// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Coordinator-side persistent state store.
//!
//! Originally just a side-channel for `(label → value)` bytes so the
//! coordinator could return the value alongside a lookup proof. With
//! the DB-anchored-state refactor, this is now the durability layer
//! for the **entire** coordinator view — slot occupancy, per-label
//! routing, the coordinator's FS chain state, the epoch-commit log,
//! and (written by each shard server, read by it on restart) per-shard
//! polynomial checkpoints.
//!
//! Two backends are supported behind the `Db` trait: `RedisDb` (a
//! network-attached Redis instance, shareable across processes) and
//! `RocksDb` (a process-local embedded LSM, single-writer). One is
//! selected at config time via [`DbSource`]; the rest of the code
//! never touches either implementation directly.
//!
//! The verifier never trusts what comes back from the DB. It always
//! re-hashes the bytes and checks the proof binds. The DB is just
//! the place the coordinator keeps "what was true at end of epoch N"
//! so it can answer probes without an RPC and resume from a cold
//! start.
//!
//! Key namespace (all binary-safe, all prefixed `aegon:`):
//!
//! | Key                                            | Contents                                              |
//! | :--                                            | :--                                                    |
//! | `aegon:value:` + label                         | raw value bytes                                        |
//! | `aegon:routing:` + label                       | `CanonicalSerialize`d `LabelRouting`                   |
//! | `aegon:slot:{shard_id}:{slot_idx}`             | label bytes (also serves as occupancy bit)             |
//! | `aegon:labels`                                 | SET of every known label (for recovery scans)          |
//! | `aegon:coord:state`                            | serialized `CoordState { epoch, r_index, r_value }`    |
//! | `aegon:coord:epoch_commit:{epoch}`             | serialized `ShardedEpochCommitment`                    |
//! | `aegon:shard:{shard_id}:state`                 | serialized shard polynomial+commitment checkpoint      |
//! | `aegon:openings:{epoch}:{shard_id}`            | serialized §6.4 `HistoryOpenings` for new placements   |

#[cfg(feature = "redis")]
mod redis;
#[cfg(feature = "rocksdb")]
mod rocksdb;

#[cfg(feature = "redis")]
pub(crate) use redis::RedisDb;
#[cfg(feature = "rocksdb")]
pub(crate) use rocksdb::RocksDb;

use super::error::AegonError;

/// Where the coordinator stores its durable state. `None` is the
/// no-DB path used by the in-process tests (lookup returns an empty
/// value, no recovery possible).
#[derive(Clone, Debug, Default)]
pub enum DbSource {
    /// No external DB. Lookups return empty values and there is no
    /// crash recovery — the in-process tests use this path because
    /// they hold the values out-of-band.
    #[default]
    None,
    /// Connect to a Redis server at this URL on coordinator setup.
    /// URL form is the standard `redis://[user[:pass]@]host[:port][/db]`.
    /// Cross-process shared (coord and shards both see each other's
    /// writes), so the slot-occupancy probe consults the same Redis.
    Redis(String),
    /// Open (or create) a RocksDB instance at this filesystem path.
    /// Process-local, single-writer — coord and shards can NOT share
    /// one directory. When this backend is selected the publish-path
    /// slot-occupancy probe automatically falls back to per-probe
    /// gRPC `is_index_slot_occupied` calls to the owning shard, since
    /// the coord's local RocksDB has no shard-side slot keys.
    Rocks(std::path::PathBuf),
}

#[cfg(feature = "redis")]
pub(crate) fn open_redis(url: &str) -> Result<Box<dyn Db>, AegonError> {
    Ok(Box::new(RedisDb::connect(url)?))
}

#[cfg(not(feature = "redis"))]
pub(crate) fn open_redis(_url: &str) -> Result<Box<dyn Db>, AegonError> {
    Err(AegonError::Config(
        "aegon was built without the `redis` feature".into(),
    ))
}

#[cfg(feature = "rocksdb")]
pub(crate) fn open_rocks(path: &std::path::Path) -> Result<Box<dyn Db>, AegonError> {
    Ok(Box::new(RocksDb::open(path)?))
}

#[cfg(not(feature = "rocksdb"))]
pub(crate) fn open_rocks(_path: &std::path::Path) -> Result<Box<dyn Db>, AegonError> {
    Err(AegonError::Config(
        "aegon was built without the `rocksdb` feature".into(),
    ))
}

/// One step of an atomic write batch — one MULTI/EXEC inside
/// `RedisDb`, one WriteBatch inside `RocksDb`.
/// New variants get added as the schema grows; the impl pipelines
/// them into a single round-trip.
#[derive(Debug, Clone)]
pub enum DbOp {
    /// `SET key value` — overwrite if present.
    Set {
        /// Key to overwrite.
        key: Vec<u8>,
        /// New value.
        value: Vec<u8>,
    },
    /// `SADD key member` — add `member` to the set at `key`.
    SAdd {
        /// Key of the set.
        key: Vec<u8>,
        /// Member to add.
        member: Vec<u8>,
    },
    /// `LPUSH key member` — prepend `member` to the head of the list
    /// at `key`. Used by the value-history sliding window: combined
    /// with an `LTrim 0 (N-1)` immediately after, this implements
    /// "keep the most recent N entries" atomically inside a publish's
    /// MULTI/EXEC.
    LPush {
        /// Key of the list.
        key: Vec<u8>,
        /// Element to prepend at the head.
        member: Vec<u8>,
    },
    /// `LTRIM key start stop` — keep only `list[start..=stop]`,
    /// discarding the rest. Negative indices count from the tail.
    /// Paired with `LPush` to bound the value-history list length.
    LTrim {
        /// Key of the list.
        key: Vec<u8>,
        /// First index to keep; negative counts from the tail.
        start: isize,
        /// Last index to keep, inclusive; negative counts from the tail.
        stop: isize,
    },
}

// Wire encoding for shipping a `Vec<DbOp>` over the shard's
// ApplyPersistenceOps gRPC. Plain length-prefixed bytes; no proto
// schema for the DbOp enum itself so it stays a crate-internal type
// while still crossing the wire.
//
// Layout:
//   u32_le n_ops
//   for each op:
//     u8 tag                          (0=Set, 1=SAdd, 2=LPush, 3=LTrim)
//     u32_le key_len, key bytes
//     for Set/SAdd/LPush: u32_le val_len, val bytes
//     for LTrim: i64_le start, i64_le stop
impl DbOp {
    /// Encode a `Vec<DbOp>` into bytes for the
    /// `ShardService::ApplyPersistenceOps` RPC. See file-level wire
    /// comment for the layout.
    pub fn encode_batch(ops: &[DbOp]) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + ops.len() * 48);
        out.extend_from_slice(&(ops.len() as u32).to_le_bytes());
        for op in ops {
            match op {
                DbOp::Set { key, value } => {
                    out.push(0);
                    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
                    out.extend_from_slice(key);
                    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
                    out.extend_from_slice(value);
                }
                DbOp::SAdd { key, member } => {
                    out.push(1);
                    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
                    out.extend_from_slice(key);
                    out.extend_from_slice(&(member.len() as u32).to_le_bytes());
                    out.extend_from_slice(member);
                }
                DbOp::LPush { key, member } => {
                    out.push(2);
                    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
                    out.extend_from_slice(key);
                    out.extend_from_slice(&(member.len() as u32).to_le_bytes());
                    out.extend_from_slice(member);
                }
                DbOp::LTrim { key, start, stop } => {
                    out.push(3);
                    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
                    out.extend_from_slice(key);
                    out.extend_from_slice(&(*start as i64).to_le_bytes());
                    out.extend_from_slice(&(*stop as i64).to_le_bytes());
                }
            }
        }
        out
    }

    /// Reverse of `encode_batch`. Returns `AegonError::Database` on
    /// any framing inconsistency (short read, unknown tag, etc.).
    pub fn decode_batch(bytes: &[u8]) -> Result<Vec<DbOp>, AegonError> {
        let mut cur = 0usize;
        let n = read_u32_le(&mut cur, bytes)? as usize;
        let mut ops = Vec::with_capacity(n);
        for i in 0..n {
            if cur >= bytes.len() {
                return Err(AegonError::Database(format!(
                    "DbOp::decode_batch: missing tag for op {i}"
                )));
            }
            let tag = bytes[cur];
            cur += 1;
            let key_len = read_u32_le(&mut cur, bytes)? as usize;
            let key = take_slice(&mut cur, key_len, bytes)?.to_vec();
            ops.push(match tag {
                0 => {
                    let v_len = read_u32_le(&mut cur, bytes)? as usize;
                    let value = take_slice(&mut cur, v_len, bytes)?.to_vec();
                    DbOp::Set { key, value }
                }
                1 => {
                    let m_len = read_u32_le(&mut cur, bytes)? as usize;
                    let member = take_slice(&mut cur, m_len, bytes)?.to_vec();
                    DbOp::SAdd { key, member }
                }
                2 => {
                    let m_len = read_u32_le(&mut cur, bytes)? as usize;
                    let member = take_slice(&mut cur, m_len, bytes)?.to_vec();
                    DbOp::LPush { key, member }
                }
                3 => {
                    let start = read_i64_le(&mut cur, bytes)? as isize;
                    let stop = read_i64_le(&mut cur, bytes)? as isize;
                    DbOp::LTrim { key, start, stop }
                }
                t => {
                    return Err(AegonError::Database(format!(
                        "DbOp::decode_batch: unknown tag {t} at op {i}"
                    )))
                }
            });
        }
        if cur != bytes.len() {
            return Err(AegonError::Database(format!(
                "DbOp::decode_batch: {} trailing bytes after {} ops",
                bytes.len() - cur,
                n
            )));
        }
        Ok(ops)
    }
}

fn take_slice<'a>(cur: &mut usize, n: usize, bytes: &'a [u8]) -> Result<&'a [u8], AegonError> {
    if *cur + n > bytes.len() {
        return Err(AegonError::Database(format!(
            "DbOp::decode_batch: short read at offset {} (need {} bytes, have {})",
            *cur,
            n,
            bytes.len() - *cur
        )));
    }
    let s = &bytes[*cur..*cur + n];
    *cur += n;
    Ok(s)
}

fn read_u32_le(cur: &mut usize, bytes: &[u8]) -> Result<u32, AegonError> {
    let s = take_slice(cur, 4, bytes)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_i64_le(cur: &mut usize, bytes: &[u8]) -> Result<i64, AegonError> {
    let s = take_slice(cur, 8, bytes)?;
    Ok(i64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

// TODO(rocksdb-caching): when we add a `RocksDb` impl of this trait
// for production-scale storage (target ~2^34 labels, ~10s of TB
// on-disk), the storage layer should grow a configurable caching
// layer. Today's `RedisDb` has no separate cache because Redis is
// itself an in-RAM store — once we move to a disk-backed engine the
// hot/cold split becomes real and caching matters.
//
// The default RocksDB block cache is ~8 MiB out of the box — fine
// for tests, **catastrophically undersized** for production. Touch
// these knobs when wiring `RocksDb::open`:
//
//   1. `BlockBasedOptions::set_block_cache(LruCache::new(N GiB))` —
//      single biggest lever. Sized to fit "the hot working set" in
//      RAM. Target 50–70% of the coord box's free RAM minus
//      whatever the moka cache below takes.
//   2. `Options::set_row_cache(LruCache::new(M GiB))` — caches
//      decoded full rows on top of block cache. Helps point-lookup-
//      heavy workloads like `aegon:value:<label>` and
//      `aegon:value_history:<label>`. ~1–2 GiB is plenty.
//   3. `BlockBasedOptions::set_bloom_filter(10, false)` — ~10 bits
//      per key, ~1% false-positive rate on `EXISTS` probes. Big win
//      for the publish-path slot-occupancy loop where most probes
//      hit empty slots that don't exist in any SSTable.
//   4. Column families per access pattern. Split the keyspace so
//      hot vs. cold tiers can have independent cache budgets:
//        * "hot_metadata" CF: `aegon:value:`, `aegon:routing:`,
//          `aegon:slot:` — small values, every lookup reads them.
//          Aggressive caching.
//        * "history" CF: `aegon:value_history:` — large, rarely
//          read per label. Smaller block cache, lean on the moka
//          layer below for hot-user hits.
//        * "audit" CF: `aegon:openings:`, `aegon:coord:epoch_commit:`
//          — write-once, almost-never-read. Minimal cache.
//   5. Optional application-level cache (e.g., the `moka` crate)
//      sitting *in front of* the RocksDB impl. Pattern:
//        `struct CachedDb { inner: RocksDb, cache: moka::Cache<...> }`
//      moka's TinyLFU eviction policy is behavior-driven by
//      definition — hot keys stay, cold keys evict, no per-user
//      configuration needed. This is the natural place to hook in
//      future per-user policies (VIP labels pinned, etc.) without
//      touching the RocksDB layer.
//
// Until any of that lands, neither backend has a tunable cache:
// Redis keeps everything in RAM by design, and RocksDB ships with
// just block-cache defaults — both work fine for the bench-scale
// workloads we run today.
//
/// Coordinator-side durable store. All writes happen via `write_atomic`
/// (one atomic batch per publish — Redis MULTI/EXEC or RocksDB
/// WriteBatch depending on the configured backend); reads are point
/// lookups + a couple of recovery-time enumeration helpers.
pub(crate) trait Db: Send + Sync {
    /// Apply every op in one atomic batch. On any error, none of the
    /// ops are applied.
    fn write_atomic(&self, ops: &[DbOp]) -> Result<(), AegonError>;
    /// Single-key read. `None` if the key doesn't exist.
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, AegonError>;
    // Set/exists helpers the current schema does not use.
    #[allow(dead_code)]
    /// Existence check — same cost as `get` but skips the payload.
    fn exists(&self, key: &[u8]) -> Result<bool, AegonError>;
    // Set/exists helpers the current schema does not use.
    #[allow(dead_code)]
    /// Pipelined batch existence check. One round-trip / batch lookup
    /// for every key in `keys`, returning a `Vec<bool>` parallel to
    /// the input. Used by the publish-path open-addressing loop
    /// where per-key `exists` round-trips dominate at large batch
    /// sizes.
    fn exists_many(&self, keys: &[Vec<u8>]) -> Result<Vec<bool>, AegonError>;
    // Set/exists helpers the current schema does not use.
    #[allow(dead_code)]
    /// Members of a SET-typed key. Used for recovery-time
    /// enumeration (the `aegon:labels` set).
    fn smembers(&self, key: &[u8]) -> Result<Vec<Vec<u8>>, AegonError>;
    /// `LRANGE key start stop` — list slice. Used by `lookup_history`
    /// to fetch the cached `aegon:value_history:{label}` window with
    /// one round-trip. Returns each element as raw bytes (canonical-
    /// serialized `StoredValueHistoryEntry`); caller decodes.
    fn lrange(&self, key: &[u8], start: isize, stop: isize) -> Result<Vec<Vec<u8>>, AegonError>;
    /// Bulk-delete every key beginning with `prefix`. The DB handle,
    /// any column families, and any in-front caching layers are kept
    /// open. RocksDB implements via a single `delete_range` over
    /// `[prefix, next_prefix)`; Redis paginates via `SCAN` + batched
    /// `DEL`. Used by `ShardedAegon::clear_dictionary` to wipe the
    /// entire `aegon:` keyspace between benchmark iterations without
    /// rebuilding the SRS or tearing down the cluster.
    fn delete_prefix(&self, prefix: &[u8]) -> Result<(), AegonError>;
}

// ---------- key-construction helpers ----------------------------------

/// `aegon:value:` prefix for the raw value-bytes side-channel.
pub(crate) fn key_value(label: &[u8]) -> Vec<u8> {
    let mut k = b"aegon:value:".to_vec();
    k.extend_from_slice(label);
    k
}

/// `aegon:coord:state` — one key, serialized `(epoch, r_index, r_value)`.
pub(crate) fn key_coord_state() -> &'static [u8] {
    b"aegon:coord:state"
}

/// `aegon:coord:epoch_commit:{epoch}` — one key per epoch with the
/// serialized `ShardedEpochCommitment`.
pub(crate) fn key_epoch_commit(epoch: u64) -> Vec<u8> {
    format!("aegon:coord:epoch_commit:{epoch}").into_bytes()
}

/// `aegon:shard:{shard_id}:state` — shard's own checkpoint.
pub(crate) fn key_shard_state(shard_id: u32) -> Vec<u8> {
    format!("aegon:shard:{shard_id}:state").into_bytes()
}

// Legacy coord-side key; `key_history_openings_local` is the one wired up.
#[allow(dead_code)]
/// `aegon:openings:{epoch}:{shard_id}` — paper §6.4 history witnesses
/// for every brand-new label `shard_id` placed during the transition
/// into `epoch`. Serialized `HistoryOpenings<E, P>` bytes. Absent when
/// the publish carried no new labels for that shard.
///
/// Legacy coord-side key. Post per-shard-DB refactor the same payload
/// lives at [`key_history_openings_local`] inside the owning shard's
/// own DB.
pub(crate) fn key_history_openings(epoch: u64, shard_id: u32) -> Vec<u8> {
    format!("aegon:openings:{epoch}:{shard_id}").into_bytes()
}

/// `aegon:openings:{epoch}` — per-shard-local variant of
/// [`key_history_openings`]. Each shard's DB is already shard-local,
/// so the `{shard_id}` suffix is redundant. Used by
/// `ShardServer::fetch_history_openings` and by the shard write side
/// of the coord-built ops batch.
pub(crate) fn key_history_openings_local(epoch: u64) -> Vec<u8> {
    format!("aegon:openings:{epoch}").into_bytes()
}

/// `aegon:value_history:{label}` — LIST-typed key storing the last
/// N `StoredValueHistoryEntry` records for a label, head-first
/// (LPUSH + LTRIM 0 N-1 pattern; both Redis and RocksDB backends
/// implement these semantics under the `Db` trait). Each list
/// element is the canonical-serialized entry bytes. Used by the
/// user-facing `lookup_history` API; absent until the label's first
/// publish.
pub(crate) fn key_value_history(label: &[u8]) -> Vec<u8> {
    let mut k = b"aegon:value_history:".to_vec();
    k.extend_from_slice(label);
    k
}

/// `aegon:label_placement:{label}` — single-value key storing the
/// canonical-serialized `StoredLabelPlacement` for a label. Written
/// exactly once, at the publish that first places the label; never
/// updated again (labels can't move in the current system). Read
/// by the user-facing `lookup_label_history` API; absent until the
/// label's placement publish.
pub(crate) fn key_label_placement(label: &[u8]) -> Vec<u8> {
    let mut k = b"aegon:label_placement:".to_vec();
    k.extend_from_slice(label);
    k
}

/// `aegon:coord:shard_fullness` — single key holding the two-layer
/// routing's per-shard fullness map. Wire format is documented on
/// `ShardedAegon::try_recover_shard_full_proofs`. Size is
/// `O(N_shards)` and never grows with the label count; this is the
/// only label-count-independent state the two-layer refactor added
/// to the coord-side keyspace.
pub(crate) fn key_shard_fullness() -> &'static [u8] {
    b"aegon:coord:shard_fullness"
}

#[cfg(test)]
mod dbop_wire_tests {
    use super::DbOp;
    #[test]
    fn roundtrip_mixed_batch() {
        let ops = vec![
            DbOp::Set {
                key: b"k1".to_vec(),
                value: b"v1".to_vec(),
            },
            DbOp::SAdd {
                key: b"set:a".to_vec(),
                member: b"member".to_vec(),
            },
            DbOp::LPush {
                key: b"list:b".to_vec(),
                member: b"entry".to_vec(),
            },
            DbOp::LTrim {
                key: b"list:b".to_vec(),
                start: 0,
                stop: 7,
            },
            // Empty key / value edges.
            DbOp::Set {
                key: vec![],
                value: vec![],
            },
            DbOp::Set {
                key: b"big".to_vec(),
                value: vec![0xAB; 1024],
            },
        ];
        let bytes = DbOp::encode_batch(&ops);
        let back = DbOp::decode_batch(&bytes).expect("decode roundtrip");
        assert_eq!(back.len(), ops.len());
        for (a, b) in ops.iter().zip(back.iter()) {
            match (a, b) {
                (DbOp::Set { key: k1, value: v1 }, DbOp::Set { key: k2, value: v2 }) => {
                    assert_eq!(k1, k2);
                    assert_eq!(v1, v2);
                }
                (
                    DbOp::SAdd {
                        key: k1,
                        member: m1,
                    },
                    DbOp::SAdd {
                        key: k2,
                        member: m2,
                    },
                ) => {
                    assert_eq!(k1, k2);
                    assert_eq!(m1, m2);
                }
                (
                    DbOp::LPush {
                        key: k1,
                        member: m1,
                    },
                    DbOp::LPush {
                        key: k2,
                        member: m2,
                    },
                ) => {
                    assert_eq!(k1, k2);
                    assert_eq!(m1, m2);
                }
                (
                    DbOp::LTrim {
                        key: k1,
                        start: s1,
                        stop: t1,
                    },
                    DbOp::LTrim {
                        key: k2,
                        start: s2,
                        stop: t2,
                    },
                ) => {
                    assert_eq!(k1, k2);
                    assert_eq!(s1, s2);
                    assert_eq!(t1, t2);
                }
                _ => panic!("variant mismatch"),
            }
        }
    }

    #[test]
    fn rejects_short_input() {
        assert!(DbOp::decode_batch(&[]).is_err());
        // Claims 1 op but no body.
        let bad = [1u8, 0, 0, 0];
        assert!(DbOp::decode_batch(&bad).is_err());
    }

    #[test]
    fn rejects_unknown_tag() {
        // 1 op, tag=99
        let bad = [1u8, 0, 0, 0, 99u8, 0, 0, 0, 0];
        assert!(DbOp::decode_batch(&bad).is_err());
    }
}
