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

use std::sync::Mutex;
use std::time::Duration;

use redis::Commands;

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

/// One step of an atomic write batch — one MULTI/EXEC inside
/// `RedisDb`, one WriteBatch inside `RocksDb`.
/// New variants get added as the schema grows; the impl pipelines
/// them into a single round-trip.
#[derive(Debug, Clone)]
pub enum DbOp {
    /// `SET key value` — overwrite if present.
    Set { key: Vec<u8>, value: Vec<u8> },
    /// `SADD key member` — add `member` to the set at `key`.
    SAdd { key: Vec<u8>, member: Vec<u8> },
    /// `LPUSH key member` — prepend `member` to the head of the list
    /// at `key`. Used by the value-history sliding window: combined
    /// with an `LTrim 0 (N-1)` immediately after, this implements
    /// "keep the most recent N entries" atomically inside a publish's
    /// MULTI/EXEC.
    LPush { key: Vec<u8>, member: Vec<u8> },
    /// `LTRIM key start stop` — keep only `list[start..=stop]`,
    /// discarding the rest. Negative indices count from the tail.
    /// Paired with `LPush` to bound the value-history list length.
    LTrim { key: Vec<u8>, start: isize, stop: isize },
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
                },
                DbOp::SAdd { key, member } => {
                    out.push(1);
                    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
                    out.extend_from_slice(key);
                    out.extend_from_slice(&(member.len() as u32).to_le_bytes());
                    out.extend_from_slice(member);
                },
                DbOp::LPush { key, member } => {
                    out.push(2);
                    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
                    out.extend_from_slice(key);
                    out.extend_from_slice(&(member.len() as u32).to_le_bytes());
                    out.extend_from_slice(member);
                },
                DbOp::LTrim { key, start, stop } => {
                    out.push(3);
                    out.extend_from_slice(&(key.len() as u32).to_le_bytes());
                    out.extend_from_slice(key);
                    out.extend_from_slice(&(*start as i64).to_le_bytes());
                    out.extend_from_slice(&(*stop as i64).to_le_bytes());
                },
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
                },
                1 => {
                    let m_len = read_u32_le(&mut cur, bytes)? as usize;
                    let member = take_slice(&mut cur, m_len, bytes)?.to_vec();
                    DbOp::SAdd { key, member }
                },
                2 => {
                    let m_len = read_u32_le(&mut cur, bytes)? as usize;
                    let member = take_slice(&mut cur, m_len, bytes)?.to_vec();
                    DbOp::LPush { key, member }
                },
                3 => {
                    let start = read_i64_le(&mut cur, bytes)? as isize;
                    let stop = read_i64_le(&mut cur, bytes)? as isize;
                    DbOp::LTrim { key, start, stop }
                },
                t => {
                    return Err(AegonError::Database(format!(
                        "DbOp::decode_batch: unknown tag {t} at op {i}"
                    )))
                },
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

#[cfg(test)]
mod dbop_wire_tests {
    use super::DbOp;
    #[test]
    fn roundtrip_mixed_batch() {
        let ops = vec![
            DbOp::Set { key: b"k1".to_vec(), value: b"v1".to_vec() },
            DbOp::SAdd { key: b"set:a".to_vec(), member: b"member".to_vec() },
            DbOp::LPush { key: b"list:b".to_vec(), member: b"entry".to_vec() },
            DbOp::LTrim { key: b"list:b".to_vec(), start: 0, stop: 7 },
            // Empty key / value edges.
            DbOp::Set { key: vec![], value: vec![] },
            DbOp::Set { key: b"big".to_vec(), value: vec![0xAB; 1024] },
        ];
        let bytes = DbOp::encode_batch(&ops);
        let back = DbOp::decode_batch(&bytes).expect("decode roundtrip");
        assert_eq!(back.len(), ops.len());
        for (a, b) in ops.iter().zip(back.iter()) {
            match (a, b) {
                (DbOp::Set { key: k1, value: v1 }, DbOp::Set { key: k2, value: v2 }) => {
                    assert_eq!(k1, k2);
                    assert_eq!(v1, v2);
                },
                (DbOp::SAdd { key: k1, member: m1 }, DbOp::SAdd { key: k2, member: m2 }) => {
                    assert_eq!(k1, k2);
                    assert_eq!(m1, m2);
                },
                (DbOp::LPush { key: k1, member: m1 }, DbOp::LPush { key: k2, member: m2 }) => {
                    assert_eq!(k1, k2);
                    assert_eq!(m1, m2);
                },
                (
                    DbOp::LTrim { key: k1, start: s1, stop: t1 },
                    DbOp::LTrim { key: k2, start: s2, stop: t2 },
                ) => {
                    assert_eq!(k1, k2);
                    assert_eq!(s1, s2);
                    assert_eq!(t1, t2);
                },
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
    /// Existence check — same cost as `get` but skips the payload.
    fn exists(&self, key: &[u8]) -> Result<bool, AegonError>;
    /// Pipelined batch existence check. One round-trip / batch lookup
    /// for every key in `keys`, returning a `Vec<bool>` parallel to
    /// the input. Used by the publish-path open-addressing loop
    /// where per-key `exists` round-trips dominate at large batch
    /// sizes.
    fn exists_many(&self, keys: &[Vec<u8>]) -> Result<Vec<bool>, AegonError>;
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

/// Redis-backed implementation. One connection behind a `Mutex` —
/// publishes serialize on the coordinator anyway (rayon parallelism
/// lives below this layer), so contention isn't a real concern.
pub(crate) struct RedisDb {
    conn: Mutex<redis::Connection>,
}

impl RedisDb {
    /// Connect eagerly (5s timeout) and PING. A misconfigured URL or
    /// an unreachable server should fail at coordinator setup, not on
    /// the first publish.
    pub(crate) fn connect(url: &str) -> Result<Self, AegonError> {
        let client = redis::Client::open(url).map_err(|e| {
            AegonError::Database(format!("invalid redis url {url:?}: {e}"))
        })?;
        let mut conn = client
            .get_connection_with_timeout(Duration::from_secs(5))
            .map_err(|e| AegonError::Database(format!("connect to {url:?}: {e}")))?;
        let pong: String = redis::cmd("PING")
            .query(&mut conn)
            .map_err(|e| AegonError::Database(format!("PING {url:?}: {e}")))?;
        if pong != "PONG" {
            return Err(AegonError::Database(format!(
                "unexpected PING reply from {url:?}: {pong:?}"
            )));
        }
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn lock_conn(&self) -> Result<std::sync::MutexGuard<'_, redis::Connection>, AegonError> {
        self.conn
            .lock()
            .map_err(|_| AegonError::Database("redis connection mutex poisoned".into()))
    }
}

impl Db for RedisDb {
    fn write_atomic(&self, ops: &[DbOp]) -> Result<(), AegonError> {
        if ops.is_empty() {
            return Ok(());
        }
        let mut conn = self.lock_conn()?;
        // MULTI/EXEC: redis::pipe().atomic() wraps the batch in a
        // transaction. Either every op lands or none of them do, which
        // is the guarantee crash recovery relies on.
        let mut pipe = redis::pipe();
        pipe.atomic();
        for op in ops {
            match op {
                DbOp::Set { key, value } => {
                    pipe.set::<&[u8], &[u8]>(key, value).ignore();
                },
                DbOp::SAdd { key, member } => {
                    pipe.sadd::<&[u8], &[u8]>(key, member).ignore();
                },
                DbOp::LPush { key, member } => {
                    pipe.lpush::<&[u8], &[u8]>(key, member).ignore();
                },
                DbOp::LTrim { key, start, stop } => {
                    pipe.ltrim::<&[u8]>(key, *start, *stop).ignore();
                },
            }
        }
        pipe.query::<()>(&mut *conn)
            .map_err(|e| AegonError::Database(format!("MULTI/EXEC ({} ops): {e}", ops.len())))
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, AegonError> {
        let mut conn = self.lock_conn()?;
        conn.get::<&[u8], Option<Vec<u8>>>(key)
            .map_err(|e| AegonError::Database(format!("GET: {e}")))
    }

    fn exists(&self, key: &[u8]) -> Result<bool, AegonError> {
        let mut conn = self.lock_conn()?;
        conn.exists::<&[u8], bool>(key)
            .map_err(|e| AegonError::Database(format!("EXISTS: {e}")))
    }

    fn exists_many(&self, keys: &[Vec<u8>]) -> Result<Vec<bool>, AegonError> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let mut conn = self.lock_conn()?;
        // Non-atomic pipeline: each EXISTS is independent, so we don't
        // need MULTI/EXEC's all-or-nothing semantics — we just want one
        // TCP round-trip instead of N. `redis::pipe()` defaults to
        // non-atomic; `.atomic()` opt-in wraps in MULTI/EXEC.
        let mut pipe = redis::pipe();
        for k in keys {
            pipe.exists::<&[u8]>(k);
        }
        pipe.query::<Vec<bool>>(&mut *conn)
            .map_err(|e| AegonError::Database(format!("pipelined EXISTS ({} keys): {e}", keys.len())))
    }

    fn smembers(&self, key: &[u8]) -> Result<Vec<Vec<u8>>, AegonError> {
        let mut conn = self.lock_conn()?;
        conn.smembers::<&[u8], Vec<Vec<u8>>>(key)
            .map_err(|e| AegonError::Database(format!("SMEMBERS: {e}")))
    }

    fn lrange(&self, key: &[u8], start: isize, stop: isize) -> Result<Vec<Vec<u8>>, AegonError> {
        let mut conn = self.lock_conn()?;
        conn.lrange::<&[u8], Vec<Vec<u8>>>(key, start, stop)
            .map_err(|e| AegonError::Database(format!("LRANGE: {e}")))
    }

    fn delete_prefix(&self, prefix: &[u8]) -> Result<(), AegonError> {
        // Redis has no `KEYS prefix*` we'd want to use in production
        // (O(N) and blocking), so SCAN cursor-iterates a chunk at a
        // time and we DEL each chunk in a non-atomic pipeline. The
        // pattern is a glob match against `prefix*`, with `*` and `?`
        // and `[` escaped so they don't expand against caller-supplied
        // bytes.
        let mut conn = self.lock_conn()?;
        let mut pattern: Vec<u8> = Vec::with_capacity(prefix.len() + 1);
        for &b in prefix {
            if matches!(b, b'*' | b'?' | b'[' | b'\\') {
                pattern.push(b'\\');
            }
            pattern.push(b);
        }
        pattern.push(b'*');
        let mut cursor: u64 = 0;
        loop {
            let (next, keys): (u64, Vec<Vec<u8>>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern[..])
                .arg("COUNT")
                .arg(1024_u64)
                .query(&mut *conn)
                .map_err(|e| AegonError::Database(format!("SCAN: {e}")))?;
            if !keys.is_empty() {
                let mut pipe = redis::pipe();
                for k in &keys {
                    pipe.del::<&[u8]>(k).ignore();
                }
                pipe.query::<()>(&mut *conn)
                    .map_err(|e| AegonError::Database(format!("DEL batch ({}): {e}", keys.len())))?;
            }
            if next == 0 {
                break;
            }
            cursor = next;
        }
        Ok(())
    }
}

// ---------- RocksDb impl ------------------------------------------------

/// RocksDB-backed implementation. Single-process, embedded LSM store
/// rooted at a directory on local disk. Designed as the target storage
/// engine for production-scale deployments (target ~2^34 labels at
/// tens of TB on-disk), where vanilla Redis can't physically hold the
/// working set in RAM.
///
/// Tradeoffs vs `RedisDb`:
///   * **Process-local**: a single `RocksDb` directory can only be
///     opened by one process at a time. This is fine in the current
///     architecture where the coord is the sole DB writer/reader for
///     all system-wide state. Shards have their own private RocksDB
///     directories (currently disabled — see
///     `SHARD_CHECKPOINT_ENABLED` in `shard_grpc.rs`) which the
///     coord never reads.
///   * **Disk-backed**: bulk of the dataset lives on SSD, with a
///     configurable RAM block cache in front. Default block cache
///     is currently RocksDB's library default (~8 MiB) — fine for
///     tests, dramatically undersized for production. See the
///     `TODO(rocksdb-caching)` note above for the knobs to turn.
///   * **Single-writer model**: the publish path is the only writer
///     to any given Redis key in our protocol, so the RMW emulation
///     of LPush/LTrim/SAdd below doesn't race in practice. A second
///     writer (e.g. concurrent recoveries) would need a
///     `OptimisticTransactionDB` or per-key locking; out of scope
///     for the current bench.
pub(crate) struct RocksDb {
    inner: rocksdb::DB,
}

impl RocksDb {
    /// Open (or create) a RocksDB instance at `path`. Uses bytewise
    /// comparator (the default) which preserves our key namespaces'
    /// lexicographic prefix-scan property for `smembers`-style
    /// iteration over `aegon:labels:`.
    pub(crate) fn open(path: &std::path::Path) -> Result<Self, AegonError> {
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        // Compression on by default at the higher levels — LZ4 is a
        // good fit (fast, ~2× compression on canonical-serialized
        // arkworks bytes which still have some structure). Level 0
        // stays uncompressed to keep flush latency tight.
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // ---- write-throughput tuning for the publish hot path ----
        //
        // Each publish emits a ~98 K-op WriteBatch (value, routing,
        // slot, history, label_placement, value_history). At
        // batch=16384 that lands in ~1.5 s of inline write time, plus
        // bursty compaction stalls if the L0 level saturates faster
        // than background compactions can drain it. Defaults target
        // small embedded workloads — we need to widen the runway.

        // Use all coord vCPUs for compaction + flush threads. Per
        // rocksdb docs, this should be set early before any other
        // background-job knobs (which override the per-pool sizes).
        // 16 matches the n2-standard-16 default coord; override with
        // AEGON_ROCKSDB_PARALLELISM if running on a different shape.
        let parallelism: i32 = std::env::var("AEGON_ROCKSDB_PARALLELISM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);
        opts.increase_parallelism(parallelism);
        // Multiple concurrent L0→L1 subcompactions so a single large
        // batch doesn't sequentialize compaction work on one core.
        opts.set_max_subcompactions(4);
        // Bigger memtable → fewer L0 flushes per publish. 256 MB +
        // 4 buffers = 1 GB total memtable budget. The bumped-tuning
        // experiment (1 GB × 8 = 8 GB budget) tried to absorb the
        // K=524k per-shard write burst, but combined with Stage B's
        // peak per-finalize op-vec footprint it pushed the n2-standard
        // shard VMs (64 GB) into OOM at fill=30%. The pre-Stage-A
        // tuning (this) is the run5-validated config that ran to 90%
        // cleanly.
        opts.set_write_buffer_size(256 * 1024 * 1024);
        // More concurrent memtables = the writer doesn't block while a
        // flush is in flight. 4 buffers ⇒ tolerate ~1 GB of in-flight
        // memtable data without back-pressure on the writer.
        opts.set_max_write_buffer_number(4);
        // Bigger SST files at every level = fewer files overall and
        // less metadata churn during compactions. 128 MB matches the
        // pre-Stage-A run5 config.
        opts.set_target_file_size_base(128 * 1024 * 1024);
        // L0 stall + stop thresholds. Default trips at 20/36; 40/60
        // gives background compaction enough headroom for hours of
        // climb at K=65536 (run5-validated).
        opts.set_level_zero_slowdown_writes_trigger(40);
        opts.set_level_zero_stop_writes_trigger(60);
        // No LRU cap on open SSTs — at 128 MB/file the full database
        // tops out at ~few thousand files even at 90% fill, so we
        // can afford to keep file descriptors for everything.
        opts.set_max_open_files(-1);
        // -1 because the coord's only writer is the bench process and
        // mid-bench durability matters less than throughput.
        opts.set_use_fsync(false);

        // Block cache: keep recently-read SST blocks in RAM so
        // compaction-side reads (and re-reads of hot keyspace) don't
        // round-trip the SSD. Default RocksDB cache is 8 MB; for the
        // large coord (64 GB RAM, ~26 GB taken by the keyspace index)
        // we have ~30 GB free, of which 16 GB makes a safe cache cap.
        // Shards have the same RAM but much smaller per-shard state,
        // so the LRU just won't fill — no waste. Override with
        // AEGON_ROCKSDB_BLOCK_CACHE_GB.
        let block_cache_gb: u64 = std::env::var("AEGON_ROCKSDB_BLOCK_CACHE_GB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);
        if block_cache_gb > 0 {
            let cache = rocksdb::Cache::new_lru_cache((block_cache_gb * 1024 * 1024 * 1024) as usize);
            let mut block_opts = rocksdb::BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            // 16 KB blocks balance random-read latency against cache
            // granularity — RocksDB default is 4 KB which fragments
            // the cache for our larger value payloads.
            block_opts.set_block_size(16 * 1024);
            opts.set_block_based_table_factory(&block_opts);
        }

        // Diagnostic stats — emitted to RocksDB's LOG file alongside
        // the DB. AEGON_ROCKSDB_STATS_DUMP_SEC=60 turns on one-minute
        // snapshots of compaction throughput, L0 file count, and
        // stall time so we can confirm whether write stalls actually
        // drive the publish-latency growth at high fill.
        let stats_dump_sec: u32 = std::env::var("AEGON_ROCKSDB_STATS_DUMP_SEC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if stats_dump_sec > 0 {
            opts.enable_statistics();
            opts.set_stats_dump_period_sec(stats_dump_sec);
        }

        let inner = rocksdb::DB::open(&opts, path)
            .map_err(|e| AegonError::Database(format!("open rocksdb {path:?}: {e}")))?;
        Ok(Self { inner })
    }

    /// Compose the internal "set member" key used to emulate Redis
    /// sets via key-as-membership encoding. Stored value is empty;
    /// presence == set membership. Iteration is via prefix scan.
    fn set_member_key(set_key: &[u8], member: &[u8]) -> Vec<u8> {
        let mut k = Vec::with_capacity(set_key.len() + 1 + member.len());
        k.extend_from_slice(set_key);
        k.push(b':');
        k.extend_from_slice(member);
        k
    }

    /// Prefix for prefix-scan over set members (`set_key + ":"`).
    fn set_member_prefix(set_key: &[u8]) -> Vec<u8> {
        let mut k = Vec::with_capacity(set_key.len() + 1);
        k.extend_from_slice(set_key);
        k.push(b':');
        k
    }
}

impl Db for RocksDb {
    fn write_atomic(&self, ops: &[DbOp]) -> Result<(), AegonError> {
        if ops.is_empty() {
            return Ok(());
        }
        // RocksDB's `WriteBatch` is atomic — either every op lands or
        // none do (assuming the WAL is properly synced, which is the
        // default). Same all-or-nothing semantics we get from Redis
        // MULTI/EXEC.
        //
        // The List ops (LPush, LTrim) need read-modify-write since
        // RocksDB has no native list type. We emulate them by storing
        // the list as a single canonical-serialized
        // `Vec<Vec<u8>>` value. Because WriteBatch is write-only, we
        // do the RMW *outside* the batch by buffering per-key state:
        // on the first list op for a key, read the current value
        // (one extra DB read); apply each subsequent list op against
        // the in-memory buffer; at the end, put each buffer's final
        // bytes into the batch. Single-writer (coord) makes this
        // safe; a multi-writer setup would need optimistic txns.
        let mut wb = rocksdb::WriteBatch::default();
        // Per-key list buffer (lazy-loaded on first list op).
        let mut list_buffers: std::collections::HashMap<Vec<u8>, Vec<Vec<u8>>> =
            std::collections::HashMap::new();
        let mut load_list = |key: &[u8]| -> Result<Vec<Vec<u8>>, AegonError> {
            let raw = self
                .inner
                .get(key)
                .map_err(|e| AegonError::Database(format!("rocksdb list-load get: {e}")))?;
            Ok(match raw {
                Some(bytes) => decode_list(&bytes)?,
                None => Vec::new(),
            })
        };
        for op in ops {
            match op {
                DbOp::Set { key, value } => {
                    wb.put(key, value);
                },
                DbOp::SAdd { key, member } => {
                    // Encoded as one zero-byte-valued key per member;
                    // smembers does a prefix scan to enumerate.
                    let composed = Self::set_member_key(key, member);
                    wb.put(&composed, &[]);
                },
                DbOp::LPush { key, member } => {
                    if !list_buffers.contains_key(key) {
                        list_buffers.insert(key.clone(), load_list(key)?);
                    }
                    // Redis LPUSH semantics: prepend to head.
                    let buf = list_buffers.get_mut(key).unwrap();
                    buf.insert(0, member.clone());
                },
                DbOp::LTrim { key, start, stop } => {
                    if !list_buffers.contains_key(key) {
                        list_buffers.insert(key.clone(), load_list(key)?);
                    }
                    let buf = list_buffers.get_mut(key).unwrap();
                    let len = buf.len() as isize;
                    // Resolve negative indices Redis-style.
                    let lo = normalize_idx(*start, len).max(0) as usize;
                    let hi_inclusive = normalize_idx(*stop, len).max(-1);
                    let new_buf: Vec<Vec<u8>> = if hi_inclusive < 0 || (lo as isize) > hi_inclusive {
                        Vec::new()
                    } else {
                        let hi = (hi_inclusive as usize + 1).min(buf.len());
                        buf[lo..hi].to_vec()
                    };
                    *buf = new_buf;
                },
            }
        }
        // Flush each list buffer into the WriteBatch.
        for (key, buf) in list_buffers {
            let bytes = encode_list(&buf)?;
            wb.put(&key, &bytes);
        }
        // WAL disabled for the bench's coordinator: every key is
        // either reconstructable from the shards (via re-publish) or
        // ephemeral (the bench tears the cluster down at the end). On
        // the publish hot path, the WAL fsync per batch is the single
        // biggest contributor to write_atomic latency on n2-standard-4
        // (~30-50% of the 1.7 s observed). Removing it leaves the
        // memtable as the only durability surface, which the bench is
        // explicitly OK with.
        let mut write_opts = rocksdb::WriteOptions::default();
        write_opts.disable_wal(true);
        self.inner
            .write_opt(wb, &write_opts)
            .map_err(|e| AegonError::Database(format!("rocksdb WriteBatch ({} ops): {e}", ops.len())))
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, AegonError> {
        self.inner
            .get(key)
            .map_err(|e| AegonError::Database(format!("rocksdb get: {e}")))
    }

    fn exists(&self, key: &[u8]) -> Result<bool, AegonError> {
        // `key_may_exist` is the bloom-filter fast path: false means
        // "definitely not present"; true means "maybe present, do a
        // real get". For our occupancy-probe workload the bloom is
        // the whole point — most slots are empty at low load factor.
        if !self.inner.key_may_exist(key) {
            return Ok(false);
        }
        Ok(self
            .inner
            .get(key)
            .map_err(|e| AegonError::Database(format!("rocksdb exists: {e}")))?
            .is_some())
    }

    fn exists_many(&self, keys: &[Vec<u8>]) -> Result<Vec<bool>, AegonError> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        // `multi_get` issues a single batched call; per-key the
        // engine still bloom-filters internally and only reads
        // SSTables for keys that might be present. Faster than
        // looping `get` for large `keys.len()`.
        let results = self.inner.multi_get(keys.iter().map(|k| k.as_slice()));
        let mut out: Vec<bool> = Vec::with_capacity(results.len());
        for r in results {
            match r {
                Ok(opt) => out.push(opt.is_some()),
                Err(e) => return Err(AegonError::Database(format!("rocksdb multi_get: {e}"))),
            }
        }
        Ok(out)
    }

    fn smembers(&self, key: &[u8]) -> Result<Vec<Vec<u8>>, AegonError> {
        // Sets are encoded as `<set_key>:<member>` keys. To
        // enumerate, prefix-scan over `<set_key>:` and strip the
        // prefix off each key. Iterator is backed by a snapshot so
        // concurrent writes don't tear the view.
        let prefix = Self::set_member_prefix(key);
        let mut out: Vec<Vec<u8>> = Vec::new();
        let iter = self.inner.prefix_iterator(&prefix);
        for kv in iter {
            let (k, _v) = kv
                .map_err(|e| AegonError::Database(format!("rocksdb smembers iter: {e}")))?;
            if !k.starts_with(&prefix) {
                // prefix_iterator can over-scan past the prefix when
                // bloom filters are off — defensive bound check.
                break;
            }
            out.push(k[prefix.len()..].to_vec());
        }
        Ok(out)
    }

    fn lrange(&self, key: &[u8], start: isize, stop: isize) -> Result<Vec<Vec<u8>>, AegonError> {
        let raw = self
            .inner
            .get(key)
            .map_err(|e| AegonError::Database(format!("rocksdb lrange get: {e}")))?;
        let list = match raw {
            Some(bytes) => decode_list(&bytes)?,
            None => return Ok(Vec::new()),
        };
        let len = list.len() as isize;
        let lo = normalize_idx(start, len).max(0) as usize;
        let hi_inclusive = normalize_idx(stop, len).max(-1);
        if hi_inclusive < 0 || (lo as isize) > hi_inclusive {
            return Ok(Vec::new());
        }
        let hi = (hi_inclusive as usize + 1).min(list.len());
        Ok(list[lo..hi].to_vec())
    }

    fn delete_prefix(&self, prefix: &[u8]) -> Result<(), AegonError> {
        // RocksDB's `delete_range(start, end)` deletes every key in
        // `[start, end)`. `end` here is the lexicographically next
        // string after the prefix — found by incrementing the last
        // byte that isn't already `0xff`. An all-`0xff` prefix has no
        // strict upper bound; we fall back to a prefix-iterator + DEL
        // scan, which is slower but never wrong.
        let next_prefix = lex_next_prefix(prefix);
        if let Some(end) = next_prefix {
            let mut wb = rocksdb::WriteBatch::default();
            wb.delete_range(prefix, &end);
            let mut write_opts = rocksdb::WriteOptions::default();
            write_opts.disable_wal(true);
            self.inner.write_opt(wb, &write_opts).map_err(|e| {
                AegonError::Database(format!("rocksdb delete_range({}): {e}", prefix.len()))
            })?;
        } else {
            let iter = self.inner.prefix_iterator(prefix);
            let mut wb = rocksdb::WriteBatch::default();
            for kv in iter {
                let (k, _v) = kv
                    .map_err(|e| AegonError::Database(format!("rocksdb prefix-scan: {e}")))?;
                if !k.starts_with(prefix) {
                    break;
                }
                wb.delete(&k);
            }
            let mut write_opts = rocksdb::WriteOptions::default();
            write_opts.disable_wal(true);
            self.inner.write_opt(wb, &write_opts).map_err(|e| {
                AegonError::Database(format!("rocksdb prefix-delete fallback: {e}"))
            })?;
        }
        // Cheap async compaction over the cleared range so the next
        // read-side iterator doesn't have to walk thousands of newly-
        // tombstoned SST entries. Bounded to the deleted range so it
        // doesn't touch any other live data.
        if let Some(end) = lex_next_prefix(prefix) {
            self.inner.compact_range(Some(prefix), Some(end.as_slice()));
        }
        Ok(())
    }

}

/// Lexicographically next prefix after `prefix`. Returns `None` when
/// every byte in `prefix` is `0xff` (no strict upper bound exists).
/// Powers RocksDB's `delete_range(prefix, end)` upper bound — `end`
/// must be exclusive and ordered after every `prefix + suffix` key.
fn lex_next_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut out = prefix.to_vec();
    while let Some(last) = out.last_mut() {
        if *last < 0xff {
            *last += 1;
            return Some(out);
        }
        out.pop();
    }
    None
}

/// Redis-style negative-index normalization: `-1` → `len-1`,
/// `-2` → `len-2`, etc. Out-of-range negatives clamp to `-1`
/// (LTRIM "delete everything" semantics).
fn normalize_idx(idx: isize, len: isize) -> isize {
    if idx < 0 {
        let n = len + idx;
        if n < 0 { -1 } else { n }
    } else {
        // Positive indices clamp to `len - 1` (past-the-end → last
        // valid index, matching Redis behavior).
        idx.min((len - 1).max(0))
    }
}

/// Canonical encoding for the RocksDB list-as-blob representation:
/// `u32 le count` followed by `count` × (`u32 le len, len bytes`).
/// Compact, no external dep, and prefix-stable so future "incremental
/// append" optimizations could append without rewriting.
fn encode_list(items: &[Vec<u8>]) -> Result<Vec<u8>, AegonError> {
    if items.len() > u32::MAX as usize {
        return Err(AegonError::Database(format!(
            "list length {} exceeds u32::MAX",
            items.len()
        )));
    }
    let total: usize = 4 + items.iter().map(|i| 4 + i.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&(items.len() as u32).to_le_bytes());
    for item in items {
        if item.len() > u32::MAX as usize {
            return Err(AegonError::Database(format!(
                "list element of size {} exceeds u32::MAX",
                item.len()
            )));
        }
        out.extend_from_slice(&(item.len() as u32).to_le_bytes());
        out.extend_from_slice(item);
    }
    Ok(out)
}

fn decode_list(bytes: &[u8]) -> Result<Vec<Vec<u8>>, AegonError> {
    if bytes.len() < 4 {
        return Err(AegonError::Database("list blob truncated (header)".into()));
    }
    let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    let mut out = Vec::with_capacity(count);
    let mut off = 4;
    for _ in 0..count {
        if off + 4 > bytes.len() {
            return Err(AegonError::Database("list blob truncated (elem len)".into()));
        }
        let elen = u32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]) as usize;
        off += 4;
        if off + elen > bytes.len() {
            return Err(AegonError::Database("list blob truncated (elem body)".into()));
        }
        out.push(bytes[off..off + elen].to_vec());
        off += elen;
    }
    Ok(out)
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
