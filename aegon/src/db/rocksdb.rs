// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! RocksDB-backed durable store.

use super::{AegonError, Db, DbOp};

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
///     `TODO(rocksdb-caching)` note in `db.rs` for the knobs to turn.
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
            let cache =
                rocksdb::Cache::new_lru_cache((block_cache_gb * 1024 * 1024 * 1024) as usize);
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

    // Set/exists helpers the current schema does not use.
    #[allow(dead_code)]
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
        let load_list = |key: &[u8]| -> Result<Vec<Vec<u8>>, AegonError> {
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
                }
                DbOp::SAdd { key, member } => {
                    // Encoded as one zero-byte-valued key per member;
                    // smembers does a prefix scan to enumerate.
                    let composed = Self::set_member_key(key, member);
                    wb.put(&composed, []);
                }
                DbOp::LPush { key, member } => {
                    if !list_buffers.contains_key(key) {
                        list_buffers.insert(key.clone(), load_list(key)?);
                    }
                    // Redis LPUSH semantics: prepend to head.
                    let buf = list_buffers.get_mut(key).unwrap();
                    buf.insert(0, member.clone());
                }
                DbOp::LTrim { key, start, stop } => {
                    if !list_buffers.contains_key(key) {
                        list_buffers.insert(key.clone(), load_list(key)?);
                    }
                    let buf = list_buffers.get_mut(key).unwrap();
                    let len = buf.len() as isize;
                    // Resolve negative indices Redis-style.
                    let lo = normalize_idx(*start, len).max(0) as usize;
                    let hi_inclusive = normalize_idx(*stop, len).max(-1);
                    let new_buf: Vec<Vec<u8>> = if hi_inclusive < 0 || (lo as isize) > hi_inclusive
                    {
                        Vec::new()
                    } else {
                        let hi = (hi_inclusive as usize + 1).min(buf.len());
                        buf[lo..hi].to_vec()
                    };
                    *buf = new_buf;
                }
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
        self.inner.write_opt(wb, &write_opts).map_err(|e| {
            AegonError::Database(format!("rocksdb WriteBatch ({} ops): {e}", ops.len()))
        })
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
            let (k, _v) =
                kv.map_err(|e| AegonError::Database(format!("rocksdb smembers iter: {e}")))?;
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
                let (k, _v) =
                    kv.map_err(|e| AegonError::Database(format!("rocksdb prefix-scan: {e}")))?;
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
        if n < 0 {
            -1
        } else {
            n
        }
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
            return Err(AegonError::Database(
                "list blob truncated (elem len)".into(),
            ));
        }
        let elen = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
            as usize;
        off += 4;
        if off + elen > bytes.len() {
            return Err(AegonError::Database(
                "list blob truncated (elem body)".into(),
            ));
        }
        out.push(bytes[off..off + elen].to_vec());
        off += elen;
    }
    Ok(out)
}
