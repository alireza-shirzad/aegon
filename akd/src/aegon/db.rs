//! Coordinator-side persistent state store.
//!
//! Originally just a side-channel for `(label → value)` bytes so the
//! coordinator could return the value alongside a lookup proof. With
//! the Redis-anchored-state refactor, this is now the durability layer
//! for the **entire** coordinator view — slot occupancy, per-label
//! routing, the coordinator's FS chain state, the epoch-commit log,
//! and (written by each shard server, read by it on restart) per-shard
//! polynomial checkpoints.
//!
//! The verifier never trusts what comes back from Redis. It always
//! re-hashes the bytes and checks the proof binds. Redis is just the
//! place the coordinator keeps "what was true at end of epoch N" so it
//! can answer probes without an RPC and resume from a cold start.
//!
//! Key namespace (all binary-safe, all prefixed `aegon:`):
//!
//! | Key                                            | Contents                                              |
//! | :--                                            | :--                                                    |
//! | `aegon:value:` + label                         | raw value bytes                                        |
//! | `aegon:routing:` + label                       | `CanonicalSerialize`d `LabelRouting`                   |
//! | `aegon:slot:{shard_id}:{slot_idx}`             | label bytes (also serves as occupancy bit)             |
//! | `aegon:labels`                                 | Redis SET of every known label (for recovery scans)    |
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
    Redis(String),
}

/// One step of an atomic write batch (a MULTI/EXEC inside RedisDb).
/// New variants get added as the schema grows; the impl pipelines
/// them into a single round-trip.
#[derive(Debug)]
pub(crate) enum DbOp {
    /// `SET key value` — overwrite if present.
    Set { key: Vec<u8>, value: Vec<u8> },
    /// `SADD key member` — add `member` to the set at `key`.
    SAdd { key: Vec<u8>, member: Vec<u8> },
}

/// Coordinator-side durable store. All writes happen via `write_atomic`
/// (one MULTI/EXEC per publish); reads are point lookups + a couple of
/// recovery-time enumeration helpers.
pub(crate) trait Db: Send + Sync {
    /// Apply every op in a single MULTI/EXEC. On Redis errors, none of
    /// the ops are applied.
    fn write_atomic(&self, ops: &[DbOp]) -> Result<(), AegonError>;
    /// Single-key read. `None` if the key doesn't exist.
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, AegonError>;
    /// Existence check — same network cost as `get` but skips the
    /// payload.
    fn exists(&self, key: &[u8]) -> Result<bool, AegonError>;
    /// Members of a Redis SET. Used for recovery-time enumeration
    /// (the `aegon:labels` set).
    fn smembers(&self, key: &[u8]) -> Result<Vec<Vec<u8>>, AegonError>;
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

    fn smembers(&self, key: &[u8]) -> Result<Vec<Vec<u8>>, AegonError> {
        let mut conn = self.lock_conn()?;
        conn.smembers::<&[u8], Vec<Vec<u8>>>(key)
            .map_err(|e| AegonError::Database(format!("SMEMBERS: {e}")))
    }
}

// ---------- key-construction helpers ----------------------------------

/// `aegon:value:` prefix for the raw value-bytes side-channel.
pub(crate) fn key_value(label: &[u8]) -> Vec<u8> {
    let mut k = b"aegon:value:".to_vec();
    k.extend_from_slice(label);
    k
}

/// `aegon:routing:` prefix for the `CanonicalSerialize`d `LabelRouting`.
pub(crate) fn key_routing(label: &[u8]) -> Vec<u8> {
    let mut k = b"aegon:routing:".to_vec();
    k.extend_from_slice(label);
    k
}

/// `aegon:slot:{shard_id}:{slot_idx}` for occupancy + reverse-index.
/// The presence of this key means "shard `shard_id`'s slot `slot_idx`
/// is occupied"; the value is the label that owns the slot, so a
/// fresh shard can reconstruct its `(index_poly, value_poly)` by
/// pairing `slot:` entries with their `value:` counterparts.
pub(crate) fn key_slot(shard_id: u32, slot_idx: usize) -> Vec<u8> {
    format!("aegon:slot:{shard_id}:{slot_idx}").into_bytes()
}

/// `aegon:labels` Redis SET of every label ever published. Used by
/// recovery to enumerate `(label, routing, value)` triples without
/// relying on `SCAN MATCH` against binary keys.
pub(crate) fn key_labels_set() -> &'static [u8] {
    b"aegon:labels"
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
pub(crate) fn key_history_openings(epoch: u64, shard_id: u32) -> Vec<u8> {
    format!("aegon:openings:{epoch}:{shard_id}").into_bytes()
}
