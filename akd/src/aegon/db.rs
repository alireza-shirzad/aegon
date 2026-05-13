//! Coordinator-side label→value storage.
//!
//! The polynomial commitments only ever commit to hashes of `(label,
//! value)`. The raw bytes are side data: needed so the coordinator
//! can return a value alongside the proof on lookup, but never trusted
//! by the verifier (which re-hashes the value itself).
//!
//! Today we ship one backend, [`RedisDb`], plus a no-op
//! "disconnected" mode for in-process tests that already know the
//! value out-of-band. New backends should implement [`Db`].

use std::sync::Mutex;
use std::time::Duration;

use redis::Commands;

use super::error::AegonError;

/// Where the coordinator stores raw `(label, value)` bytes. The
/// polynomial commitments live elsewhere (in-RAM on each shard); this
/// is a side-channel for retrieving the value bytes on lookup.
#[derive(Clone, Debug, Default)]
pub enum DbSource {
    /// No external DB. `lookup` returns an empty value vector and the
    /// caller is expected to know the value out-of-band (this is the
    /// path the in-process tests take — they pass the pre-published
    /// value straight to the verifier).
    #[default]
    None,
    /// Connect to a Redis server at this URL on coordinator setup.
    /// URL form is the standard `redis://[user[:pass]@]host[:port][/db]`.
    Redis(String),
}

/// Coordinator-side KV store interface. Implementors are responsible
/// for whatever pooling / retry behaviour suits the backend; the
/// coordinator just calls `put_batch` once per publish and `get` once
/// per lookup.
pub(crate) trait Db: Send + Sync {
    fn put_batch(&self, items: &[(Vec<u8>, Vec<u8>)]) -> Result<(), AegonError>;
    fn get(&self, label: &[u8]) -> Result<Option<Vec<u8>>, AegonError>;
}

/// Redis-backed implementation. Holds a `redis::Client` (cheap to
/// clone, thread-safe) plus a single multiplexed `Connection` behind
/// a `Mutex` — Redis pipelines/single-threaded I/O are fast enough at
/// our scale that we don't need a connection pool yet.
pub(crate) struct RedisDb {
    conn: Mutex<redis::Connection>,
}

impl RedisDb {
    /// Connect (eagerly, with a short timeout) and PING. Fails fast if
    /// the URL is wrong, the server is down, or the network won't
    /// route — the coordinator should refuse to come up rather than
    /// surface the error on the first publish.
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
}

impl Db for RedisDb {
    fn put_batch(&self, items: &[(Vec<u8>, Vec<u8>)]) -> Result<(), AegonError> {
        if items.is_empty() {
            return Ok(());
        }
        let mut conn = self
            .conn
            .lock()
            .map_err(|_| AegonError::Database("redis connection mutex poisoned".into()))?;
        let mut pipe = redis::pipe();
        for (k, v) in items {
            pipe.set::<&[u8], &[u8]>(k, v).ignore();
        }
        pipe.query::<()>(&mut *conn)
            .map_err(|e| AegonError::Database(format!("MSET batch (n={}): {e}", items.len())))
    }

    fn get(&self, label: &[u8]) -> Result<Option<Vec<u8>>, AegonError> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|_| AegonError::Database("redis connection mutex poisoned".into()))?;
        conn.get::<&[u8], Option<Vec<u8>>>(label)
            .map_err(|e| AegonError::Database(format!("GET: {e}")))
    }
}
