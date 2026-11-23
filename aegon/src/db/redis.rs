// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Redis-backed durable store.

use std::sync::Mutex;
use std::time::Duration;

use ::redis::Commands;

use super::{AegonError, Db, DbOp};

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
        let client = redis::Client::open(url)
            .map_err(|e| AegonError::Database(format!("invalid redis url {url:?}: {e}")))?;
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
                }
                DbOp::SAdd { key, member } => {
                    pipe.sadd::<&[u8], &[u8]>(key, member).ignore();
                }
                DbOp::LPush { key, member } => {
                    pipe.lpush::<&[u8], &[u8]>(key, member).ignore();
                }
                DbOp::LTrim { key, start, stop } => {
                    pipe.ltrim::<&[u8]>(key, *start, *stop).ignore();
                }
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
        pipe.query::<Vec<bool>>(&mut *conn).map_err(|e| {
            AegonError::Database(format!("pipelined EXISTS ({} keys): {e}", keys.len()))
        })
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
                pipe.query::<()>(&mut *conn).map_err(|e| {
                    AegonError::Database(format!("DEL batch ({}): {e}", keys.len()))
                })?;
            }
            if next == 0 {
                break;
            }
            cursor = next;
        }
        Ok(())
    }
}
