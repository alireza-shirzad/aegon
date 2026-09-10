// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! gRPC transport for sharded Aegon.
//!
//! The in-process `ShardedAegon` coordinator owns a `Vec<Aegon<E,P,H>>`
//! and dispatches the seven low-level shard methods locally. To make
//! shards live on different machines, those local calls become RPCs
//! against this module's `ShardService`. Everything is bytes-tunneled
//! over a tiny proto schema; the heavy lifting is arkworks
//! `CanonicalSerialize`/`Deserialize` on both sides.
//!
//! Three things live here:
//!   * [`ShardHandle`] — the trait the coordinator dispatches through.
//!     Implemented by `Aegon` (in-process) and [`GrpcShardClient`]
//!     (remote).
//!   * [`ShardServer`] — wraps an `Aegon` and exposes it over tonic.
//!     The `aegon_shard_server` binary just plumbs CLI args into
//!     `ShardServer::serve`.
//!   * [`GrpcShardClient`] — implements `ShardHandle` by making
//!     blocking RPCs from a private tokio runtime. Sync surface so
//!     the rayon-parallel publish loop in `ShardedAegon` stays the
//!     same shape regardless of transport.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use ark_ec::pairing::Pairing;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use tokio::runtime::Runtime;
use tokio::sync::RwLock as AsyncRwLock;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Identity, Server, ServerTlsConfig};
use tonic::{Request, Response, Status};

use super::config::VerifierContext;
use super::db::{
    key_history_openings_local, key_label_placement, key_shard_state, key_value, key_value_history,
    Db, DbOp, DbSource, RedisDb,
};
use super::error::AegonError;
use super::hash::{HashSuite, Sha256Hash};
use super::server::Aegon;
use super::server::AegonCheckpoint;
use super::sharded::ShardWrite;
use super::types::{AegonPcs, EpochCommitment, Label, Value};

// Generated tonic code lives in this module. `tonic-build` emits one
// rust module per proto package; ours is `aegon.shard.v1`.
// Generated code carries no rustdoc; the crate-level `warn(missing_docs)`
// cannot be satisfied for types we do not author.
#[allow(missing_docs)]
pub mod proto {
    tonic::include_proto!("aegon.shard.v1");
}

use proto::shard_service_client::ShardServiceClient;
use proto::shard_service_server::{ShardService, ShardServiceServer};
use proto::{
    ApplyPersistenceOpsRequest, ApplyPersistenceOpsResponse, CommitmentResponse, Empty,
    FetchHistoryOpeningsRequest, FetchHistoryOpeningsResponse, FetchLabelPlacementRequest,
    FetchLabelPlacementResponse, FetchLabelProofTrailRequest, FetchLabelProofTrailResponse,
    FetchValueHistoryRequest, FetchValueHistoryResponse, FetchValueRequest, FetchValueResponse,
    FindLabelSlotRequest, FindLabelSlotResponse, OpenResponse, PublishBatchRequest,
    PublishBatchResponse, PublishPhase1Request, PublishPhase1Response,
    PublishPhase2AndPersistRequest, PublishPhase2AndPersistResponse, ReconfigurePrefillRequest,
    ReconfigurePrefillResponse, SlotEpochRequest, SlotOccupiedResponse, SlotRequest,
};

// ---------- wire encoding helpers --------------------------------------

// Use uncompressed (un)serialization on the gRPC wire. Compressed
// format halves payload size but each G1Affine decode pays a Tonelli–
// Shanks square root in Fp (~5 µs/pt on Bn254); for a 10k-update
// publish the coordinator decodes ~1.8M points, which alone burned
// ~2 s of single-thread CPU per shard (×4 shards via rayon ≈ 2 s wall).
// Uncompressed is a straight memcpy and skips the curve-membership
// check, since both sides of the wire run our own binaries and we
// trust the points.
fn encode<T: CanonicalSerialize>(t: &T) -> Result<Vec<u8>, AegonError> {
    let mut buf = Vec::with_capacity(t.uncompressed_size());
    t.serialize_uncompressed(&mut buf)
        .map_err(|e| AegonError::Config(format!("encode: {e}")))?;
    Ok(buf)
}

fn decode<T: CanonicalDeserialize>(bytes: &[u8]) -> Result<T, AegonError> {
    T::deserialize_uncompressed_unchecked(bytes)
        .map_err(|e| AegonError::Config(format!("decode: {e}")))
}

fn err_to_status(e: AegonError) -> Status {
    Status::internal(format!("{e}"))
}

fn status_to_err(s: Status) -> AegonError {
    AegonError::Config(format!("grpc: {} ({})", s.message(), s.code()))
}

// ---------- ShardHandle trait ------------------------------------------

/// What the `ShardedAegon` coordinator needs from each shard. The
/// in-process implementation (`Aegon<E, P, H>`) just forwards each
/// method; the remote implementation ([`GrpcShardClient`]) makes a
/// blocking RPC. The trait is intentionally sync — callers like
/// `ShardedAegon::publish` use rayon's `par_iter_mut`, which is
/// incompatible with `async fn`.
pub trait ShardHandle<E, P, H>: Send + Sync
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    /// Apply a batch of writes at already-resolved slots and return the
    /// new `(index, value)` commitments. Phase 1 of a publish: the data
    /// polynomials are mutated, the rand polynomials are not.
    fn publish_phase_1_at_slots(
        &mut self,
        batch: &[ShardWrite<E::ScalarField>],
    ) -> Result<(P::Commitment, P::Commitment), AegonError>;

    /// Combined Phase-2 + persist: apply the cross-shard FS scalars,
    /// build the per-shard `StoredValueHistoryEntry` /
    /// `StoredLabelPlacement` DbOps locally (with empty merkle paths —
    /// the coord stitches them in at lookup time), write them
    /// atomically to the shard's local DB, and return the new
    /// `EpochCommitment`. Replaces the legacy 3-step (PublishPhase2 +
    /// FinalizePublishPersist) flow with one RPC.
    ///
    /// `shard_id` is passed in: the shard process itself doesn't
    /// track its own id — the coord knows it from the routing table.
    fn publish_phase_2_and_persist(
        &mut self,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
        shard_id: u32,
    ) -> Result<EpochCommitment<E, P>, AegonError>;

    /// Whether the open-addressing walk should skip past this slot.
    fn is_index_slot_occupied(&self, slot_bits: &[bool]) -> bool;

    /// Open the `index` polynomial at `slot_bits` in the current epoch.
    fn open_index_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    /// Open the `value` polynomial at `slot_bits` in the current epoch.
    fn open_value_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    /// Open the randomized `index` polynomial at `slot_bits` as of
    /// `epoch`. Requires that epoch to still be retained.
    fn open_rand_index_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    /// Open the randomized `value` polynomial at `slot_bits` as of
    /// `epoch`. Requires that epoch to still be retained.
    fn open_rand_value_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    /// Open the live `rand_value_poly` at `slot_bits`. Mirrors
    /// `Aegon::open_rand_value_at_slot_current`. Used by
    /// `ShardedAegon::lookup_history` to attach a freshness
    /// attestation to the history bundle.
    fn open_rand_value_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    /// Open the live `rand_index_poly` at `slot_bits`. Label-side
    /// mirror; used by `ShardedAegon::lookup_label_history`.
    fn open_rand_index_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError>;

    /// Re-mask a publish-time non-ZK value-history entry into a
    /// hiding one. The coordinator calls this during
    /// `ShardedAegon::lookup_history` for every stored entry it
    /// reads from the DB before serving the bundle to a user — the
    /// masking-server protocol is applied at user-facing time, not
    /// at publish time.
    fn remask_value_history_entry(
        &self,
        entry: super::sharded::StoredValueHistoryEntry<E, P>,
    ) -> Result<super::sharded::StoredValueHistoryEntry<E, P>, AegonError>;

    /// This shard's commitment at its current epoch.
    fn current_commitment(&self) -> EpochCommitment<E, P>;

    /// Per-shard verifier context. Only really needed at coordinator
    /// setup time; we read it from shard 0 to build the public
    /// `ShardedVerifierContext`.
    fn verifier_context(&self) -> VerifierContext<E, P>;

    /// Log2 of this shard's slot count.
    fn log_capacity(&self) -> usize;

    /// Pre-populate this shard's polynomials with `count` random
    /// `(slot, h_label, h_value)` entries. Drives the publish bench's
    /// fill-percentage sweep.
    ///
    /// Only callable on a freshly-`init`-ed shard at epoch 0 with no
    /// pending publish (matches `Aegon::prefill_random`). Default
    /// returns `AegonError::Config(...)` so the gRPC `ShardClient`
    /// impl, which has no path to call into a remote shard's prefill,
    /// can keep the no-op default — remote prefill is driven at boot
    /// time via `aegon_shard_server --prefill-count`.
    fn prefill_random_in_place(&mut self, _count: usize, _seed: u64) -> Result<(), AegonError> {
        Err(AegonError::Config(
            "prefill_random_in_place not supported via this transport — set --prefill-count at shard boot time"
                .into(),
        ))
    }

    /// Wipe this shard back to its post-setup empty state without
    /// regenerating the SRS or dropping the transport. The in-process
    /// impl calls [`Aegon::clear_dictionary`] (cheap; restores from a
    /// stashed setup baseline). The default returns `Err` so the
    /// remote-transport impl, which would need a `ClearDictionary`
    /// RPC, fails loudly until that RPC is wired up.
    fn clear_dictionary(&mut self) -> Result<(), AegonError> {
        Err(AegonError::Config(
            "clear_dictionary not supported via this transport — wire up the ClearDictionary RPC"
                .into(),
        ))
    }

    /// Two-layer routing's publish entry. Takes raw `(label, value)`
    /// tuples already routed to this shard by the coord's first-layer
    /// `H_shard`. The shard runs its own second-layer `H_slot` open-
    /// addressing internally to pick slots, then runs the phase-1
    /// crypto pipeline. Returns commitments + per-label placements +
    /// optional fullness signal — see [`PublishBatchOutcome`] for the
    /// fields. The default `Err` keeps remote transports honest:
    /// every concrete `ShardHandle` impl must opt in.
    fn publish_batch(
        &mut self,
        _batch: &[(super::types::Label, super::types::Value)],
        _vrf_proofs_shard_per_label: &[Vec<Vec<u8>>],
    ) -> Result<super::server::PublishBatchOutcome<E, P>, AegonError>
    where
        P::Commitment: Clone,
    {
        Err(AegonError::Config(
            "publish_batch not supported via this transport — wire up the PublishBatch RPC".into(),
        ))
    }

    /// Two-layer routing's lookup entry: walk the intra-shard
    /// `H_slot(slot_ctr, label)` probe trail and report the slot that
    /// stores this label (and the `slot_ctr` it took to reach it), or
    /// `None` if the label is not in this shard. Mirrors
    /// [`Aegon::find_label_slot`]. The default `Err` keeps remote
    /// transports honest — every concrete impl must opt in.
    fn find_label_slot(
        &self,
        _label: &super::types::Label,
    ) -> Result<Option<(Vec<bool>, u64)>, AegonError> {
        Err(AegonError::Config(
            "find_label_slot not supported via this transport — wire up the FindLabelSlot RPC"
                .into(),
        ))
    }

    /// Two-layer routing's combined lookup entry. Walks `H_slot`
    /// exactly like [`Self::find_label_slot`] and, for every probed
    /// slot, also returns the index-polynomial opening at that slot.
    /// The coord uses this on the destination shard during
    /// `ShardedAegon::lookup_label_two_layer` instead of issuing
    /// `find_label_slot` + N × `open_index_at_slot` separately — one
    /// network RTT instead of `slot_ctr0 + 2`. The shard walks the
    /// H_slot trail only once.
    ///
    /// `Ok(None)` mirrors `find_label_slot`'s "label not in this
    /// shard" outcome. The default implementation composes the legacy
    /// methods so any transport that already implements
    /// `find_label_slot` + `open_index_at_slot` works without further
    /// wiring; the GRPC transport overrides this to make the RPC
    /// directly and is the load-bearing path on the cluster.
    fn fetch_label_proof_trail(
        &self,
        label: &super::types::Label,
    ) -> Result<Option<super::sharded::LabelProofTrail<E, P>>, AegonError> {
        let Some((final_slot_bits, slot_ctr0)) = self.find_label_slot(label)? else {
            return Ok(None);
        };
        // Default fallback — recompose the trail by issuing one
        // `open_index_at_slot` per probe. The remote transport
        // overrides this with a single RPC.
        let mut entries: Vec<super::sharded::LabelProofTrailEntry<E, P>> =
            Vec::with_capacity((slot_ctr0 as usize) + 1);
        let log_capacity = self.log_capacity();
        for ctr in 0..=slot_ctr0 {
            // Re-derive slot_bits via the same H_slot the shard used.
            // We don't have a `&self.vrf_prover` here on the trait, so
            // use the static H::h_slot path — this default is the
            // in-process Aegon-backed `ShardHandle`, which also uses
            // the static path internally for `H::h_slot` evaluations.
            let slot_bits = if ctr == slot_ctr0 {
                final_slot_bits.clone()
            } else {
                H::h_slot(ctr, label, log_capacity)
            };
            let (evaluation, proof) = self.open_index_at_slot(&slot_bits)?;
            entries.push(super::sharded::LabelProofTrailEntry {
                slot_bits,
                evaluation,
                proof,
            });
        }
        Ok(Some(super::sharded::LabelProofTrail {
            final_slot_bits,
            slot_ctr0,
            entries,
            // The trait default has no access to a VRF prover or to
            // the coord's H_shard route, so it leaves both proof
            // vectors empty. The coord's lookup path treats empty
            // proof vectors as "no cache available — recompute" and
            // falls back to the legacy prove_h_* calls. Production
            // shards override this fn with a path that fills both.
            vrf_proofs_shard: Vec::new(),
            vrf_proofs_slot: Vec::new(),
        }))
    }

    // ---------- per-shard durable state (post-refactor) ----------
    //
    // Defaults are deliberately no-op (or empty-result) so the
    // in-process `Aegon`-backed `ShardHandle` impl can keep using the
    // legacy "no DB on the shard" path without forcing every test to
    // wire up a per-shard DB. The cluster transport
    // (`GrpcShardClient`) overrides these to make the RPCs.

    /// Apply a pre-formed `Vec<DbOp>` (encoded via
    /// [`super::db::DbOp::encode_batch`]) to this shard's local DB
    /// atomically. Used by the coord at publish persist time.
    ///
    /// Default impl is a no-op so the in-process `Aegon`-backed shard
    /// (which has no DB of its own) accepts the call silently — tests
    /// using `DbSource::None` already hold the values out-of-band.
    ///
    /// Takes `&self` (not `&mut self`) so the coord can call it from a
    /// `&self` context like `persist_publish_to_db`. The remote impl
    /// constructs a fresh `ShardServiceClient` from the cached Channel
    /// per call; the in-process default is a no-op.
    fn apply_persistence_ops(&self, _ops_bytes: &[u8]) -> Result<(), AegonError> {
        Ok(())
    }

    /// Read the raw value bytes for `label` from this shard's DB.
    /// Returns `Ok(None)` when the shard has no DB (in-process tests)
    /// or when the label is absent.
    fn fetch_value(&self, _label: &super::types::Label) -> Result<Option<Vec<u8>>, AegonError> {
        Ok(None)
    }

    /// Read the value-history sliding window for `label` from this
    /// shard's DB. Returns `Ok(empty)` when the shard has no DB or the
    /// label has no history yet.
    fn fetch_value_history(
        &self,
        _label: &super::types::Label,
    ) -> Result<Vec<Vec<u8>>, AegonError> {
        Ok(Vec::new())
    }

    /// Combined value-history RPC: collapses fetch + per-entry remask +
    /// freshness opening into a single shard round-trip. Used by
    /// `ShardedAegon::lookup_history`.
    ///
    /// The default implementation composes
    ///   `fetch_value_history` + per-entry `remask_value_history_entry`
    ///   + `open_rand_value_at_slot_current`
    /// so the in-process transport works without any extra plumbing.
    /// The GRPC client overrides this with a single RPC; the GRPC
    /// server hands off to this default impl on the shard side. Entries
    /// in the returned `FullValueHistory` carry empty
    /// `prev_merkle_path` / `post_merkle_path` — the coord stitches
    /// those in from its `epoch_commits` cache before serving the
    /// bundle to the user.
    fn fetch_full_value_history(
        &self,
        label: &super::types::Label,
    ) -> Result<super::sharded::FullValueHistory<E, P>, AegonError> {
        let mut raw_entries = self.fetch_value_history(label)?;
        if raw_entries.len() > super::sharded::HISTORY_WINDOW {
            raw_entries.truncate(super::sharded::HISTORY_WINDOW);
        }
        if raw_entries.is_empty() {
            return Ok(super::sharded::FullValueHistory {
                entries: Vec::new(),
                freshness_eval: None,
                freshness_proof: None,
            });
        }
        // Same decode + remask logic the coord used to run, just
        // executed once on the shard side.
        let entries: Vec<super::sharded::StoredValueHistoryEntry<E, P>> = raw_entries
            .into_iter()
            .map(|bytes| -> Result<super::sharded::StoredValueHistoryEntry<E, P>, AegonError> {
                let entry =
                    super::sharded::StoredValueHistoryEntry::<E, P>::deserialize_uncompressed_unchecked(
                        &bytes[..],
                    )
                    .map_err(|e| {
                        AegonError::Database(format!("decode value history entry: {e}"))
                    })?;
                self.remask_value_history_entry(entry)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Freshness opening at the latest entry's slot under THIS
        // shard's live rand_value_poly. `entries[0]` is the most
        // recently published value-change for `label`.
        let latest = &entries[0];
        let (freshness_eval, freshness_proof) =
            self.open_rand_value_at_slot_current(&latest.slot_bits)?;
        Ok(super::sharded::FullValueHistory {
            entries,
            freshness_eval: Some(freshness_eval),
            freshness_proof: Some(freshness_proof),
        })
    }

    /// Read the `StoredLabelPlacement<E,P>` bytes for `label` from
    /// this shard's DB. `Ok(None)` when missing.
    fn fetch_label_placement(
        &self,
        _label: &super::types::Label,
    ) -> Result<Option<Vec<u8>>, AegonError> {
        Ok(None)
    }

    /// Combined label-history RPC: collapses placement fetch + live
    /// rand_index opening into a single shard round-trip. Used by
    /// `ShardedAegon::lookup_label_history`.
    ///
    /// The default implementation composes `fetch_label_placement` +
    /// `open_rand_index_at_slot_current` so the in-process transport
    /// works without extra plumbing. The placement carries an empty
    /// `placement_merkle_path` — the coord stitches that in from its
    /// `epoch_commits` cache.
    fn fetch_full_label_history(
        &self,
        label: &super::types::Label,
    ) -> Result<Option<super::sharded::FullLabelHistory<E, P>>, AegonError> {
        let Some(bytes) = self.fetch_label_placement(label)? else {
            return Ok(None);
        };
        let placement =
            super::sharded::StoredLabelPlacement::<E, P>::deserialize_uncompressed_unchecked(
                &bytes[..],
            )
            .map_err(|e| AegonError::Database(format!("decode label placement: {e}")))?;
        let (freshness_eval, freshness_proof) =
            self.open_rand_index_at_slot_current(&placement.slot_bits)?;
        Ok(Some(super::sharded::FullLabelHistory {
            placement,
            freshness_eval,
            freshness_proof,
        }))
    }

    /// Read the `HistoryOpenings<E,P>` bytes for `epoch` from this
    /// shard's DB. `Ok(None)` when missing.
    fn fetch_history_openings(&self, _epoch: u64) -> Result<Option<Vec<u8>>, AegonError> {
        Ok(None)
    }
}

// ---------- in-process impl: Aegon directly is a ShardHandle -----------

impl<E, P, H> ShardHandle<E, P, H> for Aegon<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync,
    P::ProverParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Send + Sync,
    P::VerifierParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Send + Sync,
    P::Commitment: Clone
        + Send
        + Sync
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Sub<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    P::Proof: Clone + Send + Sync,
    P::State: Send + Sync,
    P::Polynomial: Send + Sync,
    P::Point: Send + Sync,
    P::Evaluation: Send + Sync,
    H: HashSuite<E::ScalarField> + Send + Sync,
{
    fn publish_phase_1_at_slots(
        &mut self,
        batch: &[ShardWrite<E::ScalarField>],
    ) -> Result<(P::Commitment, P::Commitment), AegonError> {
        Aegon::publish_phase_1_at_slots(self, batch)
    }

    fn publish_phase_2_and_persist(
        &mut self,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
        shard_id: u32,
    ) -> Result<EpochCommitment<E, P>, AegonError> {
        // Stream each chunk into this shard's own store, mirroring what
        // `ShardServiceImpl` does for a remote shard. The DB is moved
        // out for the duration so the sink can borrow it while
        // `publish_phase_2_and_persist` holds `&mut self`, then put
        // back -- including on the error path.
        //
        // With no DB attached (`DbSource::None`) the sink discards, which
        // is the historical in-memory behaviour every unit test relies on.
        let db = self.take_db();
        let res =
            Aegon::publish_phase_2_and_persist(self, new_r_index, new_r_value, shard_id, |chunk| {
                match &db {
                    Some(d) => d.write_atomic(chunk),
                    None => Ok(()),
                }
            });
        self.restore_db(db);
        res
    }

    fn fetch_value(&self, label: &super::types::Label) -> Result<Option<Vec<u8>>, AegonError> {
        match self.db() {
            Some(db) => db.get(&super::db::key_value(label)),
            None => Ok(None),
        }
    }

    fn fetch_value_history(&self, label: &super::types::Label) -> Result<Vec<Vec<u8>>, AegonError> {
        match self.db() {
            Some(db) => db.lrange(&super::db::key_value_history(label), 0, -1),
            None => Ok(Vec::new()),
        }
    }

    fn fetch_label_placement(
        &self,
        label: &super::types::Label,
    ) -> Result<Option<Vec<u8>>, AegonError> {
        match self.db() {
            Some(db) => db.get(&super::db::key_label_placement(label)),
            None => Ok(None),
        }
    }

    fn is_index_slot_occupied(&self, slot_bits: &[bool]) -> bool {
        Aegon::is_index_slot_occupied(self, slot_bits)
    }

    fn open_index_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_index_at_slot(self, slot_bits)
    }

    fn open_value_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_value_at_slot(self, slot_bits)
    }

    fn open_rand_index_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_rand_index_at_slot_in_epoch(self, slot_bits, epoch)
    }

    fn open_rand_value_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_rand_value_at_slot_in_epoch(self, slot_bits, epoch)
    }

    fn open_rand_value_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_rand_value_at_slot_current(self, slot_bits)
    }

    fn open_rand_index_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        Aegon::open_rand_index_at_slot_current(self, slot_bits)
    }

    fn remask_value_history_entry(
        &self,
        entry: super::sharded::StoredValueHistoryEntry<E, P>,
    ) -> Result<super::sharded::StoredValueHistoryEntry<E, P>, AegonError> {
        // Under a non-hiding SRS, publish_phase_2 already produced
        // plain non-ZK openings and the stored entries verify
        // directly. Pass-through.
        if !self.is_zk_srs() {
            return Ok(entry);
        }
        // Under a hiding SRS:
        //   - publish_phase_2 stored plain non-ZK proofs (cheap)
        //   - this shard has the per-epoch `tau_f` for value-side
        //     polys (kept in `EpochSnapshot.{value_tau,
        //     rand_value_tau}` independently of `retain_epoch_polys`)
        //   - the masking server (or an inline fallback) provides a
        //     fresh `MaskingPackage`
        // Combine the three to mint a hiding opening per stored
        // proof. Each `remask_value_side_proof` call internally
        // fetches/generates ONE masking package; we pay 3 packages
        // per entry. That's still ~5× cheaper than the inline-ZK
        // path at publish time because we only pay it on the
        // (presumably rare) history-lookup path, not on every batch
        // of 16K labels.
        let prev_epoch = entry.epoch.saturating_sub(1);
        let post_epoch = entry.epoch;
        // Fetch all three tau snapshots up front (cheap reads).
        let rand_value_pre_tau = self.rand_value_tau_at_epoch(prev_epoch).ok_or_else(|| {
            AegonError::Config(format!(
                "remask_value_history_entry: rand_value tau missing for epoch {prev_epoch}"
            ))
        })?;
        let rand_value_post_tau = self.rand_value_tau_at_epoch(post_epoch).ok_or_else(|| {
            AegonError::Config(format!(
                "remask_value_history_entry: rand_value tau missing for epoch {post_epoch}"
            ))
        })?;
        let value_post_tau = self.value_tau_at_epoch(post_epoch).ok_or_else(|| {
            AegonError::Config(format!(
                "remask_value_history_entry: value tau missing for epoch {post_epoch}"
            ))
        })?;

        // The three remasks are independent — each takes its own
        // masking package, has its own commitment + tau, and writes
        // its own output proof. The three commitments / evaluations /
        // proofs / taus are disjoint, so rayon can run them in
        // parallel. With an inline-generated masking package this
        // saves ~2/3 of the wall (~50 ms → ~17 ms per entry).
        //
        // Fetch all three packages FIRST, here on the calling thread.
        // Fetching blocks on the masking pool's channel, and the
        // producers filling that channel need the global rayon pool to
        // build a package. Fetching from inside the join below parks
        // every rayon worker waiting for a package that can only be
        // produced once a worker frees up — a circular wait, and the
        // hang that made the private-mode tests flaky. Only pure
        // compute goes inside the join.
        let pkg_rand_pre = self.fetch_masking_package()?;
        let pkg_rand_post = self.fetch_masking_package()?;
        let pkg_value_post = self.fetch_masking_package()?;
        // A non-hiding SRS needs no package and `remask_*` returns the
        // plain proof; the dummy is never read in that case.
        let with_pkg = |pkg: &Option<P::MaskingPackage>,
                        commitment: &P::Commitment,
                        eval: &E::ScalarField,
                        proof: P::Proof,
                        tau: &P::HidingScalar,
                        label: &'static [u8]| match pkg {
            Some(pkg) => self.remask_value_side_proof_with_package(
                commitment,
                &entry.slot_bits,
                eval,
                proof,
                tau,
                label,
                pkg,
            ),
            None => Ok(proof),
        };
        let ((rand_value_pre_proof, rand_value_post_proof), value_post_proof) = rayon::join(
            || {
                rayon::join(
                    || {
                        with_pkg(
                            &pkg_rand_pre,
                            &entry.prev_shard_commit.rand_value_commitment,
                            &entry.rand_value_pre_eval,
                            entry.rand_value_pre_proof.clone(),
                            &rand_value_pre_tau,
                            b"aegon.rand_value.open",
                        )
                    },
                    || {
                        with_pkg(
                            &pkg_rand_post,
                            &entry.post_shard_commit.rand_value_commitment,
                            &entry.rand_value_post_eval,
                            entry.rand_value_post_proof.clone(),
                            &rand_value_post_tau,
                            b"aegon.rand_value.open",
                        )
                    },
                )
            },
            || {
                with_pkg(
                    &pkg_value_post,
                    &entry.post_shard_commit.value_commitment,
                    &entry.value_post_eval,
                    entry.value_post_proof.clone(),
                    &value_post_tau,
                    b"aegon.value.open",
                )
            },
        );
        let rand_value_pre_proof = rand_value_pre_proof?;
        let rand_value_post_proof = rand_value_post_proof?;
        let value_post_proof = value_post_proof?;
        Ok(super::sharded::StoredValueHistoryEntry {
            rand_value_pre_proof,
            rand_value_post_proof,
            value_post_proof,
            ..entry
        })
    }

    fn current_commitment(&self) -> EpochCommitment<E, P> {
        Aegon::current_commitment(self)
    }

    fn verifier_context(&self) -> VerifierContext<E, P> {
        Aegon::verifier_context(self)
    }

    fn prefill_random_in_place(&mut self, count: usize, seed: u64) -> Result<(), AegonError> {
        use ark_std::rand::SeedableRng;
        // Reset back to epoch-0 first so multi-stage fill-percent
        // sweeps can call this repeatedly on the same Aegon —
        // matches the new gRPC ReconfigurePrefill semantics
        // ("wipe and refill"). For a freshly-init-ed Aegon, this is
        // a no-op (zero polys, zero commitments stay zero).
        Aegon::reset_state(self)?;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        // `shard_id`/`db_source` arguments to Aegon::prefill_random are
        // legacy — the implementation ignores both (the comment on
        // server.rs:573 spells it out). Pass `DbSource::None` + `0`.
        Aegon::prefill_random(self, &mut rng, count, &super::DbSource::None, 0)
    }

    fn clear_dictionary(&mut self) -> Result<(), AegonError> {
        Aegon::clear_dictionary(self)
    }

    fn publish_batch(
        &mut self,
        batch: &[(super::types::Label, super::types::Value)],
        vrf_proofs_shard_per_label: &[Vec<Vec<u8>>],
    ) -> Result<super::server::PublishBatchOutcome<E, P>, AegonError>
    where
        P::Commitment: Clone,
    {
        Aegon::publish_batch(self, batch, vrf_proofs_shard_per_label)
    }

    fn find_label_slot(
        &self,
        label: &super::types::Label,
    ) -> Result<Option<(Vec<bool>, u64)>, AegonError> {
        Ok(Aegon::find_label_slot(self, label))
    }

    fn fetch_label_proof_trail(
        &self,
        label: &super::types::Label,
    ) -> Result<Option<super::sharded::LabelProofTrail<E, P>>, AegonError> {
        // In-process specialization: walk H_slot once and open the
        // index polynomial at every probed slot in the same pass. We
        // could lean on the trait default, but that re-walks H_slot
        // for every probe — wasteful even in process. Mirrors the
        // exact loop the coord used to run via N×OpenIndexAtSlot.
        let Some((final_slot_bits, slot_ctr0)) = Aegon::find_label_slot(self, label) else {
            return Ok(None);
        };
        let log_capacity = Aegon::log_capacity(self);
        let mut entries: Vec<super::sharded::LabelProofTrailEntry<E, P>> =
            Vec::with_capacity((slot_ctr0 as usize) + 1);
        for ctr in 0..=slot_ctr0 {
            let slot_bits = if ctr == slot_ctr0 {
                final_slot_bits.clone()
            } else {
                H::h_slot(ctr, label, log_capacity)
            };
            let (evaluation, proof) = Aegon::open_index_at_slot(self, &slot_bits)?;
            entries.push(super::sharded::LabelProofTrailEntry {
                slot_bits,
                evaluation,
                proof,
            });
        }
        // If a VRF prover is configured and `publish_batch` cached
        // proofs for this label, ship them with the trail so the coord
        // can skip its prove_h_shard / prove_h_slot calls. Empty
        // vectors signal "no cache; recompute" to the coord.
        let (vrf_proofs_shard, vrf_proofs_slot) = match self.label_vrf_proofs(label) {
            Some(cached) => (
                cached.vrf_proofs_shard.clone(),
                cached.vrf_proofs_slot.clone(),
            ),
            None => (Vec::new(), Vec::new()),
        };
        Ok(Some(super::sharded::LabelProofTrail {
            final_slot_bits,
            slot_ctr0,
            entries,
            vrf_proofs_shard,
            vrf_proofs_slot,
        }))
    }

    fn log_capacity(&self) -> usize {
        Aegon::log_capacity(self)
    }
}

// ---------- gRPC server: wraps an Aegon, exposes 8 RPCs ----------------

/// Server-side TLS material. Build via [`Self::from_pem_files`].
#[derive(Clone)]
pub struct ShardServerTlsConfig {
    inner: ServerTlsConfig,
}

impl ShardServerTlsConfig {
    /// Load a PEM-encoded server certificate + matching private key
    /// from disk. The cert may be a chain (concatenated PEM blocks);
    /// the key must be the matching leaf.
    pub fn from_pem_files(cert_path: &Path, key_path: &Path) -> Result<Self, AegonError> {
        let cert = std::fs::read(cert_path).map_err(|e| {
            AegonError::Config(format!("read tls cert '{}': {e}", cert_path.display()))
        })?;
        let key = std::fs::read(key_path).map_err(|e| {
            AegonError::Config(format!("read tls key '{}': {e}", key_path.display()))
        })?;
        let identity = Identity::from_pem(cert, key);
        Ok(Self {
            inner: ServerTlsConfig::new().identity(identity),
        })
    }
}

/// Server-side adapter. Wraps an `Aegon` instance behind an async
/// `RwLock` so the read-only RPCs (open_*, is_index_slot_occupied,
/// current_commitment) execute concurrently. Mutating RPCs
/// (publish_phase_1/phase_2) take the write lock and serialize as
/// expected.
pub struct ShardServer<E, P, H = Sha256Hash>
where
    E: Pairing,
    P: AegonPcs<E>,
    H: HashSuite<E::ScalarField>,
{
    aegon: Arc<AsyncRwLock<Aegon<E, P, H>>>,
    /// Optional DB client (Redis or RocksDB, depending on
    /// `DbSource`). When set, the shard writes its
    /// [`AegonCheckpoint`] to `aegon:shard:{shard_id}:state` after
    /// every successful `publish_phase_2`. Powers shard-restart
    /// recovery (`aegon_shard_server --db-url` or `--db-path`).
    db: Option<Arc<dyn Db>>,
    shard_id: u32,
}

impl<E, P, H> ShardServer<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Send + Sync + 'static,
    P::VerifierParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Clone + Send + Sync + 'static,
    P::Commitment: CanonicalSerialize
        + CanonicalDeserialize
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Sub<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    P::Proof: CanonicalSerialize + Send + Sync + 'static,
    P::State: Send + Sync + 'static,
    P::Polynomial: Send + Sync + 'static,
    P::Point: Send + Sync + 'static,
    P::Evaluation: Send + Sync + 'static,
    H: HashSuite<E::ScalarField> + Send + Sync + 'static,
    EpochCommitment<E, P>: CanonicalSerialize + Send + Sync + 'static,
{
    /// Serve a shard with no attached store. Reads that need one
    /// (values, history, placements) will report a missing DB.
    pub fn new(aegon: Aegon<E, P, H>) -> Self {
        Self {
            aegon: Arc::new(AsyncRwLock::new(aegon)),
            db: None,
            shard_id: 0,
        }
    }

    /// Same as `new`, but also wires up the DB-backed durability
    /// barrier — every successful `publish_phase_2` writes the new
    /// `AegonCheckpoint` to `aegon:shard:{shard_id}:state`. The
    /// shard server binary calls this when started with `--db-url`
    /// (Redis) or `--db-path` (RocksDB).
    pub fn new_with_checkpoint(
        aegon: Aegon<E, P, H>,
        db_source: DbSource,
        shard_id: u32,
    ) -> Result<Self, AegonError> {
        let db: Arc<dyn Db> = match db_source {
            DbSource::None => {
                return Err(AegonError::Config(
                    "ShardServer::new_with_checkpoint requires a Redis or Rocks DbSource".into(),
                ));
            }
            DbSource::Redis(url) => Arc::new(RedisDb::connect(&url)?),
            DbSource::Rocks(path) => Arc::new(crate::aegon::db::RocksDb::open(&path)?),
        };
        Ok(Self {
            aegon: Arc::new(AsyncRwLock::new(aegon)),
            db: Some(db),
            shard_id,
        })
    }

    /// Bind and serve indefinitely on `addr` (plaintext HTTP/2).
    pub async fn serve(self, addr: std::net::SocketAddr) -> Result<(), tonic::transport::Error> {
        let service = Self::wrap_service(self);
        Self::tuned_builder().add_service(service).serve(addr).await
    }

    /// Bind and serve indefinitely on `addr` with TLS. The
    /// coordinator-side client connects via
    /// [`GrpcShardClientConfig::with_tls_ca`].
    pub async fn serve_with_tls(
        self,
        addr: std::net::SocketAddr,
        tls: ShardServerTlsConfig,
    ) -> Result<(), tonic::transport::Error> {
        let service = Self::wrap_service(self);
        Self::tuned_builder()
            .tls_config(tls.inner)?
            .add_service(service)
            .serve(addr)
            .await
    }

    /// Server builder with HTTP/2 flow-control + stream concurrency
    /// tuned for the coord↔shard path. Patch 10 (2026-06-23): the
    /// tonic 64 KB defaults capped sustained per-connection
    /// throughput at ~960 qps under GCP RTT. The coord opens ONE
    /// channel per shard, so this affects every coord→shard request
    /// the cluster makes (lookups + publishes). Same conn=64 MiB /
    /// stream=16 MiB tuning as the coordinator server — see
    /// `CoordinatorServer::tuned_builder` for the derivation.
    fn tuned_builder() -> Server {
        Server::builder()
            .initial_connection_window_size(64 * 1024 * 1024)
            .initial_stream_window_size(16 * 1024 * 1024)
            .max_concurrent_streams(Some(4096))
    }

    /// Wrap `self` as a tonic service with the message-size limits the
    /// cluster needs. Tonic defaults to 4 MiB, which is fine for tiny
    /// batches but breaks publish_phase_1 once §6.4 openings per shard
    /// cross that threshold (happens around batch=10k for a 4-shard
    /// cluster). Set to 1 GiB on both directions so the limit is
    /// effectively never reached.
    fn wrap_service(this: Self) -> ShardServiceServer<Self> {
        const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;
        ShardServiceServer::new(this)
            .max_decoding_message_size(MAX_MSG_BYTES)
            .max_encoding_message_size(MAX_MSG_BYTES)
    }
}

#[tonic::async_trait]
impl<E, P, H> ShardService for ShardServer<E, P, H>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync + 'static,
    P::ProverParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Send + Sync + 'static,
    P::VerifierParam: akd_core::aegon_crypto::pcs::PCSGlobalParam + Clone + Send + Sync + 'static,
    P::Commitment: CanonicalSerialize
        + CanonicalDeserialize
        + Clone
        + Send
        + Sync
        + 'static
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Sub<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    P::Proof: CanonicalSerialize + CanonicalDeserialize + Clone + Send + Sync + 'static,
    P::State: Send + Sync + 'static,
    P::Polynomial: Send + Sync + 'static,
    P::Point: Send + Sync + 'static,
    P::Evaluation: Send + Sync + 'static,
    H: HashSuite<E::ScalarField> + Send + Sync + 'static,
    EpochCommitment<E, P>: CanonicalSerialize + CanonicalDeserialize + Send + Sync + 'static,
    AegonCheckpoint<E, P>: CanonicalSerialize + Send + Sync + 'static,
{
    async fn publish_phase1_at_slots(
        &self,
        req: Request<PublishPhase1Request>,
    ) -> Result<Response<PublishPhase1Response>, Status> {
        let batch: Vec<ShardWrite<E::ScalarField>> =
            decode(&req.into_inner().batch_bytes).map_err(err_to_status)?;
        let mut aegon = self.aegon.write().await;
        let (index_com, value_com) = aegon
            .publish_phase_1_at_slots(&batch)
            .map_err(err_to_status)?;
        Ok(Response::new(PublishPhase1Response {
            index_commitment: encode(&index_com).map_err(err_to_status)?,
            value_commitment: encode(&value_com).map_err(err_to_status)?,
        }))
    }

    async fn publish_phase2_and_persist(
        &self,
        req: Request<PublishPhase2AndPersistRequest>,
    ) -> Result<Response<PublishPhase2AndPersistResponse>, Status> {
        let r = req.into_inner();
        let r_index: E::ScalarField = decode(&r.r_index).map_err(err_to_status)?;
        let r_value: E::ScalarField = decode(&r.r_value).map_err(err_to_status)?;
        let shard_id = r.shard_id;

        // Clone the Arc<dyn Db> handle so the streaming closure owns
        // a stable reference for the whole publish (cheap — Arc::clone
        // is one atomic refcount bump). The closure is `FnMut` and
        // called many times by `publish_phase_2_and_persist`, once
        // per chunk.
        let db_handle = self.db.clone();

        let mut aegon = self.aegon.write().await;
        let commit = aegon
            .publish_phase_2_and_persist(r_index, r_value, shard_id, |chunk| {
                // Streaming chunk write: each call lands as one RocksDB
                // WriteBatch (one WAL record + one memtable insert).
                // No-op when no DB is configured (in-process tests).
                if let Some(db) = &db_handle {
                    db.write_atomic(chunk)?;
                }
                Ok(())
            })
            .map_err(err_to_status)?;

        // TODO(shard-checkpoint-fault-tolerance): per-publish
        // checkpoint write into the shard's own local DB is
        // **intentionally disabled** in the current architecture.
        // The system-wide DB now belongs to the coordinator alone;
        // shards don't communicate with any DB during steady-state
        // operation. The code structure below is preserved for the
        // future fault-tolerance feature: when re-enabled, each
        // shard would persist its own `(rand_index_poly,
        // rand_value_poly, KZH aux states)` delta to a private
        // local RocksDB after every phase_2, slot-keyed for fast
        // delta writes (~few hundred bytes per publish per shard
        // at 2^34 scale) and prefix-scan recovery on restart.
        //
        // Until that's enabled, a shard process restart loses its
        // polynomial state and the shard has to be re-driven from
        // scratch by the coord. Acceptable for bench/research where
        // shards rarely restart mid-run; not acceptable for prod.
        const SHARD_CHECKPOINT_ENABLED: bool = false;
        if SHARD_CHECKPOINT_ENABLED {
            if let Some(db) = &self.db {
                let ckpt = aegon.capture_checkpoint();
                let bytes = encode(&ckpt).map_err(err_to_status)?;
                db.write_atomic(&[DbOp::Set {
                    key: key_shard_state(self.shard_id),
                    value: bytes,
                }])
                .map_err(err_to_status)?;
            }
        }
        drop(aegon);

        Ok(Response::new(PublishPhase2AndPersistResponse {
            epoch_commitment: encode(&commit).map_err(err_to_status)?,
        }))
    }

    async fn is_index_slot_occupied(
        &self,
        req: Request<SlotRequest>,
    ) -> Result<Response<SlotOccupiedResponse>, Status> {
        let slot: Vec<bool> = decode(&req.into_inner().slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        Ok(Response::new(SlotOccupiedResponse {
            occupied: aegon.is_index_slot_occupied(&slot),
        }))
    }

    async fn open_index_at_slot(
        &self,
        req: Request<SlotRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let slot: Vec<bool> = decode(&req.into_inner().slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon.open_index_at_slot(&slot).map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn open_value_at_slot(
        &self,
        req: Request<SlotRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let slot: Vec<bool> = decode(&req.into_inner().slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon.open_value_at_slot(&slot).map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn open_rand_index_at_slot_in_epoch(
        &self,
        req: Request<SlotEpochRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let r = req.into_inner();
        let slot: Vec<bool> = decode(&r.slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon
            .open_rand_index_at_slot_in_epoch(&slot, r.epoch)
            .map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn open_rand_value_at_slot_in_epoch(
        &self,
        req: Request<SlotEpochRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let r = req.into_inner();
        let slot: Vec<bool> = decode(&r.slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon
            .open_rand_value_at_slot_in_epoch(&slot, r.epoch)
            .map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn open_rand_value_at_slot_current(
        &self,
        req: Request<SlotRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let slot: Vec<bool> = decode(&req.into_inner().slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon
            .open_rand_value_at_slot_current(&slot)
            .map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn open_rand_index_at_slot_current(
        &self,
        req: Request<SlotRequest>,
    ) -> Result<Response<OpenResponse>, Status> {
        let slot: Vec<bool> = decode(&req.into_inner().slot_bits).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let (eval, proof) = aegon
            .open_rand_index_at_slot_current(&slot)
            .map_err(err_to_status)?;
        Ok(Response::new(OpenResponse {
            evaluation: encode(&eval).map_err(err_to_status)?,
            proof: encode(&proof).map_err(err_to_status)?,
        }))
    }

    async fn remask_value_history_entry(
        &self,
        req: Request<proto::RemaskValueHistoryEntryRequest>,
    ) -> Result<Response<proto::RemaskValueHistoryEntryResponse>, Status> {
        let entry: super::sharded::StoredValueHistoryEntry<E, P> =
            decode(&req.into_inner().entry_uncompressed).map_err(err_to_status)?;
        let aegon = self.aegon.read().await;
        let remasked = ShardHandle::<E, P, H>::remask_value_history_entry(&*aegon, entry)
            .map_err(err_to_status)?;
        Ok(Response::new(proto::RemaskValueHistoryEntryResponse {
            entry_uncompressed: encode(&remasked).map_err(err_to_status)?,
        }))
    }

    async fn current_commitment(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<CommitmentResponse>, Status> {
        let aegon = self.aegon.read().await;
        let commit = aegon.current_commitment();
        Ok(Response::new(CommitmentResponse {
            epoch_commitment: encode(&commit).map_err(err_to_status)?,
        }))
    }

    async fn get_verifier_context(
        &self,
        _req: Request<Empty>,
    ) -> Result<Response<proto::GetVerifierContextResponse>, Status> {
        let aegon = self.aegon.read().await;
        let vp = ShardHandle::<E, P, H>::verifier_context(&*aegon).verifier_param;
        let log_capacity = ShardHandle::<E, P, H>::log_capacity(&*aegon);
        Ok(Response::new(proto::GetVerifierContextResponse {
            verifier_param: encode(&vp).map_err(err_to_status)?,
            log_capacity: log_capacity as u64,
        }))
    }

    async fn reconfigure_prefill(
        &self,
        req: Request<ReconfigurePrefillRequest>,
    ) -> Result<Response<ReconfigurePrefillResponse>, Status> {
        let r = req.into_inner();
        let mut aegon = self.aegon.write().await;
        // Reset wipes the in-memory Aegon back to a fresh epoch-0
        // and recomputes the empty-poly commitments. Then prefill
        // sprays `count` random entries into the data polynomials.
        aegon.reset_state().map_err(err_to_status)?;
        if r.count > 0 {
            use ark_std::rand::SeedableRng;
            let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(r.seed);
            // shard_id / db_source are ignored by Aegon::prefill_random.
            aegon
                .prefill_random(&mut rng, r.count as usize, &super::DbSource::None, 0)
                .map_err(err_to_status)?;
        }
        Ok(Response::new(ReconfigurePrefillResponse {}))
    }

    async fn clear_dictionary(&self, _req: Request<Empty>) -> Result<Response<Empty>, Status> {
        let mut aegon = self.aegon.write().await;
        // Restores from the in-memory `setup_baseline` — see
        // `Aegon::clear_dictionary` on the server side. Cheap relative
        // to `reset_state` because the empty-poly commitments are
        // already stashed rather than re-derived. Fails if a publish
        // is in flight on this shard.
        aegon.clear_dictionary().map_err(err_to_status)?;
        Ok(Response::new(Empty {}))
    }

    async fn publish_batch(
        &self,
        req: Request<PublishBatchRequest>,
    ) -> Result<Response<PublishBatchResponse>, Status> {
        let inner = req.into_inner();
        let batch: Vec<(super::types::Label, super::types::Value)> =
            decode(&inner.batch_bytes).map_err(err_to_status)?;
        // Empty bytes signal "no per-label H_shard proofs shipped"
        // (legacy callers, or VRF disabled). Decoding an empty blob
        // gives an empty Vec, which `Aegon::publish_batch` treats as
        // "skip the VRF cache".
        let vrf_proofs_shard_per_label: Vec<Vec<Vec<u8>>> =
            if inner.vrf_proofs_shard_bytes.is_empty() {
                Vec::new()
            } else {
                decode(&inner.vrf_proofs_shard_bytes).map_err(err_to_status)?
            };
        let mut aegon = self.aegon.write().await;
        let outcome = aegon
            .publish_batch(&batch, &vrf_proofs_shard_per_label)
            .map_err(err_to_status)?;
        let is_full = outcome.fullness_proof.is_some();
        let fullness_proof = outcome.fullness_proof.unwrap_or_default();
        Ok(Response::new(PublishBatchResponse {
            index_commitment: encode(&outcome.index_commitment).map_err(err_to_status)?,
            value_commitment: encode(&outcome.value_commitment).map_err(err_to_status)?,
            placed_count: outcome.placed_count as u64,
            placements_bytes: encode(&outcome.placements).map_err(err_to_status)?,
            is_full,
            fullness_proof,
        }))
    }

    async fn find_label_slot(
        &self,
        req: Request<FindLabelSlotRequest>,
    ) -> Result<Response<FindLabelSlotResponse>, Status> {
        let label = req.into_inner().label;
        let aegon = self.aegon.read().await;
        match Aegon::find_label_slot(&aegon, &label) {
            Some((slot_bits, slot_ctr)) => Ok(Response::new(FindLabelSlotResponse {
                found: true,
                slot_bits: encode(&slot_bits).map_err(err_to_status)?,
                slot_ctr,
            })),
            None => Ok(Response::new(FindLabelSlotResponse {
                found: false,
                slot_bits: Vec::new(),
                slot_ctr: 0,
            })),
        }
    }

    async fn fetch_label_proof_trail(
        &self,
        req: Request<FetchLabelProofTrailRequest>,
    ) -> Result<Response<FetchLabelProofTrailResponse>, Status> {
        let label = req.into_inner().label;
        let aegon = self.aegon.read().await;
        // Reuse the in-process ShardHandle impl — it walks H_slot
        // once and emits an opening per probed slot in the same pass.
        let trail = ShardHandle::<E, P, H>::fetch_label_proof_trail(&*aegon, &label)
            .map_err(err_to_status)?;
        let Some(trail) = trail else {
            return Ok(Response::new(FetchLabelProofTrailResponse {
                found: false,
                trail_uncompressed: Vec::new(),
            }));
        };
        Ok(Response::new(FetchLabelProofTrailResponse {
            found: true,
            trail_uncompressed: encode(&trail).map_err(err_to_status)?,
        }))
    }

    // ---------- per-shard durable state (post-refactor) ----------
    //
    // Each shard owns the dictionary content for the labels routed to
    // it. Coord ships ops via ApplyPersistenceOps at the end of every
    // publish; lookups read back via the four Fetch* RPCs.

    async fn apply_persistence_ops(
        &self,
        req: Request<ApplyPersistenceOpsRequest>,
    ) -> Result<Response<ApplyPersistenceOpsResponse>, Status> {
        let ops_bytes = req.into_inner().ops_bytes;
        let ops = DbOp::decode_batch(&ops_bytes).map_err(err_to_status)?;
        let db = self.db.as_ref().ok_or_else(|| {
            Status::failed_precondition(
                "shard has no DB configured — start aegon_shard_server with --db-path",
            )
        })?;
        db.write_atomic(&ops).map_err(err_to_status)?;
        Ok(Response::new(ApplyPersistenceOpsResponse {}))
    }

    async fn fetch_value(
        &self,
        req: Request<FetchValueRequest>,
    ) -> Result<Response<FetchValueResponse>, Status> {
        let label = req.into_inner().label;
        let db = self.db.as_ref().ok_or_else(|| {
            Status::failed_precondition("shard has no DB configured for FetchValue")
        })?;
        let v = db.get(&key_value(&label)).map_err(err_to_status)?;
        Ok(Response::new(match v {
            Some(value) => FetchValueResponse { found: true, value },
            None => FetchValueResponse {
                found: false,
                value: Vec::new(),
            },
        }))
    }

    async fn fetch_value_history(
        &self,
        req: Request<FetchValueHistoryRequest>,
    ) -> Result<Response<FetchValueHistoryResponse>, Status> {
        let label = req.into_inner().label;
        let db = self.db.as_ref().ok_or_else(|| {
            Status::failed_precondition("shard has no DB configured for FetchValueHistory")
        })?;
        let entries = db
            .lrange(&key_value_history(&label), 0, -1)
            .map_err(err_to_status)?;
        Ok(Response::new(FetchValueHistoryResponse { entries }))
    }

    async fn fetch_full_value_history(
        &self,
        req: Request<proto::FetchFullValueHistoryRequest>,
    ) -> Result<Response<proto::FetchFullValueHistoryResponse>, Status> {
        let label = req.into_inner().label;
        let db = self.db.as_ref().ok_or_else(|| {
            Status::failed_precondition("shard has no DB configured for FetchFullValueHistory")
        })?;
        // Read sliding window directly from this shard's DB.
        let mut raw_entries = db
            .lrange(&key_value_history(&label), 0, -1)
            .map_err(err_to_status)?;
        if raw_entries.len() > super::sharded::HISTORY_WINDOW {
            raw_entries.truncate(super::sharded::HISTORY_WINDOW);
        }
        if raw_entries.is_empty() {
            return Ok(Response::new(proto::FetchFullValueHistoryResponse {
                found: false,
                history_uncompressed: Vec::new(),
            }));
        }
        // Decode + remask + freshness opening — all under a single
        // Aegon read lock so we don't reacquire it per entry.
        let aegon = self.aegon.read().await;
        let entries: Vec<super::sharded::StoredValueHistoryEntry<E, P>> = raw_entries
            .into_iter()
            .map(|bytes| -> Result<super::sharded::StoredValueHistoryEntry<E, P>, AegonError> {
                let entry =
                    super::sharded::StoredValueHistoryEntry::<E, P>::deserialize_uncompressed_unchecked(
                        &bytes[..],
                    )
                    .map_err(|e| {
                        AegonError::Database(format!("decode value history entry: {e}"))
                    })?;
                ShardHandle::<E, P, H>::remask_value_history_entry(&*aegon, entry)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(err_to_status)?;
        let latest = &entries[0];
        let (freshness_eval, freshness_proof) = aegon
            .open_rand_value_at_slot_current(&latest.slot_bits)
            .map_err(err_to_status)?;
        let full = super::sharded::FullValueHistory::<E, P> {
            entries,
            freshness_eval: Some(freshness_eval),
            freshness_proof: Some(freshness_proof),
        };
        Ok(Response::new(proto::FetchFullValueHistoryResponse {
            found: true,
            history_uncompressed: encode(&full).map_err(err_to_status)?,
        }))
    }

    async fn fetch_label_placement(
        &self,
        req: Request<FetchLabelPlacementRequest>,
    ) -> Result<Response<FetchLabelPlacementResponse>, Status> {
        let label = req.into_inner().label;
        let db = self.db.as_ref().ok_or_else(|| {
            Status::failed_precondition("shard has no DB configured for FetchLabelPlacement")
        })?;
        let placement_bytes = db
            .get(&key_label_placement(&label))
            .map_err(err_to_status)?;
        Ok(Response::new(match placement_bytes {
            Some(bytes) => FetchLabelPlacementResponse {
                found: true,
                placement_bytes: bytes,
            },
            None => FetchLabelPlacementResponse {
                found: false,
                placement_bytes: Vec::new(),
            },
        }))
    }

    async fn fetch_full_label_history(
        &self,
        req: Request<proto::FetchFullLabelHistoryRequest>,
    ) -> Result<Response<proto::FetchFullLabelHistoryResponse>, Status> {
        let label = req.into_inner().label;
        let db = self.db.as_ref().ok_or_else(|| {
            Status::failed_precondition("shard has no DB configured for FetchFullLabelHistory")
        })?;
        let Some(placement_bytes) = db
            .get(&key_label_placement(&label))
            .map_err(err_to_status)?
        else {
            return Ok(Response::new(proto::FetchFullLabelHistoryResponse {
                found: false,
                full_uncompressed: Vec::new(),
            }));
        };
        let placement =
            super::sharded::StoredLabelPlacement::<E, P>::deserialize_uncompressed_unchecked(
                &placement_bytes[..],
            )
            .map_err(|e| Status::internal(format!("decode label placement: {e}")))?;
        let aegon = self.aegon.read().await;
        let (freshness_eval, freshness_proof) = aegon
            .open_rand_index_at_slot_current(&placement.slot_bits)
            .map_err(err_to_status)?;
        let full = super::sharded::FullLabelHistory::<E, P> {
            placement,
            freshness_eval,
            freshness_proof,
        };
        Ok(Response::new(proto::FetchFullLabelHistoryResponse {
            found: true,
            full_uncompressed: encode(&full).map_err(err_to_status)?,
        }))
    }

    async fn fetch_history_openings(
        &self,
        req: Request<FetchHistoryOpeningsRequest>,
    ) -> Result<Response<FetchHistoryOpeningsResponse>, Status> {
        let epoch = req.into_inner().epoch;
        let db = self.db.as_ref().ok_or_else(|| {
            Status::failed_precondition("shard has no DB configured for FetchHistoryOpenings")
        })?;
        let bytes = db
            .get(&key_history_openings_local(epoch))
            .map_err(err_to_status)?;
        Ok(Response::new(match bytes {
            Some(openings_bytes) => FetchHistoryOpeningsResponse {
                found: true,
                openings_bytes,
            },
            None => FetchHistoryOpeningsResponse {
                found: false,
                openings_bytes: Vec::new(),
            },
        }))
    }
}

// ---------- gRPC client: implements ShardHandle ------------------------

/// Blocking client adapter. Holds a dedicated tokio runtime + a
/// tonic client. Each `ShardHandle` method is a sync wrapper that
/// `block_on`s the underlying async RPC. The runtime is owned by the
/// adapter; we never share it with the application's runtime, so
/// there's no "block_on inside a runtime" deadlock risk.
///
/// `verifier_context` is cached at construction time: it doesn't
/// change over the shard's lifetime, and the coordinator queries it
/// once at setup.
/// Retry policy for the client. Every RPC retries on transport-level
/// failures (server unreachable, broken connection); the body of a
/// `Status::internal` reply — which is what surfaces a protocol-level
/// error from the shard's `Aegon` — is **not** retried, since those
/// errors are deterministic.
#[derive(Clone, Debug)]
pub struct RetryPolicy {
    /// Total attempts including the first; `1` disables retrying.
    pub max_attempts: usize,
    /// Delay before the first retry.
    pub initial_backoff: Duration,
    /// Ceiling the exponential backoff is clamped to.
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            initial_backoff: Duration::from_millis(50),
            max_backoff: Duration::from_secs(2),
        }
    }
}

impl RetryPolicy {
    /// A policy that never retries.
    pub fn off() -> Self {
        Self {
            max_attempts: 1,
            initial_backoff: Duration::from_millis(0),
            max_backoff: Duration::from_millis(0),
        }
    }
}

/// Connection options for [`GrpcShardClient::connect_with`].
pub struct GrpcShardClientConfig<E: Pairing, P: AegonPcs<E>> {
    /// Shard address, e.g. `http://10.0.0.7:50051`.
    pub endpoint: String,
    /// Verifier bundle for checking what this shard returns.
    pub verifier_context: VerifierContext<E, P>,
    /// Log2 of the shard's slot count. Must match the server's.
    pub log_capacity: usize,
    /// PEM-encoded CA bundle used to verify the shard's server cert.
    /// `None` → plaintext HTTP/2 (matches the server side default).
    pub tls_ca_pem: Option<Vec<u8>>,
    /// Optional SNI / domain name override for TLS. Useful when the
    /// endpoint URL holds an IP but the cert is for a hostname.
    pub tls_domain: Option<String>,
    /// How transport failures are retried.
    pub retry: RetryPolicy,
    /// Connect timeout; `None` uses tonic's default.
    pub connect_timeout: Option<Duration>,
}

impl<E: Pairing, P: AegonPcs<E>> GrpcShardClientConfig<E, P> {
    /// A plaintext config with the default retry policy and no TLS.
    pub fn new(
        endpoint: String,
        verifier_context: VerifierContext<E, P>,
        log_capacity: usize,
    ) -> Self {
        Self {
            endpoint,
            verifier_context,
            log_capacity,
            tls_ca_pem: None,
            tls_domain: None,
            retry: RetryPolicy::default(),
            connect_timeout: Some(Duration::from_secs(10)),
        }
    }

    /// Enable TLS, verifying the shard against this PEM-encoded CA.
    pub fn with_tls_ca(mut self, ca_pem: Vec<u8>) -> Self {
        self.tls_ca_pem = Some(ca_pem);
        self
    }

    /// Override the domain matched against the shard's certificate.
    pub fn with_tls_domain(mut self, domain: impl Into<String>) -> Self {
        self.tls_domain = Some(domain.into());
        self
    }

    /// Replace the retry policy.
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }
}

/// A remote shard, driven over gRPC. Implements [`ShardHandle`] so a
/// `ShardedAegon` can treat it exactly like an in-process shard.
pub struct GrpcShardClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E>,
{
    runtime: Arc<Runtime>,
    /// Cloneable HTTP/2 channel. Each RPC clones it cheaply (Arc-bump
    /// internally) and constructs a fresh `ShardServiceClient` from
    /// the clone — HTTP/2 multiplexes the streams over the underlying
    /// connection, so concurrent calls run in parallel without the
    /// `Mutex<Client>` serialization the old design imposed.
    channel: Channel,
    cached_verifier_context: VerifierContext<E, P>,
    cached_log_capacity: usize,
    retry: RetryPolicy,
}

impl<E, P> GrpcShardClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync,
    P::Commitment: CanonicalDeserialize + Send + Sync,
    P::VerifierParam: Clone + Send + Sync,
{
    /// Convenience: plaintext connect with default retry.
    pub fn connect(
        endpoint: String,
        verifier_context: VerifierContext<E, P>,
        log_capacity: usize,
    ) -> Result<Self, AegonError> {
        Self::connect_with(GrpcShardClientConfig::new(
            endpoint,
            verifier_context,
            log_capacity,
        ))
    }

    /// Connect with full configurability (TLS, timeouts, retries).
    /// Creates its own private 64-worker runtime — fine for tests and
    /// one-off clients, but **don't** call this in a loop for many
    /// shards: 128 × 64 = 8K worker threads on a 16-CPU coord
    /// thrashes the OS scheduler and caps throughput on context
    /// switches instead of real work. Use
    /// [`build_shared_runtime`] +
    /// [`Self::connect_with_runtime`] from the coord instead.
    pub fn connect_with(cfg: GrpcShardClientConfig<E, P>) -> Result<Self, AegonError> {
        let runtime = Arc::new(build_shared_runtime()?);
        Self::connect_with_runtime(cfg, runtime)
    }

    /// Same as [`Self::connect_with`] but reuses a caller-supplied
    /// runtime instead of spawning a fresh one. The coord builds ONE
    /// runtime up-front and passes it to all 128 shard clients —
    /// without this, each `connect_with` would mint its own 64-worker
    /// runtime and the coord would land at ~8K threads on a 16-CPU
    /// VM, where OS-scheduler context switching dominates real work
    /// and caps lookup throughput in the ~1 kqps range. With one
    /// shared runtime the thread count drops to ~64 + main-runtime
    /// workers + the blocking pool.
    pub fn connect_with_runtime(
        cfg: GrpcShardClientConfig<E, P>,
        runtime: Arc<Runtime>,
    ) -> Result<Self, AegonError> {
        // Patch 10: HTTP/2 flow-control windows. Tonic defaults of
        // 64 KB conn / 64 KB stream capped the coord↔shard channel
        // at ~960 sustained RPCs/s per shard under GCP RTT.
        // Publishes + lookups both go through this channel, so the
        // cap matters for the whole cluster. See
        // `ShardServer::tuned_builder` for the matched server side.
        let mut endpoint = tonic::transport::Endpoint::from_shared(cfg.endpoint.clone())
            .map_err(|e| AegonError::Config(format!("endpoint '{}': {e}", cfg.endpoint)))?
            .initial_connection_window_size(64 * 1024 * 1024)
            .initial_stream_window_size(16 * 1024 * 1024);
        if let Some(t) = cfg.connect_timeout {
            endpoint = endpoint.connect_timeout(t);
        }
        if let Some(ca_pem) = &cfg.tls_ca_pem {
            let mut tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(ca_pem));
            if let Some(domain) = &cfg.tls_domain {
                tls = tls.domain_name(domain);
            }
            endpoint = endpoint
                .tls_config(tls)
                .map_err(|e| AegonError::Config(format!("tls config: {e}")))?;
        }

        let channel = runtime
            .block_on(endpoint.connect())
            .map_err(|e| AegonError::Config(format!("connect '{}': {e}", cfg.endpoint)))?;

        Ok(Self {
            runtime,
            channel,
            cached_verifier_context: cfg.verifier_context,
            cached_log_capacity: cfg.log_capacity,
            retry: cfg.retry,
        })
    }

    /// Construct a fresh client from the shared channel. The clone is
    /// cheap (an Arc-bump on the underlying HTTP/2 connection); each
    /// caller gets its own `&mut self` so concurrent RPCs multiplex
    /// natively over the connection instead of serializing on a
    /// `Mutex<Client>`. See `ShardServer::wrap_service` for the 8 GiB
    /// message-size rationale (same trigger as the old call sites).
    fn client(&self) -> ShardServiceClient<Channel> {
        const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;
        ShardServiceClient::new(self.channel.clone())
            .max_decoding_message_size(MAX_MSG_BYTES)
            .max_encoding_message_size(MAX_MSG_BYTES)
    }
}

/// Build the tokio runtime that backs all of one coord's shard
/// clients. The fan-out workload (lookup, plan_phase_1, audit) is
/// purely I/O-bound — each call awaits a sub-ms gRPC RTT — and tonic
/// multiplexes any number of in-flight streams over a single
/// connection. 64 workers leaves headroom for parallel publish
/// fan-out (~16K calls per shard during a fresh-cluster start) while
/// keeping total coord thread count bounded.
pub fn build_shared_runtime() -> Result<Runtime, AegonError> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(64)
        .enable_all()
        .thread_name("aegon-grpc-rt")
        .build()
        .map_err(|e| AegonError::Config(format!("tokio runtime: {e}")))
}

/// One-shot fetch of a shard's `(VerifierContext, log_capacity)` over
/// gRPC. Used by the coordinator at setup time so it doesn't have to
/// regenerate the (multi-GB, multi-minute) full SRS just to derive
/// its small verifier-side projection. Each call opens its own
/// throwaway runtime + channel — cheap because this runs once per
/// cluster boot.
///
/// Plaintext only — the bench/intra-cluster traffic is already
/// implicitly trusted (same VPC), matching the masking-server
/// rationale.
pub fn fetch_verifier_context_from_endpoint<E, P>(
    endpoint: String,
) -> Result<(VerifierContext<E, P>, usize), AegonError>
where
    E: Pairing,
    P: AegonPcs<E>,
    P::VerifierParam: CanonicalDeserialize + Clone,
{
    let runtime = Runtime::new().map_err(|e| AegonError::Config(format!("tokio runtime: {e}")))?;
    let ep = tonic::transport::Endpoint::from_shared(endpoint.clone())
        .map_err(|e| AegonError::Config(format!("endpoint '{endpoint}': {e}")))?
        .connect_timeout(Duration::from_secs(10));
    let channel = runtime
        .block_on(ep.connect())
        .map_err(|e| AegonError::Config(format!("connect '{endpoint}': {e}")))?;
    const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;
    let mut client = ShardServiceClient::new(channel)
        .max_decoding_message_size(MAX_MSG_BYTES)
        .max_encoding_message_size(MAX_MSG_BYTES);
    let resp = runtime
        .block_on(client.get_verifier_context(Request::new(Empty {})))
        .map_err(status_to_err)?
        .into_inner();
    let verifier_param: P::VerifierParam = decode(&resp.verifier_param)?;
    let log_capacity = resp.log_capacity as usize;
    let vctx = VerifierContext::new(log_capacity, verifier_param);
    Ok((vctx, log_capacity))
}

// `impl GrpcShardClient` continues below with `with_retry` and
// the rest of the client methods.
impl<E, P> GrpcShardClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync,
    P::Commitment: CanonicalDeserialize + Send + Sync,
    P::VerifierParam: Clone + Send + Sync,
{
    /// Run `op` against a fresh client, retrying on transport-level
    /// failures only (`Status::code() == Unavailable | Unknown`). The
    /// `op` closure is async, awaited from this method's owned tokio
    /// runtime. Each attempt receives its own `ShardServiceClient`
    /// constructed from the shared channel, so concurrent in-flight
    /// retries do not serialize on a shared mutex. Retries use
    /// exponential backoff capped at `retry.max_backoff`.
    fn with_retry<T, Fut, F>(&self, mut op: F) -> Result<T, AegonError>
    where
        F: FnMut(ShardServiceClient<Channel>) -> Fut,
        Fut: std::future::Future<Output = Result<T, Status>>,
    {
        self.runtime.block_on(async {
            let mut backoff = self.retry.initial_backoff;
            let mut last_err: Option<Status> = None;
            for attempt in 0..self.retry.max_attempts {
                match op(self.client()).await {
                    Ok(v) => return Ok(v),
                    Err(s) => {
                        let retriable =
                            matches!(s.code(), tonic::Code::Unavailable | tonic::Code::Unknown);
                        last_err = Some(s);
                        if !retriable || attempt + 1 == self.retry.max_attempts {
                            break;
                        }
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(self.retry.max_backoff);
                    }
                }
            }
            Err(status_to_err(last_err.expect("at least one attempt")))
        })
    }
}

impl<E, P, H> ShardHandle<E, P, H> for GrpcShardClient<E, P>
where
    E: Pairing,
    P: AegonPcs<E> + Send + Sync,
    P::ProverParam: Send + Sync,
    P::VerifierParam: Clone + Send + Sync,
    P::Commitment: CanonicalSerialize
        + CanonicalDeserialize
        + Clone
        + Send
        + Sync
        + std::ops::Add<Output = P::Commitment>
        + std::ops::Sub<Output = P::Commitment>
        + std::ops::Mul<E::ScalarField, Output = P::Commitment>,
    P::Proof: CanonicalDeserialize + Clone + Send + Sync,
    P::State: Send + Sync,
    P::Polynomial: Send + Sync,
    P::Point: Send + Sync,
    P::Evaluation: Send + Sync,
    H: HashSuite<E::ScalarField> + Send + Sync,
    EpochCommitment<E, P>: CanonicalDeserialize + Send + Sync,
{
    fn publish_phase_1_at_slots(
        &mut self,
        batch: &[ShardWrite<E::ScalarField>],
    ) -> Result<(P::Commitment, P::Commitment), AegonError> {
        let req = PublishPhase1Request {
            batch_bytes: encode(&batch.to_vec())?,
        };
        let resp = self.runtime.block_on(async {
            self.client()
                .publish_phase1_at_slots(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        let idx: P::Commitment = decode(&inner.index_commitment)?;
        let val: P::Commitment = decode(&inner.value_commitment)?;
        Ok((idx, val))
    }

    fn publish_phase_2_and_persist(
        &mut self,
        new_r_index: E::ScalarField,
        new_r_value: E::ScalarField,
        shard_id: u32,
    ) -> Result<EpochCommitment<E, P>, AegonError> {
        let req = PublishPhase2AndPersistRequest {
            r_index: encode(&new_r_index)?,
            r_value: encode(&new_r_value)?,
            shard_id,
        };
        let resp = self.runtime.block_on(async {
            self.client()
                .publish_phase2_and_persist(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        let commit: EpochCommitment<E, P> = decode(&inner.epoch_commitment)?;
        Ok(commit)
    }

    fn is_index_slot_occupied(&self, slot_bits: &[bool]) -> bool {
        // Hot path: plan_phase_1_batches calls this once per probe
        // (~16K times per fresh-cluster publish). Each call gets a
        // fresh Channel-backed client so concurrent probes multiplex
        // over the same HTTP/2 connection without serializing on a
        // shared mutex.
        let req = SlotRequest {
            slot_bits: encode(&slot_bits.to_vec()).expect("slot_bits encode"),
        };
        let mut client = self.client();
        let result = self
            .runtime
            .block_on(async move { client.is_index_slot_occupied(req).await });
        match result {
            Ok(resp) => resp.into_inner().occupied,
            // RPC unreachable; default to "occupied" so the
            // coordinator's open-addressing loop doesn't claim a slot
            // we can't actually verify.
            Err(_) => true,
        }
    }

    fn open_index_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
        };
        let resp = self.with_retry(move |mut client| {
            let req = req.clone();
            async move { client.open_index_at_slot(req).await }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn open_value_at_slot(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
        };
        let resp = self.with_retry(move |mut client| {
            let req = req.clone();
            async move { client.open_value_at_slot(req).await }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn open_rand_index_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotEpochRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
            epoch,
        };
        let resp = self.with_retry(move |mut client| {
            let req = req.clone();
            async move { client.open_rand_index_at_slot_in_epoch(req).await }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn open_rand_value_at_slot_in_epoch(
        &self,
        slot_bits: &[bool],
        epoch: u64,
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotEpochRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
            epoch,
        };
        let resp = self.with_retry(move |mut client| {
            let req = req.clone();
            async move { client.open_rand_value_at_slot_in_epoch(req).await }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn open_rand_value_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
        };
        let resp = self.with_retry(move |mut client| {
            let req = req.clone();
            async move { client.open_rand_value_at_slot_current(req).await }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn open_rand_index_at_slot_current(
        &self,
        slot_bits: &[bool],
    ) -> Result<(E::ScalarField, P::Proof), AegonError> {
        let req = SlotRequest {
            slot_bits: encode(&slot_bits.to_vec())?,
        };
        let resp = self.with_retry(move |mut client| {
            let req = req.clone();
            async move { client.open_rand_index_at_slot_current(req).await }
        })?;
        let inner = resp.into_inner();
        Ok((decode(&inner.evaluation)?, decode(&inner.proof)?))
    }

    fn remask_value_history_entry(
        &self,
        entry: super::sharded::StoredValueHistoryEntry<E, P>,
    ) -> Result<super::sharded::StoredValueHistoryEntry<E, P>, AegonError> {
        let req = proto::RemaskValueHistoryEntryRequest {
            entry_uncompressed: encode(&entry)?,
        };
        let resp = self.with_retry(move |mut client| {
            let req = req.clone();
            async move { client.remask_value_history_entry(req).await }
        })?;
        decode(&resp.into_inner().entry_uncompressed)
    }

    fn current_commitment(&self) -> EpochCommitment<E, P> {
        self.with_retry(|mut client| async move { client.current_commitment(Empty {}).await })
            .and_then(|r| decode(&r.into_inner().epoch_commitment))
            .expect("current_commitment RPC")
    }

    fn verifier_context(&self) -> VerifierContext<E, P> {
        self.cached_verifier_context.clone()
    }

    fn log_capacity(&self) -> usize {
        self.cached_log_capacity
    }

    fn prefill_random_in_place(&mut self, count: usize, seed: u64) -> Result<(), AegonError> {
        let req = ReconfigurePrefillRequest {
            count: count as u64,
            seed,
        };
        let _ = self.with_retry(move |mut client| {
            let req = req;
            async move { client.reconfigure_prefill(req).await }
        })?;
        Ok(())
    }

    fn clear_dictionary(&mut self) -> Result<(), AegonError> {
        let _ =
            self.with_retry(|mut client| async move { client.clear_dictionary(Empty {}).await })?;
        Ok(())
    }

    fn publish_batch(
        &mut self,
        batch: &[(super::types::Label, super::types::Value)],
        vrf_proofs_shard_per_label: &[Vec<Vec<u8>>],
    ) -> Result<super::server::PublishBatchOutcome<E, P>, AegonError>
    where
        P::Commitment: Clone,
    {
        // Ship the H_shard proofs alongside the batch so the shard can
        // cache them for lookup. Encoding an empty slice as empty bytes
        // keeps the wire small when VRF is not configured cluster-wide.
        let vrf_proofs_shard_bytes: Vec<u8> = if vrf_proofs_shard_per_label.is_empty() {
            Vec::new()
        } else {
            encode(&vrf_proofs_shard_per_label.to_vec())?
        };
        let req = PublishBatchRequest {
            batch_bytes: encode(&batch.to_vec())?,
            vrf_proofs_shard_bytes,
        };
        let resp = self.runtime.block_on(async {
            self.client()
                .publish_batch(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        let index_commitment: P::Commitment = decode(&inner.index_commitment)?;
        let value_commitment: P::Commitment = decode(&inner.value_commitment)?;
        let placements: Vec<super::server::ShardPlacement> = decode(&inner.placements_bytes)?;
        let fullness_proof = if inner.is_full {
            Some(inner.fullness_proof)
        } else {
            None
        };
        Ok(super::server::PublishBatchOutcome {
            placements,
            placed_count: inner.placed_count as usize,
            index_commitment,
            value_commitment,
            fullness_proof,
        })
    }

    fn find_label_slot(
        &self,
        label: &super::types::Label,
    ) -> Result<Option<(Vec<bool>, u64)>, AegonError> {
        let req = FindLabelSlotRequest {
            label: label.clone(),
        };
        let resp = self.runtime.block_on(async {
            self.client()
                .find_label_slot(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        if !inner.found {
            return Ok(None);
        }
        let slot_bits: Vec<bool> = decode(&inner.slot_bits)?;
        Ok(Some((slot_bits, inner.slot_ctr)))
    }

    fn fetch_label_proof_trail(
        &self,
        label: &super::types::Label,
    ) -> Result<Option<super::sharded::LabelProofTrail<E, P>>, AegonError> {
        let req = FetchLabelProofTrailRequest {
            label: label.clone(),
        };
        let resp = self.runtime.block_on(async {
            self.client()
                .fetch_label_proof_trail(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        if !inner.found {
            return Ok(None);
        }
        let trail: super::sharded::LabelProofTrail<E, P> = decode(&inner.trail_uncompressed)?;
        Ok(Some(trail))
    }

    fn apply_persistence_ops(&self, ops_bytes: &[u8]) -> Result<(), AegonError> {
        let req = ApplyPersistenceOpsRequest {
            ops_bytes: ops_bytes.to_vec(),
        };
        let _resp = self.runtime.block_on(async {
            self.client()
                .apply_persistence_ops(req)
                .await
                .map_err(status_to_err)
        })?;
        Ok(())
    }

    fn fetch_value(&self, label: &super::types::Label) -> Result<Option<Vec<u8>>, AegonError> {
        let req = FetchValueRequest {
            label: label.clone(),
        };
        let resp = self
            .runtime
            .block_on(async { self.client().fetch_value(req).await.map_err(status_to_err) })?;
        let inner = resp.into_inner();
        Ok(if inner.found { Some(inner.value) } else { None })
    }

    fn fetch_value_history(&self, label: &super::types::Label) -> Result<Vec<Vec<u8>>, AegonError> {
        let req = FetchValueHistoryRequest {
            label: label.clone(),
        };
        let resp = self.runtime.block_on(async {
            self.client()
                .fetch_value_history(req)
                .await
                .map_err(status_to_err)
        })?;
        Ok(resp.into_inner().entries)
    }

    fn fetch_full_value_history(
        &self,
        label: &super::types::Label,
    ) -> Result<super::sharded::FullValueHistory<E, P>, AegonError> {
        let req = proto::FetchFullValueHistoryRequest {
            label: label.clone(),
        };
        let resp = self.runtime.block_on(async {
            self.client()
                .fetch_full_value_history(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        if !inner.found {
            return Ok(super::sharded::FullValueHistory {
                entries: Vec::new(),
                freshness_eval: None,
                freshness_proof: None,
            });
        }
        let full: super::sharded::FullValueHistory<E, P> = decode(&inner.history_uncompressed)?;
        Ok(full)
    }

    fn fetch_label_placement(
        &self,
        label: &super::types::Label,
    ) -> Result<Option<Vec<u8>>, AegonError> {
        let req = FetchLabelPlacementRequest {
            label: label.clone(),
        };
        let resp = self.runtime.block_on(async {
            self.client()
                .fetch_label_placement(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        Ok(if inner.found {
            Some(inner.placement_bytes)
        } else {
            None
        })
    }

    fn fetch_full_label_history(
        &self,
        label: &super::types::Label,
    ) -> Result<Option<super::sharded::FullLabelHistory<E, P>>, AegonError> {
        let req = proto::FetchFullLabelHistoryRequest {
            label: label.clone(),
        };
        let resp = self.runtime.block_on(async {
            self.client()
                .fetch_full_label_history(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        if !inner.found {
            return Ok(None);
        }
        let full: super::sharded::FullLabelHistory<E, P> = decode(&inner.full_uncompressed)?;
        Ok(Some(full))
    }

    fn fetch_history_openings(&self, epoch: u64) -> Result<Option<Vec<u8>>, AegonError> {
        let req = FetchHistoryOpeningsRequest { epoch };
        let resp = self.runtime.block_on(async {
            self.client()
                .fetch_history_openings(req)
                .await
                .map_err(status_to_err)
        })?;
        let inner = resp.into_inner();
        Ok(if inner.found {
            Some(inner.openings_bytes)
        } else {
            None
        })
    }
}

// Unused but useful to anchor the type aliases at module-scope.
#[allow(dead_code)]
type _Label = Label;
#[allow(dead_code)]
type _Value = Value;
