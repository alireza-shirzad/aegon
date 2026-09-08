// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Aegon — a transparent dictionary built generically on top of any
//! multilinear polynomial commitment scheme. Implements the publish /
//! lookup / verify slice from §6 of the Aegon paper.
//!
//! See [`Aegon`] for the entry point. The crate is generic over:
//!   - the pairing engine `E: ark_ec::pairing::Pairing`,
//!   - the polynomial commitment scheme
//!     `P: PolynomialCommitmentScheme<E, Polynomial = DenseOrSparseMLE<F>, ...>`,
//!   - the hash suite `H: HashSuite<F>` used for index assignment.
//!
//! No KZH-k specific code lives here — the only constraint on `P` is the
//! standard multilinear-PCS interface. Tests (see `tests/`) instantiate it
//! with the workspace's KZH-k implementation as a smoke check.

pub mod audit;
pub mod audit_fs;
/// Partitioning the shard set into independent Fiat-Shamir chains.
pub mod chain_groups;
pub mod config;
pub mod consistency;
pub mod coordinator_grpc;
pub mod db;
pub mod distributed_srs;
pub mod error;
pub(crate) mod fs;
pub mod hash;
pub mod instrument;
/// IVC (Nova) auditing — fold the per-epoch invariance check into a
/// single recursive proof. See the module docs for the curve-cycle
/// argument that makes it cheap.
#[cfg(feature = "ivc_audit")]
pub mod ivc;
pub mod masking;
pub mod presets;
pub mod server;
pub mod shard_grpc;
pub mod sharded;
pub mod sigma;
#[cfg(feature = "tracing_instrument")]
pub mod tracing_init;
pub mod types;
pub mod verify;

pub use audit::verify_invariance;
pub use audit_fs::{AuditFs, AuditFsHooks};
pub use chain_groups::GroupPlan;
pub use config::{
    shard_log_capacity_for_two_layer, shard_log_capacity_from_true, true_log_capacity_from_shard,
    AegonConfig, VerifierContext, LOG2_OVER_PROVISIONING_FACTOR, OVER_PROVISIONING_FACTOR,
};
pub use consistency::verify_consistency;
pub use db::DbSource;
pub use error::AegonError;
pub use hash::{
    EcVrfHash, HashSuite, Sha256Hash, VrfProver, VrfVerifier, VrfVerifyError, BENCH_VRF_SEED,
    VRF_PROOF_BYTES, VRF_PUBLIC_KEY_BYTES,
};
pub use presets::optimal_kzh_k;
pub use server::Aegon;
pub use sharded::{
    build_merkle_path, build_merkle_root_and_paths, merkle_root, probe_at,
    rederive_sharded_fs_scalars, verify_lookup_history, verify_lookup_label_history,
    verify_lookup_label_two_layer, verify_lookup_value, verify_merkle_path,
    verify_sharded_consistency_two_layer, verify_sharded_invariance,
    verify_sharded_lookup_two_layer, EpochDigest, FreshnessAttestation, FreshnessAttestationLabel,
    LabelSlot, ShardRoutingProbe, ShardSlotProbe, ShardSlotRandPair, ShardTransport, ShardedAegon,
    ShardedAegonConfig, ShardedAegonConfigBuilder, ShardedConsistencyProofTwoLayer,
    ShardedEpochCommitment, ShardedLabelHistory, ShardedLabelProofTwoLayer,
    ShardedLookupProofTwoLayer, ShardedValueHistory, ShardedValueProof, ShardedVerifierContext,
    SrsSource, StoredLabelPlacement, StoredValueHistoryEntry, VerifiedLookupHistory,
    VerifiedLookupLabelHistory, HISTORY_WINDOW,
};
pub use sigma::BlindingEqProof;
pub use types::{
    AegonPcs, AuditState, ConsistencyProof, EpochCommitment, HistoryOpeningEntry, HistoryOpenings,
    Label, LookupProof, RandPair, ShardedAuditState, Value, ValueChangeEntry,
};
pub use verify::verify_lookup;
