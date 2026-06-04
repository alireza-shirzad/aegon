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
pub mod config;
pub mod consistency;
pub mod coordinator_grpc;
pub mod db;
pub mod distributed_srs;
pub mod error;
pub(crate) mod fs;
pub mod hash;
pub mod instrument;
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
pub use config::{
    shard_log_capacity_from_true, true_log_capacity_from_shard, AegonConfig, VerifierContext,
    LOG2_OVER_PROVISIONING_FACTOR, OVER_PROVISIONING_FACTOR,
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
pub use sigma::BlindingEqProof;
pub use sharded::{
    build_merkle_path, build_merkle_root_and_paths, merkle_root, probe_at,
    rederive_sharded_fs_scalars, verify_lookup_history, verify_lookup_label,
    verify_lookup_label_history, verify_lookup_value, verify_merkle_path,
    verify_sharded_consistency, verify_sharded_invariance, verify_sharded_lookup, EpochDigest,
    FreshnessAttestation, FreshnessAttestationLabel, LabelSlot, ShardTransport, ShardedAegon,
    ShardedAegonConfig, ShardedAegonConfigBuilder, ShardedConsistencyProof,
    ShardedEpochCommitment, ShardedLabelHistory, ShardedLabelProof, ShardedLookupProof,
    ShardedProbe, ShardedRandPair, ShardedValueHistory, ShardedValueProof,
    ShardedVerifierContext, SrsSource, StoredLabelPlacement, StoredValueHistoryEntry,
    VerifiedLookupHistory, VerifiedLookupLabelHistory, HISTORY_WINDOW,
};
pub use types::{
    AegonPcs, AuditState, ConsistencyProof, EpochCommitment, HistoryOpeningEntry, HistoryOpenings,
    Label, LookupProof, RandPair, Value, ValueChangeEntry,
};
pub use verify::verify_lookup;
