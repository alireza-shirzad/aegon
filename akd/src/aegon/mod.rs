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
pub mod db;
pub mod error;
pub(crate) mod fs;
pub mod hash;
pub mod presets;
pub mod server;
pub mod shard_grpc;
pub mod sharded;
pub mod types;
pub mod verify;

pub use audit::verify_invariance;
pub use config::{AegonConfig, VerifierContext};
pub use consistency::verify_consistency;
pub use db::DbSource;
pub use error::AegonError;
pub use hash::{HashSuite, Sha256Hash};
pub use presets::optimal_kzh_k;
pub use server::Aegon;
pub use sharded::{
    probe_at, rederive_sharded_fs_scalars, verify_merkle_path, verify_sharded_consistency,
    verify_sharded_invariance, verify_sharded_lookup, EpochDigest, ShardTransport,
    ShardedAegon, ShardedAegonConfig, ShardedAegonConfigBuilder, ShardedConsistencyProof,
    ShardedEpochCommitment, ShardedInvarianceProof, ShardedLookupProof, ShardedProbe,
    ShardedRandPair, ShardedVerifierContext, SrsSource,
};
pub use types::{
    AegonPcs, AuditState, ConsistencyProof, EpochCommitment, InvarianceProof, Label, LookupProof,
    RandPair, Value,
};
pub use verify::verify_lookup;
