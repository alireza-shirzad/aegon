//! Group-sharded auditing: split the shard set into `G` independent
//! folding chains that prove in parallel.
//!
//! ## Why this is sound
//!
//! The chain scalars `r_index` / `r_value` are Fiat–Shamir challenges
//! that weight each epoch's delta in a rolling random linear
//! combination. Their only job is to be unpredictable to the server
//! *before it fixes the commitments they weight*. A transcript over
//! one group's commitments delivers that for that group's shards
//! exactly as well as a transcript over all `η` does — the global
//! transcript is a batching convenience, not a soundness requirement.
//!
//! So `G` groups run `G` independent rolling accumulators. Soundness
//! is per group, and a union bound over `G` groups costs `log2 G`
//! bits — nothing, against a 128-bit challenge.
//!
//! What the split does *not* do is let the prover choose the
//! partition: [`GroupPlan`] is derived from the deployment shape, and
//! the verifier slices the epoch tuple itself. A prover that folded a
//! different partition simply fails the digest check.
//!
//! ## What it buys
//!
//! Every phase scales down with the group size, because the groups
//! never interact:
//!
//! * **setup** is paid *once*, not `G` times — all groups have the
//!   same shape, so they share one [`IvcAuditParams`];
//! * **folding** and **compression** cost what a `η/G`-shard chain
//!   costs, and run concurrently;
//! * **peak memory** per chain drops by the same factor, which is
//!   what makes this deployable on the existing shard machines.
//!
//! The cost is proof size: `G` proofs instead of one. They are ~11 KB
//! each, so even `G = 8` stays far below one epoch of classic audit
//! bandwidth.
//!
//! ## Where the parallelism actually pays
//!
//! On a single box, folding `G` groups concurrently is close to a
//! wash: Nova already saturates the cores with rayon-parallel MSMs,
//! so `G` chains share the same cores. The win is that each group is
//! an *independent process* — in a deployment that already runs `η`
//! shard machines, each group folds on its own host and the
//! deployment's wall-clock is the per-group cost.
//!
//! [`GroupedIvcAuditProver::fold_epoch`] therefore reports per-group
//! timings rather than pretending one machine gets a `G`× speedup.

use std::sync::Arc;

use ark_bn254::G1Affine as ArkG1Affine;
use rayon::prelude::*;


use super::circuit::SigmaWitness;
use super::fs_poseidon::{FsParams, ShardCommitments};
use super::prover::{
    compress, compressed_proof_size_bytes, proof_size_bytes, AuditProof, CompressedAuditProof,
    CompressedProverKey, CompressedVerifierKey, IvcAuditParams, IvcAuditProver,
};
use super::verifier::{verify_compressed_ivc_audit, verify_ivc_audit, VerifiedAudit};
use crate::aegon::error::AegonError;

pub use crate::aegon::chain_groups::GroupPlan;

/// Nova parameters for a group-sharded deployment.
///
/// Holds **one** [`IvcAuditParams`] shared by every group: all groups
/// fold the same circuit shape, so the (expensive) shape commitment
/// is computed once. At `η = 128, G = 8` that alone takes setup from
/// 75 s to 12 s.
pub struct GroupedIvcAuditParams {
    inner: Arc<IvcAuditParams>,
    plan: GroupPlan,
}

impl GroupedIvcAuditParams {
    /// Derive parameters for a deployment of `n_shards` shards split
    /// into `groups` chains.
    ///
    /// The convenience form of [`Self::setup`], for callers holding
    /// the group count as a runtime value — a CLI flag, a config
    /// field — rather than a prebuilt [`GroupPlan`]. `groups` must
    /// divide `n_shards` and must equal the server's
    /// [`chain_groups`](crate::aegon::ShardedAegonConfig::chain_groups);
    /// see [`group_plan_from_context`](super::adapter::group_plan_from_context)
    /// for why the two are one parameter and not two.
    pub fn for_shards(
        n_shards: usize,
        groups: usize,
        num_vars: usize,
        h: ArkG1Affine,
    ) -> Result<Self, AegonError> {
        Self::setup(GroupPlan::new(n_shards, groups)?, num_vars, h)
    }

    /// Derive parameters for `plan`, with `num_vars` and the SRS
    /// generator `h` as in [`IvcAuditParams::setup`].
    pub fn setup(plan: GroupPlan, num_vars: usize, h: ArkG1Affine) -> Result<Self, AegonError> {
        let inner = IvcAuditParams::setup(
            FsParams {
                num_vars,
                n_shards: plan.shards_per_group(),
            },
            h,
        )?;
        Ok(Self {
            inner: Arc::new(inner),
            plan,
        })
    }

    /// The per-group parameters. Shared by every chain.
    pub fn inner(&self) -> &Arc<IvcAuditParams> {
        &self.inner
    }

    /// The partition these parameters describe.
    pub fn plan(&self) -> GroupPlan {
        self.plan
    }

    /// Constraints in one group's folding step.
    pub fn constraints_per_step(&self) -> usize {
        self.inner.constraints_per_step()
    }

    /// Compression keys, shared by every group for the same reason
    /// the public parameters are.
    pub fn compression_keys(
        &self,
    ) -> Result<(CompressedProverKey, CompressedVerifierKey), AegonError> {
        self.inner.compression_keys()
    }
}

/// `G` independent folding chains advanced in lockstep.
pub struct GroupedIvcAuditProver {
    plan: GroupPlan,
    params: Arc<IvcAuditParams>,
    groups: Vec<IvcAuditProver>,
}

impl GroupedIvcAuditProver {
    /// Start every group's chain at the genesis epoch.
    pub fn new(
        params: &GroupedIvcAuditParams,
        genesis: &[ShardCommitments],
    ) -> Result<Self, AegonError> {
        let plan = params.plan;
        let slices = plan.split(genesis)?;
        let groups = slices
            .into_iter()
            .map(|g| IvcAuditProver::new(params.inner.clone(), g))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            plan,
            params: params.inner.clone(),
            groups,
        })
    }

    /// Fold one epoch transition into every group's chain.
    ///
    /// The groups are folded concurrently. On a single machine that
    /// is roughly a wash against folding them in sequence, since Nova
    /// already parallelises each MSM; in a distributed deployment
    /// each group would run on its own host and this call stands in
    /// for what all of them do at once.
    pub fn fold_epoch(
        &mut self,
        next: &[ShardCommitments],
        sigma: &[SigmaWitness],
    ) -> Result<(), AegonError> {
        let next_groups = self.plan.split(next)?;
        let sigma_groups = self.plan.split(sigma)?;
        self.groups
            .par_iter_mut()
            .zip(next_groups.into_par_iter())
            .zip(sigma_groups.into_par_iter())
            .map(|((prover, n), s)| prover.fold_epoch(n, s))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(())
    }

    /// Epoch transitions folded. Identical across groups by
    /// construction, since [`fold_epoch`](Self::fold_epoch) advances
    /// all of them or none.
    pub fn num_steps(&self) -> usize {
        self.groups.first().map(|g| g.num_steps()).unwrap_or(0)
    }

    /// The partition being folded.
    pub fn plan(&self) -> GroupPlan {
        self.plan
    }

    /// Per-group folding proofs, in group order.
    pub fn proofs(&self) -> Option<Vec<&AuditProof>> {
        self.groups.iter().map(|g| g.proof()).collect()
    }

    /// Total working-state size across every group, in bytes. This is
    /// the prover's state, not anything published; the per-group
    /// figure — this divided by `G` — is what one host holds.
    pub fn folding_state_bytes(&self) -> usize {
        self.groups
            .iter()
            .filter_map(|g| g.proof())
            .map(proof_size_bytes)
            .sum()
    }

    /// Compress every group's chain into the published proof.
    ///
    /// Compression is the expensive phase (~100 µs per 1 000
    /// constraints), and it is exactly the phase that group-sharding
    /// divides: each group compresses a `η/G`-shard circuit.
    pub fn compress_all(&self, pk: &CompressedProverKey) -> Result<GroupedAuditProof, AegonError> {
        let proofs = self
            .proofs()
            .ok_or_else(|| AegonError::Ivc("nothing folded yet".into()))?;
        let per_group = proofs
            .into_par_iter()
            .map(|p| compress(&self.params, pk, p))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GroupedAuditProof {
            per_group,
            groups: self.plan.groups(),
        })
    }
}

/// What a group-sharded deployment publishes: one compressed proof
/// per group.
pub struct GroupedAuditProof {
    /// Compressed proofs in group order.
    pub per_group: Vec<CompressedAuditProof>,
    groups: usize,
}

impl GroupedAuditProof {
    /// Total wire size an auditor downloads, in bytes.
    pub fn size_bytes(&self) -> usize {
        self.per_group
            .iter()
            .map(compressed_proof_size_bytes)
            .sum()
    }

    /// Number of groups this proof covers.
    pub fn groups(&self) -> usize {
        self.groups
    }
}

/// Verify a group-sharded audit.
///
/// The auditor supplies the **full** genesis and current epoch
/// tuples; this function does the partitioning itself using the
/// deployment's [`GroupPlan`]. That is what makes coverage automatic:
/// the prover never gets to say which shards a group contained, so a
/// proof that folded a different partition — or that quietly omitted
/// a shard — fails its group's digest check.
///
/// Every group is checked; the audit succeeds only if all of them do.
pub fn verify_grouped_ivc_audit(
    params: &GroupedIvcAuditParams,
    vk: &CompressedVerifierKey,
    proof: &GroupedAuditProof,
    num_steps: usize,
    genesis: &[ShardCommitments],
    current: &[ShardCommitments],
) -> Result<VerifiedAudit, AegonError> {
    let plan = params.plan;
    if proof.per_group.len() != plan.groups() {
        return Err(AegonError::Ivc(format!(
            "audit covers {} groups, deployment has {}",
            proof.per_group.len(),
            plan.groups()
        )));
    }
    let genesis_groups = plan.split(genesis)?;
    let current_groups = plan.split(current)?;

    proof
        .per_group
        .par_iter()
        .zip(genesis_groups.into_par_iter())
        .zip(current_groups.into_par_iter())
        .map(|((p, g0), gn)| {
            let z0 = super::verifier::initial_state(&params.inner, g0)?;
            verify_compressed_ivc_audit(&params.inner, vk, p, num_steps, &z0, gn)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(VerifiedAudit {
        epochs: num_steps as u64,
    })
}

/// [`verify_grouped_ivc_audit`] against uncompressed folding proofs.
///
/// Only useful to a party that holds the prover's state — a watchdog
/// re-verifying its own fold, say. Deployments publish the compressed
/// form.
pub fn verify_grouped_folding_proofs(
    params: &GroupedIvcAuditParams,
    proofs: &[&AuditProof],
    num_steps: usize,
    genesis: &[ShardCommitments],
    current: &[ShardCommitments],
) -> Result<VerifiedAudit, AegonError> {
    let plan = params.plan;
    if proofs.len() != plan.groups() {
        return Err(AegonError::Ivc(format!(
            "audit covers {} groups, deployment has {}",
            proofs.len(),
            plan.groups()
        )));
    }
    let genesis_groups = plan.split(genesis)?;
    let current_groups = plan.split(current)?;

    proofs
        .par_iter()
        .zip(genesis_groups.into_par_iter())
        .zip(current_groups.into_par_iter())
        .map(|((p, g0), gn)| {
            let z0 = super::verifier::initial_state(&params.inner, g0)?;
            verify_ivc_audit(&params.inner, p, num_steps, &z0, gn)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(VerifiedAudit {
        epochs: num_steps as u64,
    })
}
