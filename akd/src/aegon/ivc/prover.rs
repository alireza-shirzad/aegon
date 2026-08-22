//! Folding driver: turn a sequence of published epoch commitments
//! into a single Nova recursive proof.
//!
//! ## Who runs this
//!
//! Anyone. Every input the folding step consumes — the four
//! commitments per shard *and* each shard's Schnorr blinding-equality
//! proof — is already published in
//! [`ShardedEpochCommitment`](crate::aegon::ShardedEpochCommitment).
//! No server secret is involved, so the IVC prover is a detached
//! process tailing the bulletin board, not a privileged component of
//! the key server. A deployment can run it itself as a convenience,
//! or a watchdog can run it independently and publish the proof; a
//! sceptical auditor can even re-run it from scratch.
//!
//! ## Step accounting
//!
//! Nova's `RecursiveSNARK::new` *already performs the first step*
//! using the circuit handed to it, and the subsequent first call to
//! `prove_step` is a deliberate no-op that only advances the counter.
//! Getting this wrong silently drops or double-counts an epoch, so
//! [`IvcAuditProver`] hides it: the SNARK is constructed lazily on the
//! first [`fold_epoch`](IvcAuditProver::fold_epoch) and the step
//! counter is maintained explicitly, always equal to Nova's own `i`.

use std::sync::Arc;

use ark_bn254::G1Affine as ArkG1Affine;
use nova_snark::{
    nova::{PublicParams, RecursiveSNARK},
    provider::{poseidon::PoseidonConstantsCircuit, Bn256EngineIPA, GrumpkinEngine},
    traits::snark::{default_ck_hint, RelaxedR1CSSNARKTrait},
};

/// Which terminal SNARK a set of public parameters is built for.
///
/// The two need different commitment-key sizes, so this is fixed at
/// [`IvcAuditParams::setup_with`] time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerminalSnark {
    /// `spartan::snark` -- no floor on the commitment key.
    #[default]
    Spartan,
    /// `spartan::ppsnark`, the SNARK from the MicroNova paper --
    /// needs a key large enough to commit to the R1CS matrices.
    MicroNova,
}

use super::bridge::CircuitField;
use super::circuit::{AuditStepCircuit, AuditStepWitness, SigmaWitness, ARITY};
use super::fs_poseidon::{ro_constants, FsParams, ShardCommitments};
use crate::aegon::error::AegonError;

/// Primary Nova engine. Grumpkin, so the step circuit's field is
/// BN254's base field and point arithmetic stays native — see
/// [`super::bridge`].
pub type E1 = GrumpkinEngine;
/// Secondary Nova engine, completing the cycle.
pub type E2 = Bn256EngineIPA;

/// The recursive proof an auditor verifies.
pub type AuditProof = RecursiveSNARK<E1, E2, AuditStepCircuit>;

/// Nova public parameters plus the shape template they were derived
/// from.
///
/// Expensive to build (it commits to the R1CS shape), deterministic
/// in `(params, h)`, and reusable across every prover and verifier in
/// a deployment — so build once and share behind the [`Arc`] that
/// [`IvcAuditProver`] holds.
///
/// The SRS generator `h` is baked into the circuit as a constant, so
/// it is covered by the parameters' digest: a proof verified under
/// these parameters is necessarily a proof about *this* SRS.
pub struct IvcAuditParams {
    pp: PublicParams<E1, E2, AuditStepCircuit>,
    template: AuditStepCircuit,
    fs_params: FsParams,
    ro_consts: PoseidonConstantsCircuit<CircuitField>,
}

impl IvcAuditParams {
    /// Derive parameters for a directory of the given shape.
    ///
    /// `h` is the hiding generator from the KZH SRS verifier key
    /// (`vk.get_h()`), which is what the value-chain Schnorr proofs
    /// are stated against.
    pub fn setup(fs_params: FsParams, h: ArkG1Affine) -> Result<Self, AegonError> {
        Self::setup_with(fs_params, h, TerminalSnark::Spartan)
    }

    /// [`Self::setup`], choosing which terminal SNARK the parameters
    /// will support.
    ///
    /// This is a **setup-time** decision, not a swap at compression
    /// time. MicroNova's verifier holds a commitment to the R1CS
    /// matrices, so its commitment key must be large enough to commit
    /// to them (`A + B + C` nonzeros); plain Spartan puts no floor on
    /// the key at all. Parameters built for one will fail
    /// compression setup for the other with
    /// `InvalidCommitmentKeyLength`.
    pub fn setup_with(
        fs_params: FsParams,
        h: ArkG1Affine,
        terminal: TerminalSnark,
    ) -> Result<Self, AegonError> {
        if fs_params.n_shards == 0 {
            return Err(AegonError::Config(
                "IVC audit setup requires at least one shard".into(),
            ));
        }
        let ro_consts = ro_constants();
        let template = AuditStepCircuit::new(fs_params, ro_consts.clone(), h, None);
        let pp = match terminal {
            TerminalSnark::Spartan => {
                PublicParams::setup(&template, &*default_ck_hint(), &*default_ck_hint())
            },
            TerminalSnark::MicroNova => PublicParams::setup(
                &template,
                &*<S1Pp as RelaxedR1CSSNARKTrait<E1>>::ck_floor(),
                &*<S2Pp as RelaxedR1CSSNARKTrait<E2>>::ck_floor(),
            ),
        }
        .map_err(|e| AegonError::Ivc(format!("public parameter setup failed: {e}")))?;
        Ok(Self {
            pp,
            template,
            fs_params,
            ro_consts,
        })
    }

    /// The Nova public parameters. Auditors need these to verify.
    pub fn public_params(&self) -> &PublicParams<E1, E2, AuditStepCircuit> {
        &self.pp
    }

    /// The deployment shape these parameters were derived for.
    pub fn fs_params(&self) -> FsParams {
        self.fs_params
    }

    /// Poseidon constants shared by the circuit and the native
    /// digest helpers.
    pub fn ro_consts(&self) -> &PoseidonConstantsCircuit<CircuitField> {
        &self.ro_consts
    }

    /// Number of R1CS constraints in one folding step. Reported by
    /// the bench; also a cheap guard that the circuit is the size the
    /// design expects.
    pub fn constraints_per_step(&self) -> usize {
        self.pp.num_constraints().0
    }

    /// Constraints in the **secondary** (BN254) instance.
    ///
    /// Nova's cycle puts the step circuit on the primary and a small
    /// fixed folding-verifier circuit on the secondary. Compression
    /// proves both and `CompressedSNARK::verify` checks them under a
    /// `rayon::join`, so total verify time is `max(S1, S2)` -- which
    /// makes the ratio of these two numbers the thing that decides
    /// whether improving the secondary's PCS could ever matter.
    pub fn secondary_constraints_per_step(&self) -> usize {
        self.pp.num_constraints().1
    }

    fn circuit_for(&self, witness: AuditStepWitness) -> AuditStepCircuit {
        self.template.with_witness(witness)
    }
}

/// Incremental folding state: one step per epoch transition.
pub struct IvcAuditProver {
    params: Arc<IvcAuditParams>,
    snark: Option<AuditProof>,
    z0: Vec<CircuitField>,
    /// Commitments of the epoch the chain currently ends at. Kept so
    /// callers only supply the *new* epoch and cannot accidentally
    /// fold a discontinuous pair.
    current: Vec<ShardCommitments>,
    steps: usize,
}

impl IvcAuditProver {
    /// Start a chain at the genesis epoch.
    ///
    /// `genesis` is epoch 0's per-shard commitments — under
    /// `Server.Init` these are commitments to zero polynomials, i.e.
    /// the identity. The initial state is
    /// `z0 = [0, 0, 0, Poseidon(genesis)]`; a verifier reconstructs
    /// the same `z0` from the same public genesis commitments.
    pub fn new(
        params: Arc<IvcAuditParams>,
        genesis: &[ShardCommitments],
    ) -> Result<Self, AegonError> {
        let z0 = super::verifier::initial_state(&params, genesis)?;
        Ok(Self {
            params,
            snark: None,
            z0,
            current: genesis.to_vec(),
            steps: 0,
        })
    }

    /// Fold one epoch transition into the proof.
    ///
    /// `next` and `sigma` are the newly published epoch's per-shard
    /// commitments and value-chain Schnorr proofs. The previous epoch
    /// is whatever the chain currently ends at, so consecutive calls
    /// cannot skip an epoch.
    pub fn fold_epoch(
        &mut self,
        next: &[ShardCommitments],
        sigma: &[SigmaWitness],
    ) -> Result<(), AegonError> {
        let n = self.params.fs_params.n_shards;
        if next.len() != n || sigma.len() != n {
            return Err(AegonError::Ivc(format!(
                "expected {n} shards and {n} sigma proofs, got {} and {}",
                next.len(),
                sigma.len()
            )));
        }
        let witness = AuditStepWitness {
            prev: self.current.clone(),
            next: next.to_vec(),
            sigma: sigma.to_vec(),
        };
        let circuit = self.params.circuit_for(witness);

        match self.snark.as_mut() {
            None => {
                // `new` performs step 1 itself; the immediately
                // following `prove_step` is Nova's documented no-op
                // that just advances `i` to 1.
                let mut snark =
                    RecursiveSNARK::new(self.params.public_params(), &circuit, &self.z0).map_err(
                        |e| AegonError::Ivc(format!("initialising recursive SNARK failed: {e}")),
                    )?;
                snark
                    .prove_step(self.params.public_params(), &circuit)
                    .map_err(|e| AegonError::Ivc(format!("first fold failed: {e}")))?;
                self.snark = Some(snark);
            },
            Some(snark) => {
                snark
                    .prove_step(self.params.public_params(), &circuit)
                    .map_err(|e| {
                        AegonError::Ivc(format!("folding epoch {} failed: {e}", self.steps + 1))
                    })?;
            },
        }

        self.current = next.to_vec();
        self.steps += 1;
        debug_assert_eq!(
            self.steps,
            self.snark.as_ref().map(|s| s.num_steps()).unwrap_or(0),
            "local step counter drifted from Nova's"
        );
        Ok(())
    }

    /// The proof so far, or `None` before the first fold.
    pub fn proof(&self) -> Option<&AuditProof> {
        self.snark.as_ref()
    }

    /// Number of epoch transitions folded. Pass this as `num_steps`
    /// when verifying.
    pub fn num_steps(&self) -> usize {
        self.steps
    }

    /// The initial state the proof is anchored to.
    pub fn z0(&self) -> &[CircuitField] {
        &self.z0
    }

    /// Commitments of the epoch the chain currently ends at.
    pub fn current_epoch_commitments(&self) -> &[ShardCommitments] {
        &self.current
    }

    /// The parameters this prover folds against.
    pub fn params(&self) -> &Arc<IvcAuditParams> {
        &self.params
    }
}

/// Arity of the folded state, re-exported for callers building `z0`
/// by hand.
pub const STATE_ARITY: usize = ARITY;

/// Serialized wire size of a recursive audit proof, in bytes.
///
/// Uses the same `bincode` encoding `nova-snark` itself serializes
/// with, so the number reflects what would actually cross the network
/// to an auditor. Note this size is a function of the circuit shape,
/// **not** of the number of epochs folded — which is the property the
/// whole design exists for.
pub fn proof_size_bytes(proof: &AuditProof) -> usize {
    bincode::serde::encode_to_vec(proof, bincode::config::standard())
        .map(|v| v.len())
        .unwrap_or(0)
}

// ---------- compression ------------------------------------------------
//
// The `RecursiveSNARK` above carries the running relaxed-R1CS
// witnesses, so it is megabytes regardless of chain length. That is
// fine for a prover holding folding state, but it is not what one
// wants to publish. Compressing it with Spartan collapses it to a
// succinct proof of the same statement.
//
// Crucially, *both* sizes are independent of the number of epochs
// folded — that independence is the property the design is for. The
// compression is about the constant, not the growth.

use nova_snark::{
    nova::CompressedSNARK,
    provider::ipa_pc::EvaluationEngine,
    spartan::snark::RelaxedR1CSSNARK,
};

/// Spartan-over-IPA for the primary (Grumpkin) instance.
pub type S1 = RelaxedR1CSSNARK<E1, EvaluationEngine<E1>>;
/// Spartan-over-IPA for the secondary (BN254) instance.
pub type S2 = RelaxedR1CSSNARK<E2, EvaluationEngine<E2>>;

/// A compressed audit proof — what a deployment actually publishes.
pub type CompressedAuditProof = CompressedSNARK<E1, E2, AuditStepCircuit, S1, S2>;
/// Prover key for compression.
pub type CompressedProverKey = nova_snark::nova::ProverKey<E1, E2, AuditStepCircuit, S1, S2>;
/// Verifier key for compression. This is what auditors hold.
pub type CompressedVerifierKey = nova_snark::nova::VerifierKey<E1, E2, AuditStepCircuit, S1, S2>;

impl IvcAuditParams {
    /// Derive the compression keys. Deterministic in the public
    /// parameters, and reusable for the lifetime of a deployment.
    pub fn compression_keys(
        &self,
    ) -> Result<(CompressedProverKey, CompressedVerifierKey), AegonError> {
        CompressedSNARK::<E1, E2, AuditStepCircuit, S1, S2>::setup(&self.pp)
            .map_err(|e| AegonError::Ivc(format!("compression setup failed: {e}")))
    }
}

/// Compress a folded chain into a succinct proof.
pub fn compress(
    params: &IvcAuditParams,
    pk: &CompressedProverKey,
    proof: &AuditProof,
) -> Result<CompressedAuditProof, AegonError> {
    CompressedSNARK::<E1, E2, AuditStepCircuit, S1, S2>::prove(&params.pp, pk, proof)
        .map_err(|e| AegonError::Ivc(format!("compression failed: {e}")))
}

/// Serialized wire size of a compressed audit proof, in bytes.
pub fn compressed_proof_size_bytes(proof: &CompressedAuditProof) -> usize {
    bincode::serde::encode_to_vec(proof, bincode::config::standard())
        .map(|v| v.len())
        .unwrap_or(0)
}

// ---------- MicroNova (preprocessing) compression ----------------------
//
// `nova_snark::spartan::ppsnark` is the SNARK from the MicroNova
// paper: a *preprocessing* Spartan whose verifier holds a commitment
// to the R1CS matrices rather than re-deriving them. Its own module
// docs are explicit about when that pays off -- "beneficial when
// using a polynomial commitment scheme in which the verifier's costs
// is succinct."
//
// That caveat is load-bearing here. Our primary instance lives on
// Grumpkin, which is not pairing-friendly, so its PCS is IPA and the
// IPA verifier is *linear* in the committed vector. Preprocessing the
// matrices removes one linear term but not that one. These aliases
// exist so the claim can be measured rather than argued -- see the
// bench's `--snark-compare` mode.

/// MicroNova (preprocessing Spartan) for the primary instance.
pub type S1Pp = nova_snark::spartan::ppsnark::RelaxedR1CSSNARK<E1, EvaluationEngine<E1>>;
/// MicroNova (preprocessing Spartan) for the secondary instance.
pub type S2Pp = nova_snark::spartan::ppsnark::RelaxedR1CSSNARK<E2, EvaluationEngine<E2>>;

/// A compressed audit proof under MicroNova.
pub type CompressedAuditProofPp = CompressedSNARK<E1, E2, AuditStepCircuit, S1Pp, S2Pp>;
/// MicroNova prover key.
pub type CompressedProverKeyPp = nova_snark::nova::ProverKey<E1, E2, AuditStepCircuit, S1Pp, S2Pp>;
/// MicroNova verifier key.
pub type CompressedVerifierKeyPp =
    nova_snark::nova::VerifierKey<E1, E2, AuditStepCircuit, S1Pp, S2Pp>;

impl IvcAuditParams {
    /// Derive MicroNova compression keys. Preprocessing means this
    /// does strictly more work than [`Self::compression_keys`]: it
    /// commits to the R1CS matrices up front so the verifier need
    /// not touch them.
    pub fn compression_keys_pp(
        &self,
    ) -> Result<(CompressedProverKeyPp, CompressedVerifierKeyPp), AegonError> {
        CompressedSNARK::<E1, E2, AuditStepCircuit, S1Pp, S2Pp>::setup(&self.pp)
            .map_err(|e| AegonError::Ivc(format!("MicroNova compression setup failed: {e}")))
    }
}

/// Compress a folded chain with MicroNova.
pub fn compress_pp(
    params: &IvcAuditParams,
    pk: &CompressedProverKeyPp,
    proof: &AuditProof,
) -> Result<CompressedAuditProofPp, AegonError> {
    CompressedSNARK::<E1, E2, AuditStepCircuit, S1Pp, S2Pp>::prove(&params.pp, pk, proof)
        .map_err(|e| AegonError::Ivc(format!("MicroNova compression failed: {e}")))
}

/// Wire size of a MicroNova-compressed audit proof, in bytes.
pub fn compressed_pp_proof_size_bytes(proof: &CompressedAuditProofPp) -> usize {
    bincode::serde::encode_to_vec(proof, bincode::config::standard())
        .map(|v| v.len())
        .unwrap_or(0)
}
