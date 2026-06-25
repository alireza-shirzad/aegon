//! Hash suite for Aegon.
//!
//! Aegon's index assignment relies on two hash functions (paper §5.1):
//!   - `H_bits(ctr, label) -> {0,1}^{log N}` : maps a (probe counter, label) to a
//!     boolean point on the hypercube. Need not be collision-resistant on its
//!     own — uniqueness is enforced by the open-addressing protocol.
//!   - `H_F(label) -> F \ {0}` : the field-element value stored at a label's
//!     assigned index. Must be collision-resistant and preimage-resistant on
//!     `F`.
//!
//! In production, both are instantiated with a VRF (paper §6.1 / §6.5) so that
//! clients cannot enumerate the index distribution offline. Here we provide a
//! deterministic SHA-256 instantiation suitable for the protocol skeleton; the
//! `HashSuite` trait lets callers swap in a VRF later without touching the
//! Aegon core.

use ark_ff::PrimeField;
use sha2::{Digest, Sha256};

/// The hash functions needed by Aegon's two-layer index-assignment
/// protocol.
///
/// **Two-layer routing:** in the sharded deployment, label placement
/// is split into:
///   - `H_shard(shard_ctr, label) -> shard_id` — picks which shard
///     the label lives in. `shard_ctr` advances only when the picked
///     shard reports fullness back to the coordinator. The
///     coordinator keeps an `O(N_shards)` map of which shards have
///     reported full and skips them at routing time.
///   - `H_slot(slot_ctr, label)  -> slot_bits` — within the chosen
///     shard, picks an open-addressing slot. `slot_ctr` advances when
///     the current slot is already occupied by a *different* label
///     (collision); intra-shard probing is driven entirely by the
///     shard, with no coord involvement.
///
/// Both layers must be VRF-backed in privacy deployments so adversaries
/// cannot enumerate placements offline. The `EcVrfHash` impl wires both
/// to the same process-wide VRF key with disjoint domain-separation
/// tags so the two layers cannot be confused or replayed against each
/// other.
///
/// The legacy single-layer `H_bits` is retained for the unsharded /
/// single-shard call sites (e.g., `Aegon::publish` internal probing,
/// `verify::verify_lookup`) that don't need the shard / slot split.
pub trait HashSuite<F: PrimeField> {
    /// Single-layer hash: maps `(ctr, label)` to `num_vars` bits.
    /// Used by the unsharded code paths. The two-layer code paths use
    /// `h_shard` + `h_slot` instead.
    fn h_bits(ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool>;

    /// First-layer hash: maps `(shard_ctr, label)` to `num_vars` bits.
    /// The output is interpreted as a `shard_id` index by the
    /// coordinator. Must be domain-separated from `h_slot` and from
    /// the legacy `h_bits`. `shard_ctr` advances only when the
    /// targeted shard has reported full to the coord.
    fn h_shard(shard_ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool>;

    /// Second-layer hash: maps `(slot_ctr, label)` to `num_vars` bits.
    /// The output is interpreted as `slot_bits` within the chosen
    /// shard. Must be domain-separated from `h_shard` and from the
    /// legacy `h_bits`. `slot_ctr` advances on intra-shard slot
    /// collision (regular open addressing).
    fn h_slot(slot_ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool>;

    /// Maps `label` to a non-zero field element.
    fn h_f(label: &[u8]) -> F;
}

/// SHA-256-based deterministic hash suite. Domain-separated by tag bytes so
/// `H_bits` and `H_F` cannot collide.
pub struct Sha256Hash;

/// Shared implementation for SHA-256-based bit derivation under a
/// domain tag. Stretches SHA-256 output if `num_vars` exceeds 256
/// bits by re-hashing with an internal counter, same convention as
/// the original `h_bits`. Domain separation lives entirely in the
/// `tag` bytes — `tag` must be a unique constant per call site
/// (one of `b"aegon.h_bits"`, `b"aegon.h_shard"`, `b"aegon.h_slot"`).
fn sha256_bits_with_tag(tag: &[u8], ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool> {
    let mut bits = Vec::with_capacity(num_vars);
    let mut counter: u32 = 0;
    while bits.len() < num_vars {
        let mut h = Sha256::new();
        h.update(tag);
        h.update(ctr.to_le_bytes());
        h.update(counter.to_le_bytes());
        h.update((label.len() as u64).to_le_bytes());
        h.update(label);
        let digest = h.finalize();
        for byte in digest.iter() {
            for bit in 0..8 {
                if bits.len() == num_vars {
                    return bits;
                }
                bits.push((byte >> bit) & 1 == 1);
            }
        }
        counter += 1;
    }
    bits
}

impl<F: PrimeField> HashSuite<F> for Sha256Hash {
    fn h_bits(ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool> {
        sha256_bits_with_tag(b"aegon.h_bits", ctr, label, num_vars)
    }

    fn h_shard(shard_ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool> {
        sha256_bits_with_tag(b"aegon.h_shard", shard_ctr, label, num_vars)
    }

    fn h_slot(slot_ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool> {
        sha256_bits_with_tag(b"aegon.h_slot", slot_ctr, label, num_vars)
    }

    fn h_f(label: &[u8]) -> F {
        // Reduce SHA-256 output mod p; rehash on the (negligibly probable) zero
        // case so the post-condition `H_F(label) != 0` holds without leaking
        // anything about `label`.
        let mut counter: u32 = 0;
        loop {
            let mut h = Sha256::new();
            h.update(b"aegon.h_f");
            h.update(counter.to_le_bytes());
            h.update((label.len() as u64).to_le_bytes());
            h.update(label);
            let digest = h.finalize();
            let value = F::from_le_bytes_mod_order(&digest);
            if !value.is_zero() {
                return value;
            }
            counter += 1;
        }
    }
}

/// Convert a boolean index `bits` to the `usize` encoding used by the
/// underlying multilinear PCS's evaluation table.
///
/// Block-major C-order: `bits` is split into `dims.len()` consecutive
/// blocks of widths `dims[0], dims[1], ...`; each block is encoded
/// little-endian within itself, and earlier blocks land in the higher
/// bits of the final index. This matches KZH-k's internal layout
/// (Figure 14, paper) — earlier-numbered variables (`X_1` etc.) occupy
/// the high bits because the H tensor is stored in C-order with the
/// first axis varying slowest.
///
/// For PCSs that use the ark_poly default convention (single block,
/// variable `i` in bit `i`), pass `dims = &[bits.len()]`.
///
/// # Panics
///
/// If `dims.iter().sum::<usize>() != bits.len()`.
pub fn bool_index_to_usize(bits: &[bool], dims: &[usize]) -> usize {
    debug_assert_eq!(
        dims.iter().copied().sum::<usize>(),
        bits.len(),
        "dims must partition bits"
    );
    let mut idx: usize = 0;
    let mut start = 0;
    for &d in dims {
        let end = start + d;
        let block = bits_le_to_usize(&bits[start..end]);
        idx = (idx << d) | block;
        start = end;
    }
    idx
}

fn bits_le_to_usize(bits: &[bool]) -> usize {
    let mut acc: usize = 0;
    for (i, &b) in bits.iter().enumerate() {
        if b {
            acc |= 1 << i;
        }
    }
    acc
}

/// Convert a boolean index to a `Vec<F>` evaluation point.
pub fn bool_index_to_point<F: PrimeField>(bits: &[bool]) -> Vec<F> {
    bits.iter()
        .map(|&b| if b { F::one() } else { F::zero() })
        .collect()
}

// ---------------------------------------------------------------------------
// ECVRF-EDWARDS25519-SHA512-TAI (RFC 9381) instantiation of `H_bits`.
// ---------------------------------------------------------------------------
//
// Three pieces, layered so each can be used independently:
//   1. `vrf_alpha(ctr, label)`     — the canonical VRF input ("alpha string")
//      derived from `(ctr, label)`. Domain-separated by `b"aegon.h_bits"`.
//   2. `output_to_bits(out, k)`    — 64-byte VRF output -> first `k` bits
//      (little-endian per byte). Used by both prove and verify paths so
//      they cannot drift.
//   3. `VrfProver` / `VrfVerifier` — keyed wrappers that hold the
//      private / public key and return (bits, proof_bytes) or
//      (verified bits) respectively. These are the types call-sites in
//      sharded.rs / verify.rs / consistency.rs should use once the
//      gRPC wire format carries proofs alongside lookup responses.
//
// `EcVrfHash` is the static-method `HashSuite` shim used by the
// benchmark binaries and by sharded.rs callers that still go through
// the legacy static path. It computes the same bits as `VrfProver`
// (and is interchangeable on the server side), but it cannot surface
// proofs because the trait signature is static. Replacing call sites
// with `VrfProver::prove_h_bits` is what surfaces proofs to the wire.

/// Length of an ECVRF proof on the wire (RFC 9381, P-256/Ed25519 = 80 B).
pub const VRF_PROOF_BYTES: usize = akd_core::ecvrf::PROOF_LENGTH;
/// Length of an Ed25519 VRF public key on the wire.
pub const VRF_PUBLIC_KEY_BYTES: usize = 32;

/// Build the canonical alpha string fed to `VRF.prove` / `VRF.verify`
/// for `(domain_tag, ctr, label)`. Format: `tag || ctr_le8 ||
/// label_len_le8 || label`. The tag is the layer's domain-separation
/// string — one of `b"aegon.h_bits"`, `b"aegon.h_shard"`,
/// `b"aegon.h_slot"` — and must match between prover and verifier.
/// Stable across `EcVrfHash`, `VrfProver`, and `VrfVerifier`; changing
/// the encoding here is a wire-format break, so it is documented and
/// unit-tested.
fn vrf_alpha_with_tag(tag: &[u8], ctr: u64, label: &[u8]) -> Vec<u8> {
    let mut alpha = Vec::with_capacity(tag.len() + 8 + 8 + label.len());
    alpha.extend_from_slice(tag);
    alpha.extend_from_slice(&ctr.to_le_bytes());
    alpha.extend_from_slice(&(label.len() as u64).to_le_bytes());
    alpha.extend_from_slice(label);
    alpha
}

/// Legacy single-layer alpha (`b"aegon.h_bits"` tag). Retained so the
/// unsharded code paths and existing unit tests stay byte-stable.
/// New code should use `vrf_alpha_with_tag` with a layer-specific tag.
fn vrf_alpha(ctr: u64, label: &[u8]) -> Vec<u8> {
    vrf_alpha_with_tag(b"aegon.h_bits", ctr, label)
}

/// Slice the 64-byte VRF output into `num_vars` Boolean bits using
/// the same little-endian per-byte ordering as the SHA-256 suite.
/// Panics if `num_vars` exceeds the output's bit width (currently 512).
fn output_to_bits(out_bytes: &[u8], num_vars: usize) -> Vec<bool> {
    assert!(
        num_vars <= 8 * out_bytes.len(),
        "ECVRF output is {} bytes; cannot serve num_vars={num_vars}",
        out_bytes.len()
    );
    let mut bits = Vec::with_capacity(num_vars);
    'outer: for byte in out_bytes.iter() {
        for bit in 0..8 {
            if bits.len() == num_vars {
                break 'outer;
            }
            bits.push((byte >> bit) & 1 == 1);
        }
    }
    bits
}

/// Configuration controlling where `EcVrfHash`'s server-side VRF
/// private key comes from. The lookup is performed once per process
/// (on first `vrf_secret_key()` call) and cached in a `OnceLock`.
///
/// Priority order:
///   1. `AEGON_VRF_SEED` env var — 64 hex chars of a 32-byte Ed25519 seed.
///   2. `AEGON_VRF_KEY_PATH` env var — path to a 32-byte sealed seed
///      file; if the path does not exist, a fresh seed is generated
///      via `OsRng` and persisted there with mode `0600`.
///   3. The hard-coded benchmark seed `BENCH_VRF_SEED` — used only by
///      tests and microbenches, never by a production binary.
///
/// Production deployments should set either `AEGON_VRF_SEED` (when an
/// out-of-band key-management system, HSM, or KMS feeds the seed at
/// process start) or `AEGON_VRF_KEY_PATH` (for self-managed sealed
/// storage on the coordinator host). The pubkey is derived
/// deterministically from the secret and exposed via
/// `EcVrfHash::public_key()` so it can be packaged with the SRS or
/// served via a `/vrf-pubkey` endpoint to clients.
pub mod vrf_key_source {
    /// Env var name for the 32-byte hex-encoded seed.
    pub const SEED_HEX_ENV: &str = "AEGON_VRF_SEED";
    /// Env var name for the on-disk sealed-seed path.
    pub const KEY_PATH_ENV: &str = "AEGON_VRF_KEY_PATH";
}

/// Hard-coded VRF seed used by benches and unit tests when neither
/// env var is set. Never use in production — it's published in this
/// source file.
pub const BENCH_VRF_SEED: [u8; 32] = *b"aegon-bench-ecvrf-edwards25519!\0";

fn load_seed_from_env() -> Option<[u8; 32]> {
    use std::env;
    if let Ok(hex) = env::var(vrf_key_source::SEED_HEX_ENV) {
        let mut buf = [0u8; 32];
        if hex::decode_to_slice(hex.trim(), &mut buf).is_ok() {
            return Some(buf);
        } else {
            eprintln!(
                "warning: {} set but not a valid 64-char hex string; \
                 falling back",
                vrf_key_source::SEED_HEX_ENV
            );
        }
    }
    if let Ok(path) = env::var(vrf_key_source::KEY_PATH_ENV) {
        return Some(load_or_create_seed_file(&path));
    }
    None
}

fn load_or_create_seed_file(path: &str) -> [u8; 32] {
    use std::fs;
    use std::io::Read;
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::Path;

    let p = Path::new(path);
    if p.exists() {
        let mut f = fs::File::open(p).expect("VRF key path readable");
        let mut buf = [0u8; 32];
        f.read_exact(&mut buf)
            .expect("VRF key file must be exactly 32 bytes");
        return buf;
    }
    // Fresh seed: 32 bytes from the OS entropy source, persisted at
    // mode 0600. Read from /dev/urandom directly to avoid pulling in
    // a new feature-gated dependency on `rand`.
    let mut buf = [0u8; 32];
    let mut urandom = fs::File::open("/dev/urandom")
        .expect("/dev/urandom available for VRF key generation");
    urandom
        .read_exact(&mut buf)
        .expect("read 32 bytes from /dev/urandom");
    if let Some(parent) = p.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true).mode(0o600);
    let mut f = opts
        .open(p)
        .expect("could not create new VRF key file at AEGON_VRF_KEY_PATH");
    use std::io::Write;
    f.write_all(&buf).expect("write VRF seed");
    buf
}

/// Lazily-initialized per-process VRF private key for `EcVrfHash`.
/// Initialized on first call from the configured key source.
fn vrf_secret_key() -> &'static akd_core::ecvrf::VRFPrivateKey {
    use akd_core::ecvrf::VRFPrivateKey;
    use std::sync::OnceLock;
    static KEY: OnceLock<VRFPrivateKey> = OnceLock::new();
    KEY.get_or_init(|| {
        let seed = load_seed_from_env().unwrap_or(BENCH_VRF_SEED);
        VRFPrivateKey::try_from(seed.as_slice())
            .expect("32-byte seed is a valid Ed25519 secret")
    })
}

// ---------------------------------------------------------------------------
// Keyed prover / verifier API. These are the types call-sites in
// sharded.rs / verify.rs / consistency.rs should adopt once the gRPC
// wire format carries proofs alongside lookup responses.
// ---------------------------------------------------------------------------

/// Server-side VRF prover. Holds an Ed25519 private key, exposes the
/// matching public key, and produces `(bits, proof)` for every
/// `(ctr, label)` it's asked about.
///
/// A `VrfProver` is created once per coordinator/shard process,
/// typically by:
///   - `VrfProver::from_env()`     — read from env / file per
///     `vrf_key_source` priority order, generate-and-persist a fresh
///     key when neither is set;
///   - `VrfProver::from_seed(&[u8; 32])` — explicit injection for
///     deterministic testing.
#[derive(Clone)]
pub struct VrfProver {
    sk: akd_core::ecvrf::VRFPrivateKey,
    pk: akd_core::ecvrf::VRFPublicKey,
}

impl VrfProver {
    /// Build a prover from a 32-byte Ed25519 seed.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let sk = akd_core::ecvrf::VRFPrivateKey::try_from(seed.as_slice())
            .expect("32-byte seed is a valid Ed25519 secret");
        let pk = akd_core::ecvrf::VRFPublicKey::from(&sk);
        Self { sk, pk }
    }

    /// Build a prover using the same priority order as
    /// `EcVrfHash::public_key()` — env var, then key-file, then
    /// `BENCH_VRF_SEED`.
    pub fn from_env() -> Self {
        let seed = load_seed_from_env().unwrap_or(BENCH_VRF_SEED);
        Self::from_seed(&seed)
    }

    /// Public key paired with this prover. Ship to clients via the
    /// existing config bundle so they can construct a `VrfVerifier`.
    pub fn public_key(&self) -> &akd_core::ecvrf::VRFPublicKey {
        &self.pk
    }

    /// Compute `H_bits(ctr, label)` together with the VRF proof.
    /// The proof bytes (`VRF_PROOF_BYTES = 80`) attach to the lookup
    /// response; the verifier then recovers the same bits via
    /// `VrfVerifier::verify_h_bits`. Used by the legacy single-layer
    /// code paths; new code should use `prove_h_shard` / `prove_h_slot`.
    pub fn prove_h_bits(
        &self,
        ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> (Vec<bool>, [u8; VRF_PROOF_BYTES]) {
        self.prove_with_tag(b"aegon.h_bits", ctr, label, num_vars)
    }

    /// Compute `H_shard(shard_ctr, label)` together with the VRF
    /// proof. First layer of the two-layer routing — selects which
    /// shard. The coordinator only needs `shard_ctr > 0` when the
    /// targeted shard has reported full; in steady state `shard_ctr =
    /// 0` is sufficient.
    ///
    /// **Short-circuit for `num_vars == 0` (single-shard deployments)**:
    /// the layer-1 routing decision is structurally fixed (shard 0 is
    /// the only shard), the recovered bit vector is necessarily empty,
    /// and the VRF proof carries no information any verifier can act
    /// on. We skip the ECVRF prove call entirely and return a
    /// zero-filled proof. The matching [`VrfVerifier::verify_h_shard`]
    /// also skips verification in this case. Soundness is unaffected:
    /// the only valid shard_id in any path is 0, regardless of what
    /// the proof bytes are.
    pub fn prove_h_shard(
        &self,
        shard_ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> (Vec<bool>, [u8; VRF_PROOF_BYTES]) {
        if num_vars == 0 {
            return (Vec::new(), [0u8; VRF_PROOF_BYTES]);
        }
        self.prove_with_tag(b"aegon.h_shard", shard_ctr, label, num_vars)
    }

    /// Compute `H_slot(slot_ctr, label)` together with the VRF
    /// proof. Second layer of the two-layer routing — picks an
    /// open-addressing slot within the shard chosen by `prove_h_shard`.
    /// `slot_ctr` advances on intra-shard collision.
    pub fn prove_h_slot(
        &self,
        slot_ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> (Vec<bool>, [u8; VRF_PROOF_BYTES]) {
        self.prove_with_tag(b"aegon.h_slot", slot_ctr, label, num_vars)
    }

    /// Shared core: prove the VRF for `(tag, ctr, label)` and return
    /// `(bits, proof)`. Tag is the layer's domain-separation string.
    fn prove_with_tag(
        &self,
        tag: &[u8],
        ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> (Vec<bool>, [u8; VRF_PROOF_BYTES]) {
        use akd_core::ecvrf::Output;
        let alpha = vrf_alpha_with_tag(tag, ctr, label);
        let proof = self.sk.prove(&alpha);
        let output = Output::from(&proof);
        let bits = output_to_bits(&output.to_bytes(), num_vars);
        (bits, proof.to_bytes())
    }

    /// Bits-only variant of `prove_h_shard`. The VRF output is bit-
    /// identical to what `prove_h_shard` would derive from its proof,
    /// but skips the ZK proof generation entirely — `evaluate` costs
    /// one scalar multiplication versus `prove`'s three plus a Fiat-
    /// Shamir hash. Use this on the publish path where the proof is
    /// never serialized to the wire.
    pub fn evaluate_h_shard(
        &self,
        shard_ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> Vec<bool> {
        if num_vars == 0 {
            return Vec::new();
        }
        self.evaluate_with_tag(b"aegon.h_shard", shard_ctr, label, num_vars)
    }

    /// Bits-only variant of `prove_h_slot`. See `evaluate_h_shard` for
    /// the rationale — used by the shard's open-addressing probe loop
    /// where the slot bits are recomputable by anyone re-running the
    /// VRF, so the proof is never persisted.
    pub fn evaluate_h_slot(
        &self,
        slot_ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> Vec<bool> {
        self.evaluate_with_tag(b"aegon.h_slot", slot_ctr, label, num_vars)
    }

    /// Bits-only variant of `prove_h_bits` — used by the legacy single-
    /// layer code paths when the proof would be discarded.
    pub fn evaluate_h_bits(
        &self,
        ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> Vec<bool> {
        self.evaluate_with_tag(b"aegon.h_bits", ctr, label, num_vars)
    }

    fn evaluate_with_tag(
        &self,
        tag: &[u8],
        ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> Vec<bool> {
        let alpha = vrf_alpha_with_tag(tag, ctr, label);
        let output = self.sk.evaluate(&alpha);
        output_to_bits(&output.to_bytes(), num_vars)
    }
}

/// Client-side VRF verifier. Holds the public key and verifies that
/// a `(ctr, label, proof)` triple was generated by the matching
/// prover, returning the corresponding `H_bits` output on success.
///
/// Errors surface as `VrfVerifyError` so the trail-walking code in
/// `verify_lookup_label` etc. can attribute a verification failure
/// to a specific probe index.
#[derive(Clone)]
pub struct VrfVerifier {
    pk: akd_core::ecvrf::VRFPublicKey,
}

/// Error returned by `VrfVerifier::verify_h_bits` when the supplied
/// proof does not validate under the verifier's public key, or when
/// the proof bytes do not parse as a well-formed `Proof`.
#[derive(Debug, Clone)]
pub enum VrfVerifyError {
    /// Proof bytes failed to deserialize (length / point on curve / etc.).
    Malformed(String),
    /// Proof was well-formed but failed `VRF.verify` for this `(pk, alpha)`.
    InvalidProof(String),
}

impl core::fmt::Display for VrfVerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VrfVerifyError::Malformed(m) => write!(f, "VRF proof malformed: {m}"),
            VrfVerifyError::InvalidProof(m) => write!(f, "VRF proof invalid: {m}"),
        }
    }
}

impl std::error::Error for VrfVerifyError {}

impl VrfVerifier {
    /// Build a verifier from a 32-byte Ed25519 public key (the
    /// compressed point form returned by `VrfProver::public_key()`).
    pub fn from_public_key_bytes(pk_bytes: &[u8]) -> Result<Self, VrfVerifyError> {
        let pk = akd_core::ecvrf::VRFPublicKey::try_from(pk_bytes)
            .map_err(|e| VrfVerifyError::Malformed(format!("{e:?}")))?;
        Ok(Self { pk })
    }

    /// Build a verifier directly from a parsed `VRFPublicKey`.
    pub fn new(pk: akd_core::ecvrf::VRFPublicKey) -> Self {
        Self { pk }
    }

    /// Verify `proof` against `(ctr, label)` under the embedded public
    /// key and recover the first `num_vars` bits of the VRF output.
    /// On verification failure the returned `Err` carries the reason
    /// (malformed bytes vs invalid proof). Legacy single-layer entry;
    /// new code should use `verify_h_shard` / `verify_h_slot`.
    pub fn verify_h_bits(
        &self,
        ctr: u64,
        label: &[u8],
        proof_bytes: &[u8],
        num_vars: usize,
    ) -> Result<Vec<bool>, VrfVerifyError> {
        self.verify_with_tag(b"aegon.h_bits", ctr, label, proof_bytes, num_vars)
    }

    /// Verify a `prove_h_shard` proof and recover the shard-routing
    /// bits. The verifier consumes the same `shard_ctr` the server
    /// used to land on a non-full shard — the shard_ctr is part of the
    /// public lookup proof.
    ///
    /// **Short-circuit for `num_vars == 0` (single-shard deployments)**:
    /// returns `Ok(vec![])` without inspecting `proof_bytes`. See
    /// [`VrfProver::prove_h_shard`] for the soundness argument — at
    /// `log_n_shards = 0` there is no routing decision to attest to,
    /// so there is nothing to verify.
    pub fn verify_h_shard(
        &self,
        shard_ctr: u64,
        label: &[u8],
        proof_bytes: &[u8],
        num_vars: usize,
    ) -> Result<Vec<bool>, VrfVerifyError> {
        if num_vars == 0 {
            return Ok(Vec::new());
        }
        self.verify_with_tag(b"aegon.h_shard", shard_ctr, label, proof_bytes, num_vars)
    }

    /// Verify a `prove_h_slot` proof and recover the slot bits within
    /// the chosen shard. The verifier walks one of these per
    /// intra-shard probe in the lookup proof.
    pub fn verify_h_slot(
        &self,
        slot_ctr: u64,
        label: &[u8],
        proof_bytes: &[u8],
        num_vars: usize,
    ) -> Result<Vec<bool>, VrfVerifyError> {
        self.verify_with_tag(b"aegon.h_slot", slot_ctr, label, proof_bytes, num_vars)
    }

    /// Shared core: verify the VRF for `(tag, ctr, label)` and return
    /// the recovered bits.
    fn verify_with_tag(
        &self,
        tag: &[u8],
        ctr: u64,
        label: &[u8],
        proof_bytes: &[u8],
        num_vars: usize,
    ) -> Result<Vec<bool>, VrfVerifyError> {
        use akd_core::ecvrf::{Output, Proof};
        let proof = Proof::try_from(proof_bytes)
            .map_err(|e| VrfVerifyError::Malformed(format!("{e:?}")))?;
        let alpha = vrf_alpha_with_tag(tag, ctr, label);
        self.pk
            .verify(&proof, &alpha)
            .map_err(|e| VrfVerifyError::InvalidProof(format!("{e:?}")))?;
        let output = Output::from(&proof);
        Ok(output_to_bits(&output.to_bytes(), num_vars))
    }
}

/// ECVRF-EDWARDS25519-SHA512-TAI (RFC 9381) instantiation of
/// `HashSuite` for the static call sites (benchmarks, the existing
/// in-process flow). Computes the same bits as a `VrfProver` would,
/// but drops the proof on the floor because the `HashSuite` trait is
/// stateless. To surface proofs to the wire, replace call sites with
/// `VrfProver::prove_h_bits` and the matching
/// `VrfVerifier::verify_h_bits`.
///
/// `H_F(label) -> F\{0}` keeps the SHA-256 instantiation: the paper
/// only requires `H_F` to be collision-resistant and preimage-
/// resistant (no verifiability), so reusing `Sha256Hash::h_f` keeps
/// the field embedding identical between the two suites.
pub struct EcVrfHash;

impl EcVrfHash {
    /// The public key paired with the server-side private key
    /// `EcVrfHash::h_bits` is currently using. Ship this to clients
    /// alongside the SRS so they can build a `VrfVerifier`.
    pub fn public_key() -> akd_core::ecvrf::VRFPublicKey {
        akd_core::ecvrf::VRFPublicKey::from(vrf_secret_key())
    }

    /// Convenience: produce both the bits and the VRF proof for
    /// `(ctr, label)`, using the per-process key. Equivalent to
    /// `VrfProver::from_env().prove_h_bits(...)` but reuses the
    /// process-wide `OnceLock` key.
    pub fn prove_h_bits(
        ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> (Vec<bool>, [u8; VRF_PROOF_BYTES]) {
        Self::prove_with_tag(b"aegon.h_bits", ctr, label, num_vars)
    }

    /// Two-layer entry: prove + bits for the shard-routing hash.
    ///
    /// **Short-circuit for `num_vars == 0`**: mirrors the instance
    /// method [`VrfProver::prove_h_shard`] — skip the ECVRF prove and
    /// return a zero-filled proof. Single-shard deployments have no
    /// routing decision to attest to.
    pub fn prove_h_shard(
        shard_ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> (Vec<bool>, [u8; VRF_PROOF_BYTES]) {
        if num_vars == 0 {
            return (Vec::new(), [0u8; VRF_PROOF_BYTES]);
        }
        Self::prove_with_tag(b"aegon.h_shard", shard_ctr, label, num_vars)
    }

    /// Two-layer entry: prove + bits for the intra-shard slot hash.
    pub fn prove_h_slot(
        slot_ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> (Vec<bool>, [u8; VRF_PROOF_BYTES]) {
        Self::prove_with_tag(b"aegon.h_slot", slot_ctr, label, num_vars)
    }

    fn prove_with_tag(
        tag: &[u8],
        ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> (Vec<bool>, [u8; VRF_PROOF_BYTES]) {
        use akd_core::ecvrf::Output;
        let alpha = vrf_alpha_with_tag(tag, ctr, label);
        let proof = vrf_secret_key().prove(&alpha);
        let output = Output::from(&proof);
        let bits = output_to_bits(&output.to_bytes(), num_vars);
        (bits, proof.to_bytes())
    }

    /// Bits-only variant of `prove_with_tag` — `evaluate` runs one
    /// scalar mul vs `prove`'s three plus a Fiat-Shamir hash. The
    /// VRF output is bit-identical (both derive from the same
    /// `gamma = h_point * sk`), so any caller that discards the
    /// proof can use this directly.
    fn evaluate_with_tag(
        tag: &[u8],
        ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> Vec<bool> {
        let alpha = vrf_alpha_with_tag(tag, ctr, label);
        let output = vrf_secret_key().evaluate(&alpha);
        output_to_bits(&output.to_bytes(), num_vars)
    }
}

impl<F: PrimeField> HashSuite<F> for EcVrfHash {
    fn h_bits(ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool> {
        // The trait signature returns bits only — no caller of
        // `h_bits` can use the proof, so skipping it (one scalar mul
        // vs three plus a Fiat-Shamir hash) is pure win. Call sites
        // that need the proof use `VrfProver::prove_h_bits` or
        // `EcVrfHash::prove_h_bits` directly.
        Self::evaluate_with_tag(b"aegon.h_bits", ctr, label, num_vars)
    }

    fn h_shard(shard_ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool> {
        if num_vars == 0 {
            return Vec::new();
        }
        Self::evaluate_with_tag(b"aegon.h_shard", shard_ctr, label, num_vars)
    }

    fn h_slot(slot_ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool> {
        Self::evaluate_with_tag(b"aegon.h_slot", slot_ctr, label, num_vars)
    }

    fn h_f(label: &[u8]) -> F {
        <Sha256Hash as HashSuite<F>>::h_f(label)
    }
}

#[cfg(test)]
mod ecvrf_tests {
    use super::*;
    use ark_bn254::Fr;

    #[test]
    fn alpha_is_stable() {
        // Stability is wire-format-critical: any byte-layout change
        // here invalidates every proof a prior version emitted.
        let a = vrf_alpha(0, b"alice@example.com");
        // 12 ("aegon.h_bits") + 8 (ctr) + 8 (len) + 17 (label) = 45.
        assert_eq!(a.len(), 45);
        assert_eq!(&a[..12], b"aegon.h_bits");
        assert_eq!(&a[12..20], &0u64.to_le_bytes());
        assert_eq!(&a[20..28], &(17u64).to_le_bytes());
        assert_eq!(&a[28..], b"alice@example.com");
    }

    #[test]
    fn output_to_bits_respects_num_vars() {
        let bytes = [0xA5u8; 64]; // 0xA5 = 0b1010_0101
        let bits = output_to_bits(&bytes, 12);
        assert_eq!(bits.len(), 12);
        // Little-endian per byte: bit 0 of 0xA5 = 1, bit 1 = 0, ...
        let expected = [true, false, true, false, false, true, false, true,
                        true, false, true, false]; // first byte then first 4 bits of next
        assert_eq!(bits, expected);
    }

    #[test]
    fn prover_verifier_round_trip() {
        let prover = VrfProver::from_seed(&BENCH_VRF_SEED);
        let pk_bytes = prover.public_key().as_bytes().to_vec();
        let verifier = VrfVerifier::from_public_key_bytes(&pk_bytes).unwrap();

        let label = b"bob@example.com";
        for ctr in 0u64..5 {
            let (bits_prover, proof) = prover.prove_h_bits(ctr, label, 27);
            let bits_verifier = verifier
                .verify_h_bits(ctr, label, &proof, 27)
                .expect("verifier accepts honestly-produced proof");
            assert_eq!(
                bits_prover, bits_verifier,
                "verifier recovers the same bits as prover (ctr={ctr})"
            );
        }
    }

    #[test]
    fn verifier_rejects_tampered_proof() {
        let prover = VrfProver::from_seed(&BENCH_VRF_SEED);
        let pk_bytes = prover.public_key().as_bytes().to_vec();
        let verifier = VrfVerifier::from_public_key_bytes(&pk_bytes).unwrap();

        let label = b"carol@example.com";
        let (_bits, mut proof) = prover.prove_h_bits(7, label, 27);
        // Flip a bit in the proof; well-formed Edwards points survive
        // flipping low bits of `s`, so the verify step will reject
        // with InvalidProof rather than Malformed.
        proof[60] ^= 0x01;
        let res = verifier.verify_h_bits(7, label, &proof, 27);
        assert!(matches!(res, Err(VrfVerifyError::InvalidProof(_))));
    }

    #[test]
    fn verifier_rejects_wrong_alpha() {
        let prover = VrfProver::from_seed(&BENCH_VRF_SEED);
        let pk_bytes = prover.public_key().as_bytes().to_vec();
        let verifier = VrfVerifier::from_public_key_bytes(&pk_bytes).unwrap();

        let (_bits, proof) = prover.prove_h_bits(0, b"dave", 27);
        // Same proof, different label — must reject.
        let res = verifier.verify_h_bits(0, b"eve", &proof, 27);
        assert!(matches!(res, Err(VrfVerifyError::InvalidProof(_))));
    }

    #[test]
    fn ecvrf_hash_matches_prover() {
        // EcVrfHash::h_bits and VrfProver::prove_h_bits with the same
        // seed must produce identical bits — this is the invariant
        // that lets us refactor static call sites to surface proofs
        // without changing the bits they see.
        let prover = VrfProver::from_seed(&BENCH_VRF_SEED);
        for ctr in 0u64..3 {
            let bits_static = <EcVrfHash as HashSuite<Fr>>::h_bits(ctr, b"frank", 27);
            let (bits_keyed, _proof) = prover.prove_h_bits(ctr, b"frank", 27);
            assert_eq!(bits_static, bits_keyed);
        }
    }

    #[test]
    fn layers_are_domain_separated() {
        // The three layers (h_bits, h_shard, h_slot) must produce
        // INDEPENDENT outputs for the same (ctr, label) — that's the
        // only thing preventing a server from replaying an h_shard
        // proof as an h_slot proof (or vice versa) and confusing the
        // verifier about which layer it just consumed. Backed by
        // distinct ASCII tags in the alpha string.
        let prover = VrfProver::from_seed(&BENCH_VRF_SEED);
        let label = b"replay-attacker@example.com";
        for ctr in 0u64..3 {
            let (bits_bits,  proof_bits)  = prover.prove_h_bits(ctr,  label, 27);
            let (bits_shard, proof_shard) = prover.prove_h_shard(ctr, label, 27);
            let (bits_slot,  proof_slot)  = prover.prove_h_slot(ctr,  label, 27);
            assert_ne!(bits_bits,  bits_shard, "h_bits vs h_shard collide at ctr={ctr}");
            assert_ne!(bits_bits,  bits_slot,  "h_bits vs h_slot collide at ctr={ctr}");
            assert_ne!(bits_shard, bits_slot,  "h_shard vs h_slot collide at ctr={ctr}");
            assert_ne!(proof_bits,  proof_shard);
            assert_ne!(proof_bits,  proof_slot);
            assert_ne!(proof_shard, proof_slot);
        }
    }

    #[test]
    fn shard_layer_round_trip() {
        let prover = VrfProver::from_seed(&BENCH_VRF_SEED);
        let pk_bytes = prover.public_key().as_bytes().to_vec();
        let verifier = VrfVerifier::from_public_key_bytes(&pk_bytes).unwrap();
        let label = b"shard-layer-rt";
        for ctr in 0u64..4 {
            let (bits, proof) = prover.prove_h_shard(ctr, label, 7);
            let recovered = verifier.verify_h_shard(ctr, label, &proof, 7).unwrap();
            assert_eq!(bits, recovered);
            // An h_bits-encoded proof must not verify under h_shard.
            let (_, wrong_layer) = prover.prove_h_bits(ctr, label, 7);
            assert!(verifier.verify_h_shard(ctr, label, &wrong_layer, 7).is_err());
        }
    }

    #[test]
    fn slot_layer_round_trip() {
        let prover = VrfProver::from_seed(&BENCH_VRF_SEED);
        let pk_bytes = prover.public_key().as_bytes().to_vec();
        let verifier = VrfVerifier::from_public_key_bytes(&pk_bytes).unwrap();
        let label = b"slot-layer-rt";
        for ctr in 0u64..4 {
            let (bits, proof) = prover.prove_h_slot(ctr, label, 22);
            let recovered = verifier.verify_h_slot(ctr, label, &proof, 22).unwrap();
            assert_eq!(bits, recovered);
            // Cross-layer rejection: an h_shard proof must not verify
            // under h_slot for the same (ctr, label).
            let (_, wrong_layer) = prover.prove_h_shard(ctr, label, 22);
            assert!(verifier.verify_h_slot(ctr, label, &wrong_layer, 22).is_err());
        }
    }

    #[test]
    fn sha256_layers_domain_separated() {
        // Same property must hold for the Sha256 suite — the bench
        // benchmarks use it, and we want byte-stable outputs that
        // can't be confused across layers.
        let bits_b = <Sha256Hash as HashSuite<Fr>>::h_bits(0, b"x", 27);
        let bits_a = <Sha256Hash as HashSuite<Fr>>::h_shard(0, b"x", 27);
        let bits_s = <Sha256Hash as HashSuite<Fr>>::h_slot(0, b"x", 27);
        assert_ne!(bits_b, bits_a);
        assert_ne!(bits_b, bits_s);
        assert_ne!(bits_a, bits_s);
    }
}
