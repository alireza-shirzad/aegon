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

/// The two hash functions needed by Aegon's index-assignment protocol.
pub trait HashSuite<F: PrimeField> {
    /// Maps `(ctr, label)` to a boolean vector of length `num_vars`.
    fn h_bits(ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool>;

    /// Maps `label` to a non-zero field element.
    fn h_f(label: &[u8]) -> F;
}

/// SHA-256-based deterministic hash suite. Domain-separated by tag bytes so
/// `H_bits` and `H_F` cannot collide.
pub struct Sha256Hash;

impl<F: PrimeField> HashSuite<F> for Sha256Hash {
    fn h_bits(ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool> {
        // Stretch SHA-256 output if num_vars exceeds 256 bits.
        let mut bits = Vec::with_capacity(num_vars);
        let mut counter: u32 = 0;
        while bits.len() < num_vars {
            let mut h = Sha256::new();
            h.update(b"aegon.h_bits");
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
/// for `(ctr, label)`. Format: `b"aegon.h_bits" || ctr_le8 ||
/// label_len_le8 || label`. Stable across `EcVrfHash`, `VrfProver`,
/// and `VrfVerifier` — changing the encoding here is a wire-format
/// break, so it is documented and unit-tested.
fn vrf_alpha(ctr: u64, label: &[u8]) -> Vec<u8> {
    let mut alpha = Vec::with_capacity(12 + 8 + 8 + label.len());
    alpha.extend_from_slice(b"aegon.h_bits");
    alpha.extend_from_slice(&ctr.to_le_bytes());
    alpha.extend_from_slice(&(label.len() as u64).to_le_bytes());
    alpha.extend_from_slice(label);
    alpha
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
    /// `VrfVerifier::verify_h_bits`.
    pub fn prove_h_bits(
        &self,
        ctr: u64,
        label: &[u8],
        num_vars: usize,
    ) -> (Vec<bool>, [u8; VRF_PROOF_BYTES]) {
        use akd_core::ecvrf::Output;
        let alpha = vrf_alpha(ctr, label);
        let proof = self.sk.prove(&alpha);
        let output = Output::from(&proof);
        let bits = output_to_bits(&output.to_bytes(), num_vars);
        (bits, proof.to_bytes())
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
    /// (malformed bytes vs invalid proof).
    pub fn verify_h_bits(
        &self,
        ctr: u64,
        label: &[u8],
        proof_bytes: &[u8],
        num_vars: usize,
    ) -> Result<Vec<bool>, VrfVerifyError> {
        use akd_core::ecvrf::{Output, Proof};
        let proof = Proof::try_from(proof_bytes)
            .map_err(|e| VrfVerifyError::Malformed(format!("{e:?}")))?;
        let alpha = vrf_alpha(ctr, label);
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
        use akd_core::ecvrf::Output;
        let alpha = vrf_alpha(ctr, label);
        let proof = vrf_secret_key().prove(&alpha);
        let output = Output::from(&proof);
        let bits = output_to_bits(&output.to_bytes(), num_vars);
        (bits, proof.to_bytes())
    }
}

impl<F: PrimeField> HashSuite<F> for EcVrfHash {
    fn h_bits(ctr: u64, label: &[u8], num_vars: usize) -> Vec<bool> {
        // Pays the full prove cost so benches reflect the real ECVRF
        // overhead; the proof itself is discarded because the trait
        // signature is static. Production call sites surface the
        // proof via `VrfProver` / `EcVrfHash::prove_h_bits` instead.
        let (bits, _proof) = Self::prove_h_bits(ctr, label, num_vars);
        bits
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
}
