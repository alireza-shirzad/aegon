// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Distributed SRS generation for KZH-k.
//!
//! Replaces the "every shard re-derives the full SRS from
//! `--setup-seed`" pattern with a coordinated scheme:
//!
//!   1. One bootstrap actor (typically a `aegon_srs_bootstrap` binary
//!      run from the coordinator) samples the trapdoors `(g, h, v,
//!      mu_mat)` deterministically from `--setup-seed`.
//!   2. The bootstrap actor pushes the trapdoors to every shard via
//!      [`SrsService::BootstrapSrs`].
//!   3. Each shard computes its flat-index slab of every `H_t` tensor.
//!      Slabs partition `[0, len(H_t))` evenly across `n_shards`.
//!   4. Each shard pulls the missing slabs from its peers via the
//!      streaming [`SrsService::GetSrsSlab`] RPC.
//!   5. Each shard assembles the full SRS, writes it to a local cache
//!      file keyed on `(log_capacity, k, hash(setup_seed))`, and
//!      proceeds to prefill.
//!
//! On subsequent boots, the cache hit short-circuits the bootstrap
//! handshake — the shard loads its SRS from disk and reports ready
//! immediately.
//!
//! ## Sharing math vs. work
//!
//! Compared to the seed-replication path:
//!
//!   * **Compute**: parallelism scales with `N`. Each shard does
//!     roughly `1/N` of the `H_t` MSM work instead of every shard
//!     doing the full pass.
//!   * **Storage**: unchanged — every shard ends up with the full SRS
//!     resident, because every shard commits its own polynomial of
//!     size `2^log_capacity`. The distributed scheme only reduces
//!     compute, not memory.
//!   * **Network**: every shard pulls `(N-1)/N` of the SRS from its
//!     peers. Per-shard wall-clock is dominated by the receiver's NIC
//!     bandwidth.
//!
//! The module is KZH-k-specific (trapdoors, `H_t` slab layout). Other
//! PCS schemes fall back to the existing in-process gen.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ark_ec::{pairing::Pairing, scalar_mul::BatchMulPreprocessing, AffineRepr, CurveGroup};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{rand::Rng, One, UniformRand};
use futures::Stream;
use ndarray::{ArrayD, IxDyn};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tokio::sync::{Mutex, Notify};
use tonic::transport::Channel;
use tonic::{Request, Response, Status};

use akd_core::aegon_crypto::pcs::kzhk::srs::{
    KZHKProverParam, KZHKUniversalParams, KZHKVerifierParam,
};
use akd_core::aegon_crypto::pcs::kzhk::structs::Tensor;

use super::error::AegonError;

use proto::srs_service_client::SrsServiceClient;
use proto::srs_service_server::{SrsService, SrsServiceServer};
use proto::{
    BootstrapSrsRequest, BootstrapSrsResponse, GetMetricsRequest, GetMetricsResponse,
    GetSrsSlabRequest, PhaseEntry, SrsSlabChunk, WaitForReadyRequest, WaitForReadyResponse,
};

// Generated tonic code for `aegon.srs.v1`.
// Generated code carries no rustdoc; the crate-level `warn(missing_docs)`
// cannot be satisfied for types we do not author.
#[allow(missing_docs)]
pub mod proto {
    tonic::include_proto!("aegon.srs.v1");
}

// Match shard/coordinator/masking transports — 1 GiB cap on both sides.
pub(crate) const MAX_MSG_BYTES: usize = 8 * 1024 * 1024 * 1024;

// Target chunk size on the wire for slab streaming. 64 MiB at ~64 B per
// G1Affine works out to ~1M points per chunk on BN254. Lets the
// receiver overlap dequeue/decode with the next chunk's transmit
// without ever holding more than one chunk in flight per stream.
pub(crate) const SLAB_CHUNK_BYTES_TARGET: usize = 64 * 1024 * 1024;

// ---------- trapdoors ---------------------------------------------------

/// Trapdoors sampled once by the bootstrap actor and broadcast to every
/// shard. Small payload (KB-scale): `mu_mat` is `sum_j 2^{d_j}` field
/// elements, and `(g, h, v)` are three group elements.
///
/// Serialisation is canonical-uncompressed so encode/decode is
/// allocation-light on the shard side.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Trapdoors<E: Pairing> {
    /// `[d_1, ..., d_k]` — same as `KZHKUniversalParams::dimensions`.
    pub dimensions: Vec<usize>,
    /// G1 generator.
    pub g: E::G1Affine,
    /// The hiding generator, used for blinded commitments and by the
    /// Sigma protocol.
    pub h: E::G1Affine,
    /// G2 generator.
    pub v: E::G2Affine,
    /// `mu_mat[j][i]` for `j in [k]`, `i in [2^{d_j}]`.
    pub mu_mat: Vec<Vec<E::ScalarField>>,
}

impl<E: Pairing> Trapdoors<E> {
    /// Sample trapdoors from `rng`. Same scalar layout as
    /// `KZHKUniversalParams::gen_srs_for_testing` so that, given a
    /// matched seed, the resulting SRS is byte-identical to what
    /// in-process gen would produce.
    pub fn sample<R: Rng>(rng: &mut R, k: usize, num_vars: usize) -> Self {
        let d = num_vars / k;
        let remainder_d = num_vars % k;
        let mut dimensions = vec![d; k];
        for dim in dimensions.iter_mut().take(remainder_d) {
            *dim += 1;
        }

        let g = E::G1::rand(rng);
        let h = E::G1::rand(rng);
        let v = E::G2::rand(rng);
        let mu_mat: Vec<Vec<E::ScalarField>> = (0..k)
            .map(|j| {
                (0..(1usize << dimensions[j]))
                    .map(|_| E::ScalarField::rand(rng))
                    .collect()
            })
            .collect();

        Self {
            dimensions,
            g: g.into_affine(),
            h: h.into_affine(),
            v: v.into_affine(),
            mu_mat,
        }
    }

    /// Number of KZH-k blocks, i.e. `dimensions.len()`.
    pub fn k(&self) -> usize {
        self.dimensions.len()
    }

    /// Total variable count, `d_1 + ... + d_k`.
    pub fn num_vars(&self) -> usize {
        self.dimensions.iter().sum()
    }

    /// Canonical-uncompressed serialisation. Used by the bootstrap
    /// actor when filling `BootstrapSrsRequest.trapdoors_uncompressed`.
    pub fn encode(&self) -> Result<Vec<u8>, AegonError> {
        let mut buf = Vec::with_capacity(self.uncompressed_size());
        self.serialize_uncompressed(&mut buf)
            .map_err(|e| AegonError::Config(format!("trapdoors encode: {e}")))?;
        Ok(buf)
    }

    /// Inverse of [`Trapdoors::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, AegonError> {
        Self::deserialize_uncompressed_unchecked(bytes)
            .map_err(|e| AegonError::Config(format!("trapdoors decode: {e}")))
    }
}

// ---------- slab geometry ----------------------------------------------

/// Per-`H_t` shape, total length, and C-order strides. Helps both the
/// slab compute (which needs strides to decompose a flat index into the
/// multi-index that selects `mu_mat` entries) and the assembly step.
#[derive(Clone, Debug)]
pub struct HtGeometry {
    /// Which `H_t` this describes, `t in [0, k)`.
    pub t: usize,
    /// Extent along each axis: `2^{d_j}` for `j >= t`.
    pub shape: Vec<usize>,
    /// C-order strides, for decomposing a flat index into the
    /// multi-index that selects `mu_mat` entries.
    pub strides: Vec<usize>,
    /// Total element count, the product of `shape`.
    pub len: usize,
}

impl HtGeometry {
    /// Geometry for every `H_t`, `t in [0, k)`, from the block dims.
    pub fn all(dimensions: &[usize]) -> Vec<HtGeometry> {
        let k = dimensions.len();
        (0..k)
            .map(|t| {
                let shape: Vec<usize> = dimensions[t..].iter().map(|&dj| 1usize << dj).collect();
                let len: usize = shape.iter().product();
                let m = shape.len();
                let mut strides = vec![1usize; m];
                for a in (0..m).rev().skip(1) {
                    strides[a] = strides[a + 1] * shape[a + 1];
                }
                HtGeometry {
                    t,
                    shape,
                    strides,
                    len,
                }
            })
            .collect()
    }
}

/// Slab boundaries for shard `i` of `n_shards` against a tensor of
/// length `len`. Tiles `[0, len)` exactly (sum of widths = len) and is
/// monotonic in `shard_id`.
///
/// Returns `[start, end)`. May be empty (`start == end`) when
/// `len < n_shards` — in that case only the first `len` shards get
/// non-empty slabs.
pub fn slab_range(shard_id: usize, n_shards: usize, len: usize) -> (usize, usize) {
    debug_assert!(n_shards > 0);
    debug_assert!(shard_id < n_shards);
    let start = (shard_id * len) / n_shards;
    let end = ((shard_id + 1) * len) / n_shards;
    (start, end)
}

// ---------- slab compute -----------------------------------------------

// Same cap as the in-process gen path in `KZHKUniversalParams::gen_srs_for_testing`:
// caps the per-chunk `(exps, batch_mul output)` working set so the
// transient `exps` allocation never scales with the full slab length.
const EXPS_CHUNK_LOG: u32 = 25;

/// Compute this shard's slab of `H_t`. Each entry is
/// `g · ∏_{a=0..m} mu_mat[t+a][((global) / strides[a]) % shape[a]]`,
/// where `global` ranges over `[range_start, range_end)`.
///
/// This is the inner kernel of the distributed scheme; everything else
/// (peer-pull exchange, assembly) is glue.
pub fn compute_h_t_slab<E: Pairing>(
    trapdoors: &Trapdoors<E>,
    geom: &HtGeometry,
    range_start: usize,
    range_end: usize,
) -> Vec<E::G1Affine> {
    assert!(range_start <= range_end && range_end <= geom.len);
    let this_len = range_end - range_start;
    if this_len == 0 {
        return Vec::new();
    }

    let m = geom.shape.len();
    let mu_mat = &trapdoors.mu_mat;
    let t = geom.t;
    let g_proj = trapdoors.g.into_group();

    let exps_chunk_max: usize = 1usize << EXPS_CHUNK_LOG;
    let chunk_len = this_len.min(exps_chunk_max);
    let table_g = BatchMulPreprocessing::new(g_proj, chunk_len);

    let mut out: Vec<E::G1Affine> = Vec::with_capacity(this_len);

    let mut chunk_start = 0usize;
    while chunk_start < this_len {
        let this_chunk = (this_len - chunk_start).min(chunk_len);

        let exps_chunk: Vec<E::ScalarField> = (0..this_chunk)
            .into_par_iter()
            .map(|c| {
                let global = range_start + chunk_start + c;
                let mut prod = E::ScalarField::one();
                for a in 0..m {
                    let idx = (global / geom.strides[a]) % geom.shape[a];
                    prod *= mu_mat[t + a][idx];
                }
                prod
            })
            .collect();

        let aff_chunk: Vec<E::G1Affine> = table_g.batch_mul(&exps_chunk);
        out.extend(aff_chunk);

        chunk_start += this_chunk;
    }

    out
}

/// Compute `v_mat = [v^{mu_mat[j][i]}]_{j, i}` locally. Small and
/// duplicate across all shards by design — every shard needs the full
/// `v_mat` for opening, and the payload is at most `k * 2^{d_max}`
/// `G2Prepared` elements (a few thousand entries for production
/// parameters). Cheaper to recompute than to transfer.
pub fn compute_v_mat<E: Pairing>(trapdoors: &Trapdoors<E>) -> Vec<Vec<E::G2Prepared>> {
    let v = trapdoors.v.into_group();
    let k = trapdoors.k();

    (0..k)
        .into_par_iter()
        .map(|j| {
            let rows = 1usize << trapdoors.dimensions[j];
            let table_v = BatchMulPreprocessing::new(v, rows);
            let aff: Vec<E::G2Affine> = table_v.batch_mul(&trapdoors.mu_mat[j]);
            aff.into_iter()
                .map(<E as Pairing>::G2Prepared::from)
                .collect()
        })
        .collect()
}

// ---------- assembly ----------------------------------------------------

/// Build the full `H_t` tensor from `n_shards` consecutive slabs.
/// Caller is responsible for delivering slabs in shard-id order
/// (`slabs[i]` must be the slab owned by shard `i`).
pub fn assemble_h_tensor<E: Pairing>(
    geom: &HtGeometry,
    slabs: Vec<Vec<E::G1Affine>>,
) -> Result<Tensor<E::G1Affine>, AegonError> {
    let total: usize = slabs.iter().map(|s| s.len()).sum();
    if total != geom.len {
        return Err(AegonError::Config(format!(
            "assemble H_t (t={}): slab total {} != expected {}",
            geom.t, total, geom.len
        )));
    }
    let mut flat: Vec<E::G1Affine> = Vec::with_capacity(geom.len);
    for s in slabs {
        flat.extend(s);
    }
    let arr = ArrayD::from_shape_vec(IxDyn(&geom.shape), flat).map_err(|e| {
        AegonError::Config(format!(
            "assemble H_t (t={}): ndarray shape mismatch: {e}",
            geom.t
        ))
    })?;
    Ok(Tensor(arr))
}

/// Build the universal SRS struct from assembled tensors + trapdoor
/// public scalars + the standard zk hiding sparsity. The result is
/// byte-identical to what in-process gen with the matching seed would
/// produce, except for `hiding_sparsity` which is always set (per the
/// "one SRS, ZK-shaped, h ignored by non-ZK paths" design).
pub fn build_universal_params<E: Pairing>(
    trapdoors: &Trapdoors<E>,
    h_tensors: Vec<Tensor<E::G1Affine>>,
    v_mat: Vec<Vec<E::G2Prepared>>,
) -> KZHKUniversalParams<E> {
    let hiding_sparsity =
        Some(ceil_k_root_scaled(1u128 << trapdoors.num_vars(), trapdoors.k() as u32) as usize);
    KZHKUniversalParams::new(
        trapdoors.dimensions.clone(),
        Arc::new(h_tensors),
        Arc::new(v_mat),
        trapdoors.v,
        trapdoors.g,
        trapdoors.h,
        hiding_sparsity,
    )
}

/// Mirror of `ceil_k_root_scaled` in `pcs/kzhk/srs.rs`. Duplicated here
/// to avoid touching that file's visibility — the function is trivially
/// portable.
fn ceil_k_root_scaled(n: u128, k: u32) -> u128 {
    debug_assert!(k > 0, "k must be >= 1");
    if n == 0 {
        return 0;
    }
    if k == 1 {
        return n;
    }
    // ceil(n^(1/k)) via integer search around the floating-point estimate.
    let approx = (n as f64).powf(1.0 / k as f64);
    let mut lo = (approx as u128).saturating_sub(2).max(1);
    while lo.checked_pow(k).map(|p| p > n).unwrap_or(true) && lo > 1 {
        lo -= 1;
    }
    let mut hi = lo + 1;
    while hi.checked_pow(k).map(|p| p < n).unwrap_or(false) {
        hi += 1;
    }
    // ceil = smallest x with x^k >= n
    let ceil_root = if lo.pow(k) >= n { lo } else { hi };
    (k as u128) * ceil_root
}

// ---------- on-disk cache ----------------------------------------------

/// Cache file name for `(log_capacity, k, setup_seed)`. The seed is
/// embedded as a hex u64 so two different seeds never collide on disk.
pub fn cache_file_name(log_capacity: usize, k: usize, setup_seed: u64) -> String {
    format!(
        "aegon-srs-lc{log_capacity}-k{k}-seed{seed:016x}.cache",
        log_capacity = log_capacity,
        k = k,
        seed = setup_seed
    )
}

/// Resolve the full cache path under `dir`.
pub fn cache_file_path(dir: &Path, log_capacity: usize, k: usize, setup_seed: u64) -> PathBuf {
    dir.join(cache_file_name(log_capacity, k, setup_seed))
}

/// Atomic write: serialise to a sibling `*.tmp`, fsync, rename. Avoids
/// leaving a half-written cache file behind if the shard crashes mid
/// write. Subsequent boots either see no cache or see a complete one.
pub fn write_cache<E: Pairing>(
    path: &Path,
    universal_params: &KZHKUniversalParams<E>,
    prover_param: &KZHKProverParam<E>,
    verifier_param: &KZHKVerifierParam<E>,
) -> Result<(), AegonError>
where
    KZHKUniversalParams<E>: CanonicalSerialize,
    KZHKProverParam<E>: CanonicalSerialize,
    KZHKVerifierParam<E>: CanonicalSerialize,
{
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            AegonError::Config(format!("create srs cache dir '{}': {e}", parent.display()))
        })?;
    }
    let tmp = path.with_extension("cache.tmp");
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| {
            AegonError::Config(format!("create srs cache tmp '{}': {e}", tmp.display()))
        })?;
        universal_params
            .serialize_uncompressed(&mut f)
            .map_err(|e| AegonError::Config(format!("serialise universal_params: {e}")))?;
        prover_param
            .serialize_uncompressed(&mut f)
            .map_err(|e| AegonError::Config(format!("serialise prover_param: {e}")))?;
        verifier_param
            .serialize_uncompressed(&mut f)
            .map_err(|e| AegonError::Config(format!("serialise verifier_param: {e}")))?;
        f.flush()
            .map_err(|e| AegonError::Config(format!("flush srs cache tmp: {e}")))?;
        f.sync_all()
            .map_err(|e| AegonError::Config(format!("fsync srs cache tmp: {e}")))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| {
        AegonError::Config(format!(
            "rename srs cache '{}' -> '{}': {e}",
            tmp.display(),
            path.display()
        ))
    })?;
    Ok(())
}

/// Read a previously-written cache. Returns
/// `(universal_params, prover_param, verifier_param)`.
pub fn read_cache<E: Pairing>(
    path: &Path,
) -> Result<
    (
        KZHKUniversalParams<E>,
        KZHKProverParam<E>,
        KZHKVerifierParam<E>,
    ),
    AegonError,
>
where
    KZHKUniversalParams<E>: CanonicalDeserialize,
    KZHKProverParam<E>: CanonicalDeserialize,
    KZHKVerifierParam<E>: CanonicalDeserialize,
{
    let mut f = std::fs::File::open(path)
        .map_err(|e| AegonError::Config(format!("open srs cache '{}': {e}", path.display())))?;
    let up = KZHKUniversalParams::<E>::deserialize_uncompressed_unchecked(&mut f)
        .map_err(|e| AegonError::Config(format!("deserialise universal_params: {e}")))?;
    let pk = KZHKProverParam::<E>::deserialize_uncompressed_unchecked(&mut f)
        .map_err(|e| AegonError::Config(format!("deserialise prover_param: {e}")))?;
    let vk = KZHKVerifierParam::<E>::deserialize_uncompressed_unchecked(&mut f)
        .map_err(|e| AegonError::Config(format!("deserialise verifier_param: {e}")))?;
    Ok((up, pk, vk))
}

// ---------- slab wire codec --------------------------------------------

/// Encode a slab of `G1Affine` points into chunked
/// `SrsSlabChunk.points_uncompressed` byte buffers. Each chunk holds an
/// integer number of points, sized to ~`SLAB_CHUNK_BYTES_TARGET` on the
/// wire.
pub fn encode_slab_to_chunks<E: Pairing>(
    slab: &[E::G1Affine],
) -> Result<Vec<(Vec<u8>, u32)>, AegonError> {
    if slab.is_empty() {
        return Ok(Vec::new());
    }
    let per_point = slab[0].uncompressed_size();
    let pts_per_chunk = (SLAB_CHUNK_BYTES_TARGET / per_point).max(1);
    let mut out = Vec::with_capacity(slab.len().div_ceil(pts_per_chunk));
    for chunk in slab.chunks(pts_per_chunk) {
        let mut buf = Vec::with_capacity(chunk.len() * per_point);
        for p in chunk {
            p.serialize_uncompressed(&mut buf)
                .map_err(|e| AegonError::Config(format!("encode slab chunk: {e}")))?;
        }
        out.push((buf, chunk.len() as u32));
    }
    Ok(out)
}

/// Decode a single slab chunk (counterpart to one entry produced by
/// [`encode_slab_to_chunks`]).
pub fn decode_slab_chunk<E: Pairing>(
    bytes: &[u8],
    n_points: usize,
) -> Result<Vec<E::G1Affine>, AegonError> {
    let mut out = Vec::with_capacity(n_points);
    let mut cursor: &[u8] = bytes;
    for _ in 0..n_points {
        let p = E::G1Affine::deserialize_uncompressed_unchecked(&mut cursor)
            .map_err(|e| AegonError::Config(format!("decode slab point: {e}")))?;
        out.push(p);
    }
    Ok(out)
}

// ---------- helpers consumed by server/client glue ---------------------

/// Per-shard slab map: `slabs[t][shard_id] = Vec<G1Affine>` once all
/// peers' contributions have landed. Local slab is populated by
/// [`compute_h_t_slab`]; peer slabs are populated by the streaming
/// `GetSrsSlab` consumer.
pub type SlabMatrix<E> = Vec<BTreeMap<usize, Vec<<E as Pairing>::G1Affine>>>;

/// Convenience: convert a fully-populated [`SlabMatrix`] (every
/// `[t][shard_id]` present) into the assembled `Vec<Tensor>` ready for
/// [`build_universal_params`].
pub fn matrix_to_tensors<E: Pairing>(
    geoms: &[HtGeometry],
    matrix: SlabMatrix<E>,
    n_shards: usize,
) -> Result<Vec<Tensor<E::G1Affine>>, AegonError> {
    if matrix.len() != geoms.len() {
        return Err(AegonError::Config(format!(
            "matrix_to_tensors: matrix.len()={} != geoms.len()={}",
            matrix.len(),
            geoms.len()
        )));
    }
    geoms
        .iter()
        .zip(matrix)
        .map(|(geom, mut per_t)| {
            let mut slabs: Vec<Vec<E::G1Affine>> = Vec::with_capacity(n_shards);
            for s in 0..n_shards {
                let slab = per_t.remove(&s).ok_or_else(|| {
                    AegonError::Config(format!(
                        "matrix_to_tensors: missing slab t={} shard={}",
                        geom.t, s
                    ))
                })?;
                slabs.push(slab);
            }
            assemble_h_tensor::<E>(geom, slabs)
        })
        .collect()
}

// ---------- bootstrap state machine ------------------------------------

/// Per-shard config for the distributed-gen path. Built from CLI args
/// on the shard server and passed to [`SrsBootstrapState::new`].
#[derive(Clone, Debug)]
pub struct SrsBootstrapConfig {
    /// This shard's index in the cluster.
    pub shard_id: u32,
    /// Log2 of this shard's slot count.
    pub log_capacity: u32,
    /// Number of KZH-k blocks.
    pub k: u32,
    /// Seed the trapdoors are sampled from. NOT a ceremony -- see
    /// `SECURITY.md`.
    pub setup_seed: u64,
    /// Directory holding the on-disk SRS cache, keyed by
    /// `(log_capacity, k, setup_seed)`.
    pub cache_dir: PathBuf,
}

/// Phase of the bootstrap. Visible to the bootstrap actor via
/// [`SrsService::WaitForReady`] for human-readable progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Shard is up but has no SRS yet. Waiting for a `BootstrapSrs` RPC.
    AwaitingBootstrap,
    /// Trapdoors received; computing local slabs and exchanging with peers.
    Computing,
    /// All slabs in hand; building the assembled `KZHKUniversalParams`
    /// and writing the cache file.
    Assembling,
    /// SRS in hand; initialising the Aegon instance.
    Initializing,
    /// Aegon initialised; running prefill.
    Prefilling,
    /// Ready to serve normal traffic.
    Ready,
}

impl Phase {
    /// Stable lowercase name, for logs and the `WaitForReady` wire.
    pub fn as_str(self) -> &'static str {
        match self {
            Phase::AwaitingBootstrap => "awaiting-bootstrap",
            Phase::Computing => "computing-slabs",
            Phase::Assembling => "assembling-srs",
            Phase::Initializing => "initializing-aegon",
            Phase::Prefilling => "prefilling",
            Phase::Ready => "ready",
        }
    }
}

/// Plain-struct view of the per-shard metrics, mirroring the on-wire
/// `GetMetricsResponse` shape. Returned by
/// [`SrsBootstrapState::snapshot_metrics`] and consumed by the
/// `GetMetrics` handler + tests.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Shard these metrics came from.
    pub shard_id: u32,
    /// Whether the SRS was served from the on-disk cache rather than
    /// recomputed.
    pub cache_hit: bool,
    /// Wall-clock spent in each bootstrap phase.
    pub phases: Vec<PhaseEntry>,
    /// Slab bytes received from peers.
    pub inbound_slab_bytes: u64,
    /// Slab bytes sent to peers.
    pub outbound_slab_bytes: u64,
    /// Size of the prover key.
    pub pk_bytes: u64,
    /// Size of the verifier key.
    pub vk_bytes: u64,
    /// Size of the universal parameters before trimming.
    pub universal_bytes: u64,
    /// Seconds spent computing.
    pub compute_secs: f64,
    /// Seconds spent transferring slabs.
    pub communication_secs: f64,
}

struct Inner<E: Pairing> {
    phase: Phase,
    trapdoors: Option<Arc<Trapdoors<E>>>,
    /// `peer_endpoints[i]` = gRPC URL for shard `i`. Populated when
    /// trapdoors arrive (the bootstrap actor knows the full topology).
    peer_endpoints: Vec<String>,
    /// `local_slabs[t]` = this shard's slab of `H_t`. Populated by the
    /// main loop after [`compute_h_t_slab`] finishes for each `t`.
    /// Wrapped in `Arc` so concurrent `GetSrsSlab` handlers can share
    /// the same allocation without copying.
    local_slabs: Vec<Option<Arc<Vec<E::G1Affine>>>>,
    status: String,
    /// Phase-transition timestamps in chronological order. Each entry
    /// is `(label, monotonic_instant)`; the bootstrap actor's
    /// `GetMetrics` aggregator subtracts consecutive entries to get
    /// durations. Labels are short stable strings (see `set_phase`).
    /// May contain multiple entries with the same phase (e.g. nested
    /// "computing-local-slabs" / "pulling-peer-slabs" sub-phases under
    /// `Computing`).
    phase_log: Vec<(String, Instant)>,
    /// True once an SRS has been assembled (or loaded from cache).
    /// Used to gate cache-hit reporting in `GetMetrics`.
    cache_hit: bool,
    /// Uncompressed serialised sizes of the assembled SRS, recorded at
    /// cache-write time (or cache-load time on a cache hit). Zero
    /// before the shard reaches assembly.
    pk_bytes: u64,
    vk_bytes: u64,
    universal_bytes: u64,
    /// Wall-clock seconds spent in CPU-bound setup (slab compute +
    /// assembly + build + trim). Set by [`record_phase_durations`]
    /// once the distributed path finishes. Stays at 0 on the
    /// cache-hit path.
    compute_secs: f64,
    /// Wall-clock seconds spent in the peer slab exchange phase
    /// (`pull_slab_from_peer`). Stays at 0 on single-shard runs (no
    /// peers to talk to) and on cache-hit boots.
    communication_secs: f64,
}

/// Shared bootstrap state. Holds the trapdoors handoff, the local slab
/// table that peers pull from, and the readiness signal. `Arc`-shared
/// between the shard's main loop and the gRPC handlers.
pub struct SrsBootstrapState<E: Pairing> {
    config: SrsBootstrapConfig,
    inner: Mutex<Inner<E>>,
    /// Broadcast notify on any state mutation. Awaiters re-check the
    /// condition they care about after each wake. Coarse but correct;
    /// the actual number of waiters is small (handful of GetSrsSlab
    /// streams in flight + the main loop's trapdoor wait).
    notify: Notify,
    /// Process-start anchor for the phase log. Subtracting from any
    /// recorded `Instant` gives a `secs since boot` value that's
    /// comparable across shards (modulo clock drift, which is
    /// monotonic-instant-bounded anyway).
    started_at: Instant,
    /// Total bytes of `G1Affine` payload pulled from peers (sum across
    /// all `(t, peer)` pairs). Incremented by the slab pull loop in
    /// [`run_distributed_compute`] after each peer's stream drains.
    /// Atomic for cheap lock-free updates from the parallel pull
    /// tasks.
    inbound_slab_bytes: AtomicU64,
    /// Total bytes of slab payload served to peers. Incremented by
    /// the `GetSrsSlab` handler per request. Should match
    /// `inbound_slab_bytes` on the symmetric side once exchange ends.
    outbound_slab_bytes: AtomicU64,
}

impl<E: Pairing> SrsBootstrapState<E> {
    /// Start a bootstrap actor in the `Idle` phase.
    pub fn new(config: SrsBootstrapConfig) -> Arc<Self> {
        // Allocate slab slots up front, sized by `k`. We don't know the
        // geometry until trapdoors arrive (well, we do — k is in
        // config — but we get the same size from dimensions later).
        let k = config.k as usize;
        let now = Instant::now();
        Arc::new(Self {
            config,
            inner: Mutex::new(Inner {
                phase: Phase::AwaitingBootstrap,
                trapdoors: None,
                peer_endpoints: Vec::new(),
                local_slabs: vec![None; k],
                status: "awaiting-bootstrap".to_string(),
                phase_log: vec![("awaiting-bootstrap".to_string(), now)],
                cache_hit: false,
                pk_bytes: 0,
                vk_bytes: 0,
                universal_bytes: 0,
                compute_secs: 0.0,
                communication_secs: 0.0,
            }),
            notify: Notify::new(),
            started_at: now,
            inbound_slab_bytes: AtomicU64::new(0),
            outbound_slab_bytes: AtomicU64::new(0),
        })
    }

    /// The config this actor was built with.
    pub fn config(&self) -> &SrsBootstrapConfig {
        &self.config
    }

    /// Accept trapdoors pushed by the bootstrap actor. Idempotent: if
    /// the shard is already past `AwaitingBootstrap` (e.g. cache hit
    /// won the race), the pushed trapdoors are silently ignored and
    /// the caller learns via `cache_hit = true` in the response.
    pub async fn accept_bootstrap(
        &self,
        trapdoors: Trapdoors<E>,
        peer_endpoints: Vec<String>,
    ) -> bool {
        let mut inner = self.inner.lock().await;
        match inner.phase {
            Phase::AwaitingBootstrap => {
                inner.trapdoors = Some(Arc::new(trapdoors));
                inner.peer_endpoints = peer_endpoints;
                inner.phase = Phase::Computing;
                inner.status = "computing-slabs".to_string();
                drop(inner);
                self.notify.notify_waiters();
                false // not a cache hit
            }
            _ => true, // already past bootstrap — cache hit (or progressed past)
        }
    }

    /// Block until trapdoors arrive (via `accept_bootstrap`). Called by
    /// the shard's main loop on the cache-miss path.
    pub async fn wait_for_trapdoors(&self) -> (Arc<Trapdoors<E>>, Vec<String>) {
        loop {
            // Snapshot under the lock; await the notify outside the lock.
            let notified = self.notify.notified();
            {
                let inner = self.inner.lock().await;
                if let Some(td) = &inner.trapdoors {
                    return (Arc::clone(td), inner.peer_endpoints.clone());
                }
            }
            notified.await;
        }
    }

    /// Publish this shard's slab of `H_t` so peers can pull it via
    /// [`SrsService::GetSrsSlab`]. Called by the main loop after
    /// [`compute_h_t_slab`] returns for each `t`.
    pub async fn set_local_slab(&self, t: usize, slab: Vec<E::G1Affine>) {
        {
            let mut inner = self.inner.lock().await;
            if t < inner.local_slabs.len() {
                inner.local_slabs[t] = Some(Arc::new(slab));
            }
        }
        self.notify.notify_waiters();
    }

    /// Block until this shard's slab for `H_t` is available, then
    /// return an `Arc`-shared handle. Called by GetSrsSlab handlers.
    /// Returns an empty `Arc` only if `t` is out of range (caller
    /// should validate `t < k` before calling).
    pub async fn wait_for_slab(&self, t: usize) -> Option<Arc<Vec<E::G1Affine>>> {
        loop {
            let notified = self.notify.notified();
            {
                let inner = self.inner.lock().await;
                if t >= inner.local_slabs.len() {
                    return None;
                }
                if let Some(s) = &inner.local_slabs[t] {
                    return Some(Arc::clone(s));
                }
            }
            notified.await;
        }
    }

    /// Drop the per-`H_t` slab cache (after full assembly is done +
    /// peer pulls are complete). Frees ~(N-1)/N of the inbound slab
    /// memory before init/prefill starts. Local slab also dropped at
    /// this point since assembly has consumed it.
    pub async fn drop_local_slabs(&self) {
        let mut inner = self.inner.lock().await;
        for slot in inner.local_slabs.iter_mut() {
            *slot = None;
        }
    }

    /// Set the current phase + status string. Status is best-effort
    /// debug info surfaced via `WaitForReady` and recorded in the phase
    /// log under the same label so `GetMetrics` can replay the
    /// sub-phase breakdown the shard's main loop actually walked
    /// through.
    pub async fn set_phase(&self, phase: Phase, status: impl Into<String>) {
        let status: String = status.into();
        let now = Instant::now();
        {
            let mut inner = self.inner.lock().await;
            inner.phase = phase;
            inner.status = status.clone();
            inner.phase_log.push((status, now));
        }
        self.notify.notify_waiters();
    }

    /// Note that the assembled SRS came from disk cache, not from
    /// distributed gen. Suppresses bogus inbound/outbound numbers in
    /// `GetMetrics` (the counters would be zero anyway but the flag
    /// makes intent explicit).
    pub async fn mark_cache_hit(&self) {
        let mut inner = self.inner.lock().await;
        inner.cache_hit = true;
    }

    /// Record the uncompressed serialised sizes of the assembled SRS,
    /// so `GetMetrics` can return them without a second serialise pass.
    /// Called immediately after `write_cache` (or after a cache read).
    pub async fn record_sizes(&self, pk_bytes: u64, vk_bytes: u64, universal_bytes: u64) {
        let mut inner = self.inner.lock().await;
        inner.pk_bytes = pk_bytes;
        inner.vk_bytes = vk_bytes;
        inner.universal_bytes = universal_bytes;
    }

    /// Record the compute-vs-communication wall-clock split for this
    /// shard's setup pass. Called once at the end of
    /// [`run_distributed_compute`]. On single-shard runs
    /// `communication_secs` is 0 by construction (no peers to pull
    /// from). On the cache-hit path neither is set (both stay at 0).
    pub async fn record_phase_durations(&self, compute: Duration, communication: Duration) {
        let mut inner = self.inner.lock().await;
        inner.compute_secs = compute.as_secs_f64();
        inner.communication_secs = communication.as_secs_f64();
    }

    /// Add `n` to the inbound-slab byte counter. Called by the slab
    /// pull loop in [`run_distributed_compute`] after each peer's
    /// stream finishes.
    pub fn add_inbound_bytes(&self, n: u64) {
        self.inbound_slab_bytes.fetch_add(n, Ordering::Relaxed);
    }

    /// Add `n` to the outbound-slab byte counter. Called by the
    /// `GetSrsSlab` handler once it knows the served slab's size.
    pub fn add_outbound_bytes(&self, n: u64) {
        self.outbound_slab_bytes.fetch_add(n, Ordering::Relaxed);
    }

    /// Snapshot the per-shard metrics. Mostly consumed by the
    /// `GetMetrics` gRPC handler; exposed here so tests can poke at
    /// the state directly without going through gRPC.
    pub async fn snapshot_metrics(&self) -> MetricsSnapshot {
        let inner = self.inner.lock().await;
        let phases = inner
            .phase_log
            .iter()
            .map(|(label, instant)| PhaseEntry {
                phase: label.clone(),
                monotonic_secs: instant.duration_since(self.started_at).as_secs_f64(),
            })
            .collect();
        MetricsSnapshot {
            shard_id: self.config.shard_id,
            cache_hit: inner.cache_hit,
            phases,
            inbound_slab_bytes: self.inbound_slab_bytes.load(Ordering::Relaxed),
            outbound_slab_bytes: self.outbound_slab_bytes.load(Ordering::Relaxed),
            pk_bytes: inner.pk_bytes,
            vk_bytes: inner.vk_bytes,
            universal_bytes: inner.universal_bytes,
            compute_secs: inner.compute_secs,
            communication_secs: inner.communication_secs,
        }
    }

    /// Block until phase reaches `Ready` or `timeout` elapses. Returns
    /// `(ready, status)`. Zero timeout = block indefinitely.
    pub async fn wait_for_ready(&self, timeout: Duration) -> (bool, String) {
        let deadline = if timeout.is_zero() {
            None
        } else {
            Some(tokio::time::Instant::now() + timeout)
        };
        loop {
            let notified = self.notify.notified();
            {
                let inner = self.inner.lock().await;
                if inner.phase == Phase::Ready {
                    return (true, inner.status.clone());
                }
                if let Some(d) = deadline {
                    if tokio::time::Instant::now() >= d {
                        return (false, inner.status.clone());
                    }
                }
            }
            match deadline {
                None => notified.await,
                Some(d) => {
                    let remaining = d.saturating_duration_since(tokio::time::Instant::now());
                    if tokio::time::timeout(remaining, notified).await.is_err() {
                        // timed out — loop will catch and return on next pass
                    }
                }
            }
        }
    }

    /// Current bootstrap phase.
    pub async fn phase(&self) -> Phase {
        self.inner.lock().await.phase
    }

    /// Human-readable progress line for `WaitForReady`.
    pub async fn status(&self) -> String {
        self.inner.lock().await.status.clone()
    }
}

// ---------- gRPC server -------------------------------------------------

/// gRPC service wrapping a [`SrsBootstrapState`]. Add to the same
/// `tonic::transport::Server` as your other shard services — the SRS
/// path consumes negligible CPU once the local slab is built, so
/// sharing the port is fine.
pub struct SrsServer<E: Pairing> {
    state: Arc<SrsBootstrapState<E>>,
}

impl<E: Pairing> SrsServer<E> {
    /// Wrap a bootstrap actor as the gRPC `SrsService`.
    pub fn new(state: Arc<SrsBootstrapState<E>>) -> Self {
        Self { state }
    }

    /// Wrap this server in a tonic `SrsServiceServer` ready to add to a
    /// `Server::builder()`. Sets the 1 GiB caps that match the rest of
    /// the Aegon transports.
    pub fn into_service(self) -> SrsServiceServer<Self>
    where
        E: Send + Sync + 'static,
        E::G1Affine: Send + Sync,
    {
        SrsServiceServer::new(self)
            .max_decoding_message_size(MAX_MSG_BYTES)
            .max_encoding_message_size(MAX_MSG_BYTES)
    }
}

fn err_to_status(e: AegonError) -> Status {
    Status::internal(format!("{e}"))
}

#[tonic::async_trait]
impl<E> SrsService for SrsServer<E>
where
    E: Pairing + Send + Sync + 'static,
    E::G1Affine: Send + Sync,
{
    type GetSrsSlabStream =
        Pin<Box<dyn Stream<Item = Result<SrsSlabChunk, Status>> + Send + 'static>>;

    async fn bootstrap_srs(
        &self,
        req: Request<BootstrapSrsRequest>,
    ) -> Result<Response<BootstrapSrsResponse>, Status> {
        let req = req.into_inner();
        let cfg = self.state.config();
        if req.setup_seed != cfg.setup_seed {
            return Err(Status::failed_precondition(format!(
                "setup_seed mismatch: shard configured for {:#018x} but bootstrap pushed {:#018x}",
                cfg.setup_seed, req.setup_seed
            )));
        }
        if req.shard_id != cfg.shard_id {
            return Err(Status::failed_precondition(format!(
                "shard_id mismatch: shard configured as {} but bootstrap addressed {}",
                cfg.shard_id, req.shard_id
            )));
        }
        if req.kzh_k != cfg.k {
            return Err(Status::failed_precondition(format!(
                "kzh_k mismatch: shard configured for {} but bootstrap pushed {}",
                cfg.k, req.kzh_k
            )));
        }
        if req.log_capacity != cfg.log_capacity {
            return Err(Status::failed_precondition(format!(
                "log_capacity mismatch: shard configured for {} but bootstrap pushed {}",
                cfg.log_capacity, req.log_capacity
            )));
        }
        let n_shards = req.n_shards as usize;
        if req.peer_endpoints.len() != n_shards {
            return Err(Status::failed_precondition(format!(
                "peer_endpoints.len()={} != n_shards={}",
                req.peer_endpoints.len(),
                n_shards
            )));
        }
        let trapdoors = Trapdoors::<E>::decode(&req.trapdoors_uncompressed)
            .map_err(|e| Status::invalid_argument(format!("decode trapdoors: {e}")))?;
        let cache_hit = self
            .state
            .accept_bootstrap(trapdoors, req.peer_endpoints)
            .await;
        Ok(Response::new(BootstrapSrsResponse { cache_hit }))
    }

    async fn get_srs_slab(
        &self,
        req: Request<GetSrsSlabRequest>,
    ) -> Result<Response<Self::GetSrsSlabStream>, Status> {
        let req = req.into_inner();
        let t = req.t as usize;
        let slab = self
            .state
            .wait_for_slab(t)
            .await
            .ok_or_else(|| Status::failed_precondition(format!("t={t} out of range")))?;

        // Validate the requested range matches this shard's slab. The
        // caller derives `(range_start, range_end)` from
        // `slab_range(shard_id, n_shards, len)`, identical to what we
        // computed locally, so a mismatch indicates a topology
        // disagreement (different n_shards on different shards) and
        // should fail loudly.
        let expected_len = slab.len() as u64;
        let requested_len = req.range_end.saturating_sub(req.range_start);
        if requested_len != expected_len {
            return Err(Status::failed_precondition(format!(
                "slab len mismatch: got {} but local slab is {}",
                requested_len, expected_len
            )));
        }

        // Chunk + stream. We materialise all chunks up front because
        // the slab is already in memory — the only reason for
        // streaming is to bound the wire-level per-message size below
        // the 1 GiB cap, not to overlap with compute.
        let chunks = encode_slab_to_chunks::<E>(&slab).map_err(err_to_status)?;
        let served_bytes: u64 = chunks.iter().map(|(b, _)| b.len() as u64).sum();
        self.state.add_outbound_bytes(served_bytes);
        let stream = futures::stream::iter(
            chunks
                .into_iter()
                .map(|(bytes, n)| {
                    Ok(SrsSlabChunk {
                        points_uncompressed: bytes,
                        n_points: n,
                    })
                })
                .collect::<Vec<Result<SrsSlabChunk, Status>>>(),
        );
        Ok(Response::new(Box::pin(stream)))
    }

    async fn wait_for_ready(
        &self,
        req: Request<WaitForReadyRequest>,
    ) -> Result<Response<WaitForReadyResponse>, Status> {
        let req = req.into_inner();
        let timeout = if req.timeout_secs == 0 {
            Duration::ZERO
        } else {
            Duration::from_secs(req.timeout_secs)
        };
        let (ready, status) = self.state.wait_for_ready(timeout).await;
        Ok(Response::new(WaitForReadyResponse { ready, status }))
    }

    async fn get_metrics(
        &self,
        _req: Request<GetMetricsRequest>,
    ) -> Result<Response<GetMetricsResponse>, Status> {
        let snap = self.state.snapshot_metrics().await;
        Ok(Response::new(GetMetricsResponse {
            shard_id: snap.shard_id,
            cache_hit: snap.cache_hit,
            phases: snap.phases,
            inbound_slab_bytes: snap.inbound_slab_bytes,
            outbound_slab_bytes: snap.outbound_slab_bytes,
            pk_bytes: snap.pk_bytes,
            vk_bytes: snap.vk_bytes,
            universal_bytes: snap.universal_bytes,
            compute_secs: snap.compute_secs,
            communication_secs: snap.communication_secs,
        }))
    }
}

// ---------- gRPC clients ------------------------------------------------

/// Connect a `SrsServiceClient` against `endpoint` with the standard
/// Aegon transport caps. Async; safe to call from any tokio runtime.
pub async fn connect_srs_client(endpoint: &str) -> Result<SrsServiceClient<Channel>, AegonError> {
    let channel = tonic::transport::Endpoint::from_shared(endpoint.to_string())
        .map_err(|e| AegonError::Config(format!("srs endpoint: {e}")))?
        .connect()
        .await
        .map_err(|e| AegonError::Config(format!("srs connect '{endpoint}': {e}")))?;
    Ok(SrsServiceClient::new(channel)
        .max_decoding_message_size(MAX_MSG_BYTES)
        .max_encoding_message_size(MAX_MSG_BYTES))
}

/// Pull a single peer's slab of `H_t` over a streaming gRPC. Retries
/// with backoff on transient transport errors — peers might not be up
/// yet when we first try to connect, since the bootstrap actor doesn't
/// barrier them.
///
/// Returns `(slab, payload_bytes)` where `payload_bytes` is the sum of
/// `points_uncompressed.len()` across received chunks (used by the
/// caller to drive the inbound byte counter). The caller is responsible
/// for inserting the slab into the assembly matrix.
pub async fn pull_slab_from_peer<E: Pairing>(
    endpoint: &str,
    t: u32,
    range_start: u64,
    range_end: u64,
) -> Result<(Vec<E::G1Affine>, u64), AegonError> {
    // Connect-with-retry: each shard binds gRPC roughly simultaneously
    // but DNS / link-up jitter can leave one endpoint unroutable for a
    // few seconds. Six tries × 1-second sleep = 6s tolerance, which
    // matches what the rest of the cluster uses.
    let mut last_err: Option<AegonError> = None;
    let mut client_opt: Option<SrsServiceClient<Channel>> = None;
    for attempt in 0..6 {
        match connect_srs_client(endpoint).await {
            Ok(c) => {
                client_opt = Some(c);
                break;
            }
            Err(e) => {
                last_err = Some(e);
                tokio::time::sleep(Duration::from_secs(1)).await;
                let _ = attempt;
            }
        }
    }
    let mut client = client_opt.ok_or_else(|| {
        last_err.unwrap_or_else(|| AegonError::Config(format!("connect '{endpoint}' failed")))
    })?;

    let mut stream = client
        .get_srs_slab(Request::new(GetSrsSlabRequest {
            t,
            range_start,
            range_end,
        }))
        .await
        .map_err(|s| AegonError::Config(format!("get_srs_slab '{endpoint}': {s}")))?
        .into_inner();
    let expected: usize = (range_end - range_start) as usize;
    let mut out: Vec<E::G1Affine> = Vec::with_capacity(expected);
    let mut bytes_in: u64 = 0;
    while let Some(chunk) = stream
        .message()
        .await
        .map_err(|s| AegonError::Config(format!("stream '{endpoint}': {s}")))?
    {
        let n = chunk.n_points as usize;
        bytes_in += chunk.points_uncompressed.len() as u64;
        let pts = decode_slab_chunk::<E>(&chunk.points_uncompressed, n)?;
        out.extend(pts);
    }
    if out.len() != expected {
        return Err(AegonError::Config(format!(
            "pull from '{endpoint}': got {} pts, expected {}",
            out.len(),
            expected
        )));
    }
    Ok((out, bytes_in))
}

// ---------- orchestration helpers --------------------------------------

/// Run the cache-miss path end-to-end on one shard:
///
///   1. Wait for trapdoors via [`SrsBootstrapState::wait_for_trapdoors`].
///   2. Compute local slabs of every `H_t`; publish each as it lands so
///      peers can start pulling immediately.
///   3. Pull peer slabs in parallel from every other shard's
///      `GetSrsSlab` endpoint.
///   4. Assemble the full `H_t` tensors, build the universal SRS,
///      derive prover_param + verifier_param, write the cache file.
///
/// Returns `(universal_params, prover_param, verifier_param)`. Caller
/// proceeds to initialise Aegon and run prefill, then calls
/// `state.set_phase(Phase::Ready, ...)` to release `WaitForReady`
/// callers.
pub async fn run_distributed_compute<E: Pairing + Send + Sync + 'static>(
    state: Arc<SrsBootstrapState<E>>,
) -> Result<
    (
        KZHKUniversalParams<E>,
        KZHKProverParam<E>,
        KZHKVerifierParam<E>,
    ),
    AegonError,
>
where
    E::G1Affine: Send + Sync + 'static,
    KZHKUniversalParams<E>: CanonicalSerialize,
    KZHKProverParam<E>: CanonicalSerialize,
    KZHKVerifierParam<E>: CanonicalSerialize,
{
    let (trapdoors, peer_endpoints) = state.wait_for_trapdoors().await;
    let n_shards = peer_endpoints.len();
    let shard_id = state.config().shard_id as usize;
    let geoms = HtGeometry::all(&trapdoors.dimensions);

    // Phase-duration accounting. Two buckets — CPU-bound work
    // (`compute_acc`) and peer slab exchange wall-clock
    // (`communication_acc`). The bootstrap actor's per-shard JSON
    // reports both; cluster-wide aggregators take the max across
    // shards (since they run in parallel).
    let mut compute_acc = Duration::ZERO;
    let mut communication_acc = Duration::ZERO;

    // ---- local slab compute (in a blocking task — MSM is CPU-heavy) ----
    state
        .set_phase(Phase::Computing, "computing-local-slabs")
        .await;
    let compute_start = Instant::now();
    for (t_idx, geom) in geoms.iter().enumerate() {
        let (start, end) = slab_range(shard_id, n_shards, geom.len);
        let trapdoors_for_task = Arc::clone(&trapdoors);
        let geom_for_task = geom.clone();
        let slab = tokio::task::spawn_blocking(move || {
            compute_h_t_slab::<E>(&trapdoors_for_task, &geom_for_task, start, end)
        })
        .await
        .map_err(|e| AegonError::Config(format!("compute_h_t_slab join: {e}")))?;
        state.set_local_slab(t_idx, slab).await;
    }
    compute_acc += compute_start.elapsed();

    state
        .set_phase(Phase::Computing, "pulling-peer-slabs")
        .await;
    let comm_start = Instant::now();

    // ---- pull peer slabs in parallel --------------------------------
    // For each (t, peer), spawn an async task. tokio's scheduler handles
    // the actual concurrency; each task is mostly waiting on network IO.
    let mut matrix: SlabMatrix<E> = (0..geoms.len()).map(|_| BTreeMap::new()).collect();

    // Drain the local slabs into the matrix first — they're already in
    // memory, and consuming them frees the duplicate Arc-held copy in
    // the state's local_slabs table after assembly.
    for (t_idx, geom) in geoms.iter().enumerate() {
        let (start, end) = slab_range(shard_id, n_shards, geom.len);
        let local = state
            .wait_for_slab(t_idx)
            .await
            .ok_or_else(|| AegonError::Config(format!("missing own slab t={t_idx}")))?;
        // wait_for_slab returns an Arc — take a snapshot by cloning the
        // inner Vec. Cheap relative to the assembly cost.
        let local_vec: Vec<E::G1Affine> = (*local).clone();
        if local_vec.len() != end - start {
            return Err(AegonError::Config(format!(
                "local slab t={t_idx} len {} != range {}-{}",
                local_vec.len(),
                start,
                end
            )));
        }
        matrix[t_idx].insert(shard_id, local_vec);
    }

    // Pull from peers concurrently across all (t, peer) pairs.
    let mut pull_tasks = Vec::new();
    for (t_idx, geom) in geoms.iter().enumerate() {
        for peer_id in 0..n_shards {
            if peer_id == shard_id {
                continue;
            }
            let (start, end) = slab_range(peer_id, n_shards, geom.len);
            if start == end {
                // empty slab — nothing to fetch
                let t = t_idx;
                pull_tasks.push(tokio::spawn(async move {
                    Ok::<(usize, usize, Vec<E::G1Affine>, u64), AegonError>((
                        t,
                        peer_id,
                        Vec::new(),
                        0,
                    ))
                }));
                continue;
            }
            let endpoint = peer_endpoints[peer_id].clone();
            let t = t_idx as u32;
            pull_tasks.push(tokio::spawn(async move {
                let (slab, bytes_in) =
                    pull_slab_from_peer::<E>(&endpoint, t, start as u64, end as u64).await?;
                Ok::<(usize, usize, Vec<E::G1Affine>, u64), AegonError>((
                    t as usize, peer_id, slab, bytes_in,
                ))
            }));
        }
    }

    for task in pull_tasks {
        let (t_idx, peer_id, slab, bytes_in) = task
            .await
            .map_err(|e| AegonError::Config(format!("pull task join: {e}")))??;
        state.add_inbound_bytes(bytes_in);
        matrix[t_idx].insert(peer_id, slab);
    }

    communication_acc += comm_start.elapsed();

    // ---- assemble + write cache --------------------------------------
    state.set_phase(Phase::Assembling, "assembling-srs").await;
    let assembly_start = Instant::now();
    let h_tensors = matrix_to_tensors::<E>(&geoms, matrix, n_shards)?;
    let v_mat = compute_v_mat(&trapdoors);
    let universal_params = build_universal_params(&trapdoors, h_tensors, v_mat);
    let (prover_param, verifier_param) = {
        use akd_core::aegon_crypto::StructuredReferenceString;
        universal_params
            .trim(trapdoors.num_vars())
            .map_err(|e| AegonError::Config(format!("trim: {e}")))?
    };
    compute_acc += assembly_start.elapsed();

    state.set_phase(Phase::Assembling, "writing-cache").await;
    let cfg = state.config();
    let cache_path = cache_file_path(
        &cfg.cache_dir,
        cfg.log_capacity as usize,
        cfg.k as usize,
        cfg.setup_seed,
    );
    write_cache::<E>(
        &cache_path,
        &universal_params,
        &prover_param,
        &verifier_param,
    )?;

    // Snapshot the assembled sizes so `GetMetrics` can return them
    // without a second serialise pass.
    state
        .record_sizes(
            prover_param.uncompressed_size() as u64,
            verifier_param.uncompressed_size() as u64,
            universal_params.uncompressed_size() as u64,
        )
        .await;

    // Free the per-`H_t` slab cache before init/prefill kicks off; the
    // bytes we'd be holding on to are already in the assembled tensors.
    state.drop_local_slabs().await;

    // Final compute/communication split so `GetMetrics` returns it
    // without the consumer having to derive deltas from the phase
    // log. `compute_acc` covers slab MSM + assembly + v_mat + trim;
    // `communication_acc` covers the peer-pull wall-clock. Cache I/O
    // and trapdoor wait are deliberately excluded from both.
    state
        .record_phase_durations(compute_acc, communication_acc)
        .await;

    Ok((universal_params, prover_param, verifier_param))
}

/// Try the cache path first. Returns `Some(...)` on cache hit (and
/// advances `state` past `AwaitingBootstrap` so a stray bootstrap RPC
/// reports `cache_hit = true`); returns `None` otherwise.
pub async fn try_cache_hit<E: Pairing>(
    state: &Arc<SrsBootstrapState<E>>,
) -> Result<
    Option<(
        KZHKUniversalParams<E>,
        KZHKProverParam<E>,
        KZHKVerifierParam<E>,
    )>,
    AegonError,
>
where
    KZHKUniversalParams<E>: CanonicalDeserialize,
    KZHKProverParam<E>: CanonicalDeserialize,
    KZHKVerifierParam<E>: CanonicalDeserialize,
{
    let cfg = state.config();
    let path = cache_file_path(
        &cfg.cache_dir,
        cfg.log_capacity as usize,
        cfg.k as usize,
        cfg.setup_seed,
    );
    if !path.exists() {
        return Ok(None);
    }
    let (up, pk, vk) = read_cache::<E>(&path)?;
    // Jump straight to Initializing — the trapdoor handoff is bypassed
    // because we never needed it.
    state.mark_cache_hit().await;
    state
        .record_sizes(
            pk.uncompressed_size() as u64,
            vk.uncompressed_size() as u64,
            up.uncompressed_size() as u64,
        )
        .await;
    state.set_phase(Phase::Initializing, "cache-hit").await;
    Ok(Some((up, pk, vk)))
}
