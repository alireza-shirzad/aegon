//! Size-aware multi-scalar-multiplication wrapper used throughout KZH-k.
//!
//! # Why not call arkworks directly
//!
//! `ark_ec::VariableBaseMSM::msm_unchecked` always goes through Pippenger,
//! which pays bucket-allocation and reduction overhead even for very small
//! inputs, and (with its `parallel` feature) builds a fresh rayon
//! `ThreadPool` per call. Neither is desirable for the access patterns
//! KZH-k generates — in particular, `update_state_*` produces thousands of
//! tiny per-chunk / per-bucket MSMs, and those calls can fire from inside
//! an already-parallel region in which a nested pool build would exhaust
//! the OS thread limit.
//!
//! This wrapper adds three decisions on top of `msm_unchecked`:
//!
//! 1. **Algorithm selection by size.** For `N ≤ NAIVE_MSM_THRESHOLD` we
//!    skip Pippenger entirely and just compute `∑ sᵢ·Bᵢ` with a tight
//!    `mul_bigint` + add loop. On BN254, Pippenger only starts winning
//!    around ~32–100 terms; below that, naive summation is both simpler
//!    and faster because it has zero setup cost. The threshold is
//!    conservative (100) so that callers parallelizing an outer loop of
//!    small MSMs can rely on *every* inner call taking the naive path.
//!
//! 2. **Nested-pool avoidance.** Under `--features parallel`, if we are
//!    already inside a rayon worker (`rayon::current_thread_index()` is
//!    `Some`), we call `msm_unchecked` directly without building a pool.
//!    This uses the ambient pool's parallelism and avoids the macOS
//!    `EAGAIN` / `ThreadPoolBuildError` failures that occur when many
//!    parallel tasks each try to spawn their own OS threads.
//!
//! 3. **Sized ad-hoc pool for top-level calls.** When called from outside
//!    a rayon region and `N` is large enough to warrant parallelism, we
//!    build a short-lived rayon pool whose thread count is picked by
//!    [`threads_for_n`] — a small step function of `N`, capped by the
//!    machine's physical core count. The heuristic targets ≥ ~128 terms
//!    per worker so Pippenger's overhead is amortized; `N < 32` runs
//!    single-threaded (and in practice takes the naive path above
//!    anyway), while very large `N` (≥ 4096) is allowed up to 64
//!    threads. See the table in [`threads_for_n`].
//!
//! # Feature gating
//!
//! Without `--features parallel` the code compiles down to: small-N
//! naive path, or `msm_unchecked` sequentially. No rayon dependency is
//! pulled in.

use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::PrimeField;

use ark_ec::PrimeGroup;
use ark_std::Zero;

/// Upper bound below which `msm_wrapper_g1` uses a naive sum-of-scalar-mults
/// instead of arkworks' Pippenger. Exposed so callers can decide whether
/// their per-chunk / per-bucket MSMs are all small enough that the outer
/// loop can be parallelized without causing nested rayon pool builds
/// inside arkworks' Pippenger.
pub const NAIVE_MSM_THRESHOLD: usize = 100;
#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;
// ===============================
// Public API (G1 / G2)
// ===============================

/// MSM in `G1` sized by input length.
///
/// Dispatch rules (see module docs for the rationale):
///
/// - `N ≤ NAIVE_MSM_THRESHOLD` → naive `∑ sᵢ·Bᵢ` loop, no rayon pool.
/// - Already inside a rayon worker → `ark_ec::VariableBaseMSM::msm_unchecked`
///   directly, using the ambient pool.
/// - Top-level call, `N` large → build a short-lived rayon pool sized by
///   [`threads_for_n`] and run `msm_unchecked` inside it.
/// - `--features parallel` disabled → always sequential `msm_unchecked`
///   (after the naive path).
pub fn msm_wrapper_g1<E: Pairing>(
    bases: &[<E::G1 as CurveGroup>::Affine],
    scalars: &[E::ScalarField],
) -> E::G1
where
    E::ScalarField: PrimeField,
    <E::G1 as CurveGroup>::Affine: AffineRepr<ScalarField = E::ScalarField, Group = E::G1>,
{
    msm_wrapper_affine::<E, <E::G1 as CurveGroup>::Affine>(bases, scalars)
}

/// `G2` counterpart of [`msm_wrapper_g1`]. KZH-k currently performs most
/// multi-scalar multiplications in `G1`; this entry point exists for
/// symmetry and for any future `G2`-side aggregations.
pub fn msm_wrapper_g2<E: Pairing>(
    bases: &[<E::G2 as CurveGroup>::Affine],
    scalars: &[E::ScalarField],
) -> E::G2
where
    E::ScalarField: PrimeField,
    <E::G2 as CurveGroup>::Affine: AffineRepr<ScalarField = E::ScalarField, Group = E::G2>,
{
    msm_wrapper_affine::<E, <E::G2 as CurveGroup>::Affine>(bases, scalars)
}

// ===============================
// Core wrapper (generic Affine)
// ===============================

pub fn msm_wrapper_affine<E, A>(bases: &[A], scalars: &[E::ScalarField]) -> A::Group
where
    E: Pairing,
    E::ScalarField: PrimeField,
    A: AffineRepr<ScalarField = E::ScalarField>,
    A::Group: VariableBaseMSM,
{
    // Length check matches arkworks' msm API
    if bases.len() != scalars.len() {
        panic!()
    }

    // Small-input fast path: naive summation of scalar-mults.
    //
    // Arkworks' `msm_unchecked` always goes through Pippenger, which pays
    // bucket-allocation and reduction overhead even for very small `N`.
    // For BN254 Pippenger only starts winning around ~32 terms; below that
    // a plain `sum(s_i * B_i)` is faster. This matters a lot for sparse
    // polynomials where the per-bucket MSMs in `update_state_sparse`
    // degenerate to 1–tens of terms each (see `KZHK::update_state_sparse`).
    if bases.len() <= NAIVE_MSM_THRESHOLD {
        let mut acc = A::Group::zero();
        for (b, s) in bases.iter().zip(scalars.iter()) {
            acc += b.into_group().mul_bigint(s.into_bigint());
        }
        return acc;
    }

    // Non-parallel build: just call arkworks MSM
    #[cfg(not(feature = "parallel"))]
    {
        return <A::Group as VariableBaseMSM>::msm_unchecked(bases, scalars);
    }

    // Parallel build: run MSM inside a small pool with fixed-heuristic threads
    #[cfg(feature = "parallel")]
    {
        // If we're already inside a rayon worker thread, don't build a nested
        // `ThreadPool` — that spawns fresh OS threads per call and will
        // exhaust the per-process thread limit when the caller is itself
        // running many MSMs in parallel (e.g. `update_state_dense`'s
        // `cfg_iter_mut!` over `d_j`). The ambient pool already has
        // parallelism; arkworks' `msm_unchecked` will use it.
        if rayon::current_thread_index().is_some() {
            return <A::Group as VariableBaseMSM>::msm_unchecked(bases, scalars);
        }

        let n = bases.len();
        let phys = detect_cores();
        let threads = threads_for_n(n, phys);

        // If threads == 1, avoid building a pool
        if threads <= 1 {
            return <A::Group as VariableBaseMSM>::msm_unchecked(bases, scalars);
        }

        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("failed to build rayon pool");

        pool.install(|| <A::Group as VariableBaseMSM>::msm_unchecked(bases, scalars))
    }
}

// ===============================
// Fixed heuristic thread picker
// ===============================

/// Picks a rayon worker count for a top-level MSM of size `n`.
///
/// Fixed step function — no autotuning, no runtime sampling:
///
/// | n             | threads |
/// |---------------|---------|
/// | `< 32`        | 1       |
/// | `< 256`       | 2       |
/// | `< 512`       | 16      |
/// | `< 4096`      | 32      |
/// | `≥ 4096`      | 64      |
///
/// Then capped by the machine's physical core count.
///
/// Rationale: Pippenger's per-worker overhead dominates for very small
/// `n` (and we've already short-circuited those through the naive path);
/// as `n` grows we let more cores in, but the cap keeps us from
/// saturating a big machine on a moderate MSM and helps avoid nested
/// pool explosions when several wrappers fire concurrently.
#[cfg(feature = "parallel")]
fn threads_for_n(n: usize, phys_cores: usize) -> usize {
    let t = if n < 32 {
        1
    } else if n < 256 {
        2
    } else if n < 512 {
        16
    } else if n < 4096 {
        32
    } else {
        64
    };
    t.min(phys_cores.max(1))
}

/// Physical-core estimate used as an upper bound in [`threads_for_n`].
/// Falls back to 1 if the platform can't answer.
#[cfg(feature = "parallel")]
fn detect_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .max(1)
}
