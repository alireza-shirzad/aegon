//! Standalone MSM calibration binary.
//!
//! Build with `cargo build --release -p akd --bin aegon_calibrate_msm`,
//! scp to a shard VM, run there to get the hardware-specific
//! `NAIVE_THRESHOLD` and `THREAD_TABLE` constants for that machine, and
//! paste the printed constants into
//! `akd_core/src/aegon_crypto/pcs/kzhk/msm.rs`.
//!
//! No flags — sizes and iteration counts are baked in. Calibration on a
//! 4-vCPU VM takes ~1 minute end-to-end.

use ark_bn254::G1Projective;
use akd_core::aegon_crypto::pcs::kzhk::msm::calibrate;

fn main() {
    // Sizes spanning the regime KZH-k's per-bucket / per-chunk MSMs
    // actually hit during publishes (a handful to a few hundred K).
    let sizes: &[usize] = &[
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072,
    ];
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let iters = 5;

    println!(
        "calibrating bn254 G1 MSM, max_threads={max_threads}, iters={iters} on {} sizes",
        sizes.len()
    );
    let cal = calibrate::calibrate::<G1Projective>(sizes, max_threads, iters);
    println!("\n--- recommended constants (paste into msm.rs) ---");
    cal.print_as_constants();
}
