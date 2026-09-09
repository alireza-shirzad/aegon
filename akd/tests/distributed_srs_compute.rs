// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Distributed SRS gen verifies against the in-process reference SRS.
//!
//! The load-bearing correctness check for the distributed path: if
//! slab boundaries or per-entry math drift, every cluster-wide
//! opening/verify silently misbehaves until something pairing-checks
//! loudly. So we pin the math at trapdoor + slab + assembly time by
//! comparing serialised bytes against `gen_srs_for_testing` from the
//! same seed.

use std::collections::BTreeMap;

use akd::aegon::distributed_srs::{
    build_universal_params, cache_file_path, compute_h_t_slab, compute_v_mat, matrix_to_tensors,
    read_cache, slab_range, write_cache, HtGeometry, SlabMatrix, Trapdoors,
};
use akd_core::aegon_crypto::pcs::kzhk::srs::KZHKUniversalParams as RefParams;
use akd_core::aegon_crypto::StructuredReferenceString;
use ark_bn254::Bn254;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

type E = Bn254;

fn slab_round_trip(k: usize, num_vars: usize, n_shards: usize) {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let reference =
        RefParams::<E>::gen_srs_for_testing(&mut rng, k, true, num_vars).expect("ref gen");

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let trapdoors: Trapdoors<E> = Trapdoors::sample(&mut rng, k, num_vars);
    let geoms = HtGeometry::all(&trapdoors.dimensions);

    let mut matrix: SlabMatrix<E> = (0..geoms.len()).map(|_| BTreeMap::new()).collect();
    for (t_idx, geom) in geoms.iter().enumerate() {
        for s in 0..n_shards {
            let (start, end) = slab_range(s, n_shards, geom.len);
            let slab = compute_h_t_slab(&trapdoors, geom, start, end);
            matrix[t_idx].insert(s, slab);
        }
    }
    let h_tensors = matrix_to_tensors::<E>(&geoms, matrix, n_shards).expect("assemble");
    let v_mat = compute_v_mat(&trapdoors);
    let dist = build_universal_params(&trapdoors, h_tensors, v_mat);

    let mut ref_bytes = Vec::new();
    reference
        .serialize_uncompressed(&mut ref_bytes)
        .expect("ref serialise");
    let mut dist_bytes = Vec::new();
    dist.serialize_uncompressed(&mut dist_bytes)
        .expect("dist serialise");
    assert_eq!(
        ref_bytes.len(),
        dist_bytes.len(),
        "SRS sizes differ: ref={} dist={}",
        ref_bytes.len(),
        dist_bytes.len()
    );
    assert!(
        ref_bytes == dist_bytes,
        "distributed SRS != reference for k={k} nv={num_vars} n_shards={n_shards}"
    );
}

#[test]
fn distributed_srs_matches_ref_k3_nv6_shards4() {
    slab_round_trip(3, 6, 4);
}

#[test]
fn distributed_srs_matches_ref_k4_nv8_shards8() {
    slab_round_trip(4, 8, 8);
}

#[test]
fn distributed_srs_matches_ref_k2_nv5_shards3() {
    // Non-power-of-two n_shards and len not divisible by n_shards:
    // slab boundaries land at floor((i * len) / n_shards), which tiles
    // exactly but with non-equal widths.
    slab_round_trip(2, 5, 3);
}

#[test]
fn distributed_srs_matches_ref_when_some_slabs_empty() {
    // len(H_{k-1}) = 4 with n_shards=8 -> only the first 4 shards get
    // non-empty slabs of H_{k-1}, the rest contribute empty slabs.
    // Assembly must still concatenate correctly.
    slab_round_trip(3, 6, 8);
}

#[test]
fn slab_range_tiles_exactly() {
    for &len in &[0usize, 1, 3, 7, 16, 31, 32, 1_000_001] {
        for &n in &[1usize, 2, 3, 5, 8, 32] {
            let mut last_end = 0usize;
            for s in 0..n {
                let (start, end) = slab_range(s, n, len);
                assert_eq!(start, last_end, "gap at s={s} n={n} len={len}");
                assert!(start <= end);
                last_end = end;
            }
            assert_eq!(last_end, len, "incomplete tile n={n} len={len}");
        }
    }
}

#[test]
fn cache_round_trip() {
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    let up = RefParams::<E>::gen_srs_for_testing(&mut rng, 3, true, 6).expect("gen");
    let (pk, vk) = up.trim(6).expect("trim");

    let tmp = std::env::temp_dir().join(format!(
        "aegon-srs-cache-test-{}-{}",
        std::process::id(),
        ark_std::rand::random::<u64>()
    ));
    std::fs::create_dir_all(&tmp).expect("mkdir");
    let path = cache_file_path(&tmp, 6, 3, 0xdead_beefu64);
    write_cache::<E>(&path, &up, &pk, &vk).expect("write");

    let (up2, pk2, vk2) = read_cache::<E>(&path).expect("read");

    let mut a = Vec::new();
    let mut b = Vec::new();
    up.serialize_uncompressed(&mut a).unwrap();
    up2.serialize_uncompressed(&mut b).unwrap();
    assert_eq!(a, b, "universal_params bytes differ after round-trip");

    a.clear();
    b.clear();
    pk.serialize_uncompressed(&mut a).unwrap();
    pk2.serialize_uncompressed(&mut b).unwrap();
    assert_eq!(a, b, "prover_param bytes differ after round-trip");

    a.clear();
    b.clear();
    vk.serialize_uncompressed(&mut a).unwrap();
    vk2.serialize_uncompressed(&mut b).unwrap();
    assert_eq!(a, b, "verifier_param bytes differ after round-trip");

    let _ = std::fs::remove_dir_all(&tmp);
}
