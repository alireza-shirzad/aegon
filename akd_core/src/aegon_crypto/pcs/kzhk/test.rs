use super::*;
use crate::aegon_crypto::pcs::kzhk::structs::KZHKConfig;
use ark_bn254::{Bn254 as E, Fr};
use ark_std::{test_rng, vec::Vec, UniformRand};

fn test_single_helper(
    nv: usize,
    zk: bool,
    is_sparse: bool,
    is_boolean: bool,
    k: usize,
) -> Result<(), PCSError> {
    let mut rng = test_rng();
    let poly = if is_sparse {
        DenseOrSparseMLE::Sparse(SparseMultilinearExtension::<Fr>::rand(nv, &mut rng))
    } else {
        DenseOrSparseMLE::Dense(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng))
    };
    let mut prover_transcript = IOPTranscript::new(b"test_kzhk");
    let params = KZHK::<E>::gen_srs_for_testing(KZHKConfig::new(k, zk), &mut rng, nv)?;
    let (ck, vk) = KZHK::trim(params, None, Some(nv))?;
    let point = match is_boolean {
        true => (0..nv)
            .map(|_| Fr::from((usize::rand(&mut rng) % 2) as i64))
            .collect::<Vec<_>>(),
        false => (0..nv).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>(),
    };
    let (com, mut state) = KZHK::<E>::commit(&ck, &poly)?;
    KZHK::<E>::update_state(&ck, &poly, &com, &mut state)?;
    let (proof, value) = KZHK::<E>::open(
        &ck,
        &com,
        poly.as_ref(),
        &point,
        &state,
        &mut prover_transcript,
    )?;
    let mut verif_transcript = IOPTranscript::new(b"test_kzhk");
    assert!(KZHK::<E>::verify(
        &vk,
        &com,
        &point,
        &value,
        &proof,
        &mut verif_transcript,
    )?);

    Ok(())
}
#[test]
fn test_dense_k2() -> Result<(), PCSError> {
    test_single_helper(2, false, false, false, 2)?;
    test_single_helper(3, false, false, false, 2)?;
    test_single_helper(4, false, false, false, 2)?;
    test_single_helper(5, false, false, false, 2)?;
    test_single_helper(6, false, false, false, 2)?;
    test_single_helper(7, false, false, false, 2)?;
    test_single_helper(8, false, false, false, 2)?;
    test_single_helper(9, false, false, false, 2)?;
    test_single_helper(10, false, false, false, 2)?;
    test_single_helper(11, false, false, false, 2)?;
    test_single_helper(12, false, false, false, 2)?;
    test_single_helper(13, false, false, false, 2)?;
    test_single_helper(14, false, false, false, 2)?;
    test_single_helper(15, false, false, false, 2)?;
    Ok(())
}

#[test]
fn test_dense_zk_k2() -> Result<(), PCSError> {
    test_single_helper(2, true, false, false, 2)?;
    test_single_helper(3, true, false, false, 2)?;
    test_single_helper(4, true, false, false, 2)?;
    test_single_helper(5, true, false, false, 2)?;
    test_single_helper(6, true, false, false, 2)?;
    test_single_helper(7, true, false, false, 2)?;
    test_single_helper(8, true, false, false, 2)?;
    test_single_helper(9, true, false, false, 2)?;
    test_single_helper(10, true, false, false, 2)?;
    test_single_helper(11, true, false, false, 2)?;
    test_single_helper(12, true, false, false, 2)?;
    test_single_helper(13, true, false, false, 2)?;
    test_single_helper(14, true, false, false, 2)?;
    test_single_helper(15, true, false, false, 2)?;
    Ok(())
}

#[test]
fn test_dense_boolean_k2() -> Result<(), PCSError> {
    test_single_helper(2, false, false, true, 2)?;
    test_single_helper(3, false, false, true, 2)?;
    test_single_helper(4, false, false, true, 2)?;
    test_single_helper(5, false, false, true, 2)?;
    test_single_helper(6, false, false, true, 2)?;
    test_single_helper(7, false, false, true, 2)?;
    test_single_helper(8, false, false, true, 2)?;
    test_single_helper(9, false, false, true, 2)?;
    test_single_helper(10, false, false, true, 2)?;
    test_single_helper(11, false, false, true, 2)?;
    test_single_helper(12, false, false, true, 2)?;
    test_single_helper(13, false, false, true, 2)?;
    test_single_helper(14, false, false, true, 2)?;
    test_single_helper(15, false, false, true, 2)?;
    Ok(())
}

#[test]
fn test_dense_boolean_zk_k2() -> Result<(), PCSError> {
    test_single_helper(2, true, false, true, 2)?;
    test_single_helper(3, true, false, true, 2)?;
    test_single_helper(4, true, false, true, 2)?;
    test_single_helper(5, true, false, true, 2)?;
    test_single_helper(6, true, false, true, 2)?;
    test_single_helper(7, true, false, true, 2)?;
    test_single_helper(8, true, false, true, 2)?;
    test_single_helper(9, true, false, true, 2)?;
    test_single_helper(10, true, false, true, 2)?;
    test_single_helper(11, true, false, true, 2)?;
    test_single_helper(12, true, false, true, 2)?;
    test_single_helper(13, true, false, true, 2)?;
    test_single_helper(14, true, false, true, 2)?;
    test_single_helper(15, true, false, true, 2)?;
    Ok(())
}

#[test]
fn test_sparse_k2() -> Result<(), PCSError> {
    test_single_helper(2, false, true, false, 2)?;
    test_single_helper(3, false, true, false, 2)?;
    test_single_helper(4, false, true, false, 2)?;
    test_single_helper(5, false, true, false, 2)?;
    test_single_helper(6, false, true, false, 2)?;
    test_single_helper(7, false, true, false, 2)?;
    test_single_helper(8, false, true, false, 2)?;
    test_single_helper(9, false, true, false, 2)?;
    test_single_helper(10, false, true, false, 2)?;
    test_single_helper(11, false, true, false, 2)?;
    test_single_helper(12, false, true, false, 2)?;
    test_single_helper(13, false, true, false, 2)?;
    test_single_helper(14, false, true, false, 2)?;
    test_single_helper(15, false, true, false, 2)?;
    Ok(())
}

#[test]
fn test_sparse_zk_k2() -> Result<(), PCSError> {
    test_single_helper(2, true, true, false, 2)?;
    test_single_helper(3, true, true, false, 2)?;
    test_single_helper(4, true, true, false, 2)?;
    test_single_helper(5, true, true, false, 2)?;
    test_single_helper(6, true, true, false, 2)?;
    test_single_helper(7, true, true, false, 2)?;
    test_single_helper(8, true, true, false, 2)?;
    test_single_helper(9, true, true, false, 2)?;
    test_single_helper(10, true, true, false, 2)?;
    test_single_helper(11, true, true, false, 2)?;
    test_single_helper(12, true, true, false, 2)?;
    test_single_helper(13, true, true, false, 2)?;
    test_single_helper(14, true, true, false, 2)?;
    test_single_helper(15, true, true, false, 2)?;
    Ok(())
}

#[test]
fn test_sparse_boolean_k2() -> Result<(), PCSError> {
    test_single_helper(2, false, true, true, 2)?;
    test_single_helper(3, false, true, true, 2)?;
    test_single_helper(4, false, true, true, 2)?;
    test_single_helper(5, false, true, true, 2)?;
    test_single_helper(6, false, true, true, 2)?;
    test_single_helper(7, false, true, true, 2)?;
    test_single_helper(8, false, true, true, 2)?;
    test_single_helper(9, false, true, true, 2)?;
    test_single_helper(10, false, true, true, 2)?;
    test_single_helper(11, false, true, true, 2)?;
    test_single_helper(12, false, true, true, 2)?;
    test_single_helper(13, false, true, true, 2)?;
    test_single_helper(14, false, true, true, 2)?;
    test_single_helper(15, false, true, true, 2)?;
    Ok(())
}

#[test]
fn test_sparse_boolean_zk_k2() -> Result<(), PCSError> {
    test_single_helper(2, true, true, true, 2)?;
    test_single_helper(3, true, true, true, 2)?;
    test_single_helper(4, true, true, true, 2)?;
    test_single_helper(5, true, true, true, 2)?;
    test_single_helper(6, true, true, true, 2)?;
    test_single_helper(7, true, true, true, 2)?;
    test_single_helper(8, true, true, true, 2)?;
    test_single_helper(9, true, true, true, 2)?;
    test_single_helper(10, true, true, true, 2)?;
    test_single_helper(11, true, true, true, 2)?;
    test_single_helper(12, true, true, true, 2)?;
    test_single_helper(13, true, true, true, 2)?;
    test_single_helper(14, true, true, true, 2)?;
    test_single_helper(15, true, true, true, 2)?;
    Ok(())
}

// ---------------- k = 3 ----------------

#[test]
fn test_dense_k3() -> Result<(), PCSError> {
    test_single_helper(3, false, false, false, 3)?;
    test_single_helper(4, false, false, false, 3)?;
    test_single_helper(5, false, false, false, 3)?;
    test_single_helper(6, false, false, false, 3)?;
    test_single_helper(7, false, false, false, 3)?;
    test_single_helper(8, false, false, false, 3)?;
    test_single_helper(9, false, false, false, 3)?;
    test_single_helper(10, false, false, false, 3)?;
    test_single_helper(11, false, false, false, 3)?;
    test_single_helper(12, false, false, false, 3)?;
    test_single_helper(13, false, false, false, 3)?;
    test_single_helper(14, false, false, false, 3)?;
    test_single_helper(15, false, false, false, 3)?;
    Ok(())
}

#[test]
fn test_dense_zk_k3() -> Result<(), PCSError> {
    test_single_helper(3, true, false, false, 3)?;
    test_single_helper(4, true, false, false, 3)?;
    test_single_helper(5, true, false, false, 3)?;
    test_single_helper(6, true, false, false, 3)?;
    test_single_helper(7, true, false, false, 3)?;
    test_single_helper(8, true, false, false, 3)?;
    test_single_helper(9, true, false, false, 3)?;
    test_single_helper(10, true, false, false, 3)?;
    test_single_helper(11, true, false, false, 3)?;
    test_single_helper(12, true, false, false, 3)?;
    test_single_helper(13, true, false, false, 3)?;
    test_single_helper(14, true, false, false, 3)?;
    test_single_helper(15, true, false, false, 3)?;
    Ok(())
}

#[test]
fn test_dense_boolean_k3() -> Result<(), PCSError> {
    test_single_helper(3, false, false, true, 3)?;
    test_single_helper(4, false, false, true, 3)?;
    test_single_helper(5, false, false, true, 3)?;
    test_single_helper(6, false, false, true, 3)?;
    test_single_helper(7, false, false, true, 3)?;
    test_single_helper(8, false, false, true, 3)?;
    test_single_helper(9, false, false, true, 3)?;
    test_single_helper(10, false, false, true, 3)?;
    test_single_helper(11, false, false, true, 3)?;
    test_single_helper(12, false, false, true, 3)?;
    test_single_helper(13, false, false, true, 3)?;
    test_single_helper(14, false, false, true, 3)?;
    test_single_helper(15, false, false, true, 3)?;
    Ok(())
}

#[test]
fn test_dense_boolean_zk_k3() -> Result<(), PCSError> {
    test_single_helper(3, true, false, true, 3)?;
    test_single_helper(4, true, false, true, 3)?;
    test_single_helper(5, true, false, true, 3)?;
    test_single_helper(6, true, false, true, 3)?;
    test_single_helper(7, true, false, true, 3)?;
    test_single_helper(8, true, false, true, 3)?;
    test_single_helper(9, true, false, true, 3)?;
    test_single_helper(10, true, false, true, 3)?;
    test_single_helper(11, true, false, true, 3)?;
    test_single_helper(12, true, false, true, 3)?;
    test_single_helper(13, true, false, true, 3)?;
    test_single_helper(14, true, false, true, 3)?;
    test_single_helper(15, true, false, true, 3)?;
    Ok(())
}

#[test]
fn test_sparse_k3() -> Result<(), PCSError> {
    test_single_helper(3, false, true, false, 3)?;
    test_single_helper(4, false, true, false, 3)?;
    test_single_helper(5, false, true, false, 3)?;
    test_single_helper(6, false, true, false, 3)?;
    test_single_helper(7, false, true, false, 3)?;
    test_single_helper(8, false, true, false, 3)?;
    test_single_helper(9, false, true, false, 3)?;
    test_single_helper(10, false, true, false, 3)?;
    test_single_helper(11, false, true, false, 3)?;
    test_single_helper(12, false, true, false, 3)?;
    test_single_helper(13, false, true, false, 3)?;
    test_single_helper(14, false, true, false, 3)?;
    test_single_helper(15, false, true, false, 3)?;
    Ok(())
}

#[test]
fn test_sparse_zk_k3() -> Result<(), PCSError> {
    test_single_helper(3, true, true, false, 3)?;
    test_single_helper(4, true, true, false, 3)?;
    test_single_helper(5, true, true, false, 3)?;
    test_single_helper(6, true, true, false, 3)?;
    test_single_helper(7, true, true, false, 3)?;
    test_single_helper(8, true, true, false, 3)?;
    test_single_helper(9, true, true, false, 3)?;
    test_single_helper(10, true, true, false, 3)?;
    test_single_helper(11, true, true, false, 3)?;
    test_single_helper(12, true, true, false, 3)?;
    test_single_helper(13, true, true, false, 3)?;
    test_single_helper(14, true, true, false, 3)?;
    test_single_helper(15, true, true, false, 3)?;
    Ok(())
}

#[test]
fn test_sparse_boolean_k3() -> Result<(), PCSError> {
    test_single_helper(3, false, true, true, 3)?;
    test_single_helper(4, false, true, true, 3)?;
    test_single_helper(5, false, true, true, 3)?;
    test_single_helper(6, false, true, true, 3)?;
    test_single_helper(7, false, true, true, 3)?;
    test_single_helper(8, false, true, true, 3)?;
    test_single_helper(9, false, true, true, 3)?;
    test_single_helper(10, false, true, true, 3)?;
    test_single_helper(11, false, true, true, 3)?;
    test_single_helper(12, false, true, true, 3)?;
    test_single_helper(13, false, true, true, 3)?;
    test_single_helper(14, false, true, true, 3)?;
    test_single_helper(15, false, true, true, 3)?;
    Ok(())
}

#[test]
fn test_sparse_boolean_zk_k3() -> Result<(), PCSError> {
    test_single_helper(3, true, true, true, 3)?;
    test_single_helper(4, true, true, true, 3)?;
    test_single_helper(5, true, true, true, 3)?;
    test_single_helper(6, true, true, true, 3)?;
    test_single_helper(7, true, true, true, 3)?;
    test_single_helper(8, true, true, true, 3)?;
    test_single_helper(9, true, true, true, 3)?;
    test_single_helper(10, true, true, true, 3)?;
    test_single_helper(11, true, true, true, 3)?;
    test_single_helper(12, true, true, true, 3)?;
    test_single_helper(13, true, true, true, 3)?;
    test_single_helper(14, true, true, true, 3)?;
    test_single_helper(15, true, true, true, 3)?;
    Ok(())
}

// ---------------- k = 4 ----------------

#[test]
fn test_dense_k4() -> Result<(), PCSError> {
    test_single_helper(4, false, false, false, 4)?;
    test_single_helper(5, false, false, false, 4)?;
    test_single_helper(6, false, false, false, 4)?;
    test_single_helper(7, false, false, false, 4)?;
    test_single_helper(8, false, false, false, 4)?;
    test_single_helper(9, false, false, false, 4)?;
    test_single_helper(10, false, false, false, 4)?;
    test_single_helper(11, false, false, false, 4)?;
    test_single_helper(12, false, false, false, 4)?;
    test_single_helper(13, false, false, false, 4)?;
    test_single_helper(14, false, false, false, 4)?;
    test_single_helper(15, false, false, false, 4)?;
    Ok(())
}

#[test]
fn test_dense_zk_k4() -> Result<(), PCSError> {
    test_single_helper(4, true, false, false, 4)?;
    test_single_helper(5, true, false, false, 4)?;
    test_single_helper(6, true, false, false, 4)?;
    test_single_helper(7, true, false, false, 4)?;
    test_single_helper(8, true, false, false, 4)?;
    test_single_helper(9, true, false, false, 4)?;
    test_single_helper(10, true, false, false, 4)?;
    test_single_helper(11, true, false, false, 4)?;
    test_single_helper(12, true, false, false, 4)?;
    test_single_helper(13, true, false, false, 4)?;
    test_single_helper(14, true, false, false, 4)?;
    test_single_helper(15, true, false, false, 4)?;
    Ok(())
}

#[test]
fn test_dense_boolean_k4() -> Result<(), PCSError> {
    test_single_helper(4, false, false, true, 4)?;
    test_single_helper(5, false, false, true, 4)?;
    test_single_helper(6, false, false, true, 4)?;
    test_single_helper(7, false, false, true, 4)?;
    test_single_helper(8, false, false, true, 4)?;
    test_single_helper(9, false, false, true, 4)?;
    test_single_helper(10, false, false, true, 4)?;
    test_single_helper(11, false, false, true, 4)?;
    test_single_helper(12, false, false, true, 4)?;
    test_single_helper(13, false, false, true, 4)?;
    test_single_helper(14, false, false, true, 4)?;
    test_single_helper(15, false, false, true, 4)?;
    Ok(())
}

#[test]
fn test_dense_boolean_zk_k4() -> Result<(), PCSError> {
    test_single_helper(4, true, false, true, 4)?;
    test_single_helper(5, true, false, true, 4)?;
    test_single_helper(6, true, false, true, 4)?;
    test_single_helper(7, true, false, true, 4)?;
    test_single_helper(8, true, false, true, 4)?;
    test_single_helper(9, true, false, true, 4)?;
    test_single_helper(10, true, false, true, 4)?;
    test_single_helper(11, true, false, true, 4)?;
    test_single_helper(12, true, false, true, 4)?;
    test_single_helper(13, true, false, true, 4)?;
    test_single_helper(14, true, false, true, 4)?;
    test_single_helper(15, true, false, true, 4)?;
    Ok(())
}

#[test]
fn test_sparse_k4() -> Result<(), PCSError> {
    test_single_helper(4, false, true, false, 4)?;
    test_single_helper(5, false, true, false, 4)?;
    test_single_helper(6, false, true, false, 4)?;
    test_single_helper(7, false, true, false, 4)?;
    test_single_helper(8, false, true, false, 4)?;
    test_single_helper(9, false, true, false, 4)?;
    test_single_helper(10, false, true, false, 4)?;
    test_single_helper(11, false, true, false, 4)?;
    test_single_helper(12, false, true, false, 4)?;
    test_single_helper(13, false, true, false, 4)?;
    test_single_helper(14, false, true, false, 4)?;
    test_single_helper(15, false, true, false, 4)?;
    Ok(())
}

#[test]
fn test_sparse_zk_k4() -> Result<(), PCSError> {
    test_single_helper(4, true, true, false, 4)?;
    test_single_helper(5, true, true, false, 4)?;
    test_single_helper(6, true, true, false, 4)?;
    test_single_helper(7, true, true, false, 4)?;
    test_single_helper(8, true, true, false, 4)?;
    test_single_helper(9, true, true, false, 4)?;
    test_single_helper(10, true, true, false, 4)?;
    test_single_helper(11, true, true, false, 4)?;
    test_single_helper(12, true, true, false, 4)?;
    test_single_helper(13, true, true, false, 4)?;
    test_single_helper(14, true, true, false, 4)?;
    test_single_helper(15, true, true, false, 4)?;
    Ok(())
}

#[test]
fn test_sparse_boolean_k4() -> Result<(), PCSError> {
    test_single_helper(4, false, true, true, 4)?;
    test_single_helper(5, false, true, true, 4)?;
    test_single_helper(6, false, true, true, 4)?;
    test_single_helper(7, false, true, true, 4)?;
    test_single_helper(8, false, true, true, 4)?;
    test_single_helper(9, false, true, true, 4)?;
    test_single_helper(10, false, true, true, 4)?;
    test_single_helper(11, false, true, true, 4)?;
    test_single_helper(12, false, true, true, 4)?;
    test_single_helper(13, false, true, true, 4)?;
    test_single_helper(14, false, true, true, 4)?;
    test_single_helper(15, false, true, true, 4)?;
    Ok(())
}

#[test]
fn test_sparse_boolean_zk_k4() -> Result<(), PCSError> {
    test_single_helper(4, true, true, true, 4)?;
    test_single_helper(5, true, true, true, 4)?;
    test_single_helper(6, true, true, true, 4)?;
    test_single_helper(7, true, true, true, 4)?;
    test_single_helper(8, true, true, true, 4)?;
    test_single_helper(9, true, true, true, 4)?;
    test_single_helper(10, true, true, true, 4)?;
    test_single_helper(11, true, true, true, 4)?;
    test_single_helper(12, true, true, true, 4)?;
    test_single_helper(13, true, true, true, 4)?;
    test_single_helper(14, true, true, true, 4)?;
    test_single_helper(15, true, true, true, 4)?;
    Ok(())
}

// ---------------- k = 5 ----------------

#[test]
fn test_dense_k5() -> Result<(), PCSError> {
    // Keep your original k-argument pattern
    test_single_helper(5, false, false, false, 4)?;
    test_single_helper(6, false, false, false, 4)?;
    test_single_helper(7, false, false, false, 4)?;
    test_single_helper(8, false, false, false, 4)?;
    test_single_helper(9, false, false, false, 5)?;
    test_single_helper(10, false, false, false, 5)?;
    test_single_helper(11, false, false, false, 5)?;
    test_single_helper(12, false, false, false, 5)?;
    test_single_helper(13, false, false, false, 5)?;
    test_single_helper(14, false, false, false, 5)?;
    test_single_helper(15, false, false, false, 5)?;
    Ok(())
}

#[test]
fn test_dense_zk_k5() -> Result<(), PCSError> {
    test_single_helper(5, true, false, false, 4)?;
    test_single_helper(6, true, false, false, 4)?;
    test_single_helper(7, true, false, false, 4)?;
    test_single_helper(8, true, false, false, 4)?;
    test_single_helper(9, true, false, false, 5)?;
    test_single_helper(10, true, false, false, 5)?;
    test_single_helper(11, true, false, false, 5)?;
    test_single_helper(12, true, false, false, 5)?;
    test_single_helper(13, true, false, false, 5)?;
    test_single_helper(14, true, false, false, 5)?;
    test_single_helper(15, true, false, false, 5)?;
    Ok(())
}

#[test]
fn test_dense_boolean_k5() -> Result<(), PCSError> {
    test_single_helper(5, false, false, true, 4)?;
    test_single_helper(6, false, false, true, 4)?;
    test_single_helper(7, false, false, true, 4)?;
    test_single_helper(8, false, false, true, 4)?;
    test_single_helper(9, false, false, true, 5)?;
    test_single_helper(10, false, false, true, 5)?;
    test_single_helper(11, false, false, true, 5)?;
    test_single_helper(12, false, false, true, 5)?;
    test_single_helper(13, false, false, true, 5)?;
    test_single_helper(14, false, false, true, 5)?;
    test_single_helper(15, false, false, true, 5)?;
    Ok(())
}

#[test]
fn test_dense_boolean_zk_k5() -> Result<(), PCSError> {
    test_single_helper(5, true, false, true, 4)?;
    test_single_helper(6, true, false, true, 4)?;
    test_single_helper(7, true, false, true, 4)?;
    test_single_helper(8, true, false, true, 4)?;
    test_single_helper(9, true, false, true, 5)?;
    test_single_helper(10, true, false, true, 5)?;
    test_single_helper(11, true, false, true, 5)?;
    test_single_helper(12, true, false, true, 5)?;
    test_single_helper(13, true, false, true, 5)?;
    test_single_helper(14, true, false, true, 5)?;
    test_single_helper(15, true, false, true, 5)?;
    Ok(())
}

#[test]
fn test_sparse_k5() -> Result<(), PCSError> {
    test_single_helper(5, false, true, false, 4)?;
    test_single_helper(6, false, true, false, 4)?;
    test_single_helper(7, false, true, false, 4)?;
    test_single_helper(8, false, true, false, 4)?;
    test_single_helper(9, false, true, false, 5)?;
    test_single_helper(10, false, true, false, 5)?;
    test_single_helper(11, false, true, false, 5)?;
    test_single_helper(12, false, true, false, 5)?;
    test_single_helper(13, false, true, false, 5)?;
    test_single_helper(14, false, true, false, 5)?;
    test_single_helper(15, false, true, false, 5)?;
    Ok(())
}

#[test]
fn test_sparse_zk_k5() -> Result<(), PCSError> {
    test_single_helper(5, true, true, false, 4)?;
    test_single_helper(6, true, true, false, 4)?;
    test_single_helper(7, true, true, false, 4)?;
    test_single_helper(8, true, true, false, 4)?;
    test_single_helper(9, true, true, false, 5)?;
    test_single_helper(10, true, true, false, 5)?;
    test_single_helper(11, true, true, false, 5)?;
    test_single_helper(12, true, true, false, 5)?;
    test_single_helper(13, true, true, false, 5)?;
    test_single_helper(14, true, true, false, 5)?;
    test_single_helper(15, true, true, false, 5)?;
    Ok(())
}

#[test]
fn test_sparse_boolean_k5() -> Result<(), PCSError> {
    test_single_helper(5, false, true, true, 4)?;
    test_single_helper(6, false, true, true, 4)?;
    test_single_helper(7, false, true, true, 4)?;
    test_single_helper(8, false, true, true, 4)?;
    test_single_helper(9, false, true, true, 5)?;
    test_single_helper(10, false, true, true, 5)?;
    test_single_helper(11, false, true, true, 5)?;
    test_single_helper(12, false, true, true, 5)?;
    test_single_helper(13, false, true, true, 5)?;
    test_single_helper(14, false, true, true, 5)?;
    test_single_helper(15, false, true, true, 5)?;
    Ok(())
}

#[test]
fn test_sparse_boolean_zk_k5() -> Result<(), PCSError> {
    test_single_helper(5, true, true, true, 4)?;
    test_single_helper(6, true, true, true, 4)?;
    test_single_helper(7, true, true, true, 4)?;
    test_single_helper(8, true, true, true, 4)?;
    test_single_helper(9, true, true, true, 5)?;
    test_single_helper(10, true, true, true, 5)?;
    test_single_helper(11, true, true, true, 5)?;
    test_single_helper(12, true, true, true, 5)?;
    test_single_helper(13, true, true, true, 5)?;
    test_single_helper(14, true, true, true, 5)?;
    test_single_helper(15, true, true, true, 5)?;
    Ok(())
}

/// Masking-server protocol: `generate_masking_package` produces an
/// opening-point-agnostic auxiliary, and `open_zk_with_package`
/// consumes it to produce a verifying hiding opening. End result
/// must verify against the same SRS as a vanilla `open_zk` would.
#[test]
fn masking_package_open_zk_round_trip() -> Result<(), PCSError> {
    let nv = 8;
    let k = 2;
    let mut rng = test_rng();
    let poly =
        DenseOrSparseMLE::Sparse(SparseMultilinearExtension::<Fr>::rand(nv, &mut rng));
    let params = KZHK::<E>::gen_srs_for_testing(KZHKConfig::new(k, true), &mut rng, nv)?;
    let (ck, vk) = KZHK::trim(params, None, Some(nv))?;
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

    let (com, mut state) = KZHK::<E>::commit(&ck, &poly)?;
    KZHK::<E>::update_state(&ck, &poly, &com, &mut state)?;

    // Producer side (masking server): builds the package in isolation,
    // without ever seeing the polynomial, commitment, or point.
    let package =
        <KZHK<E> as PolynomialCommitmentScheme<E>>::generate_masking_package(&ck, nv)?;

    // Consumer side (shard): opens the polynomial with the package.
    let mut prover_transcript = IOPTranscript::new(b"test_masking_pkg");
    let (proof, value) = <KZHK<E> as PolynomialCommitmentScheme<E>>::open_zk_with_package(
        &ck,
        &com,
        poly.as_ref(),
        &point,
        &state,
        &mut prover_transcript,
        &package,
    )?;

    // Verifier-side: the proof is a valid hiding opening of `f` at `point`.
    let mut verif_transcript = IOPTranscript::new(b"test_masking_pkg");
    assert!(KZHK::<E>::verify(
        &vk,
        &com,
        &point,
        &value,
        &proof,
        &mut verif_transcript,
    )?);
    Ok(())
}

/// `remask_with_package` upgrades a precomputed non-ZK opening into
/// a hiding one. The resulting proof must verify like any other
/// hiding opening; this is the primitive the history-lookup path
/// would use to upgrade DB-stored plain openings on demand.
#[test]
fn masking_package_remask_non_zk_proof() -> Result<(), PCSError> {
    let nv = 8;
    let k = 2;
    let mut rng = test_rng();
    let poly =
        DenseOrSparseMLE::Sparse(SparseMultilinearExtension::<Fr>::rand(nv, &mut rng));
    let params = KZHK::<E>::gen_srs_for_testing(KZHKConfig::new(k, true), &mut rng, nv)?;
    let (ck, vk) = KZHK::trim(params, None, Some(nv))?;
    let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();

    let (com, mut state) = KZHK::<E>::commit(&ck, &poly)?;
    KZHK::<E>::update_state(&ck, &poly, &com, &mut state)?;

    // Produce a plain non-ZK opening of `f` at `point`. The history-
    // lookup remask path stores exactly this shape at publish time.
    let (non_zk_proof, value) = <KZHK<E> as PolynomialCommitmentScheme<E>>::open_non_zk(
        &ck,
        &com,
        poly.as_ref(),
        &point,
        &state,
    )?;
    let tau_f =
        <KZHK<E> as PolynomialCommitmentScheme<E>>::get_hiding_scalar(&state);

    // Remask using a fresh package. The verifier accepts the result
    // as a hiding opening of `f` at `point` against `com`.
    let package =
        <KZHK<E> as PolynomialCommitmentScheme<E>>::generate_masking_package(&ck, nv)?;
    let mut prover_transcript = IOPTranscript::new(b"test_remask_pkg");
    let proof = <KZHK<E> as PolynomialCommitmentScheme<E>>::remask_with_package(
        &ck,
        &com,
        &point,
        &value,
        non_zk_proof,
        &tau_f,
        &mut prover_transcript,
        &package,
    )?;

    let mut verif_transcript = IOPTranscript::new(b"test_remask_pkg");
    assert!(KZHK::<E>::verify(
        &vk,
        &com,
        &point,
        &value,
        &proof,
        &mut verif_transcript,
    )?);
    Ok(())
}

