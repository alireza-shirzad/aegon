// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use aegon_crypto::pcs::kzhk::srs::KZHKUniversalParams;
use aegon_crypto::{PCSError, StructuredReferenceString};
use ark_bn254::Bn254;
use ark_std::rand::{rngs::StdRng, SeedableRng};

#[cfg(not(feature = "test-srs"))]
#[test]
fn test_srs_generation_is_disabled_without_feature() {
    use aegon_crypto::pcs::kzhk::{structs::KZHKConfig, KZHK};
    use aegon_crypto::PolynomialCommitmentScheme;
    use ark_std::rand::RngCore;

    let mut rng = StdRng::seed_from_u64(42);
    let mut expected_rng = rng.clone();
    let high = KZHK::<Bn254>::gen_srs_for_testing(KZHKConfig::new(2, false), &mut rng, 2);
    let low = KZHKUniversalParams::<Bn254>::gen_srs_for_testing(&mut rng, 2, false, 2);
    for result in [high, low] {
        match result {
            Err(PCSError::InvalidParameters(message)) => assert!(message.contains("test-srs")),
            other => panic!("generation must be disabled without test-srs: {other:?}"),
        }
    }
    assert_eq!(
        rng.next_u64(),
        expected_rng.next_u64(),
        "disabled generation must not consume randomness"
    );
}

#[cfg(feature = "test-srs")]
#[test]
fn test_srs_feature_enables_generation() -> Result<(), PCSError> {
    let mut rng = StdRng::seed_from_u64(42);
    let srs = KZHKUniversalParams::<Bn254>::gen_srs_for_testing(&mut rng, 2, false, 2)?;
    assert_eq!(srs.get_dimensions(), &[1, 1]);
    Ok(())
}
