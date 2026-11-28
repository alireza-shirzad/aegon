// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! Tests for the `ark_transcript` variant: the [`ArkTranscript`] contract,
//! and KZH-k zk openings run over it.

use super::{ArkTranscript, FsTranscript};
use crate::pcs::kzhk::{structs::KZHKConfig, KZHK};
use crate::poly::DenseOrSparseMLE;
use crate::{PCSError, PolynomialCommitmentScheme};
use ark_bn254::{Bn254 as E, Fr};
use ark_poly::{DenseMultilinearExtension, MultilinearExtension, SparseMultilinearExtension};
use ark_std::{test_rng, UniformRand};

fn transcript(domain: &'static [u8], msg: &[u8]) -> ArkTranscript<Fr> {
    let mut t = ArkTranscript::new(domain);
    t.append_message(b"msg", msg).unwrap();
    t.append_field_element(b"field", &Fr::from(7u64)).unwrap();
    t
}

fn challenge(domain: &'static [u8], msg: &[u8]) -> Fr {
    transcript(domain, msg)
        .get_and_append_challenge(b"c")
        .unwrap()
}

#[test]
fn rejects_challenge_from_empty_transcript() {
    let mut t = ArkTranscript::<Fr>::new(b"test");
    assert!(t.get_and_append_challenge(b"c").is_err());
    assert!(t.get_and_append_challenge_vectors(b"c", 1).is_err());
    t.append_message(b"msg", b"x").unwrap();
    assert!(t.get_and_append_challenge(b"c").is_ok());
}

#[test]
fn challenge_is_deterministic_and_bound_to_inputs() {
    let base = challenge(b"domain", b"hello");
    assert_eq!(base, challenge(b"domain", b"hello"));
    assert_ne!(base, challenge(b"domain", b"hellp"));
    assert_ne!(base, challenge(b"other", b"hello"));
    let other_label = transcript(b"domain", b"hello")
        .get_and_append_challenge(b"d")
        .unwrap();
    assert_ne!(base, other_label);
}

#[test]
fn successive_challenges_differ() {
    let mut t = transcript(b"domain", b"hello");
    let a = t.get_and_append_challenge(b"c").unwrap();
    let b = t.get_and_append_challenge(b"c").unwrap();
    assert_ne!(a, b);
    let v = transcript(b"domain", b"hello")
        .get_and_append_challenge_vectors(b"c", 2)
        .unwrap();
    assert_eq!(v, vec![a, b]);
}

#[cfg(feature = "merlin")]
#[test]
fn differs_from_merlin() {
    let mut m = super::IOPTranscript::<Fr>::new(b"domain");
    m.append_message(b"msg", b"hello").unwrap();
    m.append_field_element(b"field", &Fr::from(7u64)).unwrap();
    assert_ne!(
        m.get_and_append_challenge(b"c").unwrap(),
        challenge(b"domain", b"hello")
    );
}

/// zk openings are where KZH-k draws a Fiat-Shamir challenge, so they are
/// what the transcript choice affects: plain, and through a masking package.
#[test]
fn zk_open_verify_with_ark_transcript() -> Result<(), PCSError> {
    type Ark = KZHK<E, ArkTranscript<Fr>>;
    let tr = || ArkTranscript::<Fr>::new(b"test_kzhk");

    let mut rng = test_rng();
    for (nv, k, sparse) in [(4, 2, false), (8, 2, true), (9, 3, false), (10, 5, true)] {
        let poly = if sparse {
            DenseOrSparseMLE::Sparse(SparseMultilinearExtension::<Fr>::rand(nv, &mut rng))
        } else {
            DenseOrSparseMLE::Dense(DenseMultilinearExtension::<Fr>::rand(nv, &mut rng))
        };
        let params = Ark::gen_srs_for_testing(KZHKConfig::new(k, true), &mut rng, nv)?;
        let (ck, vk) = Ark::trim(params, None, Some(nv))?;
        let point: Vec<Fr> = (0..nv).map(|_| Fr::rand(&mut rng)).collect();
        let (com, mut state) = Ark::commit(&ck, &poly)?;
        Ark::update_state(&ck, &poly, &com, &mut state)?;

        let (proof, value) = Ark::open(&ck, &com, poly.as_ref(), &point, &state, &mut tr())?;
        assert!(Ark::verify(&vk, &com, &point, &value, &proof, &mut tr())?);
        // Another domain label, or another transcript, derives a different
        // challenge, so the same proof must not verify.
        let mut other_label = ArkTranscript::<Fr>::new(b"other");
        assert!(!Ark::verify(
            &vk,
            &com,
            &point,
            &value,
            &proof,
            &mut other_label
        )?);
        #[cfg(feature = "merlin")]
        assert!(!KZHK::<E, super::IOPTranscript<Fr>>::verify(
            &vk,
            &com,
            &point,
            &value,
            &proof,
            &mut super::IOPTranscript::new(b"test_kzhk"),
        )?);

        let package = Ark::generate_masking_package(&ck, nv)?;
        let (proof, value) = Ark::open_zk_with_package(
            &ck,
            &com,
            poly.as_ref(),
            &point,
            &state,
            &mut tr(),
            &package,
        )?;
        assert!(Ark::verify(&vk, &com, &point, &value, &proof, &mut tr())?);
    }
    Ok(())
}
