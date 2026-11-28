// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//! [`FsTranscript`] over `ark-transcript` (SHAKE128).

use super::{FsTranscript, TranscriptError};
use ark_ff::PrimeField;
use std::marker::PhantomData;

/// An `ark-transcript` transcript, producing challenges over `F`. Select it
/// with e.g. `KZHK<Bn254, ArkTranscript<Fr>>`.
#[derive(Clone)]
pub struct ArkTranscript<F: PrimeField> {
    transcript: ark_transcript::Transcript,
    is_empty: bool,
    phantom: PhantomData<F>,
}

impl<F: PrimeField> FsTranscript<F> for ArkTranscript<F> {
    fn new(label: &'static [u8]) -> Self {
        Self {
            transcript: ark_transcript::Transcript::new_labeled(label),
            is_empty: true,
            phantom: PhantomData,
        }
    }

    fn append_message(&mut self, label: &'static [u8], msg: &[u8]) -> Result<(), TranscriptError> {
        self.transcript.label(label);
        self.transcript.write_bytes(msg);
        self.is_empty = false;
        Ok(())
    }

    fn get_and_append_challenge(&mut self, label: &'static [u8]) -> Result<F, TranscriptError> {
        if self.is_empty {
            return Err(TranscriptError::InvalidTranscript(
                "transcript is empty".to_string(),
            ));
        }
        let challenge = self.transcript.challenge(label).read_reduce::<F>();
        self.append_serializable_element(label, &challenge)?;
        Ok(challenge)
    }
}
