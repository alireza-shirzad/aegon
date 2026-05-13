use thiserror::Error;

#[derive(Debug, Error)]
pub enum AegonError {
    #[error("label {0:?} is not present in the dictionary")]
    UnknownLabel(Vec<u8>),

    #[error("label {0:?} is already assigned")]
    DuplicateLabel(Vec<u8>),

    #[error("dictionary is full: open-addressing exhausted {capacity} slots without finding an empty index")]
    DictionaryFull { capacity: usize },

    #[error("epoch {0} is not retained in the server's history")]
    InvalidEpoch(u64),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("PCS error: {0:?}")]
    Pcs(akd_core::aegon_crypto::pcs::prelude::PCSError),

    #[error("transcript error: {0:?}")]
    Transcript(akd_core::aegon_crypto::transcript::TranscriptError),

    #[error("proof verification failed: {0}")]
    Verification(&'static str),

    #[error("database error: {0}")]
    Database(String),
}

impl From<akd_core::aegon_crypto::transcript::TranscriptError> for AegonError {
    fn from(e: akd_core::aegon_crypto::transcript::TranscriptError) -> Self {
        AegonError::Transcript(e)
    }
}

impl From<akd_core::aegon_crypto::pcs::prelude::PCSError> for AegonError {
    fn from(e: akd_core::aegon_crypto::pcs::prelude::PCSError) -> Self {
        AegonError::Pcs(e)
    }
}
