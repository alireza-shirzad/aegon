// Copyright (c) The Aegon Authors.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use super::*;
use ark_bn254::Fr;
use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const TEST_SEED: [u8; 32] = [7u8; 32];
const CHILD_TEST_ENV: &str = "AEGON_VRF_KEY_TEST_CHILD";
const CHILD_CASE_ENV: &str = "AEGON_VRF_KEY_TEST_CASE";

struct KeyDir(PathBuf);

impl KeyDir {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "aegon-vrf-key-tests-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create isolated key directory");
        Self(path)
    }

    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, bytes).expect("write test key file");
        path
    }
}

impl Drop for KeyDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(super) fn in_test_subprocess(test_name: &str) -> bool {
    std::env::var(CHILD_TEST_ENV).as_deref() == Ok(test_name)
}

pub(super) fn isolated_test_command(test_name: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().expect("test executable"));
    command
        .args(["--exact", test_name, "--nocapture"])
        .env_clear()
        .env(CHILD_TEST_ENV, test_name);
    command
}

pub(super) fn assert_test_success(command: &mut Command) {
    let output = command
        .output()
        .expect("run isolated key-source regression");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed;"),
        "isolated regression failed or did not run: {}\n{stdout}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn missing_configuration_is_rejected() {
    assert!(matches!(
        load_seed(None, None),
        Err(VrfKeyError::MissingConfiguration)
    ));
}

#[test]
fn malformed_seed_hex_and_lengths_are_rejected() {
    for seed in [
        String::new(),
        "gg".repeat(32),
        "0".repeat(63),
        "00".repeat(31),
        "00".repeat(33),
    ] {
        assert!(
            matches!(load_seed(Some(&seed), None), Err(VrfKeyError::InvalidSeed)),
            "invalid seed must not be accepted: {seed:?}"
        );
    }
}

#[test]
fn missing_key_file_is_not_created() {
    let dir = KeyDir::new();
    let path = dir.0.join("missing-parent/key");
    assert!(matches!(
        load_seed(None, Some(&path)),
        Err(VrfKeyError::KeyFile(ref error))
            if error.kind() == std::io::ErrorKind::NotFound
    ));
    assert!(!path.exists(), "loading must not create a missing key file");
    assert!(
        !path.parent().unwrap().exists(),
        "loading must not create a missing key directory"
    );
}

#[test]
fn key_file_must_be_exactly_32_bytes() {
    let dir = KeyDir::new();
    for len in [0, 1, 31, 33, 4096] {
        let path = dir.file(&format!("key-{len}"), &vec![7u8; len]);
        assert!(
            matches!(
                load_seed(None, Some(&path)),
                Err(VrfKeyError::InvalidKeyFile)
            ),
            "key file of length {len} must be rejected"
        );
        assert_eq!(fs::metadata(path).unwrap().len(), len as u64);
    }
}

#[test]
fn non_file_key_path_is_rejected() {
    let dir = KeyDir::new();
    assert!(matches!(
        load_seed(None, Some(&dir.0)),
        Err(VrfKeyError::InvalidKeyFile)
    ));
}

#[test]
fn benchmark_seed_is_rejected_from_both_sources() {
    let dir = KeyDir::new();
    let path = dir.file("benchmark-key", &BENCH_VRF_SEED);
    let seed_hex = hex::encode(BENCH_VRF_SEED);
    assert!(matches!(
        load_seed(Some(&seed_hex), None),
        Err(VrfKeyError::BenchmarkSeed)
    ));
    assert!(matches!(
        load_seed(None, Some(&path)),
        Err(VrfKeyError::BenchmarkSeed)
    ));
}

#[test]
fn valid_hex_and_file_yield_the_same_key() {
    let dir = KeyDir::new();
    let path = dir.file("valid-key", &TEST_SEED);
    let seed_hex = hex::encode(TEST_SEED);
    let from_hex = load_seed(Some(&seed_hex), None).unwrap();
    let from_file = load_seed(None, Some(&path)).unwrap();
    assert_eq!(from_hex, TEST_SEED);
    assert_eq!(from_file, TEST_SEED);
    let hex_prover = VrfProver::from_seed(&from_hex);
    let file_prover = VrfProver::from_seed(&from_file);
    assert_eq!(
        hex_prover.public_key().as_bytes(),
        file_prover.public_key().as_bytes()
    );
    assert_eq!(
        hex_prover.evaluate_h_bits(1, b"key-source", 27),
        file_prover.evaluate_h_bits(1, b"key-source", 27)
    );
}

#[test]
fn seed_precedence_never_downgrades_to_a_file() {
    let dir = KeyDir::new();
    let path = dir.file("other-key", &[8u8; 32]);
    let seed_hex = hex::encode(TEST_SEED);
    assert_eq!(load_seed(Some(&seed_hex), Some(&path)).unwrap(), TEST_SEED);
    let missing = dir.0.join("absent");
    assert_eq!(
        load_seed(Some(&seed_hex), Some(&missing)).unwrap(),
        TEST_SEED
    );
    assert!(!missing.exists());
    for seed in ["invalid", ""] {
        assert!(matches!(
            load_seed(Some(seed), Some(&path)),
            Err(VrfKeyError::InvalidSeed)
        ));
    }
    let bench_hex = hex::encode(BENCH_VRF_SEED);
    assert!(matches!(
        load_seed(Some(&bench_hex), Some(&path)),
        Err(VrfKeyError::BenchmarkSeed)
    ));
    let bench_path = dir.file("benchmark-key", &BENCH_VRF_SEED);
    assert_eq!(
        load_seed(Some(&seed_hex), Some(&bench_path)).unwrap(),
        TEST_SEED
    );
}

fn assert_static_hash_rejects_configuration() {
    assert!(std::panic::catch_unwind(EcVrfHash::public_key).is_err());
    assert!(std::panic::catch_unwind(|| {
        <EcVrfHash as HashSuite<Fr>>::h_bits(0, b"fail-closed", 27)
    })
    .is_err());
    assert!(std::panic::catch_unwind(|| {
        <EcVrfHash as HashSuite<Fr>>::h_shard(0, b"fail-closed", 2)
    })
    .is_err());
    assert!(std::panic::catch_unwind(|| {
        <EcVrfHash as HashSuite<Fr>>::h_slot(0, b"fail-closed", 27)
    })
    .is_err());
    assert!(std::panic::catch_unwind(|| EcVrfHash::prove_h_bits(0, b"fail-closed", 27)).is_err());
    assert!(std::panic::catch_unwind(|| EcVrfHash::prove_h_shard(0, b"fail-closed", 2)).is_err());
    assert!(std::panic::catch_unwind(|| EcVrfHash::prove_h_slot(0, b"fail-closed", 27)).is_err());
}

#[test]
fn environment_loader_and_static_hash_fail_closed() {
    const TEST_NAME: &str =
        "hash::key_source_tests::environment_loader_and_static_hash_fail_closed";
    if in_test_subprocess(TEST_NAME) {
        let case = std::env::var(CHILD_CASE_ENV).expect("synthetic test case");
        if case.starts_with("valid-") {
            let prover = VrfProver::from_env().expect("explicit valid key loads");
            let expected = VrfProver::from_seed(&TEST_SEED);
            assert_eq!(
                prover.public_key().as_bytes(),
                expected.public_key().as_bytes()
            );
            assert_eq!(
                EcVrfHash::public_key().as_bytes(),
                prover.public_key().as_bytes()
            );
            assert_eq!(
                EcVrfHash::prove_h_shard(1, b"key-source", 2),
                prover.prove_h_shard(1, b"key-source", 2)
            );
            assert_eq!(
                EcVrfHash::prove_h_slot(1, b"key-source", 27),
                prover.prove_h_slot(1, b"key-source", 27)
            );
        } else {
            let error = VrfProver::from_env()
                .err()
                .expect("configuration must fail closed");
            match case.as_str() {
                "missing" => assert!(matches!(error, VrfKeyError::MissingConfiguration)),
                "invalid-seed" | "empty-seed" | "nonunicode-seed" | "no-downgrade" => {
                    assert!(matches!(error, VrfKeyError::InvalidSeed));
                }
                "empty-path" => {
                    assert!(matches!(error, VrfKeyError::InvalidKeyPath));
                }
                "missing-file" | "nonunicode-path" => assert!(matches!(
                    error,
                    VrfKeyError::KeyFile(ref error)
                        if error.kind() == std::io::ErrorKind::NotFound
                )),
                "short-file" | "long-file" | "directory-path" => {
                    assert!(matches!(error, VrfKeyError::InvalidKeyFile));
                }
                "benchmark-seed" | "benchmark-file" | "benchmark-no-downgrade" => {
                    assert!(matches!(error, VrfKeyError::BenchmarkSeed));
                }
                _ => panic!("unknown synthetic case: {case}"),
            }
            assert_static_hash_rejects_configuration();
        }
        return;
    }

    let dir = KeyDir::new();
    let valid = dir.file("valid-key", &TEST_SEED);
    let short = dir.file("short-key", &[7u8; 31]);
    let long = dir.file("long-key", &[7u8; 33]);
    let benchmark = dir.file("benchmark-key", &BENCH_VRF_SEED);
    let missing = dir.0.join("missing-parent/key");
    let seed_hex = hex::encode(TEST_SEED);
    let bench_hex = hex::encode(BENCH_VRF_SEED);
    let run = |case: &str, seed: Option<&OsStr>, path: Option<&OsStr>| {
        let mut command = isolated_test_command(TEST_NAME);
        command.env(CHILD_CASE_ENV, case);
        if let Some(seed) = seed {
            command.env(vrf_key_source::SEED_HEX_ENV, seed);
        }
        if let Some(path) = path {
            command.env(vrf_key_source::KEY_PATH_ENV, path);
        }
        assert_test_success(&mut command);
    };
    run("missing", None, None);
    run("invalid-seed", Some(OsStr::new("invalid")), None);
    run("empty-seed", Some(OsStr::new("")), Some(valid.as_os_str()));
    run("empty-path", None, Some(OsStr::new("")));
    run("missing-file", None, Some(missing.as_os_str()));
    assert!(!missing.exists(), "from_env must not create a key file");
    assert!(
        !missing.parent().unwrap().exists(),
        "from_env must not create a key directory"
    );
    run("short-file", None, Some(short.as_os_str()));
    run("long-file", None, Some(long.as_os_str()));
    run("directory-path", None, Some(dir.0.as_os_str()));
    run("benchmark-seed", Some(OsStr::new(&bench_hex)), None);
    run("benchmark-file", None, Some(benchmark.as_os_str()));
    run(
        "no-downgrade",
        Some(OsStr::new("invalid")),
        Some(valid.as_os_str()),
    );
    run(
        "benchmark-no-downgrade",
        Some(OsStr::new(&bench_hex)),
        Some(valid.as_os_str()),
    );
    run("valid-seed", Some(OsStr::new(&seed_hex)), None);
    run("valid-file", None, Some(valid.as_os_str()));
    run(
        "valid-seed-priority",
        Some(OsStr::new(&seed_hex)),
        Some(benchmark.as_os_str()),
    );
    #[cfg(unix)]
    {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        let nonunicode = OsString::from_vec(vec![0xff]);
        run(
            "nonunicode-seed",
            Some(&nonunicode),
            Some(valid.as_os_str()),
        );
        let nonunicode_path = dir.0.join(&nonunicode);
        run("nonunicode-path", None, Some(nonunicode_path.as_os_str()));
        fs::write(&nonunicode_path, TEST_SEED).expect("write key at nonunicode path");
        run(
            "valid-nonunicode-file",
            None,
            Some(nonunicode_path.as_os_str()),
        );
    }
}
