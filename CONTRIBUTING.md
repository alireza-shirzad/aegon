# Contributing

This is the reference implementation for an academic paper (see `README.md`
for the citation). Contributions are welcome, especially bug reports against
the cryptographic paths.

There is **no CLA**. By contributing you agree that your contributions are
licensed under the MIT License (`LICENSE`), the same terms as the rest of the
repository.

## Before you start

Read `SECURITY.md`. This is unaudited research code with a non-ceremonial
trusted setup; it is not production key transparency, and patches that
present it as such will not be merged.

## Pull requests

1. Fork the repo and branch from `main`.
2. Add tests for new code, and update docs when you change an API.
3. Make sure the checks CI runs are green locally:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets
   cargo clippy --workspace --all-targets --features ivc_audit
   cargo test -p akd
   cargo test -p akd --features ivc_audit
   cargo test -p akd_core
   ```

   Test the two crates in separate invocations — see "Test status" in the
   README for why. `Prerequisites` there covers `protoc` and `libclang`.

4. Do not pass `-D warnings` to clippy. The lint policy lives in
   `[workspace.lints]` in the root `Cargo.toml`, and command-line flags
   override it. The allow-list there is a backlog, not a style preference;
   removing an entry and fixing the fallout is a welcome contribution.

## Known gaps

Several tests are `#[ignore]`d with their reason in the attribute, and the
upstream SEEMless/Merkle suites sit behind the off-by-default
`upstream_tests` feature. The README tables both. Fixes for any of them are
useful contributions — start with the value-history round-trip against the
Rocks backend, which is an undiagnosed failure rather than a stale test.

There are roughly 250 `missing_docs` sites on public API. CI reports the
count without failing on it.

## Issues

Use GitHub issues. For anything that looks like a vulnerability, see
`SECURITY.md` — but note that this software is not deployed anywhere, so
there is no embargo process.

## Relationship to upstream AKD

This repository is a fork of [facebook/akd](https://github.com/facebook/akd)
with the append-only-tree backend replaced. Bugs that also affect upstream
should go to Meta through their process rather than here; `NOTICE` describes
the split.
