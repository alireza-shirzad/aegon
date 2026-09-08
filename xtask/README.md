This package supports local code-coverage reporting.

Coverage badges are not published for this repository. The upstream
`facebook/akd` badges that previously appeared here pointed at Meta's
Codecov project and did not reflect this fork's coverage, so they have
been removed rather than left to mislead.

## Viewing code coverage locally

Do this once to set it up:
```
rustup component add llvm-tools-preview
cargo install grcov
```

Subsequently, run:
```
cargo xtask coverage --dev
```
