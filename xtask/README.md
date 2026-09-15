This package supports local code-coverage reporting. Coverage badges are
not published for this repository.

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
