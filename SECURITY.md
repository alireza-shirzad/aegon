# Security

## This is research code

Aegon is a research prototype accompanying an academic paper. It has **not**
been audited, and it is **not** suitable for production key transparency.
Concretely, before anyone relies on it:

* **The trusted setup is not ceremonial.** `aegon_srs_gen` calls
  `gen_srs_for_testing` and samples the KZH-k trapdoors locally from a
  caller-supplied seed (the deploy scripts pass `--seed 42`). Anyone who
  knows the seed knows the trapdoors and can forge openings. A real
  deployment needs a multi-party setup ceremony and a binary that loads its
  output; neither is implemented here.
* **The cryptographic implementation is unreviewed.** The KZH-k commitment
  scheme, the Sigma-protocol blinding-equality proof, the Fiat-Shamir
  derivations, and the IVC step circuit have had no external review.
* **Nothing here is constant-time.** No side-channel hardening has been
  attempted anywhere in the codebase.
* **The VRF key is a fixed constant unless you override it.** `hash.rs`
  falls back to the hard-coded `BENCH_VRF_SEED` when no seed is supplied
  through the environment.
* **Transport security is optional and off by default.** The shard and
  coordinator servers do support TLS (`ShardServerTlsConfig`,
  `CoordinatorServerTlsConfig`, and the matching client configs), but it is
  opt-in and the deploy scripts in `scripts/` do not wire it up:
  coordinator-to-shard traffic runs as plaintext HTTP/2, and the Redis
  backend runs without `requirepass`. There is no peer authentication or
  authorization anywhere — a server trusts whatever reaches it. The
  benchmark topology relies entirely on VPC firewall rules. See
  `scripts/README.md` for the specifics.
* **The audit transcript is fixed for the life of a chain.** The audit path's
  Fiat-Shamir derivations default to Poseidon; `--audit-fs sha256` selects the
  original SHA-256 ones. Servers and auditors must agree, and the choice is
  baked into every epoch the server publishes, so it cannot be changed on a
  live chain -- a mismatch rejects every epoch. Poseidon is the default
  because it is the only transcript a fast-forward auditor can recompute
  affordably in circuit, and a deployment that might ever want that has to
  start on it. The Merkle commitment, lookup, consistency and history paths
  are unaffected either way and stay on SHA-256.

## Reporting a vulnerability

Open a GitHub issue. Because this is not deployed software, there is no
embargo process and no security-release channel.

If you believe you have found a flaw in the *protocol* rather than the
implementation, that is a paper correction and we would very much like to
hear about it — please include which claim in the paper you believe fails.

## Upstream

Vulnerabilities in code inherited from
[facebook/akd](https://github.com/facebook/akd) that also affect upstream
should be reported to Meta through their process, not here.
