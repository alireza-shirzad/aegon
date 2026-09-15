# A four-shard cluster on your machine

The [single-machine walkthrough](README.md) runs Aegon in one process. This
one runs it the way a deployment does: every role is a separate process
talking gRPC, and the dictionary is split across four shards.

| Process | Count | Port | Role |
| :--- | :--- | :--- | :--- |
| `aegon_masking_server` | 2 | 50061-50062 | Precomputes the blinding packages that hiding openings consume. |
| `aegon_shard_server` | 4 | 50051-50054 | Holds a quarter of the dictionary and proves against it. |
| `aegon_coordinator_server` | 1 | 50100 | Routes each request to the owning shard; the only process clients talk to. |
| `aegon_client` | 1 | — | Looks a user up and verifies the proofs itself. |

It runs in **private mode**, which is what the masking servers are for. In
private mode every opening is hiding, and each one consumes a blinding
package. Generating those is the expensive part, so it moves off the shards
onto servers you can scale separately: here, four shards share two of them.

Everything below runs on one machine, so it fits on a laptop. Nothing about
the wiring changes when the processes move to different hosts — only the
addresses do.

---

## Before you start

```bash
# Build the four binaries once.
cargo build --release -p aegon \
  --bin aegon_masking_server --bin aegon_shard_server \
  --bin aegon_coordinator_server --bin aegon_client
```

Every process must agree on four things, or setup fails:

| Flag | Value here | Why |
| :--- | :--- | :--- |
| `--shard-log-capacity` | `10` | log2 of the slots *per shard*: 1,024 each, 4,096 in total. |
| `--kzh-k` | `3` | Commitment-scheme parameter for that size. |
| `--setup-seed` | `42` | Test-mode setup: the shared seed is what makes every process derive the same one. |
| `--private` | set | Hiding commitments and openings. Leave it off everywhere and you don't need masking servers at all. |

The masking servers take `--num-vars` instead of `--shard-log-capacity`;
it's the same number.

Each command below blocks, so give each one its own terminal, or append `&`
to run it in the background.

---

## Step 1: the masking servers

Start these first — the shards connect to them at boot.

```bash
# Terminal 1
./target/release/aegon_masking_server \
  --bind 127.0.0.1:50061 --num-vars 10 --kzh-k 3 --setup-seed 42

# Terminal 2
./target/release/aegon_masking_server \
  --bind 127.0.0.1:50062 --num-vars 10 --kzh-k 3 --setup-seed 42
```

Each prints a warning that it is generating a test setup, then:

```text
aegon_masking_server listening on 127.0.0.1:50061 (num_vars=10, kzh_k=3, queue_size=256, producers=12)
```

`queue_size` is how many packages it keeps ready, and `producers` is how
many it builds in parallel (defaulting to your core count). Those two knobs
are how you match masking throughput to your lookup rate.

---

## Step 2: the four shards

Each shard gets **both** masking servers and round-robins its requests
across them.

```bash
# Terminals 3-6: one per shard, changing only the port.
./target/release/aegon_shard_server \
  --bind 127.0.0.1:50051 \
  --shard-log-capacity 10 --kzh-k 3 --setup-seed 42 --private \
  --masking-addr http://127.0.0.1:50061,http://127.0.0.1:50062
```

Repeat for ports `50052`, `50053` and `50054`. Each shard confirms the
masking wiring and then its own address:

```text
connecting to 2 masking servers (round-robin): ["http://127.0.0.1:50061", "http://127.0.0.1:50062"]
aegon_shard_server listening on 127.0.0.1:50051 (shard_log_capacity=10, kzh_k=3, private=true)
```

The shards are peers and don't know about each other. Routing is entirely
the coordinator's job.

---

## Step 3: the coordinator

The endpoint list is what makes this a four-shard dictionary; its length
must be a power of two. `--seed-batch-size` publishes test users at startup
so there is something to look up.

```bash
# Terminal 7
./target/release/aegon_coordinator_server \
  --listen 127.0.0.1:50100 \
  --endpoints http://127.0.0.1:50051,http://127.0.0.1:50052,http://127.0.0.1:50053,http://127.0.0.1:50054 \
  --shard-log-capacity 10 --kzh-k 3 --setup-seed 42 --private \
  --seed-batch-size 100
```

```text
coordinator: connecting to 4 shards (shard_log_capacity=10, kzh_k=3, log_n_shards=2)
coordinator: setup OK in 11.0 ms
coordinator: ECVRF prover attached (pubkey 028e2af4...); clients fetch via CurrentCommitment
coordinator: seeded 100 labels in 253.8 ms
coordinator: serving on 127.0.0.1:50100
```

Setup is the one moment the coordinator talks to every shard: it reads each
one's initial commitment, and checks the first shard's configured size
against its own. On a mismatch it refuses to start, rather than publish an
epoch nobody can verify.

The seeded users are `b100-s0-u0` through `b100-s0-u99`, with values `v-0`
through `v-99`. They are spread across all four shards.

---

## Step 4: look someone up

```bash
# Terminal 8. --log-n-shards 2 means 4 shards; the client needs it to
# recompute where a user should live.
./target/release/aegon_client \
  --coordinator http://127.0.0.1:50100 \
  --shard-log-capacity 10 --log-n-shards 2 --kzh-k 3 --setup-seed 42 --private \
  --label b100-s0-u0 --expected-value v-0
```

```text
client: connecting to http://127.0.0.1:50100 ...
client: current commitment OK
client: lookup_label OK in 7.8 ms → shard 3, slot len 10
client: lookup_value OK in 8.1 ms → eval matches H_F(value), proof verified
client: done.
```

`shard 3` is the coordinator telling the client where the user lives, and
the client checking that claim for itself. The client trusts nothing it is
told, so a value the server never published is rejected:

```bash
./target/release/aegon_client \
  --coordinator http://127.0.0.1:50100 \
  --shard-log-capacity 10 --log-n-shards 2 --kzh-k 3 --setup-seed 42 --private \
  --label b100-s0-u0 --expected-value not-the-value
```

```text
error: lookup_value_with_bytes: proof verification failed: value opening does not match H_F(value)
```

---

## Making it your own

- **More shards.** The endpoint count must be a power of two, and the client's
  `--log-n-shards` must be its log2. Total slots are
  `2^shard_log_capacity x n_shards`, and Aegon's default 4x headroom means
  four shards of 1,024 slots hold about 1,024 users comfortably.
- **More masking servers.** Add addresses to `--masking-addr`. Shards spread
  their requests across the list, so masking throughput scales with it.
- **Without private mode.** Drop `--private` everywhere and skip step 1
  entirely. Openings stop being hiding, and auditors can then see users'
  values.
- **Stopping.** Ctrl-C each process. State lives in memory, so the next start
  is a fresh dictionary.

`--setup-seed` is a test setup: anyone who knows the seed knows the
trapdoor. A real deployment generates one SRS with `aegon_srs_gen`, ships
the file, and passes `--srs-path` everywhere instead.

---

For spreading these processes across machines — TLS, Redis-backed storage,
systemd units, and sizing for billions of users — see
[`docs/deployment.md`](../../docs/deployment.md).
