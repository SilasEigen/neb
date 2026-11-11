# nebhash: integration notes for miner developers

nebhash is the proof of work of the NEB chain. One attempt is one integer training step of a small neural network that the chain owns: the nonce picks 16 samples of a dataset pinned by the chain, the attempt runs a forward and a backward pass in int32, and the gradient is hashed into the ticket. There is no floating point, so a CPU, a GPU kernel and a Python script give the same bytes.

This page is for developers who want to add nebhash to their miner. The normative texts are `spec/shares.md` (the attempt, the step, the chain rules) and `spec/node.md` section 7.1 (the work service). Everything in this repository is Apache-2.0, kernels included: you can copy the code into a closed-source miner.

## The algorithm in numbers

| | |
|---|---|
| network | 784 inputs, 128 hidden units, 10 outputs, weights in -8..7 (4 bits) |
| input of an attempt | 80-byte header, the model of the job (50,816 bytes; from block 8800, 13,000 for Kepler and 11,140 for the retina), the dataset the job names |
| key | `BLAKE3(header)`, nonce in bytes 76 to 79 |
| samples | 16 indices from the first 64 bytes of `ChaCha8(key)`: `idx = floor(u * N / 2^32)` |
| work per attempt | about 3.3 million small-integer multiply-adds (two 16x784x128 products, small layer-2 terms) |
| output | 8-lane digest of the clipped gradient, `ticket = BLAKE3(digest \|\| key)`, compared as a 256-bit little-endian integer |
| share target | block target times 32 |
| check of one attempt | 0.5 to 1 ms of one CPU core with the C reference |

The inner loops are int8-sized values in int32 sums (the largest value is 43,904), so dp4a and integer tensor-core instructions apply directly. Sections 5.1 to 5.4 of `spec/shares.md` give the whole attempt in about 30 lines.

## Data the miner holds

- **The dataset file**, `neb-datasets.bin` from the release (259,897,080 bytes, BLAKE3 `296a174edbf67339fa0af341f063154292bba224b3d2cacf5b18b31c70da80eb`). It holds 4 datasets, `NMDS || LE32(4) || dataset 0 || ... || dataset 3`: the two of the slots, `NMD1 || LE32(N)` then N records of 785 bytes (784 values in 0..7, then the label), 13,404 and 34,464 samples; then the two of the upgrade to nebhash v2, `NMD2 || LE32(arch) || LE32(N)` then N records of the architecture's inputs and a label, 13,404 Kepler signals for KEP1 (arch 1) and 51,696 scans of 64 x 64 for RET2 (arch 2). Load it once at start (`spec/shares.md` section 11.6, `spec/nebhash-v2.md` section 2).
- **The model of the current job.** The weights change with every block, so the node sends them before every job: a Model frame of `index (1) || weights_hash (32) || packed weights`. The first byte is the index of the dataset of the job in the file, and the dataset says which network to run. Block `h` works on slot `h mod 2`, so the dataset alternates from one job to the next. Upload the weights to the GPU on each job (50,816 bytes for v1, 13,000 for KEP1, 11,140 for RET2).

### nebhash v2, from block 8800

From block 8800 (slot 0) and 8801 (slot 1) the jobs use the convolutional networks of `spec/nebhash-v2.md`: KEP1 (two 1D branches on the Kepler views, 26,000 weights, about 180,800 multiply-adds per sample forward) and RET2 (three 2D convolutions on 64 x 64 scans, 22,280 weights, about 680,000). The key, the sample indices, the clip, the digest lanes and the ticket are the same as in v1. `pow/ref/nebhash2.c` is the reference for both, read from a table of layers; `pow/gpu/nebhash2_cuda.cu` is a CUDA kernel that matches it bit for bit; `vectors/nebhash2.txt` and `vectors/nebhash2_s1.txt` are the vectors.

## Talking to a node today

The node has a built-in work service: binary frames on TCP port 3441, `length (u32 LE) || type (u8) || payload`, where the length counts the type byte and the payload.

| Type | Name | Direction | Payload |
|---|---|---|---|
| 1 | Hello | miner to node | `version (u16, 1) \|\| name (varint length, UTF-8)` |
| 7 | Model | node to miner | `index (u8) \|\| weights_hash (32) \|\| packed weights (50,816, or 13,000 for KEP1 and 11,140 for RET2)` |
| 2 | Job | node to miner | target, share target, header and coinbase parts (`spec/node.md` 7.1) |
| 6 | Share | miner to node | `job id (u64) \|\| nonce (u32)` |
| 4 | Submit | miner to node | `job id (u64) \|\| extranonce (varint length, bytes) \|\| time (u32) \|\| nonce (u32)` |
| 5 | Result | node to miner | `accepted (u8) \|\| code (u8)` |

What matters for a miner:

- The node sends a Model frame and a Job in one write when the miner connects, when the best block changes, after a rejected Submit, and again when no job went out for 30 s. A Job of 2^32 nonces never runs dry: the next one has a later `time_start`.
- Mine on the **reference header** of the job: an extranonce of zero bytes, the time `time_start`, and roll the nonce. Building the header from the job is in `spec/node.md` 7.1 (coinbase, Merkle branch, `body_root = H("body", LE32(tx_count), h, shares_root)`).
- A nonce whose ticket meets the share target is a share: send it with a Share frame. The node checks it, keeps it and relays it, and it trains the model of the chain. Shares get no answer.
- A nonce whose ticket meets the block target is a solution: send it with Submit and read the Result (0 accepted, 1 bad submission, 2 bad header, 3 bad state).
- The node keeps the 64 newest jobs; a Share or Submit that names an older job is dropped.
- The service listens on 127.0.0.1 with no authentication, so the miner runs next to its node. `node/crates/neb-work/src/frames.rs` encodes and decodes every frame, and `node/crates/neb-cpuminer` is a complete client in Rust.

## Code to reuse

| Path | What it is |
|---|---|
| `pow/ref/shares.c`, `blake3_min.c`, `chacha8.c` | C reference of the attempt and the step; every engine is checked against it |
| `pow/gpu/shares_cuda.cu`, `scan_cuda.cu` | CUDA kernel K0: dp4a, the default kernel |
| `pow/gpu/shares_tc.cu`, `tc_common.cuh` | CUDA kernels K1 and K2: integer tensor cores, Ampere and later |
| `pow/cpu/shares_avx512.c` | AVX-512 engine for CPUs |
| `node/crates/neb-gpuminer` | Rust host of the GPU miner: work service, job handling, kernel fallback |
| `pow/oracle/shares_oracle.py` | Python oracle written from the spec alone |
| `docs/mining.md` | engines, kernels and their options |

## Checking an implementation

- `vectors/shares.txt` lists 24 model steps with, for every share, its digest, its ticket and its 16 sample indices (`spec/shares.md` section 10). A new engine that matches every line computes nebhash. `vectors/shares_s1.txt` is a variant that also exercises the clip and clamp rules of the step. `pow/tests/check_share_vectors.sh` rebuilds both files with the C reference.
- `neb-miner-gpu --selftest` and `neb-miner-cpu --synthetic --engine avx512 --selftest` compare the engines with the reference and exit with code 0 when they agree.
- The reference GPU miner checks every nonce it reports with the C reference before sending it, and checks the model of each job on a few nonces. With the tensor-core kernels it also audits one batch in 64 and falls back to K0 on any mismatch. A third-party miner should do the same: the node drops a wrong share without an answer.

## Pools and dev fee

Today a miner talks to a node, solo, and the job pays the lock that the node operator chose with `--miner-lock`. The pool protocol is specified in `spec/pool.md`: JSON lines in the style of stratum, the payout address in the login, and no change to the node or the consensus. A miner with a dev fee authorizes its own login in the same session and submits that part of its attempts under it. The reference pool, open source, is on the roadmap for Q4 2026.

A share on NEB is bound to the parent block, not to the header (`spec/shares.md` section 11.2): a miner can change job, extranonce or time at any moment without losing the shares it found, and the shares of every miner train the model, whoever finds the block.

## Links

- Code: https://github.com/SilasEigen/neb
- Specification: https://github.com/SilasEigen/neb/blob/main/spec/shares.md
- Pool protocol: https://github.com/SilasEigen/neb/blob/main/spec/pool.md
- Running a node and a miner: https://github.com/SilasEigen/neb/blob/main/docs/testnet.md
- Explorer: https://neuralenginebyte.org/#/explorer
- Discord: https://discord.gg/B6dTbScw8
