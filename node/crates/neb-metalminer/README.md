# neb-metalminer — the NEB miner on an Apple GPU

`neb-miner-metal` mines the NEB testnet on the GPU of an Apple silicon Mac (Metal). It is the same attempt as
`pow/gpu` (CUDA) and `neb-miner-cpu`, and its results are checked against the C reference of `pow/ref/shares.c`:
digest, ticket and sample indices are equal bit for bit.

```
neb-miner-metal --node 127.0.0.1:3441 --dataset neb-datasets.bin
```

## Requirements

- macOS on Apple silicon (Apple7 GPU family or later; tested on a Mac mini M4 Pro, 24 GB).
- Rust and a C compiler. **No Xcode and no offline `metal` compiler**: the kernel `src/neb_share.metal` is compiled by
  the device when the miner starts (`MTLDevice newLibraryWithSource:`), which is why this works with the Command Line
  Tools alone.

## Build and check

```
cd node
cargo build --release -p neb-metalminer
cargo test --release -p neb-metalminer       # the kernel against the C reference
```

## Run

```
# 1. the node, with the lock of a wallet that receives the coins
neb-wallet --wallet wallet.txt new
neb-node --run --dataset neb-datasets.bin --datadir data --miner-lock "$(neb-wallet --wallet wallet.txt lock)"

# 2. the GPU miner
neb-miner-metal --node 127.0.0.1:3441 --dataset neb-datasets.bin
```

The miner prints a line every 5 s: attempts per second, shares, and blocks accepted and rejected by the node.

**One engine per model slot.** `neb-datasets.bin` holds one dataset per slot, and the miner opens one `MetalEngine`
for each slot whose dataset is the 784-128-10 network of `spec/shares.md` (slots 0 and 1 today), the way
`neb-cpuminer` and `neb-gpuminer` do. Each job is mined with the engine of its `unit.slot`, and a job of a slot this
kernel cannot run waits for the next job: the kernel is that network's shape, so the nebhash v2 slots (KEP1, RET2)
are left to an engine that implements them. `--selftest` reports a skipped slot instead of failing.

```
neb-miner-metal --dataset neb-datasets.bin --selftest --seeds 8 --keys 64   # against the reference, several models
neb-miner-metal --dataset neb-datasets.bin --bench 100000                   # throughput
```

`--batch N` (default 2048) is the number of nonces of one dispatch; `--name TEXT` is the name the node shows.

## Measured on a Mac mini M4 Pro (24 GB, macOS 26)

| Miner | Attempts/s |
|---|---|
| `neb-miner-cpu`, 1 thread (C reference) | 3 052 |
| `neb-miner-cpu`, 12 threads (C reference) | 23 582 |
| `neb-miner-metal`, Apple M4 Pro GPU | **~59 000** |

2.5× the whole 12-core CPU miner and about 19× one core, with identical tickets. The genesis target is met with
probability 2^-20, so one block takes about 18 s of GPU time; a share (2^-15) comes about twice a second.

## How it works

- **One attempt per threadgroup**, 256 threads, 32 KiB of threadgroup memory: sample indices, layer 1, layer 2 and
  the output error, the propagated error, both gradients with the clip, the digest lanes, the ticket. The structure is
  that of the dp4a kernel in `pow/gpu/shares_cuda.cu`.
- Metal has no `dp4a`, so a 4-lane dot product is four multiplies and three adds on `int4` lanes built from `char4`.
  All arithmetic is integer, so the result cannot round differently from the reference.
- The 47 MB dataset, the packed weights, the per-sample pixel sums and the batch records stay in shared-storage
  `MTLBuffer`s; a job change only re-uploads `W1` and `W2` (101 632 bytes).
- Thread `tid` feeds digest lane `tid & 7` (the flat gradients have 784 and 128 entries per row, both multiples of 8),
  then the 256 partial lanes are summed and thread 0 hashes `digest || key` into the ticket.

## Trust: the GPU only proposes

`neb-miner-metal` follows the rules of `neb-engines`. Every attempt that the GPU reports as a share or a solution is
evaluated again by the C reference before it is sent, and a disagreement is counted (`mismatches`) and dropped, never
reported. `cargo test -p neb-metalminer` and `--selftest` compare the GPU with the reference over several models,
datasets and nonces.

## Limits

- Apple GPUs only. The CUDA miner and the CPU miner are unchanged.
- 32 KiB of threadgroup memory per attempt is the binding constraint: one threadgroup per GPU core, so occupancy is
  fixed by that array. A variant with 512 threads and 4 samples per thread was measured at 42 000 attempts/s, and a
  byte-lane SWAR dot product cannot be made exact (the cross terms of a 32-bit multiply leak into the high lane), so
  the kernel keeps the `int4` form above.
- The kernel is written for the shapes of the network (784-128-10, 16 samples); it is not a general matrix engine.
