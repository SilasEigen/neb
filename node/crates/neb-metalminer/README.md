# neb-metalminer — the NEB miner on an Apple GPU

`neb-miner-metal` mines the NEB testnet on the GPU of an Apple silicon Mac (Metal). It is the same attempt as
`pow/gpu` (CUDA) and `neb-miner-cpu`, and its results are checked against the C references of `pow/ref/shares.c` and
`pow/ref/nebhash2.c`: digest, ticket and sample indices are equal bit for bit. It mines the 784-128-10 network of
`spec/shares.md` and the two architectures of nebhash v2 (`spec/nebhash-v2.md`, KEP1 and RET2).

```
neb-miner-metal --node 127.0.0.1:3441 --dataset neb-datasets.bin
```

## Requirements

- macOS on Apple silicon (Apple7 GPU family or later; tested on a Mac mini M4 Pro, 24 GB).
- Rust and a C compiler. **No Xcode and no offline `metal` compiler**: the kernels `src/neb_share.metal` (784-128-10)
  and `src/neb_share2.metal` (nebhash v2), each after the shared hash code `src/neb_common.metal`, are compiled by the
  device when the miner starts (`MTLDevice newLibraryWithSource:`), which is why this works with the Command Line
  Tools alone.

## Build and check

```
cd node
cargo build --release -p neb-metalminer
cargo test --release -p neb-metalminer       # the kernels against the C references and the nebhash v2 vectors
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
for each, with the kernel of the dataset's architecture (`neb_share` for the 784-128-10 network, `neb_share2` for
KEP1 and RET2), the way `neb-cpuminer` and `neb-gpuminer` do. Each job is mined with the engine of its `unit.slot`; a
job whose model is of another network than the slot's dataset (an old dataset file after an upgrade) waits for the
next job.

```
neb-miner-metal --dataset neb-datasets.bin --selftest --seeds 8 --keys 64   # against the reference, several models
neb-miner-metal --dataset neb-datasets.bin --bench 100000                   # throughput of slot 0
neb-miner-metal --dataset neb-datasets.bin --bench 4000 --slot 3            # throughput of another slot
```

`--selftest` runs every slot of the file.

`--batch N` (default 2048) is the number of nonces of one dispatch; `--name TEXT` is the name the node shows.

## Measured on a Mac mini M4 Pro (24 GB, macOS 26)

| Miner | Attempts/s |
|---|---|
| `neb-miner-cpu`, 1 thread (C reference) | 3 052 |
| `neb-miner-cpu`, 12 threads (C reference) | 23 582 |
| `neb-miner-metal`, Apple M4 Pro GPU | **~59 000** |
| `neb-miner-metal`, KEP1 (slot 2 of the testnet file) | ~6 300 |
| `neb-miner-metal`, RET2 (slot 3) | ~1 800 |

The first three rows are the 784-128-10 network; the last two are nebhash v2 (`--bench 4000 --slot N`, batch 2048),
whose attempts cost more (180 800 and 680 000 multiply-accumulates per sample against 101 632). 2.5× the whole
12-core CPU miner and about 19× one core, with identical tickets. The genesis target is met with
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

**nebhash v2** (`src/neb_share2.metal`) follows `pow/gpu/nebhash2_cuda.cu`: one threadgroup per attempt, all threads
sharing the positions of a layer; for each of the 16 samples the forward and backward passes run layer by layer, and
the inputs and the routed errors of every layer wait in a scratch area of device memory (a barrier with
`mem_flags::mem_device` separates a write from the reads of the other threads); then each thread computes whole
entries of the gradient, clips them and adds them to the digest lanes (sums modulo 2^32, so the order of the
additions does not matter). The weights, at most 26 000 bytes, sit in threadgroup memory. One kernel serves both
architectures: the layer table is the one of the C reference (`neb2_arch_get`, read through FFI, not copied), and
the host derives the scratch layout from it. The scratch holds 512 attempts, so a batch is cut into dispatches of
512 (about 180 MB for RET2).

## Trust: the GPU only proposes

`neb-miner-metal` follows the rules of `neb-engines`. Every attempt that the GPU reports as a share or a solution is
evaluated again by the C reference before it is sent, and a disagreement is counted (`mismatches`) and dropped, never
reported. `cargo test -p neb-metalminer` and `--selftest` compare the GPU with the reference over several models,
datasets and nonces; for nebhash v2 the tests also replay `vectors/nebhash2.txt` and `vectors/nebhash2_s1.txt` (the
latter with weights that move, so the later steps run on trained models) and cross a cut between two dispatches.

## Limits

- Apple GPUs only. The CUDA miner and the CPU miner are unchanged.
- 32 KiB of threadgroup memory per attempt is the binding constraint: one threadgroup per GPU core, so occupancy is
  fixed by that array. A variant with 512 threads and 4 samples per thread was measured at 42 000 attempts/s, and a
  byte-lane SWAR dot product cannot be made exact (the cross terms of a 32-bit multiply leak into the high lane), so
  the kernel keeps the `int4` form above.
- The 784-128-10 kernel is written for the shapes of that network (16 samples); the nebhash v2 kernel reads any table
  of at most 6 layers and 26 000 weights (`NEB2_MAX_LAYERS`, `NEB2_MAX_WEIGHTS`) and is not tuned beyond the CUDA
  structure: it spends most of its time in barriers and in the scratch traffic.
