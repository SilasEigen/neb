# Mining engines

The miners evaluate attempts with the C reference unless the operator asks for an optimised engine. An engine computes the
same tickets faster; it decides nothing about which blocks and shares are valid. The node never uses an engine: it
validates every block and share with the C reference, and CI checks that `neb-node` does not depend on the crate
`neb-engines`.

| component | evaluates tickets with | optimised engines | default |
|---|---|---|---|
| node (`neb-node`) | the C reference (`neb-pow`) | none | the reference |
| CPU miner (`neb-miner-cpu`) | the C reference | AVX-512 VNNI, dense and sparse (`pow/cpu`, crate `neb-engines`): `--engine avx512` or `avx512-dense` | the reference |
| GPU miner (`neb-miner-gpu`) | the dp4a kernel K0 (`pow/gpu/shares_cuda.cu`) | the tensor-core kernels K1 and K2 (`pow/gpu/shares_tc.cu`): `--gpu-kernel k1` or `k2` | K0 |

## Requirements

| engine | needs | when it is not available |
|---|---|---|
| AVX-512 (CPU) | x86-64 with AVX-512 F, BW, VL, DQ and VNNI, and a C compiler that accepts them (gcc, clang or MSVC) | the reference is used, with a message |
| K0 (GPU) | NVIDIA GPU of compute capability 7.5 or later, CUDA 13 toolkit to build, driver 580 or later | the GPU miner is not built or does not start |
| K1, K2 (GPU) | compute capability 8.0 or later (int8 `mma.sync`) | K0 is used on that device, with a message |

## Safeguards

1. **Off by default.** No engine other than the reference or K0 runs unless the operator names it.
2. **Self-test before use.** The engine is compared with the reference on five models (the initial state, a trained model,
   all weights 7, all -8, alternating) and on two datasets (the network dataset and the synthetic one of `spec/shares.md`
   section 9). The CPU engine compares the digest, the ticket, the 16 sample indices and the clipped gradients; the GPU
   kernel compares the hits for three bounds and exact tickets. Any difference, or missing instructions, means the miner
   says why and falls back.
3. **The reference decides every reported nonce.** A nonce that an engine reports as a share or a block is evaluated again
   with the C reference before it is sent. A nonce that the reference refuses is never sent and counts as a fault.
4. **Checks during the run.** The model of each job is checked on 16 attempts (CPU) or 4 exact tickets (GPU), and one
   attempt in 4,096 (CPU) or one batch in 64 (GPU) is audited with the reference even when the engine reports no hit.
5. **Fallback.** On any difference the engine is switched off for the rest of the run (CPU: the reference; GPU: K0 on that
   device, with the batch scanned again).
6. **No silent truncation.** A scan with more hits than the device can report is split until it is complete.

A wrong engine cannot make a node accept an invalid block or share. It can make its own miner lose shares and blocks: rules
3 to 5 catch wrong hits at once, while a missed hit is caught only by the model checks and the audits, with high
probability for a systematic fault. Enable an engine on new hardware only after its self-test passes.

```bash
node/target/release/neb-miner-cpu --synthetic --engine avx512 --selftest             # exit code 0 when the engine passes
node/target/release/neb-miner-gpu --selftest                                          # K0, and K1 and K2 where supported
node/target/release/neb-miner-gpu --node 127.0.0.1:3441 --dataset ../neb-datasets.bin --gpu-kernel k2
```

## Numerical range

The model step is applied by the C reference only, in the node and in the miners; the engines compute tickets and gradients.
The C step is exact while `|R + GS|` fits in an int32 and is not `-2^31`. A block lists at most 256 shares, so
`|GS| <= 65,536` and `|R + GS| <= 98,304` in layer 1, far inside that range; `Model::try_apply_step` refuses a step outside
it, and `GradSum` refuses a share beyond `MAX_SHARES`. In the AVX-512 and tensor-core code the intermediate values are
bounded by the clip: pre-activations of layer 1 are at most `784 * 7 * 8 = 43,904` in absolute value, activations are in
`-8..7`, the sums of `G2` over 16 samples are at most 1,024, the clipped gradients are in `-256..256`, and the digest lanes
wrap modulo `2^32` as the specification defines them.

Implementation notes for the CUDA kernels are in `pow/gpu/README.md`, and for the AVX-512 code in `pow/cpu/README.md`. The
validation and the measured throughput are in `docs/experiments.md`, sections 7 and 8.
