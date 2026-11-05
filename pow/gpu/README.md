# CUDA kernels of the training share (`pow/gpu`)

| Files | Content |
|---|---|
| `shares_cuda.cu`, `shares_cuda.h` | K0: the dp4a kernel, the default of the GPU miner. |
| `shares_tc.cu`, `shares_tc.h`, `tc_common.cuh` | K1 (dense) and K2 (sparse): one training share per thread block with int8 tensor-core instructions (`mma`). K2 skips the hidden units whose propagated error is zero. |
| `scan_cuda.cu`, `scan_cuda.h` | The nonce scan of the GPU miner: header mode, bound test, hit list, and the choice of the kernel. |

Every kernel gives the digest, the ticket, the 16 sample indices and the clipped gradients of `pow/ref/shares.c`, bit for bit.
The kernels are mining optimisations, not consensus: the node never uses them, and the miner evaluates every nonce that a
device reports again with the C reference before it sends it. K1 and K2 are off by default (`--gpu-kernel k1|k2`).

- **Compute capability.** K0 needs 7.5 or later. K1 and K2 need 8.0 or later: below it the `mma` wrappers do nothing and the
  kernel traps, and the host never launches it (`neb_share_tc_supported`). The unit compiles for sm_75, and K1 and K2 cannot
  run there.
- **Staging buffer.** The host staging buffer of the model is local to each call, so that several threads can upload models to
  the same device at the same time (test `models_uploaded_from_several_threads_do_not_corrupt_each_other` in
  `node/crates/neb-gpuminer/tests/cuda_kernels.rs`).
- **Negative control.** `NEB_NEGATIVE_CONTROL_TC_CLIP` makes the clip of `G1` wrong on purpose; `pow/tests/check_shares_tc.sh`
  and `.bat` build with it and check that the differential test fails.
- **One device link per kernel family.** The objects are built without relocatable device code, and each has a 32-byte
  constant bank 3 with its BLAKE3 initial value. Device-linking K0 together with K1 and K2 merges the two banks into one
  section of 32 bytes, with `kTcB3IV` at offset 0 and `kB3IV` at offset 32, beyond its end: K0 then reads its initial value
  from outside the section and returns wrong tickets, although every file compiles and K1 and K2 are right.
  `node/crates/neb-gpuminer/build.rs` therefore builds `scan_cuda.cu` and `shares_cuda.cu` in one library and `shares_tc.cu` in
  another. Keep the two families in separate libraries.

## Checks

```
bash pow/tests/check_shares_tc.sh                          # K0, K1 and K2 against the reference (Windows: check_shares_tc.bat)
(cd node && cargo test --locked --features cuda -p neb-gpuminer)
neb-miner-gpu --selftest                                   # every kernel on every device, against the reference
```

These checks need a CUDA device. This repository records no result of them for any particular device. A miner on other
hardware relies on the self-test and on the verification of every reported nonce, and the kernels are unproven there.
