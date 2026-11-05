# AVX-512 evaluation of the training share (`pow/cpu`)

`shares_avx512.c` evaluates one training share (the attempt of `spec/shares.md`) with AVX-512 VNNI instructions. It has two
variants: dense, which computes every pixel and every hidden unit, and sparse, which skips the zero pixels and the hidden
units whose propagated error is zero. Both give the digest, the ticket, the 16 sample indices and the clipped gradients of
`pow/ref/shares.c`, bit for bit. The BLAKE3 used for the ticket is `pow/ref/blake3_min.c`.

It is a mining optimisation, not consensus: the node does not depend on it and validates with `pow/ref`. The CPU miner uses it
only when asked (`neb-miner-cpu --engine avx512`), after a self-test against the reference.

- **Build.** `node/crates/neb-engines/build.rs` compiles the unit on its own, for x86-64 only, with the AVX-512 flags for
  that unit only. It is portable to MSVC and gcc. At run time it needs AVX-512 F, BW, VL, DQ and VNNI, which the caller checks.
- **Interface.** `shares_avx512.h`: dataset, prepared weights, scratch space, one share. Allocation failures return NULL.
- **Negative controls.** `NEB_NEGATIVE_CONTROL_CLIP` and `NEB_NEGATIVE_CONTROL_SPARSE_WORD` break the code on purpose;
  `pow/tests/check_avx512.sh` builds with them and checks that the differential test fails.

## Checks

```
bash pow/tests/check_avx512.sh                            # against the reference, with the two negative controls
(cd node && cargo test --locked -p neb-engines)           # includes tests/differential.rs
neb-miner-cpu --dataset FILE --engine avx512 --selftest   # exit code 0 when the engine passes
```

The recorded runs are in `pow/bench/results/`.
