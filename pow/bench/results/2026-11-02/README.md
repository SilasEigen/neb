# Recorded runs of 2026-11-02: throughput of the CPU engine

Raw outputs of the throughput measurements of the AVX-512 engine, with the command that produced them. They describe one machine, an AMD Zen 5 desktop processor with 16 cores (32 threads, AVX-512
VNNI), under Windows 11 and under Linux in WSL 2. They say nothing about other hardware (see `docs/experiments.md` section
8). No result for a GPU is recorded here. `SHA256SUMS` lists every file of this directory.

The files are the outputs as they were written, with these substitutions: the home directory of the user is written
`<home>`, build directories are written as relative paths, the paths of the dataset and of the models are reduced to file
names, and the lines of `environment.txt` that identify a GPU and a commit are removed. No measured value was changed.

The dataset is the network dataset (`mnist-q8.bin`, built by `tools/make_mnist_dataset.py`). The models are raw dumps of
508,160 bytes of the network after 0, 10, 100, 300, 1,000, 3,000, 10,000 and 100,000 training steps, trained with the C
reference. They are not in this repository, so the rows that use them cannot be repeated without them; every other row can.
The `environment.txt` of each run holds the compiler versions, the number of logical processors and the SHA-256 of the
dataset and of the models.

## `windows/` and `linux-wsl/`: throughput

```
OUT=pow/bench/results/2026-11-02/NAME DATASET=mnist-q8.bin MODELS="model_step0.bin model_step300.bin model_step3000.bin model_step100000.bin" bash pow/bench/run_bench.sh
```

`bench_cpu.jsonl` holds the reference, the AVX-512 dense and sparse variants, and the sparse variant with the checks of a
miner, with 1 thread, half of the logical processors and all of them. A run is valid only if every engine gives the same
checksum of tickets on the same keys; the script stops otherwise. The tables of `docs/experiments.md` section 8 are
medians of these rows.

## Validation logs

The validation logs of these runs (bit-for-bit checks of the CPU path, chain replays, tests and compiler checks) are in the
release asset `neb-0.1.0-experiments.tar.gz` of version 0.1.0, with the commands that produced them. `docs/experiments.md`
section 7 summarises them.
