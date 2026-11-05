#!/bin/bash
# Reproducible throughput measurements of the ticket engines (AVX-512 on the CPU, K0, K1 and K2 on a CUDA device), as JSON
# lines in OUT. Run it on a quiet machine, from Linux, WSL or Git Bash, with a Rust toolchain (and CUDA for CUDA=1):
#   OUT=pow/bench/results/NAME DATASET=mnist-q8.bin MODELS="model_step0.bin model_step3000.bin" CUDA=1 bash pow/bench/run_bench.sh
# DATASET is the network dataset file (default: the synthetic dataset), MODELS raw model dumps of 508160 bytes (default: the
# initial model). Threads: 1 (16384 attempts, 3 repetitions), half of the logical processors and all of them (262144
# attempts, 5 repetitions). Every engine of a run evaluates the same keys: the "checksum" fields must agree (the script
# stops if they do not). The measurements are of this machine; nothing here is a claim about another.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:?set OUT to the output directory}"
mkdir -p "$OUT"
LOGICAL="$(nproc 2>/dev/null || echo 4)"
HALF=$((LOGICAL / 2))
[ "$HALF" -ge 1 ] || HALF=1
ARGS=()
[ -n "${DATASET:-}" ] && ARGS+=(--dataset "$DATASET")
for m in ${MODELS:-}; do ARGS+=(--model "$m"); done
MANIFEST=node/Cargo.toml
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit $(git rev-parse HEAD)"
  echo "uname $(uname -srm)"
  grep -m1 'model name' /proc/cpuinfo 2>/dev/null || true
  echo "logical_processors $LOGICAL"
  rustc --version
  command -v gcc >/dev/null && gcc --version | head -1 || true
  command -v nvcc >/dev/null && nvcc --version | tail -2 | head -1 || true
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,compute_cap,power.limit --format=csv,noheader || true
  echo "dataset ${DATASET:-synthetic}"
  for m in ${MODELS:-}; do echo "model $m $(sha256sum "$m" | cut -d' ' -f1)"; done
  [ -n "${DATASET:-}" ] && echo "dataset_sha256 $(sha256sum "$DATASET" | cut -d' ' -f1)" || true
} > "$OUT/environment.txt"
cargo build --release --locked --manifest-path $MANIFEST -p neb-engines --examples
BENCH_CPU="${CARGO_TARGET_DIR:-node/target}/release/examples/bench_cpu"
[ -x "$BENCH_CPU" ] || BENCH_CPU="$BENCH_CPU.exe"
: > "$OUT/bench_cpu.jsonl"
"$BENCH_CPU" "${ARGS[@]}" --threads 1 --attempts 16384 --reps 3 >> "$OUT/bench_cpu.jsonl"
"$BENCH_CPU" "${ARGS[@]}" --threads "$HALF,$LOGICAL" --attempts 262144 --reps 5 >> "$OUT/bench_cpu.jsonl"
if grep -q '"checksum_equal_to_first_engine":false' "$OUT/bench_cpu.jsonl"; then echo "INVALID: engines disagree"; exit 1; fi
if [ "${CUDA:-0}" = 1 ]; then
  cargo build --release --locked --manifest-path $MANIFEST -p neb-gpuminer --features cuda --examples
  BENCH_GPU="${CARGO_TARGET_DIR:-node/target}/release/examples/bench_gpu"
  [ -x "$BENCH_GPU" ] || BENCH_GPU="$BENCH_GPU.exe"
  "$BENCH_GPU" "${ARGS[@]}" --reps 7 --scans 16 > "$OUT/bench_gpu.jsonl"
  if grep -q '"equal_to_first_kernel":false' "$OUT/bench_gpu.jsonl"; then echo "INVALID: kernels disagree"; exit 1; fi
fi
echo "done: $OUT"
