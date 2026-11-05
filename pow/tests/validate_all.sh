#!/bin/bash
# Runs every check of the optimised engines and of the recorded chains, one after the other, and writes a log per step in
# LOGDIR with a PASS/FAIL line for each at the end. Linux or WSL, gcc, Python with numpy, a Rust toolchain; CUDA=1 adds the
# CUDA checks (nvcc and a device). The checks that need the network dataset and trained models run when DATASET and MODELS
# are set (the dataset is the file `tools/make_mnist_dataset.py` writes, the models are raw dumps of 508160 bytes).
# EXHAUSTIVE=1 adds the scan of the whole domain of the model step (7.7e10 entries, ten minutes or more); REQUIRE_AVX512=1 makes a
# missing AVX-512 a failure of the Rust tests.
#   LOGDIR=$HOME/val DATASET=mnist-q8.bin MODELS="m0.bin m3000.bin" CUDA=1 bash pow/tests/validate_all.sh
set -u
cd "$(dirname "$0")/../.."
LOGDIR="${LOGDIR:?set LOGDIR}"
mkdir -p "$LOGDIR"
export BUILD="${BUILD:-$LOGDIR/build}"
mkdir -p "$BUILD"
M=node/Cargo.toml
RESULTS=()
run() {  # name, command...
  local name="$1"; shift
  echo "== $name"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then RESULTS+=("PASS $name"); else RESULTS+=("FAIL $name"); fi
  tail -2 "$LOGDIR/$name.log"
}
REALENV=()
[ -n "${DATASET:-}" ] && REALENV+=(NEB_DATASET="$DATASET" NEB_REQUIRE_DATASET=1)
run spec_constants bash pow/tests/check_spec_constants.sh
run reference_werror bash -c 'for f in shares chacha8 blake3_min; do gcc -O2 -Wall -Wextra -Werror -Ipow/ref -c pow/ref/$f.c -o "$BUILD/$f.o" || exit 1; done'
run oracle_selftest python3 -B -I pow/oracle/shares_oracle.py --selftest
run share_vectors bash pow/tests/check_share_vectors.sh
run avx512_differential env KEYS="${KEYS:-2048}" ${DATASET:+REAL="$DATASET"} ${MODELS:+MODELS="$MODELS"} ${REAL_KEYS:+REAL_KEYS="$REAL_KEYS"} bash pow/tests/check_avx512.sh
run cargo_test_release env ${REQUIRE_AVX512:+NEB_REQUIRE_AVX512=1} "${REALENV[@]}" cargo test --locked --release --manifest-path $M
run cargo_test_debug env "${REALENV[@]}" cargo test --locked --manifest-path $M
[ "${EXHAUSTIVE:-0}" = 1 ] && run step_exhaustive cargo test --locked --release --manifest-path $M -p neb-pow --test step_boundaries -- --ignored --nocapture
run chain_replay_synthetic cargo run --locked --release --quiet --manifest-path $M -p neb-chain --example chain_fixture -- replay pow/tests/chains/synthetic-stress.txt
run oracle_chain_synthetic python3 -B -I pow/oracle/chain_oracle.py pow/tests/chains/synthetic-stress.txt
run chain_replay_pools_synthetic cargo run --locked --release --quiet --manifest-path $M -p neb-chain --example chain_fixture -- replay pow/tests/chains/pools-synthetic.txt
run oracle_chain_pools_synthetic python3 -B -I pow/oracle/chain_oracle.py pow/tests/chains/pools-synthetic.txt
if [ -n "${DATASET:-}" ]; then
  for c in pools-mnist; do
    run chain_replay_$c cargo run --locked --release --quiet --manifest-path $M -p neb-chain --example chain_fixture -- replay pow/tests/chains/$c.txt --dataset "$DATASET"
    run oracle_chain_$c python3 -B -I pow/oracle/chain_oracle.py pow/tests/chains/$c.txt --dataset "$DATASET"
  done
  run cpu_miner_selftest_avx512 bash -c "cargo run --locked --release --quiet --manifest-path $M -p neb-cpuminer -- --dataset '$DATASET' --engine avx512 --selftest"
fi
if [ "${CUDA:-0}" = 1 ]; then
  run cuda_differential env KEYS="${KEYS:-2048}" ${DATASET:+REAL="$DATASET"} ${MODELS:+MODELS="$MODELS"} ${REAL_KEYS:+REAL_KEYS="$REAL_KEYS"} bash pow/tests/check_shares_tc.sh
  run cargo_test_cuda env NEB_REQUIRE_CUDA=1 cargo test --locked --release --manifest-path $M --features cuda -p neb-gpuminer
  run gpu_miner_selftest bash -c "cargo run --locked --release --quiet --manifest-path $M -p neb-gpuminer --features cuda -- --selftest ${DATASET:+--dataset '$DATASET'}"
fi
echo
printf '%s\n' "${RESULTS[@]}" | tee "$LOGDIR/SUMMARY.txt"
! printf '%s\n' "${RESULTS[@]}" | grep -q '^FAIL'
