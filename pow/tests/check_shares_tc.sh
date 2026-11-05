#!/bin/bash
# Differential test of the CUDA kernels K0, K1 and K2 against the C reference, and a negative control (the tensor-core
# kernels built wrong on purpose, which the test must reject). Linux or WSL, with nvcc and gcc. Set NVCC if nvcc is not on the
# path, ARCH for the GPU generation (default native, the installed device; K1 and K2 need sm_80 or later), KEYS for the keys per case (default
# 2048), BUILD for the output directory (default: build). REAL=FILE and MODELS="F1 F2" add the network dataset file with raw
# model dumps (REAL_KEYS=N keys each, default 100000).
set -euo pipefail
cd "$(dirname "$0")/../.."
BUILD="${BUILD:-build}"
NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-native}"
KEYS="${KEYS:-2048}"
mkdir -p "$BUILD"
SRC="pow/tests/test_shares_tc.cu pow/gpu/shares_cuda.cu pow/gpu/shares_tc.cu pow/tests/syn_dataset.c pow/ref/shares.c pow/ref/chacha8.c pow/ref/blake3_min.c"
# the C files are compiled as C by nvcc's host compiler, the .cu files as C++
objs=()
for f in pow/tests/syn_dataset.c pow/ref/shares.c pow/ref/chacha8.c pow/ref/blake3_min.c; do
  o="$BUILD/tc_$(basename "$f" .c).o"
  gcc -O2 -Wall -Wextra -Ipow/ref -Ipow/tests -c "$f" -o "$o"
  objs+=("$o")
done
CU="pow/tests/test_shares_tc.cu pow/gpu/shares_cuda.cu pow/gpu/shares_tc.cu"
"$NVCC" -O3 -arch="$ARCH" -std=c++17 -Xcompiler -fopenmp -Ipow/ref -Ipow/gpu -Ipow/tests -o "$BUILD/test_shares_tc" $CU "${objs[@]}"
REALARGS=()
if [ -n "${REAL:-}" ]; then
  REALARGS=(--real "$REAL" --real-keys "${REAL_KEYS:-100000}")
  for m in ${MODELS:-}; do REALARGS+=(--model "$m"); done
fi
"$BUILD/test_shares_tc" "$KEYS" 64 ${REALARGS[@]+"${REALARGS[@]}"}
"$NVCC" -O3 -arch="$ARCH" -std=c++17 -Xcompiler -fopenmp -DNEB_NEGATIVE_CONTROL_TC_CLIP -Ipow/ref -Ipow/gpu -Ipow/tests -o "$BUILD/test_shares_tc_neg" $CU "${objs[@]}"
if "$BUILD/test_shares_tc_neg" 256 > "$BUILD/test_shares_tc_neg.log"; then
  echo "FAIL the negative control was not detected"
  exit 1
fi
echo "PASS negative control detected: $(tail -1 "$BUILD/test_shares_tc_neg.log")"
echo "SHARES TC CHECK PASSED"
