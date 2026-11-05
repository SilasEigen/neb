#!/bin/bash
# AVX-512 evaluation of training shares (pow/cpu): differential test against the C reference, and two negative controls
# (deliberately wrong builds, which the test must reject). Needs gcc and a processor with AVX-512 F, BW, VL, DQ and
# VNNI; prints SKIP and exits 0 without them. KEYS=N changes the number of keys per case (default 2048). REAL=FILE and MODELS="F1 F2"
# add the network dataset file with raw model dumps (REAL_KEYS=N keys each, default 100000).
# BUILD=DIR puts the binaries in DIR (default: build).
set -euo pipefail
cd "$(dirname "$0")/../.."
BUILD="${BUILD:-build}"
KEYS="${KEYS:-2048}"
mkdir -p "$BUILD"
if [ -r /proc/cpuinfo ]; then
  for f in avx512f avx512bw avx512vl avx512dq avx512_vnni; do
    grep -qw "$f" /proc/cpuinfo || { echo "SKIP no $f on this processor"; exit 0; }
  done
fi
FLAGS="-O2 -fopenmp -Wall -Wextra -Werror -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -Ipow/ref -Ipow/cpu -Ipow/tests"
SRC="pow/tests/test_shares_avx512.c pow/cpu/shares_avx512.c pow/tests/syn_dataset.c pow/ref/shares.c pow/ref/chacha8.c pow/ref/blake3_min.c"
gcc $FLAGS $SRC -o "$BUILD/test_shares_avx512"
REALARGS=()
if [ -n "${REAL:-}" ]; then
  REALARGS=(--real "$REAL" --real-keys "${REAL_KEYS:-100000}")
  for m in ${MODELS:-}; do REALARGS+=(--model "$m"); done
fi
"$BUILD/test_shares_avx512" "$KEYS" ${REALARGS[@]+"${REALARGS[@]}"}
for control in CLIP SPARSE_WORD; do
  gcc $FLAGS -DNEB_NEGATIVE_CONTROL_$control $SRC -o "$BUILD/test_shares_avx512_neg_$control"
  if "$BUILD/test_shares_avx512_neg_$control" 256 > "$BUILD/neg_$control.log"; then
    echo "FAIL the negative control $control was not detected"
    exit 1
  fi
  echo "PASS negative control $control detected: $(tail -1 "$BUILD/neg_$control.log")"
done
echo "AVX512 SHARES PASSED"
