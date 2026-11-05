#!/bin/bash
# Training shares (spec/shares.md): rebuilds the C reference (production constants and the test-only variant of section
# 10.1), regenerates both vector files and compares them byte for byte with vectors/, then replays them with the Python
# oracle (python3 and numpy needed). Run it from Linux, WSL or Git Bash. SKIP_ORACLE=1 runs only the C part (a few
# seconds). BUILD=DIR puts the binaries and the regenerated files in DIR (default: build).
set -euo pipefail
cd "$(dirname "$0")/../.."
BUILD="${BUILD:-build}"
mkdir -p "$BUILD"
SRC="pow/tests/gen_share_vectors.c pow/tests/syn_dataset.c pow/ref/shares.c pow/ref/chacha8.c pow/ref/blake3_min.c"
gcc -O2 -Wall -Wextra -Werror -Ipow/ref $SRC -o "$BUILD/gen_share_vectors"
gcc -O2 -Wall -Wextra -Werror -DNEB_SU1=9 -DNEB_SU2=6 -Ipow/ref $SRC -o "$BUILD/gen_share_vectors_s1"
"$BUILD/gen_share_vectors" 2>/dev/null > "$BUILD/shares.regen.txt"
"$BUILD/gen_share_vectors_s1" 2>/dev/null > "$BUILD/shares_s1.regen.txt"
cmp "$BUILD/shares.regen.txt" vectors/shares.txt
cmp "$BUILD/shares_s1.regen.txt" vectors/shares_s1.txt
echo "PASS C reference reproduces vectors/shares.txt and vectors/shares_s1.txt"
if [ "${SKIP_ORACLE:-0}" != "1" ]; then
  python3 -B -I pow/oracle/shares_oracle.py --selftest | tail -1
  python3 -B -I pow/oracle/shares_oracle.py --check vectors/shares.txt
  python3 -B -I pow/oracle/shares_oracle.py --check vectors/shares_s1.txt --su1 9 --su2 6
fi
echo "SHARE VECTORS PASSED"
