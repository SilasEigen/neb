#!/bin/bash
# Release build: source paths (the repository, the dependencies and the toolchain) are mapped to fixed prefixes. On Linux
# this makes neb-node, neb-wallet and neb-miner-cpu byte-identical between builds; Windows and CUDA binaries are not yet.
# Usage: tools/release-build.sh            (the binaries are in node/target/release, or in $CARGO_TARGET_DIR)
#        NEB_CUDA=1 tools/release-build.sh  also builds neb-miner-gpu (CUDA toolkit required)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
# cargo needs native paths in --remap-path-prefix; cygpath converts them under Git Bash on Windows
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else printf '%s' "$1"; fi; }
# the last matching prefix wins: the general one first
flags=""
for pair in "$HOME=/home" "${RUSTUP_HOME:-$HOME/.rustup}=/rustup" "${CARGO_HOME:-$HOME/.cargo}=/cargo" "$root=/src"; do
  flags="$flags --remap-path-prefix=$(native "${pair%=*}")=${pair##*=}"
done
cd "$root/node"
RUSTFLAGS="$flags" cargo build --release --locked
if [ "${NEB_CUDA:-0}" = 1 ]; then
  RUSTFLAGS="$flags" cargo build --release --locked -p neb-gpuminer --features cuda
fi
