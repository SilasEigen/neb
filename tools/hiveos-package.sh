#!/bin/bash
# Builds the HiveOS custom miner package: nebminer-VERSION.tar.gz, with the scripts of tools/hiveos, the node, the wallet, the
# GPU miner and the dataset file.
# Usage: tools/hiveos-package.sh BINDIR DATASET OUTDIR     (BINDIR holds the Linux neb-node, neb-wallet and neb-miner-gpu)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
bin=$(cd "$1" && pwd)
dataset=$(readlink -f "$2")
out=$(mkdir -p "$3" && cd "$3" && pwd)
version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/node/Cargo.toml" | head -1)
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/nebminer"
cp "$root"/tools/hiveos/h-*.sh "$stage/nebminer/"
sed "s/^CUSTOM_VERSION=.*/CUSTOM_VERSION=$version/" "$root/tools/hiveos/h-manifest.conf" > "$stage/nebminer/h-manifest.conf"
cp "$bin/neb-node" "$bin/neb-wallet" "$bin/neb-miner-gpu" "$stage/nebminer/"
cp "$dataset" "$stage/nebminer/mnist-q8.bin"
chmod 755 "$stage"/nebminer/h-*.sh "$stage/nebminer/neb-node" "$stage/nebminer/neb-wallet" "$stage/nebminer/neb-miner-gpu"
chmod 644 "$stage/nebminer/h-manifest.conf" "$stage/nebminer/mnist-q8.bin"
tar --owner=0 --group=0 --numeric-owner --sort=name --mtime='2026-10-30 00:00:00 UTC' -C "$stage" -czf "$out/nebminer-$version.tar.gz" nebminer
echo "$out/nebminer-$version.tar.gz"
