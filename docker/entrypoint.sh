#!/bin/sh
# Entrypoint of the node image. `run` (the default) starts a node from the environment; any other command is run as given,
# for example `neb-node --genesis` or `neb-wallet new`.
#
#   NEB_MINER_ADDRESS  neb1... address that receives the rewards of the jobs of this node (or NEB_MINER_LOCK, 64 hex)
#   NEB_SEEDS          comma-separated HOST:PORT peers that the node keeps dialled
#   NEB_MAX_PEERS      peer sessions, inbound and outbound together
#   NEB_LISTEN         peer bind address (default 0.0.0.0)
#   NEB_DATASET        path of the dataset file (default $NEB_DATA_DIR/mnist-q8.bin)
#   NEB_DATASET_URL    where to download the dataset file when it is missing; the node refuses a file with another hash
#   NEB_EXTRA_ARGS     more arguments for neb-node, split on spaces
set -eu

if [ "${1:-run}" != "run" ]; then
  exec "$@"
fi

data="${NEB_DATA_DIR:-/data}"
dataset="${NEB_DATASET:-$data/mnist-q8.bin}"

if [ ! -s "$dataset" ]; then
  if [ -z "${NEB_DATASET_URL:-}" ]; then
    echo "neb: the dataset file $dataset is missing: mount mnist-q8.bin there or set NEB_DATASET_URL" >&2
    exit 64
  fi
  echo "neb: downloading the dataset file from $NEB_DATASET_URL" >&2
  curl --fail --location --silent --show-error --retry 3 --output "$dataset.part" "$NEB_DATASET_URL"
  mv "$dataset.part" "$dataset"
fi

if [ -n "${NEB_MINER_ADDRESS:-}" ]; then
  NEB_MINER_LOCK="$(neb-wallet lock-of "$NEB_MINER_ADDRESS")"
fi

set -- neb-node --run --dataset "$dataset" --datadir "$data/chain" --listen "${NEB_LISTEN:-0.0.0.0}" --color never
[ -z "${NEB_MINER_LOCK:-}" ] || set -- "$@" --miner-lock "$NEB_MINER_LOCK"
[ -z "${NEB_MAX_PEERS:-}" ] || set -- "$@" --max-peers "$NEB_MAX_PEERS"
if [ -n "${NEB_SEEDS:-}" ]; then
  old_ifs="$IFS"; IFS=','
  for seed in $NEB_SEEDS; do set -- "$@" --seed "$seed"; done
  IFS="$old_ifs"
fi
# shellcheck disable=SC2086
exec "$@" ${NEB_EXTRA_ARGS:-}
