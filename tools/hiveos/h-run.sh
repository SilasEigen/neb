#!/usr/bin/env bash
# Runs a node and the GPU miner on the rig. The miner uses every GPU, so there is nothing to dispatch.

cd "$(dirname "$0")" || exit 1
. ./h-manifest.conf
. "$CUSTOM_CONFIG_FILENAME" || {
	echo "No config file: h-config.sh did not run"
	exit 1
}

# the Hive agent points LD_LIBRARY_PATH to its own libraries; the miner uses those of the system and of the driver
unset LD_LIBRARY_PATH

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")" data
# a node that an earlier run left behind
pkill -x neb-node 2> /dev/null && sleep 1
rm -f stats.json

node_args=(--run --dataset mnist-q8.bin --datadir data --miner-lock "$NEB_LOCK" --color never)
for seed in $NEB_SEEDS; do
	node_args+=(--seed "$seed")
done
./neb-node "${node_args[@]}" >> "$CUSTOM_LOG_BASENAME-node.log" 2>&1 &
node_pid=$!
trap 'kill $node_pid 2> /dev/null' EXIT

# the miner connects to the work service of the node, which opens after the chain is replayed
for _ in $(seq 120); do
	(exec 3<> /dev/tcp/127.0.0.1/3441) 2> /dev/null && break
	kill -0 $node_pid 2> /dev/null || {
		echo "neb-node stopped, see $CUSTOM_LOG_BASENAME-node.log"
		exit 1
	}
	sleep 1
done

# shellcheck disable=SC2086
./neb-miner-gpu --node 127.0.0.1:3441 --dataset mnist-q8.bin --plain --stats stats.json $NEB_MINER_ARGS 2>&1 | tee -a "$CUSTOM_LOG_BASENAME.log"
exit "${PIPESTATUS[0]}"
