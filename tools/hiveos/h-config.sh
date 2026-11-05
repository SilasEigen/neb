#!/usr/bin/env bash
# Included by /hive/bin/custom before every start. The wallet field of the flight sheet is a neb1 address (a suffix such as
# ".%WORKER_NAME%" is ignored), the pool URL field lists relays as HOST:PORT separated by spaces or commas, and the extra
# config arguments go to neb-miner-gpu.

dir=$(dirname "${BASH_SOURCE[0]}")
. "$dir/h-manifest.conf"

address=$(tr -d '[:space:]' <<< "$CUSTOM_TEMPLATE" | sed 's/[.%:@].*//')
if [[ -z $address ]]; then
	echo "The wallet field is empty: enter your neb1 address"
	return 1
fi
lock=$("$dir/neb-wallet" lock-of "$address") || {
	echo "Not a valid neb1 address: $address"
	return 1
}

# single quotes keep the values from being run when h-run.sh reads the file
quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
seeds=$(tr ',\n' '  ' <<< "$CUSTOM_URL" | tr -s ' ')
{
	echo "NEB_LOCK=$lock"
	echo "NEB_SEEDS=$(quote "$seeds")"
	echo "NEB_MINER_ARGS=$(quote "$CUSTOM_USER_CONFIG")"
} > "$CUSTOM_CONFIG_FILENAME"
