#!/usr/bin/env bash
# Included by the Hive agent: sets $khs and $stats from the file that neb-miner-gpu rewrites every two seconds.

stats_file=${NEB_STATS_FILE:-/hive/miners/custom/${CUSTOM_MINER:-nebminer}/stats.json}
if [[ -f $stats_file ]] && (( $(date +%s) - $(stat -c %Y "$stats_file") < 60 )); then
	khs=$(jq -r '.khs' "$stats_file")
	stats=$(jq -c 'del(.khs)' "$stats_file")
else
	khs=0
	stats=
fi
