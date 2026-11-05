#!/bin/bash
# Checks that every constant row "| NAME | value |" of the constants table of spec/shares.md equals the
# "#define NEB_NAME" of pow/ref/shares.h.
set -u
cd "$(dirname "$0")/../.."
SPEC=spec/shares.md
HDR=pow/ref/shares.h
KEYS="NIN NH NOUT NB S1 S2 SB SU1 SU2 CLIP TPOS TNEG"
[ -f "$SPEC" ] && [ -f "$HDR" ] || { echo "FAIL spec_constants_match_header: $SPEC or $HDR missing"; exit 1; }
fail=0
for k in $KEYS; do
  sv=$(grep -E "^\| $k \|" "$SPEC" | head -n1 | sed -E 's/^\| [A-Z0-9_]+ \| *(-?[0-9]+) *\|.*/\1/')
  hv=$(grep -E "^#define NEB_$k[[:space:]]" "$HDR" | head -n1 | sed -E 's/^#define [A-Z0-9_]+[[:space:]]+\(?(-?[0-9]+)\)?[uU]?.*/\1/')
  if [ -z "$sv" ] || [ -z "$hv" ] || [ "$sv" != "$hv" ]; then echo "FAIL $k: spec '$sv' header '$hv'"; fail=1; fi
done
[ $fail -eq 0 ] && echo "PASS spec_constants_match_header"
exit $fail
