#!/usr/bin/env python3
"""Check of vet.py on the calibration signals: known Kepler planets and false positives of the test stars, at their
catalog ephemeris. The duration test is left out: these hosts are not M dwarfs.

    NEB_KEPLER_CACHE=DIR NEB_KEPLER_TEST=FILE python check_vet.py      (after calib.py)

check_bls.py measures the blind search on the same planets.
"""
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib  # noqa: E402
import kb_common as K  # noqa: E402
import vet  # noqa: E402


def main():
    rows = {calib.key(r): r for r in calib.test_tces()}
    out = {0: Counter(), 1: Counter()}
    for x in csv.DictReader(open(os.path.join(K.RESULTS, "calib.csv"))):
        r = rows[x["tce"]]
        t, f = K.read_lc(os.path.join(K.LC_DIR, K.lc_name("kic-%09d" % int(r["kepid"]))))
        v = vet.vet(t, f, float(r["tce_period"]), float(r["tce_time0bk"]), float(r["tce_duration"]) / 24)
        fail = "" if v["fail"] == "too long for an M dwarf" else v["fail"]
        out[int(x["label"])][fail or "PASS"] += 1
    for lab, name in ((1, "planets"), (0, "false positives")):
        print("%s (%d): %s" % (name, sum(out[lab].values()), dict(out[lab].most_common())))


if __name__ == "__main__":
    main()
