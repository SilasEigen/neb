#!/usr/bin/env python3
"""Check of vet.py on the calibration signals: known Kepler planets and false positives of the test stars, at their
catalog ephemeris. These hosts are mostly FGK stars, not M dwarfs: the duration and shape tests use a density floor
of 0.1 solar (subgiants included) instead of the M dwarf floor.

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

RHO_FLOOR_FGK = 0.1


def main():
    rows = {calib.key(r): r for r in calib.test_tces()}
    out = {0: Counter(), 1: Counter()}
    for x in csv.DictReader(open(os.path.join(K.RESULTS, "calib.csv"))):
        r = rows[x["tce"]]
        t, f, q = K.read_lc(os.path.join(K.LC_DIR, K.lc_name("kic-%09d" % int(r["kepid"]))), quarters=True)
        keep = vet.clean_mask(t, f)
        v = vet.vet(t[keep], f[keep], float(r["tce_period"]), float(r["tce_time0bk"]), float(r["tce_duration"]) / 24,
                    quarter=q[keep], rho_min=RHO_FLOOR_FGK)
        out[int(x["label"])][v["fail"] or "PASS"] += 1
    for lab, name in ((1, "planets"), (0, "false positives")):
        print("%s (%d): %s" % (name, sum(out[lab].values()), dict(out[lab].most_common())))


if __name__ == "__main__":
    main()
