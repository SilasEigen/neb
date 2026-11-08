#!/usr/bin/env python3
"""Blind recovery check: does the search find the period of the known planets (P < 20 d) of the calibration set?

    NEB_KEPLER_CACHE=DIR NEB_KEPLER_TEST=FILE python check_bls.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib  # noqa: E402
import kb_common as K  # noqa: E402
import vet  # noqa: E402


def main():
    rows = {calib.key(r): r for r in calib.test_tces()}
    res = [x for x in csv.DictReader(open(os.path.join(K.RESULTS, "calib.csv")))
           if x["label"] == "1" and float(x["period"]) < 20]
    others = {}
    for r in csv.DictReader(open(os.path.join(calib.KEP, "dr24_tce.csv"))):
        others.setdefault(int(r["kepid"]), []).append((float(r["tce_period"]), r["av_training_set"]))
    hits = other_hits = 0
    for x in res:
        r = rows[x["tce"]]
        P, dh = float(r["tce_period"]), float(r["tce_duration"])
        t, f = K.read_lc(os.path.join(K.LC_DIR, K.lc_name("kic-%09d" % int(r["kepid"]))))
        v = vet.vet(t, f, P, float(r["tce_time0bk"]), dh / 24)
        bp, bt0, bd = vet.search(t, f)
        hit = any(abs(bp / P - k) < 0.002 * k for k in (1, 0.5, 2))
        other = "" if hit else next(("other known TCE %.3f d (%s)" % (q, lab) for q, lab in others.get(int(r["kepid"]), [])
                                     if any(abs(bp / q - k) < 0.002 * k for k in (1, 0.5, 2))), "")
        hits += hit
        other_hits += bool(other)
        print("P %7.3f dur %4.1f h  MES(true) %6s  found %7.3f  %s%s" % (P, dh, v.get("mes"), bp, "OK" if hit else "", other), flush=True)
    print("recovered %d of %d; %d more found another known signal of the same star" % (hits, len(res), other_hits))


if __name__ == "__main__":
    main()
