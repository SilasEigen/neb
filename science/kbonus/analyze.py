#!/usr/bin/env python3
"""Hunt results (results/hunt.csv) against the inverted control: survivors above MES thresholds, and checks on the
strongest real ones (neighbours sharing the period, known KOIs nearby).

    python analyze.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hunt as H  # noqa: E402
import kb_common as K  # noqa: E402

rows = [r for r in csv.DictReader(open(H.OUT)) if r.get("real_period")]
print("stars searched:", len(rows))
for th in (7.1, 8, 9, 10, 12, 15, 20, 30):
    n = {tag: sum(H.survivor(r, tag) and float(r[tag + "_mes"] or 0) >= th for r in rows) for tag in ("real", "inv")}
    print("MES >= %4.1f: real %3d  inverted %3d" % (th, n["real"], n["inv"]))

pos = K.mstar_positions()
koi = [k for k in csv.DictReader(open(K.cached(K.KOI_URL, "koi-cumulative-radec.csv"))) if k["koi_period"]]
kp = np.array([float(k["koi_period"]) for k in koi])
kra, kde = np.array([float(k["ra"]) for k in koi]), np.array([float(k["dec"]) for k in koi])


def sep(a, b):
    return 3600 * np.hypot((a[0] - b[0]) * np.cos(np.radians(a[1])), a[1] - b[1])


inv_best = max(float(r["inv_mes"] or 0) for r in rows if H.survivor(r, "inv"))
print("\nstrongest inverted survivor MES: %.1f" % inv_best)
print("real survivors above it, with checks:")
for r in sorted((r for r in rows if H.survivor(r, "real") and float(r["real_mes"]) > inv_best), key=lambda r: -float(r["real_mes"])):
    P, src = float(r["real_period"]), r["source"]
    me = pos.get(src)
    twins = [(o["source"], round(sep(me, pos[o["source"]]))) for o in rows if o["source"] != src and o["source"] in pos
             and abs(float(o["real_period"]) / P - 1) < 0.0005]
    d = 3600 * np.hypot((kra - me[0]) * np.cos(np.radians(me[1])), kde - me[1])
    near = [(koi[i]["kepoi_name"], koi[i]["koi_disposition"], round(float(d[i])), round(float(kp[i]), 4)) for i in np.argsort(d)[:3]]
    print("  %-30s P=%8.4f d dur=%.1f h depth=%6s ppm MES=%5s score=%3s" % (src, P, float(r["real_dur_h"]), r["real_depth_ppm"], r["real_mes"], r["real_score"]))
    print("      same period elsewhere (source, arcsec): %s" % (twins or "none"))
    print("      nearest KOIs (name, disposition, arcsec, period): %s" % near)
