#!/usr/bin/env python3
"""Recovery figure: a confirmed Kepler planet of a test star (never seen by the chain in training), found blind by
the search in its Kepler Bonus light curve, folded, with the answer of the chain's model.

    NEB_KEPLER_CACHE=DIR NEB_KEPLER_TEST=FILE python figure.py      -> results/figure-kepler-recovered.png

Run calib.py first: it downloads the light curves of the test hosts.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib  # noqa: E402
import kb_common as K  # noqa: E402
import vet  # noqa: E402

SURFACE, INK, INK2, BLUE, RAW = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#b9b8b2"


def pick():
    names = {}
    for r in csv.DictReader(open(K.cached(K.CONFIRMED_URL, "koi-confirmed.csv"))):
        names.setdefault(int(r["kepid"]), []).append((float(r["koi_period"]), r["kepler_name"], float(r["koi_prad"] or "nan")))
    rows = {calib.key(r): r for r in calib.test_tces()}
    w1, w2, height, wh = K.load_model()
    found = []
    for x in csv.DictReader(open(os.path.join(K.RESULTS, "calib.csv"))):
        r = rows[x["tce"]]
        kid, P = int(r["kepid"]), float(r["tce_period"])
        if x["label"] != "1" or P >= 20:
            continue
        name = next(((n, rad) for q, n, rad in names.get(kid, []) if abs(q / P - 1) < 0.002), None)
        if not name:
            continue
        t, f = vet.clean(*K.read_lc(os.path.join(K.LC_DIR, K.lc_name("kic-%09d" % kid))))
        bp, bt0, bd = vet.search(t, f)
        if abs(bp / P - 1) > 0.002:
            continue
        v = vet.vet(t, f, bp, bt0, bd)
        o = K.outputs(w1, w2, K.kepler_views(t, f, bp, bt0, bd * 24)[None])[0]
        found.append({"name": name[0], "prad": name[1], "kid": kid, "P_nasa": P, "P": bp, "t0": bt0, "dur": bd,
                      "mes": v.get("mes", 0), "depth": v.get("depth_ppm", 0), "answer": int(o.argmax()), "score": int(o[1] - o[0])})
        print(found[-1], flush=True)
    good = [c for c in found if c["answer"] == 1 and c["prad"] < 6 and c["mes"] >= 15]
    return max(good or found, key=lambda c: c["mes"]), height, wh


def draw(c, height, wh, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, f = vet.clean(*K.read_lc(os.path.join(K.LC_DIR, K.lc_name("kic-%09d" % c["kid"]))))
    P, t0, dur = c["P"], c["t0"], c["dur"]
    hours = (np.mod(t - t0 + P / 2, P) - P / 2) * 24
    win = max(4 * dur * 24, 8.0)
    m = np.abs(hours) < win
    x, y = hours[m], 1e6 * f[m]
    edges = np.linspace(-win, win, 61)
    mid = (edges[1:] + edges[:-1]) / 2
    k = np.digitize(x, edges) - 1
    med = np.array([np.median(y[k == i]) if (k == i).sum() > 5 else np.nan for i in range(len(mid))])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    lo, hi = np.nanpercentile(y, 0.5), np.nanpercentile(y, 99.5)
    ax.scatter(x, y, s=2, color=RAW, linewidths=0, rasterized=True)
    ax.plot(mid, med, color=BLUE, lw=2, marker="o", ms=4, mec=SURFACE, mew=1)
    ax.set_ylim(min(lo, np.nanmin(med)) - 0.15 * c["depth"], hi)
    ax.set_xlim(-win, win)
    ax.set_xlabel("hours from mid-transit")
    ax.set_ylabel("brightness change (ppm)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d9d8d3")
    ax.grid(axis="y", color="#ecebe6", lw=0.8)
    ax.set_axisbelow(True)
    fig.text(0.07, 0.95, "%s, found blind in its Kepler Bonus light curve" % c["name"], fontsize=15, weight="bold", color=INK)
    fig.text(0.07, 0.895, "Search found P = %.4f d (NASA: %.4f d). NEB chain model, block %d: planet (score %+d)."
             % (c["P"], c["P_nasa"], height, c["score"]), fontsize=10.5, color=INK2)
    fig.text(0.07, 0.855, "The model never saw this star during training. Gray: every measurement. Blue: median per bin.",
             fontsize=10.5, color=INK2)
    fig.text(0.07, 0.015, "Data: Kepler Bonus PSF light curves (Martinez-Palomera, Hedges & Dotson 2023), NASA Kepler mission, MAST.",
             fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.81, bottom=0.18)
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out, "weights hash", wh[:16])


if __name__ == "__main__":
    c, height, wh = pick()
    print("picked", c)
    draw(c, height, wh, os.path.join(K.RESULTS, "figure-kepler-recovered.png"))
