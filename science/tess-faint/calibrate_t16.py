"""Calibration of the pixel test on the candidates of the T16 project (Roth et al. 2026, ApJS 284, 19): 11,143
multi-transit planet candidates in sectors 1 to 13, most of them fainter than T = 12.

The pixel test runs on each candidate in the sector where T16 found it, at its period, epoch and duration
(duration = qtran x period). The only ground truth is the TESS objects of interest (TOI) at the same period (within
2%, or half or double): CP and KP (confirmed and known planets) must stay on their star, FP and FA (false positives
and alarms) are often a neighbor. Few candidates fainter than T = 13.5 are TOIs, so there the table says how often
the test decides at all and on which side, not whether it is right.

Groups: planet (TOI disposition CP or KP), fp (FP or FA), toi (any other TOI disposition), new (not a TOI at that period).

    python calibrate_t16.py [--per-cell N] [--threads K] [--out NAME]   ->  <data>/t16/calibration.csv
    python calibrate_t16.py --summary

Inputs in <data>/t16/: `candidate_table.tab` (Harvard Dataverse, doi:10.7910/DVN/CWEUGW) and `toi.csv` (NASA Exoplanet
Archive, TAP table `toi`: toi, tid, tfopwg_disp, pl_orbper). A cell is a group and a magnitude bin; N candidates are
drawn at random in each (seed 2026, so a larger N keeps the smaller sample).
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pixel_test as P  # noqa: E402
import tf_common as T  # noqa: E402

DIR = os.path.join(T.DATA, "t16")
BINS = ((0, 12), (12, 13.5), (13.5, 14.5), (14.5, 15.5), (15.5, 99))
FIELDS = ["gaia", "tic", "sector", "tmag", "group", "toi_disp", "period", "dur_h", "events", "snr_peak", "offset_px", "offset_arcsec", "cent_ra", "cent_dec", "source", "source_sep", "source_g", "verdict"]


def same_period(p, q):
    return abs(p / q - 1) < 0.02 or abs(p / q - 2) < 0.04 or abs(p / q - 0.5) < 0.01


def load():
    cands = list(csv.DictReader(open(os.path.join(DIR, "candidate_table.tab")), delimiter="\t"))
    by_tic = collections.defaultdict(list)
    for t in csv.DictReader(open(os.path.join(DIR, "toi.csv"))):
        by_tic[t["tid"]].append(t)
    for c in cands:
        c["toi_disp"] = next((t["tfopwg_disp"] for t in by_tic.get(c["TICID"], ()) if t["pl_orbper"] and same_period(float(c["P_med"]), float(t["pl_orbper"]))), "")
        c["group"] = "planet" if c["toi_disp"] in ("CP", "KP") else "fp" if c["toi_disp"] in ("FP", "FA") else "toi" if c["toi_disp"] else "new"
    return cands


def mag_bin(tmag):
    return next(i for i, (a, b) in enumerate(BINS) if a <= tmag < b)


def test_one(c):
    p = float(c["P_med"])
    row = {"gaia": c["GAIAID"], "tic": c["TICID"], "tmag": c["Tess mag"], "real_period": p, "real_t0": float(c["Tc_med"]),
           "real_dur_h": float(c["qtran"]) * p * 24.0, "ra": c["ra"], "dec": c["dec"], "sector": c["Sector"],
           "real_depth_ppm": 1e6 * float(c["rp_rs_med"] or 0) ** 2}
    return P.one(row) | {"sector": c["Sector"], "group": c["group"], "toi_disp": c["toi_disp"]}


def run(a):
    cells, rng = collections.defaultdict(list), random.Random(2026)
    for c in load():
        cells[(c["group"], mag_bin(float(c["Tess mag"])))].append(c)
    pick = []
    for key in sorted(cells):
        rng.shuffle(cells[key])
        pick += cells[key][:a.per_cell]
    out = os.path.join(DIR, a.out)
    done = {x["gaia"] for x in csv.DictReader(open(out))} if os.path.exists(out) else set()
    todo = [c for c in pick if c["GAIAID"] not in done]
    print("%d cells, %d candidates drawn, %d to do" % (len(cells), len(pick), len(todo)), flush=True)
    new, start = not os.path.exists(out), time.time()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for i in range(0, len(todo), 20):
            with cf.ThreadPoolExecutor(a.threads) as ex:
                w.writerows(list(ex.map(test_one, todo[i:i + 20])))
            fh.flush()
            print("%d done, %.0f s" % (min(i + 20, len(todo)), time.time() - start), flush=True)


def summary(a):
    table = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(os.path.join(DIR, a.out))):
        v = "error" if r["verdict"].startswith("error") else r["verdict"]
        table[(r["group"], mag_bin(float(r["tmag"])))][v] += 1
    print("%-7s %-11s %4s %5s %6s %5s %10s %6s   %s" % ("group", "T", "n", "on", "blend", "off", "no signal", "other", "on / (on + off)"))
    for (g, b), c in sorted(table.items()):
        n, on, mix, off, none = sum(c.values()), c["on target"], c["blend"], c["off target"], c["no signal"]
        share = "%.0f%%" % (100.0 * on / (on + off)) if on + off else "-"
        print("%-7s %-11s %4d %5d %6d %5d %10d %6d   %s" % (g, "%g-%g" % BINS[b], n, on, mix, off, none, n - on - mix - off - none, share))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell", type=int, default=1, help="candidates drawn in each group and magnitude bin")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", default="calibration.csv")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    summary(a) if a.summary else run(a)
