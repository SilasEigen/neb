#!/usr/bin/env python3
"""Calibration: does the Kepler model of the chain still work on Kepler Bonus (PSF) light curves?

    NEB_KEPLER_CACHE=DIR NEB_KEPLER_TEST=FILE python calib.py [--n 60]

Takes DR24 TCEs of the stars kept OUT of the chain (the 15 % test stars of tools/make_datasets.py, split replayed
with the same seed and checked against kepler-test-q8.bin), downloads the KBonus light curve of each host,
builds the 784 values with the same views, and compares the answers of the model on the NASA DV light curve (the
training kind) and on the Kepler Bonus light curve of the same signal. Writes results/calib.csv.
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_common as K  # noqa: E402

KEP = os.environ.get("NEB_KEPLER_CACHE", os.path.join(K.DATA, "kepler"))
TEST = os.environ.get("NEB_KEPLER_TEST", os.path.join(K.DATA, "kepler-test-q8.bin"))
SEED = 2026


def name(r):
    return "kplr%09d_q1_q17_dr24_tce_%02d_dvt_lc.tbl" % (int(r["kepid"]), int(r["tce_plnt_num"]))


def key(r):
    return "%09d_%02d" % (int(r["kepid"]), int(r["tce_plnt_num"]))


def test_tces():
    """the TCEs of the test stars, replaying the random draws of make_datasets.kepler()"""
    rows = [r for r in csv.DictReader(open(os.path.join(KEP, "dr24_tce.csv"))) if r["av_training_set"] in ("PC", "AFP", "NTP")]
    urls = set(re.findall(r"wget -O '([^']+)'", open(os.path.join(KEP, "dv24_wget.bat")).read()))
    rng = np.random.default_rng(SEED)
    pc = [r for r in rows if r["av_training_set"] == "PC"]
    fp = [r for r in rows if r["av_training_set"] != "PC"]
    pick = [pc[i] for i in rng.permutation(len(pc))] + [fp[i] for i in rng.permutation(len(fp))]
    jobs = [r for r in pick if name(r) in urls]
    keys = sorted(key(r) for r in jobs)
    stars = sorted({k[:9] for k in keys})
    test = set(stars[i] for i in rng.permutation(len(stars))[:int(0.15 * len(stars))])
    return [r for r in jobs if key(r)[:9] in test]


def check_split(tces):
    """the views of the replayed test TCEs must be exactly the records of kepler-test-q8.bin"""
    raw = np.fromfile(TEST, np.uint8)
    n = int.from_bytes(raw[4:8].tobytes(), "little")
    rec = raw[8:8 + n * 785].reshape(n, 785)
    ours = np.stack([np.append(np.load(os.path.join(KEP, "views", key(r) + ".npy")), int(r["av_training_set"] == "PC")) for r in tces])
    a = sorted(map(bytes, rec))
    b = sorted(map(bytes, ours.astype(np.uint8)))
    return n, len(tces), a == b


def auc(score, label):
    """probability that a random planet scores above a random false positive (ties count one half)"""
    s, y = np.asarray(score, float), np.asarray(label)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return gt / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=60, help="planets and false positives drawn (each)")
    p.add_argument("--model", default=K.MODEL)
    a = p.parse_args()

    tces = test_tces()
    n_file, n_ours, same = check_split(tces)
    print("test split replayed: %d TCEs, test file %d records, identical: %s" % (n_ours, n_file, same), flush=True)
    if not same:
        raise SystemExit("the replayed split is not the chain's test split; stop")

    rng = np.random.default_rng(7)
    pcs = [r for r in tces if r["av_training_set"] == "PC"]
    fps = [r for r in tces if r["av_training_set"] != "PC"]
    sample = [pcs[i] for i in rng.permutation(len(pcs))[:a.n]] + [fps[i] for i in rng.permutation(len(fps))[:a.n]]
    hosts = sorted({"%09d" % int(r["kepid"]) for r in sample})
    paths = K.fetch_many([(h[:4] + "/" + h, "kic-" + h) for h in hosts])
    print("KBonus light curves: %d of %d hosts downloaded" % (sum(v is not None for v in paths.values()), len(hosts)), flush=True)

    w1, w2, height, wh = K.load_model(a.model)
    out_rows, lc_cache = [], {}
    for r in sample:
        h = "%09d" % int(r["kepid"])
        path = paths.get("kic-" + h)
        if path is None:
            continue
        if h not in lc_cache:
            try:
                lc_cache[h] = K.read_lc(path)
            except Exception as e:  # a broken file skips the host
                print("skip %s: %s" % (h, e))
                lc_cache[h] = None
        if lc_cache[h] is None or len(lc_cache[h][0]) < 1000:
            continue
        t, f = lc_cache[h]
        period, t0, dur = float(r["tce_period"]), float(r["tce_time0bk"]), float(r["tce_duration"])
        v_kb = K.kepler_views(t, f, period, t0, dur)
        v_dv = np.load(os.path.join(KEP, "views", key(r) + ".npy"))
        o = K.outputs(w1, w2, np.stack([v_dv, v_kb]))
        lab = int(r["av_training_set"] == "PC")
        out_rows.append({"tce": key(r), "label": lab, "period": period, "dv_answer": int(o[0].argmax()),
                         "kb_answer": int(o[1].argmax()), "dv_score": int(o[0, 1] - o[0, 0]), "kb_score": int(o[1, 1] - o[1, 0])})

    y = np.array([x["label"] for x in out_rows])
    for src in ("dv", "kb"):
        ans = np.array([x[src + "_answer"] for x in out_rows])
        sc = np.array([x[src + "_score"] for x in out_rows])
        print("%s: accuracy %.1f %% (%d signals), planets found %d/%d, false positives rejected %d/%d, AUC %.3f" % (
            "NASA DV light curves " if src == "dv" else "Kepler Bonus PSF     ", 100 * (ans == y).mean(), len(y),
            ((ans == 1) & (y == 1)).sum(), (y == 1).sum(), ((ans != 1) & (y == 0)).sum(), (y == 0).sum(), auc(sc, y)))
    agree = np.mean([x["dv_answer"] == x["kb_answer"] for x in out_rows])
    print("same answer on both light curves: %.1f %%; model block %d, weights hash %s..." % (100 * agree, height, wh[:16]))
    with open(os.path.join(K.RESULTS, "calib.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
