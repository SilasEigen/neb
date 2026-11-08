#!/usr/bin/env python3
"""Hunt: Kepler Bonus background M dwarfs, searched and vetted like the Kepler pipeline, then judged by the Kepler
model of the chain. Every star is also run with its flux inverted (a control: what noise alone lets through).

    python hunt.py --start 300 --n 2000 [--chunk 250] [--workers 8]

Stars are taken in a fixed random order (seed 2026, ranks start .. start+n). Light curves are downloaded by chunks and
deleted after use, except for the stars that pass every test with a model score of at least 4. Results are appended
to results/hunt.csv (a rerun skips the stars already there). A survivor is a lead for astronomers, not a planet:
no pixel-level test (centroids, Gaia neighbours) is done here.
"""
import argparse
import concurrent.futures as cf
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kb_common as K  # noqa: E402
import vet  # noqa: E402

OUT = os.path.join(K.RESULTS, "hunt.csv")
SCORE_MIN = 4
FIELDS = ["source", "points"] + ["%s_%s" % (tag, k) for tag in ("real", "inv") for k in (
    "period", "t0", "dur_h", "n_transits", "depth_ppm", "mes", "odd_even_sigma", "secondary_sigma", "fail", "answer", "score")]


def one(job):
    src, path, model_path = job
    try:
        t, f = K.read_lc(path)
    except Exception as e:
        return {"source": src, "points": -1, "real_fail": "unreadable: %s" % str(e)[:60]}
    if len(t) < 3000:
        return {"source": src, "points": len(t), "real_fail": "too few points"}
    w1, w2, _, _ = K.load_model(model_path)
    row = {"source": src, "points": len(t)}
    for tag, flux in (("real", f), ("inv", -f)):
        tc, fc = vet.clean(t, flux)
        p, t0, dur = vet.search(tc, fc)
        v = vet.vet(tc, fc, p, t0, dur)
        o = K.outputs(w1, w2, K.kepler_views(tc, fc, p, t0, dur * 24)[None])[0]
        row.update({tag + "_period": round(p, 5), tag + "_t0": round(t0, 4), tag + "_dur_h": round(dur * 24, 2),
                    tag + "_answer": int(o.argmax()), tag + "_score": int(o[1] - o[0])})
        row.update({tag + "_" + k: v.get(k, "") for k in ("n_transits", "depth_ppm", "mes", "odd_even_sigma", "secondary_sigma", "fail")})
    return row


def survivor(r, tag):
    return r.get(tag + "_fail", "x") == "" and int(r.get(tag + "_score") or -99) >= SCORE_MIN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=300)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", default=K.MODEL)
    a = p.parse_args()

    bg, _, _ = K.background_mdwarfs()
    order = np.random.default_rng(2026).permutation(len(bg))
    batch = [bg[i] for i in order[a.start:a.start + a.n]]
    done = {r["source"] for r in csv.DictReader(open(OUT))} if os.path.exists(OUT) else set()
    todo = [b for b in batch if b[1] not in done]
    print("ranks %d..%d: %d stars, %d already done" % (a.start, a.start + a.n, len(batch), len(batch) - len(todo)), flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for c in range(0, len(todo), a.chunk):
            part = todo[c:c + a.chunk]
            paths = K.fetch_many(part, workers=8)
            jobs = [(s, paths[s], a.model) for _, s in part if paths.get(s)]
            with cf.ProcessPoolExecutor(a.workers) as ex:
                rows = list(ex.map(one, jobs))
            for r in rows:
                w.writerow(r)
                if not survivor(r, "real"):
                    try:
                        os.remove(paths[r["source"]])
                    except OSError:
                        pass
            fh.flush()
            print("chunk %d: %d stars done, %d real survivors so far" % (c // a.chunk + 1, min(c + a.chunk, len(todo)),
                  sum(survivor(r, "real") for r in rows)), flush=True)
    summary()


def summary():
    rows = list(csv.DictReader(open(OUT)))
    ok = [r for r in rows if r.get("real_period")]
    print("\nstars searched: %d (%d skipped)" % (len(ok), len(rows) - len(ok)))
    for tag in ("real", "inv"):
        fails = {}
        for r in ok:
            fails[r[tag + "_fail"] or "PASS"] = fails.get(r[tag + "_fail"] or "PASS", 0) + 1
        surv = [r for r in ok if survivor(r, tag)]
        print("%s: vetting %s; pass + model score >= %d: %d" % (tag, dict(sorted(fails.items(), key=lambda x: -x[1])), SCORE_MIN, len(surv)))
    pos = K.mstar_positions()
    koi = [k for k in csv.DictReader(open(K.cached(K.KOI_URL, "koi-cumulative-radec.csv"))) if k["koi_period"]]
    kp = np.array([float(k["koi_period"]) for k in koi])
    kra, kde = np.array([float(k["ra"]) for k in koi]), np.array([float(k["dec"]) for k in koi])
    allp = np.array([float(r["real_period"]) for r in ok])
    print("\nreal survivors (leads, not planets):")
    for r in sorted((r for r in ok if survivor(r, "real")), key=lambda r: -float(r["real_mes"] or 0)):
        P = float(r["real_period"])
        ra, de = pos.get(r["source"], (np.nan, np.nan))
        sep = 3600 * np.hypot((kra - ra) * np.cos(np.radians(de)), kde - de)
        match = [koi[i]["kepoi_name"] + " " + koi[i]["koi_disposition"] for i in np.flatnonzero(sep < 100)
                 if any(abs(P / kp[i] - k) < 0.001 * k for k in (1, 0.5, 2, 1 / 3, 3))]
        pile = int((np.abs(allp / P - 1) < 0.0005).sum()) - 1
        print("  %-30s P=%8.4f d dur=%4.1f h depth=%6s ppm MES=%5s n=%3s odd/even=%4s sec=%5s score=%3s  %s%s" % (
            r["source"], P, float(r["real_dur_h"]), r["real_depth_ppm"], r["real_mes"], r["real_n_transits"],
            r["real_odd_even_sigma"], r["real_secondary_sigma"], r["real_score"],
            ("NEAR KOI SAME PERIOD: " + ", ".join(match)) if match else "",
            ("  same period in %d other stars" % pile) if pile > 0 else ""))


if __name__ == "__main__":
    main()
