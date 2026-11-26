"""The search on faint stars of one TESS sector: stars fainter than T = 13.5, down to T = 16, with the TESS-Gaia
light curves (TGLC, Han & Brandt 2023) read from the public MAST bucket. Each light curve is searched for a
repeating dip, the detection tests run, and the TESS model of the main network says eclipse or junk. Every star is
also run with its flux flipped: what noise alone gets called an eclipse.

    python hunt_faint.py --fields 120 [--per 50]      ->  results/hunt-faint.csv, results/hunt-faint.json

The stars are small groups of neighbors on the sky: each field is a random place in the file tree of the sector
(seed 2026), which is ordered by Gaia identifier, and the next `per` light curves from there. A call is a signal
that is detected and called an eclipse: an eclipse of a planet or of a star, to be checked on the pixels
(pixel_test.py), not a planet.
"""
import argparse
import concurrent.futures as cf
import csv
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_common as T  # noqa: E402

BUCKET = "https://stpubdata.s3.amazonaws.com/"
ROOT = "mast/hlsp/tglc/s%04d/" % T.SECTOR
OUT = os.path.join(T.RES, "hunt-faint.csv")
KEYS = ("period", "t0", "dur_h", "n", "depth_ppm", "mes", "odd_even", "secondary", "fail", "detected", "eclipse", "margin")
FIELDS = ["gaia", "tic", "ra", "dec", "tmag", "camccd", "points"] + ["%s_%s" % (tag, k) for tag in ("real", "inv") for k in KEYS]
FAINT = 13.5  # the limit of the Quick-Look Pipeline light curves


def get(url, timeout=90):
    return urllib.request.urlopen(urllib.request.Request(url, headers=T.UA), timeout=timeout).read()


def listing(prefix, after="", n=1000, dirs=False):
    q = {"list-type": "2", "prefix": prefix, "max-keys": str(n)}
    if after:
        q["start-after"] = after
    if dirs:
        q["delimiter"] = "/"
    text = get(BUCKET + "?" + urllib.parse.urlencode(q)).decode()
    return re.findall(r"<Prefix>([^<]+)</Prefix>" if dirs else r"<Key>([^<]+_llc\.fits)</Key>", text)


def fields(n, per, rng):
    """`n` random places of the sector, `per` light curve files after each"""
    keys = []
    chips = [d for d in listing(ROOT, dirs=True) if d != ROOT]
    tops = {c: [d for d in listing(c, dirs=True) if d != c] for c in chips}
    for _ in range(n):
        top = rng.choice(tops[rng.choice(chips)])
        keys += listing(top, after="%s%04d/%04d/" % (top, rng.integers(10000), rng.integers(10000)), n=per)
    return sorted(set(keys))


def one(key):
    row = {"gaia": re.search(r"gaiaid-(\d+)", key).group(1), "camccd": key.split("/")[4], "points": 0}
    try:
        from astropy.io import fits
        h = fits.open(io.BytesIO(get(BUCKET + key)))
        hd, d = h[0].header, h[1].data
        row.update(tic=hd.get("TICID", ""), ra=round(float(hd["RA_OBJ"]), 6), dec=round(float(hd["DEC_OBJ"]), 6), tmag=round(float(hd["TESSMAG"]), 3))
        if row["tmag"] <= FAINT:
            return row | {"real_fail": "not faint"}
        t, f = np.asarray(d["time"], float), np.asarray(d["cal_aper_flux"], float)
        ok = np.isfinite(t) & np.isfinite(f) & (d["TESS_flags"] == 0) & (d["TGLC_flags"] == 0) & (f > 0)
        t, f = t[ok], f[ok]
        row["points"] = len(t)
        if len(t) < 2000:
            return row | {"real_fail": "too few points"}
        f = f / np.median(f)
        ws, _, _ = T.load_model()
        for tag, sign in (("real", 1.0), ("inv", -1.0)):
            row.update({tag + "_" + k: v for k, v in T.judge(ws, t, f, sign).items()})
        if row.get("real_detected") and row.get("real_eclipse"):
            os.makedirs(os.path.join(T.DATA, "lc"), exist_ok=True)
            np.savez_compressed(os.path.join(T.DATA, "lc", row["gaia"] + ".npz"), t=t, f=f.astype(np.float32))
    except Exception as e:
        return row | {"real_fail": "error: %s" % str(e)[:60]}
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fields", type=int, default=120)
    p.add_argument("--per", type=int, default=50)
    p.add_argument("--chunk", type=int, default=200)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    keys = fields(a.fields, a.per, np.random.default_rng(2026))
    done = {r["gaia"] for r in csv.DictReader(open(OUT))} if os.path.exists(OUT) else set()
    todo = [k for k in keys if re.search(r"gaiaid-(\d+)", k).group(1) not in done]
    _, height, weightshash = T.load_model()
    print("sector %d: %d light curves in %d fields, %d to do; model of block %d" % (T.SECTOR, len(keys), a.fields, len(todo), height), flush=True)
    new, start = not os.path.exists(OUT), time.time()
    with open(OUT, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for c in range(0, len(todo), a.chunk):
            with cf.ProcessPoolExecutor(a.workers) as ex:
                out = list(ex.map(one, todo[c:c + a.chunk]))
            w.writerows(out)
            fh.flush()
            print("%d done, %.1f s" % (min(c + a.chunk, len(todo)), time.time() - start), flush=True)
    if not todo:
        return  # nothing ran: the timing of the last run stays
    with open(os.path.join(T.RES, "hunt-faint.json"), "w") as f:
        json.dump({"sector": T.SECTOR, "light_curves": len(todo), "seconds": round(time.time() - start, 1), "processes": a.workers,
                   "model_block": height, "weights_hash": weightshash}, f)


if __name__ == "__main__":
    main()
