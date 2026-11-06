#!/usr/bin/env python3
"""Builds the dataset set file of the testnet of protocol 3: slot 0 Kepler transit signals, slot 1 retinal OCT scans.

    python -I tools/make_datasets.py --cache DIR --out /path/outside/the/tree/neb-datasets.bin

The file is `NMDS || LE32(2) || kepler || retina` (spec/shares.md section 11.6), each part an NMD1 dataset. The
script also writes, next to `--out`, the samples kept out of the chain: `kepler-test-q8.bin` (15 % of the stars) and
`retina-test-q8.bin` (the OCTMNIST test split). It prints the SHA-256 of the set file; `neb-node --dataset-hash FILE`
prints the BLAKE3 hash the network pins. Everything it downloads goes to `--cache`, never into the source tree.

Slot 0, Kepler (NASA Exoplanet Archive, public):
  - the Q1-Q17 DR24 TCE table: rowid, kepid, tce_plnt_num, tce_period, tce_time0bk, tce_duration and the label
    av_training_set (PC planet candidate, AFP astrophysical false positive, NTP non-transiting phenomenon, UNK);
  - the DR24 DV light curve of each labelled TCE (`*_dvt_lc.tbl`, about 8.6 MB of text each, 15,737 files, about
    135 GB in all; each is read once and deleted, only its 784 values stay in the cache). The column used is
    INIT_FLUX_PL, the detrended flux; the fitted transit model in the same file is not used.
  The curve is folded on the TCE period; a global view of 583 bins over the orbit and a local view of 201 bins over
  +-2 transit durations are the medians of the flux in each bin (as in Shallue and Vanderburg 2018). Each view is set
  to median 0 and minimum -1, and a value is the depth of the dip, round(-7 v) clipped to 0..7. Label 1 is a planet
  candidate, 0 a false positive (AFP or NTP). The split is by star: 15 % of the stars, drawn with a fixed seed, are
  in the test file only.

Slot 1, retina (OCTMNIST, MedMNIST v2, CC BY 4.0, https://zenodo.org/records/10519652, file octmnist.npz,
MD5 c68d92d5b585d8d81f7112f81e2d0842): 28 x 28 gray scans, labels 0 choroidal neovascularization, 1 diabetic macular
edema, 2 drusen, 3 normal. The chain holds the train and validation splits with the same number of scans per label
(8,616), drawn with a fixed seed; a value is clip((x - 30) >> 4, 0, 7) of the 8-bit pixel x, so the background stays
0, as the fixed shifts of the network expect.
"""
import argparse
import concurrent.futures as cf
import csv
import hashlib
import os
import re
import urllib.request

import numpy as np

UA = {"user-agent": "neb-make-datasets"}
TCE_URL = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+rowid,kepid,tce_plnt_num,tce_period,"
           "tce_time0bk,tce_duration,av_training_set+from+q1_q17_dr24_tce&format=csv")
DV_LIST_URL = "https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/Kepler_TCE_DV_DR24_wget.bat"
OCT_URL = "https://zenodo.org/records/10519652/files/octmnist.npz?download=1"
OCT_MD5 = "c68d92d5b585d8d81f7112f81e2d0842"
NG, NL = 583, 201
SEED = 2026


def download(url, path):
    req = urllib.request.Request(url.replace("http://", "https://").replace(":80/", "/"), headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(path + ".part", path)


def binned(phase, flux, lo, hi, n):
    """median of the flux in n equal bins of [lo, hi); empty bins take the median of the whole view"""
    k = np.floor((phase - lo) / (hi - lo) * n).astype(int)
    ok = (k >= 0) & (k < n)
    k, f = k[ok], flux[ok]
    out = np.full(n, np.nan)
    order = np.argsort(k, kind="stable")
    k, f = k[order], f[order]
    edges = np.searchsorted(k, np.arange(n + 1))
    for i in range(n):
        if edges[i + 1] > edges[i]:
            out[i] = np.median(f[edges[i]:edges[i + 1]])
    fill = np.nanmedian(out) if np.isfinite(out).any() else 0.0
    return np.where(np.isfinite(out), out, fill)


def depth(v):
    v = v - np.median(v)
    m = -v.min()
    if m <= 0:
        return np.zeros(v.size, np.uint8)
    return np.clip(np.rint(-7 * v / m), 0, 7).astype(np.uint8)


def kepler_views(time, flux, period, t0, dur_h):
    phase = np.mod(time - t0 + period / 2, period) - period / 2
    g = binned(phase, flux, -period / 2, period / 2, NG)
    w = min(2 * dur_h / 24.0, period / 2)
    return np.concatenate([depth(g), depth(binned(phase, flux, -w, w, NL))])


def kepler_one(job, views):
    key, url, period, t0, dur = job
    out = os.path.join(views, key + ".npy")
    if os.path.exists(out):
        return key, "cached"
    raw = os.path.join(views, key + ".tbl")
    try:
        download(url, raw)
        data = np.loadtxt((line for line in open(raw) if line[:1] not in "\\|"), usecols=(0, 2))
        data = data[np.isfinite(data).all(axis=1)]
        np.save(out, kepler_views(data[:, 0], data[:, 1], period, t0, dur))
        return key, "ok"
    except Exception as e:  # a missing or broken file skips the TCE
        return key, "error %s" % e
    finally:
        if os.path.exists(raw):
            os.remove(raw)


def nmd1(values, labels):
    rec = np.zeros((len(labels), 785), np.uint8)
    rec[:, :784] = values.reshape(len(labels), 784)
    rec[:, 784] = labels
    return b"NMD1" + len(labels).to_bytes(4, "little") + rec.tobytes()


def kepler(cache, workers):
    os.makedirs(os.path.join(cache, "views"), exist_ok=True)
    table, dv = os.path.join(cache, "dr24_tce.csv"), os.path.join(cache, "dv24_wget.bat")
    for url, path in ((TCE_URL, table), (DV_LIST_URL, dv)):
        if not os.path.exists(path):
            download(url, path)
    urls = {m.group(1): m.group(2) for m in re.finditer(r"wget -O '([^']+)' '([^']+)'", open(dv).read())}
    rows = [r for r in csv.DictReader(open(table)) if r["av_training_set"] in ("PC", "AFP", "NTP")]
    rng = np.random.default_rng(SEED)
    pc = [r for r in rows if r["av_training_set"] == "PC"]
    fp = [r for r in rows if r["av_training_set"] != "PC"]
    pick = [pc[i] for i in rng.permutation(len(pc))] + [fp[i] for i in rng.permutation(len(fp))]
    key = lambda r: "%09d_%02d" % (int(r["kepid"]), int(r["tce_plnt_num"]))
    jobs = []
    for r in pick:
        name = "kplr%09d_q1_q17_dr24_tce_%02d_dvt_lc.tbl" % (int(r["kepid"]), int(r["tce_plnt_num"]))
        if name in urls:
            jobs.append((key(r), urls[name], float(r["tce_period"]), float(r["tce_time0bk"]), float(r["tce_duration"])))
    print("kepler: %d labelled TCEs, %d with a light curve" % (len(pick), len(jobs)), flush=True)
    status = {}
    views = os.path.join(cache, "views")
    with cf.ThreadPoolExecutor(workers) as ex:
        for n, (k, st) in enumerate(ex.map(lambda j: kepler_one(j, views), jobs), 1):
            status[k] = st
            if n % 500 == 0 or st.startswith("error"):
                print("kepler: %d/%d %s %s" % (n, len(jobs), k, st), flush=True)
    label = {key(r): int(r["av_training_set"] == "PC") for r in pick}
    keys = sorted(k for k, st in status.items() if not st.startswith("error"))
    stars = sorted({k[:9] for k in keys})
    test_stars = set(stars[i] for i in rng.permutation(len(stars))[:int(0.15 * len(stars))])
    parts = []
    for part in ([k for k in keys if k[:9] not in test_stars], [k for k in keys if k[:9] in test_stars]):
        part = [part[i] for i in rng.permutation(len(part))]
        values = np.stack([np.load(os.path.join(views, k + ".npy")) for k in part])
        parts.append(nmd1(values, np.array([label[k] for k in part], np.uint8)))
    return parts


def retina(cache):
    src = os.path.join(cache, "octmnist.npz")
    if not os.path.exists(src):
        download(OCT_URL, src)
    if hashlib.md5(open(src, "rb").read()).hexdigest() != OCT_MD5:
        raise SystemExit("%s does not have the published MD5" % src)
    d = np.load(src)
    x = np.concatenate([d["train_images"], d["val_images"]]).astype(np.int32)
    y = np.concatenate([d["train_labels"], d["val_labels"]]).ravel()
    rng = np.random.default_rng(1)
    n = np.bincount(y).min()
    idx = rng.permutation(np.concatenate([rng.permutation(np.flatnonzero(y == c))[:n] for c in range(4)]))
    value = lambda a: np.clip((a - 30) >> 4, 0, 7).astype(np.uint8)
    test = d["test_images"].astype(np.int32)
    return [nmd1(value(x[idx]), y[idx].astype(np.uint8)), nmd1(value(test), d["test_labels"].ravel().astype(np.uint8))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="directory for the downloads and the Kepler views")
    p.add_argument("--out", required=True, help="the set file; the test files are written next to it")
    p.add_argument("--workers", type=int, default=4, help="Kepler light curves downloaded at once")
    a = p.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    k_train, k_test = kepler(a.cache, a.workers)
    r_train, r_test = retina(a.cache)
    data = b"NMDS" + (2).to_bytes(4, "little") + k_train + r_train
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(data)
    here = os.path.dirname(os.path.abspath(a.out))
    for name, part in (("kepler-test-q8.bin", k_test), ("retina-test-q8.bin", r_test)):
        with open(os.path.join(here, name), "wb") as f:
            f.write(part)
    print("%s: %d bytes, SHA-256 %s" % (a.out, len(data), hashlib.sha256(data).hexdigest()))


if __name__ == "__main__":
    main()
