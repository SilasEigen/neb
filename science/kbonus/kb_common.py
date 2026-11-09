"""Shared pieces of the Kepler Bonus search: paths, downloads, light curves, the views of the chain and its model.

The views (binned, depth, kepler_views) are copied unchanged from tools/make_datasets.py, so a signal found in a
Kepler Bonus light curve becomes the same 784 values as the training samples of slot 0.

Paths, from environment variables (all optional):
  NEB_KBONUS_DATA   downloads (light curves, catalogs, target pixel files); default: data/ next to this file
  NEB_KEPLER_CACHE  the --cache directory of tools/make_datasets.py (dr24_tce.csv, dv24_wget.bat, views/); calib.py
  NEB_KEPLER_TEST   kepler-test-q8.bin, written next to --out by tools/make_datasets.py; calib.py
"""
import concurrent.futures as cf
import csv
import os
import re
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("NEB_KBONUS_DATA", os.path.join(HERE, "data"))
LC_DIR = os.path.join(DATA, "lc")
RESULTS = os.path.join(HERE, "results")
MODEL = os.path.join(RESULTS, "neb-kepler-block1428.npz")
UA = {"user-agent": "neb-science"}
HLSP = "https://archive.stsci.edu/hlsps/kbonus-bkg"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="
TARGETS_URL = TAP + "select+distinct+kepid+from+keplerstellar"
KOI_URL = TAP + "select+kepid,kepoi_name,koi_disposition,koi_period,ra,dec+from+cumulative"
CONFIRMED_URL = TAP + "select+kepid,kepoi_name,kepler_name,koi_disposition,koi_period,koi_prad+from+cumulative+where+koi_disposition=%27CONFIRMED%27"
NG, NL = 583, 201


# --- views, identical to tools/make_datasets.py ------------------------------------------------------------------
def binned(phase, flux, lo, hi, n):
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


# --- the model of the chain: the integer forward pass of spec/shares.md ------------------------------------------
def load_model(path=MODEL):
    z = np.load(path)
    return z["w1"].astype(np.int64), z["w2"].astype(np.int64), int(z["height"]), str(z["weightshash"])


def outputs(w1, w2, px):
    r = lambda x, s: (x + (1 << (s - 1))) >> s
    a1 = np.clip(r(np.asarray(px, np.int64) @ w1.T, 5), -8, 7)
    return np.clip(r(a1 @ w2.T, 5), -8, 7)


# --- downloads -----------------------------------------------------------------------------------------------------
def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(path + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(path + ".part", path)


def cached(url, name):
    """a file under DATA, downloaded once"""
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        download(url, path)
    return path


def lc_name(source):
    """source: 'kic-000757039' or 'gaia-dr3-...' as in the Kepler Bonus catalogs"""
    return "hlsp_kbonus-bkg_kepler_kepler_%s_kepler_v1.0_lc.fits" % source


def fetch(rel_dir, source):
    """one Kepler Bonus light curve into LC_DIR (cached); the local path, or None when it cannot be downloaded"""
    path = os.path.join(LC_DIR, lc_name(source))
    if not os.path.exists(path):
        try:
            download("%s/lcs/%s/%s" % (HLSP, rel_dir, lc_name(source)), path)
        except Exception:
            return None
    return path


def fetch_many(items, workers=6):
    """items: (rel_dir, source) pairs; returns {source: path or None}"""
    with cf.ThreadPoolExecutor(workers) as ex:
        return dict(zip([s for _, s in items], ex.map(lambda it: fetch(*it), items)))


def file_list():
    """the bulk download script of Kepler Bonus: one line per light curve, with its directory"""
    return cached(HLSP + "/hlsp_kbonus-bkg_kepler_lightcurves_all_kepler_v1_bulkdl.sh", "bulkdl.sh")


LC_PATTERN = re.compile(r"--output 'kbonus-bkg/lcs/([^']+)/hlsp_kbonus-bkg_kepler_kepler_([^_]+)_kepler_v1\.0_lc\.fits'")


def background_mdwarfs():
    """(rel_dir, source) of the M dwarfs of the Kepler Bonus catalog that were not Kepler targets"""
    from astropy.io import fits
    targets = {int(r["kepid"]) for r in csv.DictReader(open(cached(TARGETS_URL, "kepler-targets.csv")))}
    cat = cached(HLSP + "/catalogs/hlsp_kbonus-bkg_kepler_kepler_mstars-catalog_kepler_v1.0_cat.fits", "mstars-cat.fits")
    mst = {s.strip() for s in np.array(fits.open(cat)[1].data["fname"]).astype(str)}
    rel = {}
    with open(file_list()) as f:
        for line in f:
            m = LC_PATTERN.search(line)
            if m and m.group(2) in mst:
                rel[m.group(2)] = m.group(1)
    out = [(d, s) for s, d in rel.items() if s.startswith("gaia-") or int(s[4:]) not in targets]
    return sorted(out, key=lambda x: x[1]), len(targets), len(mst)


def mstar_positions():
    """{source: (ra, dec)} of the Kepler Bonus M dwarf catalog"""
    from astropy.io import fits
    cat = fits.open(cached(HLSP + "/catalogs/hlsp_kbonus-bkg_kepler_kepler_mstars-catalog_kepler_v1.0_cat.fits", "mstars-cat.fits"))[1].data
    return {s.strip(): (float(a), float(d)) for s, a, d in zip(np.array(cat["fname"]).astype(str), cat["ra"], cat["dec"])}


# --- light curves --------------------------------------------------------------------------------------------------
def read_lc(path, window=2.0, quarters=False):
    """time and relative flux (0 = median) of a Kepler Bonus light curve: good cadences (Kepler quality mask 175),
    each quarter divided by a running median of `window` days, computed on 0.25-day bin medians and interpolated.
    With quarters=True, also the Kepler quarter of each point."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        d = h["LIGHTCURVE_STITCHED"].data
        t, f, q = (np.asarray(d[c], float) for c in ("TIME", "FLUX", "QUARTER"))
        qual = np.asarray(d["SAP_QUALITY"], np.int64)
        # some background sources have no PSF flux: fall back to the aperture flux
        if np.isfinite(f).mean() < 0.5:
            f = np.asarray(d["SAP_FLUX"], float)
    # 175 = attitude tweak, safe mode, coarse point, earth point, desaturation, manual exclude
    ok = np.isfinite(t) & np.isfinite(f) & ((qual & 175) == 0) & (f > 0)
    t, f, q = t[ok], f[ok], q[ok]
    rel = np.empty_like(f)
    for quarter in np.unique(q):
        m = q == quarter
        tq, fq = t[m], f[m] / np.median(f[m])
        edges = np.arange(tq.min(), tq.max() + 0.25, 0.25)
        idx = np.digitize(tq, edges)
        centers, meds = [], []
        for i in np.unique(idx):
            s = idx == i
            centers.append(tq[s].mean())
            meds.append(np.median(fq[s]))
        centers, meds = np.array(centers), np.array(meds)
        half = int(round(window / 0.25 / 2))
        trend = np.array([np.median(meds[max(0, i - half):i + half + 1]) for i in range(meds.size)])
        rel[m] = fq / np.interp(tq, centers, trend) - 1.0
    return (t, rel, q.astype(int)) if quarters else (t, rel)
