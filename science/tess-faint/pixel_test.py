"""Pixel test of the calls of hunt_faint.py: for each call, an 11 x 11 pixel cutout of the TESS full-frame images
(TESScut at MAST, the sector of the hunt) around the star, and a difference image: the median of the pixels just
outside the eclipse minus the median inside it, event by event, then averaged. A signal that sits on the star puts
the peak of the difference image on the star; a signal of a neighbor puts it on the neighbor. A TESS pixel is
21 arcsec wide.

    python pixel_test.py     ->  results/pixel-faint.csv

Verdict: `no signal` when the peak of the difference image is below 4 sigma (the test says nothing), `on target`
when its centroid is within one pixel of the star, `off target` when it is farther, `no data` when the cutout falls
off the chip.
"""
import concurrent.futures as cf
import csv
import io
import math
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_common as T  # noqa: E402

URL = "https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=%s&dec=%s&y=11&x=11&units=px&sector=%d"
FIELDS = ["gaia", "tic", "tmag", "period", "dur_h", "ra", "dec", "events", "snr_peak", "offset_px", "offset_arcsec", "cent_ra", "cent_dec", "verdict"]
PIXEL = 21.0  # arcsec


def cutout(ra, dec, sector=T.SECTOR):
    """the cutout of one sector as an open FITS file, or None after three failed tries"""
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(URL % (ra, dec, sector), headers=T.UA), timeout=240).read()
            z = zipfile.ZipFile(io.BytesIO(raw))
            return fits.open(io.BytesIO(z.read(z.namelist()[0])))
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def frames(h):
    """time and pixel values of the good cadences; a pixel off the chip stays at zero in every frame"""
    d = h[1].data
    t, f, q = np.asarray(d["TIME"], float), np.asarray(d["FLUX"], float), np.asarray(d["QUALITY"])
    ok = np.isfinite(t) & (q == 0) & (np.isfinite(f).mean(axis=(1, 2)) > 0.9)
    return t[ok], np.nan_to_num(f[ok])


def one(r):
    out = {"gaia": r["gaia"], "tic": r["tic"], "tmag": r["tmag"], "period": r["real_period"], "dur_h": r["real_dur_h"], "ra": r["ra"], "dec": r["dec"]}
    h = cutout(r["ra"], r["dec"])
    if h is None:
        return out | {"verdict": "no cutout"}
    try:
        t, f = frames(h)
        if len(t) == 0:
            return out | {"verdict": "no data"}
        period, t0, dur = float(r["real_period"]), float(r["real_t0"]), float(r["real_dur_h"]) / 24.0
        diffs = []
        for k in range(int(math.floor((t.min() - t0) / period)), int(math.ceil((t.max() - t0) / period)) + 1):
            dt = t - (t0 + k * period)
            inside, outside = np.abs(dt) < 0.35 * dur, (np.abs(dt) > 0.6 * dur) & (np.abs(dt) < 2.0 * dur)
            if inside.sum() >= 5 and outside.sum() >= 10:
                diffs.append(np.median(f[outside], axis=0) - np.median(f[inside], axis=0))
        out["events"] = len(diffs)
        if len(diffs) < 3:
            return out | {"verdict": "too few events"}
        diffs = np.array(diffs)
        image = diffs.mean(axis=0)
        sigma = diffs.std(axis=0, ddof=1) / math.sqrt(len(diffs))
        snr = image / np.where(sigma > 0, sigma, np.inf)
        out["snr_peak"] = round(float(snr.max()), 1)
        wcs = WCS(h[2].header)
        tx, ty = wcs.all_world2pix([[float(r["ra"]), float(r["dec"])]], 0)[0]
        # the peak is the 3 x 3 block with the largest sum; the centroid is taken in the 5 x 5 block around it
        sums = np.array([[image[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].sum() for j in range(image.shape[1])] for i in range(image.shape[0])])
        pi, pj = np.unravel_index(np.argmax(sums), sums.shape)
        window = np.zeros_like(image)
        window[max(pi - 2, 0):pi + 3, max(pj - 2, 0):pj + 3] = 1.0
        weight = np.clip(image, 0, None) * window
        if weight.sum() <= 0:
            return out | {"verdict": "no signal"}
        yy, xx = np.indices(image.shape)
        cx, cy = float((weight * xx).sum() / weight.sum()), float((weight * yy).sum() / weight.sum())
        offset = math.hypot(cx - tx, cy - ty)
        ra, dec = wcs.all_pix2world([[cx, cy]], 0)[0]
        out.update(offset_px=round(offset, 2), offset_arcsec=round(offset * PIXEL, 1), cent_ra=round(float(ra), 5), cent_dec=round(float(dec), 5))
        out["verdict"] = "no signal" if snr.max() < 4 else ("on target" if offset <= 1.0 else "off target")
        return out
    except Exception as e:
        return out | {"verdict": "error: %s" % str(e)[:60]}


def main():
    rows = [x for x in csv.DictReader(open(os.path.join(T.RES, "hunt-faint.csv"))) if x.get("real_detected") == "1" and x.get("real_eclipse") == "1"]
    start = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, rows))
    with open(os.path.join(T.RES, "pixel-faint.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(res)
    print("%d calls, %.0f s" % (len(res), time.time() - start))
    for v in sorted({x["verdict"] for x in res}):
        print("%-12s %d" % (v, sum(1 for x in res if x["verdict"] == v)))


if __name__ == "__main__":
    main()
