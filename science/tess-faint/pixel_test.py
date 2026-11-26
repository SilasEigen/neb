"""Pixel test of the calls of hunt_faint.py: for each call, an 11 x 11 pixel cutout of the TESS full-frame images
(TESScut at MAST, the sector of the hunt) around the star, and a difference image: the median of the pixels just
outside the eclipse minus the median inside it, event by event, then averaged. A signal that sits on the star puts
the peak of the difference image on the star; a signal of a neighbor puts it on the neighbor. A TESS pixel is
21 arcsec wide.

    python pixel_test.py [--detections] [--hunt CSV] [--out CSV]     ->  results/pixel-faint.csv

Verdict: `no signal` when the peak of the difference image is below 4 sigma (the test says nothing), `on target`
when its centroid is within one pixel of the star and no other Gaia star bright enough to make the dip is nearer to
the centroid, `blend` when such a star is nearer but the target is within a quarter of a pixel of being as near (TESS
cannot tell them apart), `off target` otherwise (`source` names the nearest such star), `no data` when the cutout
falls off the chip.
"""
import argparse
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
FIELDS = ["gaia", "tic", "tmag", "period", "dur_h", "ra", "dec", "model", "events", "snr_peak", "offset_px", "offset_arcsec", "cent_ra", "cent_dec", "source", "source_sep", "source_g", "verdict"]
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


def source(r, cra, cdec):
    """the Gaia DR3 star nearest to the centroid among those bright enough to make the dip: {source, source_sep, source_g}.
    A star `m` magnitudes fainter than the target cannot dim the target's light by more than 10^(-0.4 m), so the limit
    is the target's G plus 2.5 log10(1 / depth), or plus 2.5 when the depth is not known. Empty when VizieR does not answer:
    the verdict then rests on the distance to the target alone."""
    q = ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=I/355/gaiadr3&-c=%s%%20%s%s&-c.rs=45&-out.max=400&-out=Source,RA_ICRS,DE_ICRS,Gmag"
         % (r["ra"], "%2B" if float(r["dec"]) >= 0 else "", r["dec"]))
    try:
        lines = [l.split("\t") for l in urllib.request.urlopen(urllib.request.Request(q, headers=T.UA), timeout=120).read().decode().splitlines()
                 if l and not l.startswith("#")][3:]
        stars = [(s.strip(), float(a), float(d), float(g)) for s, a, d, g in lines if g.strip()]
    except Exception:
        return {}
    arcsec = lambda a, d, a2, d2: 3600.0 * math.hypot((a - a2) * math.cos(math.radians(d)), d - d2)  # noqa: E731
    ra, dec = float(r["ra"]), float(r["dec"])
    me = [s for s in stars if s[0] == r["gaia"]] or sorted(stars, key=lambda s: arcsec(ra, dec, s[1], s[2]))[:1]
    if not me:
        return {}
    depth = float(r.get("real_depth_ppm") or 0) / 1e6
    limit = me[0][3] + (2.5 * math.log10(1.0 / depth) if 0 < depth < 1 else 2.5)
    able = [s for s in stars if s[3] <= limit or s[0] == me[0][0]]
    best = min(able, key=lambda s: arcsec(cra, cdec, s[1], s[2]))
    return {"source": r["gaia"] if best[0] == me[0][0] else best[0], "source_sep": round(arcsec(cra, cdec, best[1], best[2]), 1), "source_g": round(best[3], 2)}


def one(r):
    out = {"gaia": r["gaia"], "tic": r["tic"], "tmag": r["tmag"], "period": r["real_period"], "dur_h": r["real_dur_h"], "ra": r["ra"], "dec": r["dec"],
           "model": r.get("real_margin", "")}
    h = cutout(r["ra"], r["dec"], int(r.get("sector") or T.SECTOR))
    if h is None:
        return out | {"verdict": "no cutout"}
    try:
        t, f = frames(h)
        if len(t) == 0:
            return out | {"verdict": "no data"}
        period, t0, dur = float(r["real_period"]), float(r["real_t0"]), float(r["real_dur_h"]) / 24.0
        # 10-minute frames (sector 40): 5 points inside and 10 outside; 30-minute frames (sectors 1 to 26): fewer fit in the same windows
        need_in, need_out = (5, 10) if (len(t) < 2 or np.median(np.diff(t)) < 0.0125) else (2, 4)
        diffs = []
        for k in range(int(math.floor((t.min() - t0) / period)), int(math.ceil((t.max() - t0) / period)) + 1):
            dt = t - (t0 + k * period)
            inside, outside = np.abs(dt) < 0.35 * dur, (np.abs(dt) > 0.6 * dur) & (np.abs(dt) < 2.0 * dur)
            if inside.sum() >= need_in and outside.sum() >= need_out:
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
        if snr.max() < 4:
            return out | {"verdict": "no signal"}
        near = source(r, float(ra), float(dec))
        out.update(near)
        # on target: the centroid is within one pixel of the star and no other star that could make the dip is nearer to it
        if offset > 1.0:
            out["verdict"] = "off target"
        elif near.get("source", r["gaia"]) == r["gaia"]:
            out["verdict"] = "on target"
        else:  # another star is nearer: a blend when the star is within the error of the centroid (a quarter of a pixel), off target beyond
            out["verdict"] = "blend" if offset * PIXEL - near["source_sep"] < 0.25 * PIXEL else "off target"
        return out
    except Exception as e:
        return out | {"verdict": "error: %s" % str(e)[:60]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hunt", default=os.path.join(T.RES, "hunt-faint.csv"), help="the .csv of hunt_faint.py")
    p.add_argument("--out", default=os.path.join(T.RES, "pixel-faint.csv"))
    p.add_argument("--detections", action="store_true", help="test every detection, whatever the model says, except periods within 0.05 day of one day")
    a = p.parse_args()
    rows = [x for x in csv.DictReader(open(a.hunt)) if x.get("real_detected") == "1"]
    if a.detections:
        rows = [x for x in rows if abs(float(x["real_period"]) - 1.0) > 0.05]  # the one-day systematic of these light curves
    else:
        rows = [x for x in rows if x.get("real_eclipse") == "1"]
    done = {x["gaia"] for x in csv.DictReader(open(a.out))} if os.path.exists(a.out) else set()
    rows = [x for x in rows if x["gaia"] not in done]
    new, start, res = not os.path.exists(a.out), time.time(), []
    with open(a.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for c in range(0, len(rows), 40):
            with cf.ThreadPoolExecutor(4) as ex:
                part = list(ex.map(one, rows[c:c + 40]))
            w.writerows(part)
            fh.flush()
            res += part
    print("%d signals, %.0f s" % (len(res), time.time() - start))
    for v in sorted({x["verdict"] for x in res}):
        print("%-12s %d" % (v, sum(1 for x in res if x["verdict"] == v)))


if __name__ == "__main__":
    main()
