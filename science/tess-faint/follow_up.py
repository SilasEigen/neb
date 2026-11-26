"""Follow-up of one call that the pixel test left on its star: the depth of the dip in the light curve kept by the
hunt, the same dip in every other TESS sector that observed the star, and the catalogs that list the star.

    python follow_up.py 2099445748765925760     ->  results/follow-up-<gaia>.json

Depth: median outside the dip minus median inside, at the ephemeris of the hunt, for all events, the odd ones, the
even ones, and at half a period (a secondary eclipse). Sectors: for each sector, an 11 x 11 pixel cutout, the sum
of the 3 x 3 pixels on the star, and the best period within 2% of the hunt's; the depth is that of the raw sum, so
it is diluted by the sky and the neighbors. Catalogs: VizieR tables within 8 arcsec, and the TESS objects of
interest of the NASA Exoplanet Archive by TIC number.
"""
import csv
import io
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tf_common as T  # noqa: E402
from pixel_test import frames  # noqa: E402

CATALOGS = (("Gaia DR3 eclipsing binaries", "I/358/veb"), ("Gaia DR3 two-body orbits", "I/357/tboprb"), ("Gaia DR3 variable classes", "I/358/vclassre"),
            ("ASAS-SN variables", "II/366/catalog"), ("ZTF periodic variables", "J/ApJS/249/18"), ("ATLAS variables", "J/AJ/156/241"),
            ("Catalina periodic variables", "J/ApJS/213/9"), ("TESS eclipsing binaries", "J/ApJS/258/16"), ("VSX", "B/vsx/vsx"))


def get(url, timeout=240):
    return urllib.request.urlopen(urllib.request.Request(url, headers=T.UA), timeout=timeout).read()


def depth(t, flat, period, t0, dur, shift=0.0, parity=None):
    """(depth, error, points inside) of the dip at t0 + shift, from medians; `parity` keeps the odd or the even events"""
    d = np.mod(t - t0 - shift + 0.5 * period, period) - 0.5 * period
    keep = np.ones(len(t), bool) if parity is None else (np.round((t - t0) / period).astype(int) % 2 == parity)
    inside, outside = keep & (np.abs(d) < 0.35 * dur), keep & (np.abs(d) > 0.7 * dur) & (np.abs(d) < 2.5 * dur)
    a, b = flat[inside], flat[outside]
    err = 1.2533 * np.sqrt(a.var() / len(a) + b.var() / len(b))
    return round(float(np.median(b) - np.median(a)), 5), round(float(err), 5), int(inside.sum())


def sectors(ra, dec, period, dur):
    """per sector: cadence, points, best period near `period`, raw depth and its ratio to the scatter of the phase bins"""
    z = zipfile.ZipFile(io.BytesIO(get("https://mast.stsci.edu/tesscut/api/v0.1/astrocut?ra=%s&dec=%s&y=11&x=11&units=px" % (ra, dec), 900)))
    out = []
    for name in sorted(z.namelist()):
        h = fits.open(io.BytesIO(z.read(name)))
        row = {"sector": int(h[0].header["SECTOR"])}
        t, f = frames(h)
        tx, ty = WCS(h[2].header).all_world2pix([[float(ra), float(dec)]], 0)[0]
        ix, iy = int(round(tx)), int(round(ty))
        flux = f[:, max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2].sum(axis=(1, 2)) if len(t) else np.array([])
        if len(t) < 500 or np.median(flux) <= 0:
            out.append(row | {"points": len(t)})
            continue
        flat, best = T.flatten(t, flux / np.median(flux)), (0.0, 0.0, None)
        for p in np.arange(0.98 * period, 1.02 * period, 0.00005):
            bins = int(p / dur * 2)
            k = (np.mod(t - t[0], p) / p * bins).astype(int)
            n = np.bincount(k, minlength=bins)
            mean = np.where(n > 3, np.bincount(k, weights=flat, minlength=bins) / np.maximum(n, 1), 0.0)
            if -mean.min() > best[0]:
                best = (float(-mean.min()), float(p), mean)
        scatter = 1.4826 * np.median(np.abs(best[2] - np.median(best[2])))
        out.append(row | {"points": len(t), "cadence_min": round(float(np.median(np.diff(t))) * 1440), "period": round(best[1], 4),
                          "depth": round(best[0], 4), "depth_over_scatter": round(float(best[0] / scatter), 1) if scatter > 0 else None})
    return out


def catalogs(ra, dec, tic):
    """the rows of each catalog within 8 arcsec, and the TESS objects of interest of this TIC"""
    found = {}
    for name, table in CATALOGS:
        q = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=%s&-c=%s%%20%s%s&-c.rs=8&-out.max=5" % (table, ra, "%2B" if float(dec) >= 0 else "", dec)
        lines = [l for l in get(q).decode("utf-8", "replace").splitlines() if l and not l.startswith("#")]
        found[name] = lines[3:]  # after the header, the units and the dashes
    toi = get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query=" + urllib.parse.quote("select toi,tfopwg_disp,pl_orbper from toi where tid = %s" % tic))
    found["TESS objects of interest"] = toi.decode().strip().splitlines()[1:]
    return found


def main():
    gaia = sys.argv[1]
    r = [x for x in csv.DictReader(open(os.path.join(T.RES, "hunt-faint.csv"))) if x["gaia"] == gaia][0]
    period, t0, dur = float(r["real_period"]), float(r["real_t0"]), float(r["real_dur_h"]) / 24.0
    z = np.load(os.path.join(T.DATA, "lc", gaia + ".npz"))
    t, flat = z["t"].astype(float), 1.0 + T.flatten(z["t"].astype(float), z["f"].astype(float))
    out = {"gaia": gaia, "tic": r["tic"], "tmag": r["tmag"], "ra": r["ra"], "dec": r["dec"], "period": period, "t0": t0, "dur_h": dur * 24,
           "depth": dict(zip(("all", "odd", "even", "half_period"), (depth(t, flat, period, t0, dur), depth(t, flat, period, t0, dur, parity=1),
                                                                   depth(t, flat, period, t0, dur, parity=0), depth(t, flat, period, t0, dur, shift=0.5 * period)))),
           "sectors": sectors(r["ra"], r["dec"], period, dur), "catalogs": catalogs(r["ra"], r["dec"], r["tic"])}
    with open(os.path.join(T.RES, "follow-up-%s.json" % gaia), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["depth"]))
    for s in out["sectors"]:
        print(s)
    for name, rows in out["catalogs"].items():
        print("%-28s %s" % (name, rows or "nothing"))


if __name__ == "__main__":
    main()
