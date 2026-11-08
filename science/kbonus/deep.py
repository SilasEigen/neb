#!/usr/bin/env python3
"""Deeper checks of a lead: depth per quarter, centroid shift in transit, Gaia DR3 star and neighbours.

    python deep.py kic-006691030
"""
import csv
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hunt as H  # noqa: E402
import kb_common as K  # noqa: E402
import vet  # noqa: E402


def main():
    src = sys.argv[1]
    row = next(r for r in csv.DictReader(open(H.OUT)) if r["source"] == src)
    P, t0, dur = float(row["real_period"]), float(row["real_t0"]), float(row["real_dur_h"]) / 24
    path = os.path.join(K.LC_DIR, K.lc_name(src))
    from astropy.io import fits
    h = fits.open(path)
    gid = h[0].header.get("GAIAID", "").replace("Gaia DR3 ", "")
    ra, dec = h[0].header.get("RA_OBJ"), h[0].header.get("DEC_OBJ")
    print("%s  Gaia DR3 %s  G=%.2f  P=%.5f d  t0=%.4f  dur=%.1f h  depth %s ppm  MES %s  model score %s" % (
        src, gid, h[0].header.get("GMAG"), P, t0, dur * 24, row["real_depth_ppm"], row["real_mes"], row["real_score"]))

    # depth per quarter, on the same detrending as the search
    t, f = vet.clean(*K.read_lc(path))
    sig = vet.duration_noise(t, f, dur)
    stitched = h["LIGHTCURVE_STITCHED"].data
    qt = np.asarray(stitched["TIME"], float)
    qq = np.asarray(stitched["QUARTER"], int)
    quarter_of = dict(zip(np.round(qt, 5), qq))
    q_of_t = np.array([quarter_of.get(round(x, 5), -1) for x in t])
    print("\ndepth per quarter (ppm, +- 1 sigma):")
    for q in sorted(set(q_of_t) - {-1}):
        m = q_of_t == q
        ev = vet.events(t[m], f[m], P, t0, dur)
        if ev:
            d = np.array([x[1] for x in ev])
            print("  Q%-2d  %3d transits  %6.0f +- %4.0f" % (q, len(d), 1e6 * d.mean(), 1e6 * sig / np.sqrt(len(d))))

    # centroid shift in transit, per quarter, from the per-quarter tables
    print("\ncentroid shift in transit (pixels, in minus out; sigma):")
    tot_c, tot_r, w_c, w_r = 0.0, 0.0, 0.0, 0.0
    for ext in h:
        if not ext.name.startswith("LIGHTCURVE_Q"):
            continue
        d = ext.data
        tt = np.asarray(d["TIME"], float)
        cc, rr = np.asarray(d["CENTROID_COLUMN"], float), np.asarray(d["CENTROID_ROW"], float)
        ok = np.isfinite(tt) & np.isfinite(cc) & np.isfinite(rr)
        tt, cc, rr = tt[ok], cc[ok], rr[ok]
        if tt.size < 100:
            continue
        x = (tt - t0) / P
        dt = np.abs(x - np.rint(x)) * P
        inn, out = dt < dur / 2, (dt >= dur) & (dt < 3 * dur)
        if inn.sum() < 5 or out.sum() < 5:
            continue
        for arr, name in ((cc, "col"), (rr, "row")):
            diff = arr[inn].mean() - arr[out].mean()
            err = arr[out].std() * np.sqrt(1 / inn.sum() + 1 / out.sum())
            if name == "col":
                tot_c += diff / err ** 2
                w_c += 1 / err ** 2
            else:
                tot_r += diff / err ** 2
                w_r += 1 / err ** 2
    if w_c and w_r:
        dc, dr = tot_c / w_c, tot_r / w_r
        print("  column %+.5f px (%.1f sigma), row %+.5f px (%.1f sigma)" % (dc, dc * np.sqrt(w_c), dr, dr * np.sqrt(w_r)))

    # Gaia DR3 through the MAST API: the star and its neighbours
    req ={"service": "Mast.Catalogs.GaiaDR3.Cone", "params": {"ra": ra, "dec": dec, "radius": 20 / 3600}, "format": "json", "pagesize": 100}
    body = urllib.parse.urlencode({"request": json.dumps(req)}).encode()
    with urllib.request.urlopen(urllib.request.Request("https://mast.stsci.edu/api/v0/invoke", body, K.UA), timeout=120) as r:
        data = json.load(r)["data"]
    for n in data:
        n["sep"] = 3600 * np.hypot((n["ra"] - ra) * np.cos(np.radians(dec)), n["dec"] - dec)
    star = next((n for n in data if str(n["source_id"]) == gid), None)
    if star:
        keys = ("phot_g_mean_mag", "bp_rp", "parallax", "parallax_error", "ruwe", "teff_gspphot", "radius_gspphot", "logg_gspphot")
        print("\nGaia DR3 star:", {k: star.get(k) for k in keys})
        if star.get("parallax") and star["parallax"] > 0:
            print("  distance about %.0f pc" % (1000 / star["parallax"]))
        if star.get("radius_gspphot"):
            rp = np.sqrt(abs(float(row["real_depth_ppm"])) * 1e-6) * star["radius_gspphot"] * 109.1
            print("  planet radius if the dip is on this star, undiluted: about %.1f Earth radii" % rp)
    print("\nGaia neighbours within 20 arcsec (Kepler pixel = 4 arcsec):")
    for n in sorted(data, key=lambda n: n["sep"]):
        print("  %s  G=%5.2f  %4.1f arcsec" % (n["source_id"], n.get("phot_g_mean_mag") or float("nan"), n["sep"]))


if __name__ == "__main__":
    main()
