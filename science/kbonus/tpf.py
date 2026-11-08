#!/usr/bin/env python3
"""Pixel-level check of a lead: difference images (out of transit minus in transit) in the Kepler target pixel files
that contain the lead, quarter by quarter. The pixel where the light goes missing shows which star dims.

    python tpf.py kic-006691030 [radius_arcsec]

For every transit: mean of the cadences within dur..3 dur around it minus the mean of the in-transit cadences, then
the mean over transits and its error. Positions of the lead and of its Gaia neighbours come from the WCS of each
target pixel file. Writes results/tpf-<source>.png (the quarter with the strongest difference) and prints a table.
"""
import csv
import json
import os
import re
import sys
import urllib.parse
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hunt as H  # noqa: E402
import kb_common as K  # noqa: E402

TPF_DIR = os.path.join(K.DATA, "tpf")
ARCHIVE = "https://archive.stsci.edu/missions/kepler/target_pixel_files"


def mast(service, params):
    body = urllib.parse.urlencode({"request": json.dumps({"service": service, "params": params, "format": "json", "pagesize": 500})}).encode()
    with urllib.request.urlopen(urllib.request.Request("https://mast.stsci.edu/api/v0/invoke", body, K.UA), timeout=120) as r:
        return json.load(r)["data"]


def kepler_targets_near(ra, dec, arcsec):
    out = set()
    for o in mast("Mast.Caom.Cone", {"ra": ra, "dec": dec, "radius": arcsec / 3600}):
        m = re.match(r"kplr0*(\d+)", str(o.get("target_name", "")))
        if o.get("obs_collection") == "Kepler" and m:
            out.add(int(m.group(1)))
    return sorted(out)


def tpf_files(kic):
    k9 = "%09d" % kic
    url = "%s/%s/%s/" % (ARCHIVE, k9[:4], k9)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=K.UA), timeout=120) as r:
            names = sorted(set(re.findall(r'href="(kplr%s-\d+_lpd-targ\.fits\.gz)"' % k9, r.read().decode())))
    except Exception:
        return []
    os.makedirs(TPF_DIR, exist_ok=True)
    paths = []
    for n in names:
        p = os.path.join(TPF_DIR, n)
        if not os.path.exists(p):
            with urllib.request.urlopen(urllib.request.Request(url + n, headers=K.UA), timeout=300) as r, open(p + ".part", "wb") as f:
                f.write(r.read())
            os.replace(p + ".part", p)
        paths.append(p)
    return paths


def difference(path, P, t0, dur):
    from astropy.io import fits
    with fits.open(path) as h:
        d = h[1].data
        t, cube, q = np.asarray(d["TIME"], float), np.asarray(d["FLUX"], float), np.asarray(d["QUALITY"])
        hdr2 = h[2].header
        quarter, channel = h[0].header["QUARTER"], h[0].header["CHANNEL"]
    # pixels outside the downloaded mask are NaN in every cadence: keep the cadences, ignore those pixels
    ok = np.isfinite(t) & (q == 0) & np.isfinite(cube).any(axis=(1, 2))
    t, cube = t[ok], cube[ok]
    x = (t - t0) / P
    n = np.rint(x).astype(int)
    dt = np.abs(x - n) * P
    diffs = []
    for e in np.unique(n[dt < dur / 2]):
        inn = (n == e) & (dt < dur / 2)
        ring = (n == e) & (dt >= dur) & (dt < 3 * dur)
        if inn.sum() >= 1 and ring.sum() >= 2:
            diffs.append(np.nanmean(cube[ring], 0) - np.nanmean(cube[inn], 0))
    if len(diffs) < 3:
        return None
    diffs = np.array(diffs)
    D = np.nan_to_num(np.nanmean(diffs, 0))
    E = np.nan_to_num(np.nanstd(diffs, 0, ddof=1) / np.sqrt(len(diffs)))
    return {"quarter": quarter, "channel": channel, "D": D, "E": E, "mean": np.nan_to_num(np.nanmean(cube, 0)), "n": len(diffs), "hdr": hdr2}


def main():
    src = sys.argv[1]
    row = next(r for r in csv.DictReader(open(H.OUT)) if r["source"] == src)
    P, t0, dur = float(row["real_period"]), float(row["real_t0"]), float(row["real_dur_h"]) / 24
    from astropy.io import fits
    from astropy.wcs import WCS
    hd = fits.open(os.path.join(K.LC_DIR, K.lc_name(src)))[0].header
    ra, dec = hd["RA_OBJ"], hd["DEC_OBJ"]
    gaia = mast("Mast.Catalogs.GaiaDR3.Cone", {"ra": ra, "dec": dec, "radius": 25 / 3600})
    radius = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    kics = kepler_targets_near(ra, dec, radius)
    print("%s  P=%.5f d  dur=%.1f h; Kepler targets within %.0f arcsec: %s" % (src, P, dur * 24, radius, kics), flush=True)
    results = []
    for kic in kics:
        for path in tpf_files(kic):
            r = difference(path, P, t0, dur)
            if r is None:
                continue
            w = WCS(r["hdr"])
            ny, nx = r["D"].shape
            lx, ly = w.all_world2pix([[ra, dec]], 0)[0]
            if not (-0.5 <= lx < nx - 0.5 and -0.5 <= ly < ny - 0.5):
                continue
            snr = np.where(r["E"] > 0, r["D"] / r["E"], 0)
            iy, ix = np.unravel_index(np.argmax(snr), snr.shape)
            # flux-weighted centroid of the positive difference in the 3x3 box around the strongest pixel
            y0, y1, x0, x1 = max(iy - 1, 0), min(iy + 2, ny), max(ix - 1, 0), min(ix + 2, nx)
            box = np.clip(r["D"][y0:y1, x0:x1], 0, None)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            cx, cy = (xx * box).sum() / box.sum(), (yy * box).sum() / box.sum()
            near = []
            for g in gaia:
                gx, gy = w.all_world2pix([[g["ra"], g["dec"]]], 0)[0]
                near.append((4.0 * np.hypot(gx - cx, gy - cy), g["phot_g_mean_mag"], str(g["source_id"]) == hd.get("GAIAID", "").replace("Gaia DR3 ", "")))
            near.sort()
            results.append({"kic": kic, "quarter": r["quarter"], "n": r["n"], "peak_snr": float(snr.max()),
                            "off_lead": 4.0 * np.hypot(cx - lx, cy - ly), "closest": near[:3], "r": r, "lead_xy": (lx, ly), "w": w, "c": (cx, cy)})
            print("  Q%-2d TPF of KIC %d: %3d transits, peak SNR %5.1f, dip centroid %4.1f arcsec from the lead; closest Gaia stars "
                  "to the dip: %s" % (r["quarter"], kic, r["n"], snr.max(), results[-1]["off_lead"],
                                      ", ".join("%.1f\" G=%.1f%s" % (s, gm or np.nan, " (lead)" if me else "") for s, gm, me in near[:3])), flush=True)
    if not results:
        print("no target pixel file contains the lead")
        return
    good = [x for x in results if x["peak_snr"] >= 4]
    if good:
        off = np.array([x["off_lead"] for x in good])
        print("quarters with peak SNR >= 4: %d; dip centroid to lead: median %.1f arcsec (min %.1f, max %.1f)" % (len(good), np.median(off), off.min(), off.max()))
    best = max(results, key=lambda x: x["peak_snr"])
    draw(src, best, gaia)


def draw(src, best, gaia):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r, w = best["r"], best["w"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=160)
    for ax, img, title in ((axes[0], r["mean"], "mean image (out of transit)"), (axes[1], r["D"] / np.where(r["E"] > 0, r["E"], np.inf), "difference out - in (SNR)")):
        im = ax.imshow(img, origin="lower", cmap="Greys" if ax is axes[0] else "Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)
        for g in gaia:
            gx, gy = w.all_world2pix([[g["ra"], g["dec"]]], 0)[0]
            if -0.5 <= gx < img.shape[1] - 0.5 and -0.5 <= gy < img.shape[0] - 0.5:
                ax.plot(gx, gy, "o", ms=6, mfc="none", mec="#eb6834", mew=1.2)
                ax.annotate("G=%.1f" % (g["phot_g_mean_mag"] or np.nan), (gx, gy), xytext=(4, 4), textcoords="offset points", fontsize=7, color="#52514e")
        ax.plot(*best["lead_xy"], "x", ms=9, color="#e34948", mew=2)
        ax.plot(*best["c"], "+", ms=12, color="#0b0b0b", mew=1.5)
        ax.set_title(title, fontsize=10)
    fig.suptitle("%s, Q%d, TPF of KIC %d: red x = lead, black + = dip centroid, circles = Gaia DR3" % (src, r["quarter"], best["kic"]), fontsize=10)
    out = os.path.join(K.RESULTS, "tpf-%s.png" % src)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
