#!/usr/bin/env python3
"""Blend test: fold the Kepler Bonus light curves of the neighbours of a lead at the lead's period and epoch.

    python neighbors.py kic-006691030

A neighbour that shows the same dip is the likely source; the lead is then a blend, not a planet of the faint star.
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
import vet  # noqa: E402


def mast_gaia(ra, dec, radius_arcsec):
    req = {"service": "Mast.Catalogs.GaiaDR3.Cone", "params": {"ra": ra, "dec": dec, "radius": radius_arcsec / 3600}, "format": "json", "pagesize": 100}
    body = urllib.parse.urlencode({"request": json.dumps(req)}).encode()
    with urllib.request.urlopen(urllib.request.Request("https://mast.stsci.edu/api/v0/invoke", body, K.UA), timeout=120) as r:
        return json.load(r)["data"]


def kic_near(ra, dec, radius_arcsec):
    """KIC numbers of the Kepler targets near a position (MAST observations of the Kepler mission; the Exoplanet
    Archive TAP service refuses cone queries on keplerstellar)"""
    req = {"service": "Mast.Caom.Cone", "params": {"ra": ra, "dec": dec, "radius": radius_arcsec / 3600}, "format": "json", "pagesize": 500}
    body = urllib.parse.urlencode({"request": json.dumps(req)}).encode()
    with urllib.request.urlopen(urllib.request.Request("https://mast.stsci.edu/api/v0/invoke", body, K.UA), timeout=120) as r:
        data = json.load(r)["data"]
    out = set()
    for o in data:
        m = re.match(r"kplr0*(\d+)", str(o.get("target_name", "")))
        if o.get("obs_collection") == "Kepler" and m:
            out.add(int(m.group(1)))
    return [{"kepid": k} for k in sorted(out)]


def main():
    src = sys.argv[1]
    row = next(r for r in csv.DictReader(open(H.OUT)) if r["source"] == src)
    P, t0, dur = float(row["real_period"]), float(row["real_t0"]), float(row["real_dur_h"]) / 24
    from astropy.io import fits
    hd = fits.open(os.path.join(K.LC_DIR, K.lc_name(src)))[0].header
    ra, dec = hd["RA_OBJ"], hd["DEC_OBJ"]
    names = {}
    pat = re.compile(r"--output 'kbonus-bkg/lcs/([^']+)/hlsp_kbonus-bkg_kepler_kepler_([^_]+)_kepler_v1\.0_lc\.fits'")
    wanted_gaia = {"gaia-dr3-%s" % n["source_id"]: n for n in mast_gaia(ra, dec, 20)}
    targets = {"kic-%09d" % int(t["kepid"]): t for t in kic_near(ra, dec, 20)}
    with open(K.file_list()) as f:
        for line in f:
            m = pat.search(line)
            if m and (m.group(2) in wanted_gaia or m.group(2) in targets):
                names[m.group(2)] = m.group(1)
    print("%s: P=%.5f d, lead depth %s ppm; KBonus light curves within 20 arcsec: %d" % (src, P, row["real_depth_ppm"], len(names)))
    paths = K.fetch_many([(d, s) for s, d in names.items()])
    for s, path in sorted(paths.items()):
        if not path:
            continue
        h = fits.open(path)[0].header
        sep = 3600 * np.hypot((h["RA_OBJ"] - ra) * np.cos(np.radians(dec)), h["DEC_OBJ"] - dec)
        t, f = vet.clean(*K.read_lc(path))
        ev = vet.events(t, f, P, t0, dur)
        sig = vet.duration_noise(t, f, dur)
        if not ev or not np.isfinite(sig):
            print("  %-30s %5.1f arcsec  G=%5.2f  no usable events" % (s, sep, h.get("GMAG") or np.nan))
            continue
        d = np.array([x[1] for x in ev])
        snr = d.mean() / (sig / np.sqrt(len(d)))
        print("  %-30s %5.1f arcsec  G=%5.2f  depth at the lead's ephemeris %7.0f ppm  (%5.1f sigma, %d events)" % (
            s, sep, h.get("GMAG") or np.nan, 1e6 * d.mean(), snr, len(d)))


if __name__ == "__main__":
    main()
