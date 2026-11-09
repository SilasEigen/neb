#!/usr/bin/env python3
"""The tests added to vet.py after the hunt (depth by Kepler season, transit shape) applied to its survivors, real and
inverted, at the ephemeris stored in results/hunt.csv. Light curves that the hunt deleted are downloaded again, and
deleted again afterwards.

    python revet.py      -> results/revet.csv and a table of survivors by MES, before and after
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hunt as H  # noqa: E402
import kb_common as K  # noqa: E402
import vet  # noqa: E402

OUT = os.path.join(K.RESULTS, "revet.csv")
THRESHOLDS = (7.1, 8, 10, 15, 20, 30)


def main():
    rows = [r for r in csv.DictReader(open(H.OUT)) if r.get("real_period")]
    todo = [r for r in rows if H.survivor(r, "real") or H.survivor(r, "inv")]
    rel = {}
    with open(K.file_list()) as f:
        for line in f:
            m = K.LC_PATTERN.search(line)
            if m:
                rel[m.group(2)] = m.group(1)
    out = []
    for r in todo:
        src = r["source"]
        path = os.path.join(K.LC_DIR, K.lc_name(src))
        fetched = not os.path.exists(path)
        if fetched and K.fetch(rel[src], src) is None:
            print("cannot download", src)
            continue
        t, f, q = K.read_lc(path, quarters=True)
        for tag, flux in (("real", f), ("inv", -f)):
            if not H.survivor(r, tag):
                continue
            keep = vet.clean_mask(t, flux)
            v = vet.vet(t[keep], flux[keep], float(r[tag + "_period"]), float(r[tag + "_t0"]),
                        float(r[tag + "_dur_h"]) / 24, quarter=q[keep])
            out.append({"source": src, "side": tag, "mes": r[tag + "_mes"], "score": r[tag + "_score"],
                        "season_sigma": v.get("season_sigma", ""), "season_ratio": v.get("season_ratio", ""),
                        "shape_dchi2": v.get("shape_dchi2", ""), "fail": v["fail"]})
        if fetched:
            os.remove(path)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print("survivors re-vetted: %d real, %d inverted" % (sum(x["side"] == "real" for x in out),
                                                         sum(x["side"] == "inv" for x in out)))
    print("MES at least   real before -> after   inverted before -> after")
    for th in THRESHOLDS:
        n = {(s, a): sum(x["side"] == s and float(x["mes"]) >= th and (a == "before" or x["fail"] == "") for x in out)
             for s in ("real", "inv") for a in ("before", "after")}
        print("%12.1f   %6d -> %-6d          %6d -> %d" % (th, n["real", "before"], n["real", "after"],
                                                         n["inv", "before"], n["inv", "after"]))
    fails = {}
    for x in out:
        if x["fail"]:
            fails[(x["side"], x["fail"])] = fails.get((x["side"], x["fail"]), 0) + 1
    print("rejected by the new tests:", fails)


if __name__ == "__main__":
    main()
