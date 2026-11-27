"""What the batch server does before a batch leaves: a light curve becomes one int8 per hour. Floats are used here
and nowhere after, so the miners all read the same integers.

    python prepare.py --compare HUNT_CSV      the integer search against the search of science/tess-faint, on the
                                              light curves that search kept
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tess-faint"))
import sky_ref as S  # noqa: E402
import tf_common as T  # noqa: E402

SCALE = 16  # units in one scatter of the hourly flux


def slots(t, f):
    """(int8 per hour from the first measurement, time of slot 0 in days): the flux as the box search of
    science/tess-faint reads it (running median removed, flares and gap edges dropped), the median of each hour,
    in sixteenths of the scatter of those medians"""
    flat = T.flatten(t, f)
    keep = T.vet.clean_mask(t, flat)
    k = np.floor((t[keep] - t[0]) * 24).astype(int)
    x = np.full(k.max() + 1, S.NO_DATA, np.int64)
    used = np.unique(k)
    med = np.array([np.median(flat[keep][k == i]) for i in used])
    sigma = 1.4826 * np.median(np.abs(med - np.median(med))) or 1e-4
    x[used] = np.clip(np.rint(med / sigma * SCALE), -127, 127)
    if x.size > S.MAX_SLOTS:
        raise ValueError("more than %d hours: cut the light curve by sector" % S.MAX_SLOTS)
    return x.astype(np.int8), float(t[0])


def compare(hunt):
    """for each kept light curve: is the period of the float search among the entries of the integer search"""
    rows = {r["gaia"]: r for r in csv.DictReader(open(hunt)) if r.get("real_period")}
    same = harmonic = n = 0
    for name in sorted(os.listdir(os.path.join(T.DATA, "lc"))):
        r = rows.get(name[:-4])
        if not r:
            continue
        z = np.load(os.path.join(T.DATA, "lc", name))
        x, _ = slots(z["t"].astype(float), z["f"].astype(float))
        found = S.search(x)[:, 1] / S.UNIT / 24
        ratio = found[found > 0] / float(r["real_period"])
        first = any(abs(q - 1) < 0.01 for q in ratio)
        other = any(abs(q * k - 1) < 0.01 or abs(q / k - 1) < 0.01 for q in ratio for k in (2, 3, 4))
        n, same, harmonic = n + 1, same + first, harmonic + (other and not first)
        print("%s  P %.4f  mes %5s  %s  best %.4f" % (name[:-4], float(r["real_period"]), r["real_mes"],
                                                    "same" if first else "harmonic" if other else "MISSING", found[0]), flush=True)
    print("%d light curves: %d with the same period, %d with a harmonic only, %d missing" % (n, same, harmonic, n - same - harmonic))


if __name__ == "__main__":
    compare(sys.argv[2])
