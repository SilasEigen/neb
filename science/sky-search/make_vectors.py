"""Writes vectors/sky-search.txt: light curves and the entries the reference returns for each, then the hashes of
the batch file and of the result file that hold them all, in this order.

    python make_vectors.py [GAIA ...]

The first light curves are real ones: TESS-Gaia light curves of sector 40 kept by science/tess-faint (their Gaia DR3
identifiers on the command line), prepared by prepare.py. The others are edge cases.
"""
import os
import sys

import numpy as np

import sky_ref as S
from prepare import T, slots

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vectors", "sky-search.txt")

curves = []
for gaia in sys.argv[1:]:
    z = np.load(os.path.join(T.DATA, "lc", gaia + ".npz"))
    curves.append(slots(z["t"].astype(float), z["f"].astype(float))[0])
rng = np.random.default_rng(1)
noise = rng.integers(-40, 41, 660).astype(np.int8)
noise[rng.random(660) < 0.1] = S.NO_DATA
worst = np.full(S.MAX_SLOTS, 127, np.int8)
worst[::2] = -127
curves += [noise, np.zeros(0, np.int8), np.zeros(5, np.int8), np.full(660, S.NO_DATA, np.int8), np.full(700, 5, np.int8), worst,
           np.where(np.arange(300) % 24 < 2, -100, 3).astype(np.int8)]  # a dip of two hours every day: ties between periods

raw = S.write_batch(curves)
result = S.run(raw)
_, rows = S.read_result(result)
with open(OUT, "w", newline="\n") as f:
    f.write("neb-sky-search-vectors 1\n")
    for i, (c, r) in enumerate(zip(curves, rows)):
        assert all(S.slow_score(c, p, a, q) == s for s, p, a, q in r if s), "fast and plain scores differ"
        f.write("curve %s %s\n" % (c.tobytes().hex(), result[44 + i * 512:44 + (i + 1) * 512].hex()))
    f.write("batch %s %s\n" % (S.blake3(raw).hex(), S.blake3(result).hex()))
print("%d light curves, %d bytes of batch, %d of result" % (len(curves), len(raw), len(result)))
