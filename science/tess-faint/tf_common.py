"""Shared pieces of the faint-star search: the TESS model of the main network (slot 1), the box search and the
detection tests of science/kbonus/vet.py set for one TESS sector, and the views of the chain. The detrending and
the views are those of tools/make_mainnet_datasets.py, so a signal found here becomes the same 784 values as the
training samples of the slot.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
for folder in (os.path.join(HERE, "..", "kbonus"), os.path.join(ROOT, "tools"), os.path.join(ROOT, "pow", "oracle")):
    sys.path.insert(0, folder)
import nebhash2_oracle as O  # noqa: E402
import vet  # noqa: E402
from fetch_model import rpc, unpack  # noqa: E402
from kb_common import kepler_views  # noqa: E402
from make_mainnet_datasets import detrend  # noqa: E402
from primitives import blake3  # noqa: E402
from score import v2_sums  # noqa: E402

DATA = os.environ.get("NEB_TESS_FAINT_DATA", os.path.join(HERE, "data"))
RES = os.path.join(HERE, "results")
MODEL = os.path.join(RES, "neb-tess-mainnet-block11457.json")
UA = {"user-agent": "neb-science"}
SECTOR = 40
ARCH = 1  # the convolutional network of both slots of the main network (spec/nebhash-v2.md section 3)

# the search of the Kepler hunt, set for one TESS sector: periods that a 27-day sector shows three times
vet.ARTIFACT_BANDS = ()
vet.PERIODS = np.exp(np.arange(np.log(0.5), np.log(9.0), 1e-4))
vet.DURATIONS = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0]) / 24
# the tests that say "a repeating dip is there"; the later ones (odd/even, secondary) tell two stars from a planet,
# which is not the job of this model: it sorts eclipses, of planets and of stars, from junk
DETECTION = ("fewer than 3 transits", "MES below 7.1", "one event dominates", "most events are not dips", "no noise estimate")


def fetch_model(url, path):
    """the reply of getmodel for slot 1 at the tip of a node, saved as it is"""
    m = rpc(url, "getmodel", [1])
    with open(path, "w") as f:
        json.dump(m, f, separators=(",", ":"))


def load_model(path=MODEL):
    """(weights by layer, block height, weights hash) of a getmodel reply, the weights checked against their hash"""
    m = json.load(open(path))
    m = m.get("result", m)
    packed = bytes.fromhex(m["weights"])
    if blake3(b"neb-model-weights" + packed).hex() != m["weightshash"]:
        raise SystemExit("the weights do not match their weights hash")
    if m["dataset"] != "tess":
        raise SystemExit("the model file holds the dataset %s, not tess" % m["dataset"])
    w, ws, at = unpack(packed).astype(np.int64), [], 0
    for layer in O.ARCHS[ARCH][1]:
        ws.append(w[at:at + layer.nw].reshape(layer.shape))
        at += layer.nw
    return ws, m["height"], m["weightshash"]


def margins(ws, x):
    """eclipse sum minus junk sum of the last layer, for rows of 784 values: positive is the node's answer "eclipse" """
    z = v2_sums(ARCH, ws, np.asarray(x, np.int64))
    return z[:, 1] - z[:, 0]


def flatten(t, f, window=1.0):
    """relative flux, 0 = a running median of `window` days: what the box search reads"""
    n = max(3, int(window / np.median(np.diff(t))) | 1)
    trend = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(f, n // 2, mode="edge"), n), axis=1)
    return f / np.where(trend > 0, trend, 1.0) - 1.0


def judge(ws, t, f, sign=1.0):
    """search, detection tests and the model on one light curve (sign -1: the flux flipped, the control)"""
    vet.CADENCE = float(np.median(np.diff(t)))  # 30 minutes in the first two years of TESS, 10 minutes or less after
    flat = sign * flatten(t, f)
    keep = vet.clean_mask(t, flat)
    tc, fc = t[keep], flat[keep]
    p, t0, dur = vet.search(tc, fc)
    v = vet.vet(tc, fc, p, t0, dur, rho_min=0.05)
    raw = 1.0 + sign * (f - 1.0)  # the flipped light curve, around 1
    x = kepler_views(t, detrend(t, raw, p, t0, dur), p, t0, dur * 24.0)
    margin = int(margins(ws, x[None])[0])
    return {"period": round(p, 5), "t0": round(t0, 4), "dur_h": round(dur * 24, 2), "n": v.get("n_transits", 0), "depth_ppm": v.get("depth_ppm", ""),
            "mes": v.get("mes", ""), "odd_even": v.get("odd_even_sigma", ""), "secondary": v.get("secondary_sigma", ""), "fail": v["fail"],
            "detected": int(v["fail"] not in DETECTION), "eclipse": int(margin > 0), "margin": margin}
