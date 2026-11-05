#!/usr/bin/env python3
"""Training shares with network delay: does the training hold when the gradients are stale? (simulation)

tools/training_shares.py applies, at every model step, 32 share gradients computed on the current state. In a chain the
shares found in round r (computed on the state S) can only be listed in block r+1, so the model trains on stale gradients.
This script simulates it with the same model step (constants, clipped shares, bounded residual, 16 samples per share,
32 shares per step) and two parameters:

    delay D            the gradients applied by block k were computed on the state at the start of block k-D
                       (D = 0: the state at the start of block k; for k < D, the initial state)
    shares_per_block   a block lists this many shares and applies shares_per_block / 32 model steps in a row,
                       all with gradients computed on the same stale state

Every configuration uses the same share nonces (step s, share m: seed * 10^9 + s * 1000 + m, so seed 0 and D = 0 with
32 shares per block is exactly tools/training_shares.py) and 96,000 shares in total. The test accuracy is measured at block
boundaries every 5 % of the shares.

    python tools/training_shares_delay.py --cache /path/outside/the/tree/mnist
    python tools/training_shares_delay.py --cache DIR --seeds 1 --steps 300 --delays 0 1 --per-block 32
    python tools/training_shares_delay.py --cache DIR --seeds 1 --one-step --per-block 64 256   # a block = one model step
    python tools/training_shares_delay.py --cache DIR --seeds 1 --su-shift -2                   # SU1 = 13 and SU2 = 10

Needs numpy and network access the first time (MNIST goes to --cache, never into the source tree).
"""
import argparse
import collections
import concurrent.futures
import json
import os
import sys
import tempfile
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread per worker process

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training_shares as T  # noqa: E402

FILES = (("train-images-idx3-ubyte.gz", True), ("train-labels-idx1-ubyte.gz", False),
         ("t10k-images-idx3-ubyte.gz", True), ("t10k-labels-idx1-ubyte.gz", False))
# the nonces of different seeds are disjoint as long as a run has fewer than 10^9 / 1000 = 10^6 steps
SEED_STRIDE = 10 ** 9
CHECKPOINTS = 20
BASE_SU = (T.CFG["SU1"], T.CFG["SU2"])


def load(cache):
    """MNIST as in training_shares.py (T.mnist), downloaded once into `cache` as .npy files."""
    os.makedirs(cache, exist_ok=True)
    out = []
    for name, images in FILES:
        path = os.path.join(cache, name + ".npy")
        if not os.path.exists(path):
            np.save(path, T.mnist(name, images))
        out.append(np.load(path))
    X, y, Xt, yt = out
    Tg = np.full((len(y), 10), -8, np.int8)
    Tg[np.arange(len(y)), y] = 7
    return X >> 5, Tg, Xt >> 5, yt   # uint8 pixels quantised to 0..7, as floor(x / 32); small, converted per batch


def batch_indices(nonces, n):
    """T.share_batch for many nonces at once."""
    j = np.arange(T.PER_SHARE, dtype=np.uint64)
    z = nonces.astype(np.uint64)[:, None] * np.uint64(T.PER_SHARE) + j[None, :]
    return (T.splitmix64(z) % np.uint64(n)).astype(np.int64)


def share_sums(S1, S2, X, Tg, nonces, clip, dt):
    """Sum of the clipped gradients of the shares `nonces`, computed on the state (S1, S2)."""
    bi = batch_indices(nonces, len(X))
    g1, g2 = T.share_gradients(S1, S2, X[bi].astype(dt), Tg[bi].astype(dt), dt)
    if clip:
        g1, g2 = np.clip(g1, -clip, clip), np.clip(g2, -clip, clip)
    return g1.sum(0), g2.sum(0)


def model_step(W1, W2, R1, R2, G1, G2):
    """The step of spec/shares.md section 6 (same lines as T.train) on the summed gradients G of the shares."""
    dt = W1.dtype.type
    R1, R2 = R1 + G1, R2 + G2
    s1 = np.clip(T.tdiv(R1, T.CFG["SU1"], dt), -1, 1)
    s2 = np.clip(T.tdiv(R2, T.CFG["SU2"], dt), -1, 1)
    R1, R2 = R1 - s1 * (1 << T.CFG["SU1"]), R2 - s2 * (1 << T.CFG["SU2"])
    R1, R2 = np.clip(R1, -(1 << T.CFG["SU1"]), 1 << T.CFG["SU1"]), np.clip(R2, -(1 << T.CFG["SU2"]), 1 << T.CFG["SU2"])
    return np.clip(W1 - s1, -8, 7), np.clip(W2 - s2, -8, 7), R1, R2


# The first block k < D applies gradients of the initial state, because the history is shorter than D + 1.
def simulate(data, seed, delay, per_block, steps, clip, dt, one_step=False, checkpoints=CHECKPOINTS):
    """With one_step, a block is a single model step over all its shares (spec/shares.md section 6) instead of
    per_block / 32 steps in a row. Returns [(shares, test accuracy)] at block boundaries, every 1/checkpoints of the shares."""
    X, Tg, Xt, yt = data
    Xtd = Xt.astype(dt)
    per = per_block // T.SHARES
    total = steps * T.SHARES
    W1, W2 = T.initial_model(dt, X.shape[1])
    R1, R2 = np.zeros_like(W1), np.zeros_like(W2)
    hist = collections.deque(maxlen=delay + 1)
    curve, nxt = [], 1
    for k in range(steps // per):
        hist.append((W1, W2))                      # state at the start of block k
        S1, S2 = hist[0]                           # state D blocks earlier (the initial state while k < D)
        G1, G2 = 0, 0
        for j in range(per):
            step = k * per + j
            nonces = seed * SEED_STRIDE + step * 1000 + np.arange(T.SHARES)
            g1, g2 = share_sums(S1, S2, X, Tg, nonces, clip, dt)
            if one_step:
                G1, G2 = G1 + g1, G2 + g2
            else:
                W1, W2, R1, R2 = model_step(W1, W2, R1, R2, g1, g2)
        if one_step:
            W1, W2, R1, R2 = model_step(W1, W2, R1, R2, G1, G2)
        done = (k + 1) * per_block
        if done * checkpoints >= nxt * total:
            curve.append((done, T.accuracy(W1, W2, Xtd, yt)))
            nxt = done * checkpoints // total + 1
    return curve


# One worker process per job. CFG is module state, so the update shifts are set in the process that uses them.
def job(args):
    cache, seed, delay, per_block, steps, clip, dtname, one_step, su_shift = args
    t0 = time.time()
    T.CFG["SU1"], T.CFG["SU2"] = BASE_SU[0] + su_shift, BASE_SU[1] + su_shift
    curve = simulate(load(cache), seed, delay, per_block, steps, clip, np.dtype(dtname).type, one_step)
    return (seed, delay, per_block), curve, time.time() - t0


def check(data, clip):
    """D = 0 with 32 shares per block is T.train, bit for bit, in float32 and in float64 (60 steps)."""
    X, Tg = data[0].astype(np.float64), data[1].astype(np.float64)
    nonces = [[s * 1000 + k for k in range(T.SHARES)] for s in range(60)]
    ref = T.train(nonces, np.float64, X, Tg, clip)
    ok = {}
    for name in ("float64", "float32"):
        dt = np.dtype(name).type
        W1, W2 = T.initial_model(dt, X.shape[1])
        R1, R2 = np.zeros_like(W1), np.zeros_like(W2)
        for s in range(60):
            G1, G2 = share_sums(W1, W2, X, Tg, np.array(nonces[s]), clip, dt)
            W1, W2, R1, R2 = model_step(W1, W2, R1, R2, G1, G2)
        ok[name] = bool(np.array_equal(W1, ref[0]) and np.array_equal(W2, ref[1]))
    return ok


def pct(x):
    return "%.1f" % (100 * x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "neb-mnist"), help="MNIST cache directory")
    p.add_argument("--steps", type=int, default=3000, help="model steps of 32 shares (3000 = 96,000 shares)")
    p.add_argument("--clip", type=int, default=256)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--delays", type=int, nargs="+", default=[0, 1, 2, 4])
    p.add_argument("--per-block", type=int, nargs="+", default=[32, 64, 256], help="shares per block, multiples of 32")
    p.add_argument("--curves", default="0:32,1:32,2:256", help="delay:shares_per_block of the curves to print")
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--one-step", action="store_true", help="a block is one model step over all its shares")
    p.add_argument("--su-shift", type=int, default=0, help="add this to SU1 and SU2 of tools/training_shares.py (positive: slower updates)")
    # float32 is exact while every integer stays below 2^24: the largest are |z1| <= 43904 and
    # |R1 + GS1| <= 2^15 + 256 * 256 = 98304 for a one-step block of 256 shares
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="both are exact here")
    p.add_argument("--json", help="write all curves to this file")
    a = p.parse_args()
    data = load(a.cache)
    print("check, D=0 and 32 shares per block against T.train in float64 (60 steps): %s" % check(data, a.clip), flush=True)
    jobs = [(a.cache, s, d, b, a.steps, a.clip, a.dtype, a.one_step, a.su_shift) for b in a.per_block for d in a.delays for s in range(a.seeds)]
    res = {}
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(a.workers) as ex:
        for key, curve, dt in ex.map(job, jobs):
            res[key] = curve
            print("done seed %d D=%d %d shares/block: %s %% (%.0f s)" % (key + (pct(curve[-1][1]), dt)), flush=True)
    print("total %.0f s on %d workers\n" % (time.time() - t0, a.workers))

    total = a.steps * T.SHARES
    print("Final test accuracy (%%) after %d shares, MNIST test set (10,000 images); seeds %d: mean [min-max]" % (total, a.seeds))
    print("| delay D | " + " | ".join("%d shares/block" % b for b in a.per_block) + " |")
    print("|---|" + "---|" * len(a.per_block))
    for d in a.delays:
        cells = []
        for b in a.per_block:
            v = [res[(s, d, b)][-1][1] for s in range(a.seeds)]
            cells.append("%s [%s-%s]" % (pct(np.mean(v)), pct(min(v)), pct(max(v))))
        print("| %d | " % d + " | ".join(cells) + " |")
    print()
    for spec in a.curves.split(","):
        d, b = (int(x) for x in spec.split(":"))
        print("Curve D=%d, %d shares/block (accuracy %% by number of shares, mean of %d seeds)" % (d, b, a.seeds))
        pts = [(res[(0, d, b)][i][0], np.mean([res[(s, d, b)][i][1] for s in range(a.seeds)])) for i in range(len(res[(0, d, b)]))]
        print("  " + "  ".join("%d:%s" % (n, pct(v)) for n, v in pts))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"%d,%d,%d" % k: v for k, v in res.items()}, f)


if __name__ == "__main__":
    main()
