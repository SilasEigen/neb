#!/usr/bin/env python3
"""Delays and attacks on the training shares (docs/experiments.md, section 2).

    python tools/training_shares_attacks.py --experiment staleness
    python tools/training_shares_attacks.py --experiment targeted
    python tools/training_shares_attacks.py --experiment adaptive

staleness  shares computed on an older model version are accepted (their gradient is stale), or rejected (their work is lost)
targeted   an attacker grinds nonces so that its shares contain mostly one class, to push the model toward that class
adaptive   an attacker grinds nonces so that its gradients point against the honest direction, with half of the shares

Uses the step of tools/training_shares.py. Needs numpy and network access (MNIST).
"""
import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training_shares as T  # noqa: E402


# Same sampling as T.share_batch, for many nonces at once.
def batch_indices(nonces, n):
    j = np.arange(T.PER_SHARE, dtype=np.uint64)
    z = nonces.astype(np.uint64)[:, None] * np.uint64(T.PER_SHARE) + j[None, :]
    return (T.splitmix64(z) % np.uint64(n)).astype(np.int64)


def predict(W1, W2, X):
    a1 = np.clip(np.floor((X @ W1.T + (1 << (T.CFG["S1"] - 1))) / (1 << T.CFG["S1"])), -8, 7)
    return (a1 @ W2.T).argmax(1)


def run(X, T_, steps, clip, plan_fn, max_delay=0):
    """Training loop with a plan of (nonce, delay) per step; delay d uses the model of d steps ago."""
    dt = np.float64
    W1, W2 = T.initial_model(dt, X.shape[1])
    R1, R2 = np.zeros_like(W1), np.zeros_like(W2)
    hist = collections.deque(maxlen=max_delay + 1)
    for step in range(steps):
        hist.appendleft((W1.copy(), W2.copy()))
        plan = plan_fn(step)
        if not plan:
            continue
        g1s, g2s = np.zeros_like(W1), np.zeros_like(W2)
        for d in sorted({dl for _, dl in plan}):
            nonces = np.array([n for n, dl in plan if dl == d], dtype=np.uint64)
            m1, m2 = hist[min(d, len(hist) - 1)]
            bi = batch_indices(nonces, len(X))
            g1, g2 = T.share_gradients(m1, m2, X[bi], T_[bi], dt)
            if clip:
                g1, g2 = np.clip(g1, -clip, clip), np.clip(g2, -clip, clip)
            g1s, g2s = g1s + g1.sum(0), g2s + g2.sum(0)
        # model step of spec/shares.md section 6
        R1, R2 = R1 + g1s, R2 + g2s
        s1 = np.clip(T.tdiv(R1, T.CFG["SU1"], dt), -1, 1)
        s2 = np.clip(T.tdiv(R2, T.CFG["SU2"], dt), -1, 1)
        R1, R2 = R1 - s1 * (1 << T.CFG["SU1"]), R2 - s2 * (1 << T.CFG["SU2"])
        R1, R2 = np.clip(R1, -(1 << T.CFG["SU1"]), 1 << T.CFG["SU1"]), np.clip(R2, -(1 << T.CFG["SU2"]), 1 << T.CFG["SU2"])
        W1, W2 = np.clip(W1 - s1, -8, 7), np.clip(W2 - s2, -8, 7)
    return W1, W2


def honest(step):
    return [(step * 1000 + k, 0) for k in range(T.SHARES)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", choices=["staleness", "targeted", "adaptive"], required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--clip", type=int, default=256)
    a = p.parse_args()
    q = lambda x: np.clip(np.floor(x.astype(np.float64) / 32.0), 0, 7)
    X = q(T.mnist("train-images-idx3-ubyte.gz", True)); y = T.mnist("train-labels-idx1-ubyte.gz", False)
    Xt = q(T.mnist("t10k-images-idx3-ubyte.gz", True)); yt = T.mnist("t10k-labels-idx1-ubyte.gz", False)
    Tg = np.full((len(y), 10), -8.0); Tg[np.arange(len(y)), y] = 7.0

    def report(label, W1, W2, target=None):
        pr = predict(W1, W2, Xt)
        line = "%-58s accuracy %5.1f %%" % (label, 100 * float((pr == yt).mean()))
        if target is not None:
            line += "   other digits called %d: %5.1f %%   recall of %d: %5.1f %%" % (
                target, 100 * float(((pr == target) & (yt != target)).sum() / (yt != target).sum()),
                target, 100 * float(((pr == target) & (yt == target)).sum() / (yt == target).sum()))
        print(line, flush=True)

    if a.experiment == "staleness":
        report("no delay, 32 shares per step", *run(X, Tg, a.steps, a.clip, honest))
        for d in (1, 4, 16):
            for frac in (0.5, 1.0):
                # the first nb shares of the step use the model of d steps ago
                nb = int(round(frac * T.SHARES))
                fn = lambda s, nb=nb, d=d: [(s * 1000 + k, d if k < nb else 0) for k in range(T.SHARES)]
                report("accepted, %3d %% of the shares %2d steps old" % (round(100 * frac), d), *run(X, Tg, a.steps, a.clip, fn, max_delay=d))
        for n_eff in (16, 8, 4, 1):
            fn = lambda s, n=n_eff: [(s * 1000 + k, 0) for k in range(n)]
            report("stale shares rejected, %2d shares per step left" % n_eff, *run(X, Tg, a.steps, a.clip, fn))
    elif a.experiment == "targeted":
        # the attacker picks, among M candidate nonces per share, the one whose 16 samples contain most images of `target`
        target = 3
        report("honest", *run(X, Tg, a.steps, a.clip, honest), target)
        for k in (8, 16):
            for M in (1, 256, 4096):
                def fn(step, k=k, M=M):
                    plan = honest(step)
                    for j in range(k):
                        cand = (step * 1000 + 500 + j * 4096 + np.arange(M)).astype(np.uint64)
                        count = (y[batch_indices(cand, len(X))] == target).sum(1)
                        plan[j] = (int(cand[int(np.argmax(count))]), 0)
                    return plan
                report("%2d of 32 shares grind %4d nonces for class %d" % (k, M, target), *run(X, Tg, a.steps, a.clip, fn), target)
    else:
        report("honest", *run(X, Tg, a.steps, a.clip, honest))
        for k, M in ((16, 64),):
            W1, W2 = T.train([[s * 1000 + j for j in range(T.SHARES)] for s in range(a.steps)], np.float64, X, Tg, a.clip, M, k)
            report("%2d of 32 shares grind %3d nonces against the gradient" % (k, M), W1, W2)


if __name__ == "__main__":
    main()
