#!/usr/bin/env python3
"""Training shares: mining shares as integer training steps of a chain-owned model (simulation).

The model is a 784-128-10 network with 4-bit signed weights and activations. Every share nonce selects 16 MNIST
images; 32 shares form one model step. The contribution of each share is clipped, an integer residual keeps the small
gradients (bounded to +-2^SU, spec/shares.md section 6), and the model is a pure function of the share nonces.

    python tools/training_shares.py                   # 3000 steps = 96,000 share nonces
    python tools/training_shares.py --steps 1000 --clip 0
    python tools/training_shares.py --steps 300 --attack 32     # 8 of 32 shares grind 32 nonces each

Needs numpy and network access (MNIST from the CVDF mirror). See docs/whitepaper.md.
"""
import argparse
import gzip
import struct
import urllib.request

import numpy as np

# Simulation only, not the protocol: nonces select samples with splitmix64 instead of ChaCha8 (spec/shares.md section 5.1) and
# the initial model is Gaussian instead of the seeded state of section 4. The step itself follows sections 5.2 and 6.
BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
# S1, S2, SB, SU1, SU2 of spec/shares.md section 2
CFG = dict(S1=5, S2=5, SU1=15, SU2=12, Sb=3)   # shifts of the forward pass, the error and the update
PER_SHARE, SHARES, HIDDEN = 16, 32, 128


def mnist(name, images):
    raw = gzip.decompress(urllib.request.urlopen(BASE + name, timeout=120).read())
    if images:
        n, r, c = struct.unpack(">III", raw[4:16])
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, r * c)
    return np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)


def splitmix64(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def share_batch(nonce, n):
    j = np.arange(PER_SHARE, dtype=np.uint64)
    return (splitmix64(np.uint64(nonce) * np.uint64(PER_SHARE) + j) % np.uint64(n)).astype(np.int64)


# The float64 back end is exact: every value is an integer below 2^53 and division by a power of two is exact, so
# floor and trunc equal the integer shift and the quotient toward zero.
def rshift(x, s, dt):
    return (x + (1 << (s - 1))) // (1 << s) if dt == np.int64 else np.floor((x + (1 << (s - 1))) / (1 << s))


def tdiv(r, s, dt):
    return np.sign(r) * (np.abs(r) >> s) if dt == np.int64 else np.trunc(r / (1 << s))


def share_gradients(W1, W2, Xb, Tb, dt):
    """Integer gradients of every share: Xb (shares, 16, 784), Tb (shares, 16, 10)."""
    y1 = rshift(Xb @ W1.T, CFG["S1"], dt)
    a1 = np.clip(y1, -8, 7)
    y2 = np.clip(rshift(a1 @ W2.T, CFG["S2"], dt), -8, 7)
    e2 = np.clip(y2 - Tb, -8, 7)
    g2 = np.einsum("sbo,sbh->soh", e2, a1)
    e1 = np.clip(rshift(e2 @ W2, CFG["Sb"], dt), -8, 7) * ((y1 >= -8) & (y1 <= 7))
    g1 = np.einsum("sbh,sbi->shi", e1, Xb)
    return g1, g2


def initial_model(dt, n_in=784):
    r = np.random.default_rng(0)
    w1 = np.clip(np.round(r.normal(0, 2, (HIDDEN, n_in))), -8, 7).astype(dt)
    w2 = np.clip(np.round(r.normal(0, 2, (10, HIDDEN))), -8, 7).astype(dt)
    return w1, w2


def train(step_nonces, dt, X, T, clip, attack=0, n_bad=8):
    """The chain state as a pure function of the share nonces (and, for the simulation, of an optional attacker)."""
    W1, W2 = initial_model(dt, X.shape[1])
    R1, R2 = np.zeros_like(W1), np.zeros_like(W2)
    Xd, Td = X.astype(dt), T.astype(dt)
    for step, nonces in enumerate(step_nonces):
        nonces = list(nonces)
        if attack > 1:
            nonces = grind(W1, W2, step, nonces, n_bad, attack, Xd, Td, clip, dt)
        bi = np.stack([share_batch(n, len(X)) for n in nonces])
        g1, g2 = share_gradients(W1, W2, Xd[bi], Td[bi], dt)
        if clip:
            g1, g2 = np.clip(g1, -clip, clip), np.clip(g2, -clip, clip)
        # model step of spec/shares.md section 6, applied at once (no delay)
        R1, R2 = R1 + g1.sum(0), R2 + g2.sum(0)
        s1 = np.clip(tdiv(R1, CFG["SU1"], dt), -1, 1)
        s2 = np.clip(tdiv(R2, CFG["SU2"], dt), -1, 1)
        R1, R2 = R1 - s1 * (1 << CFG["SU1"]), R2 - s2 * (1 << CFG["SU2"])
        R1, R2 = np.clip(R1, -(1 << CFG["SU1"]), 1 << CFG["SU1"]), np.clip(R2, -(1 << CFG["SU2"]), 1 << CFG["SU2"])
        W1, W2 = np.clip(W1 - s1, -8, 7), np.clip(W2 - s2, -8, 7)
    return W1, W2


def grind(W1, W2, step, nonces, n_bad, tries, Xd, Td, clip, dt):
    """Attacker: for each of its shares, tries `tries` nonces and keeps the one whose gradient points the most
    against the honest direction (estimated on 1024 images)."""
    # honest direction: the gradient of 1024 random images, divided by 64 = 1024 / 16 to the scale of one share
    ref = np.random.default_rng(10_000 + step).integers(0, len(Xd), size=1024)
    r1, r2 = (g[0] / 64.0 for g in share_gradients(W1, W2, Xd[ref][None], Td[ref][None], dt))
    for k in range(n_bad):
        cand = np.array([step * 1000 + 500 + k * 4096 + j for j in range(tries)])
        bi = np.stack([share_batch(n, len(Xd)) for n in cand])
        g1, g2 = share_gradients(W1, W2, Xd[bi], Td[bi], dt)
        if clip:
            g1, g2 = np.clip(g1, -clip, clip), np.clip(g2, -clip, clip)
        nonces[k] = int(cand[int(np.argmin((g1 * r1[None]).sum((1, 2)) + (g2 * r2[None]).sum((1, 2))))])
    return nonces


def accuracy(W1, W2, X, y):
    a1 = np.clip(np.floor((X @ W1.T + (1 << (CFG["S1"] - 1))) / (1 << CFG["S1"])), -8, 7)
    return float(((a1 @ W2.T).argmax(1) == y).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--clip", type=int, default=256, help="per-share clip of the gradient entries, 0 for none")
    p.add_argument("--attack", type=int, default=0, help="nonces tried by each of 8 attacker shares per step")
    p.add_argument("--no-replay-check", action="store_true")
    a = p.parse_args()
    # pixel = floor(x / 32), 0..7, as in the dataset file of tools/make_mnist_dataset.py
    quant = lambda x: np.clip(np.floor(x.astype(np.float64) / 32.0), 0, 7)
    X = quant(mnist("train-images-idx3-ubyte.gz", True)); y = mnist("train-labels-idx1-ubyte.gz", False)
    Xt = quant(mnist("t10k-images-idx3-ubyte.gz", True)); yt = mnist("t10k-labels-idx1-ubyte.gz", False)
    T = np.full((len(y), 10), -8.0)
    T[np.arange(len(y)), y] = 7.0
    nonces = [[s * 1000 + k for k in range(SHARES)] for s in range(a.steps)]
    W1, W2 = train(nonces, np.float64, X, T, a.clip, a.attack)
    print("%d steps, %d share nonces (%d KiB), clip %s, attacker tries %d: MNIST test accuracy %.1f %%"
          % (a.steps, a.steps * SHARES, a.steps * SHARES * 8 // 1024, a.clip or "none", a.attack, 100 * accuracy(W1, W2, Xt, yt)))
    if not a.no_replay_check:
        u = train(nonces[:60], np.float64, X, T, a.clip)
        v = train(nonces[:60], np.int64, X, T, a.clip)
        print("float64 and int64 back ends identical after 60 steps:", bool(np.array_equal(u[0], v[0]) and np.array_equal(u[1], v[1])))


if __name__ == "__main__":
    main()
