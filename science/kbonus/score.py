#!/usr/bin/env python3
"""Score of one star by a Kepler model of the chain: the same 784 values as the hunt, then one forward pass.

    python score.py kic-006691030                          # the testnet model of the results (block 1428)
    python score.py kic-006691030 --model results/neb-kepler-mainnet-block10570.json
    python score.py kic-006691030 --rpc URL [--save FILE]   # slot 0 of a node, in the state of its tip
    python score.py kic-006691030 --model FILE --test kepler-test-q8.bin

The light curve is cleaned and searched as in hunt.py, so the period, epoch and duration are those of the hunt. A
.npz model is the 784-128-10 network of fetch_model.py; its score is the planet output minus the false-positive
output, from -15 to +15. A .json model is the reply of the getmodel RPC: the weights are checked against the weights
hash, and for the convolutional network (dataset kepler-v2, spec/shares.md section 11.7) the score is the margin
between the two raw output sums, which is what the prediction of the protocol compares. The two scores are not on
the same scale.

With --test, the same model scores the test file of the chain (signals kept out of the training), to show where the
score of the star falls among known planets and known false positives.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "pow", "oracle"))
import kb_common as K  # noqa: E402
import vet  # noqa: E402
from fetch_model import NH, NIN, NOUT, rpc, unpack  # noqa: E402

ANSWERS = ("false positive", "planet candidate")
V2_ARCH = {"kepler-v2": 1}  # dataset name -> architecture of pow/oracle/nebhash2_oracle.py


def light_curve(source):
    """the Kepler Bonus light curve of the star, downloaded once"""
    path = os.path.join(K.LC_DIR, K.lc_name(source))
    if os.path.exists(path):
        return path
    with open(K.file_list()) as f:
        for line in f:
            m = K.LC_PATTERN.search(line)
            if m and m.group(2) == source:
                return K.fetch(m.group(1), source)
    return None


def views(source):
    """the 784 values of the star, with the ephemeris found by the search of the hunt"""
    path = light_curve(source)
    if path is None:
        raise SystemExit("no Kepler Bonus light curve for %s" % source)
    t, f = K.read_lc(path)
    keep = vet.clean_mask(t, f)
    t, f = t[keep], f[keep]
    p, t0, dur = vet.search(t, f)
    print("%s: %d points, P = %.5f d, t0 = %.4f, duration %.2f h" % (source, len(t), p, t0, dur * 24))
    return K.kepler_views(t, f, p, t0, dur * 24)


def load_test(path):
    """inputs and labels of a test file of the chain (NMD1, or NMD2 of architecture 1): 784 values and a class"""
    raw = np.fromfile(path, dtype=np.uint8)
    head, n = (12, int.from_bytes(raw[8:12].tobytes(), "little")) if raw[:4].tobytes() == b"NMD2" else (
        8, int.from_bytes(raw[4:8].tobytes(), "little"))
    rec = raw[head:head + n * (NIN + 1)].reshape(n, NIN + 1)
    return rec[:, :NIN].astype(np.int64), rec[:, NIN].astype(np.int64)


def v2_sums(arch, ws, x):
    """raw output sums of the last layer, for inputs x (B, 784): the integer forward pass of nebhash2_oracle"""
    import nebhash2_oracle as O
    layers = O.ARCHS[arch][1]
    b, outs, z = x.shape[0], [], None
    for i, l in enumerate(layers):
        if not l.src:
            a = x[:, l.off:l.off + int(np.prod(l.in_shape()))].reshape(b, *l.in_shape())
        else:
            a = np.concatenate([outs[j].reshape(b, -1) for j in l.src], axis=1).reshape(b, *l.in_shape())
        z = a @ ws[i].T if l.kind == "linear" else np.einsum("oct,bctp->bop", ws[i], O._patches(a, l))
        act = np.clip(O.rshift(z, l.s), -8, 7)
        outs.append(O._pool(act, l)[0] if l.pool > 1 else act)
    return z


def read_model(a):
    """(name, scores, exact): scores(x) gives one score per row of x (B, 784), positive for a planet; exact when
    the sign of the score is the prediction of the protocol"""
    if a.rpc or a.model.endswith(".json"):
        from primitives import blake3
        if a.rpc:
            m = rpc(a.rpc, "getmodel", [0])
        else:
            m = json.load(open(a.model))
            m = m.get("result", m)
        packed = bytes.fromhex(m["weights"])
        if blake3(b"neb-model-weights" + packed).hex() != m["weightshash"]:
            raise SystemExit("the weights do not match their weights hash")
        if a.rpc and a.save:
            with open(a.save, "w") as f:
                json.dump(m, f, separators=(",", ":"))
        name = "%s, block %d, weights hash %s..." % (m["dataset"], m["height"], m["weightshash"][:16])
        w = unpack(packed).astype(np.int64)
        if m["dataset"] in V2_ARCH:
            import nebhash2_oracle as O
            arch, ws, at = V2_ARCH[m["dataset"]], [], 0
            for l in O.ARCHS[arch][1]:
                ws.append(w[at:at + l.nw].reshape(l.shape))
                at += l.nw

            def scores(x):
                z = np.concatenate([v2_sums(arch, ws, x[k:k + 256]) for k in range(0, len(x), 256)])
                return z[:, 1] - z[:, 0]
            return name + ", margin of the output sums", scores, True
        if m["dataset"] != "kepler":
            raise SystemExit("slot 0 holds the dataset %s: no Kepler model" % m["dataset"])
        w1, w2 = w[:NH * NIN].reshape(NH, NIN), w[NH * NIN:NH * NIN + NOUT * NH].reshape(NOUT, NH)
    else:
        w1, w2, height, weightshash = K.load_model(a.model)
        name = "kepler, block %d, weights hash %s..." % (height, weightshash[:16])

    def scores(x):
        o = K.outputs(w1, w2, x)
        return o[:, 1] - o[:, 0]
    return name + ", planet output minus false-positive output", scores, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="a Kepler Bonus source, as kic-006691030")
    p.add_argument("--model", default=K.MODEL, help="a .npz of fetch_model.py or a getmodel reply in .json")
    p.add_argument("--rpc", help="read slot 0 from this node instead of --model")
    p.add_argument("--save", help="with --rpc: keep the reply of the node in this .json")
    p.add_argument("--test", help="test file of the chain, kepler-test-q8.bin")
    a = p.parse_args()

    x = views(a.source).astype(np.int64)[None]
    name, scores, exact = read_model(a)
    mine = int(scores(x)[0])
    print("model: %s" % name)
    print("answer: %s, score %+d" % (ANSWERS[mine > 0], mine))
    if a.test:
        px, lab = load_test(a.test)
        s = scores(px)
        acc = ", accuracy %.1f%%" % (100 * np.mean((s > 0) == (lab == 1))) if exact else ""
        print("test file: %d signals%s" % (len(lab), acc))
        for c in (1, 0):
            print("  %ss: %d, median score %+d, %.1f%% at %+d or more" % (
                ANSWERS[c].split()[0] if c else "false positive", int((lab == c).sum()), int(np.median(s[lab == c])),
                100 * np.mean(s[lab == c] >= mine), mine))


if __name__ == "__main__":
    main()
