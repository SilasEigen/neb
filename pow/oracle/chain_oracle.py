#!/usr/bin/env python3
"""Independent replay of a chain fixture (pow/tests/chains/*.txt) with the Python oracle of the training shares.

Usage:
  python3 -B -I pow/oracle/chain_oracle.py FIXTURE [--dataset FILE]

Written from spec/shares.md and spec/node.md only, on top of shares_oracle.py (which shares no code with the C
reference). For every block it checks: the parent link, the stand-in body root, the proof of work (the ticket of
the block on the model of its parent is not above its target), every share of the list (the ticket on the model of
the grandparent is not above the share target; a share is a record of 40 bytes, the share header of which has the
parent of the previous block, spec/shares.md section 11.2), the model step, the weights and residual hashes, and the losses of
the step recomputed entry by entry (cut residual, clipped weight). A mismatch prints `ORACLE FAILED` with the
block and exits 1. `--dataset FILE` gives the sample file of a fixture that was recorded on a file dataset; its
BLAKE3 hash must be the one the fixture names.
"""
import importlib.util
import os
import sys

import numpy as np


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = os.path.dirname(os.path.abspath(__file__))
O = _load("shares_oracle", os.path.join(HERE, "shares_oracle.py"))
SHAPE1, SHAPE2 = (O.NH, O.NIN), (O.NOUT, O.NH)
SU = (O.SU1, O.SU2)
BATCH = 32


class Failed(Exception):
    pass


def le32(v):
    return int(v).to_bytes(4, "little")


def parse(path):
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if lines[0] != "neb-chain-fixture 2":
        raise Failed("not a neb-chain-fixture 2 file")
    meta = {}
    for l in lines[1:4]:
        k, v = l.split(" ", 1)
        meta[k] = v
    blocks = []
    for l in lines[4:]:
        f = l.split(" ")
        if len(f) not in (12, 21) or f[0] != "block" or f[2] != "pool" or f[4] != "header" or f[6] != "list" or f[8] != "wh" or f[10] != "rh":
            raise Failed("malformed block line: " + l[:80])
        loss = None
        if len(f) == 21:
            v = [int(x) for x in f[13:]]
            loss = (tuple(v[:4]), tuple(v[4:]))
        records = [] if f[7] == "-" else [bytes.fromhex(x) for x in f[7].split(",")]
        if any(len(r) != 40 for r in records):
            raise Failed("a share record has not 40 bytes: " + l[:80])
        blocks.append({
            "height": int(f[1]), "pool": int(f[3]), "header": bytes.fromhex(f[5]), "list": records,
            "wh": f[9], "rh": f[11], "loss": loss,
        })
    return meta, bytes.fromhex(meta["seed"]), blocks


def target_of(bits):
    shift, mant = bits >> 24, bits & 0xFFFFFF
    if shift > 232 or not 0x800000 <= mant <= 0xFFFFFF:
        raise Failed("bits %08x is not a normalised compact target" % bits)
    return mant << shift


def share_target_of(bits):
    return min(target_of(bits) << 5, (1 << 256) - 1)


def fields(header):
    return {"parent": header[4:36], "body_root": header[36:68], "time": int.from_bytes(header[68:72], "little"),
            "bits": int.from_bytes(header[72:76], "little"), "nonce": int.from_bytes(header[76:80], "little")}


def stand_in_body_root(height, pool, lst):
    tx = O.blake3(b"sim-coinbase" + le32(height) + (le32(pool) if pool > 0 else b""))
    leaf = O.blake3(b"leaf" + tx)
    sroot = O.blake3(b"".join(lst))
    return O.blake3(b"body" + le32(1) + leaf + sroot)


class Samples:
    """pixels (n x 784) and labels of the dataset, as int64 rows"""
    def __init__(self, kind, path):
        if kind == "synthetic":
            ds = O.TestDataset()
            self.n = ds.n
            self.px = np.stack([ds.sample(i)[0] for i in range(ds.n)]).astype(np.int64)
            self.lab = np.array([ds.sample(i)[1] for i in range(ds.n)])
        else:
            raw = np.fromfile(path, dtype=np.uint8)
            if raw[:4].tobytes() != b"NMD1":
                raise Failed("bad dataset magic")
            self.n = int.from_bytes(raw[4:8].tobytes(), "little")
            rec = raw[8:].reshape(self.n, O.NIN + 1)
            self.px = rec[:, :O.NIN].astype(np.int64)
            self.lab = rec[:, O.NIN].astype(np.int64)
        self.tvec = np.stack([O.target_vector(c) for c in range(O.NOUT)])


def evaluate(model, samples, keys, want_gradients):
    """tickets (ints) of the keys on the model; with want_gradients also the sums of the clipped gradients"""
    w1, w2 = model[0], model[1]
    tickets = []
    gs1 = np.zeros(SHAPE1, np.int64)
    gs2 = np.zeros(SHAPE2, np.int64)
    for s in range(0, len(keys), BATCH):
        kb = keys[s:s + BATCH]
        idx = [O.share_indices(k, samples.n) for k in kb]
        x = samples.px[np.array(idx)]
        t = samples.tvec[samples.lab[np.array(idx)]]
        g1, g2 = O.share_gradients(w1, w2, x, t)
        lanes = O.digest_lanes(g1, g2)
        for j, k in enumerate(kb):
            tickets.append(O.ticket_int(O.ticket(O.digest_bytes(lanes[j]), k)))
        if want_gradients:
            gs1 += g1.sum(axis=0)
            gs2 += g2.sum(axis=0)
    return tickets, gs1, gs2


def losses(model, gs1, gs2, after):
    """(cut residual entries, clipped weight entries, sum |lr|, sum 2^SU |wc|) of each layer, from the 64-bit step"""
    out = []
    for k, (w, r, g) in enumerate(((model[0], model[2], gs1), (model[1], model[3], gs2))):
        t = 1 << SU[k]
        s = r + g
        d = np.clip(np.sign(s) * (np.abs(s) // t), -1, 1)
        r2 = s - d * t
        r3 = np.clip(r2, -t, t)
        w2 = w - d
        w3 = np.clip(w2, -8, 7)
        if not (np.array_equal(w3, after[k]) and np.array_equal(r3, after[2 + k])):
            raise Failed("layer %d: the step of the oracle differs from apply_sums" % (k + 1))
        lr, wc = r2 - r3, w3 - w2
        out.append((int((lr != 0).sum()), int((wc != 0).sum()), int(np.abs(lr).sum()), int(t * np.abs(wc).sum())))
    return tuple(out)


def replay(path, dataset_path=None):
    meta, seed, blocks = parse(path)
    kind = "synthetic" if meta["dataset"] == "synthetic" else "file"
    if kind == "file":
        if dataset_path is None:
            raise Failed("the fixture needs --dataset FILE (%s)" % meta["dataset"])
        want = meta["dataset"].split(" ")[1]
        got = O.blake3(open(dataset_path, "rb").read()).hex()
        if got != want:
            raise Failed("the dataset file has the wrong BLAKE3 hash %s" % got)
    samples = Samples(kind, dataset_path)
    w1, w2, r1, r2 = O.init_state(seed)
    models = [(w1, w2, r1, r2)]
    prev_id, prev = seed, None
    steps = shares = largest = 0
    total = [[0, 0, 0, 0], [0, 0, 0, 0]]
    for i, b in enumerate(blocks):
        h = i + 1
        tag = "block %d" % h
        if b["height"] != h:
            raise Failed("%s: the fixture says height %d" % (tag, b["height"]))
        f = fields(b["header"])
        if f["parent"] != prev_id:
            raise Failed("%s: the parent is not the previous block" % tag)
        if f["body_root"] != stand_in_body_root(h, b["pool"], b["list"]):
            raise Failed("%s: body_root is not the stand-in root of the list" % tag)
        lst = b["list"]
        if len(lst) > 256 or any(a >= c for a, c in zip(lst, lst[1:])):
            raise Failed("%s: the list has more than 256 records or is not strictly increasing (as byte strings)" % tag)
        if h == 1 and lst:
            raise Failed("%s: the block of height 1 lists no share" % tag)
        model_p = models[h - 1]
        model_pp = models[h - 2] if h >= 2 else model_p
        key = O.blake3(b["header"])
        tk, _, _ = evaluate(model_p, samples, [key], False)
        if tk[0] > target_of(f["bits"]):
            raise Failed("%s: the proof of work does not meet the target" % tag)
        if lst:
            pf = fields(prev)
            # the share header of a record: version 1, the parent of the previous block (the round), the body root, time
            # and nonce of the record, and the bits of the previous block
            keys = [O.blake3(le32(1) + pf["parent"] + r[:32] + r[32:36] + le32(pf["bits"]) + r[36:40]) for r in lst]
            tks, gs1, gs2 = evaluate(model_pp, samples, keys, True)
            st = share_target_of(pf["bits"])
            for r, t in zip(lst, tks):
                if t > st:
                    raise Failed("%s: share record %s does not meet the share target" % (tag, r.hex()))
            new = O.apply_sums(model_p[0], model_p[1], model_p[2], model_p[3], gs1, gs2, len(lst))
            lo = losses(model_p, gs1, gs2, new)
            if b["loss"] is not None and tuple(tuple(x) for x in b["loss"]) != lo:
                raise Failed("%s: the losses differ: fixture %s, oracle %s" % (tag, b["loss"], lo))
            for k in range(2):
                for j in range(4):
                    total[k][j] += lo[k][j]
            steps += 1
            shares += len(lst)
            largest = max(largest, len(lst))
        else:
            new = model_p
        wh, rh = O.state_hashes(*new)
        if (wh, rh) != (b["wh"], b["rh"]):
            raise Failed("%s: the state hashes differ from the fixture" % tag)
        prev_id, prev = O.blake3(b["header"]), b["header"]
        models.append(tuple(new))
    return {"name": meta["name"], "blocks": len(blocks), "steps": steps, "shares": shares, "largest_step": largest, "total": total,
            "final": (blocks[-1]["wh"], blocks[-1]["rh"])}


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    dataset = argv[argv.index("--dataset") + 1] if "--dataset" in argv else None
    try:
        r = replay(argv[0], dataset)
    except Failed as e:
        print("ORACLE FAILED: %s" % e)
        return 1
    print("ORACLE OK %s: blocks %d steps %d shares %d largest_step %d" % (r["name"], r["blocks"], r["steps"], r["shares"], r["largest_step"]))
    for k in range(2):
        t = r["total"][k]
        print("  layer %d: cut_events %d clip_events %d lr_abs %d wc_abs %d" % (k + 1, t[0], t[1], t[2], t[3]))
    print("  final weights_hash %s residual_hash %s" % r["final"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
