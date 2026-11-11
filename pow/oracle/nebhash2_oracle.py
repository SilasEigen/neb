#!/usr/bin/env python3
"""Independent Python oracle of spec/nebhash-v2.md: the architectures, the initial state, a share (forward pass,
gradient, digest, ticket), the model step and the state hashes, written from the spec with numpy integer arithmetic.
It shares no code with pow/ref/nebhash2.c; the primitives (BLAKE3, ChaCha8) come from primitives.py.

    python3 -B -I pow/oracle/nebhash2_oracle.py --check vectors/nebhash2.txt
    python3 -B -I pow/oracle/nebhash2_oracle.py --check vectors/nebhash2_s1.txt --su-drop 6
"""
import importlib.util
import os
import sys

import numpy as np

NB, CLIP, TPOS, TNEG, NOUT = 16, 256, 7, -8, 2
N_TEST = 2048
STEP_SHARES = (32, 1, 3, 32, 17, 2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nebhash2_prim", os.path.join(_HERE, "primitives.py"))
_prim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prim)
chacha8_stream, blake3 = _prim.chacha8_stream, _prim.blake3


class Layer:
    """kind conv1d / conv2d / linear; src: indices of the layers it reads (empty: the input span [off, off + size))"""

    def __init__(self, kind, cin, cout, k=1, stride=1, pad=0, pool=1, s=0, sb=0, su=15, in_len=1, off=0, src=()):
        self.kind, self.cin, self.cout, self.k, self.stride, self.pad, self.pool = kind, cin, cout, k, stride, pad, pool
        self.s, self.sb, self.su, self.in_len, self.off, self.src = s, sb, su, in_len, off, tuple(src)
        if kind == "linear":
            self.shape, self.conv_len, self.out_len = (cout, cin), 1, 1
        else:
            self.conv_len = (in_len + 2 * pad - k) // stride + 1
            self.out_len = self.conv_len // pool
            self.shape = (cout, cin, k) if kind == "conv1d" else (cout, cin, k, k)
        self.nw = int(np.prod(self.shape))

    def in_shape(self):
        return {"linear": (self.cin,), "conv1d": (self.cin, self.in_len), "conv2d": (self.cin, self.in_len, self.in_len)}[self.kind]

    def out_size(self):
        return self.cout * (self.out_len if self.kind == "conv1d" else self.out_len ** 2 if self.kind == "conv2d" else 1)


# section 3
ARCHS = {
    1: (784, [Layer("conv1d", 1, 8, 5, 1, 2, 4, s=2, su=15, in_len=583, off=0),
              Layer("conv1d", 8, 16, 5, 1, 2, 4, s=4, sb=5, su=15, in_len=145, src=[0]),
              Layer("conv1d", 1, 8, 5, 1, 2, 4, s=3, su=15, in_len=201, off=583),
              Layer("conv1d", 8, 16, 5, 1, 2, 4, s=4, sb=4, su=15, in_len=50, src=[2]),
              Layer("linear", 768, 32, s=5, sb=5, su=15, src=[1, 3]),
              Layer("linear", 32, 2, s=4, sb=3, su=12, src=[4])]),
    2: (4096, [Layer("conv2d", 1, 8, 3, 2, 1, 2, s=3, su=16, in_len=64, off=0),
               Layer("conv2d", 8, 16, 3, 1, 1, 2, s=4, sb=6, su=16, in_len=16, src=[0]),
               Layer("conv2d", 16, 32, 3, 1, 1, 2, s=5, sb=6, su=16, in_len=8, src=[1]),
               Layer("linear", 512, 32, s=5, sb=5, su=16, src=[2]),
               Layer("linear", 32, 2, s=4, sb=3, su=13, src=[3])]),
}


def le32(v):
    return int(v & 0xFFFFFFFF).to_bytes(4, "little")


def rshift(x, n):
    return x if n == 0 else (x + (1 << (n - 1))) >> n


def init_state(arch, seed):
    _, layers = ARCHS[arch]
    total = sum(l.nw for l in layers)
    b = np.frombuffer(chacha8_stream(blake3(b"neb-model-init" + seed), total), dtype=np.uint8).astype(np.int64)
    w = (b & 3) + ((b >> 2) & 3) + ((b >> 4) & 3) + ((b >> 6) & 3) - 6
    ws, at = [], 0
    for l in layers:
        ws.append(w[at:at + l.nw].reshape(l.shape).copy())
        at += l.nw
    return ws, [np.zeros(l.shape, dtype=np.int64) for l in layers]


def _patches(a, l):
    """a: (B, cin, n) or (B, cin, n, n). Returns the input under each kernel tap at each output position:
    conv1d (B, cin, k, P), conv2d (B, cin, k, k, P, P), zero outside the input."""
    st, k, p, cl = l.stride, l.k, l.pad, l.conv_len
    if l.kind == "conv1d":
        ap = np.pad(a, ((0, 0), (0, 0), (p, p)))
        return np.stack([ap[:, :, t:t + st * (cl - 1) + 1:st] for t in range(k)], axis=2)
    ap = np.pad(a, ((0, 0), (0, 0), (p, p), (p, p)))
    rows = []
    for ty in range(k):
        rows.append(np.stack([ap[:, :, ty:ty + st * (cl - 1) + 1:st, tx:tx + st * (cl - 1) + 1:st] for tx in range(k)], axis=2))
    return np.stack(rows, axis=2)


def _pool(act, l):
    """first maximum of each window in row-major order; returns pooled values and a one-hot routing mask"""
    B, c = act.shape[:2]
    p, ol = l.pool, l.out_len
    if l.kind == "conv1d":
        w = act[:, :, :ol * p].reshape(B, c, ol, p)
        arg = w.argmax(-1)
        route = np.zeros_like(act)
        hot = (np.arange(p) == arg[..., None]).astype(np.int64).reshape(B, c, ol * p)
        route[:, :, :ol * p] = hot
        return w.max(-1), route
    w = act[:, :, :ol * p, :ol * p].reshape(B, c, ol, p, ol, p).transpose(0, 1, 2, 4, 3, 5).reshape(B, c, ol, ol, p * p)
    arg = w.argmax(-1)
    hot = (np.arange(p * p) == arg[..., None]).astype(np.int64).reshape(B, c, ol, ol, p, p)
    route = np.zeros_like(act)
    route[:, :, :ol * p, :ol * p] = hot.transpose(0, 1, 2, 4, 3, 5).reshape(B, c, ol * p, ol * p)
    return w.max(-1), route


def share_gradient(arch, ws, x, labels):
    """x: (16, nin) values 0..7, labels: 16 classes. Returns the gradient of each layer (summed, not clipped)."""
    _, layers = ARCHS[arch]
    B = x.shape[0]
    ins, ys, routes, outs = [], [], [], []
    for i, l in enumerate(layers):
        if not l.src:
            a = x[:, l.off:l.off + int(np.prod(l.in_shape()))].reshape(B, *l.in_shape())
        else:
            a = np.concatenate([outs[j].reshape(B, -1) for j in l.src], axis=1).reshape(B, *l.in_shape())
        if l.kind == "linear":
            z = a @ ws[i].T
        elif l.kind == "conv1d":
            z = np.einsum("oct,bctp->bop", ws[i], _patches(a, l))
        else:
            z = np.einsum("ocyx,bcyxpq->bopq", ws[i], _patches(a, l))
        y = rshift(z, l.s)
        act = np.clip(y, -8, 7)
        if l.pool > 1:
            out, route = _pool(act, l)
        else:
            out, route = act, None
        ins.append(a)
        ys.append(y)
        routes.append(route)
        outs.append(out)
    t = np.full((B, NOUT), TNEG, dtype=np.int64)
    t[np.arange(B), labels] = TPOS
    errs = [None] * len(layers)
    errs[-1] = np.clip(outs[-1] - t, -8, 7)
    grads = [None] * len(layers)
    for i in range(len(layers) - 1, -1, -1):
        l = layers[i]
        e = errs[i]
        if l.pool > 1:
            rep = np.repeat(e, l.pool, axis=2)
            if l.kind == "conv2d":
                rep = np.repeat(rep, l.pool, axis=3)
            full = np.zeros_like(ys[i])
            if l.kind == "conv1d":
                full[:, :, :rep.shape[2]] = rep
            else:
                full[:, :, :rep.shape[2], :rep.shape[3]] = rep
            d = full * routes[i]
        else:
            d = e
        if i != len(layers) - 1:
            d = d * ((ys[i] >= -8) & (ys[i] <= 7))
        a = ins[i]
        if l.kind == "linear":
            grads[i] = d.T @ a
            m = d @ ws[i] if l.src else None
        elif l.kind == "conv1d":
            pa = _patches(a, l)
            grads[i] = np.einsum("bop,bctp->oct", d, pa)
            m = None
            if l.src:
                mp = np.zeros((B, l.cin, l.in_len + 2 * l.pad), dtype=np.int64)
                contrib = np.einsum("oct,bop->bctp", ws[i], d)
                for tap in range(l.k):
                    mp[:, :, tap:tap + l.stride * (l.conv_len - 1) + 1:l.stride] += contrib[:, :, tap]
                m = mp[:, :, l.pad:l.pad + l.in_len]
        else:
            pa = _patches(a, l)
            grads[i] = np.einsum("bopq,bcyxpq->ocyx", d, pa)
            m = None
            if l.src:
                n2 = l.in_len + 2 * l.pad
                mp = np.zeros((B, l.cin, n2, n2), dtype=np.int64)
                contrib = np.einsum("ocyx,bopq->bcyxpq", ws[i], d)
                e_ = l.stride * (l.conv_len - 1) + 1
                for ty in range(l.k):
                    for tx in range(l.k):
                        mp[:, :, ty:ty + e_:l.stride, tx:tx + e_:l.stride] += contrib[:, :, ty, tx]
                m = mp[:, :, l.pad:l.pad + l.in_len, l.pad:l.pad + l.in_len]
        if l.src:
            ep = np.clip(rshift(m.reshape(B, -1), l.sb), -8, 7)
            at = 0
            for j in l.src:
                n = layers[j].out_size()
                errs[j] = ep[:, at:at + n].reshape(outs[j].shape)
                at += n
    return grads


def digest_of(grads):
    lanes = [0] * 8
    for li, g in enumerate(grads):
        c = (g.ravel() & 0xFFFFFFFF).astype(np.uint64)
        f = np.arange(c.size, dtype=np.uint64)
        v = c ^ (f + np.uint64(7919 * (li + 1)))
        for k in range(8):
            lanes[k] = (lanes[k] + int(v[k::8].sum())) & 0xFFFFFFFF
    return b"".join(le32(x) for x in lanes)


class TestDataset:
    """section 8: P_c = ChaCha8(BLAKE3("neb-syn2-proto" || arch || c)) & 7, x = clip(P_(i mod 2) + (n & 3) - 1, 0, 7)"""

    def __init__(self, arch):
        self.arch, self.nin, self.n = arch, ARCHS[arch][0], N_TEST
        self.proto = [np.frombuffer(chacha8_stream(blake3(b"neb-syn2-proto" + bytes([arch, c])), self.nin),
                                    dtype=np.uint8).astype(np.int64) & 7 for c in range(2)]

    def get(self, i):
        noise = np.frombuffer(chacha8_stream(blake3(b"neb-syn2-sample" + bytes([self.arch]) + le32(i)), self.nin),
                              dtype=np.uint8).astype(np.int64)
        return np.clip(self.proto[i % 2] + (noise & 3) - 1, 0, 7), i % 2


def share_indices(key, n):
    u = chacha8_stream(key, 4 * NB)
    return [(int.from_bytes(u[4 * j:4 * j + 4], "little") * n) >> 32 for j in range(NB)]


def share(arch, ws, ds, key):
    idx = share_indices(key, ds.n)
    xs, labels = zip(*(ds.get(i) for i in idx))
    grads = [np.clip(g, -CLIP, CLIP) for g in share_gradient(arch, ws, np.stack(xs), np.array(labels))]
    d = digest_of(grads)
    return grads, d, blake3(d + key), idx


def apply(arch, ws, rs, sums, su_drop=0):
    _, layers = ARCHS[arch]
    for i, l in enumerate(layers):
        su = l.su - su_drop
        v = rs[i] + sums[i]
        q = np.sign(v) * (np.abs(v) >> su)
        dlt = np.clip(q, -1, 1)
        rs[i] = np.clip(v - dlt * (1 << su), -(1 << su), 1 << su)
        ws[i] = np.clip(ws[i] - dlt, -8, 7)


def hashes(ws, rs):
    w = np.concatenate([x.ravel() for x in ws]) & 15
    packed = (w[0::2] | (w[1::2] << 4)).astype(np.uint8).tobytes()
    r = np.concatenate([x.ravel() for x in rs]).astype(np.int64) & 0xFFFFFFFF
    return blake3(b"neb-model-weights" + packed), blake3(b"neb-model-residual" + r.astype("<u4").tobytes())


def share_key(arch, s, j):
    return blake3(b"neb-test-share2" + bytes([arch]) + le32(s) + le32(j))


def check(path, su_drop=0):
    lines = open(path, encoding="ascii").read().split("\n")
    assert lines[0] == "neb-nebhash2-vectors 1", "unknown vector file"
    rows = [ln.split() for ln in lines[1:] if ln]
    pos, shares, states = 0, 0, 0
    for arch in (1, 2):
        ws, rs = init_state(arch, bytes(32))
        wh, rh = hashes(ws, rs)
        tag, a, h1, h2 = rows[pos]
        assert (tag, int(a), h1, h2) == ("init", arch, wh.hex(), rh.hex()), f"init of arch {arch}"
        pos += 1
        ds = TestDataset(arch)
        for s, n in enumerate(STEP_SHARES):
            sums = [np.zeros_like(w) for w in ws]
            for j in range(n):
                grads, dig, tic, idx = share(arch, ws, ds, share_key(arch, s, j))
                row = rows[pos]
                assert row[:4] == ["share", str(arch), str(s), str(j)], f"line {pos + 2}"
                assert row[4] == dig.hex() and row[5] == tic.hex(), f"share {arch} {s} {j}: digest or ticket"
                assert [int(v) for v in row[6:]] == idx, f"share {arch} {s} {j}: indices"
                for k, g in enumerate(grads):
                    sums[k] += g
                pos += 1
                shares += 1
            apply(arch, ws, rs, sums, su_drop)
            wh, rh = hashes(ws, rs)
            assert rows[pos] == ["state", str(arch), str(s), str(n), wh.hex(), rh.hex()], f"state {arch} {s}"
            pos += 1
            states += 1
    assert pos == len(rows), "trailing lines"
    print(f"PASS {path}: {shares} shares and {states} states replayed by the Python oracle")


def main(argv):
    if len(argv) >= 2 and argv[0] == "--check":
        drop = int(argv[argv.index("--su-drop") + 1]) if "--su-drop" in argv else 0
        check(argv[1], drop)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
