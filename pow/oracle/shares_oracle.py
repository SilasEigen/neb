#!/usr/bin/env python3
"""Independent oracle for the training shares (spec/shares.md), written from the specification text only.

Usage:
  python3 -I pow/oracle/shares_oracle.py --selftest
  python3 -I pow/oracle/shares_oracle.py --check FILE [--su1 N] [--su2 N]

--check replays vectors/shares.txt (section 10 of the spec) and compares every field. The options
set the update shifts SU1 and SU2 (defaults 15 and 12); vectors/shares_s1.txt (section 10.1) is
replayed with --su1 9 --su2 6. A reported mismatch line reads `step j field expected got`: expected is
the value in the file, got is the value computed here.
Exit code: 0 all good, 1 on any mismatch or malformed file, 2 on a usage error.

All arithmetic is exact integer arithmetic on int64 numpy arrays.
"""
import importlib.util
import itertools
import os
import re
import sys

import numpy as np

# constants (spec section 2)
NIN = 784
NH = 128
NOUT = 10
NB = 16
S1 = 5
S2 = 5
SB = 3
SU1 = 15
SU2 = 12
CLIP = 256
TPOS = 7
TNEG = -8
# largest number of shares of a step: floor((2^31 - 1 - 2^SU1) / CLIP), so that R + GS fits in int32 (section 6)
N_MAX = 8388479

# test vectors (spec sections 9 and 10)
N_TEST = 2048
STEPS = 24
# n_s cycles with the step number s (section 10)
STEP_SHARES = (32, 32, 32, 1, 32, 3, 32, 17)
SEED_ZERO = bytes(32)


# primitives, loaded by absolute path
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HERE = os.path.dirname(os.path.abspath(__file__))
_prim = _load_module("shares_prim_ind", os.path.join(_HERE, "primitives.py"))
chacha8_stream = _prim.chacha8_stream
blake3_short = _prim.blake3_short
blake3_long = _prim.blake3_long
blake3 = _prim.blake3


def le32(x):
    return (x & 0xFFFFFFFF).to_bytes(4, "little")


# integer helpers (spec section 2)
def rshift(x, n):
    """(x + 2^(n-1)) >> n with an arithmetic shift: rounds halves toward +infinity."""
    return (np.asarray(x, dtype=np.int64) + (1 << (n - 1))) >> n


def clip(v, lo, hi):
    return np.minimum(np.maximum(v, lo), hi)


def tdiv(r, u):
    """Quotient of r by 2^u rounded toward zero."""
    r = np.asarray(r, dtype=np.int64)
    return np.sign(r) * (np.abs(r) >> u)


def pack(m):
    """Signed nibbles, element e = row * cols + column, even e in the low nibble (spec/shares.md)."""
    flat = np.asarray(m, dtype=np.int64).reshape(-1) & 15
    assert flat.size % 2 == 0
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8).tobytes()


# initial state (section 4)
def weights_from_bytes(s, nh=NH, nin=NIN, nout=NOUT):
    b = np.frombuffer(s[:nh * nin + nout * nh], dtype=np.uint8).astype(np.int64)
    w = (b & 3) + ((b >> 2) & 3) + ((b >> 4) & 3) + ((b >> 6) & 3) - 6
    return w[:nh * nin].reshape(nh, nin), w[nh * nin:].reshape(nout, nh)


def init_state(seed, nh=NH, nin=NIN, nout=NOUT):
    key = blake3(b"neb-model-init" + seed)
    w1, w2 = weights_from_bytes(chacha8_stream(key, nh * nin + nout * nh), nh, nin, nout)
    return w1, w2, np.zeros_like(w1), np.zeros_like(w2)


# dataset (sections 4 and 9)
def target_vector(label, nout=NOUT):
    t = np.full(nout, TNEG, dtype=np.int64)
    t[label] = TPOS
    return t


def pixels_from_bytes(proto, noise):
    return clip((np.asarray(proto, dtype=np.int64) & 7) + (np.asarray(noise, dtype=np.int64) & 3) - 1, 0, 7)


class TestDataset:
    def __init__(self, n=N_TEST, nin=NIN):
        self.n = n
        self.nin = nin
        self._proto = {}
        self._x = {}

    def proto(self, c):
        if c not in self._proto:
            st = chacha8_stream(blake3(b"neb-syn-proto" + bytes([c])), self.nin)
            self._proto[c] = np.frombuffer(st, dtype=np.uint8).astype(np.int64)
        return self._proto[c]

    def sample(self, i):
        if i not in self._x:
            st = chacha8_stream(blake3(b"neb-syn-sample" + le32(i)), self.nin)
            noise = np.frombuffer(st, dtype=np.uint8).astype(np.int64)
            self._x[i] = pixels_from_bytes(self.proto(i % 10), noise)
        return self._x[i], i % 10


# a share (section 5)
def sample_indices(u, n_samples):
    """u: 64 bytes read as 16 little-endian words; idx_j = floor(u_j * N / 2^32)."""
    return [(int.from_bytes(u[4 * j:4 * j + 4], "little") * n_samples) // (1 << 32) for j in range(len(u) // 4)]


def share_indices(key, n_samples):
    return sample_indices(chacha8_stream(key, 4 * NB), n_samples)


def share_gradients(w1, w2, x, t, s1=S1, s2=S2, sb=SB, bound=CLIP):
    """Clipped gradients of K shares on one model. x: (K, nb, nin), t: (K, nb, nout).
    Returns G1 (K, nh, nin) and G2 (K, nout, nh)."""
    # int64 products: no overflow, so the results equal the int32 sums of the C reference
    z1 = x @ w1.T
    y1 = rshift(z1, s1)
    a1 = clip(y1, -8, 7)
    z2 = a1 @ w2.T
    y2 = clip(rshift(z2, s2), -8, 7)
    e2 = clip(y2 - t, -8, 7)
    g2 = e2.transpose(0, 2, 1) @ a1
    m = e2 @ w2
    # the error passes back only where the layer-1 activation was not clipped
    e1 = np.where((y1 >= -8) & (y1 <= 7), clip(rshift(m, sb), -8, 7), 0)
    g1 = e1.transpose(0, 2, 1) @ x
    return clip(g1, -bound, bound), clip(g2, -bound, bound)


def digest_lanes(g1, g2):
    """Lanes D[0..7] of the digest for K shares: g1 (K, nh, nin), g2 (K, nout, nh). Returns (K, 8) uint32 values."""
    k = g1.shape[0]
    # a lane sums at most 100352 / 8 terms below 2^32, far below 2^63, so int64 holds the sum before the final mod 2^32
    d = np.zeros((k, 8), dtype=np.int64)
    for l, g in ((1, g1), (2, g2)):
        flat = g.reshape(k, -1)
        f = np.arange(flat.shape[1], dtype=np.int64)
        v = (flat & 0xFFFFFFFF) ^ ((f + 7919 * l) & 0xFFFFFFFF)
        for lane in range(8):
            d[:, lane] += v[:, lane::8].sum(axis=1)
    return d & 0xFFFFFFFF


def digest_bytes(lanes):
    return b"".join(le32(int(x)) for x in lanes)


def ticket(digest, key):
    return blake3_short(digest + key)


def ticket_int(t):
    return int.from_bytes(t, "little")


# a model step (section 6)
def apply_sums(w1, w2, r1, r2, gs1, gs2, n, su1=SU1, su2=SU2):
    """Step from the sums GS of the clipped gradients of n shares."""
    if not 1 <= n <= N_MAX:
        raise ValueError("a step takes 1 <= n <= %d shares" % N_MAX)
    out = []
    for w, r, gs, su in ((w1, r1, gs1, su1), (w2, r2, gs2, su2)):
        r = r + gs
        # the oracle checks the int32 range that the C reference gets from n <= N_MAX
        if np.abs(r).max() >= (1 << 31):
            raise OverflowError("R + GS does not fit in int32")
        d = clip(tdiv(r, su), -1, 1)
        r = r - d * (1 << su)
        r = clip(r, -(1 << su), 1 << su)
        out.append((clip(w - d, -8, 7), r))
    (nw1, nr1), (nw2, nr2) = out
    return nw1, nw2, nr1, nr2


def model_step(w1, w2, r1, r2, g1s, g2s, su1=SU1, su2=SU2):
    """g1s: (n, nh, nin) and g2s: (n, nout, nh), the clipped gradients of the n shares of the step."""
    return apply_sums(w1, w2, r1, r2, np.sum(g1s, axis=0), np.sum(g2s, axis=0), len(g1s), su1, su2)


# state identifiers (section 7)
def weights_hash_input(w1, w2):
    return b"neb-model-weights" + pack(w1) + pack(w2)


def residual_hash_input(r1, r2):
    body = b"".join((np.asarray(r, dtype=np.int64).reshape(-1) & 0xFFFFFFFF).astype("<u4").tobytes() for r in (r1, r2))
    return b"neb-model-residual" + body


def state_hashes(w1, w2, r1, r2):
    return blake3(weights_hash_input(w1, w2)).hex(), blake3(residual_hash_input(r1, r2)).hex()


# replay of the test vectors (sections 9 and 10)
def share_key(s, j):
    return blake3(b"neb-test-share" + le32(s) + le32(j))


def run_shares(w1, w2, ds, keys, chunk=32):
    """Indices, digests, tickets of the shares with the given keys, and the sums of their gradients."""
    idxs, digests, tickets = [], [], []
    gs1 = gs2 = None
    for c in range(0, len(keys), chunk):
        ks = keys[c:c + chunk]
        ix = [share_indices(k, ds.n) for k in ks]
        smp = [[ds.sample(i) for i in row] for row in ix]
        x = np.stack([np.stack([p[0] for p in row]) for row in smp])
        t = np.stack([np.stack([target_vector(p[1]) for p in row]) for row in smp])
        g1, g2 = share_gradients(w1, w2, x, t)
        lanes = digest_lanes(g1, g2)
        for k, row, ln in zip(ks, ix, lanes):
            dg = digest_bytes(ln)
            idxs.append(row)
            digests.append(dg)
            tickets.append(ticket(dg, k))
        gs1 = g1.sum(axis=0) if gs1 is None else gs1 + g1.sum(axis=0)
        gs2 = g2.sum(axis=0) if gs2 is None else gs2 + g2.sum(axis=0)
    return idxs, digests, tickets, gs1, gs2


class Malformed(Exception):
    pass


_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_DEC = re.compile(r"(0|[1-9][0-9]*)\Z")


def _hex64(tok, what):
    if not _HEX64.match(tok):
        raise Malformed("%s is not 64 lower-case hex digits" % what)
    return tok


def _dec(tok, what):
    if not _DEC.match(tok):
        raise Malformed("%s is not a decimal number" % what)
    return int(tok)


def parse_vectors(data):
    """Returns (init, steps); init = (wh, rh); steps = list of (shares, state) with
    shares = [(s, j, digest, ticket, [idx])] and state = (s, n, wh, rh)."""
    if b"\r" in data:
        raise Malformed("CR in file (LF endings required)")
    if not data.endswith(b"\n"):
        raise Malformed("missing final LF")
    lines = data.decode("ascii").split("\n")[:-1]
    if not lines or lines[0] != "neb-shares-vectors 1":
        raise Malformed("line 1 is not the header 'neb-shares-vectors 1'")
    if len(lines) < 2:
        raise Malformed("no init line")
    tok = lines[1].split(" ")
    if len(tok) != 3 or tok[0] != "init":
        raise Malformed("line 2 is not an init line")
    init = (_hex64(tok[1], "init weights_hash"), _hex64(tok[2], "init residual_hash"))
    steps = []
    cur = []
    for ln, line in enumerate(lines[2:], 3):
        tok = line.split(" ")
        try:
            if tok[0] == "share":
                if len(tok) != 5 + NB:
                    raise Malformed("share line has %d fields, expected %d" % (len(tok), 5 + NB))
                s = _dec(tok[1], "step")
                j = _dec(tok[2], "share number")
                if j != len(cur):
                    raise Malformed("share number %d, expected %d" % (j, len(cur)))
                cur.append((s, j, _hex64(tok[3], "digest"), _hex64(tok[4], "ticket"),
                            [_dec(t, "index") for t in tok[5:]]))
            elif tok[0] == "state":
                if len(tok) != 5:
                    raise Malformed("state line has %d fields, expected 5" % len(tok))
                steps.append((cur, (_dec(tok[1], "step"), _dec(tok[2], "n"),
                                    _hex64(tok[3], "weights_hash"), _hex64(tok[4], "residual_hash"))))
                cur = []
            else:
                raise Malformed("unknown line type '%s'" % tok[0])
        except Malformed as ex:
            raise Malformed("line %d: %s" % (ln, ex))
    if cur:
        raise Malformed("share lines after the last state line")
    if not steps:
        raise Malformed("no step")
    return init, steps


def replay(init, steps, su1=SU1, su2=SU2):
    """Replays the steps from the seed of 32 zero bytes. Returns (mismatches, details)."""
    mism = []

    def cmp(step, j, field, want, got):
        if want != got:
            mism.append((step, j, field, want, got))

    ds = TestDataset()
    w1, w2, r1, r2 = init_state(SEED_ZERO)
    wh, rh = state_hashes(w1, w2, r1, r2)
    cmp("init", "-", "weights_hash", init[0], wh)
    cmp("init", "-", "residual_hash", init[1], rh)
    cmp("-", "-", "steps", str(STEPS), str(len(steps)))
    for s, (fshares, fstate) in enumerate(steps):
        n = STEP_SHARES[s % len(STEP_SHARES)]
        keys = [share_key(s, j) for j in range(n)]
        idxs, digests, tickets, gs1, gs2 = run_shares(w1, w2, ds, keys)
        cmp(s, "-", "shares_in_file", str(n), str(len(fshares)))
        for j in range(min(n, len(fshares))):
            fs, fj, fd, ft, fi = fshares[j]
            cmp(s, j, "step_label", str(s), str(fs))
            for p in range(NB):
                cmp(s, j, "idx_%d" % p, str(fi[p]), str(idxs[j][p]))
            cmp(s, j, "digest", fd, digests[j].hex())
            cmp(s, j, "ticket", ft, tickets[j].hex())
        w1, w2, r1, r2 = apply_sums(w1, w2, r1, r2, gs1, gs2, n, su1, su2)
        wh, rh = state_hashes(w1, w2, r1, r2)
        cmp(s, "-", "state_label", str(s), str(fstate[0]))
        cmp(s, "-", "state_n", str(n), str(fstate[1]))
        cmp(s, "-", "weights_hash", fstate[2], wh)
        cmp(s, "-", "residual_hash", fstate[3], rh)
    return mism


def check(path, su1=SU1, su2=SU2):
    name = os.path.basename(path)
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as ex:
        print("shares_ind check file=%s unreadable: %s" % (name, ex))
        return 2
    try:
        init, steps = parse_vectors(data)
    except (Malformed, UnicodeDecodeError) as ex:
        print("shares_ind check file=%s malformed: %s" % (name, ex))
        return 1
    mism = replay(init, steps, su1, su2)
    k = sum(len(sh) for sh, _ in steps)
    ok = not mism
    shifts = "" if (su1, su2) == (SU1, SU2) else " su1=%d su2=%d" % (su1, su2)
    print("shares_ind check file=%s%s steps=%d shares=%d mismatches=%d %s"
          % (name, shifts, len(steps), k, len(mism), "OK" if ok else "MISMATCH"))
    for m in mism[:10]:
        print("  %s %s %s %s %s" % m)
    return 0 if ok else 1


# selftest
def _gradient_scalar(w1, w2, x, t, s1=S1, s2=S2, sb=SB, bound=CLIP):
    """Plain-loop transcription of section 5.2 for one share (lists of ints)."""
    nh, nin, nout, nb = len(w1), len(w1[0]), len(w2), len(x)

    def rs(v, n):
        return (v + (1 << (n - 1))) >> n

    def cl(v, lo, hi):
        return min(max(v, lo), hi)

    y1 = [[rs(sum(x[b][i] * w1[h][i] for i in range(nin)), s1) for h in range(nh)] for b in range(nb)]
    a1 = [[cl(v, -8, 7) for v in row] for row in y1]
    y2 = [[cl(rs(sum(a1[b][h] * w2[o][h] for h in range(nh)), s2), -8, 7) for o in range(nout)] for b in range(nb)]
    e2 = [[cl(y2[b][o] - t[b][o], -8, 7) for o in range(nout)] for b in range(nb)]
    g2 = [[sum(e2[b][o] * a1[b][h] for b in range(nb)) for h in range(nh)] for o in range(nout)]
    m = [[sum(e2[b][o] * w2[o][h] for o in range(nout)) for h in range(nh)] for b in range(nb)]
    e1 = [[cl(rs(m[b][h], sb), -8, 7) if -8 <= y1[b][h] <= 7 else 0 for h in range(nh)] for b in range(nb)]
    g1 = [[sum(e1[b][h] * x[b][i] for b in range(nb)) for i in range(nin)] for h in range(nh)]
    return ([[cl(v, -bound, bound) for v in row] for row in g1],
            [[cl(v, -bound, bound) for v in row] for row in g2])


def _digest_scalar(g1, g2):
    d = [0] * 8
    for l, g in ((1, g1), (2, g2)):
        f = 0
        for row in g:
            for v in row:
                d[f % 8] = (d[f % 8] + ((v & 0xFFFFFFFF) ^ ((f + 7919 * l) & 0xFFFFFFFF))) & 0xFFFFFFFF
                f += 1
    return d


def _step_scalar(w, r, gs, su):
    nw, nr = [], []
    for wrow, rrow, grow in zip(w, r, gs):
        wo, ro = [], []
        for wv, rv, gv in zip(wrow, rrow, grow):
            rv = rv + gv
            q = abs(rv) >> su
            q = q if rv >= 0 else -q
            d = min(max(q, -1), 1)
            rv = rv - d * (1 << su)
            ro.append(min(max(rv, -(1 << su)), 1 << su))
            wo.append(min(max(wv - d, -8), 7))
        nw.append(wo)
        nr.append(ro)
    return nw, nr


def selftest():
    results = []
    A = lambda v: np.array(v, dtype=np.int64)
    eq = np.array_equal

    def ok(name, cond):
        results.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name)

    rng = np.random.RandomState(20261029)

    # primitives
    ok("blake3_empty_known_answer",
       blake3(b"").hex() == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262")
    data = bytes(rng.randint(0, 256, 50816).astype(np.uint8))
    full = _load_module("shares_b3full_ind", os.path.join(_HERE, "blake3_full.py")).blake3_full
    ok("blake3_long_matches_short_one_chunk", blake3_long(data[:1000]) == blake3_short(data[:1000]))
    ok("blake3_long_matches_full_implementation",
       all(blake3_long(data[:n]) == full(data[:n]) for n in (1025, 2049, 50816)))
    ok("le32_layout", le32(0x01020304) == bytes([4, 3, 2, 1]) and le32(-1) == b"\xff\xff\xff\xff")

    # rshift, tdiv, clip
    ok("rshift_spec_examples_s5", [int(rshift(v, 5)) for v in (-1, 16, -16, -17)] == [0, 1, 0, -1])
    ok("rshift_ties_negative_and_positive_s5",
       eq(rshift(A([15, 47, 48, -15, -48, -49, 0, 43904, -43904]), 5), A([0, 1, 2, 0, -1, -2, 0, 1372, -1372])))
    ok("rshift_s1_s3_s13",
       eq(rshift(A([1, -1, 3, -3]), 1), A([1, 0, 2, -1])) and eq(rshift(A([3, 4, -4, -5]), 3), A([0, 1, 0, -1]))
       and eq(rshift(A([4095, 4096, -4096, -4097]), 13), A([0, 1, 0, -1])))
    ok("tdiv_rounds_toward_zero",
       eq(tdiv(A([5, -5, -1, 1, 8, -8, 15, -15, -16, 16, 0]), 3), A([0, 0, 0, 0, 1, -1, 1, -1, -2, 2, 0]))
       and eq(tdiv(A([5, -5, 7, -7, -8193, 16384, -16385]), 1), A([2, -2, 3, -3, -4096, 8192, -8192]))
       and eq(tdiv(A([-8193, 8191, 8192, 16384, -16384, -16383]), 13), A([-1, 0, 1, 2, -2, -1])))
    ok("clip_bounds_inclusive", eq(clip(A([-9, -8, -1, 7, 8, 100]), -8, 7), A([-8, -8, -1, 7, 7, 7])))

    # nibble packing
    ok("pack_2x2", pack(A([[1, -1], [7, -8]])) == bytes([0xF1, 0x87]))
    ok("pack_negative_nibbles", pack(A([[0, -2, 3, -3]])) == bytes([0xE0, 0xD3]))
    ok("pack_row_major_odd_columns", pack(A([[1, 2, 3], [4, 5, 6]])) == bytes([0x21, 0x43, 0x65]))
    ok("pack_sizes", len(pack(np.zeros((NH, NIN), dtype=np.int64))) == 50176
       and len(pack(np.zeros((NOUT, NH), dtype=np.int64))) == 640)

    # initial state weights
    s = bytes([0x00, 0xFF, 0x01, 0x55, 0xAA, 0x80, 0x03, 0x6C, 0x02, 0x40])
    w1, w2 = weights_from_bytes(s, 2, 3, 2)
    ok("weight_formula_and_fill_order",
       eq(w1, A([[-6, 6, -5], [-2, 2, -4]])) and eq(w2, A([[-3, 0], [-4, -5]])))
    iw1, iw2, ir1, ir2 = init_state(SEED_ZERO)
    st = chacha8_stream(blake3(b"neb-model-init" + SEED_ZERO), NH * NIN + NOUT * NH)
    ok("init_state_shapes_range_and_zero_residual",
       iw1.shape == (NH, NIN) and iw2.shape == (NOUT, NH) and iw1.min() >= -6 and iw1.max() <= 6
       and iw2.min() >= -6 and iw2.max() <= 6 and not ir1.any() and not ir2.any()
       and ir1.shape == iw1.shape and ir2.shape == iw2.shape)
    wf = lambda b: (b & 3) + ((b >> 2) & 3) + ((b >> 4) & 3) + ((b >> 6) & 3) - 6
    ok("init_state_first_bytes_of_stream",
       int(iw1[0][0]) == wf(st[0]) and int(iw1[127][783]) == wf(st[NH * NIN - 1])
       and int(iw2[0][0]) == wf(st[NH * NIN]) and int(iw2[9][127]) == wf(st[NH * NIN + NOUT * NH - 1]))

    # dataset
    ok("target_vector", eq(target_vector(3, 5), A([-8, -8, -8, 7, -8])))
    ok("pixels_formula", eq(pixels_from_bytes([0xFF, 0x00, 0x07, 0x05, 0x08], [0x00, 0x03, 0xFF, 0x02, 0x04]),
                            A([6, 2, 7, 6, 0])))

    # sample indices
    cases = [(0, 2048, 0), (0xFFFFFFFF, 2048, 2047), (0x80000000, 3, 1), (0x55555555, 3, 0), (0x55555556, 3, 1),
             (0x1999999A, 10, 1), (0x19999999, 10, 0), (0xFFFFFFFF, 1, 0), (0x80000000, 2048, 1024)]
    ok("sample_index_formula",
       all(sample_indices(le32(u) + bytes(60), n)[0] == want for u, n, want in cases))
    u = bytes([1, 0, 0, 0, 0, 0, 0, 0x80, 0xFF, 0xFF, 0xFF, 0xFF, 1, 2, 3, 4]) + bytes(48)
    ok("sample_index_little_endian_words",
       sample_indices(u, 1 << 32)[:4] == [1, 1 << 31, (1 << 32) - 1, 0x04030201]
       and sample_indices(u, 2048)[:4] == [0, 1024, 2047, 0x04030201 >> 21] and len(sample_indices(u, 5)) == 16)

    # gradient: hand-made cases
    X = A([[[7, 1, 0], [2, 0, 5]]])
    T = A([[[-8, 7], [7, -8]]])
    W1c = A([[3, -2, 1], [-6, 4, 5]])
    W2c = A([[2, -1], [-3, 1]])
    g1, g2 = share_gradients(W1c, W2c, X, T, s1=2, s2=1, sb=1, bound=256)
    ok("gradient_hand_small_shifts_mask_below",
       eq(g1[0], A([[33, 7, -40], [10, 0, 25]])) and eq(g2[0], A([[20, -71], [-25, 79]])))
    g1, g2 = share_gradients(W1c, W2c, X, T, s1=2, s2=1, sb=1, bound=30)
    ok("gradient_hand_clip_bound",
       eq(g1[0], A([[30, 7, -30], [10, 0, 25]])) and eq(g2[0], A([[20, -30], [-25, 30]])))
    Xd = A([[[7, 7, 7, 7, 7], [0, 1, 2, 7, 3]]])
    Td = A([[[7, -8], [-8, 7]]])
    W1d = A([[7, 7, 7, 7, 7], [-8, -8, -8, 3, 0]])
    W2d = A([[7, -8], [-8, 7]])
    g1, g2 = share_gradients(W1d, W2d, Xd, Td)
    ok("gradient_hand_default_shifts_mask_above_and_a1_clip",
       eq(g1[0], A([[0, 7, 14, 49, 21], [49, 41, 33, -7, 25]])) and eq(g2[0], A([[-7, 20], [11, -25]])))
    Xe = A([[[4, 3]]])
    W1e = A([[2, 2], [-4, 0], [4, 0], [-3, -2]])
    W2e = A([[1, -1, 1, -1]])
    g1, g2 = share_gradients(W1e, W2e, Xe, A([[[-8]]]), s1=1, s2=1, sb=1, bound=256)
    ok("gradient_hand_mask_boundaries_y1_7_minus8_8_minus9",
       eq(g1[0], A([[16, 12], [-12, -9], [0, 0], [0, 0]])) and eq(g2[0], A([[49, -56, 49, -56]])))
    xb = np.concatenate([X, rng.randint(0, 8, (2, 2, 3))])
    tb = np.concatenate([T, np.where(rng.randint(0, 2, (2, 2, 2)) == 1, 7, -8)])
    gb1, gb2 = share_gradients(W1c, W2c, xb, tb, s1=2, s2=1, sb=1)
    single = [share_gradients(W1c, W2c, xb[k:k + 1], tb[k:k + 1], s1=2, s2=1, sb=1) for k in range(3)]
    ok("gradient_batch_equals_single_shares",
       all(eq(gb1[k], single[k][0][0]) and eq(gb2[k], single[k][1][0]) for k in range(3)))

    # gradient and digest against plain loops
    def random_case(k, nb, nin, nh, nout, lo1, hi1, lo2, hi2, xlo=0, xhi=8):
        return (rng.randint(lo1, hi1 + 1, (nh, nin)), rng.randint(lo2, hi2 + 1, (nout, nh)),
                rng.randint(xlo, xhi, (k, nb, nin)),
                np.where(rng.randint(0, 2, (k, nb, nout)) == 1, TPOS, TNEG))

    good = good_d = True
    for (k, nb, nin, nh, nout, lo1, hi1, lo2, hi2, xlo, xhi) in [
            (3, 4, 7, 5, 3, -6, 6, -6, 6, 0, 8), (2, 16, 40, 20, 10, -8, 7, -8, 7, 0, 8),
            (2, 16, 50, 12, 10, 7, 7, -8, -8, 7, 8), (2, 16, 50, 12, 10, -8, -8, 7, 7, 7, 8),
            (2, 16, 50, 12, 10, -8, -8, -8, -8, 7, 8), (2, 16, 50, 12, 10, 7, 7, 7, 7, 7, 8)]:
        cw1, cw2, cx, ct = random_case(k, nb, nin, nh, nout, lo1, hi1, lo2, hi2, xlo, xhi)
        for kw in ({}, {"s1": 1, "s2": 2, "sb": 1, "bound": 40}):
            a1_, a2_ = share_gradients(cw1, cw2, cx, ct, **kw)
            lanes = digest_lanes(a1_, a2_)
            for q in range(k):
                r1_, r2_ = _gradient_scalar(cw1.tolist(), cw2.tolist(), cx[q].tolist(), ct[q].tolist(), **kw)
                good &= eq(a1_[q], A(r1_)) and eq(a2_[q], A(r2_))
                good_d &= lanes[q].tolist() == _digest_scalar(r1_, r2_)
    ok("gradient_matches_plain_loops_random_and_extreme", good)
    ok("digest_matches_plain_loops", good_d)
    ds = TestDataset()
    keys = [share_key(0, 0), share_key(0, 1)]
    ix = [share_indices(k, ds.n) for k in keys]
    smp = [[ds.sample(i) for i in row] for row in ix]
    bx = np.stack([np.stack([p[0] for p in row]) for row in smp])
    bt = np.stack([np.stack([target_vector(p[1]) for p in row]) for row in smp])
    rg1, rg2 = share_gradients(iw1, iw2, bx, bt)
    sg1, sg2 = _gradient_scalar(iw1.tolist(), iw2.tolist(), bx[1].tolist(), bt[1].tolist())
    ok("gradient_matches_plain_loops_full_size_share", eq(rg1[1], A(sg1)) and eq(rg2[1], A(sg2))
       and rg1.shape == (2, NH, NIN) and rg2.shape == (2, NOUT, NH) and np.abs(rg1).max() <= CLIP)

    # digest hand case
    dg = digest_bytes(digest_lanes(A([[[1, -1, 2, 0, 0], [3, 0, 0, 0, -2]]]), A([[[-3, 0, 4]]]))[0])
    ok("digest_hand_lane_fold_with_wraparound",
       dg.hex() == "08000000" "f4ffffff" "d75c0000" "f21e0000" "f31e0000" "f71e0000" "f51e0000" "f61e0000")

    # ticket
    ok("ticket_is_blake3_of_digest_and_key_little_endian",
       ticket(bytes(range(32)), bytes(range(32, 64))) == blake3_short(bytes(range(64)))
       and ticket_int(bytes([1]) + bytes(31)) == 1 and ticket_int(bytes(31) + bytes([1])) == 1 << 248)

    # step: hand case with SU1 = 3, SU2 = 2 so that the dead zone, the clip of D and the residual clamp occur in a few entries
    sw1 = A([[3, 3, 3, -8], [7, 0, 2, 1]])
    sr1 = A([[0, 5, -6, 0], [-8, -7, -8, 8]])
    ga = A([[2, 1, -2, 60], [-50, 4, 0, -1]])
    gb = A([[3, 3, -3, 40], [-50, -4, 0, 1]])
    sw2 = A([[0, 5, -7]])
    sr2 = A([[3, -3, 4]])
    ha = A([[1, -1, 0]])
    hb = A([[0, 1, 1]])
    nw1, nw2, nr1, nr2 = model_step(sw1, sw2, sr1, sr2, np.stack([ga, gb]), np.stack([ha, hb]), su1=3, su2=2)
    ok("step_hand_layer1_first_three_columns",
       eq(nw1[:, :3], A([[3, 2, 4], [7, 0, 3]])) and eq(nr1[:, :3], A([[5, 1, -3], [-8, -7, 0]])))
    ok("step_hand_layer1_d_clipped_and_residual_clamp",
       eq(nw1, A([[3, 2, 4, -8], [7, 0, 3, 0]])) and eq(nr1, A([[5, 1, -3, 8], [-8, -7, 0, 0]])))
    ok("step_hand_layer2", eq(nw2, A([[-1, 5, -8]])) and eq(nr2, A([[0, -3, 1]])))
    ok("step_hand_matches_plain_loops",
       _step_scalar(sw1.tolist(), sr1.tolist(), (ga + gb).tolist(), 3) == (nw1.tolist(), nr1.tolist())
       and _step_scalar(sw2.tolist(), sr2.tolist(), (ha + hb).tolist(), 2) == (nw2.tolist(), nr2.tolist()))
    good = True
    for su in (3, 12, 15):
        rw = rng.randint(-8, 8, (6, 9))
        rr = rng.randint(-(1 << su), (1 << su) + 1, (6, 9))
        rg = rng.randint(-CLIP, CLIP + 1, (5, 6, 9))
        sw_, sr_ = _step_scalar(rw.tolist(), rr.tolist(), rg.sum(axis=0).tolist(), su)
        o = apply_sums(rw, rw, rr, rr, rg.sum(axis=0), rg.sum(axis=0), 5, su1=su, su2=su)
        good &= o[0].tolist() == sw_ and o[2].tolist() == sr_
    ok("step_matches_plain_loops_random", good)

    # step with the shifts of the test-only variant (SU1 = 9, SU2 = 6), three shares
    t1 = np.stack([A([[256, 256, -256, -256], [256, -256, 0, 0]]),
                   A([[255, 255, -255, -254], [256, -256, 0, 0]]),
                   A([[0, 0, 0, 0], [256, -256, 0, -1]])])
    t2 = np.stack([A([[0, 1, 0, 100, -100]]), A([[0, 0, -1, 100, -100]]), A([[0, 0, 0, 100, 0]])])
    nw1, nw2, nr1, nr2 = model_step(A([[0, 0, 7, 3], [-8, 0, 2, -3]]), A([[1, 1, 0, -8, 7]]),
                                    A([[0, 1, -1, -1], [500, -500, 512, -512]]), A([[63, 63, -63, 10, -64]]),
                                    t1, t2, su1=9, su2=6)
    ok("step_hand_su1_9_layer1_dead_zone_edges_d_clip_clamp_saturation",
       eq(nw1, A([[0, -1, 7, 3], [-8, 1, 1, -2]])) and eq(nr1, A([[511, 0, 0, -511], [512, -512, 0, -1]])))
    ok("step_hand_su2_6_layer2_dead_zone_edges_d_clip_clamp_saturation",
       eq(nw2, A([[1, 0, 1, -8, 7]])) and eq(nr2, A([[63, 0, 0, 64, -64]])))
    big = np.stack([A([[256, -256]])] * 257)
    nw1, nw2, nr1, nr2 = model_step(A([[3, 3]]), A([[-8, 7]]), A([[0, 0]]), A([[0, 0]]), big, big)
    ok("step_hand_production_shifts_257_shares_d_clip_and_clamp",
       eq(nw1, A([[2, 4]])) and eq(nr1, A([[32768, -32768]])) and eq(nw2, A([[-8, 7]])) and eq(nr2, A([[4096, -4096]])))
    ok("step_hand_production_shifts_residual_edges",
       eq(model_step(A([[0, 0]]), A([[0, 0]]), A([[32767, -32767]]), A([[4095, -4095]]),
                     A([[[1, -1]]]), A([[[1, -1]]]))[2], A([[0, 0]])))

    # the step size limit of section 6: 2^SU1 + N_MAX * CLIP < 2^31 <= 2^SU1 + (N_MAX + 1) * CLIP
    def raises(exc, fn):
        try:
            fn()
        except exc:
            return True
        return False

    z = A([[0]])
    ok("step_size_limit_and_int32_bound",
       (1 << SU1) + N_MAX * CLIP < 1 << 31 <= (1 << SU1) + (N_MAX + 1) * CLIP and N_MAX == 8388479
       and eq(apply_sums(z, z, A([[1 << SU1]]), A([[-(1 << SU2)]]), A([[N_MAX * CLIP]]), A([[-N_MAX * CLIP]]),
                         N_MAX)[2], A([[(1 << SU1)]]))
       and raises(ValueError, lambda: apply_sums(z, z, z, z, z, z, N_MAX + 1))
       and raises(ValueError, lambda: apply_sums(z, z, z, z, z, z, 0))
       and raises(OverflowError, lambda: apply_sums(z, z, A([[1 << SU1]]), z, A([[(N_MAX + 1) * CLIP]]), z, N_MAX)))

    # the sum of the clipped gradients does not depend on the order of the shares (section 6)
    shares1 = np.stack([ga, gb, A([[-7, 9, 11, -250], [3, 3, 3, 3]])])
    shares2 = np.stack([ha, hb, A([[5, -5, 100]])])
    ref = model_step(sw1, sw2, sr1, sr2, shares1, shares2, su1=3, su2=2)
    ok("step_order_independence_hand_all_permutations",
       all(all(eq(a, b) for a, b in zip(ref, model_step(sw1, sw2, sr1, sr2, shares1[list(p)], shares2[list(p)],
                                                          su1=3, su2=2)))
           for p in itertools.permutations(range(3))))
    og1, og2 = share_gradients(W1c, W2c, xb, tb, s1=2, s2=1, sb=1)
    z1_, z2_ = np.zeros_like(W1c), np.zeros_like(W2c)
    ref = model_step(W1c, W2c, z1_, z2_, og1, og2, su1=1, su2=1)
    ok("step_order_independence_computed_shares_all_permutations",
       all(all(eq(a, b) for a, b in zip(ref, model_step(W1c, W2c, z1_, z2_, og1[list(p)], og2[list(p)],
                                                          su1=1, su2=1)))
           for p in itertools.permutations(range(3))))

    # state identifiers
    ok("residual_hash_input_layout",
       residual_hash_input(A([[1, -1]]), A([[-2147483648, 258]])) ==
       b"neb-model-residual" + bytes.fromhex("01000000" "ffffffff" "00000080" "02010000"))
    ok("weights_hash_input_layout",
       weights_hash_input(A([[1, -1]]), A([[7, -8]])) == b"neb-model-weights" + bytes([0xF1, 0x87]))
    ok("state_hashes_are_blake3_of_inputs",
       state_hashes(iw1, iw2, ir1, ir2) == (blake3_long(weights_hash_input(iw1, iw2)).hex(),
                                            blake3_long(residual_hash_input(ir1, ir2)).hex())
       and len(weights_hash_input(iw1, iw2)) == 17 + 50176 + 640
       and len(residual_hash_input(ir1, ir2)) == 18 + 4 * (NH * NIN + NOUT * NH))

    # vector parameters
    ok("test_vector_share_count", sum(STEP_SHARES[s % 8] for s in range(STEPS)) == 543)
    ok("share_key_layout", share_key(1, 2) == blake3_short(b"neb-test-share" + bytes([1, 0, 0, 0, 2, 0, 0, 0])))

    bad = results.count(False)
    print("selftest: %s" % ("PASS" if bad == 0 else "FAIL (%d failed)" % bad))
    return 0 if bad == 0 else 1


# command line
USAGE = "usage: shares_oracle.py --selftest | --check FILE [--su1 N] [--su2 N]"


def main(argv):
    if argv == ["--selftest"]:
        return selftest()
    if len(argv) >= 2 and argv[0] == "--check" and len(argv) % 2 == 0:
        shifts = {"--su1": SU1, "--su2": SU2}
        seen = set()
        for opt, val in zip(argv[2::2], argv[3::2]):
            if opt not in shifts or opt in seen or not re.fullmatch(r"[1-9][0-9]?", val) or int(val) > 30:
                print(USAGE)
                return 2
            seen.add(opt)
            shifts[opt] = int(val)
        return check(argv[1], shifts["--su1"], shifts["--su2"])
    print(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
