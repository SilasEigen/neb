#!/usr/bin/env python3
"""Independent Python oracle of spec/nebhash-v3.md: anti-shortcut MoE frozen INT8 backbones with on-chain
integer LoRA adapters, residual blocks, 75% input + 50% bottleneck dropout, and exact int32 trace commitment.
Shares no code with pow/ref/nebhash3.c.

    python3 -B -I pow/oracle/nebhash3_oracle.py --check vectors/nebhash3.txt
    python3 -B -I pow/oracle/nebhash3_oracle.py --check vectors/nebhash3_s1.txt --su-drop 6
"""
import importlib.util
import os
import sys

import numpy as np

NB3, CLIP, TPOS, TNEG, NOUT = 64, 1024, 7, -8, 2
N_TEST = 2048
STEP_SHARES = (32, 1, 3, 32, 17, 2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nebhash3_prim", os.path.join(_HERE, "primitives.py"))
_prim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prim)
chacha8_stream, blake3 = _prim.chacha8_stream, _prim.blake3

ARCHS = {
    1: dict(d=64, rank=16, nblocks=3, nexperts=16, s_a=5, s_lora=5, s_out=9, sb_b=4, sb_in=9, s_head=5, sb_head=2, su_lora=14, su_head=12),
    2: dict(d=1024, rank=16, nblocks=6, nexperts=10752, s_a=7, s_lora=7, s_out=11, sb_b=4, sb_in=11, s_head=7, sb_head=2, su_lora=16, su_head=14),
}


def le32(v):
    return int(v & 0xFFFFFFFF).to_bytes(4, "little")


def rshift(x, n):
    return x if n == 0 else (x + (1 << (n - 1))) >> n


def init_base(arch):
    cfg = ARCHS[arch]
    d, E = cfg["d"], cfg["nexperts"]
    raw = chacha8_stream(blake3(b"neb-syn3-base" + bytes([arch])), E * d * d)
    return np.frombuffer(raw, dtype=np.int8).astype(np.int64).reshape(E, d, d).copy()


def base_hash(w_base):
    return blake3(b"neb-base-weights" + w_base.astype(np.int8).tobytes())


def init_state(arch, seed):
    cfg = ARCHS[arch]
    d, r, L = cfg["d"], cfg["rank"], cfg["nblocks"]
    shapes = []
    for _ in range(L):
        shapes.append((r, d))
        shapes.append((d, r))
    shapes.append((NOUT, d))
    total = sum(a * b for a, b in shapes)
    raw = np.frombuffer(chacha8_stream(blake3(b"neb-model-init" + seed), total), dtype=np.uint8).astype(np.int64)
    w = (raw & 3) + ((raw >> 2) & 3) + ((raw >> 4) & 3) + ((raw >> 6) & 3) - 6
    ws, at = [], 0
    for sh in shapes:
        nw = sh[0] * sh[1]
        ws.append(w[at:at + nw].reshape(sh).copy())
        at += nw
    return ws, [np.zeros(sh, dtype=np.int64) for sh in shapes]


def share_indices(key, n):
    u = chacha8_stream(key, 4 * NB3)
    return [(int.from_bytes(u[4 * j:4 * j + 4], "little") * n) >> 32 for j in range(NB3)]


def dropout_masks(key, L, D, R):
    drop_key = blake3(b"neb-lora-drop" + key)
    in_bytes = (NB3 * L * D * 2 + 7) // 8
    u_bytes = (NB3 * L * R + 7) // 8
    raw = chacha8_stream(drop_key, in_bytes + u_bytes)
    in_arr = np.frombuffer(raw[:in_bytes], dtype=np.uint8)
    # Extract 2-bit pairs in little-endian order
    p0 = (in_arr >> 0) & 3
    p1 = (in_arr >> 2) & 3
    p2 = (in_arr >> 4) & 3
    p3 = (in_arr >> 6) & 3
    pairs = np.stack([p0, p1, p2, p3], axis=-1).ravel()[:NB3 * L * D]
    mask_in = (pairs != 0).astype(np.int64).reshape(NB3, L, D)

    u_arr = np.frombuffer(raw[in_bytes:], dtype=np.uint8)
    mask_u = np.unpackbits(u_arr, bitorder="little")[:NB3 * L * R].astype(np.int64).reshape(NB3, L, R)
    return mask_in, mask_u


class TraceAcc:
    def __init__(self):
        self.lanes = [0] * 8
        self.tc = 0

    def mix_scalar(self, val):
        c = self.tc
        mix = ((int(val) & 0xFFFFFFFF) ^ ((c * 0x9E3779B9 + 0x7F4A7C15) & 0xFFFFFFFF)) & 0xFFFFFFFF
        self.lanes[c & 7] = (self.lanes[c & 7] + mix) & 0xFFFFFFFF
        self.tc = c + 1

    def digest(self):
        return b"".join(le32(x) for x in self.lanes)


def evaluate_share(arch, ws, w_base, x_raw, labels, key):
    cfg = ARCHS[arch]
    d, r, L, E = cfg["d"], cfg["rank"], cfg["nblocks"], cfg["nexperts"]
    mask_in, mask_u = dropout_masks(key, L, d, r)
    tr = TraceAcc()

    x0 = x_raw.astype(np.int64) - 4
    xs = [x0]
    xms, y_as, us, y_outs, experts = [], [], [], [], []

    for l in range(L):
        xl = xs[-1]
        if l == 0:
            rh = blake3(b"neb-moe-route" + key + bytes([0]))
        else:
            rh = blake3(b"neb-moe-route" + key + bytes([l]) + xl.astype(np.int8).tobytes())
        e_l = (int.from_bytes(rh[:4], "little") * E) >> 32
        experts.append(e_l)
        tr.mix_scalar(e_l)

        Wb = w_base[e_l]
        Al, Bl = ws[2 * l], ws[2 * l + 1]
        xm = xl * mask_in[:, l, :]
        za_mat = xm @ Al.T
        zb_mat = xm @ Wb.T
        ya_mat = rshift(za_mat, cfg["s_a"])
        u_mat = np.clip(ya_mat, -8, 7) * mask_u[:, l, :]
        zl_mat = u_mat @ Bl.T
        yo_mat = xl + rshift(zb_mat + zl_mat * (1 << cfg["s_lora"]), cfg["s_out"])
        xnext = np.clip(yo_mat, -8, 7)

        for b in range(NB3):
            for ri in range(r):
                tr.mix_scalar(za_mat[b, ri])
            for o in range(d):
                tr.mix_scalar(zb_mat[b, o])

        xs.append(xnext)
        xms.append(xm)
        y_as.append(ya_mat)
        us.append(u_mat)
        y_outs.append(yo_mat)

    H = ws[2 * L]
    xL = xs[L]
    zh_mat = xL @ H.T
    for b in range(NB3):
        for o in range(NOUT):
            tr.mix_scalar(zh_mat[b, o])

    ah = np.clip(rshift(zh_mat, cfg["s_head"]), -8, 7)
    t = np.full((NB3, NOUT), TNEG, dtype=np.int64)
    t[np.arange(NB3), labels] = TPOS
    eh = np.clip(ah - t, -8, 7)

    grads = [None] * (2 * L + 1)
    grads[2 * L] = eh.T @ xL
    e = np.clip(rshift(eh @ H, cfg["sb_head"]), -8, 7)

    for l in range(L - 1, -1, -1):
        Wb = w_base[experts[l]]
        Al, Bl = ws[2 * l], ws[2 * l + 1]
        xm = xms[l]
        dv = e * ((y_outs[l] >= -8) & (y_outs[l] <= 7))
        grads[2 * l + 1] = dv.T @ us[l]
        eu = np.clip(rshift(dv @ Bl, cfg["sb_b"]), -8, 7)
        du = eu * mask_u[:, l, :] * ((y_as[l] >= -8) & (y_as[l] <= 7))
        grads[2 * l] = du.T @ xm
        if l > 0:
            m_prev = (dv @ Wb + du @ Al) * mask_in[:, l, :]
            e = np.clip(dv + rshift(m_prev, cfg["sb_in"]), -8, 7)

    clipped_grads = [np.clip(g, -CLIP, CLIP) for g in grads]
    g_lanes = [0] * 8
    for ti, g in enumerate(clipped_grads):
        c = (g.ravel() & 0xFFFFFFFF).astype(np.uint64)
        f = np.arange(c.size, dtype=np.uint64)
        v = c ^ (f + np.uint64(7919 * (ti + 1)))
        for k in range(8):
            g_lanes[k] = (g_lanes[k] + int(v[k::8].sum())) & 0xFFFFFFFF
    grad_digest = b"".join(le32(x) for x in g_lanes)
    digest = blake3(grad_digest + tr.digest())
    ticket = blake3(digest + key)
    return clipped_grads, digest, ticket, experts


class TestDataset:
    def __init__(self, arch):
        self.arch, self.d, self.n = arch, ARCHS[arch]["d"], N_TEST
        self.proto = [
            np.frombuffer(chacha8_stream(blake3(b"neb-syn3-proto" + bytes([arch, c])), self.d), dtype=np.uint8).astype(np.int64) & 7
            for c in range(2)
        ]

    def get(self, i):
        noise = np.frombuffer(chacha8_stream(blake3(b"neb-syn3-sample" + bytes([self.arch]) + le32(i)), self.d), dtype=np.uint8).astype(np.int64)
        return np.clip(self.proto[i % 2] + (noise & 3) - 1, 0, 7), i % 2


def share(arch, ws, w_base, ds, key):
    idx = share_indices(key, ds.n)
    xs, labels = zip(*(ds.get(i) for i in idx))
    grads, digest, ticket, experts = evaluate_share(arch, ws, w_base, np.stack(xs), np.array(labels), key)
    return grads, digest, ticket, idx, experts


def apply(arch, ws, rs, sums, su_drop=0):
    cfg = ARCHS[arch]
    L = cfg["nblocks"]
    for i in range(len(ws)):
        su = (cfg["su_lora"] if i < 2 * L else cfg["su_head"]) - su_drop
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
    return blake3(b"neb-test-share3" + bytes([arch]) + le32(s) + le32(j))


def run_vectors(su_drop=0):
    lines = []
    arch = 1
    w_base = init_base(arch)
    ds = TestDataset(arch)
    ws, rs = init_state(arch, b"\x00" * 32)
    wh, rh = hashes(ws, rs)
    lines.append(f"arch {arch} base {base_hash(w_base).hex()}")
    lines.append(f"arch {arch} init {wh.hex()} {rh.hex()}")
    for s, nsh in enumerate(STEP_SHARES):
        sums = [np.zeros_like(x) for x in ws]
        for j in range(nsh):
            k = share_key(arch, s, j)
            g, d, t, idx, experts = share(arch, ws, w_base, ds, k)
            for i in range(len(sums)):
                sums[i] += g[i]
            if j == 0 or j == nsh - 1:
                lines.append(f"arch {arch} step {s} share {j} {d.hex()} {t.hex()} {idx[0]} {idx[-1]} {experts[0]} {experts[-1]}")
        apply(arch, ws, rs, sums, su_drop=su_drop)
        wh, rh = hashes(ws, rs)
        lines.append(f"arch {arch} step {s} state {wh.hex()} {rh.hex()}")
    return "\n".join(lines) + "\n"


def main():
    args = sys.argv[1:]
    su_drop = 0
    if "--su-drop" in args:
        i = args.index("--su-drop")
        su_drop = int(args[i + 1])
        del args[i:i + 2]
    if len(args) == 2 and args[0] == "--check":
        got = run_vectors(su_drop=su_drop)
        with open(args[1], "r", encoding="utf-8") as f:
            want = f.read()
        if got != want:
            sys.stderr.write(f"MISMATCH against {args[1]}\n")
            for a, b in zip(got.splitlines(), want.splitlines()):
                if a != b:
                    sys.stderr.write(f"  got : {a}\n  want: {b}\n")
                    break
            sys.exit(1)
        print(f"PASS nebhash3_oracle matches {args[1]}")
        return
    sys.stdout.write(run_vectors(su_drop=su_drop))


if __name__ == "__main__":
    main()
