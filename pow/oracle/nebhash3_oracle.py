#!/usr/bin/env python3
"""Independent pure-Python oracle of spec/nebhash-v3.md: frozen INT8 base + on-chain integer LoRA adapters
with key-derived deterministic dropout. Shares no code with pow/ref/nebhash3.c and requires no third-party packages.

    python3 -B -I pow/oracle/nebhash3_oracle.py --check vectors/nebhash3.txt
    python3 -B -I pow/oracle/nebhash3_oracle.py --check vectors/nebhash3_s1.txt --su-drop 6
"""
import importlib.util
import os
import sys

NB3, CLIP, TPOS, TNEG, NOUT = 32, 256, 7, -8, 2
N_TEST = 2048
STEP_SHARES = (32, 1, 3, 32, 17, 2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("nebhash3_prim", os.path.join(_HERE, "primitives.py"))
_prim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prim)
chacha8_stream, blake3 = _prim.chacha8_stream, _prim.blake3

ARCHS = {
    1: dict(d=64, rank=8, nblocks=2, s_a=4, s_lora=1, s_out=5, sb_b=4, sb_in=5, s_head=5, sb_head=3, su_lora=14, su_head=12),
    2: dict(d=4096, rank=8, nblocks=4, s_a=7, s_lora=1, s_out=8, sb_b=5, sb_in=8, s_head=8, sb_head=3, su_lora=16, su_head=14),
}


def le32(v):
    return int(v & 0xFFFFFFFF).to_bytes(4, "little")


def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def rshift(x, n):
    return x if n == 0 else (x + (1 << (n - 1))) >> n


def _binom(raw):
    return [(b & 3) + ((b >> 2) & 3) + ((b >> 4) & 3) + ((b >> 6) & 3) - 6 for b in raw]


def init_base(arch):
    cfg = ARCHS[arch]
    d, L = cfg["d"], cfg["nblocks"]
    raw = chacha8_stream(blake3(b"neb-syn3-base" + bytes([arch])), L * d * d)
    w = _binom(raw)
    out = []
    at = 0
    for _ in range(L):
        mat = []
        for _o in range(d):
            mat.append(w[at:at + d])
            at += d
        out.append(mat)
    return out


def base_hash(w_base):
    flat = [v & 15 for mat in w_base for row in mat for v in row]
    packed = bytes(flat[i] | (flat[i + 1] << 4) for i in range(0, len(flat), 2))
    return blake3(b"neb-base-weights" + packed)


def init_state(arch, seed):
    cfg = ARCHS[arch]
    d, r, L = cfg["d"], cfg["rank"], cfg["nblocks"]
    shapes = []
    for _ in range(L):
        shapes.append((r, d))
        shapes.append((d, r))
    shapes.append((NOUT, d))
    total = sum(a * b for a, b in shapes)
    w = _binom(chacha8_stream(blake3(b"neb-model-init" + seed), total))
    ws, rs, at = [], [], 0
    for rows, cols in shapes:
        mat = []
        rmat = []
        for _i in range(rows):
            mat.append(w[at:at + cols])
            rmat.append([0] * cols)
            at += cols
        ws.append(mat)
        rs.append(rmat)
    return ws, rs


def share_indices(key, n):
    u = chacha8_stream(key, 4 * NB3)
    return [(int.from_bytes(u[4 * j:4 * j + 4], "little") * n) >> 32 for j in range(NB3)]


def dropout_masks(key, L, R):
    drop_key = blake3(b"neb-lora-drop" + key)
    nbytes = (NB3 * L * R + 7) // 8
    raw = chacha8_stream(drop_key, nbytes)
    masks = []
    for b in range(NB3):
        bm = []
        for l in range(L):
            lm = []
            for r in range(R):
                bit = (b * L + l) * R + r
                lm.append((raw[bit >> 3] >> (bit & 7)) & 1)
            bm.append(lm)
        masks.append(bm)
    return masks


def matvec(mat, vec):
    return [sum(a * b for a, b in zip(row, vec)) for row in mat]


def matvec_t(mat, vec):
    cols = len(mat[0])
    out = [0] * cols
    for row, dv in zip(mat, vec):
        if dv == 0:
            continue
        for j, w in enumerate(row):
            out[j] += dv * w
    return out


def share_gradient(arch, ws, w_base, xs_in, labels, key):
    cfg = ARCHS[arch]
    d, r, L = cfg["d"], cfg["rank"], cfg["nblocks"]
    masks = dropout_masks(key, L, r)
    grads = [[[0] * len(mat[0]) for _ in range(len(mat))] for mat in ws]

    H = ws[2 * L]
    G_H = grads[2 * L]

    for b in range(NB3):
        xs = [list(xs_in[b])]
        y_as, us, y_outs = [], [], []
        for l in range(L):
            Al, Bl = ws[2 * l], ws[2 * l + 1]
            xl = xs[-1]
            ya = [rshift(z, cfg["s_a"]) for z in matvec(Al, xl)]
            u = [clip(v, -8, 7) * m for v, m in zip(ya, masks[b][l])]
            zb = matvec(w_base[l], xl)
            zl = matvec(Bl, u)
            yo = [rshift(zb_i + (zl_i << cfg["s_lora"]), cfg["s_out"]) for zb_i, zl_i in zip(zb, zl)]
            xs.append([clip(v, -8, 7) for v in yo])
            y_as.append(ya)
            us.append(u)
            y_outs.append(yo)

        xL = xs[L]
        ah = [clip(rshift(z, cfg["s_head"]), -8, 7) for z in matvec(H, xL)]
        eh = [clip(ah[o] - (TPOS if o == labels[b] else TNEG), -8, 7) for o in range(NOUT)]
        for o in range(NOUT):
            eo = eh[o]
            if eo != 0:
                row = G_H[o]
                for j in range(d):
                    row[j] += eo * xL[j]
        e = [clip(rshift(v, cfg["sb_head"]), -8, 7) for v in matvec_t(H, eh)]

        for l in range(L - 1, -1, -1):
            Al, Bl = ws[2 * l], ws[2 * l + 1]
            G_Al, G_Bl = grads[2 * l], grads[2 * l + 1]
            xl = xs[l]
            dv = [0 if (yo < -8 or yo > 7) else ev for yo, ev in zip(y_outs[l], e)]
            for o in range(d):
                dvo = dv[o]
                if dvo != 0:
                    row = G_Bl[o]
                    ul = us[l]
                    for ri in range(r):
                        row[ri] += dvo * ul[ri]
            eu = [clip(rshift(v, cfg["sb_b"]), -8, 7) for v in matvec_t(Bl, dv)]
            du = [0 if (ya < -8 or ya > 7) else (eui * m) for ya, eui, m in zip(y_as[l], eu, masks[b][l])]
            for ri in range(r):
                dur = du[ri]
                if dur != 0:
                    row = G_Al[ri]
                    for j in range(d):
                        row[j] += dur * xl[j]
            if l > 0:
                mb = matvec_t(w_base[l], dv)
                ma = matvec_t(Al, du)
                e = [clip(rshift(a + c, cfg["sb_in"]), -8, 7) for a, c in zip(mb, ma)]
    return grads


def digest_of(grads):
    lanes = [0] * 8
    for ti, g in enumerate(grads):
        f = 0
        bias = 7919 * (ti + 1)
        for row in g:
            for v in row:
                lanes[f & 7] = (lanes[f & 7] + ((v & 0xFFFFFFFF) ^ ((f + bias) & 0xFFFFFFFF))) & 0xFFFFFFFF
                f += 1
    return b"".join(le32(x) for x in lanes)


class TestDataset:
    def __init__(self, arch):
        self.arch, self.d, self.n = arch, ARCHS[arch]["d"], N_TEST
        self.proto = [
            [b & 7 for b in chacha8_stream(blake3(b"neb-syn3-proto" + bytes([arch, c])), self.d)]
            for c in range(2)
        ]

    def get(self, i):
        noise = chacha8_stream(blake3(b"neb-syn3-sample" + bytes([self.arch]) + le32(i)), self.d)
        p = self.proto[i & 1]
        return [clip(p[j] + (noise[j] & 3) - 1, 0, 7) for j in range(self.d)], (i & 1)


def share(arch, ws, w_base, ds, key):
    idx = share_indices(key, ds.n)
    xs, labels = [], []
    for i in idx:
        x, y = ds.get(i)
        xs.append(x)
        labels.append(y)
    raw_g = share_gradient(arch, ws, w_base, xs, labels, key)
    grads = [[[clip(v, -CLIP, CLIP) for v in row] for row in mat] for mat in raw_g]
    d = digest_of(grads)
    return grads, d, blake3(d + key), idx


def apply(arch, ws, rs, sums, su_drop=0):
    cfg = ARCHS[arch]
    L = cfg["nblocks"]
    for i in range(len(ws)):
        su = (cfg["su_lora"] if i < 2 * L else cfg["su_head"]) - su_drop
        unit = 1 << su
        for r_i in range(len(ws[i])):
            wrow, rrow, srow = ws[i][r_i], rs[i][r_i], sums[i][r_i]
            for c_i in range(len(wrow)):
                v = rrow[c_i] + srow[c_i]
                q = (v >> su) if v >= 0 else -((-v) >> su)
                dlt = clip(q, -1, 1)
                v -= dlt * unit
                rrow[c_i] = clip(v, -unit, unit)
                wrow[c_i] = clip(wrow[c_i] - dlt, -8, 7)


def hashes(ws, rs):
    wflat = [v & 15 for mat in ws for row in mat for v in row]
    packed = bytes(wflat[i] | (wflat[i + 1] << 4) for i in range(0, len(wflat), 2))
    rbytes = b"".join(le32(v) for mat in rs for row in mat for v in row)
    return blake3(b"neb-model-weights" + packed), blake3(b"neb-model-residual" + rbytes)


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
        sums = [[[0] * len(row) for row in mat] for mat in ws]
        for j in range(nsh):
            k = share_key(arch, s, j)
            g, d, t, idx = share(arch, ws, w_base, ds, k)
            for i in range(len(sums)):
                for r_i in range(len(sums[i])):
                    for c_i in range(len(sums[i][r_i])):
                        sums[i][r_i][c_i] += g[i][r_i][c_i]
            if j == 0 or j == nsh - 1:
                lines.append(f"arch {arch} step {s} share {j} {d.hex()} {t.hex()} {idx[0]} {idx[-1]}")
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
