"""Primitives for the NEB oracle: ChaCha8 (original DJB layout, 64-bit
counter) and BLAKE3 (blake3_short for inputs of at most 1024 bytes, blake3_long
for any length with the chunk tree, blake3 picks one). Written from the public
specifications; cross-checked by the oracle selftest (blake3_full.py) and by
the share vectors produced with the C reference."""

import numpy as np

M32 = 0xFFFFFFFF


# ChaCha8 (vectorised over blocks)
def _rotl(x, n):
    return ((x << np.uint32(n)) | (x >> np.uint32(32 - n))).astype(np.uint32)


def chacha8_stream(key: bytes, n: int) -> bytes:
    nb = (n + 63) // 64
    ctr = np.arange(nb, dtype=np.uint64)
    kw = np.frombuffer(key, dtype="<u4")
    s = [np.full(nb, c, dtype=np.uint32) for c in (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)]
    s += [np.full(nb, int(w), dtype=np.uint32) for w in kw]
    s += [(ctr & M32).astype(np.uint32), (ctr >> np.uint64(32)).astype(np.uint32)]
    s += [np.zeros(nb, dtype=np.uint32), np.zeros(nb, dtype=np.uint32)]
    x = [a.copy() for a in s]

    def qr(a, b, c, d):
        x[a] = x[a] + x[b]; x[d] = _rotl(x[d] ^ x[a], 16)
        x[c] = x[c] + x[d]; x[b] = _rotl(x[b] ^ x[c], 12)
        x[a] = x[a] + x[b]; x[d] = _rotl(x[d] ^ x[a], 8)
        x[c] = x[c] + x[d]; x[b] = _rotl(x[b] ^ x[c], 7)

    with np.errstate(over="ignore"):
        for _ in range(4):
            qr(0, 4, 8, 12); qr(1, 5, 9, 13); qr(2, 6, 10, 14); qr(3, 7, 11, 15)
            qr(0, 5, 10, 15); qr(1, 6, 11, 12); qr(2, 7, 8, 13); qr(3, 4, 9, 14)
        out = np.stack([x[i] + s[i] for i in range(16)], axis=1)
    return out.astype("<u4").tobytes()[:n]


# BLAKE3 (inputs up to 1024 bytes)
_IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
_PERM = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]


def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & M32


def _g(v, a, b, c, d, mx, my):
    v[a] = (v[a] + v[b] + mx) & M32; v[d] = _rotr(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d]) & M32;      v[b] = _rotr(v[b] ^ v[c], 12)
    v[a] = (v[a] + v[b] + my) & M32; v[d] = _rotr(v[d] ^ v[a], 8)
    v[c] = (v[c] + v[d]) & M32;      v[b] = _rotr(v[b] ^ v[c], 7)


def _compress(cv, m, counter, block_len, flags):
    v = list(cv) + _IV[:4] + [counter & M32, counter >> 32, block_len, flags]
    m = list(m)
    for r in range(7):
        _g(v, 0, 4, 8, 12, m[0], m[1]); _g(v, 1, 5, 9, 13, m[2], m[3])
        _g(v, 2, 6, 10, 14, m[4], m[5]); _g(v, 3, 7, 11, 15, m[6], m[7])
        _g(v, 0, 5, 10, 15, m[8], m[9]); _g(v, 1, 6, 11, 12, m[10], m[11])
        _g(v, 2, 7, 8, 13, m[12], m[13]); _g(v, 3, 4, 9, 14, m[14], m[15])
        if r < 6:
            m = [m[i] for i in _PERM]
    return [v[i] ^ v[i + 8] for i in range(8)]


def blake3_short(data: bytes) -> bytes:
    assert len(data) <= 1024, "oracle BLAKE3 handles a single chunk only"
    cv = list(_IV)
    nblocks = max(1, (len(data) + 63) // 64)
    for b in range(nblocks):
        blk = data[b * 64:(b + 1) * 64]
        flags = (1 if b == 0 else 0) | ((2 | 8) if b == nblocks - 1 else 0)
        words = np.frombuffer(blk.ljust(64, b"\0"), dtype="<u4").tolist()
        cv = _compress(cv, words, 0, len(blk), flags)
    return np.array(cv, dtype="<u4").tobytes()


# BLAKE3 (any length)
_CHUNK_START, _CHUNK_END, _PARENT, _ROOT = 1, 2, 4, 8


def _chunk_cv(chunk, counter, root):
    cv = list(_IV)
    nblocks = max(1, (len(chunk) + 63) // 64)
    for b in range(nblocks):
        blk = chunk[b * 64:(b + 1) * 64]
        flags = _CHUNK_START if b == 0 else 0
        if b == nblocks - 1:
            flags |= _CHUNK_END | (_ROOT if root else 0)
        cv = _compress(cv, np.frombuffer(blk.ljust(64, b"\0"), dtype="<u4").tolist(), counter, len(blk), flags)
    return cv


def _tree_cv(data, first, n_chunks, root):
    if n_chunks == 1:
        return _chunk_cv(data[first * 1024:(first + 1) * 1024], first, root)
    left = 1 << ((n_chunks - 1).bit_length() - 1)
    lcv = _tree_cv(data, first, left, False)
    rcv = _tree_cv(data, first + left, n_chunks - left, False)
    return _compress(_IV, lcv + rcv, 0, 64, _PARENT | (_ROOT if root else 0))


def blake3_long(data: bytes) -> bytes:
    n_chunks = max(1, (len(data) + 1023) // 1024)
    return np.array(_tree_cv(data, 0, n_chunks, True), dtype="<u4").tobytes()


def blake3(data: bytes) -> bytes:
    return blake3_short(data) if len(data) <= 1024 else blake3_long(data)
