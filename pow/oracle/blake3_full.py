"""Full BLAKE3 (any length) for the NEB oracle selftest: a second,
independent implementation used only to cross-check the short and long paths
of primitives.py. Written from the public specification."""

import struct

M32 = 0xFFFFFFFF

_B3_IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
_B3_PERM = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
_CHUNK_START, _CHUNK_END, _PARENT, _ROOT = 1, 2, 4, 8


def _b3_compress(cv, m, counter, block_len, flags):
    v = list(cv) + _B3_IV[:4] + [counter & M32, counter >> 32, block_len, flags]
    m = list(m)

    def g(a, b, c, d, x, y):
        v[a] = (v[a] + v[b] + x) & M32
        v[d] = (((v[d] ^ v[a]) >> 16) | ((v[d] ^ v[a]) << 16)) & M32
        v[c] = (v[c] + v[d]) & M32
        v[b] = (((v[b] ^ v[c]) >> 12) | ((v[b] ^ v[c]) << 20)) & M32
        v[a] = (v[a] + v[b] + y) & M32
        v[d] = (((v[d] ^ v[a]) >> 8) | ((v[d] ^ v[a]) << 24)) & M32
        v[c] = (v[c] + v[d]) & M32
        v[b] = (((v[b] ^ v[c]) >> 7) | ((v[b] ^ v[c]) << 25)) & M32

    for _ in range(7):
        g(0, 4, 8, 12, m[0], m[1]); g(1, 5, 9, 13, m[2], m[3])
        g(2, 6, 10, 14, m[4], m[5]); g(3, 7, 11, 15, m[6], m[7])
        g(0, 5, 10, 15, m[8], m[9]); g(1, 6, 11, 12, m[10], m[11])
        g(2, 7, 8, 13, m[12], m[13]); g(3, 4, 9, 14, m[14], m[15])
        m = [m[i] for i in _B3_PERM]
    return [v[i] ^ v[i + 8] for i in range(8)]  # chaining value / first 32 output bytes


def _b3_chunk_cv(chunk, counter, root):
    cv = list(_B3_IV)
    blocks = [chunk[i:i + 64] for i in range(0, len(chunk), 64)] or [b""]
    for j, blk in enumerate(blocks):
        flags = (_CHUNK_START if j == 0 else 0)
        if j == len(blocks) - 1:
            flags |= _CHUNK_END | (_ROOT if root else 0)
        cv = _b3_compress(cv, struct.unpack("<16I", blk.ljust(64, b"\0")), counter, len(blk), flags)
    return cv


def _b3_tree_cv(data, first, n_chunks, root):
    if n_chunks == 1:
        return _b3_chunk_cv(data[first * 1024:(first + 1) * 1024], first, root)
    left_n = 1 << ((n_chunks - 1).bit_length() - 1)  # largest power of two strictly below n_chunks
    left = _b3_tree_cv(data, first, left_n, False)
    right = _b3_tree_cv(data, first + left_n, n_chunks - left_n, False)
    return _b3_compress(_B3_IV, left + right, 0, 64, _PARENT | (_ROOT if root else 0))


def blake3_full(data: bytes) -> bytes:
    n_chunks = max(1, (len(data) + 1023) // 1024)
    return struct.pack("<8I", *_b3_tree_cv(data, 0, n_chunks, True))
