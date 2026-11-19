// Shared helpers of the NEB Metal kernels: BLAKE3 compression of one block, ChaCha8 block 0 and the integer helpers.
// Both kernels (neb_share.metal, neb_share2.metal) are compiled as this text followed by their own, so that the
// hash code exists once.

#include <metal_stdlib>
using namespace metal;

constant char B3PERM[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};


static inline uint rotl32(uint x, uint n) { return (x << n) | (x >> (32 - n)); }
static inline uint rotr32(uint x, uint n) { return (x >> n) | (x << (32 - n)); }
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---- BLAKE3 compression of one block, chunk counter 0 ----
// flags: CHUNK_START 1, CHUNK_END 2, PARENT 4, ROOT 8.

static inline void b3_g(thread uint *v, int a, int b, int c, int d, uint mx, uint my) {
  v[a] = v[a] + v[b] + mx;
  v[d] = rotr32(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];
  v[b] = rotr32(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + my;
  v[d] = rotr32(v[d] ^ v[a], 8);
  v[c] = v[c] + v[d];
  v[b] = rotr32(v[b] ^ v[c], 7);
}

constant uint B3IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu,
                          0x1F83D9ABu, 0x5BE0CD19u};

static void b3_compress(thread const uint cv[8], thread const uint blk[16], uint blen, uint flags, thread uint out[8]) {
  uint m[16];
  uint v[16];
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = blk[i];
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = cv[i];
  v[8] = B3IV[0]; v[9] = B3IV[1]; v[10] = B3IV[2]; v[11] = B3IV[3];
  v[12] = 0; v[13] = 0; v[14] = blen; v[15] = flags;
#pragma unroll
  for (int r = 0; r < 7; r++) {
    b3_g(v, 0, 4, 8, 12, m[0], m[1]);
    b3_g(v, 1, 5, 9, 13, m[2], m[3]);
    b3_g(v, 2, 6, 10, 14, m[4], m[5]);
    b3_g(v, 3, 7, 11, 15, m[6], m[7]);
    b3_g(v, 0, 5, 10, 15, m[8], m[9]);
    b3_g(v, 1, 6, 11, 12, m[10], m[11]);
    b3_g(v, 2, 7, 8, 13, m[12], m[13]);
    b3_g(v, 3, 4, 9, 14, m[14], m[15]);
    if (r < 6) {
      uint t[16];
#pragma unroll
      for (int i = 0; i < 16; i++) t[i] = m[B3PERM[i]];
#pragma unroll
      for (int i = 0; i < 16; i++) m[i] = t[i];
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

// ---- ChaCha8 ----

#define NEB_QR(a, b, c, d)                                                                  \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16); x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12); \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8); x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);

// Block 0 of ChaCha8(key): out[j] is the little-endian word u_j of spec section 5.1.
static void chacha8_block0(thread const uint key[8], thread uint out[16]) {
  uint s[16], x[16];
  s[0] = 0x61707865u; s[1] = 0x3320646eu; s[2] = 0x79622d32u; s[3] = 0x6b206574u;
#pragma unroll
  for (int i = 0; i < 8; i++) s[4 + i] = key[i];
  s[12] = 0; s[13] = 0; s[14] = 0; s[15] = 0;
#pragma unroll
  for (int i = 0; i < 16; i++) x[i] = s[i];
#pragma unroll
  for (int r = 0; r < 4; r++) {
    NEB_QR(0, 4, 8, 12) NEB_QR(1, 5, 9, 13) NEB_QR(2, 6, 10, 14) NEB_QR(3, 7, 11, 15)
    NEB_QR(0, 5, 10, 15) NEB_QR(1, 6, 11, 12) NEB_QR(2, 7, 8, 13) NEB_QR(3, 4, 9, 14)
  }
#pragma unroll
  for (int i = 0; i < 16; i++) out[i] = x[i] + s[i];
}
