// Device helpers of the tensor-core kernels K1 and K2 (shares_tc.cu). The ChaCha8 block and the BLAKE3 compression are
// those of the K0 kernel (shares_cuda.cu): same code, so the same bits. The mma wrappers do nothing below sm_80, where the host never launches the kernels.
// See pow/gpu/README.md.
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#define NIN 784
#define NH 128
#define NOUT 10
#define NB 16
#define N1 (NH * NIN)
#define N2 (NOUT * NH)
#define SHIFT1 5
#define SHIFT2 5
#define SHIFTB 3
#define GCLIP 256
#define TPOS 7
#define TNEG (-8)
#define OUT_BYTES 128  // digest (8 words), ticket (8 words), sample indices (16 words)

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ __forceinline__ int rsh(int x, int n) { return (x + (1 << (n - 1))) >> n; }

#define NEB_QR(a, b, c, d) \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16); x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12); \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8);  x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);

// Block 0 of ChaCha8(key): out[j] is the word u_j of spec section 5.1.
static __device__ void chacha8_block0(const uint32_t key[8], uint32_t out[16]) {
  uint32_t s[16], x[16];
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

__device__ __constant__ uint32_t kTcB3IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                              0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

__device__ __forceinline__ void b3_g(uint32_t *v, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  v[a] = v[a] + v[b] + mx; v[d] = rotr32(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];      v[b] = rotr32(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + my; v[d] = rotr32(v[d] ^ v[a], 8);
  v[c] = v[c] + v[d];      v[b] = rotr32(v[b] ^ v[c], 7);
}

// BLAKE3 compression of one block, chunk counter 0, first 8 words. flags: CHUNK_START 1, CHUNK_END 2, PARENT 4, ROOT 8.
static __device__ void b3_compress(const uint32_t cv[8], const uint32_t blk[16], uint32_t blen, uint32_t flags, uint32_t out[8]) {
  constexpr uint8_t perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t m[16], v[16];
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = blk[i];
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = cv[i];
  v[8] = kTcB3IV[0]; v[9] = kTcB3IV[1]; v[10] = kTcB3IV[2]; v[11] = kTcB3IV[3];
  v[12] = 0; v[13] = 0; v[14] = blen; v[15] = flags;
#pragma unroll
  for (int r = 0; r < 7; r++) {
    b3_g(v, 0, 4, 8, 12, m[0], m[1]);   b3_g(v, 1, 5, 9, 13, m[2], m[3]);
    b3_g(v, 2, 6, 10, 14, m[4], m[5]);  b3_g(v, 3, 7, 11, 15, m[6], m[7]);
    b3_g(v, 0, 5, 10, 15, m[8], m[9]);  b3_g(v, 1, 6, 11, 12, m[10], m[11]);
    b3_g(v, 2, 7, 8, 13, m[12], m[13]); b3_g(v, 3, 4, 9, 14, m[14], m[15]);
    if (r < 6) {
      uint32_t t[16];
#pragma unroll
      for (int i = 0; i < 16; i++) t[i] = m[perm[i]];
#pragma unroll
      for (int i = 0; i < 16; i++) m[i] = t[i];
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

// Tensor-core products, int8 x int8 -> int32 (sm_80 and later). Fragment layouts of the PTX ISA, g = lane >> 2, t = lane & 3:
// A (row-major 16 x K): a0 = A[g][4t..4t+3], a1 = A[g+8][4t..], a2 = A[g][16+4t..], a3 = A[g+8][16+4t..] (k32; k16 has a0, a1 only)
// B (column-major K x 8): b0 = B[4t..4t+3][g], b1 = B[16+4t..16+4t+3][g] (k32; k16 has b0 only)
// C (16 x 8): c0 = C[g][2t], c1 = C[g][2t+1], c2 = C[g+8][2t], c3 = C[g+8][2t+1]
__device__ __forceinline__ void mma_k32(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ void mma_k16(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
#endif
}

// Key of attempt blockIdx.x, from a 32-byte key or from an 80-byte header (key = BLAKE3(header), spec section 11.1), then the
// 16 sample indices and labels (spec section 5.1). Runs in one thread.
template <bool FROM_HEADER>
__device__ void derive_key_indices(const uint8_t *in, size_t share, uint32_t n, const uint8_t *labels, uint32_t *skey, uint32_t *sidx, int *slabel) {
  uint32_t key[8];
  if (FROM_HEADER) {
    const uint32_t *hw = (const uint32_t *)(in + 80 * share);
    uint32_t b0[16], b1[16], iv[8], cv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) iv[i] = kTcB3IV[i];
#pragma unroll
    for (int i = 0; i < 16; i++) { b0[i] = hw[i]; b1[i] = i < 4 ? hw[16 + i] : 0u; }
    b3_compress(iv, b0, 64, 1u, cv);
    b3_compress(cv, b1, 16, 2u | 8u, key);
  } else {
    const uint32_t *kw = (const uint32_t *)(in + 32 * share);
#pragma unroll
    for (int i = 0; i < 8; i++) key[i] = kw[i];
  }
#pragma unroll
  for (int i = 0; i < 8; i++) skey[i] = key[i];
  uint32_t u[16];
  chacha8_block0(key, u);
#pragma unroll
  for (int j = 0; j < NB; j++) {
    sidx[j] = (uint32_t)(((unsigned long long)u[j] * n) >> 32);
    slabel[j] = labels[sidx[j]];
  }
}

// ticket = BLAKE3(digest || key): one 64-byte block (CHUNK_START | CHUNK_END | ROOT); writes the 128-byte record.
static __device__ void finish_record(const uint32_t *digest, const uint32_t *key, const uint32_t *idx, uint8_t *out, size_t share) {
  uint32_t blk[16], iv[8], res[8];
#pragma unroll
  for (int i = 0; i < 8; i++) { blk[i] = digest[i]; blk[8 + i] = key[i]; iv[i] = kTcB3IV[i]; }
  b3_compress(iv, blk, 64, 1u | 2u | 8u, res);
  uint32_t *o32 = (uint32_t *)(out + (size_t)OUT_BYTES * share);
#pragma unroll
  for (int i = 0; i < 8; i++) { o32[i] = digest[i]; o32[8 + i] = res[i]; }
#pragma unroll
  for (int j = 0; j < NB; j++) o32[16 + j] = idx[j];
}
