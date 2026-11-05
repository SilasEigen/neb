// Training shares on CUDA, one thread block per share: spec/shares.md sections 5.1 to 5.4. The contract is in
// shares_cuda.h. The results must equal pow/ref/shares.c bit for bit: dp4a accumulates exact int32 sums, and the digest
// lanes are sums modulo 2^32, which do not depend on the order of the entries.
#include "shares_cuda.h"
#include <string.h>
#include <vector>
#include "shares.h"  // the constants of the C reference; only macros and types are used here

#define NIN NEB_NIN
#define NH NEB_NH
#define NOUT NEB_NOUT
#define NB NEB_NB
#define NW (NIN / 4)   // 196 words of 4 pixels or weights
#define NTHREADS 256  // two threads per hidden unit, one per half of the 16 samples
static_assert(NTHREADS % 8 == 0 && NH == 128 && NB == 16 && NIN % 4 == 0, "the kernel layout assumes these sizes");

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

#define NEB_QR(a, b, c, d) \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 16); x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 12); \
  x[a] += x[b]; x[d] = rotl32(x[d] ^ x[a], 8);  x[c] += x[d]; x[b] = rotl32(x[b] ^ x[c], 7);

// Block 0 of ChaCha8(key) (counter 0, zero IV). The output words are the little-endian words of the keystream, so
// out[j] is the word u_j of spec section 5.1.
__device__ void chacha8_block0(const uint32_t key[8], uint32_t out[16]) {
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

__device__ __constant__ uint32_t kB3IV[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
                                              0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};

__device__ __forceinline__ void b3_g(uint32_t *v, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
  v[a] = v[a] + v[b] + mx; v[d] = rotr32(v[d] ^ v[a], 16);
  v[c] = v[c] + v[d];      v[b] = rotr32(v[b] ^ v[c], 12);
  v[a] = v[a] + v[b] + my; v[d] = rotr32(v[d] ^ v[a], 8);
  v[c] = v[c] + v[d];      v[b] = rotr32(v[b] ^ v[c], 7);
}

// BLAKE3 compression of one block with chunk counter 0, returning the first 8 words: the chaining value, or the hash when
// ROOT is set. Only single-chunk inputs are hashed here (the 80-byte header and the 64-byte digest || key).
// flags: CHUNK_START 1, CHUNK_END 2, PARENT 4, ROOT 8
__device__ void b3_compress(const uint32_t cv[8], const uint32_t blk[16], uint32_t blen, uint32_t flags, uint32_t out[8]) {
  constexpr uint8_t perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
  uint32_t m[16], v[16];
#pragma unroll
  for (int i = 0; i < 16; i++) m[i] = blk[i];
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = cv[i];
  v[8] = kB3IV[0]; v[9] = kB3IV[1]; v[10] = kB3IV[2]; v[11] = kB3IV[3];
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

__device__ __forceinline__ int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
__device__ __forceinline__ int rsh(int x, int n) { return (x + (1 << (n - 1))) >> n; }

// Shared memory of one share: 33,888 bytes, below the 48 KiB static limit. The 16 samples are held twice, packed along
// the pixels (xrow, for the forward product) and along the samples (xcol, for the G1 product), so that dp4a always
// sums along the contracted index.
struct Smem {
  int32_t xrow[NB * NW];        // sample b, pixels packed 4 per word along i
  int32_t xcol[4 * NIN];        // pixel i, samples packed 4 per word along b: group g at g * NIN + i
  int8_t a1[NB * NH];
  uint8_t mask[NB * NH];        // 1 where the layer-1 activation was not clipped (-8 <= y1 <= 7)
  int32_t e1p[NH * 4];          // E1 of unit h, packed along b
  int8_t e2[NB * NOUT];
  int8_t w2[NOUT * NH];
  uint32_t red[NTHREADS];
  uint32_t key[8];
  uint32_t idx[NB];
  int label[NB];
  uint32_t digest[8];
};

// One block of NTHREADS threads evaluates share blockIdx.x. Phases, each ended by __syncthreads: key, sample indices
// and labels (thread 0); copy of the samples; layer 1; layer 2 and the output error; the propagated error; gradients,
// clip and digest lanes; ticket (thread 0). In the layer and error phases thread tid handles hidden unit h = tid & 127
// for the samples 8 * (tid >> 7) .. 8 * (tid >> 7) + 7.
template <bool FROM_HEADER>
__global__ void __launch_bounds__(NTHREADS) neb_share_kernel(const uint8_t *in, const int32_t *w1t, const int8_t *w2g, const uint8_t *data,
                                                              const uint8_t *labels, uint32_t n, uint8_t *out) {
  __shared__ Smem S;
  const int tid = threadIdx.x;
  const size_t share = blockIdx.x;

  if (tid == 0) {
    uint32_t key[8];
    if (FROM_HEADER) {
      // key = BLAKE3(header): one chunk of two blocks, bytes 0..63 (CHUNK_START) and bytes 64..79 zero-padded with
      // block length 16 (CHUNK_END | ROOT).
      const uint32_t *hw = (const uint32_t *)(in + 80 * share);
      uint32_t b0[16], b1[16], iv[8], cv[8];
#pragma unroll
      for (int i = 0; i < 8; i++) iv[i] = kB3IV[i];
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
    for (int i = 0; i < 8; i++) S.key[i] = key[i];
    uint32_t u[16];
    chacha8_block0(key, u);
#pragma unroll
    for (int j = 0; j < NB; j++) {
      // u_j and n are below 2^32, so the product fits in 64 bits and the quotient by 2^32 is below n.
      S.idx[j] = (uint32_t)(((unsigned long long)u[j] * n) >> 32);
      S.label[j] = labels[S.idx[j]];
    }
  }
  // W2 (1280 bytes) to shared memory as 32-bit words; the barrier below also orders this copy before layer 2
  for (int i = tid; i < NOUT * NH / 4; i += NTHREADS) ((int32_t *)S.w2)[i] = ((const int32_t *)w2g)[i];
  __syncthreads();

  // Samples as words; pixels are 0..7, so the bytes are valid int8 values for dp4a.
  for (int e = tid; e < NB * NW; e += NTHREADS) {
    const int b = e / NW, w = e - b * NW;
    S.xrow[e] = ((const int32_t *)(data + (size_t)S.idx[b] * NIN))[w];
  }
  __syncthreads();
  // Transpose to the sample-packed form: xcol[g * NIN + i] holds X[4g .. 4g+3][i], sample 4g in the low byte. It is
  // first read in the gradient phase, after later barriers.
  const int8_t *xb = (const int8_t *)S.xrow;
  for (int e = tid; e < 4 * NIN; e += NTHREADS) {
    const int g = e / NIN, i = e - g * NIN;
    const uint32_t b0 = (uint8_t)xb[(4 * g + 0) * NIN + i], b1 = (uint8_t)xb[(4 * g + 1) * NIN + i];
    const uint32_t b2 = (uint8_t)xb[(4 * g + 2) * NIN + i], b3 = (uint8_t)xb[(4 * g + 3) * NIN + i];
    S.xcol[e] = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));
  }

  // Layer 1: z1[b][h] = sum over i of X[b][i] * W1[h][i], |z1| <= 43904; consecutive threads read consecutive words of
  // w1t. y1 = rsh(z1, S1) gives the activation a1; mask marks the unclipped ones, where the error passes back.
  {
    const int h = tid & (NH - 1), b0 = (tid >> 7) * 8;
    int acc[8];
#pragma unroll
    for (int k = 0; k < 8; k++) acc[k] = 0;
    for (int w = 0; w < NW; w++) {
      const int32_t wv = w1t[w * NH + h];
#pragma unroll
      for (int k = 0; k < 8; k++) acc[k] = __dp4a(S.xrow[(b0 + k) * NW + w], wv, acc[k]);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int y = rsh(acc[k], NEB_S1);
      S.a1[(b0 + k) * NH + h] = (int8_t)clampi(y, -8, 7);
      S.mask[(b0 + k) * NH + h] = (y >= -8 && y <= 7);
    }
  }
  __syncthreads();

  // Layer 2 and the output error for the 160 (sample, class) pairs, |z2| <= 8192:
  // E2 = clip(clip(rsh(z2, S2), -8, 7) - T, -8, 7), with T = TPOS for the label and TNEG otherwise (section 5.2).
  if (tid < NB * NOUT) {
    const int b = tid / NOUT, o = tid - b * NOUT;
    int z = 0;
    for (int w = 0; w < NH / 4; w++) z = __dp4a(((const int32_t *)S.a1)[b * (NH / 4) + w], ((const int32_t *)S.w2)[o * (NH / 4) + w], z);
    const int y2 = clampi(rsh(z, NEB_S2), -8, 7);
    const int t = (o == S.label[b]) ? NEB_TPOS : NEB_TNEG;
    S.e2[tid] = (int8_t)clampi(y2 - t, -8, 7);
  }
  __syncthreads();

  // Propagated error: E1[b][h] = mask ? clip(rsh(m, SB), -8, 7) : 0 with m = sum over o of E2[b][o] * W2[o][h],
  // |m| <= 640. The thread packs its 8 samples into the two words e1p[h * 4 + 2 * half + 0 and 1], 4 samples per word.
  {
    const int h = tid & (NH - 1), b0 = (tid >> 7) * 8;
    uint32_t pk[2] = {0u, 0u};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      int m = 0;
#pragma unroll
      for (int o = 0; o < NOUT; o++) m += (int)S.e2[(b0 + k) * NOUT + o] * (int)S.w2[o * NH + h];
      const int e1 = S.mask[(b0 + k) * NH + h] ? clampi(rsh(m, NEB_SB), -8, 7) : 0;
      pk[k >> 2] |= ((uint32_t)(uint8_t)e1) << (8 * (k & 3));
    }
    S.e1p[h * 4 + (tid >> 7) * 2 + 0] = (int32_t)pk[0];
    S.e1p[h * 4 + (tid >> 7) * 2 + 1] = (int32_t)pk[1];
  }
  __syncthreads();

  // Gradients, clip and digest lanes (sections 5.2 and 5.3); before the clip |G2| <= 1024 and |G1| <= 896. The flat
  // index is f = tid + 256 k, so f mod 8 = tid mod 8: a thread feeds only the lane tid & 7. The lane constant is
  // 7919 * l with l = 2 for G2 and l = 1 for G1.
  uint32_t lane = 0;
  for (int f = tid; f < NOUT * NH; f += NTHREADS) {
    const int o = f / NH, h = f - o * NH;
    int g = 0;
#pragma unroll
    for (int b = 0; b < NB; b++) g += (int)S.e2[b * NOUT + o] * (int)S.a1[b * NH + h];
    lane += (uint32_t)clampi(g, -NEB_CLIP, NEB_CLIP) ^ ((uint32_t)f + 7919u * 2u);
  }
  for (int f = tid; f < NH * NIN; f += NTHREADS) {
    const int h = f / NIN, i = f - h * NIN;
    int g = __dp4a(S.e1p[h * 4 + 0], S.xcol[0 * NIN + i], 0);
    g = __dp4a(S.e1p[h * 4 + 1], S.xcol[1 * NIN + i], g);
    g = __dp4a(S.e1p[h * 4 + 2], S.xcol[2 * NIN + i], g);
    g = __dp4a(S.e1p[h * 4 + 3], S.xcol[3 * NIN + i], g);
    lane += (uint32_t)clampi(g, -NEB_CLIP, NEB_CLIP) ^ ((uint32_t)f + 7919u * 1u);
  }
  // Lane d is the sum of the NTHREADS / 8 = 32 partial lanes of the threads with tid & 7 = d.
  S.red[tid] = lane;
  __syncthreads();
  if (tid < 8) {
    uint32_t d = 0;
    for (int k = 0; k < NTHREADS / 8; k++) d += S.red[tid + 8 * k];
    S.digest[tid] = d;
  }
  __syncthreads();

  // ticket = BLAKE3(digest || key): one 64-byte block, CHUNK_START | CHUNK_END | ROOT. The output record is the
  // digest, the ticket and the sample indices.
  if (tid == 0) {
    uint32_t blk[16], iv[8], res[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { blk[i] = S.digest[i]; blk[8 + i] = S.key[i]; iv[i] = kB3IV[i]; }
    b3_compress(iv, blk, 64, 1u | 2u | 8u, res);
    uint32_t *o32 = (uint32_t *)(out + (size_t)NEB_SHARE_OUT_BYTES * share);
#pragma unroll
    for (int i = 0; i < 8; i++) { o32[i] = S.digest[i]; o32[8 + i] = res[i]; }
#pragma unroll
    for (int j = 0; j < NB; j++) o32[16 + j] = S.idx[j];
  }
}

// w1t[w * NH + h] = the 4 weights W1[h][4w .. 4w+3], column 4w in the low byte; consecutive h are consecutive words, so
// the kernel reads coalesced.
static void pack_w1t(const int8_t *w1, int32_t *w1t) {
  for (int h = 0; h < NH; h++)
    for (int w = 0; w < NW; w++) {
      const int8_t *p = w1 + (size_t)h * NIN + 4 * w;
      w1t[(size_t)w * NH + h] = (int32_t)((uint32_t)(uint8_t)p[0] | ((uint32_t)(uint8_t)p[1] << 8) | ((uint32_t)(uint8_t)p[2] << 16) | ((uint32_t)(uint8_t)p[3] << 24));
    }
}

cudaError_t neb_share_gpu_set_model(neb_share_gpu *g, const int8_t *w1, const int8_t *w2) {
  // 100,352 bytes, local to the call: two threads that update two devices at the same time share nothing. (Before, this
  // buffer was a static, and a model uploaded while another thread was packing its own could be corrupted.)
  std::vector<int32_t> host_w1t((size_t)NW * NH);
  pack_w1t(w1, host_w1t.data());
  cudaError_t e = cudaMemcpy(g->w1t, host_w1t.data(), host_w1t.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return e;
  return cudaMemcpy(g->w2, w2, NOUT * NH, cudaMemcpyHostToDevice);
}

cudaError_t neb_share_gpu_init(neb_share_gpu *g, const int8_t *w1, const int8_t *w2, const uint8_t *data, const uint8_t *labels, uint32_t n) {
  memset(g, 0, sizeof *g);
  g->n = n;
  cudaError_t e;
  if ((e = cudaMalloc(&g->w1t, sizeof(int32_t) * NW * NH)) != cudaSuccess) return e;
  if ((e = cudaMalloc(&g->w2, NOUT * NH)) != cudaSuccess) return e;
  if ((e = cudaMalloc(&g->data, (size_t)n * NIN)) != cudaSuccess) return e;
  if ((e = cudaMalloc(&g->labels, n)) != cudaSuccess) return e;
  if ((e = cudaMemcpy(g->data, data, (size_t)n * NIN, cudaMemcpyHostToDevice)) != cudaSuccess) return e;
  if ((e = cudaMemcpy(g->labels, labels, n, cudaMemcpyHostToDevice)) != cudaSuccess) return e;
  return neb_share_gpu_set_model(g, w1, w2);
}

void neb_share_gpu_free(neb_share_gpu *g) {
  cudaFree(g->w1t); cudaFree(g->w2); cudaFree(g->data); cudaFree(g->labels);
  memset(g, 0, sizeof *g);
}

cudaError_t neb_share_gpu_launch(const neb_share_gpu *g, const uint8_t *d_in, int input_is_header, size_t count, uint8_t *d_out) {
  if (count == 0 || count > 0x7FFFFFFFu) return cudaErrorInvalidValue;
  if (input_is_header) neb_share_kernel<true><<<(unsigned)count, NTHREADS>>>(d_in, g->w1t, g->w2, g->data, g->labels, g->n, d_out);
  else neb_share_kernel<false><<<(unsigned)count, NTHREADS>>>(d_in, g->w1t, g->w2, g->data, g->labels, g->n, d_out);
  return cudaGetLastError();
}
