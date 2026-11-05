// K1 and K2: training shares on tensor cores (int8 mma), one thread block of 256 threads per share, for the GPU miner.
// Mining optimisation, not consensus: the node validates with pow/ref, and the miner evaluates every nonce that this code
// reports again with the C reference before it sends it. Bit for bit equal to pow/ref/shares.c (digest, ticket, the 16
// sample indices and the clipped gradients), checked by the self-test of neb-miner-gpu and pow/tests/test_shares_tc.cu.
//
// The interface is shares_tc.h. The host staging buffer of the model is local to the call, so that two threads can upload
// models at the same time, and the kernel does nothing on a device older than sm_80 (the host refuses to launch it
// there). See pow/gpu/README.md.
//
// One template, SPARSE = false (K1) or true (K2). K2 differs from K1 only in the G1 phase: it computes G1 only for the hidden
// units whose propagated error is not zero (plus inactive units as padding up to a multiple of 16 rows) and for the 8-pixel
// tiles that are not zero in all 16 samples. The entries that are skipped are exactly zero, so each would add (f + 7919) to
// its digest lane; their sum has a closed form and is added once, and the computed entries are corrected by what they would
// have added if they were zero. Units whose bound 7 * sum |E1| is at most the clip skip the clip. Phases of a block
// (spec/shares.md sections 5.1 to 5.4):
//  A  thread 0: key, sample indices, labels (as the frozen kernel); all threads: W2 to shared memory
//  B  the 16 samples are loaded once from global memory and written twice to shared memory: rows of pixels (xrow, A operand of
//     the forward product) and 4-sample words per pixel (xcol, B operand of the G1 product), by a 4x4 byte transpose
//  C  forward product Z[b][h] = sum_i X[b][i] W1[h][i] with mma m16n8k32: M = 16 samples, N = 128 hidden units (2 n-tiles per
//     warp), K = 800 (784 padded with zero pixels); W1 is read from global memory already in B-fragment order
//  D  layer 2 and the output error (dp4a, 160 threads), propagated error E1 (all threads)
//  E  G2 by scalar sums, G1 = E1^T X by mma m16n8k16 (M = 128 hidden units, one m-tile per warp, N = 784 pixels), each product
//     clipped and folded into the digest lanes in registers; lane f mod 8 of a G1 entry is 2t + (column parity) because 784 = 0 mod 8
//  F  digest lanes added in shared memory, ticket by a separate kernel
#include "shares_tc.h"
#include "tc_common.cuh"
#include <stdlib.h>
#include <string.h>
#include <vector>

constexpr int XS = 816;        // bytes per row of xrow: 204 words, so that the lanes of a warp read 32 different banks
constexpr int CS = 792;        // words per plane of xcol: 792 = 24 mod 32, same reason
constexpr int AS = 132;        // bytes per row of a1 and mask
constexpr int W2S = 33;        // words per row of W2 in shared memory (no bank conflicts when 10 rows are read in parallel)
constexpr int KSTEPS = 25;     // 25 steps of 32 pixels = 800
constexpr int NTHREADS = 256;
constexpr int NWARP = 8;
[[maybe_unused]] constexpr int NPIXQ = NIN / 4; // 196 words of 4 pixels
#ifdef NEB_NEGATIVE_CONTROL_TC_CLIP
#define CLIP_LO (-GCLIP + 1)  // deliberately wrong: pow/tests/check_shares_tc.* build with it and the test must fail
#else
#define CLIP_LO (-GCLIP)
#endif
#ifndef MINB
#define MINB 3  // blocks per SM the compiler must allow (registers); the shared memory of a block allows 3
#endif

// Per-attempt data produced by derive_kernel and consumed by the main kernel and the ticket kernel (128 bytes).
struct __align__(16) Aux {
  uint32_t key[8];
  uint32_t idx[NB];
  uint8_t label[NB];
  uint32_t pad[4];
};
static_assert(sizeof(Aux) == 128, "Aux is one 128-byte record");

template <bool FROM_HEADER>
__global__ void __launch_bounds__(128) derive_kernel(const uint8_t *in, uint32_t n, const uint8_t *labels, size_t count, Aux *aux) {
  const size_t share = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (share >= count) return;
  uint32_t key[8], idx[NB];
  int lab[NB];
  // derive_key_indices indexes the input with its own share number, so it is called on the slice of this attempt
  derive_key_indices<FROM_HEADER>(in + share * (FROM_HEADER ? 80 : 32), 0, n, labels, key, idx, lab);
  Aux a;
#pragma unroll
  for (int i = 0; i < 8; i++) a.key[i] = key[i];
#pragma unroll
  for (int j = 0; j < NB; j++) { a.idx[j] = idx[j]; a.label[j] = (uint8_t)lab[j]; }
  a.pad[0] = a.pad[1] = a.pad[2] = a.pad[3] = 0;
  aux[share] = a;
}

// ticket = BLAKE3(digest || key); the main kernel has written the digest and the sample indices of the record
__global__ void __launch_bounds__(128) ticket_kernel(const Aux *aux, size_t count, uint8_t *out) {
  const size_t share = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (share >= count) return;
  uint32_t *o32 = (uint32_t *)(out + (size_t)OUT_BYTES * share);
  uint32_t blk[16], iv[8], res[8];
#pragma unroll
  for (int i = 0; i < 8; i++) { blk[i] = o32[i]; blk[8 + i] = aux[share].key[i]; iv[i] = kTcB3IV[i]; }
  b3_compress(iv, blk, 64, 1u | 2u | 8u, res);
#pragma unroll
  for (int i = 0; i < 8; i++) o32[8 + i] = res[i];
}

__device__ __forceinline__ int sumabs4(uint32_t w) {
  int s = 0;
#pragma unroll
  for (int k = 0; k < 4; k++) { const int b = (int)(int8_t)(w >> (8 * k)); s += b < 0 ? -b : b; }
  return s;
}

struct __align__(16) S1 {
  uint8_t xrow[NB * XS];
  uint32_t xcol[4 * CS];
  int8_t a1[NB * AS];
  uint8_t mask[NB * AS];
  int8_t e2[NB * NOUT];
  uint32_t w2[NOUT * W2S];
  uint32_t idx[NB], digest[8];
  int label[NB];
  uint8_t nzq[200];  // K2: byte q is 1 when the pixel quad q (pixels 4q..4q+3) is not zero in some sample
};

template <bool SPARSE, bool DEBUG>
__global__ void __launch_bounds__(NTHREADS, MINB) k1_kernel(const Aux *aux, const uint4 *w1p, const uint32_t *w2g, const uint8_t *data,
                                                         uint8_t *out, int16_t *dbg) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  __trap();  // the tensor-core instructions below need sm_80; the host never launches this kernel on an older device
#else
  __shared__ S1 S;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const size_t share = blockIdx.x;
  uint32_t *const e1p = (uint32_t *)S.xrow;  // aliases xrow, written only after the barrier that ends the forward phase
  if (tid < NB) { S.idx[tid] = aux[share].idx[tid]; S.label[tid] = aux[share].label[tid]; }
  for (int e = tid; e < NOUT * NH / 4; e += NTHREADS) S.w2[(e >> 5) * W2S + (e & 31)] = w2g[e];
  if (tid < 8) S.digest[tid] = 0;
  if (SPARSE && tid < 50) ((uint32_t *)S.nzq)[tid] = 0;
  if (tid < NB * 4) ((uint32_t *)(S.xrow + (tid >> 2) * XS))[NPIXQ + (tid & 3)] = 0;  // pixels 784..799 are zero
  __syncthreads();

  // B: sample rows and their transposed copy. Task (tq, q): the 4 rows 4tq..4tq+3, the word q of each (pixels 4q..4q+3).
  // The 784 tasks give at most 4 per thread; their loads are issued together.
  {
    uint32_t v[4][4];
#pragma unroll
    for (int it = 0; it < 4; it++) {
      const int e = tid + it * NTHREADS;
      if (e < 4 * NPIXQ) {
        const int tq = e / NPIXQ, q = e - tq * NPIXQ;
#pragma unroll
        for (int k = 0; k < 4; k++) {
          v[it][k] = __ldg(((const uint32_t *)(data + (size_t)S.idx[4 * tq + k] * NIN)) + q);
        }
      }
    }
#pragma unroll
    for (int it = 0; it < 4; it++) {
      const int e = tid + it * NTHREADS;
      if (e < 4 * NPIXQ) {
        const int tq = e / NPIXQ, q = e - tq * NPIXQ;
#pragma unroll
        for (int k = 0; k < 4; k++) ((uint32_t *)(S.xrow + (4 * tq + k) * XS))[q] = v[it][k];
        if (SPARSE && (v[it][0] | v[it][1] | v[it][2] | v[it][3]) != 0) S.nzq[q] = 1;
        // 4x4 byte transpose: word j of the result holds byte j of v[0..3], v[0] in the low byte
        const uint32_t t0 = __byte_perm(v[it][0], v[it][1], 0x5140), t1 = __byte_perm(v[it][2], v[it][3], 0x5140);
        const uint32_t t2 = __byte_perm(v[it][0], v[it][1], 0x7362), t3 = __byte_perm(v[it][2], v[it][3], 0x7362);
        uint4 w;
        w.x = __byte_perm(t0, t1, 0x5410); w.y = __byte_perm(t0, t1, 0x7632);
        w.z = __byte_perm(t2, t3, 0x5410); w.w = __byte_perm(t2, t3, 0x7632);
        *(uint4 *)&S.xcol[tq * CS + 4 * q] = w;
      }
    }
  }
  __syncthreads();

  // C: forward product. A = samples (g, g+8), B = W1 fragments of the hidden units 16 * warp + 8 * j + g.
  {
    int c[2][4];
#pragma unroll
    for (int j = 0; j < 2; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0;
    const uint4 *wp = w1p + (size_t)warp * KSTEPS * 32 + lane;
    const uint32_t *xr = (const uint32_t *)S.xrow;
#pragma unroll 5
    for (int ks = 0; ks < KSTEPS; ks++) {
      const uint4 B = __ldg(wp + ks * 32);
      const uint32_t a0 = xr[g * (XS / 4) + ks * 8 + t], a1 = xr[(g + 8) * (XS / 4) + ks * 8 + t];
      const uint32_t a2 = xr[g * (XS / 4) + ks * 8 + 4 + t], a3 = xr[(g + 8) * (XS / 4) + ks * 8 + 4 + t];
      mma_k32(c[0], a0, a1, a2, a3, B.x, B.y);
      mma_k32(c[1], a0, a1, a2, a3, B.z, B.w);
    }
    // c[j][0..1] = samples g, hidden 16 warp + 8 j + 2t + {0,1}; c[j][2..3] = sample g + 8. |z1| <= 43904.
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
      for (int half = 0; half < 2; half++) {
        const int h = 16 * warp + 8 * j + 2 * t, b = g + 8 * half;
        const int y0 = rsh(c[j][2 * half], SHIFT1), y1 = rsh(c[j][2 * half + 1], SHIFT1);
        const uint32_t av = (uint32_t)(uint8_t)clampi(y0, -8, 7) | ((uint32_t)(uint8_t)clampi(y1, -8, 7) << 8);
        const uint32_t mv = (uint32_t)(y0 >= -8 && y0 <= 7) | ((uint32_t)(y1 >= -8 && y1 <= 7) << 8);
        *(uint16_t *)(S.a1 + b * AS + h) = (uint16_t)av;
        *(uint16_t *)(S.mask + b * AS + h) = (uint16_t)mv;
      }
  }
  __syncthreads();

  // D: layer 2 and the output error for the 160 (sample, class) pairs, |z2| <= 8192
  if (tid < NB * NOUT) {
    const int b = tid / NOUT, o = tid - b * NOUT;
    int z = 0;
#pragma unroll
    for (int w = 0; w < NH / 4; w++) z = __dp4a(((const int32_t *)S.a1)[b * (AS / 4) + w], (int32_t)S.w2[o * W2S + w], z);
    const int y2 = clampi(rsh(z, SHIFT2), -8, 7);
    const int tt = (o == S.label[b]) ? TPOS : TNEG;
    S.e2[tid] = (int8_t)clampi(y2 - tt, -8, 7);
  }
  __syncthreads();
  // propagated error, |m| <= 640; the thread packs its 8 samples into two words of e1p (4 samples per word, sample 4k in the low byte)
  {
    const int h = tid & (NH - 1), b0 = (tid >> 7) * 8;
    const int8_t *w2b = (const int8_t *)S.w2;
    uint32_t pk[2] = {0u, 0u};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      int m = 0;
#pragma unroll
      for (int o = 0; o < NOUT; o++) m += (int)S.e2[(b0 + k) * NOUT + o] * (int)w2b[o * (W2S * 4) + h];
      const int e1 = S.mask[(b0 + k) * AS + h] ? clampi(rsh(m, SHIFTB), -8, 7) : 0;
      pk[k >> 2] |= ((uint32_t)(uint8_t)e1) << (8 * (k & 3));
    }
    e1p[h * 4 + (tid >> 7) * 2 + 0] = pk[0];
    e1p[h * 4 + (tid >> 7) * 2 + 1] = pk[1];
  }
  __syncthreads();

  // E: gradients, clip and digest lanes. G2 first (flat index f = tid + 256 k, so the lane is tid & 7), l = 2 for G2 and 1 for G1.
  uint32_t lane2 = 0;
  for (int f = tid; f < N2; f += NTHREADS) {
    const int o = f >> 7, h = f & (NH - 1);
    int gs = 0;
#pragma unroll
    for (int b = 0; b < NB; b++) gs += (int)S.e2[b * NOUT + o] * (int)S.a1[b * AS + h];
    const int cc = clampi(gs, -GCLIP, GCLIP);
    lane2 += (uint32_t)cc ^ ((uint32_t)f + 7919u * 2u);
    if (DEBUG) dbg[share * (size_t)(N1 + N2) + N1 + f] = (int16_t)cc;
  }
  if constexpr (!SPARSE) {
    const uint32_t ea = e1p[(16 * warp + g) * 4 + t], eb = e1p[(16 * warp + g + 8) * 4 + t];
    const uint32_t *xc = &S.xcol[t * CS + g];
    uint32_t d0 = 0, d1 = 0;
    const uint32_t f0b = (uint32_t)(16 * warp + g) * NIN + 2 * t + 7919u, f8b = f0b + 8u * NIN;
#pragma unroll 2
    for (int nt = 0; nt < NIN / 8; nt++) {
      int c[4] = {0, 0, 0, 0};
      mma_k16(c, ea, eb, xc[8 * nt]);
      // c0, c1: hidden unit 16 warp + g, pixels 8 nt + 2t + {0,1}; c2, c3: hidden unit + 8. |G1| <= 896 before the clip.
      const uint32_t f0 = f0b + 8u * nt, f8 = f8b + 8u * nt;
      const int c0 = clampi(c[0], CLIP_LO, GCLIP), c1 = clampi(c[1], CLIP_LO, GCLIP);
      const int c2 = clampi(c[2], CLIP_LO, GCLIP), c3 = clampi(c[3], CLIP_LO, GCLIP);
      d0 += ((uint32_t)c0 ^ f0) + ((uint32_t)c2 ^ f8);
      d1 += ((uint32_t)c1 ^ (f0 + 1u)) + ((uint32_t)c3 ^ (f8 + 1u));
      if (DEBUG) {
        const size_t o = share * (size_t)(N1 + N2);
        const int h0 = 16 * warp + g, p0 = 8 * nt + 2 * t;
        dbg[o + (size_t)h0 * NIN + p0] = (int16_t)c0; dbg[o + (size_t)h0 * NIN + p0 + 1] = (int16_t)c1;
        dbg[o + (size_t)(h0 + 8) * NIN + p0] = (int16_t)c2; dbg[o + (size_t)(h0 + 8) * NIN + p0 + 1] = (int16_t)c3;
      }
    }
    // lanes with the same t hold the same two digest lanes: sum over g (lane bits 2..4) then add into shared memory
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) { d0 += __shfl_xor_sync(0xffffffffu, d0, o); d1 += __shfl_xor_sync(0xffffffffu, d1, o); }
    if (lane < 4) { atomicAdd(&S.digest[2 * t], d0); atomicAdd(&S.digest[2 * t + 1], d1); }
  }
  else {
    // lists in the memory of xrow (dead since the forward phase), after the 2048 bytes of e1p
    uint8_t *const unitsS = S.xrow + 2048, *const unitsU = unitsS + 128, *const unitsZ = unitsU + 128, *const tiles = unitsZ + 128;
    uint32_t *const meta = (uint32_t *)(tiles + 128);  // 0 S_n, 1 U_n, 2 tilesS, 3 combined, 4 m-tiles, 5 sum of row ids, 6 tiles, 7 sum of tile ids
    if (warp == 0) {
      // hidden units: S = error not zero and 7 * sum |E1| <= clip (no clip needed), U = error not zero, Z = error zero
      uint32_t cS = 0, cU = 0, cZ = 0, sumSU = 0;
      const unsigned lt = (1u << lane) - 1u;
#pragma unroll
      for (int it = 0; it < 4; it++) {
        const int u = it * 32 + lane;
        const uint4 ev = *(const uint4 *)&e1p[u * 4];
        const bool any = (ev.x | ev.y | ev.z | ev.w) != 0;
        const bool safe = (sumabs4(ev.x) + sumabs4(ev.y) + sumabs4(ev.z) + sumabs4(ev.w)) * 7 <= GCLIP;
        const unsigned mS = __ballot_sync(0xffffffffu, any && safe), mU = __ballot_sync(0xffffffffu, any && !safe), mZ = __ballot_sync(0xffffffffu, !any);
        if (any && safe) unitsS[cS + __popc(mS & lt)] = (uint8_t)u;
        if (any && !safe) unitsU[cU + __popc(mU & lt)] = (uint8_t)u;
        if (!any) unitsZ[cZ + __popc(mZ & lt)] = (uint8_t)u;
        cS += __popc(mS); cU += __popc(mU); cZ += __popc(mZ);
        if (any) sumSU += (uint32_t)u;
      }
      sumSU = __reduce_add_sync(0xffffffffu, sumSU);
      const int tS = (int)(cS + 15) >> 4, tU = (int)(cU + 15) >> 4;
      const int comb = (tS + tU) > 8;  // two groups of tiles need distinct padding units: not always possible
      const int nM = comb ? (int)(cS + cU + 15) >> 4 : tS + tU;
      const int pads = 16 * nM - (int)(cS + cU);  // inactive units used as zero rows
      __syncwarp();
      uint32_t sp = 0;
      for (int i = lane; i < pads; i += 32) sp += unitsZ[i];
      sp = __reduce_add_sync(0xffffffffu, sp);
      if (lane == 0) { meta[0] = cS; meta[1] = cU; meta[2] = (uint32_t)tS; meta[3] = (uint32_t)comb; meta[4] = (uint32_t)nM; meta[5] = sumSU + sp; }
    } else if (warp == 1) {
      // pixel tiles of 8 pixels that are not zero in all 16 samples
      uint32_t cT = 0, sumT = 0;
      const unsigned lt = (1u << lane) - 1u;
#pragma unroll
      for (int it = 0; it < 4; it++) {
        const int nt = it * 32 + lane;
        const bool act = nt < NIN / 8 && (S.nzq[2 * nt] | S.nzq[2 * nt + 1]) != 0;
        const unsigned m = __ballot_sync(0xffffffffu, act);
        if (act) tiles[cT + __popc(m & lt)] = (uint8_t)nt;
        cT += __popc(m);
        sumT += act ? (uint32_t)nt : 0u;
      }
      sumT = __reduce_add_sync(0xffffffffu, sumT);
      if (lane == 0) { meta[6] = cT; meta[7] = sumT; }
    }
    __syncthreads();
    const int S_n = (int)meta[0], U_n = (int)meta[1], tS = (int)meta[2], comb = (int)meta[3], nM = (int)meta[4], NT = (int)meta[6];
    const uint32_t sumH = meta[5], sumNT = meta[7];
    const uint32_t *xc = &S.xcol[t * CS + g];
    uint32_t d0 = 0, d1 = 0;
    for (int mt = 0; mt < nM; mt++) {
      // hidden units of rows g and g + 8 of this m-tile: the active ones first, then zero units as padding
      int rowA, rowB;
#pragma unroll
      for (int half = 0; half < 2; half++) {
        const int s2 = 16 * mt + g + 8 * half;
        int u;
        if (comb) u = s2 < S_n ? unitsS[s2] : (s2 < S_n + U_n ? unitsU[s2 - S_n] : unitsZ[s2 - S_n - U_n]);
        else if (mt < tS) u = s2 < S_n ? unitsS[s2] : unitsZ[s2 - S_n];
        else { const int pos = s2 - 16 * tS; u = pos < U_n ? unitsU[pos] : unitsZ[(16 * tS - S_n) + (pos - U_n)]; }
        if (half == 0) rowA = u; else rowB = u;
      }
      const bool noclamp = !comb && mt < tS;
      const uint32_t ea = e1p[rowA * 4 + t], eb = e1p[rowB * 4 + t];
      const uint32_t f0b = (uint32_t)rowA * NIN + 2 * t + 7919u, f8b = (uint32_t)rowB * NIN + 2 * t + 7919u;
      for (int k = warp; k < NT; k += NWARP) {
        const int nt = tiles[k];
        int c[4] = {0, 0, 0, 0};
        mma_k16(c, ea, eb, xc[8 * nt]);
        const uint32_t f0 = f0b + 8u * nt, f8 = f8b + 8u * nt;
        int c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
        if (!noclamp) { c0 = clampi(c0, CLIP_LO, GCLIP); c1 = clampi(c1, CLIP_LO, GCLIP); c2 = clampi(c2, CLIP_LO, GCLIP); c3 = clampi(c3, CLIP_LO, GCLIP); }
        d0 += ((uint32_t)c0 ^ f0) + ((uint32_t)c2 ^ f8);
        d1 += ((uint32_t)c1 ^ (f0 + 1u)) + ((uint32_t)c3 ^ (f8 + 1u));
        if (DEBUG) {
          const size_t o = share * (size_t)(N1 + N2);
          const int p0 = 8 * nt + 2 * t;
          dbg[o + (size_t)rowA * NIN + p0] = (int16_t)c0; dbg[o + (size_t)rowA * NIN + p0 + 1] = (int16_t)c1;
          dbg[o + (size_t)rowB * NIN + p0] = (int16_t)c2; dbg[o + (size_t)rowB * NIN + p0 + 1] = (int16_t)c3;
        }
      }
    }
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) { d0 += __shfl_xor_sync(0xffffffffu, d0, o); d1 += __shfl_xor_sync(0xffffffffu, d1, o); }
    if (lane < 4) { atomicAdd(&S.digest[2 * t], d0); atomicAdd(&S.digest[2 * t + 1], d1); }
    if (tid < 8) {
      // all N1 entries of G1 would add f + 7919 to lane f mod 8 if they were zero: Dc is that sum, K the same sum over the computed
      // entries (rows R = 16 nM of sum of ids sumH, NT tiles of sum of ids sumNT), whose real contributions are already in d0, d1
      const uint32_t M = (uint32_t)(N1 / 8), R = 16u * (uint32_t)nM, NTn = (uint32_t)NT, lane7919 = (uint32_t)tid + 7919u;
      const uint32_t Dc = 4u * M * (M - 1u) + M * lane7919;
      const uint32_t K = NTn * (uint32_t)NIN * sumH + R * 8u * sumNT + R * NTn * lane7919;
      atomicAdd(&S.digest[tid], Dc - K);
    }
  }
#pragma unroll
  for (int o = 8; o < 32; o <<= 1) lane2 += __shfl_xor_sync(0xffffffffu, lane2, o);
  if (lane < 8) atomicAdd(&S.digest[lane], lane2);
  __syncthreads();

  if (tid < 8) ((uint32_t *)(out + (size_t)OUT_BYTES * share))[tid] = S.digest[tid];
  if (tid >= 8 && tid < 8 + NB) ((uint32_t *)(out + (size_t)OUT_BYTES * share))[16 + (tid - 8)] = S.idx[tid - 8];
#endif
}

// ---- host side
struct neb_share_tc {
  int sparse;
  uint4 *w1p;      // 8 warps x 25 k-steps x 32 lanes of 16 bytes: the B fragments of the forward product
  uint32_t *w2;    // 320 words, row-major 10 x 128 int8
  uint8_t *data, *labels;
  uint32_t n;
  Aux *aux;
  size_t aux_cap;
};

// w1p[((warp * 25 + ks) * 32 + lane)] = { W1[16 w + g][32 ks + 4t ..], W1[16 w + g][32 ks + 16 + 4t ..],
//                                         W1[16 w + 8 + g][32 ks + 4t ..], W1[16 w + 8 + g][32 ks + 16 + 4t ..] }, zero beyond pixel 783
static void pack_w1(const int8_t *w1, uint32_t *w1p) {
  auto word = [&](int h, int k0) {
    uint32_t v = 0;
    for (int j = 0; j < 4; j++) {
      const int k = k0 + j;
      const uint32_t byte = k < NIN ? (uint32_t)(uint8_t)w1[(size_t)h * NIN + k] : 0u;
      v |= byte << (8 * j);
    }
    return v;
  };
  for (int w = 0; w < NWARP; w++)
    for (int ks = 0; ks < KSTEPS; ks++)
      for (int lane = 0; lane < 32; lane++) {
        const int g = lane >> 2, t = lane & 3;
        uint32_t *o = w1p + (((size_t)w * KSTEPS + ks) * 32 + lane) * 4;
        o[0] = word(16 * w + g, 32 * ks + 4 * t);
        o[1] = word(16 * w + g, 32 * ks + 16 + 4 * t);
        o[2] = word(16 * w + 8 + g, 32 * ks + 4 * t);
        o[3] = word(16 * w + 8 + g, 32 * ks + 16 + 4 * t);
      }
}

extern "C" {

int neb_share_tc_supported(int device) {
  int major = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) return 0;
  return major >= 8;
}

cudaError_t neb_share_tc_set_model(neb_share_tc *k, const int8_t *w1, const int8_t *w2) {
  std::vector<uint32_t> host((size_t)NWARP * KSTEPS * 32 * 4);  // local to the call: nothing is shared between threads
  pack_w1(w1, host.data());
  cudaError_t e = cudaMemcpy(k->w1p, host.data(), host.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
  if (e != cudaSuccess) return e;
  return cudaMemcpy(k->w2, w2, NOUT * NH, cudaMemcpyHostToDevice);
}

cudaError_t neb_share_tc_open(neb_share_tc **out, int sparse, const int8_t *w1, const int8_t *w2, const uint8_t *data, const uint8_t *labels, uint32_t n) {
  *out = NULL;
  if (n == 0) return cudaErrorInvalidValue;
  neb_share_tc *k = (neb_share_tc *)calloc(1, sizeof *k);
  if (!k) return cudaErrorMemoryAllocation;
  k->n = n;
  k->sparse = sparse;
  cudaError_t e;
  if ((e = cudaMalloc(&k->w1p, sizeof(uint4) * NWARP * KSTEPS * 32)) != cudaSuccess) goto fail;
  if ((e = cudaMalloc(&k->w2, NOUT * NH)) != cudaSuccess) goto fail;
  if ((e = cudaMalloc(&k->data, (size_t)n * NIN)) != cudaSuccess) goto fail;
  if ((e = cudaMalloc(&k->labels, n)) != cudaSuccess) goto fail;
  if ((e = cudaMemcpy(k->data, data, (size_t)n * NIN, cudaMemcpyHostToDevice)) != cudaSuccess) goto fail;
  if ((e = cudaMemcpy(k->labels, labels, n, cudaMemcpyHostToDevice)) != cudaSuccess) goto fail;
  if ((e = neb_share_tc_set_model(k, w1, w2)) != cudaSuccess) goto fail;
  *out = k;
  return cudaSuccess;
fail:
  neb_share_tc_close(k);
  return e;
}

void neb_share_tc_close(neb_share_tc *k) {
  if (!k) return;
  cudaFree(k->w1p); cudaFree(k->w2); cudaFree(k->data); cudaFree(k->labels); cudaFree(k->aux);
  free(k);
}

}  // extern "C"

template <bool SPARSE>
static cudaError_t launch_impl(neb_share_tc *k, const uint8_t *d_in, int header, size_t count, uint8_t *d_out, int16_t *d_dbg) {
  if (count == 0 || count > 0x7FFFFFFFu) return cudaErrorInvalidValue;
  if (count > k->aux_cap) {  // the buffer holds one 128-byte record per attempt of the largest launch so far
    cudaFree(k->aux);
    k->aux = NULL;
    k->aux_cap = 0;
    cudaError_t e = cudaMalloc(&k->aux, count * sizeof(Aux));
    if (e != cudaSuccess) return e;
    k->aux_cap = count;
  }
  const unsigned gridd = (unsigned)((count + 127) / 128), grid = (unsigned)count;
  if (header) derive_kernel<true><<<gridd, 128>>>(d_in, k->n, k->labels, count, k->aux);
  else derive_kernel<false><<<gridd, 128>>>(d_in, k->n, k->labels, count, k->aux);
  if (d_dbg) k1_kernel<SPARSE, true><<<grid, NTHREADS>>>(k->aux, k->w1p, k->w2, k->data, d_out, d_dbg);
  else k1_kernel<SPARSE, false><<<grid, NTHREADS>>>(k->aux, k->w1p, k->w2, k->data, d_out, NULL);
  ticket_kernel<<<gridd, 128>>>(k->aux, count, d_out);
  return cudaGetLastError();
}

extern "C" cudaError_t neb_share_tc_launch(neb_share_tc *k, const uint8_t *d_in, int input_is_header, size_t count, uint8_t *d_out) {
  return k->sparse ? launch_impl<true>(k, d_in, input_is_header, count, d_out, NULL) : launch_impl<false>(k, d_in, input_is_header, count, d_out, NULL);
}

extern "C" cudaError_t neb_share_tc_launch_debug(neb_share_tc *k, const uint8_t *d_in, int input_is_header, size_t count, uint8_t *d_out, int16_t *d_gradients) {
  return k->sparse ? launch_impl<true>(k, d_in, input_is_header, count, d_out, d_gradients) : launch_impl<false>(k, d_in, input_is_header, count, d_out, d_gradients);
}
