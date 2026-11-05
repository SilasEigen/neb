/* AVX-512 VNNI implementations of one training share (spec/shares.md sections 5.1 to 5.4), integer only.
 *
 * Mining optimisation, not consensus: the node validates with pow/ref/shares.c. A miner that uses this code evaluates
 * its attempts here and the reference re-evaluates every nonce it reports (neb-engines, crates/neb-engines).
 *   dense : every pixel word and every (hidden unit, pixel chunk) of the gradient is computed.
 *   sparse: skips pixel words that are zero in a sample, hidden units whose propagated error is zero for the 16
 *           samples, and pixel chunks that are zero in the 16 samples; the digest of the skipped (zero) gradient
 *           entries is a constant per lane (an entry f of layer 1 adds f + 7919, taken mod 2^32) that is corrected
 *           in closed form.
 * Both must equal pow/ref/shares.c bit for bit (digest, ticket, sample indices, clipped gradients); the sample
 * indices use the reference code, the ticket uses the reference BLAKE3. All intermediate values are exact in
 * their types: the layer 1 pre-activations are at most 784 * 7 * 8 = 43,904 in absolute value in int32, the activations are in
 * -8..7, the G2 sums over 16 samples are at most 1024 in int16, and the digest lanes wrap mod 2^32 by design.
 *
 * Portable to MSVC and gcc (alignment, bit-scan and unroll macros); the interface is shares_avx512.h. Needs
 * AVX-512 F, BW, VL, DQ and VNNI at run time (checked by the caller). See pow/cpu/README.md.
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "blake3_min.h"
#include "shares.h"
#include "shares_avx512.h"

#define NIN NEB_NIN
#define NH NEB_NH
#define NOUT NEB_NOUT
#define NB NEB_NB

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define ALIGNED64(decl) __declspec(align(64)) decl
#define NEB_UNROLL8
static inline int ctz32(unsigned m) { unsigned long i; _BitScanForward(&i, m); return (int)i; }
static inline int ctz64(uint64_t m) { unsigned long i; _BitScanForward64(&i, m); return (int)i; }
#else
#define ALIGNED64(decl) decl __attribute__((aligned(64)))
#define NEB_UNROLL8 _Pragma("GCC unroll 8")
static inline int ctz32(unsigned m) { return __builtin_ctz(m); }
static inline int ctz64(uint64_t m) { return __builtin_ctzll(m); }
#endif

/* A flat dataset: n samples of NIN pixels (0..7) and n labels (0..9), in 64-byte aligned memory. */
struct neb_cpu_dataset {
  uint32_t n;
  uint8_t *px;
  uint8_t *lab;
};
typedef struct neb_cpu_dataset ds_t;

#define NW (NIN / 4)   /* 196 words of 4 pixels */
#define NCH (NIN / 16) /* 49 chunks of 16 pixels */

typedef struct {
  int8_t *w1p;      /* [k 196][j 8][l 16][p 4] = w1[(16j+l)*784 + 4k + p] */
  int16_t *w2pp;    /* [hp 64][o 16][t 2] = w2[o*128 + 2hp + t], 0 for o >= 10 */
  int16_t *w2op;    /* [op 5][h 128][t 2] = w2[(2op+t)*128 + h] */
  uint32_t dconst1[8]; /* sum over f < NEB_N1 with f & 7 == j of (f + 7919) */
} prep_t;

typedef struct {
  ALIGNED64(int32_t z1[NB][NH]);
  ALIGNED64(int16_t a1[NB][NH]);
  ALIGNED64(int16_t e2s[NB][16]);
  ALIGNED64(int8_t e1[NB][NH]);
  ALIGNED64(uint8_t e1t[NH][16]);
  ALIGNED64(uint8_t xt[4][NIN][4]);
  uint16_t mk[NB][8];
  uint16_t mw[NB][13];
} scr_t;

static prep_t *cpu_prepare(const int8_t *w1, const int8_t *w2) {
  prep_t *P = (prep_t *)calloc(1, sizeof *P);
  if (!P) return NULL;
  P->w1p = (int8_t *)_mm_malloc(NW * 512, 64);
  P->w2pp = (int16_t *)_mm_malloc(64 * 16 * 2 * 2, 64);
  P->w2op = (int16_t *)_mm_malloc(5 * NH * 2 * 2, 64);
  for (int k = 0; k < NW; k++)
    for (int j = 0; j < 8; j++)
      for (int l = 0; l < 16; l++)
        for (int p = 0; p < 4; p++) P->w1p[((k * 8 + j) * 16 + l) * 4 + p] = w1[(16 * j + l) * NIN + 4 * k + p];
  for (int hp = 0; hp < 64; hp++)
    for (int o = 0; o < 16; o++)
      for (int t = 0; t < 2; t++) P->w2pp[(hp * 16 + o) * 2 + t] = o < NOUT ? w2[o * NH + 2 * hp + t] : 0;
  for (int op = 0; op < 5; op++)
    for (int h = 0; h < NH; h++)
      for (int t = 0; t < 2; t++) P->w2op[(op * NH + h) * 2 + t] = w2[(2 * op + t) * NH + h];
  for (uint32_t f = 0; f < NEB_N1; f++) P->dconst1[f & 7] += f + 7919u;
  return P;
}
static void cpu_release(void *p) {
  prep_t *P = (prep_t *)p;
  if (!P) return;
  _mm_free(P->w1p);
  _mm_free(P->w2pp);
  _mm_free(P->w2op);
  free(P);
}
static void *cpu_scratch(void) {
  scr_t *S = (scr_t *)_mm_malloc(sizeof *S, 64);
  if (S) memset(S, 0, sizeof *S);
  return S;
}

static inline __m512i clampv(__m512i v, int lo, int hi) { return _mm512_min_epi32(_mm512_max_epi32(v, _mm512_set1_epi32(lo)), _mm512_set1_epi32(hi)); }

/* layer 1 -> activations, masks; layer 2; output error; propagated error; G2 and its digest lanes */
static void small_part(const prep_t *P, scr_t *S, const uint8_t labels[NB], uint32_t D2[8], int32_t *g2, uint64_t act[2]) {
  const __m512i half = _mm512_set1_epi32(1 << (NEB_S1 - 1));
  for (int b = 0; b < NB; b++)
    for (int j = 0; j < 8; j++) {
      const __m512i y = _mm512_srai_epi32(_mm512_add_epi32(_mm512_load_si512(&S->z1[b][16 * j]), half), NEB_S1);
      const __m512i a = clampv(y, -8, 7);
      S->mk[b][j] = _mm512_cmpeq_epi32_mask(y, a);
      _mm256_storeu_si256((__m256i *)&S->a1[b][16 * j], _mm512_cvtepi32_epi16(a));
    }
  for (int b = 0; b < NB; b++) {
    const int32_t *ap = (const int32_t *)S->a1[b];
    __m512i z0 = _mm512_setzero_si512(), z1 = z0, z2 = z0, z3 = z0;
    for (int hp = 0; hp < 64; hp += 4) {
      z0 = _mm512_add_epi32(z0, _mm512_madd_epi16(_mm512_set1_epi32(ap[hp + 0]), _mm512_load_si512(P->w2pp + (hp + 0) * 32)));
      z1 = _mm512_add_epi32(z1, _mm512_madd_epi16(_mm512_set1_epi32(ap[hp + 1]), _mm512_load_si512(P->w2pp + (hp + 1) * 32)));
      z2 = _mm512_add_epi32(z2, _mm512_madd_epi16(_mm512_set1_epi32(ap[hp + 2]), _mm512_load_si512(P->w2pp + (hp + 2) * 32)));
      z3 = _mm512_add_epi32(z3, _mm512_madd_epi16(_mm512_set1_epi32(ap[hp + 3]), _mm512_load_si512(P->w2pp + (hp + 3) * 32)));
    }
    const __m512i z = _mm512_add_epi32(_mm512_add_epi32(z0, z1), _mm512_add_epi32(z2, z3));
    const __m512i y2 = clampv(_mm512_srai_epi32(_mm512_add_epi32(z, _mm512_set1_epi32(1 << (NEB_S2 - 1))), NEB_S2), -8, 7);
    const __m512i t = _mm512_mask_blend_epi32((__mmask16)(1u << labels[b]), _mm512_set1_epi32(NEB_TNEG), _mm512_set1_epi32(NEB_TPOS));
    const __m512i e = _mm512_maskz_mov_epi32(0x3FF, clampv(_mm512_sub_epi32(y2, t), -8, 7));
    _mm256_storeu_si256((__m256i *)S->e2s[b], _mm512_cvtepi32_epi16(e));
  }
  __m512i orv0 = _mm512_setzero_si512(), orv1 = orv0;
  for (int b = 0; b < NB; b++) {
    const int32_t *ep = (const int32_t *)S->e2s[b]; /* pairs (E2[2op], E2[2op+1]) as int16 */
    for (int j = 0; j < 8; j++) {
      __m512i m = _mm512_setzero_si512();
      for (int op = 0; op < 5; op++)
        m = _mm512_add_epi32(m, _mm512_madd_epi16(_mm512_set1_epi32(ep[op]), _mm512_load_si512(P->w2op + (op * NH + 16 * j) * 2)));
      const __m512i e = _mm512_maskz_mov_epi32(S->mk[b][j], clampv(_mm512_srai_epi32(_mm512_add_epi32(m, _mm512_set1_epi32(1 << (NEB_SB - 1))), NEB_SB), -8, 7));
      _mm_storeu_si128((__m128i *)&S->e1[b][16 * j], _mm512_cvtepi32_epi8(e));
    }
    orv0 = _mm512_or_si512(orv0, _mm512_loadu_si512(&S->e1[b][0]));
    orv1 = _mm512_or_si512(orv1, _mm512_loadu_si512(&S->e1[b][64]));
  }
  act[0] = _mm512_test_epi8_mask(orv0, orv0);
  act[1] = _mm512_test_epi8_mask(orv1, orv1);
  for (int h = 0; h < NH; h++)
    for (int b = 0; b < NB; b++) S->e1t[h][b] = (uint8_t)S->e1[b][h];
  /* G2[o][h] = sum over b of E2[b][o] * a1[b][h] (|sum| <= 1024, exact in int16), clip, digest lane constant 7919 * 2 */
  __m512i accD = _mm512_setzero_si512();
  const __m512i iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  for (int q = 0; q < 4; q++) {
    __m512i acc[NOUT];
    for (int o = 0; o < NOUT; o++) acc[o] = _mm512_setzero_si512();
    for (int b = 0; b < NB; b++) {
      const __m512i a = _mm512_loadu_si512(&S->a1[b][32 * q]);
      for (int o = 0; o < NOUT; o++) acc[o] = _mm512_add_epi16(acc[o], _mm512_mullo_epi16(_mm512_set1_epi16(S->e2s[b][o]), a));
    }
    for (int o = 0; o < NOUT; o++) {
      const __m512i c16 = _mm512_min_epi16(_mm512_max_epi16(acc[o], _mm512_set1_epi16(-NEB_CLIP)), _mm512_set1_epi16(NEB_CLIP));
      for (int hh = 0; hh < 2; hh++) {
        const __m512i c = _mm512_cvtepi16_epi32(hh ? _mm512_extracti64x4_epi64(c16, 1) : _mm512_castsi512_si256(c16));
        const __m512i f = _mm512_add_epi32(_mm512_set1_epi32(o * NH + 32 * q + 16 * hh + 7919 * 2), iota);
        accD = _mm512_add_epi32(accD, _mm512_xor_si512(c, f));
        if (g2) _mm512_storeu_si512(g2 + o * NH + 32 * q + 16 * hh, c);
      }
    }
  }
  uint32_t lanes[16]; _mm512_storeu_si512(lanes, accD);
  for (int j = 0; j < 8; j++) D2[j] = lanes[j] + lanes[j + 8];
}

static inline void build_xt(scr_t *S, const uint8_t *xp[NB], int c) {
  for (int g = 0; g < 4; g++) {
    const __m128i a = _mm_loadu_si128((const __m128i *)(xp[4 * g + 0] + 16 * c)), b = _mm_loadu_si128((const __m128i *)(xp[4 * g + 1] + 16 * c));
    const __m128i cc = _mm_loadu_si128((const __m128i *)(xp[4 * g + 2] + 16 * c)), dd = _mm_loadu_si128((const __m128i *)(xp[4 * g + 3] + 16 * c));
    const __m128i t0 = _mm_unpacklo_epi8(a, b), t1 = _mm_unpacklo_epi8(cc, dd), t2 = _mm_unpackhi_epi8(a, b), t3 = _mm_unpackhi_epi8(cc, dd);
    __m128i *o = (__m128i *)&S->xt[g][16 * c][0];
    _mm_storeu_si128(o + 0, _mm_unpacklo_epi16(t0, t1));
    _mm_storeu_si128(o + 1, _mm_unpackhi_epi16(t0, t1));
    _mm_storeu_si128(o + 2, _mm_unpacklo_epi16(t2, t3));
    _mm_storeu_si128(o + 3, _mm_unpackhi_epi16(t2, t3));
  }
}

static inline void finish(const uint32_t D[8], const uint8_t key[32], uint8_t digest[32], uint8_t ticket[32]) {
  uint8_t in[64];
  for (int d = 0; d < 8; d++) { digest[4 * d] = (uint8_t)D[d]; digest[4 * d + 1] = (uint8_t)(D[d] >> 8); digest[4 * d + 2] = (uint8_t)(D[d] >> 16); digest[4 * d + 3] = (uint8_t)(D[d] >> 24); }
  memcpy(in, digest, 32); memcpy(in + 32, key, 32);
  blake3_hash(in, 64, ticket);
}

static inline int32_t ld32(const uint8_t *p) { int32_t v; memcpy(&v, p, 4); return v; }

static inline void eval_cpu(const void *prep, void *scr, const ds_t *d, const uint8_t key[32], uint8_t digest[32], uint8_t ticket[32],
                            uint32_t idx[NB], int32_t *g1, int32_t *g2, const int sparse) {
  const prep_t *P = (const prep_t *)prep; scr_t *S = (scr_t *)scr;
  const uint8_t *xp[NB]; uint8_t lab[NB];
  neb_share_indices(key, d->n, idx);
  for (int b = 0; b < NB; b++) { xp[b] = d->px + (size_t)idx[b] * NIN; lab[b] = d->lab[idx[b]]; }

  /* ---- layer 1 */
  if (!sparse) {
    for (int b = 0; b < NB; b += 2) {
      __m512i a0[8], a1[8];
      for (int j = 0; j < 8; j++) a0[j] = a1[j] = _mm512_setzero_si512();
      for (int k = 0; k < NW; k++) {
        const __m512i x0 = _mm512_set1_epi32(ld32(xp[b] + 4 * k)), x1 = _mm512_set1_epi32(ld32(xp[b + 1] + 4 * k));
NEB_UNROLL8
        for (int j = 0; j < 8; j++) {
          const __m512i w = _mm512_load_si512(P->w1p + k * 512 + j * 64);
          a0[j] = _mm512_dpbusd_epi32(a0[j], x0, w);
          a1[j] = _mm512_dpbusd_epi32(a1[j], x1, w);
        }
      }
      for (int j = 0; j < 8; j++) { _mm512_store_si512(&S->z1[b][16 * j], a0[j]); _mm512_store_si512(&S->z1[b + 1][16 * j], a1[j]); }
    }
  } else {
    for (int b = 0; b < NB; b++) {
      for (int q = 0; q < 12; q++) { const __m512i v = _mm512_loadu_si512(xp[b] + 64 * q); S->mw[b][q] = _mm512_test_epi32_mask(v, v); }
      { const __m128i v = _mm_loadu_si128((const __m128i *)(xp[b] + 768)); S->mw[b][12] = _mm_test_epi32_mask(v, v) & 0xF; }
      __m512i acc[8];
      for (int j = 0; j < 8; j++) acc[j] = _mm512_setzero_si512();
      for (int q = 0; q < 13; q++) {
        unsigned m = S->mw[b][q];
        while (m) {
          const int k = q * 16 + ctz32(m); m &= m - 1;
#ifdef NEB_NEGATIVE_CONTROL_SPARSE_WORD
          if (k == 97) continue;
#endif
          const __m512i x = _mm512_set1_epi32(ld32(xp[b] + 4 * k));
NEB_UNROLL8
          for (int j = 0; j < 8; j++) acc[j] = _mm512_dpbusd_epi32(acc[j], x, _mm512_load_si512(P->w1p + k * 512 + j * 64));
        }
      }
      for (int j = 0; j < 8; j++) _mm512_store_si512(&S->z1[b][16 * j], acc[j]);
    }
  }

  uint32_t D2[8]; uint64_t act[2];
  small_part(P, S, lab, D2, g2, act);

  /* ---- G1 = sum over b of E1[b][h] X[b][i], clip, digest lanes with constant 7919 * 1 */
  uint32_t D1[8];
  const __m512i iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  #ifdef NEB_NEGATIVE_CONTROL_CLIP
  const __m512i lo = _mm512_set1_epi32(-NEB_CLIP + 1), hi = _mm512_set1_epi32(NEB_CLIP);
#else
  const __m512i lo = _mm512_set1_epi32(-NEB_CLIP), hi = _mm512_set1_epi32(NEB_CLIP);
#endif
  if (g1) memset(g1, 0, sizeof(int32_t) * NEB_N1);
  __m512i accD = _mm512_setzero_si512();
  if (!sparse) {
    for (int c = 0; c < NCH; c++) build_xt(S, xp, c);
    for (int c = 0; c < NCH; c++) {
      const __m512i X0 = _mm512_loadu_si512(&S->xt[0][16 * c][0]), X1 = _mm512_loadu_si512(&S->xt[1][16 * c][0]);
      const __m512i X2 = _mm512_loadu_si512(&S->xt[2][16 * c][0]), X3 = _mm512_loadu_si512(&S->xt[3][16 * c][0]);
      __m512i kv = _mm512_add_epi32(_mm512_set1_epi32(16 * c + 7919), iota);
      const __m512i step = _mm512_set1_epi32(NIN);
      __m512i acc2 = _mm512_setzero_si512();
      for (int h = 0; h < NH; h += 2) {
        const int32_t *e = (const int32_t *)S->e1t[h];
        __m512i ga = _mm512_dpbusd_epi32(_mm512_setzero_si512(), X0, _mm512_set1_epi32(e[0]));
        __m512i gb = _mm512_dpbusd_epi32(_mm512_setzero_si512(), X0, _mm512_set1_epi32(e[4]));
        ga = _mm512_dpbusd_epi32(ga, X1, _mm512_set1_epi32(e[1])); gb = _mm512_dpbusd_epi32(gb, X1, _mm512_set1_epi32(e[5]));
        ga = _mm512_dpbusd_epi32(ga, X2, _mm512_set1_epi32(e[2])); gb = _mm512_dpbusd_epi32(gb, X2, _mm512_set1_epi32(e[6]));
        ga = _mm512_dpbusd_epi32(ga, X3, _mm512_set1_epi32(e[3])); gb = _mm512_dpbusd_epi32(gb, X3, _mm512_set1_epi32(e[7]));
        ga = _mm512_min_epi32(_mm512_max_epi32(ga, lo), hi); gb = _mm512_min_epi32(_mm512_max_epi32(gb, lo), hi);
        const __m512i kv2 = _mm512_add_epi32(kv, step);
        accD = _mm512_add_epi32(accD, _mm512_xor_si512(ga, kv));
        acc2 = _mm512_add_epi32(acc2, _mm512_xor_si512(gb, kv2));
        if (g1) { _mm512_storeu_si512(g1 + (size_t)h * NIN + 16 * c, ga); _mm512_storeu_si512(g1 + (size_t)(h + 1) * NIN + 16 * c, gb); }
        kv = _mm512_add_epi32(kv2, step);
      }
      accD = _mm512_add_epi32(accD, acc2);
    }
    uint32_t lanes[16]; _mm512_storeu_si512(lanes, accD);
    for (int j = 0; j < 8; j++) D1[j] = lanes[j] + lanes[j + 8];
  } else {
    /* active hidden units A and nonzero chunks C */
    uint8_t alist[NH]; int na = 0; uint32_t sumh = 0;
    for (int w = 0; w < 2; w++) { uint64_t m = act[w]; while (m) { const int h = 64 * w + ctz64(m); m &= m - 1; alist[na++] = (uint8_t)h; sumh += (uint32_t)h; } }
    uint8_t clist[NCH]; int nc = 0; uint32_t sumc = 0;
    for (int c = 0; c < NCH; c++) {
      const int q = (4 * c) >> 4, sh = (4 * c) & 15;
      unsigned any = 0;
      for (int b = 0; b < NB; b++) any |= (S->mw[b][q] >> sh) & 0xF;
      if (any) { clist[nc++] = (uint8_t)c; sumc += (uint32_t)c; build_xt(S, xp, c); }
    }
    for (int ci = 0; ci < nc; ci++) {
      const int c = clist[ci];
      const __m512i X0 = _mm512_loadu_si512(&S->xt[0][16 * c][0]), X1 = _mm512_loadu_si512(&S->xt[1][16 * c][0]);
      const __m512i X2 = _mm512_loadu_si512(&S->xt[2][16 * c][0]), X3 = _mm512_loadu_si512(&S->xt[3][16 * c][0]);
      const __m512i kc = _mm512_add_epi32(_mm512_set1_epi32(16 * c + 7919), iota);
      for (int ai = 0; ai < na; ai++) {
        const int h = alist[ai];
        const int32_t *e = (const int32_t *)S->e1t[h];
        __m512i g = _mm512_dpbusd_epi32(_mm512_setzero_si512(), X0, _mm512_set1_epi32(e[0]));
        g = _mm512_dpbusd_epi32(g, X1, _mm512_set1_epi32(e[1]));
        g = _mm512_dpbusd_epi32(g, X2, _mm512_set1_epi32(e[2]));
        g = _mm512_dpbusd_epi32(g, X3, _mm512_set1_epi32(e[3]));
        g = _mm512_min_epi32(_mm512_max_epi32(g, lo), hi);
        accD = _mm512_add_epi32(accD, _mm512_xor_si512(g, _mm512_add_epi32(kc, _mm512_set1_epi32(h * NIN))));
        if (g1) _mm512_storeu_si512(g1 + (size_t)h * NIN + 16 * c, g);
      }
    }
    uint32_t lanes[16]; _mm512_storeu_si512(lanes, accD);
    /* K_l = sum over processed (h, c) of (h * 784 + 16 c + l + 7919) */
    const uint32_t nA = (uint32_t)na, nC = (uint32_t)nc;
    for (int j = 0; j < 8; j++) {
      uint32_t kk = 0;
      for (int l = j; l < 16; l += 8) kk += nC * (uint32_t)NIN * sumh + nA * 16u * sumc + nA * nC * (uint32_t)(l + 7919);
      D1[j] = P->dconst1[j] + lanes[j] + lanes[j + 8] - kk;
    }
  }
  uint32_t Dt[8];
  for (int j = 0; j < 8; j++) Dt[j] = D1[j] + D2[j];
  finish(Dt, key, digest, ticket);
}

/* ------------------------------------------------------------------------------------------------ public interface */

neb_cpu_dataset *neb_cpu_dataset_new(const uint8_t *pixels, const uint8_t *labels, uint32_t n) {
  if (n == 0) return NULL;
  ds_t *d = (ds_t *)calloc(1, sizeof *d);
  if (!d) return NULL;
  d->n = n;
  d->px = (uint8_t *)_mm_malloc((size_t)n * NIN + 64, 64);
  d->lab = (uint8_t *)malloc(n);
  if (!d->px || !d->lab) {
    neb_cpu_dataset_free(d);
    return NULL;
  }
  memcpy(d->px, pixels, (size_t)n * NIN);
  memset(d->px + (size_t)n * NIN, 0, 64);
  memcpy(d->lab, labels, n);
  return d;
}

void neb_cpu_dataset_free(neb_cpu_dataset *d) {
  if (!d) return;
  if (d->px) _mm_free(d->px);
  free(d->lab);
  free(d);
}

neb_cpu_model *neb_cpu_model_new(const int8_t *w1, const int8_t *w2) { return (neb_cpu_model *)cpu_prepare(w1, w2); }

void neb_cpu_model_free(neb_cpu_model *m) { cpu_release(m); }

neb_cpu_scratch *neb_cpu_scratch_new(void) { return (neb_cpu_scratch *)cpu_scratch(); }

void neb_cpu_scratch_free(neb_cpu_scratch *s) {
  if (s) _mm_free(s);
}

void neb_cpu_share_eval(const neb_cpu_model *m, neb_cpu_scratch *s, const neb_cpu_dataset *d, const uint8_t key[32], int sparse, uint8_t digest[32],
                        uint8_t ticket[32], uint32_t idx[NEB_NB], int32_t *g1, int32_t *g2) {
  eval_cpu(m, s, d, key, digest, ticket, idx, g1, g2, sparse);
}
