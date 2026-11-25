/* Reference implementation of spec/nebhash-v3.md: anti-shortcut MoE frozen INT8 backbones with on-chain integer
 * LoRA adapters, residual blocks, full-dimension input + bottleneck dropout, and exact int32 trace commitment.
 * Scalar, portable, exact integer arithmetic only, zero signed-overflow UB. */
#include "nebhash3.h"
#include <stdlib.h>
#include <string.h>
#include "blake3_min.h"
#include "chacha8.h"

static const neb3_arch LORA_TINY = {
    NEB3_LORA_TINY, 64, 16, 3, 16, NEB3_NOUT,
    5, 5, 9, 4, 9, 5, 2, 14, 12,
    (size_t)3 * 2 * 64 * 16 + (size_t)NEB3_NOUT * 64,
    (size_t)16 * 64 * 64};

static const neb3_arch QWEN3_8B_LORA = {
    NEB3_QWEN3_8B_LORA, 1024, 16, 6, 10752, NEB3_NOUT,
    7, 7, 11, 4, 11, 7, 2, 16, 14,
    (size_t)6 * 2 * 1024 * 16 + (size_t)NEB3_NOUT * 1024,
    (size_t)10752 * 1024 * 1024};

const neb3_arch *neb3_arch_get(int id) {
  if (id == NEB3_LORA_TINY) return &LORA_TINY;
  if (id == NEB3_QWEN3_8B_LORA) return &QWEN3_8B_LORA;
  return NULL;
}

static int clipi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static int64_t clip64(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

static int32_t rshift(int32_t x, int n) { return n == 0 ? x : (x + ((int32_t)1 << (n - 1))) >> n; }

static void put_le32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static uint32_t get_le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline void trace_mix(uint32_t lanes[8], uint32_t *tc, uint32_t val) {
  const uint32_t c = *tc;
  lanes[c & 7u] += val ^ (c * 0x9E3779B9u + 0x7F4A7C15u);
  *tc = c + 1u;
}

void neb3_share_indices(const uint8_t key[32], uint32_t n, uint32_t idx[NEB3_NB]) {
  uint8_t u[4 * NEB3_NB];
  chacha8_stream(key, u, sizeof u);
  for (int j = 0; j < NEB3_NB; j++) {
    const uint64_t r = get_le32(u + 4 * j);
    idx[j] = (uint32_t)((r * (uint64_t)n) >> 32);
  }
}

int neb3_model_init(int arch, const uint8_t seed[32], neb3_model *m) {
  const neb3_arch *A = neb3_arch_get(arch);
  if (!A || !m) return -1;
  uint8_t in[14 + 32], key[32];
  memcpy(in, "neb-model-init", 14);
  memcpy(in + 14, seed, 32);
  blake3_hash(in, sizeof in, key);
  uint8_t *s = (uint8_t *)malloc(A->nweights);
  int8_t *w = (int8_t *)malloc(A->nweights);
  int32_t *r = (int32_t *)calloc(A->nweights, sizeof(int32_t));
  if (!s || !w || !r) {
    free(s); free(w); free(r);
    return -1;
  }
  chacha8_stream(key, s, A->nweights);
  for (size_t k = 0; k < A->nweights; k++) {
    const int b = s[k];
    w[k] = (int8_t)((b & 3) + ((b >> 2) & 3) + ((b >> 4) & 3) + ((b >> 6) & 3) - 6);
  }
  free(s);
  m->arch = arch;
  m->nweights = A->nweights;
  m->w = w;
  m->r = r;
  return 0;
}

void neb3_model_free(neb3_model *m) {
  if (!m) return;
  free(m->w);
  free(m->r);
  m->w = NULL;
  m->r = NULL;
  m->nweights = 0;
  m->arch = 0;
}

int neb3_base_hash(const neb3_base *base, uint8_t out_hash[32]) {
  static const char tb[] = "neb-base-weights";
  const size_t lb = sizeof tb - 1;
  if (!base || !base->w_base || base->nweights == 0) return -1;
  uint8_t *buf = (uint8_t *)malloc(lb + base->nweights);
  if (!buf) return -1;
  memcpy(buf, tb, lb);
  memcpy(buf + lb, base->w_base, base->nweights);
  blake3_hash(buf, lb + base->nweights, out_hash);
  free(buf);
  return 0;
}

int neb3_share_eval(const neb3_model *m, const neb3_base *base, const neb3_dataset *ds, const uint8_t key[32],
                    int32_t *g, uint8_t digest[32], uint8_t ticket[32], uint32_t idx_out[NEB3_NB],
                    uint32_t experts_out[NEB3_MAX_BLOCKS], uint32_t *nclip) {
  if (!m || !base || !ds || !g || m->arch != base->arch) return -1;
  const neb3_arch *A = neb3_arch_get(m->arch);
  if (!A || m->nweights != A->nweights || base->nweights != A->base_weights || ds->n == 0) return -1;

  const int D = A->d, R = A->rank, L = A->nblocks, E = A->nexperts, NOUT = A->nout;
  const size_t block_ab = (size_t)R * (size_t)D;
  const size_t expert_sz = (size_t)D * (size_t)D;
  const size_t batch_d = (size_t)NEB3_NB * (size_t)D;

  /* Key-derived dropout stream: first 2-bit pairs for input features, then 1-bit masks for bottleneck */
  uint8_t drop_in[13 + 32], drop_key[32];
  memcpy(drop_in, "neb-lora-drop", 13);
  memcpy(drop_in + 13, key, 32);
  blake3_hash(drop_in, sizeof drop_in, drop_key);

  const size_t in_bytes = ((size_t)NEB3_NB * (size_t)L * (size_t)D * 2u + 7u) / 8u;
  const size_t u_bytes = ((size_t)NEB3_NB * (size_t)L * (size_t)R + 7u) / 8u;
  uint8_t *drop_buf = (uint8_t *)malloc(in_bytes + u_bytes);

  uint8_t *x_raw = (uint8_t *)malloc((size_t)D);
  int8_t *xs = (int8_t *)malloc((size_t)(L + 1) * batch_d);
  int8_t *xm_all = (int8_t *)malloc((size_t)L * batch_d);
  int32_t *y_out = (int32_t *)malloc((size_t)L * batch_d * sizeof(int32_t));
  int32_t *y_a = (int32_t *)malloc((size_t)L * (size_t)NEB3_NB * (size_t)R * sizeof(int32_t));
  int8_t *u_buf = (int8_t *)malloc((size_t)L * (size_t)NEB3_NB * (size_t)R);
  uint8_t *route_buf = (uint8_t *)malloc(13u + 32u + 1u + batch_d);
  int32_t *e_cur = (int32_t *)malloc((size_t)D * sizeof(int32_t));
  int32_t *e_next = (int32_t *)malloc((size_t)D * sizeof(int32_t));
  int32_t *d_vec = (int32_t *)malloc((size_t)D * sizeof(int32_t));
  int32_t *d_u = (int32_t *)malloc((size_t)R * sizeof(int32_t));

  if (!drop_buf || !x_raw || !xs || !xm_all || !y_out || !y_a || !u_buf || !route_buf ||
      !e_cur || !e_next || !d_vec || !d_u) {
    free(drop_buf); free(x_raw); free(xs); free(xm_all); free(y_out); free(y_a); free(u_buf);
    free(route_buf); free(e_cur); free(e_next); free(d_vec); free(d_u);
    return -1;
  }

  chacha8_stream(drop_key, drop_buf, in_bytes + u_bytes);
  const uint8_t *in_stream = drop_buf;
  const uint8_t *u_stream = drop_buf + in_bytes;

  uint32_t idx[NEB3_NB];
  neb3_share_indices(key, ds->n, idx);
  if (idx_out) memcpy(idx_out, idx, sizeof idx);
  memset(g, 0, A->nweights * sizeof(int32_t));

  /* Gather and center initial batch x_0 in -4..3 */
  for (int b = 0; b < NEB3_NB; b++) {
    const int label = ds->get(ds->ctx, idx[b], x_raw);
    (void)label;
    int8_t *x0b = xs + (size_t)b * (size_t)D;
    for (int j = 0; j < D; j++) x0b[j] = (int8_t)((int)x_raw[j] - 4);
  }
  /* Store labels once */
  int labels[NEB3_NB];
  for (int b = 0; b < NEB3_NB; b++) labels[b] = ds->get(ds->ctx, idx[b], x_raw);

  uint32_t t_lanes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint32_t tc = 0;
  uint32_t routed_experts[NEB3_MAX_BLOCKS];

  /* Layer-by-layer forward pass with sequential data-dependent MoE routing */
  for (int l = 0; l < L; l++) {
    const int8_t *xl_batch = xs + (size_t)l * batch_d;
    int8_t *xnext_batch = xs + (size_t)(l + 1) * batch_d;
    int8_t *xm_batch = xm_all + (size_t)l * batch_d;

    uint8_t rh[32];
    memcpy(route_buf, "neb-moe-route", 13);
    memcpy(route_buf + 13, key, 32);
    route_buf[45] = (uint8_t)l;
    if (l == 0) {
      blake3_hash(route_buf, 46, rh);
    } else {
      memcpy(route_buf + 46, xl_batch, batch_d);
      blake3_hash(route_buf, 46u + batch_d, rh);
    }
    const uint32_t e_l = (uint32_t)(((uint64_t)get_le32(rh) * (uint64_t)E) >> 32);
    routed_experts[l] = e_l;
    if (experts_out) experts_out[l] = e_l;
    trace_mix(t_lanes, &tc, e_l);

    const int8_t *Wb = base->w_base + (size_t)e_l * expert_sz;
    const int8_t *Al = m->w + (size_t)l * 2u * block_ab;
    const int8_t *Bl = Al + block_ab;

    for (int b = 0; b < NEB3_NB; b++) {
      const int8_t *xl = xl_batch + (size_t)b * (size_t)D;
      int8_t *xm = xm_batch + (size_t)b * (size_t)D;
      int8_t *xnext = xnext_batch + (size_t)b * (size_t)D;
      int32_t *yal = y_a + ((size_t)l * (size_t)NEB3_NB + (size_t)b) * (size_t)R;
      int8_t *ul = u_buf + ((size_t)l * (size_t)NEB3_NB + (size_t)b) * (size_t)R;
      int32_t *yol = y_out + ((size_t)l * (size_t)NEB3_NB + (size_t)b) * (size_t)D;

      for (int j = 0; j < D; j++) {
        const size_t idx2 = ((size_t)b * (size_t)L + (size_t)l) * (size_t)D + (size_t)j;
        const int pair = (in_stream[idx2 >> 2] >> ((idx2 & 3u) * 2u)) & 3;
        xm[j] = (pair != 0) ? xl[j] : 0;
      }

      for (int r = 0; r < R; r++) {
        int32_t za = 0;
        const int8_t *arow = Al + (size_t)r * (size_t)D;
        for (int j = 0; j < D; j++) za += (int32_t)arow[j] * xm[j];
        trace_mix(t_lanes, &tc, (uint32_t)za);
        const int32_t ya = rshift(za, A->s_a);
        yal[r] = ya;
        const size_t bit = ((size_t)b * (size_t)L + (size_t)l) * (size_t)R + (size_t)r;
        const int mask = (u_stream[bit >> 3] >> (bit & 7u)) & 1;
        ul[r] = (int8_t)(clipi(ya, -8, 7) * mask);
      }

      for (int o = 0; o < D; o++) {
        int32_t zb = 0, zl = 0;
        const int8_t *wbrow = Wb + (size_t)o * (size_t)D;
        const int8_t *brow = Bl + (size_t)o * (size_t)R;
        for (int j = 0; j < D; j++) zb += (int32_t)wbrow[j] * xm[j];
        trace_mix(t_lanes, &tc, (uint32_t)zb);
        for (int r = 0; r < R; r++) zl += (int32_t)brow[r] * ul[r];
        const int32_t yo = (int32_t)xl[o] + rshift(zb + zl * ((int32_t)1 << A->s_lora), A->s_out);
        yol[o] = yo;
        xnext[o] = (int8_t)clipi(yo, -8, 7);
      }
    }
  }

  const size_t h_off = (size_t)L * 2u * block_ab;
  const int8_t *H = m->w + h_off;
  int32_t *G_H = g + h_off;
  const int8_t *xL_batch = xs + (size_t)L * batch_d;

  /* Output head forward (with trace commitment) + full backward pass per sample */
  for (int b = 0; b < NEB3_NB; b++) {
    const int8_t *xL = xL_batch + (size_t)b * (size_t)D;
    int32_t e_h[NEB3_NOUT];
    for (int o = 0; o < NOUT; o++) {
      int32_t zh = 0;
      const int8_t *hrow = H + (size_t)o * (size_t)D;
      for (int j = 0; j < D; j++) zh += (int32_t)hrow[j] * xL[j];
      trace_mix(t_lanes, &tc, (uint32_t)zh);
      const int32_t ah = clipi(rshift(zh, A->s_head), -8, 7);
      const int t = (o == labels[b]) ? NEB_TPOS : NEB_TNEG;
      e_h[o] = clipi(ah - t, -8, 7);
    }

    memset(e_cur, 0, (size_t)D * sizeof(int32_t));
    for (int o = 0; o < NOUT; o++) {
      const int32_t eo = e_h[o];
      if (eo == 0) continue;
      int32_t *ghrow = G_H + (size_t)o * (size_t)D;
      const int8_t *hrow = H + (size_t)o * (size_t)D;
      for (int j = 0; j < D; j++) {
        ghrow[j] += eo * xL[j];
        e_cur[j] += eo * hrow[j];
      }
    }
    for (int j = 0; j < D; j++) e_cur[j] = clipi(rshift(e_cur[j], A->sb_head), -8, 7);

    for (int l = L - 1; l >= 0; l--) {
      const uint32_t e_l = routed_experts[l];
      const int8_t *Wb = base->w_base + (size_t)e_l * expert_sz;
      const int8_t *Al = m->w + (size_t)l * 2u * block_ab;
      const int8_t *Bl = Al + block_ab;
      int32_t *G_Al = g + (size_t)l * 2u * block_ab;
      int32_t *G_Bl = G_Al + block_ab;
      const int8_t *xm = xm_all + ((size_t)l * (size_t)NEB3_NB + (size_t)b) * (size_t)D;
      const int32_t *yal = y_a + ((size_t)l * (size_t)NEB3_NB + (size_t)b) * (size_t)R;
      const int8_t *ul = u_buf + ((size_t)l * (size_t)NEB3_NB + (size_t)b) * (size_t)R;
      const int32_t *yol = y_out + ((size_t)l * (size_t)NEB3_NB + (size_t)b) * (size_t)D;

      for (int o = 0; o < D; o++) {
        d_vec[o] = (yol[o] < -8 || yol[o] > 7) ? 0 : e_cur[o];
      }

      memset(d_u, 0, (size_t)R * sizeof(int32_t));
      for (int o = 0; o < D; o++) {
        const int32_t dv = d_vec[o];
        if (dv == 0) continue;
        int32_t *gbrow = G_Bl + (size_t)o * (size_t)R;
        const int8_t *brow = Bl + (size_t)o * (size_t)R;
        for (int r = 0; r < R; r++) {
          gbrow[r] += dv * ul[r];
          d_u[r] += dv * brow[r];
        }
      }

      for (int r = 0; r < R; r++) {
        const int32_t eu = clipi(rshift(d_u[r], A->sb_b), -8, 7);
        const size_t bit = ((size_t)b * (size_t)L + (size_t)l) * (size_t)R + (size_t)r;
        const int mask = (u_stream[bit >> 3] >> (bit & 7u)) & 1;
        d_u[r] = (yal[r] < -8 || yal[r] > 7) ? 0 : (eu * mask);
      }

      for (int r = 0; r < R; r++) {
        const int32_t dur = d_u[r];
        if (dur == 0) continue;
        int32_t *garow = G_Al + (size_t)r * (size_t)D;
        for (int j = 0; j < D; j++) garow[j] += dur * xm[j];
      }

      if (l > 0) {
        memset(e_next, 0, (size_t)D * sizeof(int32_t));
        for (int o = 0; o < D; o++) {
          const int32_t dv = d_vec[o];
          if (dv == 0) continue;
          const int8_t *wbrow = Wb + (size_t)o * (size_t)D;
          for (int j = 0; j < D; j++) e_next[j] += dv * wbrow[j];
        }
        for (int r = 0; r < R; r++) {
          const int32_t dur = d_u[r];
          if (dur == 0) continue;
          const int8_t *arow = Al + (size_t)r * (size_t)D;
          for (int j = 0; j < D; j++) e_next[j] += dur * arow[j];
        }
        for (int j = 0; j < D; j++) {
          const size_t idx2 = ((size_t)b * (size_t)L + (size_t)l) * (size_t)D + (size_t)j;
          const int pair = (in_stream[idx2 >> 2] >> ((idx2 & 3u) * 2u)) & 3;
          const int32_t m_masked = (pair != 0) ? e_next[j] : 0;
          e_cur[j] = clipi(d_vec[j] + rshift(m_masked, A->sb_in), -8, 7);
        }
      }
    }
  }

  free(drop_buf); free(x_raw); free(xs); free(xm_all); free(y_out); free(y_a); free(u_buf);
  free(route_buf); free(e_cur); free(e_next); free(d_vec); free(d_u);

  uint32_t clipped = 0, g_lanes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const int ntensors = 2 * L + 1;
  size_t off = 0;
  for (int t = 0; t < ntensors; t++) {
    const size_t nw = (t < 2 * L) ? block_ab : (size_t)NOUT * (size_t)D;
    int32_t *gt = g + off;
    for (size_t f = 0; f < nw; f++) {
      const int32_t c = clipi(gt[f], -NEB3_CLIP, NEB3_CLIP);
      clipped += (uint32_t)(c != gt[f]);
      gt[f] = c;
      g_lanes[f & 7u] += (uint32_t)c ^ ((uint32_t)f + 7919u * (uint32_t)(t + 1));
    }
    off += nw;
  }
  if (nclip) *nclip = clipped;

  uint8_t dt_buf[64], tk_buf[64];
  for (int d = 0; d < 8; d++) {
    put_le32(dt_buf + 4 * d, g_lanes[d]);
    put_le32(dt_buf + 32 + 4 * d, t_lanes[d]);
  }
  blake3_hash(dt_buf, sizeof dt_buf, digest);
  memcpy(tk_buf, digest, 32);
  memcpy(tk_buf + 32, key, 32);
  blake3_hash(tk_buf, sizeof tk_buf, ticket);
  return 0;
}

void neb3_model_apply(neb3_model *m, const int32_t *gs) {
  if (!m || !gs) return;
  const neb3_arch *A = neb3_arch_get(m->arch);
  if (!A || m->nweights != A->nweights) return;
  const size_t lora_end = (size_t)A->nblocks * 2u * (size_t)A->rank * (size_t)A->d;
  for (size_t f = 0; f < A->nweights; f++) {
    const int su = (f < lora_end ? A->su_lora : A->su_head) - NEB3_SU_DROP;
    const int64_t unit = (int64_t)1 << su;
    int64_t v = (int64_t)m->r[f] + (int64_t)gs[f];
    const int64_t q = v >= 0 ? (v >> su) : -((-v) >> su);
    const int64_t d = clip64(q, -1, 1);
    v -= d * unit;
    m->r[f] = (int32_t)clip64(v, -unit, unit);
    m->w[f] = (int8_t)clipi((int)m->w[f] - (int)d, -8, 7);
  }
}

int neb3_model_hashes(const neb3_model *m, uint8_t weights_hash[32], uint8_t residual_hash[32]) {
  static const char tw[] = "neb-model-weights", tr[] = "neb-model-residual";
  const size_t lw = sizeof tw - 1, lr = sizeof tr - 1;
  if (!m || !m->w || !m->r || (m->nweights & 1u) != 0) return -1;
  const size_t n = m->nweights;
  uint8_t *buf = (uint8_t *)malloc(lr + 4u * n);
  if (!buf) return -1;
  memcpy(buf, tw, lw);
  for (size_t j = 0; j < n / 2; j++) {
    buf[lw + j] = (uint8_t)((m->w[2 * j] & 15) | ((m->w[2 * j + 1] & 15) << 4));
  }
  blake3_hash(buf, lw + n / 2, weights_hash);
  memcpy(buf, tr, lr);
  for (size_t f = 0; f < n; f++) put_le32(buf + lr + 4u * f, (uint32_t)m->r[f]);
  blake3_hash(buf, lr + 4u * n, residual_hash);
  free(buf);
  return 0;
}

int neb3_model_predict(const neb3_model *m, const neb3_base *base, const uint8_t *x) {
  if (!m || !base || !x || m->arch != base->arch) return -1;
  const neb3_arch *A = neb3_arch_get(m->arch);
  if (!A || m->nweights != A->nweights || base->nweights != A->base_weights) return -1;
  const int D = A->d, R = A->rank, L = A->nblocks, E = A->nexperts, NOUT = A->nout;
  const size_t block_ab = (size_t)R * (size_t)D;
  const size_t expert_sz = (size_t)D * (size_t)D;

  int8_t *cur = (int8_t *)malloc((size_t)D);
  int8_t *nxt = (int8_t *)malloc((size_t)D);
  int8_t *u = (int8_t *)malloc((size_t)R);
  if (!cur || !nxt || !u) { free(cur); free(nxt); free(u); return -1; }

  for (int j = 0; j < D; j++) cur[j] = (int8_t)((int)x[j] - 4);
  for (int l = 0; l < L; l++) {
    const int e_l = l % E;
    const int8_t *Wb = base->w_base + (size_t)e_l * expert_sz;
    const int8_t *Al = m->w + (size_t)l * 2u * block_ab;
    const int8_t *Bl = Al + block_ab;
    for (int r = 0; r < R; r++) {
      int32_t za = 0;
      const int8_t *arow = Al + (size_t)r * (size_t)D;
      for (int j = 0; j < D; j++) za += (int32_t)arow[j] * cur[j];
      /* Scale by 3/4 (input keep) and 1/2 (bottleneck keep) for deterministic inference expectation */
      u[r] = (int8_t)(clipi(rshift((za * 3) >> 2, A->s_a), -8, 7) >> 1);
    }
    for (int o = 0; o < D; o++) {
      int32_t zb = 0, zl = 0;
      const int8_t *wbrow = Wb + (size_t)o * (size_t)D;
      const int8_t *brow = Bl + (size_t)o * (size_t)R;
      for (int j = 0; j < D; j++) zb += (int32_t)wbrow[j] * cur[j];
      zb = (zb * 3) >> 2;
      for (int r = 0; r < R; r++) zl += (int32_t)brow[r] * u[r];
      nxt[o] = (int8_t)clipi((int32_t)cur[o] + rshift(zb + zl * ((int32_t)1 << A->s_lora), A->s_out), -8, 7);
    }
    memcpy(cur, nxt, (size_t)D);
  }
  const int8_t *H = m->w + (size_t)L * 2u * block_ab;
  int best = 0;
  int32_t bestz = 0;
  for (int o = 0; o < NOUT; o++) {
    int32_t zh = 0;
    const int8_t *hrow = H + (size_t)o * (size_t)D;
    for (int j = 0; j < D; j++) zh += (int32_t)hrow[j] * cur[j];
    if (o == 0 || zh > bestz) { best = o; bestz = zh; }
  }
  free(cur); free(nxt); free(u);
  return best;
}
