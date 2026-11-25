/* Reference implementation of spec/nebhash-v3.md: frozen INT8 base + on-chain integer LoRA adapters with
 * key-derived deterministic dropout. Scalar, portable, exact integer arithmetic only. */
#include "nebhash3.h"
#include <stdlib.h>
#include <string.h>
#include "blake3_min.h"
#include "chacha8.h"

static const neb3_arch LORA_TINY = {
    NEB3_LORA_TINY, 64, 8, 2, NEB3_NOUT,
    4, 1, 5, 4, 5, 5, 3, 14, 12,
    (size_t)2 * 2 * 64 * 8 + (size_t)NEB3_NOUT * 64,
    (size_t)2 * 64 * 64};

static const neb3_arch QWEN3_8B_LORA = {
    NEB3_QWEN3_8B_LORA, 4096, 8, 4, NEB3_NOUT,
    7, 1, 8, 5, 8, 8, 3, 16, 14,
    (size_t)4 * 2 * 4096 * 8 + (size_t)NEB3_NOUT * 4096,
    (size_t)4 * 4096 * 4096};

const neb3_arch *neb3_arch_get(int id) {
  if (id == NEB3_LORA_TINY) return &LORA_TINY;
  if (id == NEB3_QWEN3_8B_LORA) return &QWEN3_8B_LORA;
  return NULL;
}

static int clipi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static int32_t rshift(int32_t x, int n) { return n == 0 ? x : (x + ((int32_t)1 << (n - 1))) >> n; }

static void put_le32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static uint32_t get_le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
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
  if (!base || !base->w_base || (base->nweights & 1u) != 0) return -1;
  const size_t packed = base->nweights / 2;
  uint8_t *buf = (uint8_t *)malloc(lb + packed);
  if (!buf) return -1;
  memcpy(buf, tb, lb);
  for (size_t j = 0; j < packed; j++) {
    buf[lb + j] = (uint8_t)((base->w_base[2 * j] & 15) | ((base->w_base[2 * j + 1] & 15) << 4));
  }
  blake3_hash(buf, lb + packed, out_hash);
  free(buf);
  return 0;
}

int neb3_share_eval(const neb3_model *m, const neb3_base *base, const neb3_dataset *ds, const uint8_t key[32],
                    int32_t *g, uint8_t digest[32], uint8_t ticket[32], uint32_t idx_out[NEB3_NB], uint32_t *nclip) {
  if (!m || !base || !ds || !g || m->arch != base->arch) return -1;
  const neb3_arch *A = neb3_arch_get(m->arch);
  if (!A || m->nweights != A->nweights || base->nweights != A->base_weights || ds->n == 0) return -1;

  const int D = A->d, R = A->rank, L = A->nblocks, NOUT = A->nout;
  const size_t block_ab = (size_t)R * (size_t)D;

  /* Key-derived deterministic dropout stream (section 4.1) */
  uint8_t drop_in[13 + 32], drop_key[32];
  memcpy(drop_in, "neb-lora-drop", 13);
  memcpy(drop_in + 13, key, 32);
  blake3_hash(drop_in, sizeof drop_in, drop_key);
  const size_t drop_bits = (size_t)NEB3_NB * (size_t)L * (size_t)R;
  const size_t drop_bytes = (drop_bits + 7u) / 8u;
  uint8_t *drop_stream = (uint8_t *)malloc(drop_bytes);

  uint8_t *x_raw = (uint8_t *)malloc((size_t)D);
  int8_t *xs = (int8_t *)malloc((size_t)(L + 1) * (size_t)D);
  int32_t *y_out = (int32_t *)malloc((size_t)L * (size_t)D * sizeof(int32_t));
  int32_t *y_a = (int32_t *)malloc((size_t)L * (size_t)R * sizeof(int32_t));
  int8_t *u_buf = (int8_t *)malloc((size_t)L * (size_t)R);
  int32_t *e_cur = (int32_t *)malloc((size_t)D * sizeof(int32_t));
  int32_t *e_next = (int32_t *)malloc((size_t)D * sizeof(int32_t));
  int32_t *d_vec = (int32_t *)malloc((size_t)D * sizeof(int32_t));
  int32_t *d_u = (int32_t *)malloc((size_t)R * sizeof(int32_t));

  if (!drop_stream || !x_raw || !xs || !y_out || !y_a || !u_buf || !e_cur || !e_next || !d_vec || !d_u) {
    free(drop_stream); free(x_raw); free(xs); free(y_out); free(y_a); free(u_buf);
    free(e_cur); free(e_next); free(d_vec); free(d_u);
    return -1;
  }

  chacha8_stream(drop_key, drop_stream, drop_bytes);
  uint32_t idx[NEB3_NB];
  neb3_share_indices(key, ds->n, idx);
  if (idx_out) memcpy(idx_out, idx, sizeof idx);
  memset(g, 0, A->nweights * sizeof(int32_t));

  const size_t h_off = (size_t)L * 2u * block_ab;
  const int8_t *H = m->w + h_off;
  int32_t *G_H = g + h_off;

  for (int b = 0; b < NEB3_NB; b++) {
    const int label = ds->get(ds->ctx, idx[b], x_raw);
    for (int j = 0; j < D; j++) xs[j] = (int8_t)x_raw[j];

    /* Forward pass through L frozen+LoRA blocks */
    for (int l = 0; l < L; l++) {
      const int8_t *xl = xs + (size_t)l * (size_t)D;
      int8_t *xnext = xs + (size_t)(l + 1) * (size_t)D;
      const int8_t *Wb = base->w_base + (size_t)l * (size_t)D * (size_t)D;
      const int8_t *Al = m->w + (size_t)l * 2u * block_ab;
      const int8_t *Bl = Al + block_ab;
      int32_t *yal = y_a + (size_t)l * (size_t)R;
      int8_t *ul = u_buf + (size_t)l * (size_t)R;
      int32_t *yol = y_out + (size_t)l * (size_t)D;

      for (int r = 0; r < R; r++) {
        int32_t za = 0;
        const int8_t *arow = Al + (size_t)r * (size_t)D;
        for (int j = 0; j < D; j++) za += (int32_t)arow[j] * xl[j];
        const int32_t ya = rshift(za, A->s_a);
        yal[r] = ya;
        const size_t bit = ((size_t)b * (size_t)L + (size_t)l) * (size_t)R + (size_t)r;
        const int mask = (drop_stream[bit >> 3] >> (bit & 7u)) & 1;
        ul[r] = (int8_t)(clipi(ya, -8, 7) * mask);
      }

      for (int o = 0; o < D; o++) {
        int32_t zb = 0, zl = 0;
        const int8_t *wbrow = Wb + (size_t)o * (size_t)D;
        const int8_t *brow = Bl + (size_t)o * (size_t)R;
        for (int j = 0; j < D; j++) zb += (int32_t)wbrow[j] * xl[j];
        for (int r = 0; r < R; r++) zl += (int32_t)brow[r] * ul[r];
        const int32_t yo = rshift(zb + (zl << A->s_lora), A->s_out);
        yol[o] = yo;
        xnext[o] = (int8_t)clipi(yo, -8, 7);
      }
    }

    /* Output head forward & unmasked output error */
    const int8_t *xL = xs + (size_t)L * (size_t)D;
    int32_t e_h[NEB3_NOUT];
    for (int o = 0; o < NOUT; o++) {
      int32_t zh = 0;
      const int8_t *hrow = H + (size_t)o * (size_t)D;
      for (int j = 0; j < D; j++) zh += (int32_t)hrow[j] * xL[j];
      const int32_t ah = clipi(rshift(zh, A->s_head), -8, 7);
      const int t = (o == label) ? NEB_TPOS : NEB_TNEG;
      e_h[o] = clipi(ah - t, -8, 7);
    }

    /* Head backward */
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

    /* Backward pass through L blocks (train A and B only; W_base is frozen) */
    for (int l = L - 1; l >= 0; l--) {
      const int8_t *xl = xs + (size_t)l * (size_t)D;
      const int8_t *Wb = base->w_base + (size_t)l * (size_t)D * (size_t)D;
      const int8_t *Al = m->w + (size_t)l * 2u * block_ab;
      const int8_t *Bl = Al + block_ab;
      int32_t *G_Al = g + (size_t)l * 2u * block_ab;
      int32_t *G_Bl = G_Al + block_ab;
      const int32_t *yal = y_a + (size_t)l * (size_t)R;
      const int8_t *ul = u_buf + (size_t)l * (size_t)R;
      const int32_t *yol = y_out + (size_t)l * (size_t)D;

      for (int o = 0; o < D; o++) {
        d_vec[o] = (yol[o] < -8 || yol[o] > 7) ? 0 : e_cur[o];
      }

      /* G_B += d (x) u, and m_u = B^T * d */
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
        const int mask = (drop_stream[bit >> 3] >> (bit & 7u)) & 1;
        d_u[r] = (yal[r] < -8 || yal[r] > 7) ? 0 : (eu * mask);
      }

      /* G_A += d_u (x) xl */
      for (int r = 0; r < R; r++) {
        const int32_t dur = d_u[r];
        if (dur == 0) continue;
        int32_t *garow = G_Al + (size_t)r * (size_t)D;
        for (int j = 0; j < D; j++) garow[j] += dur * xl[j];
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
        for (int j = 0; j < D; j++) e_cur[j] = clipi(rshift(e_next[j], A->sb_in), -8, 7);
      }
    }
  }

  free(drop_stream); free(x_raw); free(xs); free(y_out); free(y_a); free(u_buf);
  free(e_cur); free(e_next); free(d_vec); free(d_u);

  /* Clip and digest over trainable tensors t = 0 .. 2*L (A[0], B[0], ..., H) */
  uint32_t clipped = 0, lanes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const int ntensors = 2 * L + 1;
  size_t off = 0;
  for (int t = 0; t < ntensors; t++) {
    const size_t nw = (t < 2 * L) ? block_ab : (size_t)NOUT * (size_t)D;
    int32_t *gt = g + off;
    for (size_t f = 0; f < nw; f++) {
      const int32_t c = clipi(gt[f], -NEB_CLIP, NEB_CLIP);
      clipped += (uint32_t)(c != gt[f]);
      gt[f] = c;
      lanes[f & 7u] += (uint32_t)c ^ ((uint32_t)f + 7919u * (uint32_t)(t + 1));
    }
    off += nw;
  }
  if (nclip) *nclip = clipped;
  uint8_t in[64];
  for (int d = 0; d < 8; d++) put_le32(digest + 4 * d, lanes[d]);
  memcpy(in, digest, 32);
  memcpy(in + 32, key, 32);
  blake3_hash(in, sizeof in, ticket);
  return 0;
}

void neb3_model_apply(neb3_model *m, const int32_t *gs) {
  if (!m || !gs) return;
  const neb3_arch *A = neb3_arch_get(m->arch);
  if (!A || m->nweights != A->nweights) return;
  const size_t lora_end = (size_t)A->nblocks * 2u * (size_t)A->rank * (size_t)A->d;
  for (size_t f = 0; f < A->nweights; f++) {
    const int su = (f < lora_end ? A->su_lora : A->su_head) - NEB3_SU_DROP;
    const int32_t unit = (int32_t)1 << su;
    int32_t v = m->r[f] + gs[f];
    const int32_t q = v >= 0 ? (v >> su) : -((-v) >> su);
    const int32_t d = clipi(q, -1, 1);
    v -= d * unit;
    m->r[f] = clipi(v, -unit, unit);
    m->w[f] = (int8_t)clipi(m->w[f] - d, -8, 7);
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
