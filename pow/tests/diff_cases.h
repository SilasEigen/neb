#ifndef NEB_DIFF_CASES_H
#define NEB_DIFF_CASES_H
/* Models and datasets of the differential tests of the optimised evaluations of training shares (AVX-512 CPU, CUDA K1 and
 * K2) against the C reference. Valid C and C++. The cases reach the corners of the arithmetic:
 *   models   : initial state, a model trained for 10 steps, all weights at 7, all at -8, alternating -8 and 7, all zero
 *   datasets : the synthetic test dataset (spec/shares.md section 9), a mostly-zero image set (80 % of the pixels are 0, as
 *              in MNIST), a set with all-zero and all-7 images
 * Include after shares.h, blake3_min.h and syn_dataset.h. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const char *name;
  uint32_t n;
  uint8_t *px;
  uint8_t *lab;
} flat_t;

static inline int flat_get(void *ctx, uint32_t i, uint8_t x[NEB_NIN]) {
  const flat_t *f = (const flat_t *)ctx;
  memcpy(x, f->px + (size_t)i * NEB_NIN, NEB_NIN);
  return f->lab[i];
}

static uint64_t rng_state = 0x6E656231;
static inline uint64_t rnd(void) {
  uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static inline uint64_t rnd_seeded(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static inline void put_le32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

static inline void key_of(uint32_t model, uint32_t ds, uint32_t k, uint8_t key[32]) {
  uint8_t in[16 + 12];
  memcpy(in, "neb-avx512-test-", 16);
  put_le32(in + 16, model);
  put_le32(in + 20, ds);
  put_le32(in + 24, k);
  blake3_hash(in, sizeof in, key);
}

static inline void make_flat_syn(flat_t *f) {
  f->name = "synthetic";
  f->n = NEB_SYN_N;
  f->px = (uint8_t *)malloc((size_t)f->n * NEB_NIN);
  f->lab = (uint8_t *)malloc(f->n);
  for (uint32_t i = 0; i < f->n; i++) f->lab[i] = (uint8_t)neb_syn_get(NULL, i, f->px + (size_t)i * NEB_NIN);
}

static inline void make_flat_sparse(flat_t *f) {
  f->name = "mostly-zero";
  f->n = 512;
  f->px = (uint8_t *)malloc((size_t)f->n * NEB_NIN);
  f->lab = (uint8_t *)malloc(f->n);
  for (uint32_t i = 0; i < f->n; i++) {
    for (int p = 0; p < NEB_NIN; p++) f->px[(size_t)i * NEB_NIN + p] = (rnd() % 5 == 0) ? (uint8_t)(1 + rnd() % 7) : 0;
    f->lab[i] = (uint8_t)(rnd() % 10);
  }
}

static inline void make_flat_extreme(flat_t *f) {
  f->name = "all-zero-and-all-7";
  f->n = 64;
  f->px = (uint8_t *)malloc((size_t)f->n * NEB_NIN);
  f->lab = (uint8_t *)malloc(f->n);
  for (uint32_t i = 0; i < f->n; i++) {
    memset(f->px + (size_t)i * NEB_NIN, (i & 1) ? 7 : 0, NEB_NIN);
    if (i % 5 == 4) for (int p = 0; p < NEB_NIN; p += 3) f->px[(size_t)i * NEB_NIN + p] = (uint8_t)(rnd() & 7);
    f->lab[i] = (uint8_t)(i % 10);
  }
}

static inline void train(neb_model *m, const neb_dataset *nd, int steps, int shares) {
  int32_t *g1 = (int32_t *)malloc(sizeof(int32_t) * NEB_N1), *g2 = (int32_t *)malloc(sizeof(int32_t) * NEB_N2);
  int32_t *s1 = (int32_t *)malloc(sizeof(int32_t) * NEB_N1), *s2 = (int32_t *)malloc(sizeof(int32_t) * NEB_N2);
  for (int s = 0; s < steps; s++) {
    memset(s1, 0, sizeof(int32_t) * NEB_N1);
    memset(s2, 0, sizeof(int32_t) * NEB_N2);
    for (int j = 0; j < shares; j++) {
      uint8_t key[32], dg[32], tk[32];
      key_of(99, (uint32_t)s, (uint32_t)j, key);
      neb_share_eval(m, nd, key, g1, g2, dg, tk, NULL, NULL);
      for (int e = 0; e < NEB_N1; e++) s1[e] += g1[e];
      for (int e = 0; e < NEB_N2; e++) s2[e] += g2[e];
    }
    neb_model_apply(m, s1, s2);
  }
  free(g1); free(g2); free(s1); free(s2);
}


#define DIFF_MODELS 6
static const char *const diff_model_names[DIFF_MODELS] = {"initial", "trained", "all-7", "all--8", "alternating", "zero"};

/* Allocates and fills the DIFF_MODELS models (the caller keeps them for the whole run). */
static inline void diff_make_models(neb_model *models[DIFF_MODELS]) {
  for (int i = 0; i < DIFF_MODELS; i++) models[i] = (neb_model *)calloc(1, sizeof(neb_model));
  uint8_t seed[32] = {0};
  neb_model_init(seed, models[0]);
  neb_model_init(seed, models[1]);
  const neb_dataset syn = {NEB_SYN_N, neb_syn_get, NULL};
  train(models[1], &syn, 10, 16);
  memset(models[2]->w1, 7, NEB_N1);
  memset(models[2]->w2, 7, NEB_N2);
  memset(models[3]->w1, 0xF8, NEB_N1);
  memset(models[3]->w2, 0xF8, NEB_N2);
  for (int e = 0; e < NEB_N1; e++) models[4]->w1[e] = (e & 1) ? 7 : -8;
  for (int e = 0; e < NEB_N2; e++) models[4]->w2[e] = ((e / 3) & 1) ? -8 : 7;
}

/* The network dataset file (spec/shares.md section 11.6: "NMD1", LE32 n, n records of 784 pixels and a label). 0 on success. */
static inline int load_flat_file(const char *path, flat_t *f) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return 1;
  uint8_t hdr[8];
  if (fread(hdr, 1, 8, fp) != 8 || memcmp(hdr, "NMD1", 4)) { fclose(fp); return 2; }
  f->name = "network-file";
  f->n = (uint32_t)hdr[4] | ((uint32_t)hdr[5] << 8) | ((uint32_t)hdr[6] << 16) | ((uint32_t)hdr[7] << 24);
  f->px = (uint8_t *)malloc((size_t)f->n * NEB_NIN);
  f->lab = (uint8_t *)malloc(f->n);
  uint8_t rec[NEB_NIN + 1];
  for (uint32_t i = 0; i < f->n; i++) {
    if (fread(rec, 1, NEB_NIN + 1, fp) != NEB_NIN + 1) { fclose(fp); return 3; }
    memcpy(f->px + (size_t)i * NEB_NIN, rec, NEB_NIN);
    f->lab[i] = rec[NEB_NIN];
  }
  fclose(fp);
  return 0;
}

/* A raw model dump (struct neb_model: W1, W2, R1, R2, 508160 bytes). 0 on success. */
static inline int load_model_file(const char *path, neb_model *m) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return 1;
  const size_t r = fread(m, 1, sizeof *m, fp);
  fclose(fp);
  return r == sizeof *m ? 0 : 2;
}

#endif
