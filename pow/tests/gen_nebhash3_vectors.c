/* Deterministic vector generator for spec/nebhash-v3.md (LORA_TINY, arch 1).
 * Prints base hash, init state hashes, 6 steps of (32, 1, 3, 32, 17, 2) shares and state hashes. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "blake3_min.h"
#include "chacha8.h"
#include "nebhash3.h"

#define N_TEST 2048u

typedef struct {
  int arch;
  int d;
  uint8_t *proto[2];
} syn3_ctx;

static void put_le32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static void print_hex(const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++) printf("%02x", p[i]);
}

static int syn3_init(syn3_ctx *ctx, int arch, int d) {
  ctx->arch = arch;
  ctx->d = d;
  for (int c = 0; c < 2; c++) {
    uint8_t in[14 + 2], key[32];
    memcpy(in, "neb-syn3-proto", 14);
    in[14] = (uint8_t)arch;
    in[15] = (uint8_t)c;
    blake3_hash(in, sizeof in, key);
    ctx->proto[c] = (uint8_t *)malloc((size_t)d);
    if (!ctx->proto[c]) return -1;
    chacha8_stream(key, ctx->proto[c], (size_t)d);
    for (int j = 0; j < d; j++) ctx->proto[c][j] &= 7u;
  }
  return 0;
}

static void syn3_free(syn3_ctx *ctx) {
  free(ctx->proto[0]);
  free(ctx->proto[1]);
}

static int syn3_get(void *vctx, uint32_t idx, uint8_t *x) {
  const syn3_ctx *ctx = (const syn3_ctx *)vctx;
  uint8_t in[15 + 1 + 4], key[32];
  memcpy(in, "neb-syn3-sample", 15);
  in[15] = (uint8_t)ctx->arch;
  put_le32(in + 16, idx);
  blake3_hash(in, sizeof in, key);
  uint8_t *noise = (uint8_t *)malloc((size_t)ctx->d);
  if (!noise) return 0;
  chacha8_stream(key, noise, (size_t)ctx->d);
  const int label = (int)(idx & 1u);
  const uint8_t *p = ctx->proto[label];
  for (int j = 0; j < ctx->d; j++) {
    int v = (int)p[j] + (int)(noise[j] & 3u) - 1;
    if (v < 0) v = 0;
    if (v > 7) v = 7;
    x[j] = (uint8_t)v;
  }
  free(noise);
  return label;
}

static int8_t *make_syn3_base(int arch, size_t nweights) {
  uint8_t in[13 + 1], key[32];
  memcpy(in, "neb-syn3-base", 13);
  in[13] = (uint8_t)arch;
  blake3_hash(in, sizeof in, key);
  uint8_t *raw = (uint8_t *)malloc(nweights);
  int8_t *w = (int8_t *)malloc(nweights);
  if (!raw || !w) { free(raw); free(w); return NULL; }
  chacha8_stream(key, raw, nweights);
  for (size_t k = 0; k < nweights; k++) {
    const int b = raw[k];
    w[k] = (int8_t)((b & 3) + ((b >> 2) & 3) + ((b >> 4) & 3) + ((b >> 6) & 3) - 6);
  }
  free(raw);
  return w;
}

static void test_share_key(int arch, uint32_t s, uint32_t j, uint8_t out[32]) {
  uint8_t in[15 + 1 + 8];
  memcpy(in, "neb-test-share3", 15);
  in[15] = (uint8_t)arch;
  put_le32(in + 16, s);
  put_le32(in + 20, j);
  blake3_hash(in, sizeof in, out);
}

int main(void) {
  static const int STEP_SHARES[6] = {32, 1, 3, 32, 17, 2};
  const int arch = NEB3_LORA_TINY;
  const neb3_arch *A = neb3_arch_get(arch);
  if (!A) return 1;

  int8_t *w_base = make_syn3_base(arch, A->base_weights);
  if (!w_base) return 1;
  neb3_base base = {arch, A->base_weights, w_base};

  syn3_ctx ds_ctx;
  if (syn3_init(&ds_ctx, arch, A->d) != 0) return 1;
  neb3_dataset ds = {N_TEST, syn3_get, &ds_ctx};

  uint8_t zero_seed[32];
  memset(zero_seed, 0, sizeof zero_seed);
  neb3_model m;
  if (neb3_model_init(arch, zero_seed, &m) != 0) return 1;

  uint8_t bh[32], wh[32], rh[32];
  if (neb3_base_hash(&base, bh) != 0) return 1;
  if (neb3_model_hashes(&m, wh, rh) != 0) return 1;

  printf("arch %d base ", arch); print_hex(bh, 32); printf("\n");
  printf("arch %d init ", arch); print_hex(wh, 32); printf(" "); print_hex(rh, 32); printf("\n");

  int32_t *g = (int32_t *)malloc(A->nweights * sizeof(int32_t));
  int32_t *sums = (int32_t *)malloc(A->nweights * sizeof(int32_t));
  if (!g || !sums) return 1;

  for (int s = 0; s < 6; s++) {
    memset(sums, 0, A->nweights * sizeof(int32_t));
    const int nsh = STEP_SHARES[s];
    for (int j = 0; j < nsh; j++) {
      uint8_t key[32], digest[32], ticket[32];
      uint32_t idx[NEB3_NB];
      test_share_key(arch, (uint32_t)s, (uint32_t)j, key);
      if (neb3_share_eval(&m, &base, &ds, key, g, digest, ticket, idx, NULL) != 0) return 1;
      for (size_t f = 0; f < A->nweights; f++) sums[f] += g[f];
      if (j == 0 || j == nsh - 1) {
        printf("arch %d step %d share %d ", arch, s, j);
        print_hex(digest, 32);
        printf(" ");
        print_hex(ticket, 32);
        printf(" %u %u\n", idx[0], idx[NEB3_NB - 1]);
      }
    }
    neb3_model_apply(&m, sums);
    if (neb3_model_hashes(&m, wh, rh) != 0) return 1;
    printf("arch %d step %d state ", arch, s);
    print_hex(wh, 32);
    printf(" ");
    print_hex(rh, 32);
    printf("\n");
  }

  free(g);
  free(sums);
  neb3_model_free(&m);
  syn3_free(&ds_ctx);
  free(w_base);
  return 0;
}
