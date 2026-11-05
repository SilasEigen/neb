/* Produces vectors/shares.txt (spec/shares.md section 10) on the standard output with the C reference of pow/ref.
 *   gen_share_vectors            the vector file
 *   gen_share_vectors --train N  trains N steps of 32 shares on the test dataset and prints the accuracy
 * Statistics go to the standard error. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "blake3_min.h"
#include "chacha8.h"
#include "shares.h"
#include "syn_dataset.h"

static void put_le32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

static void hex(const uint8_t *b, int n) { for (int i = 0; i < n; i++) printf("%02x", b[i]); }

static void share_key(uint32_t s, uint32_t j, uint8_t key[32]) {
  uint8_t in[14 + 8];
  memcpy(in, "neb-test-share", 14);
  put_le32(in + 14, s);
  put_le32(in + 18, j);
  blake3_hash(in, sizeof in, key);
}

int main(int argc, char **argv) {
  /* n_s of spec/shares.md section 10: full steps, single shares and odd sizes */
  static const int counts[8] = {32, 32, 32, 1, 32, 3, 32, 17};
  const neb_dataset ds = {NEB_SYN_N, neb_syn_get, NULL};
  uint8_t seed[32] = {0}, wh[32], rh[32];
  neb_model *m = (neb_model *)malloc(sizeof *m);
  int32_t *g1 = (int32_t *)malloc(sizeof(int32_t) * NEB_N1), *g2 = (int32_t *)malloc(sizeof(int32_t) * NEB_N2);
  int32_t *s1 = (int32_t *)malloc(sizeof(int32_t) * NEB_N1), *s2 = (int32_t *)malloc(sizeof(int32_t) * NEB_N2);
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY); /* the vector file has LF endings */
#endif
  neb_model_init(seed, m);
  /* train mode: 32 shares per step and no vector lines; the statistics show how often the clip and the clamp act */
  const int train = (argc == 3 && strcmp(argv[1], "--train") == 0) ? atoi(argv[2]) : 0;
  const int steps = train ? train : 24;
  if (!train) {
    neb_model_hashes(m, wh, rh);
    printf("neb-shares-vectors 1\ninit ");
    hex(wh, 32); printf(" "); hex(rh, 32); printf("\n");
  }
  long clipped_total = 0, entries_total = 0, moved = 0, rclamp = 0;
  for (int s = 0; s < steps; s++) {
    const int n = train ? 32 : counts[s % 8];
    /* entry-wise sums of the clipped gradients: the input of neb_model_apply */
    memset(s1, 0, sizeof(int32_t) * NEB_N1);
    memset(s2, 0, sizeof(int32_t) * NEB_N2);
    for (int j = 0; j < n; j++) {
      uint8_t key[32], digest[32], ticket[32];
      uint32_t idx[NEB_NB], nclip;
      share_key((uint32_t)s, (uint32_t)j, key);
      neb_share_eval(m, &ds, key, g1, g2, digest, ticket, idx, &nclip);
      clipped_total += nclip;
      entries_total += NEB_N1 + NEB_N2;
      for (int e = 0; e < NEB_N1; e++) s1[e] += g1[e];
      for (int e = 0; e < NEB_N2; e++) s2[e] += g2[e];
      if (!train) {
        printf("share %d %d ", s, j); hex(digest, 32); printf(" "); hex(ticket, 32);
        for (int b = 0; b < NEB_NB; b++) printf(" %u", idx[b]);
        printf("\n");
      }
    }
    int8_t *before = (int8_t *)malloc(NEB_N1);
    memcpy(before, m->w1, NEB_N1);
    neb_model_apply(m, s1, s2);
    for (int e = 0; e < NEB_N1; e++) moved += (before[e] != m->w1[e]);
    free(before);
    for (int e = 0; e < NEB_N2; e++) rclamp += (m->r2[e] == (1 << NEB_SU2) || m->r2[e] == -(1 << NEB_SU2));
    if (!train) {
      neb_model_hashes(m, wh, rh);
      printf("state %d %d ", s, n); hex(wh, 32); printf(" "); hex(rh, 32); printf("\n");
    }
  }
  fprintf(stderr, "steps %d: gradient entries clipped %.3f %%, layer-1 weights moved %ld, layer-2 residuals at the clamp %ld\n",
          steps, 100.0 * (double)clipped_total / (double)entries_total, moved, rclamp);
  if (train) {
    int ok = 0;
    for (uint32_t i = 0; i < ds.n; i++) { uint8_t x[NEB_NIN]; const int c = neb_syn_get(NULL, i, x); ok += (neb_model_predict(m, x) == c); }
    printf("%d steps of 32 shares: accuracy on the %u test-dataset samples %.1f %%\n", steps, ds.n, 100.0 * ok / ds.n);
  }
  free(m); free(g1); free(g2); free(s1); free(s2);
  return 0;
}
