/* Produces vectors/nebhash2.txt (spec/nebhash-v2.md section 8) on the standard output with the C reference of pow/ref.
 * For each architecture: the initial state of a zero seed, then STEPS model steps of n_s shares on the test dataset of
 * the architecture, each share keyed by BLAKE3("neb-test-share2" || arch || LE32(s) || LE32(j)). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "blake3_min.h"
#include "nebhash2.h"
#include "syn2_dataset.h"

#define STEPS 6
static const int counts[STEPS] = {32, 1, 3, 32, 17, 2};

static void put_le32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static void hex(const uint8_t *b, int n) { for (int i = 0; i < n; i++) printf("%02x", b[i]); }

static void share_key(int arch, uint32_t s, uint32_t j, uint8_t key[32]) {
  uint8_t in[15 + 1 + 8];
  memcpy(in, "neb-test-share2", 15);
  in[15] = (uint8_t)arch;
  put_le32(in + 16, s);
  put_le32(in + 20, j);
  blake3_hash(in, sizeof in, key);
}

int main(void) {
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  neb2_model *m = (neb2_model *)malloc(sizeof *m);
  int32_t *g = (int32_t *)malloc(sizeof(int32_t) * NEB2_MAX_WEIGHTS);
  int32_t *sum = (int32_t *)malloc(sizeof(int32_t) * NEB2_MAX_WEIGHTS);
  if (!m || !g || !sum) return 1;
  printf("neb-nebhash2-vectors 1\n");
  for (int arch = NEB2_KEP1; arch <= NEB2_RET2; arch++) {
    const neb2_arch *A = neb2_arch_get(arch);
    const neb2_dataset ds = {NEB_SYN2_N, neb_syn2_get, &arch};
    uint8_t seed[32] = {0}, wh[32], rh[32];
    if (neb2_model_init(arch, seed, m) != 0) return 1;
    neb2_model_hashes(m, wh, rh);
    printf("init %d ", arch); hex(wh, 32); printf(" "); hex(rh, 32); printf("\n");
    for (int s = 0; s < STEPS; s++) {
      memset(sum, 0, sizeof(int32_t) * (size_t)A->nweights);
      for (int j = 0; j < counts[s]; j++) {
        uint8_t key[32], digest[32], ticket[32];
        uint32_t idx[NEB_NB];
        share_key(arch, (uint32_t)s, (uint32_t)j, key);
        if (neb2_share_eval(m, &ds, key, g, digest, ticket, idx, NULL) != 0) return 1;
        for (int f = 0; f < A->nweights; f++) sum[f] += g[f];
        printf("share %d %d %d ", arch, s, j); hex(digest, 32); printf(" "); hex(ticket, 32);
        for (int b = 0; b < NEB_NB; b++) printf(" %u", idx[b]);
        printf("\n");
      }
      neb2_model_apply(m, sum);
      neb2_model_hashes(m, wh, rh);
      printf("state %d %d %d ", arch, s, counts[s]); hex(wh, 32); printf(" "); hex(rh, 32); printf("\n");
    }
  }
  free(m); free(g); free(sum);
  return 0;
}
