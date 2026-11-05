/* Differential test of the AVX-512 evaluation of training shares (pow/cpu/shares_avx512.c) against the C reference
 * (pow/ref/shares.c): digest, ticket, the 16 sample indices and, on every 16th key, the full clipped gradients, for
 * both variants (dense and sparse), on several models and datasets chosen to reach the corners of the arithmetic
 * (pow/tests/diff_cases.h).
 * Usage: test_shares_avx512 [KEYS_PER_CASE] [--real DATASET_FILE --model MODEL_FILE ... [--real-keys N]]
 *   KEYS_PER_CASE   keys per model and dataset of the built-in suite (default 2048)
 *   --real FILE     also run the network dataset file with each --model (raw model dumps) on N keys (default 100000)
 * Exit code 0 when nothing differs. Needs AVX-512 F, BW, VL, DQ and VNNI. Built with -DNEB_NEGATIVE_CONTROL_CLIP or
 * -DNEB_NEGATIVE_CONTROL_SPARSE_WORD, the AVX code is deliberately wrong and this test must fail
 * (pow/tests/check_avx512.sh checks that it does). Uses OpenMP for the reference when available. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "blake3_min.h"
#include "shares.h"
#include "shares_avx512.h"
#include "syn_dataset.h"
#include "diff_cases.h"

/* One model on one dataset: `keys` keys, both variants; returns the number of differences. */
static long run_case(const char *mname, const neb_model *model, const flat_t *f, long keys, uint32_t mi, uint32_t di, long *evals, long *grads) {
  neb_cpu_model *cm = neb_cpu_model_new(model->w1, model->w2);
  neb_cpu_dataset *cd = neb_cpu_dataset_new(f->px, f->lab, f->n);
  if (!cm || !cd) { fprintf(stderr, "out of memory\n"); exit(2); }
  const neb_dataset nd = {f->n, flat_get, (void *)f};
  long bad = 0, ev = 0, gr = 0;
#pragma omp parallel reduction(+ : bad, ev, gr)
  {
    int32_t *rg1 = (int32_t *)malloc(sizeof(int32_t) * NEB_N1), *rg2 = (int32_t *)malloc(sizeof(int32_t) * NEB_N2);
    int32_t *xg1 = (int32_t *)malloc(sizeof(int32_t) * NEB_N1), *xg2 = (int32_t *)malloc(sizeof(int32_t) * NEB_N2);
    neb_cpu_scratch *scr = neb_cpu_scratch_new();
    if (!scr || !rg1 || !rg2 || !xg1 || !xg2) { fprintf(stderr, "out of memory\n"); exit(2); }
#pragma omp for schedule(dynamic, 16)
    for (long k = 0; k < keys; k++) {
      uint8_t key[32], rd[32], rt[32];
      uint32_t ri[NEB_NB];
      key_of(mi, di, (uint32_t)k, key);
      const int full = (k % 16) == 0;
      neb_share_eval(model, &nd, key, rg1, rg2, rd, rt, ri, NULL);
      for (int sparse = 0; sparse < 2; sparse++) {
        uint8_t xd[32], xt[32];
        uint32_t xi[NEB_NB];
        if (full) { memset(xg1, 0x5a, sizeof(int32_t) * NEB_N1); memset(xg2, 0x5a, sizeof(int32_t) * NEB_N2); }
        neb_cpu_share_eval(cm, scr, cd, key, sparse, xd, xt, xi, full ? xg1 : NULL, full ? xg2 : NULL);
        int ok = !memcmp(rd, xd, 32) && !memcmp(rt, xt, 32) && !memcmp(ri, xi, sizeof ri);
        if (full) {
          ok = ok && !memcmp(rg1, xg1, sizeof(int32_t) * NEB_N1) && !memcmp(rg2, xg2, sizeof(int32_t) * NEB_N2);
          gr++;
        }
        ev++;
        if (!ok) bad++;
      }
    }
    neb_cpu_scratch_free(scr);
    free(rg1); free(rg2); free(xg1); free(xg2);
  }
  printf("model %-24s dataset %-19s: %ld keys x 2 variants, %ld differences\n", mname, f->name, keys, bad);
  neb_cpu_model_free(cm);
  neb_cpu_dataset_free(cd);
  *evals += ev;
  *grads += gr;
  return bad;
}

int main(int argc, char **argv) {
  long keys = 2048, real_keys = 100000;
  const char *real = NULL, *model_files[32];
  int nmodels = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--real") && i + 1 < argc) real = argv[++i];
    else if (!strcmp(argv[i], "--model") && i + 1 < argc && nmodels < 32) model_files[nmodels++] = argv[++i];
    else if (!strcmp(argv[i], "--real-keys") && i + 1 < argc) real_keys = atol(argv[++i]);
    else if (argv[i][0] != '-') keys = atol(argv[i]);
    else { fprintf(stderr, "usage: test_shares_avx512 [KEYS] [--real FILE --model FILE ... [--real-keys N]]\n"); return 2; }
  }
  flat_t sets[3];
  make_flat_syn(&sets[0]);
  make_flat_sparse(&sets[1]);
  make_flat_extreme(&sets[2]);
  neb_model *models[DIFF_MODELS];
  diff_make_models(models);
  long total = 0, bad = 0, grads = 0;
  for (int mi = 0; mi < DIFF_MODELS; mi++)
    for (int di = 0; di < 3; di++) bad += run_case(diff_model_names[mi], models[mi], &sets[di], keys, (uint32_t)mi, (uint32_t)di, &total, &grads);
  if (real) {
    flat_t f;
    const int rc = load_flat_file(real, &f);
    if (rc) { fprintf(stderr, "cannot read the dataset %s (%d)\n", real, rc); return 2; }
    neb_model *m = (neb_model *)malloc(sizeof *m);
    for (int i = 0; i < nmodels; i++) {
      if (load_model_file(model_files[i], m)) { fprintf(stderr, "cannot read the model %s\n", model_files[i]); return 2; }
      const char *base = strrchr(model_files[i], '/');
      base = base ? base + 1 : model_files[i];
      bad += run_case(base, m, &f, real_keys, 100 + (uint32_t)i, 100, &total, &grads);
    }
  }
  printf("TOTAL %ld evaluations (%ld with full gradients), %ld differences: %s\n", total, grads, bad, bad ? "FAIL" : "PASS");
  return bad != 0;
}
