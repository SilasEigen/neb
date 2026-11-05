// Differential test of the CUDA kernels K0 (dp4a), K1 and K2 (tensor cores) against the C reference: digest, ticket and the
// 16 sample indices in key mode and in header mode, and the full clipped gradients of K1 and K2 on the first keys of each
// case, for the models and datasets of pow/tests/diff_cases.h.
//   test_shares_tc [KEYS_PER_CASE [GRADIENT_KEYS]] [--real DATASET_FILE --model MODEL_FILE ... [--real-keys N]]
//   (defaults 2048 keys and 64 gradient keys; --real adds the network dataset file with each raw model dump, N keys each,
//   default 100000, in batches)
// Exit code 0 when nothing differs. Needs a device of compute capability 8.0 or later for K1 and K2. Built with
// -DNEB_NEGATIVE_CONTROL_TC_CLIP the tensor-core kernels are deliberately wrong and this test must fail. The reference runs
// on all cores when compiled with OpenMP.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
extern "C" {
#include "blake3_min.h"
#include "shares.h"
#include "syn_dataset.h"
}
#include "diff_cases.h"
#include "shares_cuda.h"
#include "shares_tc.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(2); } } while (0)

struct Rec { uint8_t digest[32], ticket[32]; uint32_t idx[NEB_NB]; };

static void ref_records(const neb_model *m, const neb_dataset *nd, const std::vector<uint8_t> &keys, size_t n, std::vector<Rec> &out, std::vector<int16_t> *grads, size_t gkeys) {
  out.resize(n);
  if (grads) grads->assign(gkeys * (NEB_N1 + NEB_N2), 0);
#pragma omp parallel
  {
    std::vector<int32_t> g1(NEB_N1), g2(NEB_N2);
#pragma omp for schedule(dynamic, 8)
    for (long i = 0; i < (long)n; i++) {
      neb_share_eval(m, nd, &keys[32 * i], g1.data(), g2.data(), out[i].digest, out[i].ticket, out[i].idx, NULL);
      if (grads && (size_t)i < gkeys) {
        for (int e = 0; e < NEB_N1; e++) (*grads)[i * (NEB_N1 + NEB_N2) + e] = (int16_t)g1[e];
        for (int e = 0; e < NEB_N2; e++) (*grads)[i * (NEB_N1 + NEB_N2) + NEB_N1 + e] = (int16_t)g2[e];
      }
    }
  }
}

static bool same(const uint8_t *o, const Rec &r) {
  if (memcmp(o, r.digest, 32) || memcmp(o + 32, r.ticket, 32)) return false;
  for (int b = 0; b < NEB_NB; b++) {
    uint32_t v;
    memcpy(&v, o + 64 + 4 * b, 4);
    if (v != r.idx[b]) return false;
  }
  return true;
}

struct Bufs { uint8_t *d_in, *d_out; int16_t *d_dbg; size_t cap, gkeys; };

// One model on one dataset, all kernels; `keys` keys in batches of at most b.cap, a quarter as many headers.
static void run_case(const Bufs &b, bool tc, const char *mname, const neb_model *model, const flat_t *f, size_t keys, uint32_t mi, uint32_t di, long *total, long *bad) {
  const neb_dataset nd = {f->n, flat_get, (void *)f};
  const char *kname[3] = {"K0", "K1", "K2"};
  long case_bad[3] = {0, 0, 0}, case_n[3] = {0, 0, 0};
  neb_share_gpu g0;
  neb_share_tc *t1[3] = {NULL, NULL, NULL};
  CK(neb_share_gpu_init(&g0, model->w1, model->w2, f->px, f->lab, f->n));
  for (int kern = 1; kern < 3 && tc; kern++) CK(neb_share_tc_open(&t1[kern], kern == 2, model->w1, model->w2, f->px, f->lab, f->n));
  std::vector<uint8_t> out(NEB_SHARE_OUT_BYTES * b.cap);
  for (size_t base = 0; base < keys; base += b.cap) {
    const size_t nk = keys - base < b.cap ? keys - base : b.cap, nh = nk / 4;
    std::vector<uint8_t> kb(32 * nk), hb(80 * nh), hk(32 * nh);
    for (size_t k = 0; k < nk; k++) key_of(mi, di, (uint32_t)(base + k), &kb[32 * k]);
    uint64_t st = 0x4E454233ULL + 1000 * mi + di + 77 * base;
    for (size_t i = 0; i < hb.size(); i += 8) { const uint64_t r = rnd_seeded(&st); memcpy(&hb[i], &r, 8); }
    for (size_t i = 0; i < nh; i++) blake3_hash(&hb[80 * i], 80, &hk[32 * i]);
    std::vector<Rec> rk, rh;
    std::vector<int16_t> rgrad;
    const size_t gk = base == 0 ? (b.gkeys < nk ? b.gkeys : nk) : 0;
    ref_records(model, &nd, kb, nk, rk, gk ? &rgrad : NULL, gk);
    ref_records(model, &nd, hk, nh, rh, NULL, 0);
    for (int kern = 0; kern < 3; kern++) {
      if (kern > 0 && !tc) continue;
      for (int mode = 0; mode < 2; mode++) {  // 0: keys, 1: 80-byte headers
        const size_t n = mode ? nh : nk;
        if (n == 0) continue;
        CK(cudaMemcpy(b.d_in, mode ? hb.data() : kb.data(), (mode ? 80 : 32) * n, cudaMemcpyHostToDevice));
        if (kern == 0) CK(neb_share_gpu_launch(&g0, b.d_in, mode, n, b.d_out));
        else CK(neb_share_tc_launch(t1[kern], b.d_in, mode, n, b.d_out));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out.data(), b.d_out, NEB_SHARE_OUT_BYTES * n, cudaMemcpyDeviceToHost));
        const std::vector<Rec> &want = mode ? rh : rk;
        for (size_t i = 0; i < n; i++) { case_n[kern]++; if (!same(&out[NEB_SHARE_OUT_BYTES * i], want[i])) case_bad[kern]++; }
      }
      if (kern > 0 && gk) {
        CK(cudaMemcpy(b.d_in, kb.data(), 32 * gk, cudaMemcpyHostToDevice));
        CK(neb_share_tc_launch_debug(t1[kern], b.d_in, 0, gk, b.d_out, b.d_dbg));
        CK(cudaDeviceSynchronize());
        std::vector<int16_t> got(gk * (NEB_N1 + NEB_N2));
        CK(cudaMemcpy(got.data(), b.d_dbg, got.size() * sizeof(int16_t), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < gk; i++) {
          case_n[kern]++;
          if (memcmp(&got[i * (NEB_N1 + NEB_N2)], &rgrad[i * (NEB_N1 + NEB_N2)], sizeof(int16_t) * (NEB_N1 + NEB_N2))) case_bad[kern]++;
        }
      }
    }
  }
  for (int kern = 0; kern < 3; kern++) {
    if (kern > 0 && !tc) continue;
    printf("%s model %-24s dataset %-19s: %zu keys + %zu headers%s, %ld differences\n", kname[kern], mname, f->name, keys, keys / 4, kern ? " + gradients" : "", case_bad[kern]);
    *total += case_n[kern];
    *bad += case_bad[kern];
  }
  neb_share_gpu_free(&g0);
  for (int kern = 1; kern < 3; kern++) neb_share_tc_close(t1[kern]);
}

int main(int argc, char **argv) {
  size_t keys = 2048, gkeys = 64, real_keys = 100000;
  const char *real = NULL, *model_files[32];
  int nmodels = 0, npos = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--real") && i + 1 < argc) real = argv[++i];
    else if (!strcmp(argv[i], "--model") && i + 1 < argc && nmodels < 32) model_files[nmodels++] = argv[++i];
    else if (!strcmp(argv[i], "--real-keys") && i + 1 < argc) real_keys = (size_t)atol(argv[++i]);
    else if (argv[i][0] != '-') { if (npos++ == 0) keys = (size_t)atol(argv[i]); else gkeys = (size_t)atol(argv[i]); }
    else { fprintf(stderr, "usage: test_shares_tc [KEYS [GRADIENT_KEYS]] [--real FILE --model FILE ... [--real-keys N]]\n"); return 2; }
  }
  int device = 0;
  CK(cudaGetDevice(&device));
  const bool tc = neb_share_tc_supported(device) != 0;
  if (!tc) printf("SKIP K1 and K2: the device has compute capability below 8.0\n");
  Bufs b;
  b.cap = 65536;
  if (keys > b.cap) b.cap = keys;
  b.gkeys = gkeys;
  CK(cudaMalloc(&b.d_in, 80 * b.cap + 64));
  CK(cudaMalloc(&b.d_out, NEB_SHARE_OUT_BYTES * b.cap));
  CK(cudaMalloc(&b.d_dbg, sizeof(int16_t) * (NEB_N1 + NEB_N2) * (gkeys ? gkeys : 1)));
  flat_t sets[3];
  make_flat_syn(&sets[0]);
  make_flat_sparse(&sets[1]);
  make_flat_extreme(&sets[2]);
  neb_model *models[DIFF_MODELS];
  diff_make_models(models);
  long total = 0, bad = 0;
  for (int mi = 0; mi < DIFF_MODELS; mi++)
    for (int di = 0; di < 3; di++) run_case(b, tc, diff_model_names[mi], models[mi], &sets[di], keys, (uint32_t)mi, (uint32_t)di, &total, &bad);
  if (real) {
    flat_t f;
    const int rc = load_flat_file(real, &f);
    if (rc) { fprintf(stderr, "cannot read the dataset %s (%d)\n", real, rc); return 2; }
    neb_model *m = (neb_model *)malloc(sizeof *m);
    for (int i = 0; i < nmodels; i++) {
      if (load_model_file(model_files[i], m)) { fprintf(stderr, "cannot read the model %s\n", model_files[i]); return 2; }
      const char *base = strrchr(model_files[i], '/');
      base = base ? base + 1 : model_files[i];
      run_case(b, tc, base, m, &f, real_keys, 100 + (uint32_t)i, 100, &total, &bad);
    }
  }
  printf("TOTAL %ld comparisons, %ld differences: %s\n", total, bad, bad ? "FAIL" : "PASS");
  return bad != 0;
}
