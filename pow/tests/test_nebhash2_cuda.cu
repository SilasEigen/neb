// Checks the nebhash v2 CUDA scan (pow/gpu/nebhash2_cuda.cu) against the C reference (pow/ref/nebhash2.c): for each
// architecture, on the test dataset of spec/nebhash-v2.md section 8 and on several models (the initial state, a trained
// state and three with extreme weights), the nonces whose ticket is not above a bound must be the same, for bounds taken
// at several ranks of the reference tickets and for each exact ticket of a few nonces (that ticket is a hit, the value
// below it is not). Then prints the attempts per second. Exit status 0 when every case passes.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <set>
#include <vector>
extern "C" {
#include "blake3_min.h"
}
#include "nebhash2.h"
#include "nebhash2_cuda.h"
#include "syn2_dataset.h"

#define NONCES 384

static void ticket_of(const neb2_model *m, const neb2_dataset *ds, const uint8_t header[80], uint32_t nonce, uint8_t t[32],
                      int32_t *g) {
  uint8_t h[80], key[32], d[32];
  memcpy(h, header, 80);
  memcpy(h + 76, &nonce, 4);
  blake3_hash(h, 80, key);
  neb2_share_eval(m, ds, key, g, d, t, NULL, NULL);
}

// a <= b as 256-bit little-endian integers
static bool le256(const uint8_t *a, const uint8_t *b) {
  for (int i = 31; i >= 0; i--)
    if (a[i] != b[i]) return a[i] < b[i];
  return true;
}

static int check(int arch, const char *name, const neb2_model *m, const neb2_dataset *ds, const std::vector<uint8_t> &values,
                 const std::vector<uint8_t> &labels) {
  const neb2_arch *A = neb2_arch_get(arch);
  uint8_t header[80];
  for (int i = 0; i < 80; i++) header[i] = (uint8_t)(i * 37 + 11 + arch);
  std::vector<int32_t> g(NEB2_MAX_WEIGHTS);
  std::vector<std::vector<uint8_t>> tickets(NONCES, std::vector<uint8_t>(32));
  for (uint32_t k = 0; k < NONCES; k++) ticket_of(m, ds, header, k, tickets[k].data(), g.data());
  neb2_scan *s = NULL;
  int st = neb2_scan_open(0, arch, values.data(), labels.data(), (uint32_t)labels.size(), m->w, NONCES, NONCES, &s);
  if (st) { printf("FAIL open %d\n", st); return 1; }
  std::vector<uint32_t> hits(NONCES);
  auto run = [&](const uint8_t *bound, std::set<uint32_t> &got) {
    uint32_t n = 0;
    const int e = neb2_scan_run(s, header, 0, NONCES, bound, hits.data(), &n);
    got.clear();
    for (uint32_t i = 0; i < n && i < NONCES; i++) got.insert(hits[i]);
    return e;
  };
  std::vector<std::vector<uint8_t>> sorted = tickets;
  std::sort(sorted.begin(), sorted.end(), [](const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
    return le256(a.data(), b.data()) && a != b;
  });
  int fails = 0;
  for (int rank : {0, 3, 50, 191, NONCES - 1}) {
    std::set<uint32_t> want, got;
    for (uint32_t k = 0; k < NONCES; k++)
      if (le256(tickets[k].data(), sorted[rank].data())) want.insert(k);
    if (run(sorted[rank].data(), got) || got != want) {
      printf("FAIL arch %d %s: bound at rank %d, %zu hits instead of %zu\n", arch, name, rank, got.size(), want.size());
      fails++;
    }
  }
  for (uint32_t k = 0; k < 8; k++) {
    std::set<uint32_t> got;
    uint8_t below[32];
    memcpy(below, tickets[k].data(), 32);
    int i = 0;
    while (i < 32 && below[i] == 0) below[i++] = 0xFF;  // ticket - 1
    if (i < 32) below[i]--;
    run(tickets[k].data(), got);
    const bool hit = got.count(k) > 0;
    run(below, got);
    if (!hit || got.count(k)) {
      printf("FAIL arch %d %s: exact ticket of nonce %u\n", arch, name, k);
      fails++;
    }
  }
  if (!fails) printf("PASS arch %d %s: %d nonces, 5 ranks and 8 exact tickets equal to the reference\n", arch, name, NONCES);
  neb2_scan_close(s);
  (void)A;
  return fails;
}

int main(int argc, char **argv) {
  int fails = 0;
  for (int arch = NEB2_KEP1; arch <= NEB2_RET2; arch++) {
    const neb2_arch *A = neb2_arch_get(arch);
    const neb2_dataset ds = {NEB_SYN2_N, neb_syn2_get, &arch};
    std::vector<uint8_t> values((size_t)NEB_SYN2_N * A->nin), labels(NEB_SYN2_N);
    for (uint32_t i = 0; i < NEB_SYN2_N; i++) labels[i] = (uint8_t)neb_syn2_get(&arch, i, values.data() + (size_t)i * A->nin);
    neb2_model *m = (neb2_model *)malloc(sizeof *m);
    uint8_t seed[32] = {9};
    neb2_model_init(arch, seed, m);
    fails += check(arch, "initial", m, &ds, values, labels);
    // a trained state: 4 steps of 32 shares
    std::vector<int32_t> g(NEB2_MAX_WEIGHTS), sum(NEB2_MAX_WEIGHTS);
    for (int st = 0; st < 4; st++) {
      std::fill(sum.begin(), sum.end(), 0);
      for (int j = 0; j < 32; j++) {
        uint8_t key[32] = {(uint8_t)st, (uint8_t)j, 7}, d[32], t[32];
        neb2_share_eval(m, &ds, key, g.data(), d, t, NULL, NULL);
        for (int f = 0; f < A->nweights; f++) sum[f] += g[f];
      }
      neb2_model_apply(m, sum.data());
    }
    fails += check(arch, "trained", m, &ds, values, labels);
    const int8_t extremes[3] = {7, -8, 0};
    const char *names[3] = {"all 7", "all -8", "alternating"};
    for (int x = 0; x < 3; x++) {
      for (int f = 0; f < A->nweights; f++) m->w[f] = x < 2 ? extremes[x] : (int8_t)((f % 2) ? 7 : -8);
      fails += check(arch, names[x], m, &ds, values, labels);
    }
    // throughput on the initial state
    neb2_model_init(arch, seed, m);
    neb2_scan *s = NULL;
    const uint32_t count = argc > 1 ? (uint32_t)atoi(argv[1]) : 8192;
    if (neb2_scan_open(0, arch, values.data(), labels.data(), NEB_SYN2_N, m->w, count, 16, &s) == 0) {
      uint8_t header[80] = {1}, bound[32] = {0};
      uint32_t hits[16], n = 0;
      neb2_scan_run(s, header, 0, 1024, bound, hits, &n);
      const auto t0 = std::chrono::steady_clock::now();
      neb2_scan_run(s, header, 1024, count, bound, hits, &n);
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      printf("arch %d: %.0f attempts/s\n", arch, count / sec);
      neb2_scan_close(s);
    }
    free(m);
  }
  printf(fails ? "NEBHASH2 CUDA CHECK FAILED\n" : "NEBHASH2 CUDA CHECK PASSED\n");
  return fails ? 1 : 0;
}
