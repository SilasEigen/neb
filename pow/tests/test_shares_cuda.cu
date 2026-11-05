// Checks the CUDA training shares against the C reference and the vector file (spec/shares.md section 10).
//   test_shares_cuda vectors/shares.txt
// For every share of every step the GPU result (sample indices, digest, ticket) must equal the C reference and the line of the
// file; the model is advanced with the C reference. Then 2000 block headers are evaluated in header mode against the C reference.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <map>
extern "C" {
#include "blake3_min.h"
#include "shares.h"
#include "syn_dataset.h"
}
#include "shares_cuda.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)

static std::string hexs(const uint8_t *b, int n) { static const char *d = "0123456789abcdef"; std::string s; for (int i = 0; i < n; i++) { s += d[b[i] >> 4]; s += d[b[i] & 15]; } return s; }
static void put_le32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static uint32_t get_le32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

int main(int argc, char **argv) {
  if (argc != 2) { printf("usage: test_shares_cuda <vector file>\n"); return 2; }
  FILE *f = fopen(argv[1], "rb");
  if (!f) { printf("cannot open %s\n", argv[1]); return 2; }
  std::map<std::string, std::string> share_line, state_line;  // "s j" -> "digest ticket idx...", "s" -> "weights residual"
  char line[4096];
  while (fgets(line, sizeof line, f)) {
    std::string l(line);
    while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
    if (l.rfind("share ", 0) == 0) { size_t a = l.find(' ', 6), b = l.find(' ', a + 1); share_line[l.substr(6, b - 6)] = l.substr(b + 1); }
    else if (l.rfind("state ", 0) == 0) { size_t a = l.find(' ', 6), b = l.find(' ', a + 1); state_line[l.substr(6, a - 6)] = l.substr(b + 1); }
  }
  fclose(f);

  const neb_dataset ds = {NEB_SYN_N, neb_syn_get, NULL};
  std::vector<uint8_t> data((size_t)ds.n * NEB_NIN), labels(ds.n);
  for (uint32_t i = 0; i < ds.n; i++) labels[i] = (uint8_t)neb_syn_get(NULL, i, &data[(size_t)i * NEB_NIN]);
  neb_model *m = (neb_model *)malloc(sizeof *m);
  uint8_t seed[32] = {0};
  neb_model_init(seed, m);
  neb_share_gpu gpu;
  CK(neb_share_gpu_init(&gpu, m->w1, m->w2, data.data(), labels.data(), ds.n));
  // buffers for up to 4096 shares or headers per launch (a step has at most 32 shares, the header test 2000)
  uint8_t *d_in, *d_out;
  CK(cudaMalloc(&d_in, 80 * 4096));
  CK(cudaMalloc(&d_out, NEB_SHARE_OUT_BYTES * 4096));
  std::vector<int32_t> g1(NEB_N1), g2(NEB_N2), s1(NEB_N1), s2(NEB_N2);
  std::vector<uint8_t> out(NEB_SHARE_OUT_BYTES * 4096);
  static const int counts[8] = {32, 32, 32, 1, 32, 3, 32, 17};
  long mism = 0, shares = 0;
  for (int s = 0; s < 24; s++) {
    const int n = counts[s % 8];
    std::vector<uint8_t> keys(32 * n);
    for (int j = 0; j < n; j++) {
      uint8_t in[22];
      memcpy(in, "neb-test-share", 14); put_le32(in + 14, (uint32_t)s); put_le32(in + 18, (uint32_t)j);
      blake3_hash(in, sizeof in, &keys[32 * j]);
    }
    // the kernel does not apply model steps: upload the state that the C reference left
    CK(neb_share_gpu_set_model(&gpu, m->w1, m->w2));
    CK(cudaMemcpy(d_in, keys.data(), 32 * n, cudaMemcpyHostToDevice));
    CK(neb_share_gpu_launch(&gpu, d_in, 0, (size_t)n, d_out));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data(), d_out, (size_t)NEB_SHARE_OUT_BYTES * n, cudaMemcpyDeviceToHost));
    memset(s1.data(), 0, 4 * NEB_N1); memset(s2.data(), 0, 4 * NEB_N2);
    for (int j = 0; j < n; j++) {
      uint8_t digest[32], ticket[32];
      uint32_t idx[NEB_NB];
      neb_share_eval(m, &ds, &keys[32 * j], g1.data(), g2.data(), digest, ticket, idx, NULL);
      for (int e = 0; e < NEB_N1; e++) s1[e] += g1[e];
      for (int e = 0; e < NEB_N2; e++) s2[e] += g2[e];
      const uint8_t *o = &out[(size_t)NEB_SHARE_OUT_BYTES * j];
      std::string got = hexs(o, 32) + " " + hexs(o + 32, 32), want = hexs(digest, 32) + " " + hexs(ticket, 32);
      for (int b = 0; b < NEB_NB; b++) { got += " " + std::to_string(get_le32(o + 64 + 4 * b)); want += " " + std::to_string(idx[b]); }
      // a mismatch against the C reference points to the kernel; one against the file only, to the C reference or the file
      const std::string key = std::to_string(s) + " " + std::to_string(j);
      shares++;
      if (got != want) { mism++; if (mism <= 5) printf("MISMATCH vs C reference at step %d share %d\n", s, j); }
      if (got != share_line[key]) { mism++; if (mism <= 5) printf("MISMATCH vs vector file at step %d share %d\n", s, j); }
    }
    neb_model_apply(m, s1.data(), s2.data());
    uint8_t wh[32], rh[32];
    neb_model_hashes(m, wh, rh);
    if (hexs(wh, 32) + " " + hexs(rh, 32) != state_line[std::to_string(s)]) { mism++; printf("MISMATCH state after step %d\n", s); }
  }

  // header mode: the key is BLAKE3 of an 80-byte header, with the model left by the 24 steps
  const int NH_TEST = 2000;
  std::vector<uint8_t> hdr((size_t)80 * NH_TEST);
  uint64_t st = 0x4E454233ULL;
  for (size_t i = 0; i < hdr.size(); i += 8) { const uint64_t r = splitmix64(&st); memcpy(&hdr[i], &r, 8); }
  CK(neb_share_gpu_set_model(&gpu, m->w1, m->w2));
  CK(cudaMemcpy(d_in, hdr.data(), hdr.size(), cudaMemcpyHostToDevice));
  CK(neb_share_gpu_launch(&gpu, d_in, 1, NH_TEST, d_out));
  CK(cudaDeviceSynchronize());
  std::vector<uint8_t> out2((size_t)NEB_SHARE_OUT_BYTES * NH_TEST);
  CK(cudaMemcpy(out2.data(), d_out, out2.size(), cudaMemcpyDeviceToHost));
  // the GPU hashes the header itself; the host computes BLAKE3 of the same 80 bytes
  long hmism = 0;
  for (int i = 0; i < NH_TEST; i++) {
    uint8_t key[32], digest[32], ticket[32];
    uint32_t idx[NEB_NB];
    blake3_hash(&hdr[(size_t)80 * i], 80, key);
    neb_share_eval(m, &ds, key, g1.data(), g2.data(), digest, ticket, idx, NULL);
    const uint8_t *o = &out2[(size_t)NEB_SHARE_OUT_BYTES * i];
    bool ok = memcmp(o, digest, 32) == 0 && memcmp(o + 32, ticket, 32) == 0;
    for (int b = 0; b < NEB_NB; b++) ok = ok && get_le32(o + 64 + 4 * b) == idx[b];
    if (!ok) { hmism++; if (hmism <= 5) printf("MISMATCH header %d\n", i); }
  }
  printf("shares_cuda check file=%s steps=24 shares=%ld mismatches=%ld header_shares=%d header_mismatches=%ld %s\n",
         argv[1], shares, mism, NH_TEST, hmism, (mism == 0 && hmism == 0) ? "OK" : "MISMATCH");
  return (mism == 0 && hmism == 0) ? 0 : 1;
}
