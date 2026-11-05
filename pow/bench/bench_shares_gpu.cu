// Throughput of the training shares on CUDA, in header mode (a full attempt: key from the header, sample indices, gradient,
// digest, ticket). Usage: bench_shares_gpu [seconds] [attempts per launch]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
extern "C" {
#include "shares.h"
#include "syn_dataset.h"
}
#include "shares_cuda.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)

static uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

int main(int argc, char **argv) {
  const double secs = argc > 1 ? atof(argv[1]) : 4.0;
  const size_t N = argc > 2 ? (size_t)atol(argv[2]) : 16384;
  std::vector<uint8_t> data((size_t)NEB_SYN_N * NEB_NIN), labels(NEB_SYN_N);
  for (uint32_t i = 0; i < NEB_SYN_N; i++) labels[i] = (uint8_t)neb_syn_get(NULL, i, &data[(size_t)i * NEB_NIN]);
  neb_model *m = (neb_model *)malloc(sizeof *m);
  uint8_t seed[32] = {0};
  neb_model_init(seed, m);
  neb_share_gpu gpu;
  CK(neb_share_gpu_init(&gpu, m->w1, m->w2, data.data(), labels.data(), NEB_SYN_N));
  std::vector<uint8_t> h(N * 80);
  uint64_t st = 0x4E454234ULL;
  for (size_t i = 0; i < h.size(); i += 8) { const uint64_t r = splitmix64(&st); memcpy(&h[i], &r, 8); }
  uint8_t *d_in, *d_out;
  CK(cudaMalloc(&d_in, h.size()));
  CK(cudaMalloc(&d_out, N * NEB_SHARE_OUT_BYTES));
  CK(cudaMemcpy(d_in, h.data(), h.size(), cudaMemcpyHostToDevice));
  CK(neb_share_gpu_launch(&gpu, d_in, 1, N, d_out));  // warm-up
  CK(cudaDeviceSynchronize());
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a);
  long total = 0;
  float ms = 0;
  while (ms < secs * 1000) {
    CK(neb_share_gpu_launch(&gpu, d_in, 1, N, d_out));
    total += (long)N;
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms, a, b);
  }
  CK(cudaGetLastError());
  const double rate = total / (ms / 1000.0);
  // one attempt is about 3.3e6 multiply-accumulates: 2 * 16 * 128 * 784 for the layer-1 forward product and G1, plus
  // 3 * 16 * 1280 for the layer-2 product, m and G2
  printf("training shares on CUDA: %.0f attempts/s (%.2f s, %zu per launch), about %.2f TMAC/s of products\n", rate, ms / 1000.0, N, rate * 3.3e6 / 1e12);
  return 0;
}
