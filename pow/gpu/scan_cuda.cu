// Nonce scan on CUDA: the contract is in scan_cuda.h. The attempts are evaluated by shares_cuda.cu; this file builds the
// headers on the device and keeps only the nonces whose ticket is not above the bound, so that the host reads a few words
// instead of 128 bytes per attempt.
#include "scan_cuda.h"
#include "shares_cuda.h"
#include "shares_tc.h"
#include <stdlib.h>
#include <string.h>

struct neb_scan {
  int device;
  int kernel;
  neb_share_gpu gpu;     // K0
  neb_share_tc *tc;      // K1 or K2
  uint32_t capacity, max_hits;
  uint8_t *d_headers, *d_out;
  uint32_t *d_hits;  // d_hits[0] counts the hits, the nonces follow
};

struct Header { uint32_t w[20]; };
struct Bound { uint32_t w[8]; };

__global__ void fill_headers(uint32_t *out, Header h, uint32_t start, uint32_t count) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  uint32_t *o = out + (size_t)i * 20;
#pragma unroll
  for (int k = 0; k < 19; k++) o[k] = h.w[k];
  o[19] = start + i;
}

// The ticket is words 8..15 of the output (shares_cuda.h) and is compared with the bound as a 256-bit little-endian
// integer: word 7 is the most significant.
__global__ void select_hits(const uint8_t *out, uint32_t count, uint32_t start, Bound b, uint32_t *hits, uint32_t max_hits) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t *t = (const uint32_t *)(out + (size_t)i * NEB_SHARE_OUT_BYTES) + 8;
  bool ok = true;
  for (int k = 7; k >= 0; k--) {
    if (t[k] != b.w[k]) { ok = t[k] < b.w[k]; break; }
  }
  if (!ok) return;
  const uint32_t at = atomicAdd(hits, 1u);
  if (at < max_hits) hits[1 + at] = start + i;
}

extern "C" {

int neb_scan_device_count(int *count) { return (int)cudaGetDeviceCount(count); }

int neb_scan_device_name(int device, char *name, int len) {
  cudaDeviceProp p;
  const cudaError_t e = cudaGetDeviceProperties(&p, device);
  if (e != cudaSuccess) return (int)e;
  int i = 0;
  for (; i + 1 < len && p.name[i]; i++) name[i] = p.name[i];
  name[i] = 0;
  return 0;
}

int neb_scan_device_bus(int device, int *bus) { return (int)cudaDeviceGetAttribute(bus, cudaDevAttrPciBusId, device); }

int neb_scan_device_capability(int device, int *major, int *minor) {
  const cudaError_t e = cudaDeviceGetAttribute(major, cudaDevAttrComputeCapabilityMajor, device);
  if (e != cudaSuccess) return (int)e;
  return (int)cudaDeviceGetAttribute(minor, cudaDevAttrComputeCapabilityMinor, device);
}

int neb_scan_kernel_supported(int device, int kernel) {
  if (kernel == NEB_SCAN_KERNEL_K0) return 1;
  if (kernel == NEB_SCAN_KERNEL_K1 || kernel == NEB_SCAN_KERNEL_K2) return neb_share_tc_supported(device);
  return 0;
}

#define TRY(x) do { const cudaError_t e_ = (x); if (e_ != cudaSuccess) { return (int)e_; } } while (0)

int neb_scan_open_kernel(int device, int kernel, const uint8_t *pixels, const uint8_t *labels, uint32_t n, const int8_t *w1, const int8_t *w2,
                         uint32_t capacity, uint32_t max_hits, neb_scan **out) {
  if (capacity == 0 || max_hits == 0 || !neb_scan_kernel_supported(device, kernel)) return (int)cudaErrorInvalidValue;
  neb_scan *s = (neb_scan *)calloc(1, sizeof *s);
  if (!s) return (int)cudaErrorMemoryAllocation;
  s->device = device;
  s->kernel = kernel;
  s->capacity = capacity;
  s->max_hits = max_hits;
  cudaError_t e = cudaSetDevice(device);
  if (e == cudaSuccess) {
    if (kernel == NEB_SCAN_KERNEL_K0) e = neb_share_gpu_init(&s->gpu, w1, w2, pixels, labels, n);
    else e = neb_share_tc_open(&s->tc, kernel == NEB_SCAN_KERNEL_K2, w1, w2, pixels, labels, n);
  }
  if (e == cudaSuccess) e = cudaMalloc(&s->d_headers, (size_t)capacity * 80);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_out, (size_t)capacity * NEB_SHARE_OUT_BYTES);
  if (e == cudaSuccess) e = cudaMalloc(&s->d_hits, ((size_t)max_hits + 1) * 4);
  if (e != cudaSuccess) {
    neb_scan_close(s);
    return (int)e;
  }
  *out = s;
  return 0;
}

int neb_scan_open(int device, const uint8_t *pixels, const uint8_t *labels, uint32_t n, const int8_t *w1, const int8_t *w2,
                  uint32_t capacity, uint32_t max_hits, neb_scan **out) {
  return neb_scan_open_kernel(device, NEB_SCAN_KERNEL_K0, pixels, labels, n, w1, w2, capacity, max_hits, out);
}

int neb_scan_set_model(neb_scan *s, const int8_t *w1, const int8_t *w2) {
  TRY(cudaSetDevice(s->device));
  if (s->kernel == NEB_SCAN_KERNEL_K0) return (int)neb_share_gpu_set_model(&s->gpu, w1, w2);
  return (int)neb_share_tc_set_model(s->tc, w1, w2);
}

int neb_scan_run(neb_scan *s, const uint8_t *header, uint32_t start, uint32_t count, const uint8_t *bound, uint32_t *hits,
                 uint32_t *nhits) {
  if (count == 0 || count > s->capacity) return (int)cudaErrorInvalidValue;
  TRY(cudaSetDevice(s->device));
  Header h;
  Bound b;
  memcpy(h.w, header, 80);
  memcpy(b.w, bound, 32);
  const int threads = 256;
  const int blocks = (int)((count + threads - 1) / threads);
  TRY(cudaMemsetAsync(s->d_hits, 0, 4));
  fill_headers<<<blocks, threads>>>((uint32_t *)s->d_headers, h, start, count);
  TRY(cudaGetLastError());
  if (s->kernel == NEB_SCAN_KERNEL_K0) TRY(neb_share_gpu_launch(&s->gpu, s->d_headers, 1, count, s->d_out));
  else TRY(neb_share_tc_launch(s->tc, s->d_headers, 1, count, s->d_out));
  select_hits<<<blocks, threads>>>(s->d_out, count, start, b, s->d_hits, s->max_hits);
  TRY(cudaGetLastError());
  uint32_t n = 0;
  TRY(cudaMemcpy(&n, s->d_hits, 4, cudaMemcpyDeviceToHost));
  *nhits = n;
  if (n > 0) TRY(cudaMemcpy(hits, s->d_hits + 1, (size_t)(n < s->max_hits ? n : s->max_hits) * 4, cudaMemcpyDeviceToHost));
  return 0;
}

void neb_scan_close(neb_scan *s) {
  if (!s) return;
  cudaSetDevice(s->device);
  cudaFree(s->d_headers);
  cudaFree(s->d_out);
  cudaFree(s->d_hits);
  if (s->kernel == NEB_SCAN_KERNEL_K0) neb_share_gpu_free(&s->gpu);
  else neb_share_tc_close(s->tc);
  free(s);
}

const char *neb_scan_error(int status) { return cudaGetErrorString((cudaError_t)status); }

}  // extern "C"
