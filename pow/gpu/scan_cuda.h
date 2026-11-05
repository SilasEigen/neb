#ifndef NEB_SCAN_CUDA_H
#define NEB_SCAN_CUDA_H
// Nonce scan of one reference header on a CUDA device (spec/shares.md section 11.5): the attempts of shares_cuda.h for
// the headers that differ from the reference header in the nonce field only, and the nonces whose ticket is not above a
// bound. The status values are cudaError_t codes; 0 is success. A handle belongs to one device and must be used from one
// thread at a time.
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct neb_scan neb_scan;

int neb_scan_device_count(int *count);
// Writes the name of a device as a zero-terminated string of at most len bytes.
int neb_scan_device_name(int device, char *name, int len);
// The PCI bus number of a device, as in the first byte of `lspci` (0a:00.0 is 10).
int neb_scan_device_bus(int device, int *bus);
// The compute capability of a device (8.6 is major 8, minor 6).
int neb_scan_device_capability(int device, int *major, int *minor);
// Kernels: 0 is K0 (dp4a, pow/gpu/shares_cuda.cu), 1 is K1 and 2 is K2 (tensor cores, pow/gpu/shares_tc.cu, compute
// capability 8.0 or later). All give the same results as the C reference; the miner checks them against it.
#define NEB_SCAN_KERNEL_K0 0
#define NEB_SCAN_KERNEL_K1 1
#define NEB_SCAN_KERNEL_K2 2
// 1 when the kernel can run on the device.
int neb_scan_kernel_supported(int device, int kernel);
// Opens a handle on a device. pixels is n x 784 values 0..7, labels n values 0..9, w1 is 128 x 784 and w2 10 x 128
// (int8, row-major). One scan covers at most capacity nonces and reports at most max_hits of them. neb_scan_open uses K0.
int neb_scan_open(int device, const uint8_t *pixels, const uint8_t *labels, uint32_t n, const int8_t *w1, const int8_t *w2,
                  uint32_t capacity, uint32_t max_hits, neb_scan **out);
int neb_scan_open_kernel(int device, int kernel, const uint8_t *pixels, const uint8_t *labels, uint32_t n, const int8_t *w1,
                         const int8_t *w2, uint32_t capacity, uint32_t max_hits, neb_scan **out);
// Replaces the model after a model step.
int neb_scan_set_model(neb_scan *s, const int8_t *w1, const int8_t *w2);
// Evaluates the nonces start .. start + count - 1 of the 80-byte header (the nonce field, bytes 76..80, is replaced by the
// nonce). bound is a 256-bit little-endian value as 32 bytes. hits receives, in no particular order, the nonces whose
// ticket is not above the bound, at most max_hits of them; *nhits is the number of such nonces, which can exceed max_hits
// (the caller must then scan a smaller range: a result that is cut is not a complete result).
int neb_scan_run(neb_scan *s, const uint8_t *header, uint32_t start, uint32_t count, const uint8_t *bound, uint32_t *hits,
                 uint32_t *nhits);
void neb_scan_close(neb_scan *s);
const char *neb_scan_error(int status);

#ifdef __cplusplus
}
#endif
#endif
