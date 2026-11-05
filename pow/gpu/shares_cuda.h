#ifndef NEB_SHARES_CUDA_H
#define NEB_SHARES_CUDA_H
// Training shares on CUDA: device implementation of spec/shares.md sections 5.1 to 5.4 (sample indices, gradient, digest,
// ticket). One thread block evaluates one share. The integer products use the 8-bit dot-product instruction dp4a
// (sm_61 and later); every sum is exact in int32, so the results equal pow/ref/shares.c bit for bit
// (pow/tests/test_shares_cuda.cu checks it). The model step (section 6) is not implemented here: the host applies it
// with the C reference and uploads the new model with neb_share_gpu_set_model. The node and the CPU miner use the C
// reference. A little-endian host is assumed: keys, headers and results are exchanged as uint32 words.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

// Device state; all pointers are device memory.
typedef struct {
  int32_t *w1t;     // W1 in kernel layout: w1t[w * 128 + h] holds the 4 int8 weights W1[h][4w .. 4w+3], column 4w in the low byte
  int8_t *w2;       // 10 x 128, row-major
  uint8_t *data;    // n x 784 pixels, values 0..7
  uint8_t *labels;  // n
  uint32_t n;
} neb_share_gpu;

// Output of one share, as little-endian uint32 words: 8 words of digest, 8 words of ticket, 16 words of sample indices
// (128 bytes). Digest and ticket have the byte order of the C reference.
#define NEB_SHARE_OUT_BYTES 128

// Allocates the device buffers and uploads the dataset (n >= 1) and the model. w1 (128 x 784) and w2 (10 x 128) are
// int8 host arrays, row-major.
cudaError_t neb_share_gpu_init(neb_share_gpu *g, const int8_t *w1, const int8_t *w2, const uint8_t *data, const uint8_t *labels, uint32_t n);
// Replaces the model after a model step. Keeps no state shared with other calls, so devices can be updated from different
// threads (it staged W1 in one static host buffer until the race that this allowed was removed).
cudaError_t neb_share_gpu_set_model(neb_share_gpu *g, const int8_t *w1, const int8_t *w2);
void neb_share_gpu_free(neb_share_gpu *g);

// Evaluates count shares. d_in is device memory with count 32-byte share keys, or with count 80-byte block headers when
// input_is_header is not zero; the key is then BLAKE3(header) (spec/shares.md section 11.1), so a share nonce is
// evaluated by writing it into the header (section 11.2). d_in and d_out must be 4-byte aligned. d_out receives
// count * NEB_SHARE_OUT_BYTES bytes. Asynchronous on the default stream: synchronize before reading d_out. Returns
// cudaErrorInvalidValue for count == 0 or a count above 0x7FFFFFFF (the limit of the grid size).
cudaError_t neb_share_gpu_launch(const neb_share_gpu *g, const uint8_t *d_in, int input_is_header, size_t count, uint8_t *d_out);

#endif
