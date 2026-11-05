#ifndef NEB_SHARES_TC_H
#define NEB_SHARES_TC_H
// Training shares on CUDA with tensor cores (int8 mma): the kernels K1 (dense) and K2 (sparse) of pow/gpu/shares_tc.cu.
// Same results as pow/gpu/shares_cuda.h (the dp4a kernel K0) and as pow/ref/shares.c: digest, ticket and the 16 sample
// indices as 128-byte little-endian records (NEB_SHARE_OUT_BYTES). Needs a device of compute capability 8.0 or later
// (neb_share_tc_supported); the miner checks the device and runs a self-test against the C reference before it uses them.
// Mining only: the node and the consensus use pow/ref.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct neb_share_tc neb_share_tc;

// 1 when the device has compute capability 8.0 or later (the mma.sync int8 instructions used by the kernels).
int neb_share_tc_supported(int device);

// Allocates the device buffers on the current device and uploads the dataset (n >= 1 samples of 784 pixels and their labels)
// and the model (w1: 128 x 784 and w2: 10 x 128 int8, row-major). sparse != 0 selects K2, otherwise K1.
cudaError_t neb_share_tc_open(neb_share_tc **out, int sparse, const int8_t *w1, const int8_t *w2, const uint8_t *data, const uint8_t *labels, uint32_t n);

// Replaces the model. Uses no state shared with other calls, so devices can be updated from different threads.
cudaError_t neb_share_tc_set_model(neb_share_tc *k, const int8_t *w1, const int8_t *w2);

// Evaluates count attempts. d_in is device memory with count 32-byte keys, or count 80-byte block headers when
// input_is_header != 0 (the key is then BLAKE3(header)). d_out receives count * 128 bytes. Asynchronous on the default
// stream. The first launch of a size allocates the scratch records.
cudaError_t neb_share_tc_launch(neb_share_tc *k, const uint8_t *d_in, int input_is_header, size_t count, uint8_t *d_out);

// As neb_share_tc_launch, also writing the clipped gradients of every attempt to d_gradients: count * 101632 int16 values
// (G1 row-major 128 x 784, then G2 row-major 10 x 128).
cudaError_t neb_share_tc_launch_debug(neb_share_tc *k, const uint8_t *d_in, int input_is_header, size_t count, uint8_t *d_out, int16_t *d_gradients);

void neb_share_tc_close(neb_share_tc *k);

#ifdef __cplusplus
}
#endif

#endif
