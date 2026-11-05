#ifndef NEB_SYN_DATASET_H
#define NEB_SYN_DATASET_H
/* The test dataset of spec/shares.md section 9, for the tests and the benchmarks of the training shares. */
#include <stdint.h>
#include "shares.h"

#ifdef __cplusplus
extern "C" {
#endif
#define NEB_SYN_N 2048u
/* Fills x (NEB_NIN values) with sample i and returns its label; ctx is unused. */
int neb_syn_get(void *ctx, uint32_t i, uint8_t x[NEB_NIN]);
#ifdef __cplusplus
}
#endif

#endif
