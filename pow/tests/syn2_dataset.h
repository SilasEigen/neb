#ifndef NEB_SYN2_DATASET_H
#define NEB_SYN2_DATASET_H
/* The test datasets of spec/nebhash-v2.md section 8, one per architecture, for the vectors and the tests. */
#include <stdint.h>
#include "nebhash2.h"

#ifdef __cplusplus
extern "C" {
#endif
#define NEB_SYN2_N 2048u
/* ctx points to an int holding the architecture id. Fills x (the architecture's nin values) with sample i and returns
 * its label i mod 2. */
int neb_syn2_get(void *ctx, uint32_t i, uint8_t *x);
#ifdef __cplusplus
}
#endif

#endif
