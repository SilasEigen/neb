#ifndef NEB_SHARES_AVX512_H
#define NEB_SHARES_AVX512_H
/* AVX-512 VNNI evaluation of training shares: a faster path to the same digest, ticket and sample indices as
 * neb_share_eval (pow/ref/shares.h). Mining only: the consensus uses the reference. The caller must have checked
 * that the processor has AVX-512 F, BW, VL, DQ and VNNI before it calls anything but the *_free functions.
 *
 * A dataset and a prepared model are read-only after creation and may be shared by threads; a scratch buffer
 * belongs to one thread. Every constructor returns NULL when memory runs out. */
#include <stdint.h>

#define NEB_NB 16 /* samples per share, must equal pow/ref/shares.h */

typedef struct neb_cpu_dataset neb_cpu_dataset;
typedef struct neb_cpu_model neb_cpu_model;
typedef struct neb_cpu_scratch neb_cpu_scratch;

/* Copies n >= 1 samples: pixels is n x 784 bytes (values 0..7), labels is n bytes (values 0..9). */
neb_cpu_dataset *neb_cpu_dataset_new(const uint8_t *pixels, const uint8_t *labels, uint32_t n);
void neb_cpu_dataset_free(neb_cpu_dataset *d);

/* Repacks the weights: w1 is 128 x 784 and w2 is 10 x 128 int8, row-major, values -8..7. The residuals play no part
 * in a ticket. */
neb_cpu_model *neb_cpu_model_new(const int8_t *w1, const int8_t *w2);
void neb_cpu_model_free(neb_cpu_model *m);

neb_cpu_scratch *neb_cpu_scratch_new(void);
void neb_cpu_scratch_free(neb_cpu_scratch *s);

/* One share: digest and ticket (32 bytes each, the byte order of the reference) and the 16 sample indices. sparse != 0
 * selects the sparse variant. g1 (128 x 784) and g2 (10 x 128), when not NULL, receive the clipped gradients. */
void neb_cpu_share_eval(const neb_cpu_model *m, neb_cpu_scratch *s, const neb_cpu_dataset *d, const uint8_t key[32], int sparse,
                        uint8_t digest[32], uint8_t ticket[32], uint32_t idx[NEB_NB], int32_t *g1, int32_t *g2);

#endif
