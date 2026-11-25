#ifndef NEB_NEBHASH3_H
#define NEB_NEBHASH3_H
/* nebhash v3: frozen quantized INT8 base backbones with on-chain integer LoRA adapters and key-derived deterministic
 * dropout. Normative text: spec/nebhash-v3.md. Everything not redefined here follows shares.h (clipping, targets,
 * ticket, model step rule, state hashes). */
#include <stddef.h>
#include <stdint.h>
#include "shares.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NEB3_LORA_TINY 1      /* 64-dim, rank 8, 2 blocks: CI & deterministic test vectors */
#define NEB3_QWEN3_8B_LORA 2  /* 4096-dim, rank 8, 4 blocks: Apache-2.0 Qwen3-8B INT8 frozen base profile */

#define NEB3_NB 32            /* batch size per share, aligned to sm_80+ m16 INT8 Tensor Core tiles */
#define NEB3_NOUT 2

#ifndef NEB3_SU_DROP
#define NEB3_SU_DROP 0
#endif

typedef struct {
  int id;
  int d;          /* hidden dimension D (input/output width of each frozen block) */
  int rank;       /* LoRA bottleneck rank R */
  int nblocks;    /* number of LoRA-adapted frozen blocks L */
  int nout;       /* output head classes (2) */
  int s_a, s_lora, s_out;
  int sb_b, sb_in;
  int s_head, sb_head;
  int su_lora, su_head;
  /* derived counts */
  size_t nweights;      /* trainable weights: L * 2 * D * R + NOUT * D */
  size_t base_weights;  /* frozen base weights: L * D * D */
} neb3_arch;

const neb3_arch *neb3_arch_get(int id);

/* Trainable LoRA + head state allocated on the heap (supports both TINY and 8B profiles cleanly). */
typedef struct {
  int32_t arch;
  size_t nweights;
  int8_t *w;   /* [nweights] in -8..7: for l=0..L-1: A[l] (R*D), B[l] (D*R), then H (NOUT*D) */
  int32_t *r;  /* [nweights] residuals */
} neb3_model;

/* Frozen base backbone view: w_base holds L * D * D int8 entries in -8..7 (may be mmapped on mainnet). */
typedef struct {
  int32_t arch;
  size_t nweights;
  const int8_t *w_base;
} neb3_base;

typedef struct {
  uint32_t n;
  int (*get)(void *ctx, uint32_t index, uint8_t *x);
  void *ctx;
} neb3_dataset;

/* Allocate and initialize trainable weights from a 32-byte seed; residuals start at 0. Returns 0 or -1. */
int neb3_model_init(int arch, const uint8_t seed[32], neb3_model *m);
void neb3_model_free(neb3_model *m);

/* Base weights commitment: BLAKE3("neb-base-weights" || pack(W_base)). Returns 0 or -1. */
int neb3_base_hash(const neb3_base *base, uint8_t out_hash[32]);

/* Sample indices for a v3 share (NEB3_NB = 32). */
void neb3_share_indices(const uint8_t key[32], uint32_t n, uint32_t idx[NEB3_NB]);

/* Evaluate one v3 share: clipped trainable gradients g[m->nweights], 32-byte digest and ticket.
 * idx_out and nclip may be NULL. Returns 0 or -1. */
int neb3_share_eval(const neb3_model *m, const neb3_base *base, const neb3_dataset *ds, const uint8_t key[32],
                    int32_t *g, uint8_t digest[32], uint8_t ticket[32], uint32_t idx_out[NEB3_NB], uint32_t *nclip);

/* Apply summed clipped gradients gs[m->nweights] to trainable LoRA + head weights. */
void neb3_model_apply(neb3_model *m, const int32_t *gs);

/* State hashes of trainable weights and residuals. Returns 0 or -1 on OOM. */
int neb3_model_hashes(const neb3_model *m, uint8_t weights_hash[32], uint8_t residual_hash[32]);

#ifdef __cplusplus
}
#endif

#endif
