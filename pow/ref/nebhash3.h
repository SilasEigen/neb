#ifndef NEB_NEBHASH3_H
#define NEB_NEBHASH3_H
/* nebhash v3: anti-shortcut MoE frozen INT8 backbones with on-chain integer LoRA adapters, residual blocks,
 * full-dimension input + bottleneck dropout, and exact int32 forward trace commitment.
 * Normative text: spec/nebhash-v3.md. */
#include <stddef.h>
#include <stdint.h>
#include "shares.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NEB3_LORA_TINY 1      /* D=64, R=16, L=3, E=16 (64 KiB base): CI & deterministic test vectors */
#define NEB3_QWEN3_8B_LORA 2  /* D=1024, R=16, L=6, E=10752 (10.50 GiB base): Qwen3-8B mainnet profile */

#define NEB3_NB 64            /* batch size per share, aligned to sm_80+ m16 INT8 Tensor Core tiles */
#define NEB3_CLIP 1024        /* gradient clip bound scaled 4x for NB3 = 64 */
#define NEB3_NOUT 2
#define NEB3_MAX_BLOCKS 8

#ifndef NEB3_SU_DROP
#define NEB3_SU_DROP 0
#endif

typedef struct {
  int id;
  int d;          /* hidden dimension D */
  int rank;       /* LoRA bottleneck rank R (16) */
  int nblocks;    /* number of residual MoE+LoRA blocks L */
  int nexperts;   /* number of frozen D x D full-range int8 experts E */
  int nout;       /* output head classes (2) */
  int s_a, s_lora, s_out;
  int sb_b, sb_in;
  int s_head, sb_head;
  int su_lora, su_head;
  size_t nweights;      /* trainable weights: L * 2 * D * R + NOUT * D */
  size_t base_weights;  /* frozen full-int8 weights: E * D * D */
} neb3_arch;

const neb3_arch *neb3_arch_get(int id);

typedef struct {
  int32_t arch;
  size_t nweights;
  int8_t *w;   /* [nweights] in -8..7: A[0], B[0], ..., A[L-1], B[L-1], H */
  int32_t *r;  /* [nweights] residuals */
} neb3_model;

typedef struct {
  int32_t arch;
  size_t nweights;
  const int8_t *w_base; /* [E * D * D] full-range int8 in -128..127 */
} neb3_base;

typedef struct {
  uint32_t n;
  int (*get)(void *ctx, uint32_t index, uint8_t *x);
  void *ctx;
} neb3_dataset;

int neb3_model_init(int arch, const uint8_t seed[32], neb3_model *m);
void neb3_model_free(neb3_model *m);

/* Base weights commitment: BLAKE3("neb-base-weights" || raw_int8_bytes(W_base)). */
int neb3_base_hash(const neb3_base *base, uint8_t out_hash[32]);

void neb3_share_indices(const uint8_t key[32], uint32_t n, uint32_t idx[NEB3_NB]);

/* Evaluate one v3 share: clipped trainable gradients g[m->nweights], 32-byte combined digest
 * BLAKE3(grad_digest || trace_digest), and 32-byte ticket BLAKE3(digest || key).
 * idx_out, experts_out, and nclip may be NULL. Returns 0 or -1. */
int neb3_share_eval(const neb3_model *m, const neb3_base *base, const neb3_dataset *ds, const uint8_t key[32],
                    int32_t *g, uint8_t digest[32], uint8_t ticket[32], uint32_t idx_out[NEB3_NB],
                    uint32_t experts_out[NEB3_MAX_BLOCKS], uint32_t *nclip);

/* Exact 64-bit intermediate step avoiding any signed 32-bit overflow UB even on adversarial inputs. */
void neb3_model_apply(neb3_model *m, const int32_t *gs);

int neb3_model_hashes(const neb3_model *m, uint8_t weights_hash[32], uint8_t residual_hash[32]);

/* Deterministic inference helper (no dropout, deterministic expert 0..L-1) for accuracy evaluation.
 * Not part of consensus. Returns predicted class 0..1 or -1 on error. */
int neb3_model_predict(const neb3_model *m, const neb3_base *base, const uint8_t *x);

#ifdef __cplusplus
}
#endif

#endif
