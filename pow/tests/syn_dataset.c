/* Test dataset of spec/shares.md section 9: 10 class prototypes plus per-sample noise, both expanded with ChaCha8 from
 * BLAKE3 keys. Pure function of the sample index, as neb_dataset requires. Not thread-safe: the prototypes are built on
 * the first call. */
#include "syn_dataset.h"
#include <string.h>
#include "blake3_min.h"
#include "chacha8.h"

/* P_c of section 9: pixels 0..7 */
static uint8_t proto[10][NEB_NIN];
static int proto_ready;

static void put_le32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static void make_protos(void) {
  for (int c = 0; c < 10; c++) {
    uint8_t in[14], key[32], s[NEB_NIN];
    memcpy(in, "neb-syn-proto", 13);
    in[13] = (uint8_t)c;
    blake3_hash(in, 14, key);
    chacha8_stream(key, s, NEB_NIN);
    for (int p = 0; p < NEB_NIN; p++) proto[c][p] = s[p] & 7;
  }
  proto_ready = 1;
}

int neb_syn_get(void *ctx, uint32_t i, uint8_t x[NEB_NIN]) {
  (void)ctx;
  if (!proto_ready) make_protos();
  uint8_t in[14 + 4], key[32], n[NEB_NIN];
  memcpy(in, "neb-syn-sample", 14);
  put_le32(in + 14, i);
  blake3_hash(in, sizeof in, key);
  chacha8_stream(key, n, NEB_NIN);
  const int c = (int)(i % 10);
  /* the noise (n & 3) - 1 is in -1..2; the clamp keeps the pixel in 0..7 */
  for (int p = 0; p < NEB_NIN; p++) {
    const int v = (int)proto[c][p] + (n[p] & 3) - 1;
    x[p] = (uint8_t)(v < 0 ? 0 : (v > 7 ? 7 : v));
  }
  return c;
}
