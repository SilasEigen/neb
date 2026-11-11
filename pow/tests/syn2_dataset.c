/* Test datasets of spec/nebhash-v2.md section 8: two class prototypes per architecture plus per-sample noise, both
 * expanded with ChaCha8 from BLAKE3 keys. A pure function of the architecture and the sample index. */
#include "syn2_dataset.h"
#include <string.h>
#include "blake3_min.h"
#include "chacha8.h"

static void put_le32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

int neb_syn2_get(void *ctx, uint32_t i, uint8_t *x) {
  const int arch = *(const int *)ctx;
  const int nin = neb2_arch_get(arch)->nin;
  const int c = (int)(i % 2);
  uint8_t in[15 + 1 + 4], key[32];
  uint8_t proto[NEB2_MAX_NIN], noise[NEB2_MAX_NIN];
  /* P_c = ChaCha8(BLAKE3("neb-syn2-proto" || arch || c)) & 7 */
  memcpy(in, "neb-syn2-proto", 14);
  in[14] = (uint8_t)arch;
  in[15] = (uint8_t)c;
  blake3_hash(in, 16, key);
  chacha8_stream(key, proto, (size_t)nin);
  /* noise = ChaCha8(BLAKE3("neb-syn2-sample" || arch || LE32(i))) */
  memcpy(in, "neb-syn2-sample", 15);
  in[15] = (uint8_t)arch;
  put_le32(in + 16, i);
  blake3_hash(in, 20, key);
  chacha8_stream(key, noise, (size_t)nin);
  for (int p = 0; p < nin; p++) {
    const int v = (int)(proto[p] & 7) + (noise[p] & 3) - 1;
    x[p] = (uint8_t)(v < 0 ? 0 : (v > 7 ? 7 : v));
  }
  return c;
}
