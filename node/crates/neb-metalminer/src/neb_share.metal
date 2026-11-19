// NEB training-share attempt on Metal (spec/shares.md sections 5.1 to 5.4), one threadgroup per attempt.
// The contract is the C reference (pow/ref/shares.c, crate neb-pow): the digest, the ticket and the sample indices
// written by this kernel must equal it bit for bit. All arithmetic is integer, so there is no rounding to differ.
//
// Layout follows the dp4a kernel of pow/gpu/shares_cuda.cu with 256 threads per attempt: thread tid owns hidden unit
// h = tid & 127 for the samples 8 * (tid >> 7) .. +7. Metal has no dp4a, so a 4-lane dot product is four multiplies
// and three adds, which is exact and needs no saturation.


// The helpers (BLAKE3, ChaCha8, rotations, clampi) are in neb_common.metal, which precedes this text.

constant int NIN = 784;
constant int NH = 128;
constant int NOUT = 10;
constant int NB = 16;
constant int NW = NIN / 4; // 196 words of 4 pixels

constant int S1 = 5;
constant int S2 = 5;
constant int SB = 3;
constant int CLIP = 256;
constant int TPOS = 7;
constant int TNEG = -8;


// rshift(x, n) = (x + 2^(n-1)) >> n, arithmetic shift (spec section 2.1).
static inline int rsh(int x, int n) { return (x + (1 << (n - 1))) >> n; }

static inline int4 widen(char4 c) { return int4((int)c.x, (int)c.y, (int)c.z, (int)c.w); }

static inline int dot4(char4 a, char4 b) {
  int4 x = widen(a);
  int4 y = widen(b);
  return (x.x * y.x + x.y * y.y) + (x.z * y.z + x.w * y.w);
}

struct Params {
  uint n;          // samples of the dataset
  uint nonce_base; // nonce of the first attempt of the batch
  uint key_mode;   // 1: the key is hdr[0..8] (the self-test); 0: key = BLAKE3(header) with the nonce at word 19
};

// Record of one attempt: digest (8 words), ticket (8 words), sample indices (16 words).
constant int RECORD_WORDS = 8 + 8 + NB;

kernel void neb_share(device const uint *hdr [[buffer(0)]],           // 20 words, nonce field (word 19) zero
                      device const char4 *w1t [[buffer(1)]],         // w * NH + h, pixels 4w..4w+3 of row h
                      device const char *w2g [[buffer(2)]],          // NOUT * NH, row major o * NH + h
                      device const uchar *data [[buffer(3)]],        // n rows of NIN pixels
                      device const uchar *labels [[buffer(4)]],
                      device uint *out [[buffer(5)]],                // RECORD_WORDS per attempt
                      constant Params &p [[buffer(6)]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint tg [[threadgroup_position_in_grid]]) {
  threadgroup char xrow[NB * NIN];   // sample b, pixel i
  threadgroup int xcol[4 * NIN];     // pixel i, samples 4g..4g+3 in the four bytes of word g * NIN + i
  threadgroup char a1[NB * NH];      // activation of layer 1
  threadgroup int e1p[NH * 4];       // E1 of unit h, samples 4g..4g+3
  threadgroup char e2[NB * NOUT];
  threadgroup char w2[NOUT * NH];
  threadgroup uint red[256];
  threadgroup uint key[8];
  threadgroup uint idx[NB];
  threadgroup int label[NB];
  threadgroup uint digest[8];

  if (tid == 0) {
    uint key_l[8];
    if (p.key_mode != 0u) {
#pragma unroll
      for (int i = 0; i < 8; i++) key_l[i] = hdr[i];
    } else {
      // key = BLAKE3(header): one chunk of two blocks, bytes 0..63 then bytes 64..79 zero padded, length 16, end | root.
      uint b0[16], b1[16], iv[8], cv[8];
#pragma unroll
      for (int i = 0; i < 16; i++) b0[i] = hdr[i];
#pragma unroll
      for (int i = 0; i < 4; i++) b1[i] = hdr[16 + i];
      b1[3] = p.nonce_base + tg; // the nonce occupies bytes 76..80, that is word 19 of the header
#pragma unroll
      for (int i = 4; i < 16; i++) b1[i] = 0u;
#pragma unroll
      for (int i = 0; i < 8; i++) iv[i] = B3IV[i];
      b3_compress(iv, b0, 64u, 1u, cv);
      b3_compress(cv, b1, 16u, 2u | 8u, key_l);
    }
#pragma unroll
    for (int i = 0; i < 8; i++) key[i] = key_l[i];
    uint u[16];
    chacha8_block0(key_l, u);
#pragma unroll
    for (int j = 0; j < NB; j++) {
      idx[j] = (uint)(((ulong)u[j] * (ulong)p.n) >> 32);
      label[j] = (int)labels[idx[j]];
    }
  }

  // W2 to threadgroup memory (320 words), as the 32-bit words of the byte array.
  for (int i = tid; i < NOUT * NH / 4; i += 256) ((threadgroup int *)w2)[i] = ((device const int *)w2g)[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The 16 sample rows, one word at a time; the dataset row is 784 bytes, a multiple of 4.
  for (int e = tid; e < NB * (NIN / 4); e += 256) {
    int b = e / (NIN / 4), w = e - b * (NIN / 4);
    ((threadgroup int *)xrow)[e] = ((device const int *)(data + (ulong)idx[b] * NIN))[w];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Transpose to the sample-packed form used by the G1 product.
  for (int e = tid; e < 4 * NIN; e += 256) {
    int g = e / NIN, i = e - g * NIN;
    uint v0 = (uchar)xrow[(4 * g + 0) * NIN + i];
    uint v1 = (uchar)xrow[(4 * g + 1) * NIN + i];
    uint v2 = (uchar)xrow[(4 * g + 2) * NIN + i];
    uint v3 = (uchar)xrow[(4 * g + 3) * NIN + i];
    xcol[e] = (int)(v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
  }

  // Layer 1: z1[b][h] = sum over i of X[b][i] * W1[h][i], |z1| <= 43904; the activation is the shifted, clipped sum.
  int h = (int)(tid & (NH - 1));
  int b0i = (int)(tid >> 7) * 8;
  int acc[8];
#pragma unroll
  for (int k = 0; k < 8; k++) acc[k] = 0;
  for (int w = 0; w < NW; w++) {
    char4 wv = w1t[w * NH + h];
    int4 wi = widen(wv);
#pragma unroll
    for (int k = 0; k < 8; k++) {
      char4 xv = *((threadgroup const char4 *)(xrow + (b0i + k) * NIN + 4 * w));
      int4 xi = widen(xv);
      acc[k] += (xi.x * wi.x + xi.y * wi.y) + (xi.z * wi.z + xi.w * wi.w);
    }
  }
  bool opens[8];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    int y = rsh(acc[k], S1);
    a1[(b0i + k) * NH + h] = (char)clampi(y, -8, 7);
    opens[k] = (y >= -8 && y <= 7);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Layer 2 and the output error: E2 = clip(clip(rsh(z2, S2), -8, 7) - T, -8, 7).
  if (tid < NB * NOUT) {
    int b = (int)tid / NOUT, o = (int)tid - b * NOUT;
    int z = 0;
    for (int u = 0; u < NH; u++) z += (int)a1[b * NH + u] * (int)w2[o * NH + u];
    int y2 = clampi(rsh(z, S2), -8, 7);
    int t = (o == label[b]) ? TPOS : TNEG;
    e2[tid] = (char)clampi(y2 - t, -8, 7);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Propagated error E1[b][h] = opens ? clip(rsh(m, SB), -8, 7) : 0, m = sum over o of E2[b][o] * W2[o][h].
  {
    int pk[2] = {0, 0};
#pragma unroll
    for (int k = 0; k < 8; k++) {
      int m = 0;
#pragma unroll
      for (int o = 0; o < NOUT; o++) m += (int)e2[(b0i + k) * NOUT + o] * (int)w2[o * NH + h];
      int e1 = opens[k] ? clampi(rsh(m, SB), -8, 7) : 0;
      pk[k >> 2] |= (int)(uint)(uchar)(char)e1 << (8 * (k & 3));
    }
    e1p[h * 4 + (int)(tid >> 7) * 2 + 0] = pk[0];
    e1p[h * 4 + (int)(tid >> 7) * 2 + 1] = pk[1];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Gradients, clip, digest lanes (sections 5.2 and 5.3). Entries f = tid + 256 k keep f mod 8 = tid mod 8.
  uint lane = 0u;
  for (int f = (int)tid; f < NOUT * NH; f += 256) {
    int o = f / NH, hh = f - o * NH;
    int g = 0;
#pragma unroll
    for (int b = 0; b < NB; b++) g += (int)e2[b * NOUT + o] * (int)a1[b * NH + hh];
    lane += (uint)clampi(g, -CLIP, CLIP) ^ ((uint)f + 7919u * 2u);
  }
  for (int f = (int)tid; f < NH * NIN; f += 256) {
    int hh = f / NIN, i = f - hh * NIN;
    int g = 0;
#pragma unroll
    for (int gq = 0; gq < 4; gq++)
      g += dot4(as_type<char4>(e1p[hh * 4 + gq]), as_type<char4>(xcol[gq * NIN + i]));
    lane += (uint)clampi(g, -CLIP, CLIP) ^ ((uint)f + 7919u * 1u);
  }
  // Lane d is the sum of the 32 partial lanes of the threads with tid mod 8 = d.
  red[tid] = lane;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 8) {
    uint d = 0u;
#pragma unroll
    for (int k = 0; k < 256 / 8; k++) d += red[tid + 8 * k];
    digest[tid] = d;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ticket = BLAKE3(digest || key): one 64-byte block, chunk start | end | root.
  if (tid == 0) {
    uint blk[16], iv[8], res[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      blk[i] = digest[i];
      blk[8 + i] = key[i];
    }
#pragma unroll
    for (int i = 0; i < 8; i++) iv[i] = B3IV[i];
    b3_compress(iv, blk, 64u, 1u | 2u | 8u, res);
    device uint *rec = out + (ulong)RECORD_WORDS * tg;
#pragma unroll
    for (int i = 0; i < 8; i++) rec[i] = digest[i];
#pragma unroll
    for (int i = 0; i < 8; i++) rec[8 + i] = res[i];
#pragma unroll
    for (int j = 0; j < NB; j++) rec[16 + j] = idx[j];
  }
}
