// nebhash v2 attempt on Metal (spec/nebhash-v2.md section 4), one threadgroup of 256 threads per attempt.
// The contract is the C reference (pow/ref/nebhash2.c, crate neb-pow): the digest, the ticket and the sample indices
// written by this kernel must equal it bit for bit. All arithmetic is integer.
//
// The layout is the one of pow/gpu/nebhash2_cuda.cu: for each of the 16 samples the group runs the forward and the
// backward pass layer by layer, all threads sharing the positions of a layer, and keeps the inputs and the routed
// errors of every layer in a scratch area of device memory; then each thread computes whole entries of the gradient
// (no atomics on the gradient, so no order effect), clips them and adds them to the digest lanes, which are sums
// modulo 2^32 and so do not depend on the order of the additions. The architecture is a table of layers (the one of
// neb2_arch_get in the C reference), passed in `Plan` together with the offsets of the scratch layout; one kernel
// serves KEP1 and RET2.
//
// Device memory written by one thread and read by another of the group is separated by a barrier with
// mem_flags::mem_device. The weights (at most 26 000 bytes) sit in threadgroup memory.
//
// The helpers (BLAKE3, ChaCha8, clampi) are in neb_common.metal, which precedes this text.

constant int NB = 16;
constant int CLIP = 256;
constant int TPOS = 7;
constant int TNEG = -8;
constant int NOUT = 2;
constant int MAXW = 26000;
constant int MAXL = 6;

constant int CONV1D = 1;
constant int CONV2D = 2;
constant int LINEAR = 3;

// A layer of the table with its derived sizes and scratch offsets, all ints, in the order of Rust's `Layer`:
// the fields of neb2_layer (kind .. w_off, with src split in two), then nin_in (input values), nconv (values of the
// convolution grid), nout (values after pooling) and the offsets inside one sample's part (off_in, off_d) and inside
// the part common to the samples (off_a, off_out, off_e).
struct Layer {
  int kind, cin, cout, k, stride, pad, pool;
  int s, sb, su;
  int in_len, in_off, nsrc, src0, src1;
  int conv_len, out_len, nw, w_off;
  int nin_in, nconv, nout;
  int off_in, off_d, off_a, off_out, off_e;
};

struct Plan {
  int nlayers, nweights, nin, sample_bytes, common_bytes, block_bytes, pad0, pad1;
  Layer l[MAXL];
};

struct Params {
  uint n;          // samples of the dataset
  uint nonce_base; // nonce of the first attempt of the batch
  uint key_mode;   // 1: the key is hdr[0..8] (the self-test); 0: key = BLAKE3(header) with the nonce at word 19
};

constant int RECORD_WORDS = 8 + 8 + NB;

#define BAR threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)

// rshift(x, n) = (x + 2^(n-1)) >> n with rshift(x, 0) = x.
static inline int rsh0(int x, int n) { return n == 0 ? x : (x + (1 << (n - 1))) >> n; }

// The value of an activation byte: low nibble, sign extended. Bit 4 is set when y was outside -8..7.
static inline int nib(char a) { return ((int)(char)(a << 4)) >> 4; }

// The first maximum of the pooling window that holds grid position g, or -1 when g is in no full window; *j is the
// pooled index of the window.
static int window_argmax(constant Layer &L, device const char *a, int g, thread int &j) {
  const int cl = L.conv_len, ol = L.out_len, p = L.pool;
  if (L.kind == CONV1D) {
    const int o = g / cl, pos = g % cl, w = pos / p;
    if (w >= ol) return -1;
    const int s = o * cl + w * p;
    int best = s;
    for (int t = 1; t < p; t++)
      if (nib(a[s + t]) > nib(a[best])) best = s + t;
    j = o * ol + w;
    return best;
  }
  const int o = g / (cl * cl), py = (g / cl) % cl, px = g % cl, wy = py / p, wx = px / p;
  if (wy >= ol || wx >= ol) return -1;
  int best = (o * cl + wy * p) * cl + wx * p;
  for (int ty = 0; ty < p; ty++)
    for (int tx = 0; tx < p; tx++) {
      const int q = (o * cl + wy * p + ty) * cl + wx * p + tx;
      if (nib(a[q]) > nib(a[best])) best = q;
    }
  j = (o * ol + wy) * ol + wx;
  return best;
}

kernel void neb_share2(device const uint *hdr [[buffer(0)]],      // 20 words, nonce field (word 19) zero
                       device const char *wg [[buffer(1)]],       // the weights in model order
                       device const uchar *data [[buffer(2)]],    // n rows of nin values
                       device const uchar *labels [[buffer(3)]],
                       device char *scratch [[buffer(4)]],        // block_bytes per attempt of the dispatch
                       device uint *out [[buffer(5)]],            // RECORD_WORDS per attempt
                       constant Params &p [[buffer(6)]],
                       constant Plan &P [[buffer(7)]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint tg [[threadgroup_position_in_grid]]) {
  threadgroup char W[MAXW];
  threadgroup uint key[8];
  threadgroup uint idx[NB];
  threadgroup int label[NB];
  threadgroup atomic_uint lanes[8];
  const int T = 256;

  for (int f = (int)tid; f < P.nweights; f += T) W[f] = wg[f];
  if (tid < 8) atomic_store_explicit(&lanes[tid], 0u, memory_order_relaxed);
  if (tid == 0) {
    uint key_l[8];
    if (p.key_mode != 0u) {
#pragma unroll
      for (int i = 0; i < 8; i++) key_l[i] = hdr[i];
    } else {
      // key = BLAKE3(header): one chunk of two blocks, bytes 0..63 then bytes 64..79 zero padded, length 16, end | root.
      uint b0[16], b1[16], iv[8], cv[8];
      for (int i = 0; i < 16; i++) b0[i] = hdr[i];
      for (int i = 0; i < 4; i++) b1[i] = hdr[16 + i];
      b1[3] = p.nonce_base + tg; // the nonce occupies bytes 76..80, that is word 19 of the header
      for (int i = 4; i < 16; i++) b1[i] = 0u;
      for (int i = 0; i < 8; i++) iv[i] = B3IV[i];
      b3_compress(iv, b0, 64u, 1u, cv);
      b3_compress(cv, b1, 16u, 2u | 8u, key_l);
    }
    for (int i = 0; i < 8; i++) key[i] = key_l[i];
    uint u[16];
    chacha8_block0(key_l, u);
    for (int j = 0; j < NB; j++) {
      idx[j] = (uint)(((ulong)u[j] * (ulong)p.n) >> 32);
      label[j] = (int)labels[idx[j]];
    }
  }
  BAR;

  device char *base = scratch + (ulong)tg * (ulong)P.block_bytes;
  device char *common = base + (ulong)NB * (ulong)P.sample_bytes;
  const int last = P.nlayers - 1;

  for (int b = 0; b < NB; b++) {
    device char *smp = base + (ulong)b * (ulong)P.sample_bytes;
    device const uchar *x = data + (ulong)idx[b] * (ulong)P.nin;

    // ---- forward ----
    for (int i = 0; i < P.nlayers; i++) {
      constant Layer &L = P.l[i];
      device char *in = smp + L.off_in;
      device char *a = common + L.off_a;
      device char *outp = common + L.off_out;
      if (L.nsrc == 0) {
        for (int q = (int)tid; q < L.nin_in; q += T) in[q] = (char)x[L.in_off + q];
      } else {
        int at = 0;
        for (int s = 0; s < L.nsrc; s++) {
          const int src = s == 0 ? L.src0 : L.src1;
          device const char *so = common + P.l[src].off_out;
          const int ns = P.l[src].nout;
          for (int q = (int)tid; q < ns; q += T) in[at + q] = so[q];
          at += ns;
        }
      }
      BAR;
      threadgroup const char *w = W + L.w_off;
      for (int g = (int)tid; g < L.nconv; g += T) {
        int z = 0;
        if (L.kind == LINEAR) {
          threadgroup const char *wr = w + g * L.cin;
          for (int q = 0; q < L.cin; q++) z += (int)wr[q] * (int)in[q];
        } else if (L.kind == CONV1D) {
          const int cl = L.conv_len, o = g / cl, pp = g % cl, nn = L.in_len;
          for (int c = 0; c < L.cin; c++) {
            threadgroup const char *wr = w + (o * L.cin + c) * L.k;
            for (int t = 0; t < L.k; t++) {
              const int q = pp * L.stride + t - L.pad;
              if (q >= 0 && q < nn) z += (int)wr[t] * (int)in[c * nn + q];
            }
          }
        } else {
          const int cl = L.conv_len, o = g / (cl * cl), py = (g / cl) % cl, px = g % cl, nn = L.in_len, k = L.k;
          for (int c = 0; c < L.cin; c++) {
            threadgroup const char *wr = w + (o * L.cin + c) * k * k;
            for (int ky = 0; ky < k; ky++) {
              const int qy = py * L.stride + ky - L.pad;
              if (qy < 0 || qy >= nn) continue;
              for (int kx = 0; kx < k; kx++) {
                const int qx = px * L.stride + kx - L.pad;
                if (qx >= 0 && qx < nn) z += (int)wr[ky * k + kx] * (int)in[(c * nn + qy) * nn + qx];
              }
            }
          }
        }
        const int y = rsh0(z, L.s);
        const int v = clampi(y, -8, 7);
        a[g] = (char)((v & 15) | ((y < -8 || y > 7) ? 16 : 0));
      }
      BAR;
      for (int j = (int)tid; j < L.nout; j += T) {
        if (L.pool <= 1) {
          outp[j] = (char)nib(a[j]);
          continue;
        }
        // the window of pooled index j and its first maximum
        int g0;
        if (L.kind == CONV1D) {
          g0 = (j / L.out_len) * L.conv_len + (j % L.out_len) * L.pool;
        } else {
          const int ol = L.out_len, o = j / (ol * ol), jy = (j / ol) % ol, jx = j % ol;
          g0 = (o * L.conv_len + jy * L.pool) * L.conv_len + jx * L.pool;
        }
        int jj = 0;
        const int best = window_argmax(L, a, g0, jj);
        outp[j] = (char)nib(a[best]);
      }
      BAR;
    }

    // ---- the error of the output layer: E[o] = clip(a[o] - T[o], -8, 7) ----
    {
      device char *e = common + P.l[last].off_e;
      device const char *o = common + P.l[last].off_out;
      if ((int)tid < NOUT) e[tid] = (char)clampi((int)o[tid] - ((int)tid == label[b] ? TPOS : TNEG), -8, 7);
    }
    BAR;

    // ---- backward: route through the pooling, mask (not on the output layer), send the error to the sources ----
    for (int i = last; i >= 0; i--) {
      constant Layer &L = P.l[i];
      device const char *a = common + L.off_a;
      device const char *e = common + L.off_e;
      device char *d = smp + L.off_d;
      for (int g = (int)tid; g < L.nconv; g += T) {
        int v = 0;
        if (L.pool <= 1) {
          v = (int)e[g];
        } else {
          int j = 0;
          if (window_argmax(L, a, g, j) == g) v = (int)e[j];
        }
        if (i != last && (a[g] & 16) != 0) v = 0;
        d[g] = (char)v;
      }
      BAR;
      if (L.nsrc > 0) {
        threadgroup const char *w = W + L.w_off;
        const int s0 = L.src0, s1 = L.src1;
        const int n0 = P.l[s0].nout;
        for (int q = (int)tid; q < L.nin_in; q += T) {
          int m = 0;
          if (L.kind == LINEAR) {
            for (int o = 0; o < L.cout; o++) m += (int)w[o * L.cin + q] * (int)d[o];
          } else if (L.kind == CONV1D) {
            const int nn = L.in_len, c = q / nn, t = q % nn, cl = L.conv_len;
            for (int o = 0; o < L.cout; o++)
              for (int kk = 0; kk < L.k; kk++) {
                const int pp = t + L.pad - kk;
                if (pp < 0 || pp % L.stride != 0) continue;
                const int pos = pp / L.stride;
                if (pos < cl) m += (int)w[(o * L.cin + c) * L.k + kk] * (int)d[o * cl + pos];
              }
          } else {
            const int nn = L.in_len, c = q / (nn * nn), ty = (q / nn) % nn, tx = q % nn, cl = L.conv_len, k = L.k;
            for (int o = 0; o < L.cout; o++)
              for (int ky = 0; ky < k; ky++) {
                const int ppy = ty + L.pad - ky;
                if (ppy < 0 || ppy % L.stride != 0) continue;
                const int py = ppy / L.stride;
                if (py >= cl) continue;
                for (int kx = 0; kx < k; kx++) {
                  const int ppx = tx + L.pad - kx;
                  if (ppx < 0 || ppx % L.stride != 0) continue;
                  const int px = ppx / L.stride;
                  if (px < cl) m += (int)w[((o * L.cin + c) * k + ky) * k + kx] * (int)d[(o * cl + py) * cl + px];
                }
              }
          }
          // q indexes the concatenated input: the source is s0 for q < nout(s0), else s1
          const int src = (L.nsrc == 1 || q < n0) ? s0 : s1;
          const int at = (L.nsrc == 1 || q < n0) ? 0 : n0;
          device char *es = common + P.l[src].off_e;
          es[q - at] = (char)clampi(rsh0(m, L.sb), -8, 7);
        }
      }
      BAR;
    }
  }

  // ---- gradient, clip and digest: thread tid computes the entries f = tid, tid + 256, ... of every layer ----
  uint lane[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
  for (int i = 0; i < P.nlayers; i++) {
    constant Layer &L = P.l[i];
    for (int f = (int)tid; f < L.nw; f += T) {
      int G = 0;
      for (int b = 0; b < NB; b++) {
        device const char *smp = base + (ulong)b * (ulong)P.sample_bytes;
        device const char *in = smp + L.off_in;
        device const char *d = smp + L.off_d;
        if (L.kind == LINEAR) {
          G += (int)d[f / L.cin] * (int)in[f % L.cin];
        } else if (L.kind == CONV1D) {
          const int o = f / (L.cin * L.k), c = (f / L.k) % L.cin, kk = f % L.k, cl = L.conv_len, nn = L.in_len;
          for (int pp = 0; pp < cl; pp++) {
            const int q = pp * L.stride + kk - L.pad;
            if (q >= 0 && q < nn) G += (int)d[o * cl + pp] * (int)in[c * nn + q];
          }
        } else {
          const int k = L.k, o = f / (L.cin * k * k), c = (f / (k * k)) % L.cin, ky = (f / k) % k, kx = f % k;
          const int cl = L.conv_len, nn = L.in_len;
          for (int py = 0; py < cl; py++) {
            const int qy = py * L.stride + ky - L.pad;
            if (qy < 0 || qy >= nn) continue;
            for (int px = 0; px < cl; px++) {
              const int qx = px * L.stride + kx - L.pad;
              if (qx >= 0 && qx < nn) G += (int)d[(o * cl + py) * cl + px] * (int)in[(c * nn + qy) * nn + qx];
            }
          }
        }
      }
      const int cg = clampi(G, -CLIP, CLIP);
      lane[f & 7] += (uint)cg ^ ((uint)f + 7919u * (uint)(i + 1));
    }
  }
  for (int k = 0; k < 8; k++) atomic_fetch_add_explicit(&lanes[k], lane[k], memory_order_relaxed);
  BAR;

  // ticket = BLAKE3(digest || key): one 64-byte block, chunk start | end | root.
  if (tid == 0) {
    uint blk[16], iv[8], res[8];
    for (int i = 0; i < 8; i++) {
      blk[i] = atomic_load_explicit(&lanes[i], memory_order_relaxed);
      blk[8 + i] = key[i];
      iv[i] = B3IV[i];
    }
    b3_compress(iv, blk, 64u, 1u | 2u | 8u, res);
    device uint *rec = out + (ulong)RECORD_WORDS * tg;
    for (int i = 0; i < 8; i++) {
      rec[i] = blk[i];
      rec[8 + i] = res[i];
    }
    for (int j = 0; j < NB; j++) rec[16 + j] = idx[j];
  }
}
