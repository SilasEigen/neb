// Metal host of the NEB share attempt: device, pipeline, buffers and one dispatch per batch of nonces.
// The kernel is neb_share.metal, compiled from source at run time: no Xcode and no offline metal compiler is needed.
// The caller (Rust, crates/neb-metalminer) owns the model and the dataset and hands them here.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NEB_NIN 784
#define NEB_NH 128
#define NEB_NOUT 10
#define NEB_NW (NEB_NIN / 4)
#define NEB_RECORD_WORDS (8 + 8 + 16)
#define NEB_THREADS 256

typedef struct NebMetal NebMetal;
void neb_metal_free(NebMetal *g);

struct NebMetal {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLComputePipelineState> pipeline;
  id<MTLBuffer> w1t;   // NEB_NW * NEB_NH * 4 bytes, char4 at w * NH + h
  id<MTLBuffer> w2;    // NEB_NOUT * NEB_NH bytes
  id<MTLBuffer> data;  // n * NIN pixels
  id<MTLBuffer> labels;
  id<MTLBuffer> hdr;   // 20 words
  id<MTLBuffer> out;   // max_batch * RECORD_WORDS words
  uint32_t n;
  uint32_t max_batch;
  char name[256];
};

static void set_error(char *err, size_t cap, NSString *message) {
  if (!err || cap == 0) return;
  const char *s = [message UTF8String];
  snprintf(err, cap, "%s", s ? s : "unknown error");
}

static id<MTLBuffer> make_buffer(id<MTLDevice> device, const void *bytes, size_t length) {
  return [device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
}

NebMetal *neb_metal_new(const char *msl_source, const void *pixels, const void *labels, uint32_t n, uint32_t max_batch,
                        char *err, size_t errcap) {
  if (n == 0 || max_batch == 0) {
    set_error(err, errcap, @"the dataset is empty or the batch is zero");
    return NULL;
  }
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) {
    set_error(err, errcap, @"no Metal device");
    return NULL;
  }
  NebMetal *g = calloc(1, sizeof *g);
  if (!g) {
    set_error(err, errcap, @"out of memory");
    return NULL;
  }
  g->device = device; // owned by the create rule
  g->n = n;
  g->max_batch = max_batch;
  snprintf(g->name, sizeof g->name, "%s", [[device name] UTF8String]);

  NSError *e = nil;
  NSString *source = msl_source ? [NSString stringWithUTF8String:msl_source] : nil;
  if (!source) {
    set_error(err, errcap, @"the kernel source is empty");
    neb_metal_free(g);
    return NULL;
  }
  id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&e];
  if (!library) {
    set_error(err, errcap, [NSString stringWithFormat:@"cannot compile the kernel: %@", e.localizedDescription ?: @"?"]);
    neb_metal_free(g);
    return NULL;
  }
  id<MTLFunction> fn = [library newFunctionWithName:@"neb_share"];
  if (!fn) {
    set_error(err, errcap, @"the kernel neb_share is missing");
    neb_metal_free(g);
    return NULL;
  }
  g->pipeline = [device newComputePipelineStateWithFunction:fn error:&e];
  if (!g->pipeline) {
    set_error(err, errcap, [NSString stringWithFormat:@"cannot build the pipeline: %@", e.localizedDescription ?: @"?"]);
    neb_metal_free(g);
    return NULL;
  }
  // The kernel is one threadgroup of NEB_THREADS threads per attempt: a GPU that cannot run it says so here, not with
  // a failed dispatch on every batch.
  if (g->pipeline.maxTotalThreadsPerThreadgroup < NEB_THREADS) {
    set_error(err, errcap,
              [NSString stringWithFormat:@"this GPU runs at most %lu threads per threadgroup, the kernel needs %d",
                                         (unsigned long)g->pipeline.maxTotalThreadsPerThreadgroup, NEB_THREADS]);
    neb_metal_free(g);
    return NULL;
  }
  g->queue = [device newCommandQueue];
  g->data = make_buffer(device, pixels, (size_t)n * NEB_NIN);
  g->labels = make_buffer(device, labels, n);
  g->hdr = make_buffer(device, pixels, 80);  // overwritten by every scan
  g->out = [device newBufferWithLength:(size_t)max_batch * NEB_RECORD_WORDS * 4 options:MTLResourceStorageModeShared];
  g->w1t = [device newBufferWithLength:(size_t)NEB_NW * NEB_NH * 4 options:MTLResourceStorageModeShared];
  g->w2 = [device newBufferWithLength:(size_t)NEB_NOUT * NEB_NH options:MTLResourceStorageModeShared];
  if (!g->data || !g->labels || !g->hdr || !g->out || !g->w1t || !g->w2) {
    set_error(err, errcap, @"cannot allocate the device buffers");
    neb_metal_free(g);
    return NULL;
  }
  return g;
}

void neb_metal_free(NebMetal *g) {
  if (!g) return;
  [g->w1t release];
  [g->w2 release];
  [g->data release];
  [g->labels release];
  [g->hdr release];
  [g->out release];
  [g->pipeline release];
  [g->queue release];
  [g->device release];
  free(g);
}

const char *neb_metal_name(NebMetal *g) { return g ? g->name : ""; }
uint32_t neb_metal_max_batch(NebMetal *g) { return g ? g->max_batch : 0; }
uint32_t *neb_metal_out(NebMetal *g) { return g ? (uint32_t *)[g->out contents] : NULL; }

// w1t[w * NH + h] holds W1[h][4w .. 4w+3]: consecutive h are consecutive words, so a warp reads coalesced.
bool neb_metal_set_model(NebMetal *g, const int8_t *w1, const int8_t *w2, char *err, size_t errcap) {
  (void)err;
  (void)errcap;
  if (!g) return false;
  char *dst = (char *)[g->w1t contents];
  for (int h = 0; h < NEB_NH; h++) {
    for (int w = 0; w < NEB_NW; w++) {
      const int8_t *src = w1 + (size_t)h * NEB_NIN + 4 * w;
      char *out = dst + (size_t)(w * NEB_NH + h) * 4;
      out[0] = (char)src[0];
      out[1] = (char)src[1];
      out[2] = (char)src[2];
      out[3] = (char)src[3];
    }
  }
  memcpy([g->w2 contents], w2, NEB_NOUT * NEB_NH);
  return true;
}

// One batch: attempt `nonce_base + i` for i in 0 .. count-1, on the header `hdr` (20 words, nonce word zero).
// The results are in neb_metal_out: RECORD_WORDS words per attempt.
bool neb_metal_scan(NebMetal *g, const uint32_t *hdr, uint32_t nonce_base, uint32_t count, uint32_t key_mode, char *err,
                    size_t errcap) {
  if (!g || count == 0 || count > g->max_batch) {
    set_error(err, errcap, @"the batch is empty or above the maximum");
    return false;
  }
  memcpy([g->hdr contents], hdr, 80);
  struct {
    uint32_t n;
    uint32_t nonce_base;
    uint32_t key_mode;
  } params = {g->n, nonce_base, key_mode};

  // Called from a Rust thread, which has no autorelease pool: without this the command buffer and the encoder are
  // never released and a long run grows without bound.
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [g->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g->pipeline];
    [enc setBuffer:g->hdr offset:0 atIndex:0];
    [enc setBuffer:g->w1t offset:0 atIndex:1];
    [enc setBuffer:g->w2 offset:0 atIndex:2];
    [enc setBuffer:g->data offset:0 atIndex:3];
    [enc setBuffer:g->labels offset:0 atIndex:4];
    [enc setBuffer:g->out offset:0 atIndex:5];
    [enc setBytes:&params length:sizeof params atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(NEB_THREADS, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      set_error(err, errcap,
                [NSString stringWithFormat:@"the command buffer failed: %@", cb.error.localizedDescription ?: @"?"]);
      return false;
    }
  }
  return true;
}
