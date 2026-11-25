# nebhash v3: frozen quantized base models with on-chain integer LoRA

Status: draft proposal (RFC) for protocol activation via `NMD3` datasets. It defines the attempt, key-derived
deterministic dropout, LoRA gradient, and model step for frozen INT8 base backbones with compact trainable low-rank
adapters (`A`, `B`) and a linear head, in the exact integer arithmetic of `spec/shares.md`. Everything that this
text does not change is as in `spec/shares.md`: the key of an attempt (3.5), the ticket (5.4), the model step rule
(6), the chain integration (11), and dataset upgrades (11.7).

## 1. Motivation and What Changes

`nebhash v2` (`spec/nebhash-v2.md`) trains small convolutional models (11–13 KB packed weights) over `NB = 16`
independent samples per attempt. Because samples within a share do not interact and the attempt key only selects
sample indices, miners can precompute a per-sample gradient table once per model state (`pow/gpu/nebhash2_table_cuda.cu`),
reducing subsequent nonce attempts to table lookups, integer additions, and BLAKE3 hashing.

`nebhash v3` introduces two changes while keeping bit-exact integer consensus, compact P2P work frames, and
millisecond-class single-core CPU verification for winning shares:

1. **Key-derived deterministic dropout (anti-table)**: Each share derives a deterministic 1-bit dropout mask from its
   32-byte `key` for every LoRA bottleneck activation. Different attempts apply different masks to the same sample,
   making per-sample gradient table precomputation impossible while acting as standard regularization during training.
2. **Frozen quantized base + on-chain INT8 LoRA**: A large, immutable INT8 base backbone (committed by its BLAKE3
   digest, never broadcast per block) is paired with compact trainable LoRA adapters (`A`, `B` in `-8..7`) and a
   trainable classification head. Only the LoRA and head weights (~132 KB packed for the 8B-class mainnet profile)
   are updated per block, stored in state hashes, and distributed in work frames.

| | `nebhash v2` (`NMD2`) | `nebhash v3` (`NMD3`) |
|---|---|---|
| batch size `NB3` | 16 | 32 (aligned to `sm_80+` `m16` INT8 `mma` tiles) |
| per-sample table precomputation | possible | prevented by key-derived LoRA dropout |
| base model | none | frozen INT8 base committed by `base_hash` |
| trainable on-chain weights | 22,280 – 26,000 (11–13 KB) | `LORA_TINY`: 2,176 (~1 KB); `QWEN3_8B_LORA`: 270,336 (~132 KB) |
| VRAM requirement (mainnet profile) | < 64 MB | > 8.2 GB (excludes 6 GB / 8 GB legacy non-Tensor-Core cards) |

## 2. Datasets (`NMD3`)

A v3 dataset begins with the 4-byte magic `NMD3`, followed by:
- `arch_id` as a little-endian `u32` (`1` for `LORA_TINY`, `2` for `QWEN3_8B_LORA`),
- `base_hash` as 32 bytes (`BLAKE3("neb-base-weights" || packed_base_weights)` committing to the frozen INT8 base),
- `N` as a little-endian `u32` (`N >= 1`), the sample count,
- `N` records of `DIN + 1` bytes: `DIN` quantized input features in `0..7` (representing frozen upstream hidden
  states / pooled embeddings or synthetic test inputs) and the class label in `0..1`.

An unknown `arch_id`, a feature value above `7`, or a label above `1` makes the dataset invalid.

## 3. Architectures

Each `NMD3` architecture consists of `L` residual LoRA blocks (`l = 0 .. L-1`) operating at hidden width `D` with
bottleneck rank `R`, followed by one trainable linear output head `H` (`NOUT = 2` classes).

For each block `l`:
- Frozen base matrix `W_base[l]` has shape `[D][D]`, entries in `-8..7` (row-major).
- Trainable LoRA down-projection `A[l]` has shape `[R][D]`, entries in `-8..7` (row-major).
- Trainable LoRA up-projection `B[l]` has shape `[D][R]`, entries in `-8..7` (row-major).
- Output head `H` has shape `[NOUT][D]`, entries in `-8..7` (row-major).

All trainable weights are stored in flat order: for `l = 0 .. L-1`, `A[l]` then `B[l]`, followed by `H`.

### 3.1 `LORA_TINY` (`arch_id = 1`, deterministic test & CI profile)

- `D = 64`, `R = 8`, `L = 2`, `NOUT = 2`
- Trainable weights: `2 * (8 * 64 + 64 * 8) + 2 * 64 = 2,176` (`1,088` bytes packed)
- Shifts: `s_a = 4`, `s_lora = 1`, `s_out = 5`, `sb_b = 4`, `sb_in = 5`, `s_head = 5`, `sb_head = 3`, `su_lora = 14`, `su_head = 12`

### 3.2 `QWEN3_8B_LORA` (`arch_id = 2`, mainnet 8B-class Apache-2.0 profile)

Targets the top 4 Transformer projection blocks (`D = 4096`) of an Apache-2.0 quantized **Qwen3-8B** INT8 frozen
backbone (~8.2 GB VRAM footprint, naturally excluding 6 GB / 8 GB legacy cards without Tensor Cores while saturating
`sm_80+` `mma.sync.aligned.m16n8k32.s32.s8.s8.s32` tiles):

- `D = 4096`, `R = 8`, `L = 4`, `NOUT = 2`
- Trainable weights: `4 * (8 * 4096 + 4096 * 8) + 2 * 4096 = 270,336` (`135,168` bytes packed, ~132 KB)
- Shifts: `s_a = 7`, `s_lora = 1`, `s_out = 8`, `sb_b = 5`, `sb_in = 8`, `s_head = 8`, `sb_head = 3`, `su_lora = 16`, `su_head = 14`

## 4. A Share

### 4.1 Sample Indices and Key-Derived Dropout Masks

Given the 32-byte share `key`:
1. Draw `NB3 = 32` sample indices from `ChaCha8(key, 4 * NB3)` using the exact reduction `(le32(u_j) * N) >> 32`.
2. Derive the dropout key:
   ```
   drop_key = BLAKE3("neb-lora-drop" || key)
   ```
3. Generate `ceil(NB3 * L * R / 8)` bytes from `ChaCha8(drop_key)`. For sample `b` (`0..NB3-1`), block `l` (`0..L-1`),
   and rank index `r` (`0..R-1`), let bit index `bit = (b * L + l) * R + r`. The dropout mask is:
   ```
   mask[b][l][r] = (stream[bit >> 3] >> (bit & 7)) & 1
   ```

### 4.2 Forward Pass

For each sample `b` in `0 .. NB3-1`, let `x_0` be the `D` input values (in `0..7`). For `l = 0 .. L-1`:
```
z_base = W_base[l] * x_l                        in int32, length D
z_a    = A[l] * x_l                             in int32, length R
y_a    = rshift(z_a, s_a)
u_clip = clip(y_a, -8, 7)
u      = u_clip * mask[b][l]                    1-bit key-derived dropout
z_lora = B[l] * u                               in int32, length D
y_out  = rshift(z_base + (z_lora << s_lora), s_out)
x_{l+1} = clip(y_out, -8, 7)
```
Then the output head:
```
z_h = H * x_L                                   in int32, length NOUT
y_h = rshift(z_h, s_head)
a_h = clip(y_h, -8, 7)
```

### 4.3 Backward Pass and Gradients

For each sample `b` with target `T[o] = TPOS (7)` if `o == label` else `TNEG (-8)`:
```
E_h[o] = clip(a_h[o] - T[o], -8, 7)             output head error (unmasked, as in v1/v2)
G_H   += E_h (x) x_L
e      = clip(rshift(H^T * E_h, sb_head), -8, 7)
```
Then for `l = L-1` down to `0`:
```
d      = e, except d[i] = 0 where y_out[i] < -8 or y_out[i] > 7
G_B[l] += d (x) u
m_u    = B[l]^T * d                             in int32, length R
e_u    = clip(rshift(m_u, sb_b), -8, 7)
d_u[r] = e_u[r] * mask[b][l][r], except 0 where y_a[r] < -8 or y_a[r] > 7
G_A[l] += d_u (x) x_l
if l > 0:
    m_prev = W_base[l]^T * d + A[l]^T * d_u     in int32, length D
    e      = clip(rshift(m_prev, sb_in), -8, 7)
```
After summing over all `NB3 = 32` samples, every entry of every gradient tensor (`G_A[l]`, `G_B[l]`, `G_H`) is
clipped to `[-CLIP, CLIP]` (`CLIP = 256`).

### 4.4 Gradient Digest and Ticket

Over the trainable tensors in flat order `t = 0 .. 2*L` (`A[0], B[0], ..., A[L-1], B[L-1], H`):
```
lane[f mod 8] += (uint32(G_t[f]) ^ (f + 7919 * (t + 1)))   (mod 2^32)
```
`digest` is the 8 little-endian `u32` lanes (32 bytes), and `ticket = BLAKE3(digest || key)`.

## 5. Model Step and State Hashes

Identical to `spec/shares.md` section 6 and `spec/nebhash-v2.md` section 6:
- `A[l]` and `B[l]` step with `su_lora`; `H` steps with `su_head`.
- Trainable weights are packed 2 nibbles per byte (low nibble first) in flat trainable order:
  ```
  weights_hash  = BLAKE3("neb-model-weights" || pack(W_trainable))
  residual_hash = BLAKE3("neb-model-residual" || LE32(R_trainable))
  ```
- Frozen base weights are initialized (in synthetic tests) or loaded (on mainnet) independently and committed by:
  ```
  base_hash = BLAKE3("neb-base-weights" || pack(W_base))
  ```

## 6. Test Datasets and Vectors

For `LORA_TINY` (`arch_id = 1`), the deterministic test base weights are drawn from
`ChaCha8(BLAKE3("neb-syn3-base" || arch_id))` using the same centered binomial mapping `(b&3)+((b>>2)&3)+((b>>4)&3)+((b>>6)&3)-6`
as model init. The synthetic dataset has `N = 2048` samples:
`P_c = ChaCha8(BLAKE3("neb-syn3-proto" || arch_id || c)) & 7`,
`x_i = clip(P_(i mod 2) + (noise_i & 3) - 1, 0, 7)` where `noise_i = ChaCha8(BLAKE3("neb-syn3-sample" || arch_id || le32(i)))`.

Test vectors in `vectors/nebhash3.txt` and `vectors/nebhash3_s1.txt` (`-DNEB3_SU_DROP=6`) verify 6 model steps of
`(32, 1, 3, 32, 17, 2)` shares with keys `BLAKE3("neb-test-share3" || arch_id || le32(s) || le32(j))`, cross-checked
bit-for-bit between `pow/ref/nebhash3.c` and `pow/oracle/nebhash3_oracle.py`.
