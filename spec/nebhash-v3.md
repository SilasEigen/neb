# nebhash v3: anti-shortcut MoE frozen INT8 backbones with on-chain integer LoRA

Status: draft proposal (RFC) for protocol activation via `NMD3` datasets. It defines the attempt, sequential
data-dependent expert routing, full-dimension input + bottleneck dropout, forward trace commitment, LoRA gradient,
and model step in exact integer arithmetic. Everything not redefined here follows `spec/shares.md`.

## 1. Security Model & Why `v3` Changes `v2`

A Proof-of-Useful-Work training algorithm faces four adversarial GPU-kernel shortcuts:

1. **Per-sample gradient table precomputation (`nebhash2_table_cuda.cu`)**: In `v2`, sample gradients do not depend
   on the nonce `key`, allowing miners to evaluate the dataset once per block and reduce mining to table additions.
2. **Nibble-packing / lossless VRAM compression**: If frozen base weights are restricted to 4-bit values (`-8..7`),
   an 8 GB model packs into 4 GB of VRAM and fits onto 6 GB legacy mining cards (e.g., CMP 30HX / GTX 1660S).
3. **Mega-batch PCIe layer swapping**: If every nonce traverses the exact same static sequence of large weight layers
   `W_0 -> W_1 -> ...`, a 6 GB card can queue 65,536 nonces and stream one layer at a time over PCIe from host RAM,
   amortizing PCIe transfer time and bypassing the VRAM floor.
4. **Lossy / approximate mining**: If `ticket` only commits to heavily quantized/clipped final gradients, a miner can
   use approximate math (e.g., pruning small channels or skipping weak early-layer gradients) and still produce valid
   tickets a large fraction of the time while doing less work per attempt.

`nebhash v3` closes all four attack vectors simultaneously while preserving genuine on-chain LoRA training, compact
~97 KB P2P work frames, bit-exact integer determinism, and millisecond-class single-core CPU verification:

- **Full-range `int8` (`-128..127`) frozen expert pool (`10.50 GiB` on mainnet)**: Mainnet weights use the full
  8-bit signed range (`E = 10,752` experts of `1024 x 1024` bytes = `10.50 GiB`), preventing 4-bit nibble packing
  and exceeding 6 GB / 8 GB VRAM even under entropy coding.
- **Sequential data-dependent MoE routing + `NB3 = 64` Tensor-Core batching**: All `NB3 = 64` samples in a share run
  together as a dense `[64, D] x [D, D]` INT8 GEMM (arithmetic intensity ~128 OP/byte, saturating `sm_80+` `mma`
  Tensor Cores), while the expert index `e_l` for layer `l >= 1` is derived from `BLAKE3(key || l || x_l)`. Because
  different nonces jump to unpredictable experts across the `10.50 GiB` pool at every layer, mega-batch PCIe swapping
  is defeated.
- **75% Input Feature Dropout (`D` bits) + 50% LoRA Bottleneck Dropout (`R = 16` bits)**: Derived from `key` per
  sample and per layer, creating `2^(NB3 * L * D)` mask entropy and defeating any lookup-table decomposition while
  acting as standard dropout regularization.
- **Exact `int32` Forward Trace Commitment (`trace_digest`)**: `ticket = BLAKE3(grad_digest || trace_digest || key)`
  commits to every unshifted `int32` inner product (`z_base`, `z_a`) and expert index `e_l`. Any single-LSB
  approximation or skipped computation triggers a 50% bit-flip avalanche in `ticket`.

## 2. Datasets (`NMD3`)

A v3 dataset begins with the 4-byte magic `NMD3`, followed by:
- `arch_id`: little-endian `u32` (`1` for `LORA_TINY`, `2` for `QWEN3_8B_LORA`),
- `base_hash`: 32 bytes (`BLAKE3("neb-base-weights" || raw_int8_bytes(W_base))`),
- `N`: little-endian `u32` (`N >= 1`), the sample count,
- `N` records of `D + 1` bytes: `D` input values in `0..7` and the class label in `0..1`.

An unknown `arch_id`, an input value above `7`, or a label above `1` makes the dataset invalid.

## 3. Architectures

Each `NMD3` architecture has hidden dimension `D`, LoRA rank `R = 16`, depth `L` adapted blocks, frozen expert pool
size `E` (each expert `W_base[e]` has shape `[D][D]` with `int8` entries in `-128..127`), batch size `NB3 = 64`, and
`NOUT = 2` output classes.

Trainable weights (entries in `-8..7`, updated on-chain every block) are shared across routed experts at each depth
`l = 0 .. L-1` (representing domain/task LoRA adapters over the frozen MoE backbone), stored in flat order:
`A[0] (R x D), B[0] (D x R), ..., A[L-1] (R x D), B[L-1] (D x R), H (NOUT x D)`.

| Parameter | `LORA_TINY` (`arch_id = 1`, CI & vectors) | `QWEN3_8B_LORA` (`arch_id = 2`, mainnet profile) |
|---|---|---|
| `D` / `R` / `L` / `E` | `64` / `16` / `3` / `16` | `1024` / `16` / `6` / `10,752` |
| `NB3` / `NOUT` | `64` / `2` | `64` / `2` |
| frozen `int8` base size | `64 KiB` (`65,536` bytes) | **`10.50 GiB`** (`11,274,289,152` bytes, Qwen3-8B slices) |
| trainable weights (packed) | `6,272` (`3,136` bytes) | `198,656` (**`99,328` bytes**, ~97 KiB) |
| `s_a` / `s_lora` / `s_out` | `5` / `5` / `9` | `7` / `7` / `11` |
| `sb_b` / `sb_in` | `4` / `9` | `4` / `11` |
| `s_head` / `sb_head` | `5` / `2` | `7` / `2` |
| `su_lora` / `su_head` | `14` / `12` | `16` / `14` |

## 4. A Share

### 4.1 Sample Indices and Key-Derived Dropout Stream

Given the 32-byte share `key`:
1. Draw `NB3 = 64` sample indices from `ChaCha8(key, 4 * NB3)` via `(le32(u_j) * N) >> 32`.
2. Derive `drop_key = BLAKE3("neb-lora-drop" || key)` and generate `ChaCha8(drop_key)`:
   - First `ceil(NB3 * L * D * 2 / 8)` bytes (`in_stream`) encode 2-bit pairs for input feature dropout: for sample `b`,
     layer `l`, feature `j`, let `idx2 = (b * L + l) * D + j`, `pair = (in_stream[idx2 >> 2] >> ((idx2 & 3) * 2)) & 3`.
     `mask_in[b][l][j] = 1` if `pair != 0` else `0` (exact 75% retention).
   - Next `ceil(NB3 * L * R / 8)` bytes (`u_stream`) encode 1-bit bottleneck dropout: for `bit = (b * L + l) * R + r`,
     `mask_u[b][l][r] = (u_stream[bit >> 3] >> (bit & 7)) & 1` (exact 50% retention).

### 4.2 Forward Pass, Sequential MoE Routing, and Trace Commitment

Initialize 8 trace accumulator lanes `T_lane[0..7] = 0` (`uint32`) and counter `tc = 0`. Helper `trace_mix(val)`:
```
T_lane[tc mod 8] += uint32(val) ^ (uint32(tc) * 0x9E3779B9u + 0x7F4A7C15u)   (mod 2^32)
tc += 1
```

For sample `b = 0 .. NB3-1`, center the raw input `0..7` around zero: `x_0[b][j] = int8(raw_x[b][j]) - 4` (in `-4..3`).

For layer `l = 0 .. L-1`:
1. **Data-dependent expert selection**:
   - If `l == 0`: `rh = BLAKE3("neb-moe-route" || key || u8(0))`
   - If `l > 0`: `rh = BLAKE3("neb-moe-route" || key || u8(l) || bytes(x_l[0..NB3-1][0..D-1]))`
   - `e_l = (uint64(le32(rh[0..4])) * E) >> 32`
   - Call `trace_mix(e_l)`.
2. **Masked forward computation** (for `b = 0 .. NB3-1` in order):
   - `x_m[b][j] = x_l[b][j] * mask_in[b][l][j]`
   - For `r = 0 .. R-1`:
     - `z_a = sum_j A[l][r][j] * x_m[b][j]` (`int32`), call `trace_mix(z_a)`
     - `y_a[b][l][r] = rshift(z_a, s_a)`
     - `u[b][l][r] = clip(y_a[b][l][r], -8, 7) * mask_u[b][l][r]`
   - For `o = 0 .. D-1`:
     - `z_base = sum_j W_base[e_l][o][j] * x_m[b][j]` (`int32`), call `trace_mix(z_base)`
     - `z_lora = sum_r B[l][o][r] * u[b][l][r]` (`int32`)
     - `y_out[b][l][o] = x_l[b][o] + rshift(z_base + z_lora * (1 << s_lora), s_out)` (Transformer residual block)
     - `x_{l+1}[b][o] = clip(y_out[b][l][o], -8, 7)`

Then the output head for `b = 0 .. NB3-1`:
```
z_h[b][o] = sum_j H[o][j] * x_L[b][j]              call trace_mix(z_h[b][o]) for o = 0 .. NOUT-1
a_h[b][o] = clip(rshift(z_h[b][o], s_head), -8, 7)
```
`trace_digest` is the 8 little-endian `u32` values `T_lane[0..7]` (32 bytes).

### 4.3 Backward Pass and Gradients

For each sample `b = 0 .. NB3-1` with target `T[o] = TPOS (7)` if `o == label` else `TNEG (-8)`:
```
E_h[o] = clip(a_h[b][o] - T[o], -8, 7)
G_H   += E_h (x) x_L[b]
e      = clip(rshift(H^T * E_h, sb_head), -8, 7)
```
For `l = L-1` down to `0` (using the routed expert `e_l` and masked input `x_m[b]` from the forward pass):
```
d      = e, except d[o] = 0 where y_out[b][l][o] < -8 or y_out[b][l][o] > 7
G_B[l] += d (x) u[b][l]
e_u    = clip(rshift(B[l]^T * d, sb_b), -8, 7)
d_u[r] = e_u[r] * mask_u[b][l][r], except 0 where y_a[b][l][r] < -8 or y_a[b][l][r] > 7
G_A[l] += d_u (x) x_m[b]
if l > 0:
    m_prev[j] = ((W_base[e_l]^T * d)[j] + (A[l]^T * d_u)[j]) * mask_in[b][l][j]
    e[j]      = clip(d[j] + rshift(m_prev[j], sb_in), -8, 7)
```
Every entry of `G_A[0..L-1]`, `G_B[0..L-1]`, `G_H` is clipped to `[-NEB3_CLIP, NEB3_CLIP]` (`NEB3_CLIP = 1024`, scaled
4x from `256` for `NB3 = 64`).

### 4.4 Digest and Ticket

`grad_digest` (32 bytes) is computed over the `2*L + 1` trainable tensors exactly as in `nebhash v2` (`D[f mod 8] += uint32(c) ^ (f + 7919 * (t + 1))`).
The share `digest` exposed to work frames is `BLAKE3(grad_digest || trace_digest)` (32 bytes), and:
```
ticket = BLAKE3(digest || key)
```
Thus `ticket` remains a standard 32-byte hash of `(digest || key)` compatible with existing pool/node framing while
cryptographically binding both the exact clipped gradient and every intermediate forward inner product and MoE route.

## 5. Model Step and State Hashes

As in `spec/shares.md` section 6 using exact 64-bit intermediate sum `v = int64(r[f]) + int64(gs[f])` so no input
can ever trigger signed 32-bit overflow undefined behavior:
- `A[l]` and `B[l]` step with `su_lora`; `H` steps with `su_head`.
- `weights_hash = BLAKE3("neb-model-weights" || pack4(W_trainable))`
- `residual_hash = BLAKE3("neb-model-residual" || LE32(R_trainable))`
- `base_hash = BLAKE3("neb-base-weights" || raw_int8_bytes(W_base))`

## 6. Synthetic Test Vectors (`LORA_TINY`)

For offline CI and deterministic vector verification (`arch_id = 1` only; mainnet uses real quantized Qwen3-8B weights
committed by `base_hash`), `W_base` (`65,536` full-range `int8` bytes) is drawn directly as raw signed bytes from
`ChaCha8(BLAKE3("neb-syn3-base" || arch_id))`. `vectors/nebhash3.txt` and `vectors/nebhash3_s1.txt` (`-DNEB3_SU_DROP=6`)
record 6 steps of `(32, 1, 3, 32, 17, 2)` shares verified bit-for-bit by `pow/ref/nebhash3.c` and `pow/oracle/nebhash3_oracle.py`.
