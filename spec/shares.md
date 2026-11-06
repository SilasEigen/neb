# NEB training shares: specification

This text specifies the proof of work of the NEB chain and the model step that it drives: how an attempt is computed, which attempts are shares, and how a set of shares trains a model that the chain owns. Sections 1 to 10 define the attempt, the step and the test vectors; section 11 defines how the chain carries the shares. The text is self-contained: every primitive that it uses is defined in sections 2 and 3. `docs/whitepaper.md` gives the motivation, the security argument and the measurements. The C reference (`pow/ref/shares.c`) implements exactly this text; the Python oracle (`pow/oracle/shares_oracle.py`) is written from this text only. Both replay `vectors/shares.txt` and the test-only variant `vectors/shares_s1.txt`. The CUDA implementation (`pow/gpu/shares_cuda.cu`) is checked against the same files by `pow/tests/check_shares_cuda.bat`.

## 1. Scope

A share is an attempt of the proof of work whose ticket is below an easier share target. This text defines what a share computes (its gradient and its ticket), and how the shares of one model step change the model. The share target, the block format and the dataset of the network are defined in section 11. The dataset is an abstract array of `N` samples (section 4); the test dataset of section 9 exists only to produce the test vectors.

## 2. Constants

| name | value | meaning |
|---|---|---|
| NIN | 784 | inputs per sample |
| NH | 128 | hidden units |
| NOUT | 10 | classes |
| NB | 16 | samples per share |
| S1 | 5 | forward shift of layer 1 |
| S2 | 5 | forward shift of layer 2 |
| SB | 3 | shift of the error propagated to layer 1 |
| SU1 | 15 | update shift of layer 1 |
| SU2 | 12 | update shift of layer 2 |
| CLIP | 256 | bound of every gradient entry of one share |
| TPOS | 7 | target of the true class |
| TNEG | -8 | target of the other classes |

The update shifts are set so that the model still trains when the gradients applied by a step were computed on a state one or two steps older (`docs/whitepaper.md`, section 7.2). With `SU1 = 13` and `SU2 = 10` the accuracy collapses after one step of delay.

### 2.1 Integer arithmetic

All arithmetic is integer: no floating point appears anywhere, and every sum is exact in int32 (the bounds are given where the sums are defined). `uint32` operations wrap modulo 2^32.

- `rshift(x, n) = (x + 2^(n-1)) >> n` for a signed int32 `x` and `1 <= n <= 15`, with an arithmetic shift (the sign is replicated). It rounds half up: `rshift(16, 5) = 1`, `rshift(-16, 5) = 0`, `rshift(-17, 5) = -1`, `rshift(-1, 5) = 0`. The largest operand in this text is `|z1| <= 43904`, so `x + 2^(n-1)` cannot overflow.
- `clip(v, lo, hi) = min(max(v, lo), hi)`.
- `tdiv(r, 2^u)` is the quotient of `r` by `2^u` rounded toward zero: `sign(r) * (|r| >> u)`.
- `LE32(x)` is the 4-byte little-endian encoding of a uint32; an int32 is encoded as its two's complement. "The uint32 read from bytes" is the inverse, in little-endian order.

## 3. Primitives

### 3.1 BLAKE3

`BLAKE3(x)` is the 32-byte output of BLAKE3 in its default hash mode (no key, no key derivation), for an input of any length. Inputs of up to 1024 bytes are one chunk; longer inputs use the chunk tree. The state identifiers of section 7 hash inputs of 50833 and 406546 bytes, so the chunk tree is required. A hash is written as the hex of its 32 bytes in order.

### 3.2 ChaCha8

`ChaCha8(key)` is a keystream of bytes for a 32-byte key, in the original layout of D. J. Bernstein with 8 rounds (4 double rounds). The 16 state words are: words 0 to 3 are the constants of "expand 32-byte k" (`0x61707865`, `0x3320646e`, `0x79622d32`, `0x6b206574`); words 4 to 11 are the key read as 8 little-endian uint32; words 12 and 13 are the low and high halves of the 64-bit block counter; words 14 and 15 (the IV) are 0. Block `c` (c = 0, 1, 2, ...) is the state after the rounds plus the input state, word by word modulo 2^32, with the 16 words written little-endian (64 bytes). The keystream is the blocks for c = 0, 1, 2, ... concatenated and consumed in order. A use that needs `n` bytes takes the first `n` bytes; the others are never used.

### 3.3 Tags

A tag is a string of ASCII bytes without terminator, placed in front of the other parts of a hash input with no separator. The tags of this text are `neb-model-init` (section 4), `neb-model-weights` and `neb-model-residual` (section 7), `neb-syn-proto` and `neb-syn-sample` (section 9), and `neb-test-share` (section 10).

### 3.4 Signed nibbles and packing

A signed nibble is a 4-bit two's complement value: the bits `v` in 0..15 mean `v` for `v < 8` and `v - 16` otherwise (range -8..7). A value `w` in -8..7 is stored as the nibble `w & 15`. A matrix whose entries are in -8..7 and are numbered `e = row * cols + column` (row-major) is packed into `ceil(M / 2)` bytes for `M` entries: element `e` is the low nibble (bits 3..0) of byte `e / 2` when `e` is even and the high nibble (bits 7..4) when `e` is odd. `pack(W1)` and `pack(W2)` are the packings of the two weight matrices.

### 3.5 Key of an attempt

An attempt has a 32-byte key. In the proof of work the key is `BLAKE3(header)` of the 80-byte block header that the attempt tries (section 11.1); the key of a share is derived in the same way from the share header of its record (section 11.2). The key selects the samples of the attempt (section 5.1) and is hashed into the ticket (section 5.4); it is the only input of an attempt besides the model and the dataset.

## 4. Model, state and dataset

The model is `W1` (NH x NIN) and `W2` (NOUT x NH) with entries in -8..7. The state is `(W1, W2, R1, R2)`, where the residuals `R1` and `R2` have the same shapes and hold int32 values with `|R_l| <= 2^SU_l` (section 6 keeps this invariant, and the initial state satisfies it).

The dataset has `N` samples (`N >= 1`). Sample `i` is `X_i`, NIN values in 0..7, and a label `c_i` in 0..9. The target vector is `T_i[o] = TPOS` if `o = c_i`, and `TNEG` otherwise.

**Initial state.** Given a 32-byte seed, `key = BLAKE3("neb-model-init" || seed)`, and `s` is the first NH * NIN + NOUT * NH = 101632 bytes of `ChaCha8(key)`. Byte `k` gives the weight `w = (b & 3) + ((b >> 2) & 3) + ((b >> 4) & 3) + ((b >> 6) & 3) - 6` (range -6..6), where `b = s[k]`. Bytes `0 .. NH*NIN - 1` fill `W1` in row-major order, the next NOUT*NH bytes fill `W2`. `R1 = 0` and `R2 = 0`.

## 5. A share

An attempt has a 32-byte key (section 3.5). It is a share when its ticket (section 5.4) is at most the share target of section 11.2.

### 5.1 Samples of a share

`u` is the first 64 bytes of `ChaCha8(key)`, read as 16 words `u_j = LE32(u[4j .. 4j+3])`. The share uses the samples `idx_j = floor(u_j * N / 2^32)`, for `j` in 0..NB-1, in this order (repeats are allowed). Let `X[b]` be `X_(idx_b)` and `T[b]` the target of that sample.

### 5.2 Gradient

With the model `(W1, W2)` of the current state and `b` in 0..NB-1:

```
z1[b][h] = sum over i of X[b][i] * W1[h][i]
y1[b][h] = rshift(z1[b][h], S1)             a1[b][h] = clip(y1[b][h], -8, 7)
z2[b][o] = sum over h of a1[b][h] * W2[o][h]
y2[b][o] = clip(rshift(z2[b][o], S2), -8, 7)
E2[b][o] = clip(y2[b][o] - T[b][o], -8, 7)
G2[o][h] = sum over b of E2[b][o] * a1[b][h]
m[b][h]  = sum over o of E2[b][o] * W2[o][h]
E1[b][h] = clip(rshift(m[b][h], SB), -8, 7)  if -8 <= y1[b][h] <= 7, else 0
G1[h][i] = sum over b of E1[b][h] * X[b][i]
```

Then every entry of `G1` and `G2` is replaced by `clip(g, -CLIP, CLIP)`. All sums are exact in int32: `|z1| <= 784 * 7 * 8`, `|z2| <= 128 * 8 * 8`, `|G1| <= 16 * 8 * 7` and `|G2| <= 16 * 8 * 8`.

### 5.3 Gradient digest

There is one set of eight lanes `D`, shared by both layers, which starts at zero. For each layer `l` in 1..2, and each entry of its clipped gradient with flat row-major index `f` (`G1[h][i]` has `f = h * NIN + i`, `G2[o][h]` has `f = o * NH + h`) and value `g`: `D[f mod 8] += (uint32)g ^ (uint32)(f + 7919 * l)`, where `(uint32)g` is the two's complement and `+` wraps modulo 2^32. `digest = LE32(D[0]) || ... || LE32(D[7])` (32 bytes).

### 5.4 Ticket

`ticket = BLAKE3(digest || key)` (64-byte input), read as a 256-bit little-endian integer when it is compared with a target. The share is valid for a share target when `ticket <= target`. The ticket depends on the model through the gradient, so a share is valid only for the state on which it was computed.

## 6. A model step

Input: a state and an ordered list of `n` shares (`1 <= n <= 8388479`) computed on that state. For each layer `l` (1 for `W1`, 2 for `W2`), with `SU = SU_l`:

```
GS_l = sum over the n shares of the clipped G_l of section 5.2
R_l  = R_l + GS_l
D_l  = clip(tdiv(R_l, 2^SU), -1, 1)
R_l  = R_l - D_l * 2^SU
R_l  = clip(R_l, -2^SU, 2^SU)
W_l  = clip(W_l - D_l, -8, 7)
```

all entry by entry. The sums are exact in int32: `|GS_l| <= n * CLIP`, and `|R_l + GS_l| <= 2^SU + n * CLIP < 2^31` for every `n` of this range (the largest `n` is set by layer 1, with `SU1 = 15`). The result does not depend on the order of the shares in the list.

The residual keeps the small gradients that the shift `2^SU` would round to zero. The clamp keeps it bounded, which also bounds the influence of an adversary over time.

## 7. State identifiers

```
weights_hash  = BLAKE3("neb-model-weights" || pack(W1) || pack(W2))
residual_hash = BLAKE3("neb-model-residual" || LE32(R1 entries, row-major) || LE32(R2 entries, row-major))
```

`pack(W1)` has 50176 bytes and `pack(W2)` has 640 bytes (section 3).

## 8. Bounds

One share changes an entry of `GS_l` by at most `2 * CLIP`, and so a step changes an entry of `W_l` by at most 1. If an adversary replaces `k` of the `n` shares of a step, each step entry of the model differs by at most `min(2, ceil(2 * k * CLIP / 2^SU))`; the proof is in `docs/whitepaper.md`, section 6.

## 9. Test dataset

The test vectors use a synthetic dataset of `N = 2048` samples. Class `c` has a prototype `P_c`: the first NIN bytes of `ChaCha8(BLAKE3("neb-syn-proto" || c))` (c as one byte), each `& 7`. Sample `i` has label `c_i = i mod 10` and pixels `X_i[p] = clip(P_c[p] + (n[p] & 3) - 1, 0, 7)`, where `n` is the first NIN bytes of `ChaCha8(BLAKE3("neb-syn-sample" || LE32(i)))`.

## 10. Test vectors

`vectors/shares.txt` is produced by `pow/tests/gen_share_vectors.c` from these rules, with the seed of 32 zero bytes. It has 24 steps, numbered `s = 0 .. 23`; step `s` has `n_s` shares, where `n_s` cycles through `32, 32, 32, 1, 32, 3, 32, 17` (so `n_0 = 32`). The key of share `j` (from 0) of step `s` is `BLAKE3("neb-test-share" || LE32(s) || LE32(j))`. Each step is computed on the state left by the preceding step. Format, lines in this order, LF endings, lower-case hex, indices in decimal:

```
neb-shares-vectors 1
init <weights_hash> <residual_hash>
share <s> <j> <digest> <ticket> <idx_0> ... <idx_15>        (n_s lines per step)
state <s> <n_s> <weights_hash> <residual_hash>              (after the step)
```

A hash, a digest and a ticket are written as the hex of their 32 bytes in order. The `init` line is the state of section 4.

### 10.1 Test-only variant

A step of `n` shares changes an entry of `R_l` by at most `n * CLIP`. The clip of `D` to +-1 and the clamp of the residual need `|R_l + GS_l| >= 2^(SU+1)`, and so `|GS_l| >= 2^SU`, because `|R_l| <= 2^SU`. The steps of the vectors have `n <= 32`, so `|GS_l| <= 8192`. With the production constants `2^SU1 = 32768` is above this bound: layer 1 never reaches the clip or the clamp in the vectors, and the 24 steps move only 6 of its weights. Layer 2 (`2^SU2 = 4096`) reaches the clip, the clamp and the saturation of the weights.

`vectors/shares_s1.txt` is the same procedure with `SU1 = 9` and `SU2 = 6` (`2^SU` is 512 and 64, far below 8192). In its 24 steps all three rules occur in both layers. Only the update shifts change: `S1`, `S2`, `SB` and `CLIP` keep their values, and the initial state, the share counts `n_s`, the keys and the sample indices are those of `vectors/shares.txt`, so the `init` line and the `share` lines of step 0 are identical in the two files. The C reference builds it with `-DNEB_SU1=9 -DNEB_SU2=6`, and the oracle replays it with `--su1 9 --su2 6`; this is for tests only and is never a production setting.

### 10.2 Hashes of the vector files

```
4df370e1f268bb2ca0b150a7ab503487227895fa617e1aa7b48db06a063b34fb  vectors/shares.txt
0b69199f393bf1d53563c3c2617b1d8c63bbe8eab951ef2f2df9ebb10b2349b0  vectors/shares_s1.txt
```

`pow/tests/check_share_vectors.sh` regenerates both files with the C reference and compares them byte for byte with these files, and the oracle replays them. A change of a constant or of a rule means different files and a different table. The files have LF line endings.

## 11. Chain integration

This section defines how the share step above becomes the proof of work and the training of a chain. `spec/node.md` refers to it. The constants are:

| name | value | meaning |
|---|---|---|
| SHARE_RATIO_LOG2 | 5 | the share target is the block target times 2^5 |
| MAX_SHARES_PER_BLOCK | 256 | share records one block lists |
| SHARE_RECORD_BYTES | 40 | bytes of a share record (section 11.2) |
| MODEL_DEPTH | 100 | levels below the tip whose states a node keeps |
| D | 1 | blocks between the state of a share and the state it updates, for K = 1 |
| K | number of datasets of the network | model slots of the chain state (2 on the testnet of protocol 3) |

### 11.1 Proof of work

The key of an attempt is the block id: for a block `B` of height `h` with header `H_B` (80 bytes, `spec/node.md` section 4.1), `key = BLAKE3(H_B)`. `B` has a valid proof of work when

```
ticket(M(P)[h mod K], D[h mod K], BLAKE3(H_B)) <= target(B.bits)
```

where `P` is the parent of `B`, `ticket` is the ticket of section 5.4 evaluated with the weights of the model of slot `h mod K` in the state `M(P)` (section 11.2) and the samples of the dataset `D[h mod K]` of that slot (section 11.6), and the target is the compact target of `spec/node.md` section 4.1, compared as 256-bit little-endian integers. The genesis block carries no proof. A header alone cannot be checked, because `M(P)` depends on the bodies of the ancestors of `B`.

### 11.2 States and share lists

A network has K >= 1 datasets `D[0] .. D[K-1]` (section 11.6), and the state of a block is a list of K models, one per dataset: `M(B) = (M(B)[0], ..., M(B)[K-1])`, each a state of section 4. The block of height `h` works on slot `h mod K`. With K = 1 the state is one model and the rules below are those of protocol 2.

The state of the genesis block is the initial state of section 4 for each slot: with K = 1, `M(genesis)[0] = init(genesis id)`; with K >= 2, `M(genesis)[i] = init(BLAKE3("neb-model-slot" || genesis id || LE32(i)))`.

Let `B` be a block, `P` its parent and `PP` the parent of `P` (`PP = P` when `P` is the genesis block). `B` lists a sequence of share records `r_1 < r_2 < ... < r_k` (`k <= MAX_SHARES_PER_BLOCK`). A record has 40 bytes, `body_root (32) || LE32(time) || LE32(nonce)`, and the records are in strictly increasing order as byte strings, so a list has no duplicate and one encoding. The block of height 1 lists none.

The share header of a record `r` is the 80-byte header of `spec/node.md` section 4.1 with version 1, `parent` equal to the id of `PP`, the `body_root`, `time` and `nonce` of `r`, and the `bits` of `P`. `r` is a valid share of the round of `P` when, with `t = (h - 1) mod K` the slot of `P` (height `h - 1`),

```
ticket(M(PP)[t], D[t], BLAKE3(share header of r)) <= share_target(P.bits)
share_target(bits) = min(target(bits) * 2^SHARE_RATIO_LOG2, 2^256 - 1)
```

A share is bound to the parent of the round, not to the header of `P`. Any header whose parent is `PP`, whatever its coinbase, share list or time, and whether or not it became a block, is a share of the same round. The header of `P` and the headers of competing blocks with the parent `PP` are valid shares of it: their ticket is at most the block target, which is below the share target. A round yields about 32 shares for one block, from the attempts of all miners on `PP`.

The state of `B` is `M(B) = M(P)` when `k = 0`. When `k >= 1`, the key of share `j` is `BLAKE3(share header of r_j)`; the clipped gradients of the `k` shares are computed on `M(PP)[t]` with the samples of `D[t]` (section 5.2) and summed in int32; the model step of section 6 applies this sum to `M(P)[t]`, which gives `M(B)[t]`; every other slot of `M(B)` equals that of `M(P)`. A share trains the model it was mined on. With K = 1 the shares are computed on the state one block older than the state they update; with K >= 2, `M(PP)[t] = M(P)[t]` (the parent of `B` updated another slot), so a share is computed on the model it updates. The sum does not depend on the order of the list.

The hash of a state is the weights hash of its model when K = 1, and `BLAKE3("neb-models" || weights_hash(M[0]) || ... || weights_hash(M[K-1]))` (section 7) when K >= 2.

### 11.3 Body and commitment

The body of a block is its transactions followed by its share list:

```
body = tx_count (varint) || tx_1 || ... || tx_count || k (varint) || r_1 || ... || r_k
shares_root = BLAKE3(r_1 || ... || r_k)          (BLAKE3 of the empty input when k = 0)
body_root = H("body", LE32(tx_count), merkle_root, shares_root)
```

where each `r_j` is a 40-byte share record (section 11.2), and with `merkle_root` as in `spec/node.md` section 4.2. A decoder rejects a list of more than 256 records, a list that is not strictly increasing as byte strings, a count in a longer form than the shortest and any trailing byte.

### 11.4 States kept

A node keeps the state of every connected block of the last `MODEL_DEPTH + 1` levels of its best chain, and of the side-branch blocks it has checked in the same levels. An implementation may keep the weights and the residuals only. When a state is missing it is rebuilt by replaying the blocks from the genesis block. After a reorganisation to a shorter chain the kept states are rebuilt the same way before the next block is checked. A reorganisation of at most 60 blocks (`spec/node.md` section 6.5) finds the state of the common ancestor and of its parent among the kept states, and gives the state that a replay of the replacing chain would give.

### 11.5 Work service

The node mines a job for a child of the tip `P` (height `h`) on `M(P)[h mod K]` and sends it in two frames, `spec/node.md` section 7.1:

- the `model` frame (type 7, node to miner): `slot (1) || weights_hash (32) || pack(W1) || pack(W2)` (50 816 bytes of weights), the slot `h mod K` and the weights of `M(P)[h mod K]` with the hash of section 7. The miner evaluates the attempts of the next job on the samples of the dataset of that slot. The node sends it immediately before every job, in the same write: when a miner connects, when the best block changes and after a rejected submission;
- the job (type 2), which carries the block target, the share target, the share-list commitment `shares_root` of the block and the data to build the coinbase and the header (`spec/node.md` section 7.1). The ticket of an attempt is evaluated on the model of the last `model` frame.

The reference header of a job is the header the miner would build with an extranonce of `extranonce_size` zero bytes, the time `time_start` and a nonce set to 0. A miner makes its attempts on the reference header, rolling the nonce: an attempt whose ticket on `M(P)` meets the share target of the job is a share, and one that meets the target of the job is a solution. A share is bound to the parent `P`, not to the header (section 11.2), so a miner may change job, extranonce or time at any moment without losing the shares it found.

- The `share` frame (type 6, miner to node): `job id (u64) || nonce (u32)`. The node builds the share record of the reference header of the job with this nonce (`body_root`, `time`, `nonce`), evaluates its ticket on the model of the job and keeps the record in its share store when the ticket meets the share target. It does not answer; a nonce whose ticket is above the share target is dropped without a reply. The node remembers the 64 newest jobs: a `share` or `submit` frame that names an older job is dropped (a `submit` gets result code 1).
- The `submit` frame (type 4) carries the extranonce, the time and the nonce of a solution (`spec/node.md` section 7.1).
- The share store holds, for the tip `P` and for the parent `Q` of the tip, the share records whose share header has that parent and whose ticket on the model of that parent meets the share target. Records come from the node's miners and from peers (`shares`, `spec/node.md` section 8.2); each is evaluated once. The store keeps at most 1024 records per parent.
- The job for a child of the tip `P` lists, as its share list, the records kept for the parent `Q` of `P`, checked again against `M(Q)`, sorted as byte strings and at most 256. A node lists the shares found by every miner whose records reach it, not only those of its own miners. The records kept for `P` itself are listed by the grandchildren of `P`.

### 11.6 Dataset file

The samples of a network are a file that the network parameters pin by its BLAKE3 hash. A dataset is the 4 bytes `NMD1`, the sample count `N` as a little-endian u32 (`N >= 1`), then `N` records of 785 bytes: 784 values in 0..7 and the label in 0..9. The file is one dataset (K = 1), or a set of K datasets, `NMDS || LE32(K) || dataset 0 || ... || dataset K-1` with `1 <= K <= 16`, in slot order. A node refuses to start on a file with another hash, and a file whose size, values or labels do not fit the format, or that has bytes after its last dataset, is invalid.

The testnet of protocol 3 has K = 2: slot 0 holds the transit signals of the Kepler Q1-Q17 DR24 TCE table of the NASA Exoplanet Archive (label 1 for a planet candidate, 0 for an astrophysical false positive or a non-transiting phenomenon; 583 values of a global view of the folded light curve and 201 of a local view around the transit, each the depth of the dip below the median in 0..7), slot 1 the retinal OCT scans of OCTMNIST (MedMNIST v2, CC BY 4.0; labels 0 to 3; value `clip((x - 30) >> 4, 0, 7)` of the 8-bit pixel `x`, the same number of scans per label). `tools/make_datasets.py` builds the file; 15 % of the Kepler stars and the OCTMNIST test split are kept out of it, to measure the models on samples they never trained on. The pixels of the MNIST network file are `floor(x / 32)` of the 8-bit pixels `x`, in the order of the 60 000 training images.
