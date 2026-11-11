# nebhash v2: convolutional models

Status: active on the testnet of protocol 3 from height 8800 (slot 0, KEP1) and 8801 (slot 1, RET2), by the
upgrades of `spec/shares.md` section 11.7. It defines the attempt, the gradient and the model step of two small
convolutional networks, one per model slot of the testnet, in the integer arithmetic of `spec/shares.md`. Everything
that this text does not change is as in `spec/shares.md`: the key of an attempt (3.5), the samples of a share (5.1),
the ticket (5.4), the model step rule (6), the chain integration (11) and the dataset upgrades (11.7).

## 1. What changes

| | nebhash v1 (`spec/shares.md`) | nebhash v2 |
|---|---|---|
| network | 784-128-10, one hidden layer | one architecture per slot (section 3) |
| layers | 2 linear | convolutions, max pooling and linear layers |
| outputs | 10 | 2 |
| inputs | 784 values in 0..7 | 784 (Kepler) or 4096 (retina, 64 x 64) values in 0..7 |
| weights | 101 632 | 26 000 (Kepler) or 22 280 (retina) |
| forward multiply-accumulates per sample | 101 632 | 180 800 (Kepler) or 680 000 (retina) |

A slot moves to v2 by an upgrade of `spec/shares.md` section 11.7 whose dataset is a v2 dataset (section 2): its
architecture id selects the network of section 3. The constants NB, CLIP, TPOS, TNEG and the rules of section 2.1 of
`spec/shares.md` are unchanged. An attempt costs more than in v1: about 2 ms of one CPU core for KEP1 and 7 ms for RET2
with the reference code, against 0.4 ms for v1.

## 2. Datasets

A v2 dataset is the 4 bytes `NMD2`, the architecture id as a little-endian u32 (1 for KEP1, 2 for RET2), the sample
count `N` as a little-endian u32 (`N >= 1`), then `N` records of `NIN + 1` bytes: the `NIN` inputs of the architecture
(784 for KEP1, 4096 for RET2) in 0..7 and the label in 0..1. A set file may mix `NMD1` and `NMD2` datasets
(`spec/shares.md` section 11.6); a model trains with the architecture of its dataset, and the model of a `NMD1` dataset
is the network of `spec/shares.md`. An unknown id, a value above 7 or a label above 1 makes the file invalid.

## 3. Architectures

A layer is a convolution or a linear map with integer weights in -8..7, no bias, followed by a right shift `S`, a clip
to -8..7 and, for some layers, a max pooling. The layers are numbered from 1 in the order of the tables; the weights of
a layer are a tensor `W[out][in][k]` (1D), `W[out][in][ky][kx]` (2D) or `W[out][in]` (linear), stored in row-major
order. `SB` is the shift of the error that a layer passes back to the output of the layer before it (none for the
layers that read the input). `SU` is the update shift of the layer (`spec/shares.md` section 6).

### 3.1 KEP1, Kepler transit signals (slot 0)

The input is the global view `g` (values 0..582) and the local view `l` (values 583..783) of `spec/shares.md` section
11.6, read as two 1D signals of one channel. Each view goes through its own branch.

| layer | input | operation | output | S | SB | SU |
|---|---|---|---|---|---|---|
| 1 | g: 1 x 583 | conv1d 8 filters, kernel 5, padding 2, stride 1; max pool 4 | 8 x 145 | 2 | - | 15 |
| 2 | 8 x 145 | conv1d 16 filters, kernel 5, padding 2; max pool 4 | 16 x 36 | 4 | 5 | 15 |
| 3 | l: 1 x 201 | conv1d 8 filters, kernel 5, padding 2; max pool 4 | 8 x 50 | 3 | - | 15 |
| 4 | 8 x 50 | conv1d 16 filters, kernel 5, padding 2; max pool 4 | 16 x 12 | 4 | 4 | 15 |
| 5 | 768 = layer 2 (576) then layer 4 (192), each channel-major | linear | 32 | 5 | 5 | 15 |
| 6 | 32 | linear, output layer | 2 | 4 | 3 | 12 |

### 3.2 RET2, retinal scans at 64 x 64 (slot 1)

The input is a 64 x 64 image of one channel, row by row.

| layer | input | operation | output | S | SB | SU |
|---|---|---|---|---|---|---|
| 1 | 1 x 64 x 64 | conv2d 8 filters, 3 x 3, padding 1, stride 2; max pool 2 x 2 | 8 x 16 x 16 | 3 | - | 16 |
| 2 | 8 x 16 x 16 | conv2d 16 filters, 3 x 3, padding 1; max pool 2 x 2 | 16 x 8 x 8 | 4 | 6 | 16 |
| 3 | 16 x 8 x 8 | conv2d 32 filters, 3 x 3, padding 1; max pool 2 x 2 | 32 x 4 x 4 | 5 | 6 | 16 |
| 4 | 512, channel-major | linear | 32 | 5 | 5 | 16 |
| 5 | 32 | linear, output layer | 2 | 4 | 3 | 13 |

The shifts were chosen on the initial states so that about 2 % of the values of each layer saturate, as in v1.

## 4. A share

### 4.1 Forward pass

For each sample `b` of the share and each layer, in table order:

```
z = W * a_in                          convolution (zero padding, the stride of the table) or matrix product, in int32
y = rshift(z, S)                      rshift(x, 0) is x
a = clip(y, -8, 7)
a_out = maxpool(a)                    for the layers with a pooling, otherwise a
```

`a_in` of a layer that reads the input is the input values. A pooling of size `p` keeps, for each window of `p` (1D) or
`p x p` (2D) consecutive positions starting at 0, the largest value; the positions after the last full window are
dropped. The **position of the maximum** is the first position of the window, in row-major order, that holds the
largest value. The output layer has no pooling and its `a` is `y2` of `spec/shares.md` section 5.2.

### 4.2 Gradient

The error of the output layer is `E[o] = clip(a[o] - T[o], -8, 7)` with `T` the target of the sample
(`spec/shares.md` section 4). Then, from the last layer to the first:

```
e        = the error on a_out of the layer (E for the output layer)
e_full   = e routed back through the pooling: each value goes to the position of its maximum, the others get 0
           (the identity for a layer without pooling)
d        = e_full,  except that d = 0 where y is outside -8..7       (not for the output layer: d = e_full)
G       += d (x) a_in                 the weight gradient: the correlation of d with the input, summed over the samples
m        = W^T * d                    the transposed convolution or the transposed matrix product, in int32
e_prev   = clip(rshift(m, SB), -8, 7) the error on a_out of the layer before (not computed for the layers that read
                                      the input)
```

As in v1, the output layer keeps its error where its `y` is outside -8..7: with a mask there, an output saturated on
the wrong side would never receive an error again, and simulations with large steps stop learning for good. For layer
5 of KEP1, `e_prev` is split back into the outputs of layers 2 and 4 in the order of the concatenation.
Every entry of every `G` is then replaced by `clip(g, -CLIP, CLIP)`.

All sums are exact in int32. The largest are the weight gradients of the first layers: at most
`16 x 1024 x 8 x 7 = 917 504` for layer 1 of RET2 and `16 x 583 x 8 x 7 = 522 368` for layer 1 of KEP1.

### 4.3 Gradient digest

As `spec/shares.md` section 5.3, over the layers `l = 1 .. L` of the architecture in table order, `f` being the flat
row-major index of the entry in the weight tensor of layer `l`.

## 5. A model step

As `spec/shares.md` section 6, layer by layer, with the `SU` of each layer. The bound of section 8 holds per layer.

## 6. Initial state and identifiers

The initial weights of a model of an architecture are drawn as in `spec/shares.md` section 4 from one keystream, in
layer order then row-major order inside each tensor; the residuals start at zero.

```
weights_hash  = BLAKE3("neb-model-weights" || pack(W_1) || ... || pack(W_L))
residual_hash = BLAKE3("neb-model-residual" || LE32(R_1 entries) || ... || LE32(R_L entries))
```

The packed weights take 13 000 bytes for KEP1 and 11 140 bytes for RET2; the model frame of the work service carries
them in place of the 50 816 bytes of v1.

## 7. Measured behaviour (simulation, not normative)

A simulation of the step rule on the testnet datasets, with 32 shares of 16 samples per step, gave:

| model | v1 network, chain rule | v2 architecture, chain rule | v2 architecture, ordinary float training |
|---|---|---|---|
| Kepler (test set, 2 333 signals) | 88.3 % at 2 000 steps | 90.8 % at 3 000 steps | 91.6 % |
| retina, normal or abnormal (balanced test, 500 scans) | about 78 % | 81 to 83 % from 5 000 to 8 000 steps | 92.0 % |

These figures come from one run each and move by about 1 point (Kepler) and 2 points (retina) between runs. A clip
mask on the output layer adds about 2 points on the retina and nothing on Kepler, but it can stop the training for good
(section 4.2), so the rule keeps the output layer unmasked.

## 8. Test datasets and vectors

The test dataset of an architecture has 2 048 samples. With `a` the architecture id as one byte, the prototype of
class `c` (0 or 1) is `P_c = ChaCha8(BLAKE3("neb-syn2-proto" || a || c)) & 7` over `NIN` bytes, the noise of sample
`i` is `n = ChaCha8(BLAKE3("neb-syn2-sample" || a || LE32(i)))` over `NIN` bytes, the sample is
`clip(P_(i mod 2) + (n & 3) - 1, 0, 7)` value by value and its label is `i mod 2`.

`vectors/nebhash2.txt` holds, for KEP1 then RET2, the state hashes of the initial model of the zero seed, then 6 model
steps of 32, 1, 3, 32, 17 and 2 shares on the test dataset, share `j` of step `s` keyed by
`BLAKE3("neb-test-share2" || a || LE32(s) || LE32(j))`: for each share its digest, its ticket and its sample indices,
and after each step the state hashes. `vectors/nebhash2_s1.txt` is the same with every update shift lowered by 6, so
that a few shares move the weights. `pow/tests/check_share_vectors.sh` rebuilds both files with the C reference
(`pow/ref/nebhash2.c`) and replays them with an independent Python oracle (`pow/oracle/nebhash2_oracle.py`).
