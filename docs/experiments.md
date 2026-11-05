# NEB: experiments and validation

This document holds the detail behind the results of the whitepaper (`docs/whitepaper.md`, section 7): the simulations, the
measurements on the node software, the studies, the bit-for-bit validation of the implementation and the throughput of the
CPU engine. The whitepaper keeps the protocol and the main results; every number it quotes is here with its method.

## 1. Kinds of results

- A **simulation** runs a model of the protocol with the scripts of `tools/` (sections 2 and 3).
- A **measurement on the node software** runs the code of this repository: nodes with their peer-to-peer layer and work
  service, the chain rules and the model, with miners simulated by a program that uses the work service as a real miner
  does (sections 4 and 5).
- A **study** was run outside this repository, with the exact engines or with an event simulation of the network. Its
  scripts are not part of the repository, and the text marks it as a study (section 6).

All runs used one machine. None comes from a public network; the testnet will provide those.

**Recorded outputs.** The raw outputs, the bench sources, the scripts and the validation logs of the runs of 2026-11-02 and
2026-11-04 are published as the release asset `neb-0.1.0-experiments.tar.gz` of version 0.1.0, listed in its signed
`SHA256SUMS`. Its `pools/scripts/check_tables.py` checks that the tables of sections 4 and 5 are the output of
`make_wp_tables.py` on those results. The throughput rows of section 8 are in `pow/bench/results/2026-11-02/`.

**First rule.** Several tables compare the rule of this version with an earlier design, called the first rule here, in which
a share was bound to the header that wins its round instead of the parent of the round. It was never released; the bench
built the same program against both to measure the difference.

## 2. Simulation: training and attacks

The scripts of `tools/` simulate the step on MNIST with a 784-128-10 network (101,632 weights), pixels quantised to 0..7,
and the 10,000 test images that the model never sees during training. They need numpy and download MNIST. They differ from
the protocol in two places: the nonce selects its 16 samples with splitmix64 instead of ChaCha8, and the initial model is
Gaussian instead of derived from the genesis block id. The model step is that of the whitepaper (section 4), applied without
delay in this section. Commands are run from the root of the repository; runtimes are wall-clock on a shared 32-core machine.

**Training.** `python tools/training_shares.py --steps S` runs `S` steps of 32 shares (the nonces of step `s` are
`s * 1000 + k`, `k = 0..31`).

| Command | Steps | Share nonces | Test accuracy | Runtime |
|---|---|---|---|---|
| `python tools/training_shares.py --steps 300` | 300 | 9,600 | 90.2 % | 22 s |
| `python tools/training_shares.py --steps 1000 --no-replay-check` | 1,000 | 32,000 | 93.2 % | 45 s |
| `python tools/training_shares.py --no-replay-check` | 3,000 | 96,000 | 96.0 % | 108 s |

One run each; the runtimes are upper bounds. The residual keeps the small gradients that plain rounding loses. The update
shifts are two bits larger than a faster choice (`SU1 = 13`, `SU2 = 10`) that reaches 97.3 % after 96,000 shares without
delay; the price of the slower step is 1.4 points at 96,000 shares, and in exchange the model tolerates the delay of
section 3.

**Clipping.** After 300 steps, `python tools/training_shares.py --steps 300 --clip C --no-replay-check` gives 90.2 % without
clipping (`--clip 0`), 90.2 % with `C = 512`, 90.2 % with `C = 256` and 90.1 % with `C = 128` (9 to 22 s each).

**Grinding attack.** The attacker controls 8 of the 32 shares of every step. For each of them it tries `M` nonces and keeps
the one whose gradient points the most against the honest direction (estimated on 1,024 images).
`python tools/training_shares.py --steps 300 --attack M --no-replay-check`, after 300 steps:

| Clip | Honest (22 s) | M = 32 (116 s) | M = 128 (406 s) |
|---|---|---|---|
| 256 | 90.2 % | 90.1 % | 90.1 % |

The damage is at most 0.1 point, within the run-to-run noise. The attacker can only choose among real images of the dataset,
so it cannot build a degenerate input. This is one heuristic attacker and one seed per cell.

**Stale shares.** The gradient of a share depends on the exact weights, so a share computed on another model state does not
give the same digest and ticket on the state it is checked against, and the verification rejects it.
`python tools/training_shares_attacks.py --experiment staleness` (76 s, 300 steps) shows what lost shares cost, and what
happens when gradients that are 1, 4 or 16 steps old are accepted:

| Case | Accuracy |
|---|---|
| 32 shares per step (reference) | 90.2 % |
| stale shares rejected, 16 shares left per step | 88.0 % |
| stale shares rejected, 8 left | 85.2 % |
| stale shares rejected, 4 left | 81.6 % |
| stale shares rejected, 1 left | 60.9 % |
| stale gradients accepted, half of the shares 1 step old | 90.1 % |
| stale gradients accepted, all shares 1 step old | 90.0 % |
| stale gradients accepted, half of the shares 4 steps old | 89.1 % |
| stale gradients accepted, all shares 4 steps old | 42.1 % |
| stale gradients accepted, half of the shares 16 steps old | 75.0 % |
| stale gradients accepted, all shares 16 steps old | 13.8 % |

Losing shares degrades the model gradually. Gradients one step old cost nothing measurable at these update shifts; 4 steps
old for all shares cost much more, and 16 steps old break the training. The protocol accepts one fixed delay and no other.

**Targeted poisoning.** The attacker wants the model to call other digits "3". For each of its shares it tries `M` nonces and
keeps the one whose 16 samples contain the most 3s. `python tools/training_shares_attacks.py --experiment targeted` (87 s,
300 steps):

| Run | Accuracy | Other digits called 3 | Recall of the 3 |
|---|---|---|---|
| honest | 90.2 % | 1.4 % | 90.6 % |
| 8 of 32 shares, 1 nonce (no grinding) | 90.3 % | 1.1 % | 88.9 % |
| 16 of 32 shares, 1 nonce (no grinding) | 90.1 % | 1.2 % | 88.4 % |
| 8 of 32 shares, 256 nonces | 90.0 % | 1.7 % | 92.2 % |
| 8 of 32 shares, 4,096 nonces | 89.5 % | 2.1 % | 92.8 % |
| 16 of 32 shares, 256 nonces | 89.6 % | 2.4 % | 94.1 % |
| 16 of 32 shares, 4,096 nonces | 88.5 % | 3.3 % | 94.7 % |

With 4,096 nonces the accuracy falls by 0.7 point with a quarter of the shares and by 1.7 points with half of them. The runs
without grinding differ by 0.2 point. In the chain each of the 4,096 candidates would have to be a valid share, about 128
block-equivalents of work per share (whitepaper, section 6).

**Half of the shares against the gradient.** The attacker controls 16 of the 32 shares of every step and tries 64 nonces
for each. After 300 steps the accuracy is 89.3 %, against 90.2 % without the attacker. Command:
`python tools/training_shares_attacks.py --experiment adaptive` (about 7 minutes, one run).

**Replay.** The float64 and int64 back ends give identical weights after 60 steps from the same nonces (printed by
`python tools/training_shares.py --steps 300`). The integer step is checked bit for bit against the vectors by
`pow/tests/check_share_vectors.sh`.

## 3. Simulation: delayed gradients

**Choice of the update shifts.** The step moves a weight by one unit as soon as its residual passes `2^SU`, and the residual
then empties at once. With `SU1 = 13` and `SU2 = 10`, one block of delay breaks the training: the accuracy falls from 97.3 %
to 61.2 % (2 seeds). Slowing the update by two bits (`SU + 2`) removes the problem.

Final test accuracy (%) after 96,000 shares, MNIST test set, mean [min-max] over the seeds. A block lists 32 shares and
applies one model step of 32 shares, computed on the model of `D` blocks earlier. The seeds change the share nonces only.

| Delay `D` (blocks) | `SU1 = 13`, `SU2 = 10` (2 seeds) | `SU1 = 15`, `SU2 = 12` (4 seeds) | `SU1 = 16`, `SU2 = 13` (4 seeds) |
|---|---|---|---|
| 0 | 97.3 [97.2-97.3] | 95.9 [95.8-96.0] | 94.4 [94.3-94.4] |
| 1 | 61.2 [47.7-74.7] | 95.7 [95.7-95.8] | 94.3 [94.3-94.3] |
| 2 | 44.2 [26.3-62.1] | 95.4 [95.2-95.7] | 93.4 [91.6-94.1] |
| 4 | 22.9 [18.9-26.9] | 93.7 [91.7-94.4] | 94.3 [94.1-94.6] |

The case of the chain is the row `D = 1` of the middle column: 95.7 %, 0.2 points below the same run without delay.

```
python tools/training_shares_delay.py --cache DIR --seeds 4 --per-block 32 --curves 1:32                    # SU1 = 15, SU2 = 12
python tools/training_shares_delay.py --cache DIR --seeds 4 --per-block 32 --su-shift 1 --curves 1:32       # SU1 = 16, SU2 = 13
python tools/training_shares_delay.py --cache DIR --seeds 2 --per-block 32 --su-shift -2 --curves 1:32      # SU1 = 13, SU2 = 10
```

`DIR` is a cache directory for MNIST outside the source tree. Each run takes about 9 minutes with 16 worker processes. The
tables come from runs that also listed 64 shares per block (`--per-block 32 64`, and 64 and 256 for `--su-shift -2`); the
32-share column does not depend on the other block sizes.

- With `SU + 2`, one block of delay costs 0.2 point and 2 blocks cost 0.5 point; 4 blocks cost 2.2 points on the mean, with
  one seed at 91.7 %.
- With `SU + 3`, a delay of up to 4 blocks costs at most 1.0 point on the mean, but the accuracy without delay is 1.5 points
  lower than with `SU + 2`. `SU + 1` was not measured.
- The price of `SU + 2` without delay is 1.4 points (95.9 % against 97.3 %); the curves still rise at 96,000 shares.
- The runs use whole-block delays, fixed and equal for all shares, on MNIST.

## 4. Availability of shares (node software)

With several pools, the number of shares that reach the model depends on the rule that binds a share to a round. Under the
first rule only the node that built the winning header could list the shares of its round, and shares were not relayed: with
`f_i` the share of the hash rate of pool `i`, the model receives `sum_i f_i^2 (1 + 31 f_i)` shares per block, that is
`32.0, 8.25, 1.44, 0.41` and `0.13` with `1, 2, 5, 10` and `20` equal pools. Under the rule of this version a share is bound
to the parent of its round, and a node lists the records its own miners find and those its peers relay with the `shares`
message (`spec/node.md` section 8.2).

**Bench.** The bench starts `N` nodes of this repository in one process, joined by TCP on the loopback. Each has a simulated
miner that holds a fraction of a total attempt rate and uses the work service of its node: a job and its model, the tickets
of the reference header, a `Share` frame for a share and a `Submit` frame for a solution. The lab network gives a block
every 2 s and keeps the ratio of 32 between the share target and the block target. The same program built against the first
rule gives the comparison. Each row is one run of 120 to 150 blocks; the number of shares of a round is geometric, so the
standard error of a row is 2 to 3 shares per block near 30 shares and 0.4 or less below 2 shares.

Shares listed per block, nodes joined without delay:

| pools | formula for the first rule | first rule | this version | listed / found |
|---|---|---|---|---|
| 1 | 32.00 | 35.04 | 30.1 | 1.01 |
| 2 | 8.25 | 8.41 | 30.6 | 0.96 |
| 5 | 1.44 | 1.40 | 32.0 | 0.99 |
| 10 | 0.41 | 0.63 | 30.7 | 0.98 |
| 20 | 0.13 | 0.09 | 29.8 | 0.95 |

Each first-rule row is within about one standard error of its formula. This version lists 30 to 32 shares per block at
every number of pools; the last column is the ratio of the shares listed to the shares the miners reported. The rows for 1,
2 and 5 pools ran for 120 blocks; the rows of this version for 10 and 20 pools ran for 150 blocks on an otherwise idle
machine.

Accuracy of the model after 300 blocks, on the first 10,000 samples of the MNIST dataset of the network, one run per cell:

| pools | first rule | this version |
|---|---|---|
| 1 | - | 89.7 % |
| 10 | 36.2 % | 89.9 % |
| 20 | 24.4 % | 89.4 % |

An earlier study with the exact engines gave 36.1 % for the first rule with 10 pools between blocks 200 and 300. Across
1,440 evaluation points of exact runs, accuracy is one function of the number of shares applied (mean absolute error
2.8 points), which is why the number of shares listed per block predicts the accuracy.

Delay on every link, 10 and 20 pools, 150 blocks per row, otherwise idle machine:

| pools | delay per link | shares listed per block | shares found per block | listed / found | block time (s) | blocks lost to forks |
|---|---|---|---|---|---|---|
| 10 | 0 ms | 30.7 | 31.2 | 0.98 | 1.99 | 4 of 150 |
| 10 | 50 ms | 34.9 | 37.4 | 0.93 | 2.05 | 17 of 150 |
| 10 | 100 ms | 30.6 | 41.0 | 0.75 | 2.05 | 35 of 150 |
| 20 | 0 ms | 29.8 | 31.3 | 0.95 | 2.05 | 9 of 150 |
| 20 | 50 ms | 30.1 | 37.6 | 0.80 | 2.10 | 35 of 150 |
| 20 | 100 ms | 33.5 | 49.1 | 0.68 | 2.16 | 54 of 150 |

The lab blocks last 2 s, so 100 ms per link loses 23 to 36 % of the blocks to forks, much more than 60 s blocks would
(100 ms against 60 s corresponds to about 3 ms against 2 s). The shares listed per block stay between 30 and 35 at every
delay. The shares found per block rise to 41 and 49 because miners also report the shares of rounds whose block lost a fork.

**Another dataset.** The same network and update constants on the 60,000 training images of Fashion-MNIST (quantised the
same way, same file format), one run per row, 1,000 blocks, accuracy on the first 10,000 samples:

| pools | node | shares listed per block | accuracy after 300, 600 and 1,000 blocks |
|---|---|---|---|
| 1 | this version | 30.1 | 61.4 / 76.0 / 80.7 % |
| 10 | this version | 32.2 | 74.7 / 79.3 / 82.0 % |
| 10 | first rule | 0.35 | 23.3 / 54.9 / 61.2 % |

Training on this dataset is slower and noisier, with constants chosen for MNIST: the 10-pool run is at 74.7 % after 300
blocks and 59.9 % at 401 blocks, and earlier runs of 300 blocks gave 41.3, 66.3 and 49.6 % for 1, 10 and 20 pools. The
difference between 1 and 10 pools is within that spread; the difference with the first rule is not. The blocks list 30 to
33 shares at every number of pools.

## 5. Deviating pools (node software)

One pool among 10 deviates over 300 blocks, on the MNIST dataset of the network. It reports none of its shares
(`withhold`); it reports only the quarter of its shares that push the model most toward one class, chosen by the sum of the
clipped gradient on the output weights of that class (`skew`); or it does that and its links drop the `shares` messages, so
that it lists only its own shares when it finds a block and nobody lists its shares (`skew-private`).

| run | behaviour of pool 0 | power asked | share of the attempts reached | kept share | shares listed per block | accuracy at 100, 200, 300 blocks | samples predicted as class 3 |
|---|---|---|---|---|---|---|---|
| adv-honest-n10 | honest | - | - | - | 35.1 | 85.2 % / 87.2 % / 88.1 % | - |
| final-honest | honest | - | 10 % | - | 28.1 | 85.7 % / 88.4 % / 89.3 % | 10.7 % |
| rep-honest-a | honest | - | - | - | 33.7 | 85.6 % / 88.7 % / 87.4 % | - |
| rep-honest-b | honest | - | - | - | 33.2 | 85.9 % / 88.4 % / 88.8 % | - |
| strong-honest | honest | - | - | - | 30.8 | 82.6 % / 88.3 % / 90.0 % | 10.0 % |
| adv-skew-0.10 | skew | 10 % | ~9 % | 25 % | 29.0 | 84.8 % / 88.4 % / 89.3 % | - |
| final-skew-0.33-keep0.05 | skew | 33 % | 30 % | 5 % | 24.0 | 81.1 % / 87.6 % / 89.8 % | 9.9 % |
| adv-skew-private-0.10 | skew-private | 10 % | ~8 % | 25 % | 26.0 | 86.1 % / 88.2 % / 89.3 % | - |
| final-skew-private-0.25-keep0.25 | skew-private | 25 % | 25 % | 25 % | 20.3 | 85.7 % / 88.0 % / 89.0 % | 10.2 % |
| final-skew-private-0.33-keep0.05 | skew-private | 33 % | 33 % | 5 % | 13.1 | 81.7 % / 85.0 % / 86.9 % | 10.1 % |
| final-skew-private-0.33-keep0.25 | skew-private | 33 % | 28 % | 25 % | 16.3 | 83.7 % / 86.5 % / 88.0 % | 9.8 % |
| final-withhold-0.33 | withhold | 33 % | 30 % | - | 23.0 | 85.2 % / 87.7 % / 88.6 % | 9.8 % |

The run names are those of the recorded outputs. One run per row. The five honest runs end at 87.4 to 90.0 % after 300
blocks (mean 88.7 %, standard deviation 1.0 point), so a difference of one or two points is within the spread between runs.
A pool asking for 25 % or 33 % of the attempts reaches it only in the `final-*` rows, which ran on 3 cores; on 6 cores one
miner thread is limited to about 17 %, and those runs are kept only for the honest pools and the pool of 10 %.

Choosing shares by their bias does not move the model toward the class: the samples predicted as class 3 stay between 9.8
and 10.2 % with a deviating pool (10.0 to 10.7 % in the honest runs; class 3 is 10.2 % of the samples), even when the pool
holds 30 % of the attempts and reports only 5 % of its shares. Withholding every share of a pool with 30 % of the attempts
costs nothing visible (88.6 %). The largest effect is the censoring producer (`skew-private`) with a third of the attempts:
a third of the blocks list only its own few shares, so a block lists 13 shares on average instead of 30, and the accuracy
trails the honest runs by about 3 points after 200 blocks (85.0 %) and by 1.8 points after 300 blocks (86.9 %). The
training slows down and shows no bias. The adversaries are fixed strategies, not an optimisation of the loss of the model with
look-ahead.

## 6. Studies

These results were obtained outside this repository and are not reproduced from it.

- **A long chain.** One chain of 3,000 blocks with the exact engines, a delay of one block, real hashes and a single virtual
  miner: 96,022 shares applied and a test accuracy of 95.91 % (the simulation of section 3 gives 95.7 %). The number of
  shares per round is geometric (mean 32.3, standard deviation 32.3 over 4,050 rounds; two rounds exceeded the 256-share
  limit).
- **Residual accounting.** For every entry and every step, `2^SU * W' - R' = 2^SU * W - R - G + 2^SU * wc + lr`, where `wc`
  and `lr` are the charges that the bounds of the weight (`-8..7`) and of the residual (`+-2^SU`) discard. It holds for any
  grouping of the shares and any delay, and was checked on 7.7e10 entries of the whole domain of the consensus, on 139
  trainings of 16,384 shares and on two real 100-block chains with 88 replays of their shares, with three implementations.
  The residual bound discards charge when `|R + G| > 2T`, which cannot happen with at most 16 shares per step in layer 2 or
  128 in layer 1. On two chains of 100 blocks mined at the start of training under the first rule (accuracy 78.1 % and
  82.6 % at block 100), layer 2 discarded 25.8 % and 9.6 % of its gradient charge and layer 1 almost nothing.
- **Grouping and delay.** With the shares of those chains applied one by one, the final accuracy is 86.4 % and 87.0 %; with
  the chain's own groups and a delay of one block, 78.1 % and 82.6 %; with a delay of two blocks, 30 % and 45 %. In 139
  trainings of 16,384 shares with groups of 4 and 16 shares and delays of 0 to 2 blocks, the accuracy stayed within
  1.6 points of the one-by-one run; a group of 32 shares with a delay of 2 blocks lost 7.7 points at 4,096 shares.
- **One-step bound in practice.** An adversary with 10 %, 25 % or 33 % of the power finds about 3, 8 or 11 of the 32 shares
  of a round. The exact maximum difference of one step is 2 only when `k C >= 2^SU`, so Theorem 1 is not tight.
- **Mechanisms that keep the first rule.** Listing the solution nonce of the parent adds at most one share per block;
  relaying the shares of the producer with the block has the ceiling `1 + 31 sum_i f_i^2` (4.04 shares per block with 10
  equal pools); a collection after the fact gives 20.1 shares per block with a window of 30 s. Binding the share to the
  parent removes the ceiling.
- **Paying the shares.** An event engine of the network (1,620 runs for availability with ten seeds and 440 runs for
  incentives with twenty seeds, 1,000 blocks each) compared shares bound to the parent and not paid (V1), paid at a fixed
  price with a commission for the producer (V2), and paid from a budget of the reward shared among the listed shares (V3).
  All three give about 32 shares per block. Without payment the number listed depends on the default behaviour of the
  software (30 % of the power that does not relay lowers it from 32.2 to 23.5 with 10 pools). With V3, a pool that lists
  only its own shares earns 33 to 48 % more. With V2, listing only one's own shares stops paying for pools with 10 % of the
  power or less once the commission reaches 10 %, and still earns a pool with 34 % of the power 3.9 % more. This version
  does not pay shares.

## 7. Validation of the implementation

All comparisons are against the C reference `pow/ref/shares.c`. The logs are in the experiments archive (section 1). The
trained models and the dataset file used for the real-data rows are not in the repository.

| what is compared | with what | cases | result |
|---|---|---|---|
| model step of the Rust wrapper, both layers, the whole domain of the consensus | an independent 64-bit step | 77,312,098,320 entries (68,721,049,608 in layer 1, 8,591,048,712 in layer 2) | 0 differences, Windows (MSVC) and WSL (gcc 15) |
| AVX-512, dense and sparse, C level | the C reference | 14 models (6 synthetic, 8 trained states at 0 to 100,000 steps) on 3 synthetic datasets and the network dataset: 1,673,728 evaluations, 104,608 with the full clipped gradients | 0 differences; 2 negative controls (the engine built wrong on purpose) detected, with 2,408 and 1,775 differences |
| AVX-512, dense and sparse, Rust test | the C reference | 5 models on 3 synthetic datasets, 20,000 keys per case: 600,000 evaluations, 37,500 with gradients | 0 differences |
| CPU miner self-test | the C reference | network dataset: 960 tickets and 80 clipped gradients per variant | passed |
| recorded chains: the functions of the node and the independent Python oracle | each other, and the fixtures | `synthetic-stress` (41 blocks, 3,351 shares), `pools-synthetic` (4 pools, 40 blocks, 1,285 shares) and `pools-mnist` (3 pools, 60 blocks, 2,000 shares) | the same final weights and residual hashes and the same counts of cut and clipped entries |
| tests of the repository | themselves | 180 tests passed, 1 ignored (the exhaustive step), in release; CI also runs them in debug on Ubuntu | 0 failures, Windows (with the network dataset) and WSL |
| compilation | the compilers | AVX-512 unit with MSVC at warning level 4 and warnings as errors; the reference with gcc `-Wall -Wextra -Werror` | no warnings |

The checks in the repository: `pow/tests/check_avx512.sh`, `node/crates/neb-pow/tests` (boundaries and determinism),
`node/crates/neb-engines/tests`, `node/crates/neb-chain/tests/chain_fixtures.rs` and `pow/oracle/chain_oracle.py`. Each
differential test has a negative control: the AVX-512 code can be built wrong on purpose, and the test must fail.

The GPU kernels K0, K1 and K2 have differential tests against the reference (`pow/tests/check_shares_tc.sh`,
`cargo test --features cuda -p neb-gpuminer`) and a self-test in the miner (`neb-miner-gpu --selftest`). No GPU result is
recorded here.

## 8. Throughput of the CPU engine

Attempts per second (one attempt is one ticket), medians of repetitions, measured with `pow/bench/run_bench.sh` on one AMD
Zen 5 desktop processor (16 cores, 32 threads, AVX-512 VNNI). The dataset is the network dataset; the models are four
states of the network trained with the C reference for 0, 300, 3,000 and 100,000 steps (a more trained model has more
hidden units with no error, which the sparse variant skips). Every engine evaluates the same keys on the same model, and a
run is valid only if the checksum of all tickets is the same for all engines. The raw rows are in
`pow/bench/results/2026-11-02/`.

**Windows 11 (MSVC 19.44), 32 threads.**

| model (steps trained) | reference, 1 thread | AVX-512 dense, 1 thread | AVX-512 sparse, 1 thread | reference, 32 threads | AVX-512 dense, 32 threads | AVX-512 sparse, 32 threads | sparse with checks, 32 threads | sparse / reference at 32 threads |
|---|---|---|---|---|---|---|---|---|
| 0 | 2 692 | 82 982 | 93 224 | 42 451 | 1 125 309 | 1 280 848 | 1 401 175 | x30 |
| 300 | 2 658 | 85 575 | 97 975 | 39 659 | 1 084 127 | 1 219 640 | 1 390 807 | x31 |
| 3000 | 2 667 | 85 020 | 108 953 | 41 549 | 1 155 651 | 1 550 493 | 1 592 078 | x37 |
| 100000 | 2 696 | 83 907 | 123 648 | 42 353 | 1 191 208 | 1 957 840 | 2 003 266 | x46 |

**Linux in WSL 2 (gcc 15), 24 logical processors visible.**

| model (steps trained) | reference, 1 thread | AVX-512 dense, 1 thread | AVX-512 sparse, 1 thread | reference, 24 threads | AVX-512 dense, 24 threads | AVX-512 sparse, 24 threads | sparse with checks, 24 threads | sparse / reference at 24 threads |
|---|---|---|---|---|---|---|---|---|
| 0 | 1 823 | 101 453 | 105 533 | 24 649 | 1 249 003 | 1 389 037 | 1 554 624 | x56 |
| 300 | 1 824 | 100 664 | 104 377 | 24 556 | 1 232 270 | 1 403 247 | 1 560 957 | x57 |
| 3000 | 1 826 | 103 404 | 121 885 | 24 606 | 1 242 919 | 1 578 347 | 1 737 513 | x64 |
| 100000 | 1 833 | 97 172 | 144 975 | 24 681 | 1 249 528 | 2 008 383 | 2 047 493 | x81 |

The reference runs at 2.7 thousand attempts per second per thread with MSVC and 1.8 thousand with gcc 15 at `-O2` (the flag
of `neb-pow/build.rs`), which is why the ratio is larger on Linux. The column "with checks" is the sparse engine run as a
miner runs it (reference verification of hits and one audit in 4,096 attempts); it is within the noise of the plain sparse
engine, whose repetitions differ by up to 10 % on 32 threads. These rates belong to this machine; other processors, GPUs and
energy per attempt were not measured.
