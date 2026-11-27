# NEB sky search job, version 1 (draft)

## 1. Scope

This document specifies a job that mining software can run next to mining: a search for repeating dips in the
light curves of stars. It is not part of the consensus rules. Nothing changes in the node, in the pool protocol
(`spec/pool.md`) or in a block, and a miner that ignores this document mines as before.

The job is defined in integers only, so every implementation, on any processor or card, returns the same bytes for
the same batch. A server can then compare the results of two miners that do not know each other.

Sections 3 to 8 (the light curve, the computation, the two file formats) are implemented by a Python reference
and a Rust crate that agree on the shared vectors (section 11). Section 10, the transport, is not final: no batch
server is public yet.

## 2. Roles

```
batch server                                   miners
  light curves from a public archive
  -> one int8 per hour (section 3)
  -> batch file (section 7) ---------------->  box search (sections 4 to 6)
  <- result file (section 8) ----------------  32 entries per light curve
  compares two results of the same batch,
  checks the kept signals on the pixels
```

The server does everything that needs floating point or a catalog: it reads the archive, removes the slow trend
of each light curve, turns it into integers, and after the search it runs the tests that tell which star dims
(`science/tess-faint`). The miners do the part that costs the most arithmetic: the box search over every trial
period.

## 3. The light curve

A light curve is a row of `n` slots, `0 <= n <= 2048`. Slot `i` covers one hour: hour `i` after the first
measurement. Each slot is one signed byte `x[i]`:

- `-128`: no data in that hour;
- `-127` to `127`: the flux of that hour in sixteenths of the light curve's scatter, where 0 is the level of the
  star. A dip is negative. Deeper values are clamped to `-127`.

The job does not cover how the server gets these values, and a miner never needs to know.
`science/sky-search/prepare.py` shows what the reference server does: the flux divided by a running median of one
day, flares and the edges of gaps dropped, the median of each hour, divided by the robust scatter of those medians
and multiplied by 16.

## 4. Constants

| Name | Value | Meaning |
|---|---|---|
| `UNIT` | 4096 | time units in one slot (one hour) |
| `BIN` | 1024 | time units in one phase bin (a quarter of an hour) |
| `WIDTHS` | 4, 6, 8, 12, 16, 24 | box widths in phase bins: 1, 1.5, 2, 3, 4 and 6 hours |
| `P_MIN` | 49,152 | shortest trial period, in time units (half a day) |
| `P_MAX` | 884,736 | longest trial period (nine days) |
| `GROUP` | 64 | consecutive trial periods that compete for one entry |
| `KEEP` | 32 | entries returned per light curve |
| `MIN_IN` | 3 | fewest slots inside a box, and outside it |
| `MAX_SLOTS` | 2048 | longest light curve |

The trial periods are `p[0] = P_MIN` and `p[k+1] = p[k] + floor(p[k] / 8192)`, as long as `p[k] <= P_MAX`. There are
24,343 of them. The last one is 884,652.

## 5. The score of a box

All values are integers, and `div` and `mod` are the floor division and its remainder on non-negative numbers.

Let `V` be the slots with data (`x[i] != -128`), `n` their count and `T` the sum of their values.

For a trial period `p`, the number of phase bins is `m = (p + BIN - 1) div BIN`, and the phase bin of slot `i` is

```
bin(i) = ((i * UNIT + UNIT div 2) mod p) div BIN
```

By design, the last phase bin is narrower than the others when `BIN` does not divide `p`.

A box is a period `p`, a start `a` with `0 <= a < m`, and a width index `w` with `q = WIDTHS[w]`. It holds the
phase bins `a, a+1, ..., a+q-1`, each taken `mod m`. Let `s` be the sum of `x[i]` and `r` the count, over the
slots `i` of `V` whose phase bin is in the box. Then

```
d = r * T - n * s
score = (d * d) div (r * (n - r))     when d > 0, r >= MIN_IN and n - r >= MIN_IN
score = 0                             otherwise
```

`d` is positive when the mean inside the box is below the mean of the light curve: a dip. With `n <= 2048`,
`|d| <= 254 * n * r`, so `d * d` stays below `2^63`: signed 64-bit arithmetic is enough everywhere.

The score is `n^2` times the usual statistic of a box least squares search, so it orders the boxes of one light
curve. Scores of two light curves with different `n` cannot be compared directly.

## 6. The entries of a light curve

1. A light curve with `n < 2 * MIN_IN` has no box: its 32 entries are zero.
2. For each trial period, the best box is the one with the highest score. Ties go to the lower width index, then
   to the lower start.
3. The trial periods are taken in groups of `GROUP` in order: periods 0 to 63, 64 to 127, and so on. The last
   group is shorter. The winner of a group is the best box of its periods with the highest score. Ties go to the
   shorter period.
4. The winners are sorted by score, highest first. Ties go to the group of shorter periods. The first `KEEP`
   are the entries.
5. An entry with a score of zero is written as zeros: score, period, start and width index all zero.

Groups keep one strong signal from filling the result with its own neighbors in period. Its harmonics still take
entries of their own.

## 7. The batch file

Little-endian, no padding.

| Bytes | Field |
|---|---|
| 4 | `NEBS` |
| 2 | version, 1 |
| 2 | zero |
| 4 | number of light curves |
| then, for each light curve | |
| 2 | `n`, at most 2048 |
| `n` | the slots, one signed byte each |

A reader rejects the whole file when the magic or the version is different, a length is above 2048, a light curve
runs past the end of the file, or bytes follow the last light curve. A reader does not look at the two zero bytes
of the header.

The batch hash is the BLAKE3 hash (32 bytes) of the file.

## 8. The result file

Little-endian, no padding.

| Bytes | Field |
|---|---|
| 4 | `NEBR` |
| 2 | version, 1 |
| 2 | zero |
| 32 | batch hash |
| 4 | number of light curves |
| then, for each light curve in the order of the batch, 32 entries of 16 bytes | |
| 8 | score |
| 4 | period, in time units |
| 2 | start, in phase bins |
| 1 | width index, 0 to 5 |
| 1 | zero |

The file is `44 + 512 * count` bytes. Two correct implementations return the same file, byte for byte.

## 9. How the server checks results

- For a few light curves of each batch, the server computes the entries itself before the batch leaves. A result
  that differs on one of them is refused. A miner cannot tell those light curves from the others.
- The server checks a few entries of each result one by one, in one pass over the light curve each (`slow_score`
  in the reference).
- Each batch goes to at least two miners. When two of them return the same file, the batch is agreed. When two
  files differ, the server runs the search itself and drops the wrong one.

Some light curves of a batch are real ones with the flux flipped. They measure what noise alone gives, and a
miner cannot tell them apart either.

Once a batch is agreed, the server runs the detection tests of `science/tess-faint` on the slots of each light
curve. The pixel test needs about 10 MB of images for each signal, so the server queues the strong signals and
runs it for the strongest first, at a pace the operator sets (`science/sky-search/sift.py`).

## 10. Transport (not final)

`science/sky-search/batch_server.py` is the batch server. These paths may still change.

| Request | Reply |
|---|---|
| `GET /sky/v1/batch` | 200 and a batch file, or 204 when there is no batch for this miner |
| `POST /sky/v1/result`, the result file as the body | 200 when the result is taken |
| `GET /sky/v1/status` | 200 and the counters of the server, in JSON |
| `GET /sky/v1/signals` | 200 and the counts of each step with the signals the pixel test decided, in JSON |

A refused request gets one of these codes:

| Code | Meaning |
|---|---|
| 404 | unknown path, or a result for a batch the server does not have |
| 409 | the batch was settled while the miner searched it; nothing is counted against the miner |
| 413 | a body that is not the size of a result file |
| 422 | a result that fails a check of section 9; the body says which |
| 429 | more than 30 requests in a minute from one network address |
| 403 | five refused results in a day |

Both miner requests take three optional text fields in the query string, 64 characters at most each: `software`
(the name and version of the mining software), `device` (the model of the card or processor) and `address` (a NEB
address). None of them changes the result file. The server tells miners apart by the address, or by the network
address when none is given, and never hands the same batch twice to one miner.

Miners speak plain HTTP, as they speak plain TCP to a pool: nothing secret travels, and the server checks every
result. The server can also speak HTTPS (`--cert`, `--key`) or sit behind a proxy (`--behind-proxy`).

This protocol does not interrupt mining: the software decides when to run a batch, and the server never talks to
a pool.

## 11. Reference and vectors

- `science/sky-search/sky_ref.py`: the Python reference. `python sky_ref.py BATCH RESULT` writes the result file
  of a batch file, `python sky_ref.py --check` runs its self-check.
- `node/crates/neb-sky`: the same job in Rust, written the plain way, one period after the other, and the miner
  side of the transport (`client`). Its example `run` writes the result file of a batch file.
- `neb-sky`, the program of that crate, runs the search next to any miner and does not mine.
  `neb-sky --sky http://HOST:PORT` takes batches until it is stopped. `--sky-threads N` sets the search threads,
  `--sky-address ADDRESS` gives the server a NEB address, `--sky-device TEXT` names the cards or the processor, and
  `--once` stops when the server has no batch left.
- `neb-miner-cpu` and `neb-miner-gpu` take the same `--sky` options and run the search inside the miner. The search
  stays off until you pass `--sky`. In both miners it runs on processor threads in this version.
- `science/sky-search/batch_server.py`, `sift.py` and `sky_client.py`: the server, what it does with the entries,
  and a miner as small as it gets. `sky_client.py --engine COMMAND` runs any implementation of the job against a
  server. `science/sky-search/README.md` says how to run them.
- `vectors/sky-search.txt`: 11 light curves. The first four are real ones, TESS light curves of faint stars with
  an eclipse. The others are edge cases: noise with gaps, an empty light curve, five slots, no data at all, a
  constant, the largest values the format allows, and a dip every 24 hours that makes ties.

The vector file starts with the line `neb-sky-search-vectors 1`. Each `curve` line holds the slots of one light
curve in hexadecimal, then its 512 bytes of entries. The `batch` line holds the hash of the batch file made of
all the light curves in order, then the BLAKE3 hash of its result file.

```
cargo test --manifest-path node/Cargo.toml -p neb-sky
```

## 12. Cost

For one light curve, each trial period costs one pass over the `n` slots and one pass over the `m` phase bins
for each of the six widths: about `24,343 * (n + 6 * m)` additions in all, about `6 * 10^7` for a light curve of
27 days. Every period is independent of the others, and so is every light curve.

## 13. Reading the entries

A high score is not a detection. On real light curves, noise alone gives as many repeating dips as the stars do
(`science/tess-faint`, the flipped light curves). The entries are candidates for the tests that follow on the
server, and the pixel test sorts them.
