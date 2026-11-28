# Sky search: reference, batch server and test miner

The sky search is a job that mining software can run next to mining: a box search for repeating dips in the light
curves of faint TESS stars. [`spec/sky-search.md`](../../spec/sky-search.md) defines it. This folder holds the
reference of the job and everything the server side does. Nothing here touches the consensus rules.

The stars are the ones [`science/tess-faint`](../tess-faint) searched by hand: fainter than T = 13.5, which the
main TESS searches leave out. Miners run the box search. The server prepares the light curves, checks the results
and runs the tests that say which star dims.

## Files

| file | what it does |
|---|---|
| `sky_ref.py` | the reference of the job, integers only |
| `prepare.py` | a light curve becomes one int8 per hour: the only step with floating point before the miners |
| `batch_server.py` | reads the archive, cuts batches, hands them to miners, checks and compares the results |
| `sift.py` | detection tests on the slots of an agreed batch, then the pixel test on the strongest signals |
| `sky_client.py` | a miner as small as it gets; `--engine COMMAND` tries any implementation against a server |
| `make_vectors.py` | writes `vectors/sky-search.txt` |

The Rust side is the crate `node/crates/neb-sky`: the same job, and the client that `neb-miner-cpu` and
`neb-miner-gpu` start with `--sky`.

## Run it

The scripts need Python 3 with the packages of `science/kbonus/requirements.txt`.

A server is two processes on one SQLite file:

```
python batch_server.py serve sky.db --host 0.0.0.0 --port 8787
python batch_server.py watch sky.db --pace 1 --pixel-every 60 --feed /path/to/site/science/sky.json
```

`watch` walks the sectors of the TESS-Gaia light curves in the public MAST bucket, the newest first, and looks for
new sectors every six hours. `--pace` is the most light curve files it reads per second, `--pixel-every` the
seconds between two pixel tests, `--ahead` the most batches that wait for miners before it stops reading, and `--page`
the files it reads between two rounds of sorting and writing the feed. `--feed` is the file the science page of the
site reads. It holds the counts of each step, the signals the pixel test decided, and when miners searched: the last
search, the miners of the last day and week, and the stars searched each day. It carries no miner address.

A miner can be any of these:

```
neb-sky --sky http://SERVER:8787 --sky-threads 2
neb-miner-cpu --node HOST:PORT --dataset FILE --sky http://SERVER:8787 --sky-threads 2
python sky_client.py --url http://SERVER:8787
```

`neb-sky` runs next to any miner and does not mine. It uses a quarter of the processor threads unless `--sky-threads`
sets another number. When the server has no batch, it asks again every minute.

## What a server costs

| what | size | how often |
|---|---|---|
| a light curve file of the archive | 152,640 bytes | each star the server reads |
| a cutout of the full-frame images (pixel test) | 9.8 MB, four cutouts, all the same size | 19 of the 300 stars of the table below, so about 6 stars in 100 |
| one star in the database | about 2 KB: its slots, the results of two miners, its signal row | each star |
| preparing a light curve | a few milliseconds of one processor core | each star |

The images are most of the traffic: about 620 KB per star searched, against 153 KB for the star itself. Read
at one star per second, the light curves are about 400 GB a month and the images, if every signal were tested,
about 1.6 TB. Two settings bound this:

- `--pace` is the most light curve files read per second: 1 means about 400 GB a month.
- `--pixel-every` is the seconds between two pixel tests. Signals wait in a queue, strongest first, and the page
  counts them. 60 seconds means about 420 GB a month.

Memory and processor stay small at these rates.

## What was checked

- The Python reference and the Rust crate return the same bytes on the 11 light curves of
  `vectors/sky-search.txt`.
- On 80 light curves that `science/tess-faint` kept, the period that search found is among the 32 entries of the
  integer search 80 times (`python prepare.py --compare HUNT_CSV`).
- The seven signals of the second run of `science/tess-faint`, sent through the whole chain (slots, search, server,
  detection tests on the slots, pixel test): seven at the same period, and the pixel test puts six on their star
  and one on its neighbor, as before. One of the six was a blend there and is on target here.
- The detection tests on the slots against the same tests on the full light curves, 300 stars of that run
  (`python sift.py --compare HUNT_CSV`):

  | sent to the pixel test, away from one day | full light curves | slots | both | flipped, full | flipped, slots |
  |---|---|---|---|---|---|
  | MES of 10 or more | 15 | 40 | 14 | 8 | 39 |
  | MES of 12 or more | 12 | 23 | 11 | 6 | 19 |
  | MES of 15 or more | 11 | 19 | 9 | 3 | 13 |
  | MES of 18 or more | 6 | 12 | 5 | 0 | 8 |
  | MES of 22 or more | 3 | 7 | 3 | 0 | 3 |

  The integer search tries more boxes than the search of `science/tess-faint`, so more light curves pass a given
  cut, flipped ones included: about twice as many. The server sends a signal to the pixel test from a MES of 15.
  The seven signals above measure 18.5 to 38.3 on the slots.

## Limits

- The search runs on processor threads in both miners. There is no kernel for cards yet.
- The server downloads every light curve it hands out. At 1 file per second, a sector of several million stars takes
  months. A faster rate needs miners that download their own light curves, which this version does not do.
- Miners speak plain HTTP. The server checks every result, and it tells miners apart by the address they give or
  by their network address: one person can pose as two. The light curves whose entries the server computed itself
  are the check that counts.
- The depth of a signal is read from the slots, and a slot cannot go below about 8 times the scatter. On a deep
  eclipse the depth is a lower bound, and the feed marks it.
- The archive holds sectors 1 to 55. A new sector is read when its light curves appear there, which is later than
  the release of its images.
- A signal on its star is a candidate. Noise alone gives repeating dips too, and the flipped light curves of each
  batch count them.
