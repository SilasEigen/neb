# Kepler Bonus background-star transit search

A search for transiting planets around faint M dwarfs that Kepler recorded but never targeted, using the Kepler
model trained by the miners of the NEB chain (slot 0) as the last step after a Kepler-style vetting.

Kepler Bonus extracted PSF light curves for 606,900 sources in the Kepler field, 406,548 of them background sources
that were not Kepler targets. This folder searches the background M dwarfs of that catalog, vets every signal like
the Kepler pipeline, runs the same pipeline on the inverted flux as a control, and asks the chain's model whether a
surviving signal looks like a planet or a false positive.

Nothing here touches the protocol, the node or the miners: these are standalone Python scripts that read public data
and a copy of the model weights.

## Status

One transit candidate, **KIC 6691030**, with two warning flags. It is not a confirmed planet. A planet would need
high-resolution imaging, spectroscopy and a statistical validation by astronomers. See [Candidate](#candidate-kic-6691030).

## Data and credit

- Kepler Bonus PSF light curves and M dwarf catalog: Martinez-Palomera, Hedges & Dotson (2023), MAST high-level
  science product KBONUS-BKG, [doi:10.17909/7jbr-w430](https://doi.org/10.17909/7jbr-w430), CC BY 4.0.
- Kepler mission data (NASA): target pixel files from MAST.
- NASA Exoplanet Archive: DR24 TCE table and DV light curves (the training data of the model), the cumulative KOI
  table, the Kepler stellar table.
- Gaia DR3 (ESA) and TIC v8, through the MAST API.

## The model

The weights of slot 0 (Kepler, 784-128-10, signed 4-bit) were read with the `getmodel` RPC at block 1428:

- file: `results/neb-kepler-block1428.npz`
- weights hash: `d8d471ac7f9ee7a0b28f6f8aa1d5e6e647331a2b10f3929807155f1a14b4334a`
- accuracy on the chain's Kepler test split (`kepler-test-q8.bin`, 2,333 signals never used in training): 88.34 %

The forward pass is the integer one of `spec/shares.md`. The input is the 784 values of `tools/make_datasets.py`
(583 bins of the whole orbit and 201 bins around the transit, 8 levels). The score of a signal is the planet output
minus the false-positive output. `fetch_model.py` fetches the current weights and checks them against their hash
with the oracle of `pow/oracle`.

## Method

1. **Light curve.** PSF flux of Kepler Bonus (aperture flux when the PSF flux is missing), cadences with Kepler quality
   bits 175 removed, each quarter divided by a 2-day running median.
2. **Search.** Flares (above 4 sigma) removed, one day after each data gap and a quarter day before it removed (the
   thermal settling after the monthly downlinks makes false dips near 15.5 and 31 days). Box search from 0.7 to 20
   days on 1-hour bins, without the 12.5 to 14.5 day band where detections pile up in Kepler Bonus light curves.
   The 50 strongest peaks of the raw and of the normalised spectrum are candidates; the one with the largest robust
   multiple event statistic (MES) is kept.
3. **Vetting** (`vet.py`), in the order of the Kepler pipeline: MES at least 7.1, at least 3 transits, no single
   event dominating, most events are dips, odd and even depths within 3 sigma, no secondary eclipse above 3 sigma,
   duration at most 1.5 times the longest central transit for a star of density 2 (solar units), depth at most 20 %.
   The noise is measured on the light curve averaged over the transit duration, so red noise counts.
4. **Model.** A signal that passes the vetting with a model score of at least 4 is a survivor.
5. **Control.** Every star is searched again with its flux inverted. Inverted survivors show what noise alone lets
   through; only real survivors above the strongest inverted one are studied further.
6. **Follow-up of a lead.** Depth per quarter, centroid shift, Gaia DR3 neighbours (`deep.py`); neighbours folded at
   the same period (`neighbors.py`); difference images in the Kepler target pixel files (`tpf.py`); transit shape,
   stellar density and phase curve (`candidate.py`).

## Results

**Calibration** (`calib.py`): 116 known signals of the chain's test stars (59 planets, 57 false positives), scored
on the NASA DV light curve and on the Kepler Bonus light curve of the same signal.

| light curve | accuracy | AUC | planets found | false positives rejected |
|---|---|---|---|---|
| NASA DV (training kind) | 76.7 % | 0.877 | 35 / 59 | 54 / 57 |
| Kepler Bonus PSF | 82.8 % | 0.923 | 42 / 59 | 54 / 57 |

The model works on Kepler Bonus light curves at least as well as on the kind it was trained on. It sorts signals that
are already detected; it does not separate noise from transits on its own, hence the search and vetting before it.

**Vetting** (`check_vet.py`), on the same signals at their catalog ephemeris (duration test left out, these hosts are
not M dwarfs): 39 of 59 planets pass, 4 of 57 false positives pass.

**Blind recovery** (`check_bls.py`): on the 40 planets of the calibration set with P < 20 days, the search finds the
known period (or half or twice it) for 22, and another known signal of the same star for 12 more. Of the 17 confirmed
planets found blind, the model calls 9 planets; the 8 it misses are all smaller than 2.3 Earth radii. `figure.py`
draws one of them, Kepler-114 d (`results/figure-kepler-recovered.png`).

**Hunt** (`hunt.py`, `analyze.py`): 2,000 background M dwarfs in a fixed random order (seed 2026, ranks 300 to 2300),
1,585 with usable light curves (`results/hunt.csv`). Survivors by MES threshold:

| MES at least | 7.1 | 8 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|
| real | 57 | 36 | 9 | 4 | 4 | 2 |
| inverted (control) | 41 | 21 | 9 | 3 | 2 | 0 |

The first 1,790 stars were searched with a sliding-window normalisation of the box-search spectrum; it ran out of
memory, and the rest used the block normalisation now in `vet.py`. A rerun of those stars with the current code can
differ slightly; the three leads below give the same period, MES and score with it.

Up to MES 20 the real survivors are not clearly above the control. The strongest inverted survivor has MES 23.0;
three real ones are above it:

- `kic-009966110`, P = 8.9474 d: a known false positive, KOI K06075.01, 15 arcsec away, same period. Rejected.
- `kic-009151544`, P = 1.8835 d, depth 2.2 %: seen in one quarter of target pixel data, on the star, but about
  9 Earth radii if the dip is on it; a stellar or brown-dwarf companion is more likely. Low priority.
- `kic-006691030`, P = 4.7720 d: the candidate below.

## Candidate KIC 6691030

KIC 6691030 = TIC 63005412 = Gaia DR3 2101746884862893952, G = 17.62, about 470 to 500 pc, an M dwarf
(TIC: Teff 3468 +- 158 K, R = 0.474 Rsun, density 4.4 +- 0.5 solar).

| | |
|---|---|
| period | 4.77200 d |
| transits in the data | 268 |
| depth (trapezoid fit) | 3,806 ppm |
| duration T14 | 1.8 h |
| MES | 30.9 |
| odd / even difference | 0.9 sigma |
| secondary eclipse | -0.8 sigma |
| model score (block 1428) | +8 |
| radius if the dip is on this star, undiluted | 3.2 +- 0.4 Earth radii |

Pixel level (`tpf.py`): the star falls in the target pixel files of KIC 6691035 for 18 quarters. In 16 quarters the
dip is detected at 4 sigma or more, and its centroid lies a median 1.4 arcsec from the candidate star (0.5 to 3.0);
the other Gaia stars are 8 arcsec or more away. The depth changes from season to season, also in the pixel data,
which points to a different dilution per season rather than to another source.

Warning flags (`candidate.py`):

1. **Stellar density.** The transit shape gives a density of 0.9 solar, against 4.4 from the TIC. Forcing the TIC
   density costs delta chi2 = 7.7. A grazing transit, an eccentric orbit, or a blend can each explain it.
2. **Phase curve.** Out of transit there is a 74 ppm signal at P/2, above 99 % of random trial periods (and 76 ppm at
   P, above 97 %). Ellipsoidal light at P/2 is the sign of an eclipsing binary.

An eclipsing binary, grazing or blended within 2 arcsec, is not excluded. The vetting sheet is
`results/candidate-kic-006691030.png`.

## Running it

Python 3.10 or later.

```
pip install -r requirements.txt
python fetch_model.py              # optional: current weights; the results use results/neb-kepler-block1428.npz
python hunt.py --start 300 --n 2000
python analyze.py
python deep.py kic-006691030
python neighbors.py kic-006691030
python tpf.py kic-006691030
python candidate.py
```

Downloads go to `data/` next to the scripts (ignored by git), or to the directory in `NEB_KBONUS_DATA`. A Kepler
Bonus light curve is about 7 MB; `hunt.py` downloads by chunks and deletes the light curves of the stars that do not
survive. The bulk download list of Kepler Bonus (175 MB) is fetched once.

`calib.py`, `check_vet.py`, `check_bls.py` and `figure.py` also need the Kepler cache of `tools/make_datasets.py`
(`dr24_tce.csv`, `dv24_wget.bat`, `views/`) in `NEB_KEPLER_CACHE`, and `kepler-test-q8.bin` in `NEB_KEPLER_TEST`.
`calib.py` stops if the replayed test split is not byte for byte the chain's test file.

## Files

| file | what it does |
|---|---|
| `kb_common.py` | paths, downloads, light curve reading, the views of the chain, the integer forward pass |
| `vet.py` | search and vetting |
| `fetch_model.py` | weights of slot 0 from a node, checked against the weights hash |
| `calib.py` | the model on known signals, NASA DV against Kepler Bonus light curves |
| `check_vet.py` | the vetting on the known signals at their catalog ephemeris |
| `check_bls.py` | blind recovery of the known planets |
| `figure.py` | figure of a known planet recovered blind |
| `hunt.py` | the hunt, with the inverted control |
| `analyze.py` | survivors against the control, known KOIs nearby |
| `deep.py`, `neighbors.py`, `tpf.py` | follow-up of a lead |
| `candidate.py` | transit fit, density, phase curve, vetting sheet |
| `results/` | the outputs used above and the model weights at block 1428 |

## License

The code is under the Apache License 2.0 of this repository. The files in `results/` derive from the data listed
above; cite those sources when you use them.
