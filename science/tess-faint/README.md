# TESS faint-star eclipse search, first test

A search for repeating dips in TESS stars fainter than T = 13.5, which the main TESS searches do not cover, with
a pixel test after it and the TESS model trained by the miners of the NEB chain (slot 1) as one of the filters.

Lastovka et al. (2026, [doi:10.3847/1538-3881/aebbbc](https://doi.org/10.3847/1538-3881/aebbbc)) write that most
stars observed by TESS only in full-frame images have not been searched for transiting planets, because the search
is computationally expensive and the standard vetting needs manual inspection. This folder is a first test of an
automatic chain on those stars: 1,977 of them, in one sector.

Nothing here touches the protocol, the node or the miners: these are standalone Python scripts that read public data
and a copy of the model weights.

## Status

Two runs on sector 40, 6,221 faint stars in all. Every signal the pixel test leaves on its star and that the other
checks confirm is an eclipsing binary, none a planet: one in the first run, absent from the catalogs queried, and five
in the second, all five already cataloged. The search alone does not sort: flipped light curves give as many dips as
the real ones, and the model of the chain did not help on these stars (see [The control](#the-control)). The pixel
test does the sorting, and [a check on published candidates](#a-check-on-published-candidates) measures how it
behaves on known planets.

## Data and credit

- TESS-Gaia Light Curves (TGLC): Han & Brandt (2023), MAST high-level science product,
  [doi:10.17909/610m-9474](https://doi.org/10.17909/610m-9474), read from the public MAST bucket.
- TESS full-frame images (NASA), cut out with TESScut at MAST (Brasseur et al. 2019).
- TESS Input Catalog and Gaia DR3 (ESA), through MAST and VizieR.
- NASA Exoplanet Archive: the table of TESS objects of interest.
- VizieR (CDS) for the catalogs of variable stars listed below and for Gaia DR3.
- T16 candidate catalog: Roth et al. (2026), Harvard Dataverse, [doi:10.7910/DVN/CWEUGW](https://doi.org/10.7910/DVN/CWEUGW).

## The model

The weights of slot 1 (dataset `tess`, the convolutional network of `spec/nebhash-v2.md` section 3) were read with
the `getmodel` RPC at block 11457 of the main network:

- file: `results/neb-tess-mainnet-block11457.json`
- weights hash: `ee76d212fd7df1f1f264d48a815381d68a1b27b233b8f92cf017c1eba90aaef3`
- on the 1,860 test signals kept out of the chain (`tools/make_mainnet_datasets.py` writes them): 95.0% right, 223
  of 250 eclipses found, 66 of 1,610 junk signals called an eclipse. Answering "junk" every time scores 86.6%.

The model was trained on signals of the Quick-Look Pipeline, whose stars are brighter than T = 13.6. The stars
searched here are fainter, with light curves from another pipeline.

## Method

1. **Stars.** 120 random places in the file tree of TGLC sector 40 (seed 2026), and up to 50 light curves after
   each: 2,454 light curves. The 472 stars at T = 13.5 or brighter are set aside, and 5 with fewer than 2,000 good
   points. That leaves 1,977 stars, median T = 15.2.
2. **Light curve.** The calibrated aperture flux of TGLC, cadences with both flags at 0, divided by a 1-day running
   median.
3. **Search.** The box search of `science/kbonus/vet.py`, periods from 0.5 to 9 days, durations from 1 to 6 hours.
4. **Detection tests.** MES at least 7.1, at least 3 events, no single event dominating, most events are dips.
5. **Model.** The light curve is folded into the 784 values of the chain's dataset, with the same code, and the
   model answers eclipse or junk. A **call** is a signal that is detected and called an eclipse.
6. **Control.** Every star is searched again with its flux flipped. A flipped light curve holds no eclipse, so
   whatever is called there is noise.
7. **Pixel test** (`pixel_test.py`). For each call, an 11 x 11 pixel cutout of the full-frame images and a
   difference image: pixels just outside the eclipse minus pixels inside it. The centroid of its peak says which
   star dims: the test reads the Gaia DR3 stars within 45 arcsec and names the one nearest to the centroid among
   those bright enough to make the dip. A TESS pixel is 21 arcsec wide. Below 4 sigma the test says nothing.
8. **Follow-up** (`follow_up.py`) of what stays on its star: depth of the odd and even events, the same dip in the
   other sectors, and the catalogs.

## Results

| step | light curves |
|---|---|
| read | 2,454 |
| faint and long enough | 1,977 |
| detected | 1,099 |
| detected and called an eclipse by the model | 56 |
| pixel test: off target | 19 |
| pixel test: no signal above 4 sigma | 28 |
| pixel test: cutout off the chip | 8 |
| pixel test: on target | 1 |

924 of the 1,099 detections have a period between 0.95 and 1.05 days. That is a systematic of these light curves,
not a signal, and the pixel test puts none of those calls on its star.

The search took 1,189 seconds for the 2,454 light curves with 8 processes.

### The control

On the flipped light curves the search detects 1,165 signals and the model calls 319 of them an eclipse, against
56 on the real ones. A filter that calls more eclipses in noise than in data is not sorting. On these stars the
model's answer carries no weight, and the calls that follow are the pixel test's work.

### The one that stays

Gaia DR3 2099445748765925760 = TIC 122061758, T = 13.94, in the Kepler field but with no Kepler observation at
MAST. The TIC gives Teff 5724 K and a radius of 2.1 solar radii.

| | |
|---|---|
| period | 2.7452 d |
| depth, sector 40 | 23.1 +- 0.6% |
| odd / even depth | 23.6% / 22.7% |
| dip at half a period | -0.3 +- 0.3% |
| events in sector 40 | 8 |
| difference image | centroid 1.3 arcsec from the star, peak at 11.6 sigma |

A 23% dip on a star of that size needs a companion about as large as the Sun: an eclipsing binary. A star 7 times
brighter in G lies 20.5 arcsec to the east; the centroid of the difference image falls 1.1 arcsec to the west of
the target, on the far side from that neighbor. A second call 44 arcsec away, at the same period, has the same
centroid: it is the light of this star.

Other sectors (`results/follow-up-2099445748765925760.json`): 9 sectors hold the star and 8 have data. The best
period within 2% of the hunt's gives a dip at 6 or more times the scatter of the phase bins in 6 of them (sectors
14, 40, 41, 55, 75, 80), 4.4 in sector 81 and 3.3 in sector 54. Those depths are measured on the raw sum of 3 x 3
pixels, so they are diluted, and the period is fitted sector by sector.

Catalogs with no entry within 8 arcsec: Gaia DR3 eclipsing binaries, two-body orbits and variable classes, ASAS-SN,
ZTF periodic variables, ATLAS, Catalina, the TESS eclipsing binary catalog, VSX. It is not a TESS object of
interest, and the Kepler Eclipsing Binary Catalog has no entry for KIC 3740108. SIMBAD and the literature were not
searched, so this is a star absent from those lists, not a new one.

## Second run: 5,000 light curves

The same search on 5,000 other light curves of sector 40 (`--fields 5000`, stopped after 5,000), with the order the
first run asked for: the pixel test runs on every detection, whatever the model says. Detections with a period
within 0.05 day of one day are set aside first.

| step | light curves |
|---|---|
| read | 5,000 |
| faint and long enough | 4,244 |
| detected | 641 |
| detected, period away from one day | 405 |
| pixel test: no signal above 4 sigma | 358 |
| pixel test: off target | 31 |
| pixel test: cutout off the chip | 10 |
| pixel test: on target | 4 |
| pixel test: blend | 2 |

The search took 1,312 seconds with 8 processes, the pixel test 393 seconds with 4 threads. On the flipped light
curves the search detects 830 signals, 409 of them away from one day: as many as on the real ones. A detection
alone says nothing here.

The six signals the pixel test keeps (on target or blend) and one it traced to a neighbor:

| Gaia DR3 | T | period (d) | depth | verdict | catalogs |
|---|---|---|---|---|---|
| 4518328655090595456 | 14.9 | 0.918 | 49% | on target | Gaia DR3 eclipsing binary, VSX type E, same period |
| 4518328655080966016 | 14.4 | 0.918 | 30% | blend | the star above, 3 arcsec away |
| 4580894405750393344 | 15.1 | 0.687 | 26% | on target | ZTF J175724.83+242752.5, type EA, same period |
| 2040467773254816000 | 14.8 | 0.899 | 51% | on target | WISE J185444.6+283659, type EW, half this period |
| 2028132038769311104 | 14.3 | 1.268 | 25% | blend | ZTF J194224.01+273301.2, type EA, twice this period |
| 2100930673578962816 | 15.1 | 0.677 | 26% | off target | ZTF J191828.02+384639.7, type EW, same period, 16 arcsec away |
| 4508383194455845632 | 13.6 | 1.576 | 6.5% | on target | none within 240 arcsec at this period |

Five eclipsing binaries, all already cataloged, found with no catalog given to the search. The model of the chain
calls an eclipse on each of the six kept signals, and on 41 of the 399 others.

The last row is not a result. The sum of the 3 x 3 raw pixels on the star does not show the dip in any of the four
sectors that hold it (26, 40, 53, 80), sector 40 included: 1.7 to 2.0 times the scatter of the phase bins. A dip of
6.5% on a star of T = 13.6 should show there. It may come from the light curve itself.

The first version of the verdict looked only at the distance between the centroid and the target, with a limit of
one pixel. It left the sixth row on target: the binary 16 arcsec away is inside that pixel. Naming the nearest Gaia
star fixed it, and the first run gives the same counts with either rule.

## A check on published candidates

`calibrate_t16.py` runs the pixel test on candidates of the T16 project (Roth et al. 2026, ApJS 284, 19,
[doi:10.3847/1538-4365/ae5b6c](https://doi.org/10.3847/1538-4365/ae5b6c); catalog at
[doi:10.7910/DVN/CWEUGW](https://doi.org/10.7910/DVN/CWEUGW)): 11,143 candidates from sectors 1 to 13, 7,805 of them
fainter than T = 13.5. Each candidate is tested in the sector where T16 found it, at its period, epoch and duration.
The ground truth is the TESS objects of interest at the same period: confirmed and known planets must stay on
their star.

498 candidates, up to 40 drawn at random in each group and magnitude bin (`results/calibration-t16.csv`):

| group | tested | on target | blend | off target | no signal | other |
|---|---|---|---|---|---|---|
| confirmed or known planets | 97 | 58 | 1 | 1 | 26 | 11 |
| other objects of interest (PC, APC) | 127 | 67 | 5 | 2 | 44 | 9 |
| false positives of the TOI list | 74 | 20 | 9 | 12 | 24 | 9 |
| T16 candidates that are not objects of interest | 200 | 19 | 5 | 70 | 79 | 27 |

"Other" is a cutout off the chip, a cutout the service did not return, or fewer than three events in the sector.

What this shows: the test leaves 58 of the 60 known planets it decides on their star. With the first version of the
verdict it rejected 4 of the 41 false positives it decided; with the nearest-star rule, 12, and it calls 9 a blend.
A false positive of the TOI list is often an eclipsing binary on the target, so this group does not have to move off.

What this does not show: that the 70 T16 candidates found off target are wrong. At these magnitudes the centroid is
less precise than the quarter of a pixel the blend rule assumes, only 3 known planets fainter than T = 14.5 are in
the sample, one sector is tested per candidate, and the result was not compared with the vetting plots T16
published. It is a table to put next to theirs.

## Limits

- One sector, 6,221 stars, six eclipsing binaries, five of them already cataloged. No rate of real eclipses or of
  false alarms can be read from it.
- The pixel test leaves most signals undecided: 358 of 405 in the second run. At these magnitudes the difference
  image is often below 4 sigma.
- In the first run the model ran before the pixel test, and it should not: the control shows it calls noise an
  eclipse more often than data. The second run puts the pixel test right after the detection tests.
- The calibration has almost no ground truth fainter than T = 14.5.
- The verdict asks VizieR for the Gaia stars around each decided signal. When VizieR does not answer, it falls back
  on the distance to the target alone.

## Running it

Python 3.12 or later, with the packages of `science/kbonus/requirements.txt` (numpy, astropy).

```
python hunt_faint.py --fields 120
python pixel_test.py
python follow_up.py 2099445748765925760
```

The second run, with outputs in a folder of your choice:

```
python hunt_faint.py --fields 5000 --out OUT/hunt
python pixel_test.py --detections --hunt OUT/hunt.csv --out OUT/pixel.csv
```

The calibration needs `candidate_table.tab` of the T16 dataset and the `toi` table of the NASA Exoplanet Archive
(columns toi, tid, tfopwg_disp, pl_orbper, as CSV in `toi.csv`), both in `data/t16/`:

```
python calibrate_t16.py --per-cell 40
python calibrate_t16.py --summary
```

`hunt_faint.py` skips the stars already in `results/hunt-faint.csv`; remove the file to run the search again. The
light curves of the calls go to `data/` next to the scripts (ignored by git), or to the directory in
`NEB_TESS_FAINT_DATA`. A TGLC light curve is about 150 KB, a cutout about 10 MB.

## Files

| file | what it does |
|---|---|
| `tf_common.py` | the model, the search set for one TESS sector, the views of the chain |
| `hunt_faint.py` | the search, the detection tests, the model, the flipped control |
| `pixel_test.py` | the difference image of each call or detection, and the star nearest to its centroid |
| `calibrate_t16.py` | the pixel test on published candidates, by group and magnitude |
| `follow_up.py` | depth, other sectors and catalogs for one star |
| `results/` | the outputs used above (`-5000` for the second run) and the model weights at block 11457 |

## License

The code is under the Apache License 2.0 of this repository. The files in `results/` derive from the data listed
above; cite those sources when you use them.
