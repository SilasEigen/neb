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

One signal passed every step: an eclipsing binary, not a planet. The model of the chain did not help on these stars
(see [The control](#the-control)); the pixel test did the sorting. One sector and one case give no rate.

## Data and credit

- TESS-Gaia Light Curves (TGLC): Han & Brandt (2023), MAST high-level science product,
  [doi:10.17909/610m-9474](https://doi.org/10.17909/610m-9474), read from the public MAST bucket.
- TESS full-frame images (NASA), cut out with TESScut at MAST (Brasseur et al. 2019).
- TESS Input Catalog and Gaia DR3 (ESA), through MAST and VizieR.
- NASA Exoplanet Archive: the table of TESS objects of interest.
- VizieR (CDS) for the catalogs of variable stars listed below.

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
   star dims. A TESS pixel is 21 arcsec wide. Below 4 sigma the test says nothing.
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

The search took 1,189 seconds for the 2,454 light curves with 8 processes on one desktop processor.

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

## Limits

- One sector, 1,977 stars, one case. No rate of real eclipses or of false alarms can be read from it.
- The pixel test left 28 of 56 calls undecided: at these magnitudes the difference image is often below 4 sigma.
- The model runs before the pixel test here, and it should not: the control shows it calls noise an eclipse more
  often than data. The next run puts the pixel test right after the detection tests.
- No calibration on stars with a known answer at these magnitudes yet.

## Running it

Python 3.12 or later, with the packages of `science/kbonus/requirements.txt` (numpy, astropy).

```
python hunt_faint.py --fields 120
python pixel_test.py
python follow_up.py 2099445748765925760
```

`hunt_faint.py` skips the stars already in `results/hunt-faint.csv`; remove the file to run the search again. The
light curves of the calls go to `data/` next to the scripts (ignored by git), or to the directory in
`NEB_TESS_FAINT_DATA`. A TGLC light curve is about 150 KB, a cutout about 10 MB.

## Files

| file | what it does |
|---|---|
| `tf_common.py` | the model, the search set for one TESS sector, the views of the chain |
| `hunt_faint.py` | the search, the detection tests, the model, the flipped control |
| `pixel_test.py` | the difference image of each call |
| `follow_up.py` | depth, other sectors and catalogs for one star |
| `results/` | the outputs used above and the model weights at block 11457 |

## License

The code is under the Apache License 2.0 of this repository. The files in `results/` derive from the data listed
above; cite those sources when you use them.
