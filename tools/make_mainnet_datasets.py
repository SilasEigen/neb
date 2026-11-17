#!/usr/bin/env python3
"""Builds the dataset set file of the main network, `NMDS || LE32(2) || kepler-v2 || tess` (spec/shares.md section
11.6): two NMD2 datasets of the KEP1 network (spec/nebhash-v2.md section 2).

    python -I tools/make_mainnet_datasets.py TESTNET-neb-datasets.bin --tess DIR --out /path/outside/the/tree/neb-datasets.bin

Slot 0, `kepler-v2`: the Kepler DR24 transit signals of the set file of the test network of protocol 3
(`tools/make_datasets.py`), its third dataset, copied unchanged.

Slot 1, `tess`: the signals of Astronet-Triage-v2 (Tey et al. 2023, AJ 165, 95), CC BY 4.0,
https://zenodo.org/records/7411579. `--tess DIR` holds the two files of that record:
  - `tces_with_labels_v3.csv` (MD5 130a96933e75b4fe69905b31566c568c): 24,926 signals found by a box least squares
    search in the QLP light curves of three years of TESS, each with its epoch, period and duration, the time range
    used, the split of the authors (train, val or test) and the consensus label of at least three people: E a
    periodic eclipse (a planet or an eclipsing binary), J junk, B a contact binary, S a single transit, N not sure;
  - `LC.tar.gz` (6,479,668,693 bytes, MD5 6b3cff6aff56d39fd281e2bfd805b7a7): one FITS light curve per star, named
    `astronet_tess_ffi_*`, read in one pass without unpacking it. The columns used are TIME, SAP_FLUX and QUALITY.
    The archive also holds 35 files named `hlsp_qlp_tess_ffi_*`, second copies of stars it already has, two of them
    cut short; they are not read.
  A signal keeps the samples of quality 0 inside its time range. Its slow trend is removed: the flux is divided by
  its running median over max(1 day, 3 durations), taken after the samples within 0.75 duration of a transit are
  replaced by a linear interpolation. The curve is then folded and turned into the two views of the Kepler dataset
  by the same code (583 + 201 values of 0..7, the depth of the dip). Label 1 is an eclipse (E), 0 is junk (J); the
  signals with another consensus label or with none are left out. The chain holds every eclipse of the train and
  val splits and twice as many junk signals, drawn with a fixed seed; the eclipses and the junk of the test split
  go to `tess-test-q8.bin`, written next to `--out` and kept out of the chain. The views are cached in `DIR/views`.

The FITS files are read with astropy. The script prints the SHA-256 of the set file; `neb-node --mainnet
--dataset-hash FILE` prints the BLAKE3 hash the network pins.
"""
import argparse
import csv
import hashlib
import io
import os
import struct
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the folder of the script out of the path
from make_datasets import kepler_views, nmd2  # noqa: E402

SEED = 2026
JUNK_PER_ECLIPSE = 2
TESS_LABELS = {"E": 1, "J": 0}


def datasets(data):
    """the datasets of a set file, each as its bytes"""
    if data[:4] != b"NMDS":
        raise SystemExit("not a dataset set file")
    count, at, parts = struct.unpack_from("<I", data, 4)[0], 8, []
    inputs = {1: 784, 2: 4096}  # values per sample of the KEP1 and RET2 networks
    for _ in range(count):
        if data[at:at + 4] == b"NMD1":
            size = 8 + struct.unpack_from("<I", data, at + 4)[0] * 785
        elif data[at:at + 4] == b"NMD2":
            arch, n = struct.unpack_from("<II", data, at + 4)
            size = 12 + n * (inputs[arch] + 1)
        else:
            raise SystemExit("unknown dataset at byte %d" % at)
        parts.append(data[at:at + size])
        at += size
    if at != len(data):
        raise SystemExit("bytes after the last dataset")
    return parts


def tess_signals(folder):
    """the rows of the label table, in file order, each with its row number as `key`"""
    with open(os.path.join(folder, "tces_with_labels_v3.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))[1:]  # the first row holds the units
    for i, row in enumerate(rows):
        row["key"] = "%05d" % i
    return rows


def detrend(time, flux, period, t0, dur):
    """the flux divided by its running median, the transits interpolated over before the median"""
    inside = np.abs(np.mod(time - t0 + period / 2, period) - period / 2) < 0.75 * dur
    base = flux.copy()
    if inside.any() and (~inside).sum() > 10:
        base[inside] = np.interp(time[inside], time[~inside], flux[~inside])
    # the window is a count of samples, odd, from the median cadence; a gap in the curve is not treated apart
    n = min(int(max(1.0, 3 * dur) / np.median(np.diff(time))) | 1, (len(base) - 1) | 1)
    trend = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(base, n // 2, mode="edge"), n), axis=1)
    return flux / np.where(trend > 0, trend, 1.0)


def tess_one(raw, row):
    """the 784 values of a signal from the FITS file of its star, None when fewer than 200 samples are left"""
    from astropy.io import fits
    with fits.open(io.BytesIO(raw)) as hdus:
        table = hdus[1].data
        time, flux = np.asarray(table["TIME"], float), np.asarray(table["SAP_FLUX"], float)
        quality = np.asarray(table["QUALITY"])
    keep = (np.isfinite(time) & np.isfinite(flux) & (quality == 0)
            & (time >= float(row["MinT"]) - 1e-6) & (time <= float(row["MaxT"]) + 1e-6))
    time, flux = time[keep], flux[keep]
    if len(time) < 200:
        return None
    period, t0, dur = float(row["Period"]), float(row["Epoch"]), float(row["Duration"])
    return kepler_views(time, detrend(time, flux, period, t0, dur), period, t0, dur * 24.0)


def tess_views(folder, rows):
    """one pass over the archive: the views of every labelled signal that is not in the cache yet"""
    views = os.path.join(folder, "views")
    os.makedirs(views, exist_ok=True)
    by_star = {}
    for row in rows:
        if row["Consensus Label"] in TESS_LABELS and not os.path.exists(os.path.join(views, row["key"] + ".npy")):
            by_star.setdefault(int(row["TIC ID"]), []).append(row)
    if not by_star:
        return
    done = 0
    with tarfile.open(os.path.join(folder, "LC.tar.gz"), "r|gz") as tar:
        for member in tar:
            name = os.path.basename(member.name)
            if not member.isfile() or not name.startswith("astronet_tess_ffi_"):
                continue
            star = int(name.split("-")[1].split("_")[0])
            if star not in by_star:
                continue
            raw = tar.extractfile(member).read()
            for row in by_star[star]:
                values = tess_one(raw, row)
                if values is not None:
                    np.save(os.path.join(views, row["key"] + ".npy"), values)
                done += 1
                if done % 2000 == 0:
                    print("tess: %d signals read" % done, flush=True)


def tess(folder, out_dir):
    """The dataset of slot 1 and, next to the set file, the test file kept out of the chain."""
    rows = tess_signals(folder)
    tess_views(folder, rows)
    rng = np.random.default_rng(SEED)
    parts = {"chain": [], "test": []}
    for row in rows:
        path = os.path.join(folder, "views", row["key"] + ".npy")
        if row["Consensus Label"] in TESS_LABELS and os.path.exists(path):
            parts["test" if row["Split"] == "test" else "chain"].append((path, TESS_LABELS[row["Consensus Label"]]))
    eclipses = [s for s in parts["chain"] if s[1] == 1]
    junk = [s for s in parts["chain"] if s[1] == 0]
    junk = [junk[i] for i in sorted(rng.permutation(len(junk))[:JUNK_PER_ECLIPSE * len(eclipses)])]
    parts["chain"] = eclipses + junk
    built = {}
    for name, part in parts.items():
        part = [part[i] for i in rng.permutation(len(part))]
        values = np.stack([np.load(path) for path, _ in part]).astype(np.uint8)
        labels = np.array([label for _, label in part], np.uint8)
        built[name] = (values, labels)
        print("tess %s: %d signals, %d eclipses" % (name, len(part), int(labels.sum())), flush=True)
    values, labels = built["test"]
    with open(os.path.join(out_dir, "tess-test-q8.bin"), "wb") as f:
        f.write(b"NMD1" + len(labels).to_bytes(4, "little")
                + np.concatenate([values, labels.reshape(-1, 1)], axis=1).tobytes())
    return nmd2(1, *built["chain"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("testnet_set", help="the set file of four datasets of the test network")
    p.add_argument("--tess", required=True, help="folder with tces_with_labels_v3.csv and LC.tar.gz")
    p.add_argument("--out", required=True, help="the set file to write, outside the source tree")
    a = p.parse_args()
    with open(a.testnet_set, "rb") as f:
        parts = datasets(f.read())
    if len(parts) != 4 or parts[2][:8] != b"NMD2" + struct.pack("<I", 1):
        raise SystemExit("%s is not the set file of four datasets of the test network" % a.testnet_set)
    out = b"NMDS" + struct.pack("<I", 2) + parts[2] + tess(a.tess, os.path.dirname(os.path.abspath(a.out)))
    with open(a.out, "wb") as f:
        f.write(out)
    print("%s: %d bytes, SHA-256 %s" % (a.out, len(out), hashlib.sha256(out).hexdigest()))


if __name__ == "__main__":
    main()
