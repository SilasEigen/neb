#!/usr/bin/env python3
"""Converts the 60,000 MNIST training images into the dataset file of the testnet network.

    python tools/make_mnist_dataset.py --out /path/outside/the/tree/mnist-q8.bin
    python tools/make_mnist_dataset.py --out FILE --cache DIR     # reuse or fill a cache of the raw MNIST arrays

File format (little-endian): the 4 bytes "NMD1", the sample count N as a u32, then N records of 785 bytes: 784 pixels
in 0..7 and the label in 0..9. The pixels are quantised exactly as in tools/training_shares.py: floor(x / 32), clipped
to 0..7. The script prints the BLAKE3 hash of the file (the hash the network pins; needs the blake3 package) and its
SHA-256. MNIST is downloaded once (as in tools/training_shares.py) into the cache, never into the source tree.
"""
import argparse
import hashlib
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import training_shares as T  # noqa: E402

FILES = (("train-images-idx3-ubyte.gz", True), ("train-labels-idx1-ubyte.gz", False))


def load(cache):
    os.makedirs(cache, exist_ok=True)
    out = []
    for name, images in FILES:
        path = os.path.join(cache, name + ".npy")
        if not os.path.exists(path):
            np.save(path, T.mnist(name, images))
        out.append(np.load(path))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="the dataset file to write")
    p.add_argument("--cache", help="directory of the raw MNIST arrays (default: a directory next to --out)")
    a = p.parse_args()
    cache = a.cache or os.path.join(os.path.dirname(os.path.abspath(a.out)), "mnist")
    images, labels = load(cache)
    pixels = np.clip(np.floor(images.astype(np.float64) / 32.0), 0, 7).astype(np.uint8)
    assert pixels.shape == (60000, 784) and labels.shape == (60000,) and labels.max() <= 9
    records = np.concatenate([pixels, labels.astype(np.uint8)[:, None]], axis=1)
    data = b"NMD1" + struct.pack("<I", len(records)) + records.tobytes()
    with open(a.out, "wb") as f:
        f.write(data)
    print("wrote %s: %d samples, %d bytes" % (a.out, len(records), len(data)))
    print("sha256 %s" % hashlib.sha256(data).hexdigest())
    try:
        import blake3
        print("blake3 %s" % blake3.blake3(data).hexdigest())
    except ImportError:
        print("blake3: install the blake3 package, or run `neb-node --dataset-hash FILE`")


if __name__ == "__main__":
    main()
