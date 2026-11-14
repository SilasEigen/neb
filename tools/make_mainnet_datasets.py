#!/usr/bin/env python3
"""Builds the dataset set file of the main network from the set file of the test network of protocol 3
(`tools/make_datasets.py`): its last two datasets, the Kepler transit signals for the KEP1 network and the retinal
scans at 64 x 64 for the RET2 network, as `NMDS || LE32(2) || kepler-v2 || retina64` (spec/shares.md section 11.6).

    python -I tools/make_mainnet_datasets.py TESTNET-neb-datasets.bin /path/outside/the/tree/neb-datasets.bin

It prints the SHA-256 of the file; `neb-node --dataset-hash FILE` prints the BLAKE3 hash the network pins.
"""
import hashlib
import struct
import sys

INPUTS = {1: 784, 2: 4096}  # values per sample of the KEP1 and RET2 networks (spec/nebhash-v2.md section 2)


def datasets(data):
    """the datasets of a set file, each as its bytes"""
    if data[:4] != b"NMDS":
        raise SystemExit("not a dataset set file")
    count, at, parts = struct.unpack_from("<I", data, 4)[0], 8, []
    for _ in range(count):
        if data[at:at + 4] == b"NMD1":
            size = 8 + struct.unpack_from("<I", data, at + 4)[0] * 785
        elif data[at:at + 4] == b"NMD2":
            arch, n = struct.unpack_from("<II", data, at + 4)
            size = 12 + n * (INPUTS[arch] + 1)
        else:
            raise SystemExit("unknown dataset at byte %d" % at)
        parts.append(data[at:at + size])
        at += size
    if at != len(data):
        raise SystemExit("bytes after the last dataset")
    return parts


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    parts = datasets(open(sys.argv[1], "rb").read())
    if len(parts) != 4 or any(p[:4] != b"NMD2" for p in parts[2:]):
        raise SystemExit("%s is not the set file of four datasets of the test network" % sys.argv[1])
    out = b"NMDS" + struct.pack("<I", 2) + parts[2] + parts[3]
    with open(sys.argv[2], "wb") as f:
        f.write(out)
    print("%s: %d bytes, SHA-256 %s" % (sys.argv[2], len(out), hashlib.sha256(out).hexdigest()))


if __name__ == "__main__":
    main()
