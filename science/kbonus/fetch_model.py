#!/usr/bin/env python3
"""Fetch the Kepler model of the chain (slot 0) in the state of the tip and check it against its weights hash.

    python fetch_model.py [--rpc URL] [--out FILE]

The getmodel RPC gives the packed 4-bit weights and the weights hash (BLAKE3 of "neb-model-weights" and the packed
weights, spec/shares.md section 7). The hash is recomputed with the independent oracle of pow/oracle, then W1
(128 x 784) and W2 (10 x 128) are written as int8 in a .npz with the height, block id and weights hash. Pass the file
to calib.py or hunt.py with --model. The results in results/ were made with results/neb-kepler-block1428.npz.
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "pow", "oracle"))
import kb_common as K  # noqa: E402
from primitives import blake3  # noqa: E402

NIN, NH, NOUT = 784, 128, 10


def rpc(url, method, params):
    body = json.dumps({"jsonrpc": "1.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(url, body, {"content-type": "application/json", **K.UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        reply = json.load(r)
    if reply.get("error"):
        raise RuntimeError(reply["error"])
    return reply["result"]


def unpack(packed):
    """signed 4-bit weights: element e in the low nibble of byte e/2 when e is even, in the high nibble when odd"""
    b = np.frombuffer(packed, dtype=np.uint8)
    out = np.empty(b.size * 2, dtype=np.int8)
    out[0::2] = ((b & 15).astype(np.int8) << 4) >> 4
    out[1::2] = b.astype(np.int8) >> 4
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rpc", default="https://neuralenginebyte.org/rpc")
    p.add_argument("--out", default=os.path.join(K.DATA, "neb-kepler-tip.npz"))
    a = p.parse_args()

    m = rpc(a.rpc, "getmodel", [0])
    packed = bytes.fromhex(m["weights"])
    if blake3(b"neb-model-weights" + packed).hex() != m["weightshash"]:
        raise SystemExit("the weights do not match the weights hash given by the node")
    w = unpack(packed)
    w1 = w[:NH * NIN].reshape(NH, NIN)
    w2 = w[NH * NIN:NH * NIN + NOUT * NH].reshape(NOUT, NH)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, w1=w1, w2=w2, height=m["height"], blockhash=m["blockhash"], weightshash=m["weightshash"])
    print("slot %s (%s), block %d, weights hash %s... -> %s" % (m["slot"], m["dataset"], m["height"], m["weightshash"][:16], a.out))


if __name__ == "__main__":
    main()
