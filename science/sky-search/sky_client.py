"""A miner of the sky search, as small as it gets: asks the batch server for a batch, runs the search, posts the
result (spec/sky-search.md).

    python sky_client.py [--url http://127.0.0.1:8787] [--address NEB_ADDRESS] [--engine COMMAND] [--once]

Without `--engine` the Python reference runs the search. With it, COMMAND is run with two more arguments, the
batch file and the result file to write: any implementation of the job can be tried against a server this way.
"""
import argparse
import os
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import sky_ref as S


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8787")
    p.add_argument("--address", default="")
    p.add_argument("--software", default="sky_client 1")
    p.add_argument("--device", default="")
    p.add_argument("--engine", default="")
    p.add_argument("--once", action="store_true", help="stop when the server has no batch left")
    a = p.parse_args()
    query = urllib.parse.urlencode({"software": a.software, "device": a.device, "address": a.address})
    while True:
        with urllib.request.urlopen("%s/sky/v1/batch?%s" % (a.url, query), timeout=60) as r:
            raw = r.read() if r.status == 200 else b""
        if not raw:
            if a.once:
                return
            time.sleep(30)
            continue
        began = time.time()
        if a.engine:
            with tempfile.TemporaryDirectory() as d:
                batch, out = os.path.join(d, "batch"), os.path.join(d, "result")
                open(batch, "wb").write(raw)
                subprocess.run(shlex.split(a.engine) + [batch, out], check=True)
                result = open(out, "rb").read()
        else:
            result = S.run(raw)
        try:
            with urllib.request.urlopen(urllib.request.Request("%s/sky/v1/result?%s" % (a.url, query), result), timeout=60) as r:
                answer = r.read().decode()
        except urllib.error.HTTPError as e:
            answer = "refused: " + e.read().decode()
        print("batch %s: %d light curves in %.1f s, %s" % (S.blake3(raw).hex()[:12], len(S.read_batch(raw)), time.time() - began, answer), flush=True)


if __name__ == "__main__":
    main()
