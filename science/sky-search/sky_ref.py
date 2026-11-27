"""Reference of the sky search job, version 1 (spec/sky-search.md). Integers only: every implementation must return
the same bytes for the same batch.

    python sky_ref.py BATCH RESULT        the result file of a batch file
    python sky_ref.py --check             the self-check

A light curve is a row of hourly slots, each an int8: the flux in sixteenths of its scatter, -128 for no data. The
job is a box search for a repeating dip: for every trial period the slots are folded into phase bins of a quarter
of an hour, and every box of 1 to 6 hours is scored against the rest of the light curve.
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pow", "oracle"))
from primitives import blake3  # noqa: E402

UNIT = 4096  # time units in one slot (one hour)
BIN = 1024  # time units in one phase bin (a quarter of an hour)
WIDTHS = (4, 6, 8, 12, 16, 24)  # box widths in phase bins: 1, 1.5, 2, 3, 4 and 6 hours
P_MIN, P_MAX = 12 * UNIT, 216 * UNIT  # half a day to nine days
GROUP = 64  # consecutive trial periods that compete for one entry of the result
KEEP = 32  # entries of the result per light curve
MIN_IN = 3  # fewest slots inside a box, and outside it
MAX_SLOTS = 2048  # keeps every product below 2^63
NO_DATA = -128
BATCH_MAGIC, RESULT_MAGIC, VERSION = b"NEBS", b"NEBR", 1


def periods():
    """the trial periods in time units: each is the one before plus its 8192nd part, rounded down"""
    out, p = [], P_MIN
    while p <= P_MAX:
        out.append(p)
        p += p >> 13
    return np.array(out, np.int64)


PERIODS = periods()


def search(x):
    """KEEP entries (score, period, start bin, width index) of one light curve, best first; zeros fill the end"""
    x = np.asarray(x, np.int64)
    slots = np.flatnonzero(x != NO_DATA)
    v, n, total = x[slots], slots.size, int(x[slots].sum())
    best = np.zeros((PERIODS.size, 4), np.int64)  # per period: score, period, start, width index
    best[:, 1] = PERIODS
    bins = (PERIODS + BIN - 1) // BIN
    at = slots * UNIT + UNIT // 2  # the middle of each slot
    for m in np.unique(bins) if n >= 2 * MIN_IN else ():
        rows = np.flatnonzero(bins == m)
        b = (at[None, :] % PERIODS[rows, None]) // BIN + m * np.arange(rows.size)[:, None]
        hs = np.bincount(b.ravel(), np.broadcast_to(v, b.shape).ravel(), rows.size * m).astype(np.int64).reshape(-1, m)
        hc = np.bincount(b.ravel(), minlength=rows.size * m).reshape(-1, m)
        cs, cc = (np.cumsum(np.hstack([np.zeros((rows.size, 1), np.int64), h, h[:, :WIDTHS[-1]]]), axis=1) for h in (hs, hc))
        for qi, q in enumerate(WIDTHS):  # ascending, and argmax takes the first: ties go to the lower width, then start
            s, r = cs[:, q:q + m] - cs[:, :m], cc[:, q:q + m] - cc[:, :m]
            num = r * total - n * s  # positive when the box is below the rest
            ok = (num > 0) & (r >= MIN_IN) & (n - r >= MIN_IN)
            score = np.where(ok, num * num // np.where(ok, r * (n - r), 1), 0)
            a = score.argmax(axis=1)
            top = score[np.arange(rows.size), a]
            better = top > best[rows, 0]
            best[rows[better], 0], best[rows[better], 2], best[rows[better], 3] = top[better], a[better], qi
    groups = [best[g:g + GROUP] for g in range(0, PERIODS.size, GROUP)]
    winners = np.array([g[g[:, 0].argmax()] for g in groups])  # ties go to the shorter period
    order = np.argsort(-winners[:, 0], kind="stable")[:KEEP]  # ties go to the group of shorter periods
    out = winners[order]
    out[out[:, 0] == 0] = 0
    return out


def read_batch(raw):
    """the light curves of a batch file"""
    magic, version, _, count = struct.unpack_from("<4sHHI", raw)
    if magic != BATCH_MAGIC or version != VERSION:
        raise ValueError("not a sky search batch of version %d" % VERSION)
    curves, at = [], 12
    for _ in range(count):
        n, = struct.unpack_from("<H", raw, at)
        if n > MAX_SLOTS or at + 2 + n > len(raw):
            raise ValueError("bad light curve length")
        curves.append(np.frombuffer(raw, np.int8, n, at + 2))
        at += 2 + n
    if at != len(raw):
        raise ValueError("bytes after the last light curve")
    return curves


def write_batch(curves):
    return struct.pack("<4sHHI", BATCH_MAGIC, VERSION, 0, len(curves)) + b"".join(
        struct.pack("<H", len(c)) + np.asarray(c, np.int8).tobytes() for c in curves)


def run(raw):
    """the result file of a batch file"""
    curves = read_batch(raw)
    body = b"".join(struct.pack("<QIHBB", int(s), int(p), int(a), int(q), 0) for c in curves for s, p, a, q in search(c))
    return struct.pack("<4sHH32sI", RESULT_MAGIC, VERSION, 0, blake3(raw), len(curves)) + body


def read_result(raw):
    """(batch hash, entries by light curve) of a result file"""
    magic, version, _, batch, count = struct.unpack_from("<4sHH32sI", raw)
    if magic != RESULT_MAGIC or version != VERSION or len(raw) != 44 + count * KEEP * 16:
        raise ValueError("not a sky search result of version %d" % VERSION)
    rows = [struct.unpack_from("<QIHBB", raw, 44 + 16 * i)[:4] for i in range(count * KEEP)]
    return batch, [rows[i:i + KEEP] for i in range(0, len(rows), KEEP)]


def slow_score(x, period, start, qi):
    """the score of one box, written the plain way"""
    m, n, total, s, r = (period + BIN - 1) // BIN, 0, 0, 0, 0
    for i, value in enumerate(x):
        if value == NO_DATA:
            continue
        n, total = n + 1, total + int(value)
        if ((i * UNIT + UNIT // 2) % period // BIN - start) % m < WIDTHS[qi]:
            s, r = s + int(value), r + 1
    num = r * total - n * s
    return num * num // (r * (n - r)) if num > 0 and r >= MIN_IN and n - r >= MIN_IN else 0


def check():
    rng = np.random.default_rng(1)
    x = rng.integers(-40, 41, 660).astype(np.int8)
    x[rng.random(660) < 0.1] = NO_DATA
    period, width = 3 * 24 * UNIT + 1234, 3  # a dip of 3 hours every 3.01 days
    inside = (np.arange(660) * UNIT + UNIT // 2 - 20 * UNIT) % period < width * UNIT
    x[inside & (x != NO_DATA)] -= 60
    out = search(x)
    assert abs(int(out[0, 1]) - period) < period // 500, out[0]
    assert all(slow_score(x, int(p), int(a), int(q)) == s for s, p, a, q in out), "fast and plain scores differ"
    assert (np.diff(out[:, 0]) <= 0).all() and len(out) == KEEP
    assert not search(np.full(660, NO_DATA, np.int8)).any() and not search(np.zeros(5, np.int8)).any()
    worst = np.full(MAX_SLOTS, 127, np.int64)
    worst[::2] = -127
    assert int(search(worst)[0, 0]) == max(slow_score(worst, int(p), int(a), int(q)) for _, p, a, q in search(worst)[:1])
    raw = write_batch([x, x[:100]])
    batch, rows = read_result(run(raw))
    assert batch == blake3(raw) and len(rows) == 2 and rows[0][0] == tuple(int(k) for k in out[0])
    print("ok: %d periods, %d groups, best period %.4f d" % (PERIODS.size, -(-PERIODS.size // GROUP), out[0, 1] / UNIT / 24))


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
    else:
        open(sys.argv[2], "wb").write(run(open(sys.argv[1], "rb").read()))
