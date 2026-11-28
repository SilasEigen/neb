"""What the batch server does with the entries of a light curve once a batch is agreed: the detection tests of
science/tess-faint on the hourly slots, then, for a detected signal of a real light curve, the pixel test.

    python sift.py --compare HUNT_CSV [--count 300] [--engine COMMAND]
                                the tests on the slots against the same tests on the full light curves, for stars
                                of an earlier search; COMMAND runs the search (as in sky_client.py), the Python
                                reference when it is not given

The tests read the slots, not the file of the star: nothing is read from the archive a second time. The pixel test
downloads about 10 MB of images for each signal, so it runs from a queue: the strongest signal first, one every
`every` seconds (`pixels`). At 60 seconds that is about 420 GB a month.

    python sift.py --check              the queue against a stand-in for the pixel test, without network
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tess-faint"))
import sky_ref as S  # noqa: E402
import tf_common as T  # noqa: E402
from prepare import SCALE, slots  # noqa: E402

vet = T.vet
ONE_DAY = 0.05  # periods this close to one day are set aside: a systematic of the light curves
# The pixel test runs from this MES up. In the second run of science/tess-faint, the pixel test decided 9 of the 248
# signals below 10 it was given, all nine off target, and every signal it kept on its star is above 18 here. The
# table of README.md gives what each cut sends.
MES_PIXELS = 15.0


def to_pixels(d, mes=None):
    """True for a detected signal the pixel test is worth asking about"""
    return bool(d and d["detected"] and (d["mes"] or 0) >= (mes or MES_PIXELS) and abs(d["period"] - 1.0) > ONE_DAY)


def box(entry):
    """(period, middle of the first dip after slot 0, duration) of an entry, in days"""
    _, p, start, q = entry
    width = S.WIDTHS[q] * S.BIN
    return p / S.UNIT / 24.0, (start * S.BIN + width / 2) / S.UNIT / 24.0, width / S.UNIT / 24.0


def detect(x, rows, sigma):
    """the entry with the largest robust MES and what the detection tests say of it, or None without an entry.
    Entries away from one day are tried first.
    `sigma` is the scatter of the slots in relative flux, so depths come out in parts per million."""
    x = np.asarray(x)
    keep = x != S.NO_DATA
    t, f = (np.flatnonzero(keep) + 0.5) / 24.0, x[keep].astype(float) / SCALE * sigma
    vet.CADENCE = 1.0 / 24.0
    scored = []
    for e in rows:
        if e[0]:
            period, t0, dur = box(e)
            scored.append((vet.robust_mes(t, f, period, t0, dur, vet.duration_noise(t, f, dur)), period, t0, dur))
    if not scored:
        return None

    def tested(choice):
        _, period, t0, dur = choice
        v = vet.vet(t, f, period, t0, dur, rho_min=0.05)
        return {"period": round(period, 5), "t0": round(t0, 4), "dur_h": round(dur * 24, 2), "n": v.get("n_transits", 0),
                "depth_ppm": v.get("depth_ppm", 0), "mes": v.get("mes", 0.0), "fail": v["fail"], "detected": int(v["fail"] not in T.DETECTION)}

    # the one-day systematic must not hide a signal at another period: the best entry away from one day goes first
    away = [c for c in scored if abs(c[1] - 1.0) > ONE_DAY]
    first = tested(max(away)) if away else None
    if first and first["detected"]:
        return first
    overall = tested(max(scored))
    return overall if overall["detected"] or first is None else first


def fold(x, sigma, d):
    """the slots folded on the dip, for a plot: points and bin medians in hours and parts per million"""
    x = np.asarray(x)
    keep = x != S.NO_DATA
    t, f = (np.flatnonzero(keep) + 0.5) / 24.0, x[keep].astype(float) / SCALE * sigma * 1e6
    period, dur_h = d["period"], d["dur_h"]
    h = (np.mod(t - d["t0"] + 0.5 * period, period) - 0.5 * period) * 24.0
    half = min(period * 12, max(8.0, 4 * dur_h))
    near = np.abs(h) <= half
    width = max(dur_h / 6, 2 * half / 60)
    edges, bins = np.arange(-half, half + width, width), []
    for i in range(len(edges) - 1):
        v = f[near][(h[near] >= edges[i]) & (h[near] < edges[i + 1])]
        if v.size >= 3:
            bins.append([round(float(edges[i] + width / 2), 2), int(round(float(np.median(v)))),
                         int(round(float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(v.size))))])
    return {"points": [[round(float(a), 2), int(round(float(b)))] for a, b in zip(h[near], f[near])], "bins": bins,
            "half": round(half, 2), "measurements": int(t.size)}


def listed(ra, dec, tic):
    """names of the catalogs with a row within 8 arcsec of the star; empty when the catalog service does not answer"""
    try:
        import follow_up
        return [name for name, rows in follow_up.catalogs(ra, dec, tic).items() if rows]
    except Exception:
        return []


# a detected signal of a real light curve, strong enough and away from one day, that the pixel test has not decided yet
PENDING = "flipped = 0 and detected = 1 and verdict = '' and mes >= %s and abs(period - 1.0) > %s" % (MES_PIXELS, ONE_DAY)
TRIES = 3  # times the cutout of a signal is asked for before the signal is given up


def detect_new(con, after=None):
    """the detection tests on every light curve of the agreed batches that were not sifted yet; `after()` is called each
    time a batch is done, so a long line of batches does not stop the caller from doing its other work. No network."""
    for batch, raw, answer in con.execute("select hash, raw, answer from batch where state = 'agreed' and sifted = 0").fetchall():
        body = con.execute("select raw from result where batch = ? and hash = ? limit 1", (batch, answer)).fetchone()[0]
        curves, rows = S.read_batch(raw), S.read_result(body)[1]
        stars = con.execute("select idx, gaia, flipped, sector, start, scale from curve where batch = ?", (batch,)).fetchall()
        found = []  # computed first, written in one short transaction: the server must not wait on this to take a result
        for idx, gaia, flipped, sector, start, scale in stars:
            d = detect(curves[idx], rows[idx], scale) or {"period": 0, "t0": 0, "dur_h": 0, "n": 0, "depth_ppm": 0, "mes": 0, "detected": 0}
            found.append((batch, idx, gaia, sector, flipped, d["period"], start + d["t0"], d["dur_h"], d["n"], d["depth_ppm"] or 0, d["mes"] or 0,
                          d["detected"], time.time()))
        # the batch may have been reopened while the tests ran: then nothing is written
        if con.execute("select 1 from batch where hash = ? and state = 'agreed' and answer = ?", (batch, answer)).fetchone():
            con.executemany("insert or replace into signal values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '{}', ?)", found)
            con.execute("update batch set sifted = 1 where hash = ?", (batch,))
        con.commit()
        print("batch %s sifted" % batch[:12], flush=True)
        if after:
            after()


def pixels(con, every=60.0, limit=None):
    """the pixel test on the pending signals, the strongest first, with `every` seconds between two cutouts (each is
    about 10 MB of download). Returns how many signals it worked on; at most `limit`."""
    import pixel_test
    count, began = 0, 0.0
    while limit is None or count < limit:
        row = con.execute("select batch, idx, gaia, sector, period, t0, dur_h, depth_ppm, detail from signal where " + PENDING
                          + " order by mes desc, batch, idx limit 1").fetchone()
        if not row:
            break
        time.sleep(max(0.0, began + every - time.time()))
        began = time.time()
        batch, idx, gaia, sector, period, t0, dur_h, depth, detail = row
        tic, ra, dec, tmag, start, scale = con.execute("select tic, ra, dec, tmag, start, scale from curve where batch = ? and idx = ?", (batch, idx)).fetchone()
        out = pixel_test.one({"gaia": gaia, "tic": tic, "tmag": tmag, "ra": ra, "dec": dec, "sector": sector, "real_period": period, "real_t0": t0,
                              "real_dur_h": dur_h, "real_depth_ppm": depth})
        count += 1
        tries = json.loads(detail).get("tries", 0) + 1
        if out["verdict"] == "no cutout" and tries < TRIES:  # the service may be away: the signal stays in the queue
            con.execute("update signal set detail = ? where batch = ? and idx = ?", (json.dumps({"tries": tries}), batch, idx))
            con.commit()
            continue
        x = S.read_batch(con.execute("select raw from batch where hash = ?", (batch,)).fetchone()[0])[idx]
        detail = {k: out[k] for k in ("snr_peak", "offset_arcsec", "source", "source_sep") if k in out}
        # a slot cannot go below -127: on a deep dip the depth read from the slots is a lower bound
        detail["clamped"] = bool((np.asarray(x) == -127).any())
        if out["verdict"] in ("on target", "blend"):
            detail["curve"], detail["catalogs"] = fold(x, scale, {"period": period, "t0": t0 - start, "dur_h": dur_h}), listed(ra, dec, tic)
        con.execute("update signal set verdict = ?, detail = ?, at = ? where batch = ? and idx = ?",
                    (out["verdict"], json.dumps(detail, separators=(",", ":")), time.time(), batch, idx))
        con.commit()
    return count


def run(con, every=60.0):
    """detection on every agreed batch, then the whole pixel queue"""
    detect_new(con)
    pixels(con, every)


DAY = 86400.0


def activity(con, now=None, days=14):
    """when miners searched, for the page of the site: the time of the last search, the miners of the last 24 hours and of
    the last 7 days, and for each of the last `days` UTC days the miners that sent a result and the stars of the batches
    that two miners agreed on and the server has sorted (counted on the day of the last result of the batch, so the days
    add up to the stars searched). It carries no miner address."""
    now = time.time() if now is None else float(now)
    start = now - now % DAY - (days - 1) * DAY
    rows = con.execute("select at, miner from result where miner != 'server' and at >= ?", (start,)).fetchall()
    agreed = con.execute("""select max(r.at), (select count(*) from curve c where c.batch = r.batch and c.flipped = 0)
                            from result r join batch b on b.hash = r.batch where r.miner != 'server' and b.state = 'agreed' and b.sifted = 1
                            group by r.batch having max(r.at) >= ?""", (start,)).fetchall()
    out = [{"day": time.strftime("%Y-%m-%d", time.gmtime(start + i * DAY)), "stars": 0, "who": set()} for i in range(days)]
    for at, miner in rows:
        i = int((at - start) // DAY)
        if 0 <= i < days:
            out[i]["who"].add(miner)
    for at, stars in agreed:
        i = int((at - start) // DAY)
        if 0 <= i < days:
            out[i]["stars"] += stars
    since = lambda seconds: len({m for at, m in rows if at >= now - seconds})  # noqa: E731
    last = con.execute("select coalesce(max(at), 0) from result where miner != 'server'").fetchone()[0]
    return {"now": now, "last": last, "miners24": since(DAY), "miners7": since(7 * DAY),
            "days": [{"day": d["day"], "stars": d["stars"], "miners": len(d["who"])} for d in out]}


def feed(con, now=None):
    """what the page of the site shows: the counts of each step, the signals the pixel test decided, newest first, and when
    miners searched"""
    one = lambda q: con.execute(q).fetchone()[0]  # noqa: E731
    away = "abs(period - 1.0) > %s" % ONE_DAY
    verdicts = dict(con.execute("select verdict, count(*) from signal where verdict != '' group by verdict").fetchall())
    decided = ("on target", "blend", "off target")
    summary = {"sectors": [s for s, in con.execute("select distinct sector from curve order by sector")],
               "read": one("select count(*) from curve where flipped = 0"), "searched": one("select count(*) from signal where flipped = 0"),
               "detected": one("select count(*) from signal where flipped = 0 and detected = 1"),
               "oneDay": one("select count(*) from signal where flipped = 0 and detected = 1 and not " + away),
               "tested": sum(verdicts.values()), "waiting": one("select count(*) from signal where " + PENDING),
               "noise": {"searched": one("select count(*) from signal where flipped = 1"), "detected": one("select count(*) from signal where flipped = 1 and detected = 1"),
                         "tested": one("select count(*) from signal where flipped = 1 and detected = 1 and " + away)},
               "verdicts": {v: verdicts.get(v, 0) for v in decided + ("no signal",)} | {"no data": sum(n for v, n in verdicts.items() if v not in decided + ("no signal",))},
               "updated": one("select coalesce(max(at), 0) from signal")}
    signals = []
    for row in con.execute("""select s.gaia, c.tic, c.tmag, s.sector, s.period, s.dur_h, s.depth_ppm, s.n, s.verdict, s.detail, s.at from signal s
                              join curve c on c.batch = s.batch and c.idx = s.idx where s.verdict in (?, ?, ?) order by s.at desc limit 200""", decided):
        gaia, tic, tmag, sector, period, dur_h, depth, n, verdict, detail, at = row
        d = json.loads(detail)
        names = d.get("catalogs", [])
        signals.append({"gaia": gaia, "tic": int(tic) if str(tic).isdigit() else 0, "tmag": tmag, "sector": sector, "period": period, "dur_h": dur_h,
                        "depth_ppm": depth, "n": n, "snr": d.get("snr_peak", 0), "offset": d.get("offset_arcsec", 0), "verdict": verdict,
                        "status": "off" if verdict == "off target" else "cataloged" if names else "unchecked", "note": ", ".join(names),
                        "source": d.get("source", ""), "sourceSep": d.get("source_sep"), "curve": d.get("curve"), "clamped": d.get("clamped", False), "at": at})
    return {"summary": summary, "signals": signals, "activity": activity(con, now)}


def compare(hunt, count, engine=""):
    """re-reads `count` stars of an earlier search and runs both paths: detections and periods side by side"""
    import concurrent.futures as cf

    import hunt_faint as H
    from batch_server import flip
    rows = [r for r in csv.DictReader(open(hunt)) if r.get("real_period")]
    rows = [rows[i] for i in np.random.default_rng(7).choice(len(rows), min(count, len(rows)), replace=False)]

    def one(r):
        g = r["gaia"].zfill(21)
        key = "%s%s/%s/hlsp_tglc_tess_ffi_gaiaid-%s-s%04d-%s_tess_v1_llc.fits" % (
            H.ROOT, r["camccd"], "/".join(g[i:i + 4] for i in range(0, 16, 4)), r["gaia"], T.SECTOR, r["camccd"])
        _, t, f = H.load(key)
        x, _, sigma = slots(t, f)
        return r, x, sigma

    with cf.ThreadPoolExecutor(8) as ex:
        stars = list(ex.map(one, rows))
    raw = S.write_batch([c for _, x, _ in stars for c in (x, flip(x))])
    if engine:
        import shlex
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "batch"), "wb").write(raw)
            subprocess.run(shlex.split(engine) + [os.path.join(d, "batch"), os.path.join(d, "result")], check=True)
            found = S.read_result(open(os.path.join(d, "result"), "rb").read())[1]
    else:
        found = S.read_result(S.run(raw))[1]
    out = [(r, detect(x, found[2 * i], sigma), detect(flip(x), found[2 * i + 1], sigma)) for i, (r, x, sigma) in enumerate(stars)]
    table = np.zeros((2, 2), int)
    same = both = inv_old = inv_new = 0
    as_dict = lambda r, tag: {"detected": int(r[tag + "_detected"] or 0), "mes": float(r[tag + "_mes"] or 0), "period": float(r[tag + "_period"] or 0)}  # noqa: E731
    for r, new, inv in out:
        old = int(r["real_detected"])
        table[old, new["detected"] if new else 0] += 1
        if old and new and new["detected"]:
            both += 1
            ratio = new["period"] / float(r["real_period"])
            same += any(abs(ratio * k - 1) < 0.01 or abs(ratio / k - 1) < 0.01 for k in (1, 2, 3))
        inv_old, inv_new = inv_old + int(r["inv_detected"] or 0), inv_new + (inv["detected"] if inv else 0)
    print("%d stars. detected on the full light curve / on the slots:" % len(out))
    print("  both %d, full only %d, slots only %d, neither %d" % (table[1, 1], table[1, 0], table[0, 1], table[0, 0]))
    print("  of the %d detected by both, %d at the same period or a harmonic" % (both, same))
    print("  flipped light curves detected: %d on the full light curve, %d on the slots" % (inv_old, inv_new))
    print("sent to the pixel test, away from one day:   full  slots  both   flipped full  flipped slots")
    for cut in (10, 12, 15, 18, 22, 27):
        full = {r["gaia"] for r, _, _ in out if to_pixels(as_dict(r, "real"), cut)}
        here = {r["gaia"] for r, new, _ in out if to_pixels(new, cut)}
        print("  MES >= %2d                                  %4d  %5d  %4d   %12d  %13d" % (
            cut, len(full), len(here), len(full & here), sum(to_pixels(as_dict(r, "inv"), cut) for r, _, _ in out), sum(to_pixels(inv, cut) for _, _, inv in out)))


def check():
    """the pixel queue against a stand-in for the pixel test: the strongest signal first, signals that are weak, flipped
    or at one day never asked, a missing cutout asked again up to TRIES times"""
    import hashlib
    import sqlite3
    import types

    import batch_server as B
    con = sqlite3.connect(":memory:")
    con.executescript(B.SCHEMA)
    rng = np.random.default_rng(3)
    stars = [({"gaia": str(i), "tic": str(100 + i), "ra": 10.0 + i, "dec": 5.0, "tmag": 14.0}, rng.integers(-40, 41, 300).astype(np.int8), 0.0, 0, 40, 0.001)
             for i in range(6)]
    B.cut(con, stars, rng)
    batch = con.execute("select hash from batch").fetchone()[0]
    index = {g: i for i, g in con.execute("select idx, gaia from curve")}
    # gaia: (MES, period, flipped): 3 is too weak, 4 is flipped, 5 is at one day
    for g, (mes, period, flipped) in {"0": (40.0, 2.0, 0), "1": (20.0, 3.1, 0), "2": (16.0, 2.5, 0), "3": (12.0, 2.2, 0), "4": (50.0, 2.4, 1), "5": (60.0, 1.01, 0)}.items():
        con.execute("insert into signal values (?, ?, ?, 40, ?, ?, 2000.0, 2.0, 5, 10000, ?, 1, '', '{}', 0)", (batch, index[g], g, flipped, period, mes))
    asked = []
    verdicts = {"0": "on target", "1": "no cutout", "2": "off target"}

    def one(r):
        asked.append(r["gaia"])
        return {"verdict": verdicts[r["gaia"]], "snr_peak": 6.0, "offset_arcsec": 1.0}

    sys.modules["pixel_test"] = types.SimpleNamespace(one=one)
    globals()["listed"] = lambda ra, dec, tic: ["stub"]
    waiting = lambda: feed(con)["summary"]["waiting"]  # noqa: E731
    assert waiting() == 3
    assert pixels(con, 0.0, 1) == 1 and asked == ["0"] and waiting() == 2
    assert pixels(con, 0.0) == 4 and asked == ["0", "1", "1", "1", "2"] and waiting() == 0, asked
    got = {g: con.execute("select verdict, detail from signal where gaia = ?", (g,)).fetchone() for g in verdicts}
    assert [got[g][0] for g in "012"] == ["on target", "no cutout", "off target"], got
    assert json.loads(got["0"][1])["catalogs"] == ["stub"] and json.loads(got["0"][1])["curve"]["bins"] is not None
    assert all(con.execute("select verdict from signal where gaia = ?", (g,)).fetchone() == ("",) for g in "345")
    assert pixels(con, 0.0) == 0 and len(asked) == 5
    # when miners searched: results at known times, a result of the server itself and one older than the window left out
    now = 1_800_000_000.0  # 08:00 UTC
    assert activity(con, now)["last"] == 0 and sum(d["stars"] + d["miners"] for d in activity(con, now)["days"]) == 0
    B.cut(con, [({"gaia": str(10 + i), "tic": str(200 + i), "ra": 20.0 + i, "dec": 5.0, "tmag": 14.0}, rng.integers(-40, 41, 300).astype(np.int8), 0.0, 0, 40, 0.001)
                for i in range(3)], rng)
    other = con.execute("select hash from batch where hash != ?", (batch,)).fetchone()[0]
    files = {b: S.run(con.execute("select raw from batch where hash = ?", (b,)).fetchone()[0]) for b in (batch, other)}
    # two miners on the first batch today, two on the second batch yesterday, then results of batches of nothing
    for k, (miner, at, on) in enumerate((("a", now - 3600, batch), ("d", now - 1800, batch), ("b", now - 100000, other), ("e", now - 90000, other),
                                         ("a", now - 10 * DAY, "x4"), ("c", now - 20 * DAY, "x5"), ("server", now - 60, "x6"))):
        body = files.get(on, b"")
        con.execute("insert into result values (?, ?, ?, ?, ?, '', '', '')", (on, miner, hashlib.sha256(body).hexdigest(), body, at))
    for b in (batch, other):
        B.agree(con, b)
    act = feed(con, now)["activity"]
    assert act["last"] == now - 1800, act
    assert (act["miners24"], act["miners7"]) == (2, 4), act
    assert len(act["days"]) == 14 and act["days"][-1]["day"] == time.strftime("%Y-%m-%d", time.gmtime(now)), act["days"][-1]
    assert [d["miners"] for d in act["days"]] == [0] * 3 + [1] + [0] * 8 + [2, 2], [d["miners"] for d in act["days"]]
    # a batch counts when two miners agree and the server has sorted it, on the day of its last result: before that it waits
    assert [d["stars"] for d in act["days"]] == [0] * 14, [d["stars"] for d in act["days"]]
    done = []
    detect_new(con, lambda: done.append(1))
    assert len(done) == 2, done
    assert [d["stars"] for d in feed(con, now)["activity"]["days"]] == [0] * 12 + [3, 6]
    print("ok: queue order, retries, skips and activity")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
        sys.exit(0)
    option = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default  # noqa: E731
    compare(sys.argv[2], int(option("--count", 300)), option("--engine", ""))
