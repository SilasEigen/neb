"""The batch server of the sky search (spec/sky-search.md).

    python batch_server.py serve DB [--port 8787] [--host 127.0.0.1] [--cert PEM --key PEM] [--behind-proxy]
    python batch_server.py watch DB [--pace 2] [--pixel-every 60] [--ahead 50] [--size 64] [--every 6] [--feed FILE]
    python batch_server.py ingest DB --fields 4 [--per 50] [--size 64] [--pace 2]
    python batch_server.py settle DB
    python batch_server.py sift DB
    python batch_server.py status DB

DB is one SQLite file. Two processes share it: `serve` talks to the miners, `watch` does the rest.

`watch` lists the sectors of the TESS-Gaia light curves in the public MAST bucket, the newest first, and walks each
one file by file, at most `pace` files per second (153 KB each). It turns each light curve of a faint star into
hourly integers (prepare.py) and cuts batches. It stops reading while `ahead` batches wait for miners, and looks for
new sectors every `every` hours. Between two reads it settles the disputed batches, runs the detection tests on the
agreed ones (sift.py) and rewrites `feed`, the file the page of the site reads. The pixel test downloads about
10 MB of images per signal, so it runs from a queue, the strongest signal first, one every `pixel-every` seconds.
`ingest` reads a few random places of one sector instead: for tests.

A tenth of the light curves go in a second time with the flux flipped: what noise alone gives. For a few light
curves of each batch the server computes the entries itself and keeps them: a result that differs on one of
those is refused. A batch is agreed when two miners return the same result file. Two different files make it
disputed, and the server computes the right one. A miner is known by the NEB address it gives, or by its network
address: one person can pose as two, so the entries the server computed itself are the check that counts.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import ssl
import sys
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tess-faint"))
import sky_ref as S  # noqa: E402

SCHEMA = """
create table if not exists batch (hash text primary key, raw blob, made real, state text default 'open', answer text,
                                  sifted integer default 0);
create table if not exists curve (batch text, idx integer, gaia text, tic text, ra real, dec real, tmag real,
                                  start real, flipped integer, known blob, sector integer, scale real,
                                  primary key (batch, idx));
create table if not exists result (batch text, miner text, hash text, raw blob, at real, software text,
                                   device text, address text, primary key (batch, miner));
create table if not exists refused (miner text, batch text, why text, at real);
create table if not exists signal (batch text, idx integer, gaia text, sector integer, flipped integer, period real,
                                   t0 real, dur_h real, n integer, depth_ppm integer, mes real, detected integer,
                                   verdict text, detail text, at real, primary key (batch, idx));
create table if not exists state (key text primary key, value text);
create index if not exists curve_gaia on curve (gaia, sector);
create index if not exists refused_miner on refused (miner, at);
"""
ARCHIVE = "mast/hlsp/tglc/"
KNOWN = 3  # light curves of a batch whose entries the server computes before the batch leaves
SPOT = 8  # entries of each result checked one by one with the plain score
TEXT = 64  # longest value of a text field
MOST_RESULTS = 4  # results kept for one batch
MOST_REFUSALS = 5  # refusals in a day before a miner is turned away
PER_MINUTE = 30  # requests of one network address
MOST_BYTES = 44 + 4096 * S.KEEP * 16  # longest result file read


def db(path, create=False):
    con = sqlite3.connect(path, timeout=30)
    if create:
        con.executescript(SCHEMA)
        con.execute("pragma journal_mode=wal")
    return con


def entries(result, i):
    """the 512 bytes of light curve `i` in a result file"""
    return result[44 + i * S.KEEP * 16:44 + (i + 1) * S.KEEP * 16]


def flip(x):
    """the light curve with the flux flipped; no data stays no data"""
    return np.where(x == S.NO_DATA, S.NO_DATA, -x.astype(np.int16)).astype(np.int8)


def cut(con, rows, rng):
    """one batch from the rows read so far, in a random order"""
    order = rng.permutation(len(rows))
    raw = S.write_batch([rows[i][1] for i in order])
    h = S.blake3(raw).hex()
    known = set(rng.choice(len(rows), min(KNOWN, len(rows)), replace=False).tolist())
    con.execute("insert into batch (hash, raw, made) values (?, ?, ?)", (h, raw, time.time()))
    for idx, i in enumerate(order):
        r, x, start, flipped, sector, scale = rows[i]
        answer = b"".join(S.struct.pack("<QIHBB", *(int(v) for v in e), 0) for e in S.search(x)) if idx in known else None
        con.execute("insert into curve values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (h, idx, r["gaia"], str(r["tic"]), r["ra"], r["dec"], r["tmag"], start, flipped, answer, sector, scale))
    con.commit()
    print("batch %s: %d light curves, %d bytes" % (h[:12], len(rows), len(raw)), flush=True)
    rows.clear()


def take(con, keys, rows, a, rng, progress=None):
    """reads light curve files into `rows` and cuts a batch each time `a.size` rows are there; `progress(key)` is
    called after a cut: every file up to `key` is then in a batch"""
    import hunt_faint as H
    from prepare import slots
    for k in keys:
        began, sector = time.time(), int(re.search(r"/s(\d{4})/", k).group(1))
        gaia = re.search(r"gaiaid-(\d+)", k).group(1)
        if con.execute("select 1 from curve where gaia = ? and sector = ?", (gaia, sector)).fetchone():
            continue
        try:
            r, t, f = H.load(k)
            if t is not None:
                x, start, scale = slots(t, f)
                rows.append((r, x, start, 0, sector, scale))
                if rng.random() < 0.1:
                    rows.append((r, flip(x), start, 1, sector, scale))
        except Exception as e:  # one unreadable file must not stop a run of weeks
            print("skipped %s: %s" % (k.rsplit("/", 1)[-1], str(e)[:60]), flush=True)
        if len(rows) >= a.size:
            cut(con, rows, rng)
            if progress:
                progress(k)
        time.sleep(max(0.0, 1.0 / a.pace - (time.time() - began)))


def ingest(a):
    import hunt_faint as H
    con, rng, rows = db(a.db, True), np.random.default_rng(a.seed), []
    take(con, H.fields(a.fields, a.per, rng), rows, a, rng)
    if rows:
        cut(con, rows, rng)


def watch(a):
    import hunt_faint as H
    import sift
    con, rng, rows = db(a.db, True), np.random.default_rng(), []
    state = lambda k: (con.execute("select value from state where key = ?", (k,)).fetchone() or [""])[0]  # noqa: E731

    def remember(k, v):
        con.execute("insert or replace into state values (?, ?)", (k, v))
        con.commit()

    # `position` is where the reading has got to, `cursor:` in the database where it can start again: the light
    # curves read since the last batch are in memory only, so the cursor moves when a batch is cut
    todo, position, next_look, last_pixel, last_feed = [], {}, 0.0, 0.0, 0.0
    while True:
        settle(a, con)
        sift.detect_new(con)
        if time.time() - last_pixel >= a.pixel_every and sift.pixels(con, a.pixel_every, 1):
            last_pixel = time.time()
        if a.feed and time.time() - last_feed >= 60:  # the file the page of the site reads, replaced in one step
            with open(a.feed + ".part", "w") as f:
                json.dump(sift.feed(con) | {"server": counts(con)}, f, separators=(",", ":"))
            os.replace(a.feed + ".part", a.feed)
            last_feed = time.time()
        if con.execute("select count(*) from batch where state = 'open'").fetchone()[0] >= a.ahead:
            time.sleep(30)
            continue
        if time.time() >= next_look:  # the newest sector first
            sectors = sorted((d for d in H.listing(ARCHIVE, dirs=True) if re.search(r"/s\d{4}/$", d)), reverse=True)
            todo, next_look = [s for s in sectors if not state("done:" + s)], time.time() + a.every * 3600
        if not todo:
            time.sleep(30)
            continue
        sector = todo[0]
        position.setdefault(sector, state("cursor:" + sector))
        keys = H.listing(sector, after=position[sector], n=100)
        if not keys:
            if rows:
                cut(con, rows, rng)
            remember("done:" + sector, "1")
            todo.pop(0)
            print("sector %s read to the end" % sector[-6:-1], flush=True)
            continue
        take(con, keys, rows, a, rng, lambda k, sector=sector: remember("cursor:" + sector, k))
        position[sector] = keys[-1]


def check(con, batch, raw, body):
    """why a result file is refused, or '' """
    count = S.struct.unpack_from("<I", raw, 8)[0]
    try:
        named, rows = S.read_result(body)
    except (ValueError, S.struct.error):
        return "not a result file"
    if named.hex() != batch or len(rows) != count:
        return "not the result of this batch"
    for idx, known in con.execute("select idx, known from curve where batch = ? and known is not null", (batch,)):
        if entries(body, idx) != known:
            return "wrong entries"
    curves = S.read_batch(raw)
    for _ in range(SPOT):
        i, k = random.randrange(count), random.randrange(S.KEEP)
        s, p, start, q = rows[i][k]
        if s and (q >= len(S.WIDTHS) or not S.P_MIN <= p <= S.P_MAX or S.slow_score(curves[i], p, start, q) != s):
            return "wrong score"
    return ""


def agree(con, batch):
    """sets the state of a batch from its results: agreed when two miners sent the same file"""
    hashes = [h for h, in con.execute("select hash from result where batch = ?", (batch,))]
    same = [h for h in set(hashes) if hashes.count(h) >= 2]
    if same:
        con.execute("update batch set state = 'agreed', answer = ? where hash = ?", (same[0], batch))
    elif len(set(hashes)) >= 2:
        con.execute("update batch set state = 'disputed' where hash = ?", (batch,))


def settle(a, con=None):
    """computes the result of each disputed batch and drops the results that differ from it"""
    con = con or db(a.db, True)
    for batch, raw in con.execute("select hash, raw from batch where state = 'disputed'").fetchall():
        right = S.run(raw)
        name = hashlib.sha256(right).hexdigest()
        wrong = [m for m, h in con.execute("select miner, hash from result where batch = ?", (batch,)) if h != name]
        con.execute("update batch set state = 'agreed', answer = ? where hash = ?", (name, batch))
        con.executemany("insert into refused values (?, ?, 'lost a dispute', ?)", [(m, batch, time.time()) for m in wrong])
        con.execute("delete from result where batch = ? and hash != ?", (batch, name))
        con.execute("insert or ignore into result values (?, 'server', ?, ?, ?, 'batch_server', '', '')", (batch, name, right, time.time()))
        con.commit()
        print("batch %s settled, %d wrong result(s)" % (batch[:12], len(wrong)), flush=True)


def counts(con):
    one = lambda q: con.execute(q).fetchone()[0]  # noqa: E731
    return {"batches": one("select count(*) from batch"), "agreed": one("select count(*) from batch where state = 'agreed'"),
            "disputed": one("select count(*) from batch where state = 'disputed'"),
            "light_curves": one("select count(*) from curve where flipped = 0"),
            "searched": one("select count(*) from curve c join batch b on b.hash = c.batch where b.state = 'agreed' and c.flipped = 0"),
            "results": one("select count(*) from result"), "miners": one("select count(distinct miner) from result where miner != 'server'"),
            "refused": one("select count(*) from refused")}


class Handler(BaseHTTPRequestHandler):
    path_db, behind_proxy = "", False
    server_version, timeout = "neb-sky/1", 20
    seen, lock = {}, threading.Lock()
    feed = (0.0, b"")

    def reply(self, code, body=b"", kind="application/octet-stream"):
        self.send_response(code)
        self.send_header("content-type", kind)
        self.send_header("content-length", str(len(body)))
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def who(self):
        """(text fields, miner, network address): the miner is its NEB address when it gives one"""
        q = parse_qs(urlparse(self.path).query)
        f = {k: "".join(c for c in q.get(k, [""])[0][:TEXT] if c.isprintable()) for k in ("software", "device", "address")}
        net = self.client_address[0]
        if self.behind_proxy:  # the last hop is the one the proxy saw
            net = (self.headers.get("x-forwarded-for") or net).split(",")[-1].strip()[:TEXT]
        return f, f["address"] or net, net

    def allowed(self, con, miner, net):
        """False after the reply when this address asks too often or this miner was refused too many times"""
        now = time.time()
        with self.lock:
            if len(self.seen) > 100000:
                self.seen.clear()
            tokens, last = self.seen.get(net, (float(PER_MINUTE), now))
            tokens = min(float(PER_MINUTE), tokens + (now - last) * PER_MINUTE / 60.0)
            self.seen[net] = (max(tokens - 1.0, 0.0), now)
        if tokens < 1.0:
            self.reply(429, b"too many requests", "text/plain")
            return False
        if con.execute("select count(*) from refused where miner in (?, ?) and at > ?", (miner, net, now - 86400)).fetchone()[0] >= MOST_REFUSALS:
            self.reply(403, b"too many refused results", "text/plain")
            return False
        return True

    def do_GET(self):
        with closing(db(self.path_db)) as con:
            route = urlparse(self.path).path
            if route == "/sky/v1/status":
                return self.reply(200, json.dumps(counts(con)).encode(), "application/json")
            if route == "/sky/v1/signals":
                if time.time() - Handler.feed[0] > 30:
                    import sift
                    Handler.feed = (time.time(), json.dumps(sift.feed(con) | {"server": counts(con)}, separators=(",", ":")).encode())
                return self.reply(200, Handler.feed[1], "application/json")
            if route != "/sky/v1/batch":
                return self.reply(404)
            _, miner, net = self.who()
            if not self.allowed(con, miner, net):
                return None
            # a batch this miner has not answered, the one with the fewest results first
            row = con.execute("""select b.raw from batch b where b.state = 'open'
                                 and not exists (select 1 from result r where r.batch = b.hash and r.miner = ?)
                                 order by (select count(*) from result r where r.batch = b.hash), random() limit 1""", (miner,)).fetchone()
            return self.reply(200, row[0]) if row else self.reply(204)

    def do_POST(self):
        if urlparse(self.path).path != "/sky/v1/result":
            return self.reply(404)
        size = int(self.headers.get("content-length") or 0) if (self.headers.get("content-length") or "0").isdigit() else 0
        if not 44 <= size <= MOST_BYTES:
            return self.reply(413)
        with closing(db(self.path_db)) as con:
            f, miner, net = self.who()
            if not self.allowed(con, miner, net):
                return None
            body = self.rfile.read(size)
            batch = body[8:40].hex()
            row = con.execute("select raw, state from batch where hash = ?", (batch,)).fetchone()
            if not row:
                return self.reply(404, b"unknown batch", "text/plain")
            if row[1] != "open" or con.execute("select count(*) from result where batch = ?", (batch,)).fetchone()[0] >= MOST_RESULTS:
                return self.reply(409, b"this batch is settled", "text/plain")
            why = check(con, batch, row[0], body)
            if why:
                con.execute("insert into refused values (?, ?, ?, ?)", (miner, batch, why, time.time()))
                con.commit()
                return self.reply(422, why.encode(), "text/plain")
            con.execute("insert or ignore into result values (?, ?, ?, ?, ?, ?, ?, ?)",
                        (batch, miner, hashlib.sha256(body).hexdigest(), body, time.time(), f["software"], f["device"], f["address"]))
            agree(con, batch)
            con.commit()
            return self.reply(200, b"ok", "text/plain")

    def do_OPTIONS(self):
        self.reply(204)

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("serve", "watch", "ingest", "settle", "sift", "status"))
    p.add_argument("db")
    p.add_argument("--fields", type=int, default=4, help="ingest: random places of the sector")
    p.add_argument("--per", type=int, default=50, help="ingest: light curve files after each place")
    p.add_argument("--size", type=int, default=64, help="light curves in a batch")
    p.add_argument("--pace", type=float, default=2.0, help="most files read from the archive per second")
    p.add_argument("--pixel-every", type=float, default=60.0, help="seconds between two pixel tests (about 10 MB each)")
    p.add_argument("--ahead", type=int, default=50, help="watch: batches that may wait for miners")
    p.add_argument("--every", type=float, default=6.0, help="watch: hours between two looks for new sectors")
    p.add_argument("--feed", help="watch: file to rewrite with what the page of the site shows")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--cert", help="serve: certificate chain, PEM; with --key the server speaks HTTPS")
    p.add_argument("--key")
    p.add_argument("--behind-proxy", action="store_true", help="serve: take the network address from x-forwarded-for")
    a = p.parse_args()
    if a.command == "ingest":
        ingest(a)
    elif a.command == "watch":
        watch(a)
    elif a.command == "settle":
        settle(a)
    elif a.command == "sift":
        import sift
        sift.run(db(a.db, True), a.pixel_every)
    elif a.command == "status":
        print(json.dumps(counts(db(a.db, True))))
    else:
        db(a.db, True).close()
        Handler.path_db, Handler.behind_proxy = a.db, a.behind_proxy
        httpd = ThreadingHTTPServer((a.host, a.port), Handler)
        if a.cert:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(a.cert, a.key)
            httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print("serving on %s://%s:%d" % ("https" if a.cert else "http", a.host, a.port), flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
