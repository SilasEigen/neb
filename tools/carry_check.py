# Independent check of a carry-over list: reads a chain dump (one block in hexadecimal per line) with its own
# parser, sums the outputs of every lock and compares the tenth of each balance with the list.
# Valid only for a chain whose blocks hold nothing but their coinbase (no output is ever spent).
import sys
dump, listfile = sys.argv[1], sys.argv[2]
class R:
    def __init__(s, b): s.b, s.i = b, 0
    def take(s, n): v = s.b[s.i:s.i + n]; assert len(v) == n; s.i += n; return v
    def varint(s):
        f = s.take(1)[0]
        return f if f < 0xfd else int.from_bytes(s.take({0xfd: 2, 0xfe: 4, 0xff: 8}[f]), "little")
bal, project, blocks = {}, None, 0
for line in open(dump):
    r = R(bytes.fromhex(line.strip())); r.take(80)
    assert r.varint() == 1, "a block holds a transaction besides its coinbase"
    r.take(1)
    for _ in range(r.varint()):
        prev = r.take(32); idx = r.take(4); r.take(r.varint())
        assert prev == bytes(32) and idx == b"\xff" * 4, "not a coinbase"
    outs = [(int.from_bytes(r.take(8), "little"), r.take(1)[0], r.take(32).hex()) for _ in range(r.varint())]
    if project is None: project = (outs[1][1], outs[1][2])
    for amount, t, lock in outs: bal[(t, lock)] = bal.get((t, lock), 0) + amount
    blocks += 1
excluded = {(1, "00" * 32), project}
text = "".join(f"{t} {lock} {b // 10}\n" for (t, lock), b in sorted(bal.items(), key=lambda kv: (kv[0][0], bytes.fromhex(kv[0][1]))) if (t, lock) not in excluded and b // 10 > 0)
ref = open(listfile, newline="").read().replace("\r\n", "\n")
print("blocks", blocks, "locks", text.count("\n"), "total", sum(int(l.split()[2]) for l in text.splitlines()), "mined by listed locks", sum(b for k, b in bal.items() if k not in excluded))
print("IDENTICAL" if text == ref else "DIFFERENT")
