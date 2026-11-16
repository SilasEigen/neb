# Independent check of a carry-over list: reads a chain dump (one block in hexadecimal per line, heights 1, 2, 3 and
# so on) with its own parser, keeps its own set of unspent outputs up to HEIGHT (the whole dump when omitted) and
# compares the tenth of the balance of every lock with the list. It checks no proof of work and no signature:
# the dump is that of a chain a node accepted, and `examples/carry_over.rs` of neb-node replays it with every rule.
#
#     python -I tools/carry_check.py DUMP LIST [HEIGHT]
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pow", "oracle"))
from blake3_full import blake3_full

dump, listfile = sys.argv[1], sys.argv[2]
height = int(sys.argv[3]) if len(sys.argv) > 3 else None


class R:
    def __init__(s, b): s.b, s.i = b, 0
    def take(s, n): v = s.b[s.i:s.i + n]; assert len(v) == n; s.i += n; return v
    def varint(s):
        f = s.take(1)[0]
        return f if f < 0xfd else int.from_bytes(s.take({0xfd: 2, 0xfe: 4, 0xff: 8}[f]), "little")


def varint(n):
    for limit, flag, size in ((0xfc, b"", 1), (0xffff, b"\xfd", 2), (0xffffffff, b"\xfe", 4)):
        if n <= limit: return flag + n.to_bytes(size, "little")
    return b"\xff" + n.to_bytes(8, "little")


utxo, project, blocks, transfers = {}, None, 0, 0
for line in open(dump):
    if blocks == height: break
    r = R(bytes.fromhex(line.strip())); r.take(80)
    for position in range(r.varint()):
        start = r.i
        version, ins = r.take(1), []
        for _ in range(r.varint()):
            ins.append(r.take(32) + r.take(4)); r.take(r.varint())
        outs = [(r.take(8), r.take(1), r.take(32)) for _ in range(r.varint())]
        coinbase = ins == [bytes(32) + b"\xff" * 4]
        assert coinbase == (position == 0), "the coinbase is the first transaction of a block and the only one"
        if coinbase:
            # the txid of a coinbase commits to its witness, that of another transaction does not
            txid = blake3_full(b"txid" + r.b[start:r.i])
            if project is None: project = (outs[1][1][0], outs[1][2].hex())
        else:
            body = version + varint(len(ins)) + b"".join(ins) + varint(len(outs)) + b"".join(a + t + d for a, t, d in outs)
            txid = blake3_full(b"txid" + body)
            for i in ins: del utxo[i]  # a KeyError here is an input that spends no unspent output
            transfers += 1
        for n, (a, t, d) in enumerate(outs):
            utxo[txid + n.to_bytes(4, "little")] = (int.from_bytes(a, "little"), t[0], d.hex())
    blocks += 1
assert height is None or blocks == height, "the dump stops before the height"
bal = {}
for amount, t, lock in utxo.values(): bal[(t, lock)] = bal.get((t, lock), 0) + amount
excluded = {(1, "00" * 32), project}
text = "".join(f"{t} {lock} {b // 10}\n" for (t, lock), b in sorted(bal.items(), key=lambda kv: (kv[0][0], bytes.fromhex(kv[0][1]))) if (t, lock) not in excluded and b // 10 > 0)
ref = open(listfile, newline="").read().replace("\r\n", "\n")
print("blocks", blocks, "transfers", transfers, "locks", text.count("\n"), "total", sum(int(l.split()[2]) for l in text.splitlines()), "unspent", sum(a for a, _, _ in utxo.values()))
print("IDENTICAL" if text == ref else "DIFFERENT")
