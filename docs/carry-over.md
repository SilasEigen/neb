# Carry-over of the test networks

A tenth of what a lock held on the two test networks is paid to the same lock in the genesis block of the main network.
The rule is in `spec/node.md` section 6.6; this page gives the lists and how anyone rebuilds them.

## The rule

- The balance of a lock on a test network is the sum of its unspent outputs at one block of that network, coinbase
  outputs of any age included.
- The lock receives `floor(balance / 10)` base units for each test network: 1,000 coins held give 100 coins. There is no
  cap.
- The two results are added. A lock is the same on every network, so the wallet file that held the coins spends them:
  nothing has to be claimed.
- Two locks receive nothing: the lock of 32 zero bytes, which holds the unspendable reward of each genesis block, and the
  project-share lock.
- The outputs are coinbase outputs of height 0 of the main network. They can be spent after 100 blocks, like any reward.
- The carried coins add to the supply of the main network, on top of the block rewards. The totals are in the table
  below. The genesis block creates no other coin besides the reward of height 0.

## The lists

A list has one line per lock, by increasing lock: the lock type, the lock data in hexadecimal and the base units. The
lists are in `node/crates/neb-node/data/`, and the node builds the genesis block from them.

| network | block | locks | carried over | BLAKE3 of the list |
|---|---|---|---|---|
| first test network (protocol 2) | height 3001, its last block, `f2d0fe4dd450d5d56fd2a6959daf7c28cf71e6d52c00aa83146eb35c50b30f16` | 69 | 5,701.9 coins | `05b0f48dc4d698530016c147e9ba56aef24f4237183fae568233fb9d42eda32c` |
| second test network (protocol 3) | height 12900, announced before it was reached, `0b58879dafb768ca484d0a8894f8afc34b246c4813be5a77869c831f1f713106` | 159 | 24,509.99999952 coins | `a0fea9e6628da361568ebce88a8d942ab68b0c667bb1355350d257f9a3049916` |

Thirteen locks are in both lists. The genesis block pays 215 locks, 30,211.89999952 coins in total.

## Rebuilding a list

The second test network, from a dump of its chain (one block per line, as the hexadecimal of `getblock`, heights 1, 2, 3
and so on):

```
cargo run --release -p neb-node --example carry_over -- neb-datasets.bin chain-dump.txt HEIGHT > list.txt
```

`neb-datasets.bin` is here the dataset file of the test network (four datasets, release 0.3.1), and `HEIGHT` is 12900.
The tool replays the chain with the rules of the node, every proof of work and every share checked, and stops at
`HEIGHT`. It prints the list on the standard output and, on the standard error, the block id at that height, the number of
locks, the total and the BLAKE3 hash of the list. The block id can be compared with `getblockhash HEIGHT` of any node.

`tools/carry_check.py` is a second computation with a parser and a set of unspent outputs of its own. It checks no
proof of work and no signature, only the balances:

```
python -I tools/carry_check.py chain-dump.txt node/crates/neb-node/data/carry-testnet2.txt 12900
```

It prints `IDENTICAL` when the list it computes is the list of the repository.

The first test network ran protocol 2, which this release does not replay. Its list was made by replaying its chain with
release 0.1.1, and the same script rebuilds it from the dump of that chain, where no block holds a transaction besides
its coinbase:

```
python -I tools/carry_check.py testnet1-dump.txt node/crates/neb-node/data/carry-testnet1.txt
```

The dumps of the two chains are published with the release of the main network.
