# Main network

This guide gives the parameters of the NEB main network and what differs from the test network. The consensus rules are those of `spec/node.md`, `spec/shares.md` (section 11) and `spec/nebhash-v2.md`. Running a node, a relay, the miners and the wallet works as described in [testnet.md](testnet.md), without the `--testnet` option.

## Summary

| Item | Value |
|---|---|
| network name | mainnet |
| software version | 1.0.0 |
| protocol version | 3 |
| address prefix | `neb` (bech32m, 32-byte lock data) |
| peer-to-peer port | 3440 |
| work service port | 3441, loopback only |
| JSON-RPC port | 8432, loopback only |
| default data directory | `./neb-data` |
| models | two, from the genesis block: slot 0 `kepler-v2` (KEP1 network), slot 1 `retina64` (RET2 network) |
| dataset file | `neb-datasets.bin`, 222,320,684 bytes, two datasets |
| dataset file hash (BLAKE3) | `fbabd0bdde0b43aef946e7031ff6e037c8a2a5d0ae2616b49b2b49b11af8010e` |
| dataset file hash (SHA-256) | `500846f682669743b1305776cfbf19edc11ad3db1c809bd5a4fe5b33cf196fa4` |
| dataset upgrades | none |
| block spacing | 60 s (ASERT, half-life 600 s) |
| reward | 20 coins per block, halving every 450,000 blocks; 5 % of each reward to the project output |
| supply | the rewards sum to 17,999,999.9415 coins; the genesis block adds the carry-over of the test networks |
| coinbase maturity | 100 blocks |
| proof-of-work limit bits | `0xd5800000` (2^236) |
| share target | the block target times 32 |
| genesis time | 1795010400, 2026-11-18 14:00:00 UTC: the start of the network |
| genesis target bits | `0xc7b38c00`, 32 times the target of block 12900 of the second test network: about 1.2 * 10^10 attempts per block |
| beacon | `00000000000000000002038eee1b35e0033afa7f0d6f142bf07b941611b748a8`, Bitcoin block 976143 |
| genesis id | `fcf3b2bdbb0a3f7621c14db4e229459375a1614d0480b4aa87a9aa01d3d58f88` |
| network magic | `e37b638b` |
| genesis coinbase | 217 outputs: the reward, the project share and 215 locks of the carry-over, 30,211.89999952 coins |
| default relays | `seed1.neuralenginebyte.org:3440`, `seed2.neuralenginebyte.org:3440` |

The ports are those of the test network. A machine that runs a node of each network gives one of them other ports (`--port`, `--work-port`, `--rpc-port`).

## The start

Block 1 cannot be dated before the genesis time, and a node refuses a block dated more than 120 s ahead of its clock: a node started earlier connects to its peers, says how long is left and stays at height 0. A miner can connect to it at any time: it receives its first job about two minutes before the genesis time. The first blocks are mined at the genesis target. When the attempt rate of the network is above what that target expects, they come a few seconds apart and the target halves every 10 blocks until the spacing is 60 s; when it is below, block 1 takes longer and the target then follows the real block times (ASERT, `spec/node.md` section 6.2).

The beacon was not chosen: it is the lowest Bitcoin block whose timestamp (1794841795) is later than the time of block 12900 of the second test network (1794840900), a rule fixed before that block existed. The genesis id depends on it, so no block of this network could be prepared before 2026-11-16.

## What differs from the test network

- **The models.** Both slots hold a convolutional network from block 0: KEP1 on the Kepler transit signals (26,000 weights) and RET2 on the retinal scans at 64 x 64, normal or abnormal (22,280 weights). Each starts from the untrained model derived from the genesis id and its slot (`spec/shares.md` section 11.2). The test network started with two 784-128-10 models and moved to these networks at blocks 8800 and 8801.
- **The dataset file.** It holds the last two datasets of the file of the test network, and `tools/make_mainnet_datasets.py` rebuilds it from that file. A node refuses a file with another hash, so the file of one network cannot be used on the other.
- **The genesis block.** After the reward of height 0 and the project share, its coinbase pays the carry-over of the two test networks: one output per lock (next section).
- **The tags.** `neb-mainnet-beacon-1`, `neb-mainnet-extranonce` and `neb-mainnet-magic` (`spec/node.md` sections 6.6 and 8.2), so the two networks have different genesis ids and magics and their nodes do not talk to each other.

One attempt takes about 2 ms of one CPU core on KEP1 and about 7 ms on RET2. A block alternates between the two slots, so the attempt rate of a miner changes from one block to the next.

## Carry-over of the test networks

Each lock receives, in the genesis block, a tenth of the balance it held on each of the two test networks: 1,000 test coins give 100 coins. The rule, the lists and the commands that rebuild them are in [carry-over.md](carry-over.md).

- A lock is the same on every network. The wallet file that mined or received coins on a test network holds the carried coins, with the same address. There is nothing to claim.
- The outputs are coinbase outputs of height 0. They can be spent after 100 blocks, like any reward. Before that the wallet shows them as immature.
- Keep the wallet file and its password. Nobody can pay a carry-over to another lock after the genesis block.

```
neb-wallet --rpc http://127.0.0.1:8432 --wallet wallet.txt balance
```

## Running a node and a miner

```
neb-node --run --dataset neb-datasets.bin --datadir data --miner-lock "$(neb-wallet --wallet wallet.txt lock)"
neb-miner-cpu --node 127.0.0.1:3441 --dataset neb-datasets.bin
```

The options are those of [testnet.md](testnet.md), sections "Running a node" to "Dashboard of the miners". A node of the main network must not be given the data directory of a test network node: the store belongs to one genesis.

A node finds its peers from the default relays of the network, then from its address book: it asks its peers for the addresses they reached and keeps two connections to them, so it stays on the network when the relays are down (`spec/node.md` section 8.4). A relay or a pool node that should be found by others listens on a public address (`--listen 0.0.0.0`).

## Checking the genesis block

```
neb-node --genesis
```

prints the beacon, the genesis id, time and target, the network magic, the dataset hash and the number of outputs of the genesis coinbase. Every node prints the genesis id at start, and two nodes with different ids do not connect. The carry-over outputs of the block are the merged lists of [carry-over.md](carry-over.md): anyone can rebuild both lists from the published chains of the test networks and compare.

## The test network

The test network of protocol 3 stays in the software for trials of future upgrades:

```
neb-node --run --testnet --dataset TESTNET-neb-datasets.bin
```

It needs its own dataset file (four datasets, 259,897,080 bytes, from release 0.3.1) and uses its own data directory, `./neb-testnet2-data`. Its coins have no value.
