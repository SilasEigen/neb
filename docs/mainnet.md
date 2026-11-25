# Main network

This guide gives the parameters of the NEB main network and what differs from the test network. The consensus rules are those of `spec/node.md`, `spec/shares.md` (section 11) and `spec/nebhash-v2.md`. Running a node, a relay, the miners and the wallet works as described in [testnet.md](testnet.md), without the `--testnet` option.

## Summary

| Item | Value |
|---|---|
| network name | mainnet |
| software version | 1.0.9 |
| protocol version | 3 |
| address prefix | `neb` (bech32m, 32-byte lock data) |
| peer-to-peer port | 3440 |
| work service port | 3441, loopback only |
| JSON-RPC port | 8432, loopback only |
| default data directory | `./neb-data` |
| models | two, from the genesis block: slot 0 `kepler-v2`, slot 1 `tess`, both with the KEP1 network |
| dataset file | `neb-datasets.bin`, 16,087,037 bytes, two datasets |
| dataset file hash (BLAKE3) | `790b57fbd830f3dcbe7de9d1146aa89a0a20799118e30dd48a343a5c98f19e72` |
| dataset file hash (SHA-256) | `57e22a4b8aacecb2ebae62f76307ddccb04198ed5c4ba15c1dee90ca7a7c2174` |
| dataset upgrades | none |
| block spacing | 60 s (ASERT, half-life 600 s) |
| reward | 20 coins per block, halving every 450,000 blocks; 5% of each reward to the project output |
| supply | the rewards sum to 17,999,999.9415 coins; the genesis block adds the carry-over of the test networks |
| coinbase maturity | 100 blocks |
| proof-of-work limit bits | `0xd5800000` (2^236) |
| share target | the block target times 32 |
| genesis time | 1795010400, 2026-11-18 14:00:00 UTC: the start of the network |
| genesis target bits | `0xc7b38c00`, 32 times the target of block 12900 of the second test network: about 1.2 * 10^10 attempts per block |
| beacon | `00000000000000000002038eee1b35e0033afa7f0d6f142bf07b941611b748a8`, Bitcoin block 976143 |
| genesis id | `da2f5af7bb4c5e2bcc8d558b2f4f01d4c88557a943e6cff9cd035638fe9439ae` |
| network magic | `ad33cae5` |
| genesis coinbase | 217 outputs: the reward, the project share and 215 locks of the carry-over, 30,211.89999952 coins |
| default relays | `seed1.neuralenginebyte.org:3440` to `seed4.neuralenginebyte.org:3440` |

The ports are those of the test network. A machine that runs a node of each network gives one of them other ports (`--port`, `--work-port`, `--rpc-port`).

## The start

Block 1 cannot be dated before the genesis time, and a node refuses a block dated more than 120 s ahead of its clock: a node started earlier connects to its peers, says how long is left and stays at height 0. A miner can connect to it at any time: it receives its first job about two minutes before the genesis time. The first blocks are mined at the genesis target. When the attempt rate of the network is above what that target expects, they come a few seconds apart and the target halves every 10 blocks until the spacing is 60 s; when it is below, block 1 takes longer and the target then follows the real block times (ASERT, `spec/node.md` section 6.2).

The beacon was not chosen: it is the lowest Bitcoin block whose timestamp (1794841795) is later than the time of block 12900 of the second test network (1794840900), a rule fixed before that block existed. The genesis id depends on it, so no block of this network could be prepared before 2026-11-16.

## What differs from the test network

- **The models.** Both slots hold a KEP1 network (26,000 weights) from block 0. Slot 0 learns the Kepler transit signals, planet candidate or false positive (13,404 signals). Slot 1 learns the TESS signals of Astronet-Triage-v2 (Tey et al. 2023, CC BY 4.0), eclipse or junk: the 2,363 eclipses of its train and validation splits and 4,726 junk signals, in the two views of the Kepler dataset. The first sorts what a search finds, the second tells the planets among what is left. Each starts from the untrained model derived from the genesis id and its slot (`spec/shares.md` section 11.2). The test network started with two 784-128-10 models and moved to these networks at blocks 8800 and 8801.
- **The dataset file.** It holds the Kepler dataset of the file of the test network, copied unchanged, and the TESS dataset; `tools/make_mainnet_datasets.py` rebuilds it from that file and from the public record of the TESS signals, and writes the 1,860 test signals kept out of the chain. A node refuses a file with another hash, so the file of one network cannot be used on the other.
- **The genesis block.** After the reward of height 0 and the project share, its coinbase pays the carry-over of the two test networks: one output per lock (next section).
- **The tags.** `neb-mainnet-beacon-2`, `neb-mainnet-extranonce` and `neb-mainnet-magic` (`spec/node.md` sections 6.6 and 8.2), so the two networks have different genesis ids and magics and their nodes do not talk to each other.

One attempt takes about 2 ms of one CPU core. A block alternates between the two slots, which run the same network, so the attempt rate of a miner is the same on every block.

Version 1.0.0 named the retinal scans of the test network for slot 1 and had another genesis id (`fcf3b2bd...`) and magic; no block was mined on it. Its nodes and its dataset file do not work with this network, and a node refuses a data directory made by 1.0.0: that directory holds no block and is deleted; the wallet file is not in it.

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
