# Testnet

This guide covers the NEB test network: parameters, genesis, running a node, a CPU miner and a GPU miner, the peer-to-peer protocol, JSON-RPC and the wallet. Coins on this network have no value, and the key of the genesis output is published (see "Keys"). The consensus rules are those of `spec/node.md` and `spec/shares.md` (section 11): the proof of work is a training-share attempt, and every block trains a 784-128-10 model that the chain owns.

## Summary

| Item | Value |
|---|---|
| network name | testnet |
| peer-to-peer port | 3440 |
| work service port | 3441 (loopback) |
| JSON-RPC port | 8432 (loopback) |
| default data directory | `./neb-testnet2-data` |
| genesis target | `0xd5800000` (2^236); the proof-of-work limit is the same |
| training samples | four datasets in one file pinned by hash: slot 0, 13,404 Kepler DR24 transit signals (NASA Exoplanet Archive; 1 planet candidate, 0 false positive); slot 1, 34,464 retinal OCT scans (OCTMNIST, CC BY 4.0; 4 labels); then, for the upgrade to nebhash v2 at blocks 8800 and 8801, the same Kepler signals for the KEP1 network and 51,696 retinal OCT scans at 64 x 64 labelled 0 normal or 1 abnormal for the RET2 network (see "Upgrade to nebhash v2 at block 8800"). A block of height h trains slot h mod 2 |

The release has this file as `neb-datasets.bin` (260 MB) and as `neb-datasets.zip` (26 MB, the same file compressed); download one and check its hash, or create it once, outside the source tree, and check its hash. `tools/make_datasets.py` needs Python with numpy; it downloads the DR24 TCE table and the DV light curves from the NASA Exoplanet Archive (about 135 GB that pass through the cache once; only 784 values per signal are kept) and OCTMNIST from Zenodo (55 MB, and 312 MB at 64 x 64), and also writes the samples kept out of the chain (`kepler-test-q8.bin`, `retina-test-q8.bin`, `retina64-test-q8.bin`):

```
python tools/make_datasets.py --cache /path/to/cache --out /path/to/neb-datasets.bin
neb-node --dataset-hash /path/to/neb-datasets.bin
296a174edbf67339fa0af341f063154292bba224b3d2cacf5b18b31c70da80eb
```

Start a node and a CPU miner:

```
neb-node --run --dataset /path/to/neb-datasets.bin --datadir ./data-a
neb-miner-cpu --node 127.0.0.1:3441 --dataset /path/to/neb-datasets.bin --threads 8
```

The node refuses to start when the dataset file does not have the pinned hash. The miner shows a dashboard (see "Dashboard of the miners") or, with `--plain`, a status line every 5 s: attempts per second, shares, and blocks found (accepted and rejected by the node). An attempt meets the genesis target with probability 2^-20 and the share target with probability 2^-15, so the expected time to a block is 2^20 divided by the attempt rate, and a block comes with about 32 shares. The shares of a block train the model one block later.

`neb-node --model-accuracy --dataset FILE --datadir DIR` replays the chain of a stopped node and prints, for each model, how many of the first 2000 samples of its dataset the initial model and the chain model classify correctly (`--samples N` changes the count; a value that is not a non-negative integer is a usage error with exit status 2).

## Network parameters

| Item | Value |
|---|---|
| network name | testnet |
| address prefix | `neb` (bech32m, 32-byte lock data) |
| peer-to-peer port | 3440 |
| work service port | 3441, loopback only |
| JSON-RPC port | 8432, loopback only |
| software version | 0.3.0 |
| protocol version | 3 (one model per dataset; a node of protocol 2 or 1 does not connect) |
| network magic | `1db70ba3` (four bytes, hex) |
| default maximum peers | 8 (`--max-peers`) |
| genesis timestamp | 1794078000 (2026-11-07 19:00:00 UTC) |
| genesis target bits | `0xd5800000` (2^236) |
| proof-of-work limit bits | `0xd5800000` (2^236, the genesis target) |
| share target | the block target times 32 (2^241 at the genesis target) |
| genesis id | `8271952aba2ac5340c1dfe884caa1f2c6ee765487fa2117ddcd90710ce87248d` |
| dataset file hash (BLAKE3) | `296a174edbf67339fa0af341f063154292bba224b3d2cacf5b18b31c70da80eb` |
| dataset upgrades | from block 8800, slot 0 works on the third dataset of the file with the KEP1 network, and from block 8801 slot 1 on the fourth with the RET2 network, each from a new model (`spec/shares.md` section 11.7, `spec/nebhash-v2.md`) |

The magic is the first four bytes of `BLAKE3("neb-testnet-magic" || genesis_id)`, so two chains with different genesis blocks cannot exchange messages by accident. The initial model is derived from the genesis id (`spec/shares.md` section 11.2).

The compact form of 2^236 is `0x800000 << 213`, and 213 is `0xd5`. A ticket is a 256-bit value, so an attempt meets the genesis target with probability 2^-20 and a block takes about 1,048,576 attempts. At 16,000 attempts per second this is one block in about 65 s, close to the 60 s spacing. ASERT then follows the real block times. The limit equals the genesis target, so the chain never gets easier than at launch.

The subsidy, halving schedule, project share, transaction format, block limits, signature rules and ASERT algorithm are fixed by the specification (`spec/node.md` sections 5 and 6). This file gives the values of the network.

## Upgrade to nebhash v2 at block 8800

From block 8800 the Kepler model, and from block 8801 the retina model, are small convolutional networks (`spec/nebhash-v2.md`). The retina model also learns a new task: tell a normal scan from an abnormal one (choroidal neovascularization, diabetic macular edema or drusen), on scans of 64 x 64 instead of 28 x 28. Each starts from a new model at its block (`spec/shares.md` section 11.7). The chain, the blocks before 8800 and the coins do not change.

- **Nodes** need release 0.3.0 and the new `neb-datasets.bin` (four datasets, 259,897,080 bytes, hash above) before block 8800. Up to block 8799 the release accepts the same blocks as the previous one, so it can be installed at any time before. A node without it refuses block 8800 and every block after it.
- **Miners** need release 0.3.0 (the CPU miner and the GPU miner both have the new networks) and the new `neb-datasets.bin`. A miner with an older release or the old file stops mining at block 8800.
- **One attempt costs more**: about 2 ms of one CPU core for KEP1 and 7 ms for RET2, against 0.4 ms before, and tens to hundreds of times a v1 attempt on a GPU. Hash rates in attempts per second drop by that factor at the upgrade and the difficulty follows (ASERT, 600 s half-life). The two slots keep one difficulty, so the Kepler blocks come faster than the retina blocks and the average stays at 60 s.

## Genesis

The genesis block is deterministic. Every node derives the same bytes from the public beacon string

```
000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
```

which is the hash of the first block of an external public proof-of-work chain (spec/node.md section 6.6). The coinbase witness is

```
height (u32, 0) || extranonce (8 bytes) || miner data (32 bytes)
```

where the extranonce is the first eight bytes of `BLAKE3("neb-testnet-extranonce" || beacon)` and the miner data is `BLAKE3("neb-testnet-beacon-2" || beacon)` (`spec/node.md` section 6.6). The beacon block already exists, so on this network it gives no protection against a pre-mined start.

The genesis coinbase pays the reward of height 0: 19 coins to a lock that no key can unlock, so they can never be spent, and the 1 coin project share to a one-of-one multisig lock of the project public key (see "Keys"). The header timestamp is fixed and the nonce is zero. The genesis block is valid by definition and its proof of work is not checked; every later block is checked normally.

`neb-node --genesis` prints the genesis values, the genesis miner lock and the project lock without starting a node. The first lines of its output are:

```
network     : testnet
beacon      : 000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f
genesis id  : 8271952aba2ac5340c1dfe884caa1f2c6ee765487fa2117ddcd90710ce87248d
genesis time: 1794078000
genesis bits: 0xd5800000
pow limit   : 0xd5800000
magic       : 1db70ba3
p2p port    : 3440
work port   : 3441
rpc port    : 8432
dataset hash: 296a174edbf67339fa0af341f063154292bba224b3d2cacf5b18b31c70da80eb
```

### Keys

The miner output of the genesis block is locked to 32 zero bytes. A single-key lock is the BLAKE3 hash of a public key, and no key is known whose hash is zero, so these 19 coins can never be spent.

A node pays the reward of its jobs to the lock given with `--miner-lock`. Without it, the node runs (a relay needs no lock) but its work service refuses every miner.

The project share (spec/node.md section 6.3) goes to the one-of-one multisig lock of a project key whose secret is held by the project and is not published:

```
project public key  25dadba82b1372fd76c1901524cc742224661d6fc5f0ba1b4e95f526a3aa72d3
project lock        270bbd6380c0c7022176e202083d5671ed6bb7bc0c27350bcbd134846ce87dd5
```

The holder of the project key sees and spends the project share with `neb-wallet`: it recognises the one-of-one multisig lock of a wallet key (`neb-wallet public-key` prints the public key of the first key).

## Storage

A node keeps its data in the directory given by `--datadir` (default `./neb-testnet-data`):

- `blocks.dat` holds complete blocks, with their share lists, in an append-only log. A record is `magic "BLK1" | length u32 | block bytes | checksum u32`, where the checksum is the first four bytes of BLAKE3 over the block bytes. Every record is synced before the append returns.
- `index.dat` holds the chain history as append-only connect and disconnect events. Each connect event carries the cumulative work of the tip it connects. The node recomputes the value from the block headers when it reads the index, so a damaged value cannot change the chain.

The files hold no model. On start the node replays the best chain from the genesis block, checks every block and rebuilds the UTXO set and the model. The replay costs one ticket evaluation per block and one per share. A crash can leave one torn record at the end of either file; opening the store truncates that tail and keeps every complete record before it. If the index is missing or damaged, it is rebuilt from the block log by following parent links and choosing the tip with the most cumulative work.

## Build

```
cd node
cargo build --release --locked     # neb-node, neb-miner-cpu and neb-wallet; a C compiler is needed
cargo test --locked
```

The binaries are in `node/target/release`.

## Running a node

```
neb-node --run --dataset /path/to/neb-datasets.bin --datadir ./data-a
```

The node opens or creates the store, loads the genesis block, listens for peers on 3440, serves JSON-RPC on 127.0.0.1:8432 and serves miners on 127.0.0.1:3441. A second node on the same machine needs other ports:

```
neb-node --run --dataset /path/to/neb-datasets.bin --datadir ./data-b --port 3450 --work-port 3451 --rpc-port 8433 --connect 127.0.0.1:3440
```

Both nodes must have the same genesis id; the node prints it at start. Options:

```
--dataset FILE     dataset file with the pinned hash (required)
--datadir DIR      store directory (default ./neb-testnet-data)
--listen ADDR      peer bind address (default 127.0.0.1)
--port N           peer port (default 3440)
--work-port N      miner work service port on 127.0.0.1 (default 3441)
--rpc-port N       JSON-RPC port on 127.0.0.1 (default 8432)
--connect HOST:PORT  peer to dial once at startup, repeatable
--seed HOST:PORT     peer to keep dialled and re-dial after a loss, repeatable
                     (--addnode is an alias); HOST is an IP address or a name,
                     resolved at every dial
--no-seeds         do not dial the default seeds of the network
--max-peers N      peer sessions, inbound and outbound together (1 to 1024,
                   default 8)
--miner-lock HEX   lock paid by the jobs, 64 hex characters, as printed by
                   `neb-wallet --wallet FILE lock`; without it the work service
                   refuses miners
--pow-budget N     ticket evaluations per second that one peer may cost
                   (default 200, at least 1)
--color MODE       auto, always or never
```

`neb-node --check-hardware` prints the CPU and the proof-check path. `neb-node --selftest`, `--shares-selftest`, `--chain-selftest` and `--work-selftest` run the built-in checks. With the default `--listen`, only peers on the same machine can connect; bind a public address to accept remote peers.

## Running a public relay

A relay is a node that accepts many inbound peers; the nodes of the miners dial it. It needs no miner and no wallet. By estimate, two vCPUs and 2 GiB of RAM should be enough (not measured on a deployed relay): a block costs about 17 ms of one core to check, the node keeps the dataset file (260 MB) and the model states (about 46 MB), and the traffic is small.

```
neb-node --run --dataset /srv/neb/neb-datasets.bin --datadir /srv/neb/data --listen 0.0.0.0 --max-peers 256 --seed other-relay.example.org:3440 --color never
```

- **Ports.** Open TCP 3440 inbound. The work service (3441) and the JSON-RPC (8432) stay on the loopback interface of the relay.
- **Clock.** Run NTP. A node refuses a block that is more than 120 s ahead of its clock, so a drifting clock refuses valid blocks.
- **Several relays.** Give each relay a `--seed` for the other, so that the relays stay one network.
- **Names.** Publish the relays under host names (A records, and AAAA only if the node listens on IPv6). Nodes resolve the name at every dial, so a relay moves to another address by changing the record. A node keeps one session per name, so each relay needs its own name: a name with two addresses links a node to one of them. The two relays of the test network are `seed1.neuralenginebyte.org` (New York) and `seed2.neuralenginebyte.org` (London), with the records set to "DNS only" (an HTTP proxy does not carry port 3440); they are the default seeds (`DEFAULT_SEEDS` in `node/crates/neb-node/src/params.rs`).
- **Load.** Each peer session may cost up to `--pow-budget` ticket evaluations per second, and an IP address that sends an invalid proof is banned for 24 hours.
- **Monitoring.** `getinfo` gives the height and the number of connections, `getpeerinfo` lists the sessions.
- **Public JSON-RPC.** The JSON-RPC has no authentication and `neb-wallet` speaks plain HTTP. A public endpoint needs a reverse proxy in front of the loopback port, with TLS and a limit on the methods.

A systemd unit:

```
[Unit]
Description=NEB relay
After=network-online.target

[Service]
User=neb
ExecStart=/usr/local/bin/neb-node --run --dataset /var/lib/neb/neb-datasets.bin --datadir /var/lib/neb/data --listen 0.0.0.0 --max-peers 256 --color never
Restart=on-failure
LimitNOFILE=4096

[Install]
WantedBy=multi-user.target
```

## Running the CPU miner

```
neb-miner-cpu --node HOST:PORT (--dataset FILE | --synthetic) [--threads N] [--engine ref|avx512|avx512-sparse|avx512-dense] [--plain]
neb-miner-cpu (--dataset FILE | --synthetic) --engine NAME --selftest
```

`--node` is the work service of a node (3441 by default). `--dataset` must be the file the node runs on: a miner on another dataset computes other tickets, and the node refuses its shares and blocks. `--synthetic` uses the synthetic dataset of `spec/shares.md` section 9 and is for demonstration networks. `--threads` defaults to the number of logical CPUs.

The node sends the model of the parent block (50,816 bytes of packed weights, and from block 8800 13,000 for Kepler and 11,140 for the retina) and then a job, at each block. The miner makes one attempt per nonce of the reference header of the job, sends every ticket that meets the share target as a share and the first that meets the block target as a solution. It keeps the job until the next one arrives (`spec/shares.md` section 11.5).

`--engine` chooses how an attempt is evaluated. The default, `ref`, is the C reference that the node itself runs. `avx512` (the sparse variant, which skips the zero pixels and the hidden units with no error) and `avx512-dense` use AVX-512 VNNI instructions, need a processor with AVX-512 F, BW, VL, DQ and VNNI, and give the same tickets. They are mining optimisations, not consensus, and never the default. At start-up the miner runs a self-test of the engine against the reference (five models, the dataset of the node and the synthetic one; digest, ticket, sample indices and clipped gradients) and uses the engine only if it passes; otherwise it prints why and uses the reference. While mining it checks each new model on 16 attempts, evaluates every share and every solution again with the reference before it sends it, and audits one attempt in 4,096 that the engine reported as no hit. If the engine and the reference ever differ, the engine is switched off for the rest of the run and the miner continues with the reference. A hit that an engine misses (a false negative) cannot be seen by checking hits; the audits and the model check make a systematic fault unlikely to go unnoticed but do not remove the risk. `--selftest` runs the self-test alone and exits with 0 when the requested engine passed.

## Running the GPU miner

```
neb-miner-gpu --node HOST:PORT (--dataset FILE | --synthetic) [--devices LIST] [--gpu-kernel k0|k1|k2] [--plain] [--stats FILE]
neb-miner-gpu --selftest [--gpu-kernel k0|k1|k2] [--dataset FILE | --synthetic] [--devices LIST]
```

The GPU miner takes the place of the CPU miner on the same work service, with the same options for `--node`, `--dataset` and `--synthetic`. It needs an NVIDIA GPU of compute capability 7.5 or later (Turing and later; the 8-bit dot-product instruction `dp4a`) and a driver for CUDA 13 (580 or later); the CUDA runtime is linked into the binary. The release has it for Windows and Linux (x86_64, glibc 2.34 or later). `--devices` is a comma-separated list of device indexes (default: all devices); every device mines the same job, each on its own ranges of nonces. `--stats FILE` rewrites a JSON file every two seconds with the rate of each device (the HiveOS package reads it).

A device evaluates 65,536 nonces of the reference header at a time, keeps the nonces whose ticket meets the share target, and the miner evaluates those nonces again with the C reference before it sends them as shares (and as a solution when the ticket also meets the block target). A nonce that the device reports and the check refuses is counted and printed as a warning. A scan that has more hits than the device can report (4,096) is split into halves until every hit is reported; a result is never cut.

`--gpu-kernel` chooses the kernel. `k0`, the default, is the dp4a kernel of `pow/gpu/shares_cuda.cu`. `k1` and `k2` use the int8 tensor-core instructions (`pow/gpu/shares_tc.cu`; K2 skips the zero entries) and need compute capability 8.0 or later (Ampere and later); on an older device the miner says so and uses K0. They are mining optimisations, not consensus, and never the default. A device uses `k1` or `k2` only after a self-test against the C reference (the hits of three bounds and exact tickets, for five models, the dataset of the node and the synthetic one). While mining, the miner checks each model it uploads on four nonces, audits the exact ticket of one nonce every 64 batches, and treats a nonce that the reference refuses as a fault; on any fault the device goes back to K0, uploads the model again and scans the batch again. `--selftest` runs the self-test of the requested kernel on each device (by default K0, K1 and K2, skipping the kernels a device does not support; the synthetic dataset unless `--dataset` is given) and exits with 0 when every tested kernel equals the reference.

Build, with the CUDA toolkit (13.x) and, on Windows, the MSVC build tools:

```
cd node
cargo build --release --locked -p neb-gpuminer --features cuda
```

`NEB_CUDA_ARCHS` sets the GPU generations to compile for, as compute capabilities without the dot (default `75,86,89,90,120`; the tensor-core kernels are built for all of them but only run from 8.0 on). The build makes three libraries, one for K0 and the scan, one for K1 and K2 and one for the nebhash v2 kernel: K0 and K1/K2 must not be device-linked together (`pow/gpu/README.md`). On Linux set `CUDA_PATH` to the toolkit (`/usr/local/cuda-13.3`, for example) and put its `bin` in `PATH`; the build needs `gcc` and `nvcc`.

## Mining on HiveOS

The release has a package for the custom miner of HiveOS, `nebminer-0.3.0.tar.gz`. It holds the node, the wallet, the GPU miner and the dataset file, and the four scripts of the HiveOS interface (`tools/hiveos`). `tools/hiveos-package.sh` builds it. The miner uses every GPU of the rig by itself, so the flight sheet does not choose cards, and an option such as `--devices 0,1` in the extra arguments narrows the choice.

Create a flight sheet with a custom miner:

| Field | Value |
|---|---|
| Miner | Custom |
| Miner name | `nebminer` |
| Installation URL | the link of `nebminer-0.3.0.tar.gz` on the release page |
| Hash algorithm | leave empty or choose any |
| Wallet and worker template | your `neb1...` address (a suffix such as `.%WORKER_NAME%` is ignored) |
| Pool URL | one or more relays as `HOST:PORT`, separated by spaces |
| Pass | empty |
| Extra config arguments | arguments for `neb-miner-gpu`, for example `--devices 0,1` |

The rig needs an image based on Ubuntu 22.04 ("jammy"; the 20.04 images have a glibc that is too old), an NVIDIA driver of the 580 series or later (chosen in the settings of the rig) and cards of compute capability 7.5 or later (Turing and later). The coins go to the address of the flight sheet; the rig needs no wallet file and no secret key. `h-run.sh` starts the node (data in the package folder, log `/var/log/miner/nebminer/nebminer-node.log`) and the miner (log `/var/log/miner/nebminer/nebminer.log`, shown as the miner screen). `h-stats.sh` reads the stats file that the miner rewrites every two seconds with `--stats`: the rate of each card in kH/s, temperature, fan, PCI bus numbers, uptime and the blocks that the node accepted and refused. With the Pool URL field empty the node dials the default seeds of the network.

## Dashboard of the miners

`neb-miner-cpu` and `neb-miner-gpu` draw a dashboard on the alternate screen of a terminal of at least 120 columns and 30 rows; a smaller terminal shows a message instead. With `--plain`, or when the output is not a terminal, they print a status line every 5 s.

- **Chain.** Each block that the miner found is a cube that fills with its confirmations, with a loading bar and a status: `PENDING n / 100` below 60 confirmations, `FINAL` from 60 (a node never reorganises more than 60 blocks, `spec/node.md` section 6.5), `SPENDABLE` from 100 (coinbase maturity). A block that the node refuses is removed from the list.
- **Miner.** The rate of the last two seconds and the average, the attempts and shares since the start, the number of blocks, the target of the current job (`bits`) and the height of the tip.
- **Last block.** Nonce, block id, whether the node accepted it, and the seconds from the job to the solution. The title flashes when the node accepts a block.
- **Hardware.** The threads of the CPU miner, or for each GPU the temperature, power, shader clock and utilisation read through NVML (part of the driver) every second, and the efficiency in kH/J.
- **Keys.** On Windows consoles, `q` quits, `b` replays the flash of a new block, and the arrow keys, Page Up, Page Down, Home, End, `j` and `k` scroll the list of blocks. In other terminals the keys are not read and Ctrl-C quits.

## Peer-to-peer protocol

Messages use the framing of `spec/node.md` section 8.2:

```
magic (4) || command (12, ASCII, zero padded) || length (u32) || checksum (4) || payload
```

The checksum is the first four bytes of BLAKE3 over the payload and the maximum payload is 2 MiB. The commands are `version`, `verack`, `ping`, `pong`, `getaddr`, `addr`, `getheaders`, `headers`, `getblocks`, `inv`, `getdata`, `notfound`, `block` and `tx`.

The handshake exchanges the protocol version, the chain height, a random nonce and the genesis id. A peer that sends the node's own nonce is a self connection and is closed. After `verack` the node asks for the block ids that follow its tip with `getblocks`. The announced ids are fetched with `getdata`, 16 blocks at a time, and each block is checked and connected before the next batch is asked for. The proof of a block needs the model built by the bodies of its ancestors, so there is no header-first step. Blocks and transactions are announced with `inv`.

At most `--max-peers` peers are kept (8 by default), seeds included. `--connect` dials a peer once at startup. `--seed` (alias `--addnode`) keeps a peer dialled: a lost connection is dialled again with a back-off of 1 s that doubles up to 30 s, and the peer cap applies to every dial. A peer is `HOST:PORT`; a name is resolved at every dial. A node that is given no `--connect`, no `--seed` and no `--no-seeds` dials the default seeds of the network (`DEFAULT_SEEDS` in `params.rs`; local policy, outside the consensus rules). Peer state is in memory, so addresses are not remembered across restarts. There is no address discovery: a node knows the peers it is given and the peers that dial it.

### Ticket budget and ban

A bad block costs the receiver ticket evaluations and costs the sender nothing, so the node follows `spec/node.md` section 8.1. The cheap checks (version, known parent, expected `bits`, time window, body shape) run first and the ticket evaluations run last.

- Each peer session has a budget of `--pow-budget` ticket evaluations per second (default 200), counted as a token bucket that holds one second of tokens.
- A block from a peer takes one token for its proof and one for each of its shares, when it reaches the proof check. A session without a token waits for the next one. No block is dropped, and a flood is limited to the budget of its own session.
- A peer that sends a block that passes the cheap checks but has an invalid proof of work or an invalid share is disconnected, and its IP address is banned for 24 hours. Inbound connections from a banned address are closed at once and the node does not dial it, including for `--connect` and `--seed`. Orphan and duplicate blocks are not punished.
- Blocks from the local work service are neither limited nor banned.

The ban list is in memory, so a restart clears it.

## JSON-RPC

The server listens on 127.0.0.1:8432 by default and accepts HTTP POST with a JSON body. The methods are:

- `getblockcount`
- `getbestblockhash`
- `getblockhash height`
- `getblock hash` returns the block, share list included, as hexadecimal
- `getrawmempool` returns transaction ids
- `getpeerinfo`
- `getinfo` returns version, network, height, best block id, chain id, connection and miner counts, `bits`, ports, and `weightshash`, the weights hash of the model of the tip
- `getmodel [SLOT]` returns the packed weights of the model of slot SLOT (0 Kepler, 1 retina; default 0) in the state of the tip, with its height, block id, slot, dataset name and weights hash; `getmodel [SLOT, "genesis"]` returns those of the initial model of that slot. `getinfo` gives the hash of the whole state (`weightshash`) and the number of slots. `neb-node --model-accuracy --dataset FILE --datadir DIR` replays a stopped node's chain and counts, per slot, the samples the initial model and the chain model classify correctly.
- `sendrawtransaction hex`

Example:

```
curl -s -X POST http://127.0.0.1:8432 \
  -H 'content-type: application/json' \
  -d '{"jsonrpc":"1.0","id":1,"method":"getblockcount","params":[]}'
```

Parameters and results are listed in `spec/node.md` section 7.2.

## Wallet

`neb-wallet` keeps one or more Ed25519 secrets in a file, derives the `neb1...` address of each key and scans the node for coins. The file is encrypted with a password unless an empty password was chosen at creation. The default node is `http://127.0.0.1:8432` and the default file is `wallet.txt`. Commands:

```
neb-wallet --rpc http://127.0.0.1:8432 --wallet wallet.txt new
neb-wallet --wallet wallet.txt encrypt
neb-wallet --rpc http://127.0.0.1:8432 --wallet wallet.txt import SECRET
neb-wallet --rpc http://127.0.0.1:8432 --wallet wallet.txt list
neb-wallet --rpc http://127.0.0.1:8432 --wallet wallet.txt address
neb-wallet --wallet wallet.txt lock
neb-wallet lock-of ADDRESS
neb-wallet --rpc http://127.0.0.1:8432 --wallet wallet.txt balance
neb-wallet --rpc http://127.0.0.1:8432 --wallet wallet.txt send neb1... 1.5
```

`new` creates a wallet with one key and asks for a password (typed twice; an empty password leaves the file in clear). `encrypt` encrypts a plain wallet file. `import` adds a key to an existing wallet or creates the file. `list` prints every address and `address` prints the first one. `lock` prints the 64-hex lock of the first key, the value that `neb-node --miner-lock` takes to pay mined coins to the wallet, and `lock-of ADDRESS` prints the lock of any `neb1` address, so that a rig can mine for an address without the wallet file. `balance` scans the best chain through `getblock`, `getblockhash` and `getblockcount` and reports confirmed and immature coinbase amounts. `send` selects confirmed coins from largest to smallest, pays a fixed fee of 0.0001 coin, signs every input and submits the transaction with `sendrawtransaction`. Change returns to the first key of the wallet. A change amount below 0.00001 coin is added to the fee. Every command that reads an encrypted file asks for its password on the terminal, or reads it from `NEB_WALLET_PASSWORD` in scripts. The key is derived from the password with Argon2id (64 MiB, 3 passes) and the secrets are sealed with XChaCha20-Poly1305; a wrong password and a modified file give the same error. A lost password cannot be recovered: keep a copy of the file and of the password. On Unix the file is created readable by its owner only (mode 0600).

The address payload is the 32-byte single-key lock data, encoded with bech32m and the prefix `neb`.

## Operating notes

- Peers are given with `--connect` and `--seed`, or come from the default seeds.
- Peer state and the ban list are kept in memory. Bans are by IP address, so peers behind one shared address share a ban.
- Tickets are evaluated with the portable C reference on the peer session threads and on the work service threads.
- The node keeps no orphan block pool and no pruned history. It keeps the model of the last 100 levels of the best chain and of up to 64 checked side-branch blocks; a block whose parent model is not kept is refused as an orphan.
