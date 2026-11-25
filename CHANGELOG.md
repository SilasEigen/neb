# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The protocol version and the genesis of the network are listed separately from
the software version: a node checks them, not the software version.

## [Unreleased]

## [1.0.9] - 2026-11-25

Same protocol, network, genesis and dataset file as 1.0.8, and the nodes and miners of 1.0.2 to 1.0.9 work together.
No consensus rule changes: what follows is relay and JSON-RPC policy. Node and pool operators should upgrade.

- Node: one thread writes the state file from one buffer that it keeps, where each write started a thread and
  took a buffer of its own.
- JSON-RPC: `getutxos` reads the outputs of the lock that is asked, from a list of the outpoints of each lock,
  where it walked the whole set of unspent outputs under the node lock. It takes two more optional parameters, a
  lock type and a minimum amount. The state file is unchanged: the list is rebuilt when the file is read.
- Mempool: a transaction with an output below 522 base units is not taken or relayed. That is what the spend of an
  output costs: its 41 bytes and the 133 bytes of the input that spends it, at 3 base units per byte. A block that
  holds such an output is valid.
- Node: the transactions of the blocks a reorganisation removes return to the mempool, except those the new branch
  holds or conflicts with.
- Node: a transaction given to `sendrawtransaction` is announced again every minute until a peer asks for it or it
  leaves the mempool, so a node without a peer at that moment does not lose it.

## [1.0.8] - 2026-11-25

Same protocol, network, genesis and dataset file as 1.0.7, and the nodes and miners of 1.0.2 to 1.0.8 work together.
Node and pool operators should upgrade.

- Node: a start no longer replays the whole chain. The node writes its state to `state.dat` in the data directory
  every 10 blocks and after a reorganisation, and at the next start reads it and replays the blocks stored after
  it. Without the file, with a damaged one, or with one of another tip, it replays the whole chain as before. On
  Linux and macOS a stop by SIGTERM or SIGINT writes the file first, so the next start replays no block. The
  first start after the upgrade replays the chain once and writes the file.
- Node: the shares of a block are checked on several threads, so a replay of the whole chain is shorter.
- Node: a reorganisation rebuilds the models from the genesis block only when it lacks the levels it needs. One
  that followed a reorganisation to a shorter chain always did.
- Work service: a job leaves out a transaction that spends a coinbase output the next block cannot spend yet, as
  can remain in the pool after a reorganisation to a shorter chain.
- Node: `seed3.neuralenginebyte.org` and `seed4.neuralenginebyte.org` join the default relays of the main network.

## [1.0.7] - 2026-11-24

Same protocol, network, genesis and dataset file as 1.0.6, and the nodes and miners of 1.0.2 to 1.0.7 work together.
Node and pool operators should upgrade.

- Peers: the frames for a peer wait in a queue of that peer, written by a thread of its own. A relay to all peers
  returns at once, where it wrote to them in turn: one peer that reads slowly no longer delays the others, nor the
  result sent to the miner of a block. While more than 2 MiB wait for a peer the node reads nothing more from it; a
  peer that takes nothing for 10 s is disconnected.
- Peers: the node answers a relayed block whose parent it lacks with `getblocks` to its sender, at most once for
  each tip of its own chain. It did so only for a block sent unasked, so after two blocks at one height a node could
  stay on the lighter branch until a new session.
- Work service: a result or a job leaves at once (`TCP_NODELAY` on the sessions).

## [1.0.6] - 2026-11-24

An optional release for pools: same protocol, network, genesis and dataset file as 1.0.5, and the nodes and miners
of 1.0.2 to 1.0.6 work together. Nobody has to upgrade.

- Work service: a job starts at the later of the clock and the median time plus one, where it started at the parent
  time plus one. On a parent dated ahead of the clock, the first block of the job is no longer past the 120 s limit.
- Work service: the listener keeps accepting sessions after an accept error, as the JSON-RPC listener does.
- Work service: a write to a session is limited to 10 s. A session that stopped reading is closed, and no longer
  holds the job of the other sessions.
- JSON-RPC: a request body can be about 2 MB, so `submitblock` takes a block of the largest size. The limit was
  1 MiB, half of such a block as hexadecimal text.

## [1.0.5] - 2026-11-23

An optional release: same protocol, network, genesis and dataset file as 1.0.4, and the nodes and miners of 1.0.2,
1.0.3, 1.0.4 and 1.0.5 work together. Nobody has to upgrade.

- Wallet: `balance` and `send` read the unspent outputs of the wallet keys with `getutxos`, where they read every
  block of the chain. Their time no longer grows with the height. Against a node before 1.0.3 the wallet reads the
  blocks, as before.
- JSON-RPC: the node serves a call when it arrives. A call on a new connection waited up to 50 ms.
- Node: after a block of a local miner, a line warns when no peer has completed the handshake.

## [1.0.4] - 2026-11-21

An optional release for pools: same protocol, network, genesis and dataset file as 1.0.3, and the nodes and miners
of 1.0.2, 1.0.3 and 1.0.4 work together. Nobody has to upgrade.

- Work service: a miner that connects, or whose submission was rejected, gets the newest job while it is on the
  best block and less than 30 s old, where it got a job of its own. A pool with many sessions no longer pushes
  valid jobs out of the 64 the node keeps.
- Work service: the first job id of a run is drawn at random, so an id of an earlier run of the node names no job.
- Work service: result code 4, unknown job. It was part of code 1.
- JSON-RPC: `getjob` gives the transactions and the share list of a job, and `submitblock` takes a whole block, so
  a pool can hand a solution to a second node when the node of the job does not answer.

## [1.0.3] - 2026-11-21

An optional release: same protocol, network, genesis and dataset file as 1.0.2, and the nodes and miners of both
versions work together. It adds calls for pools, exchanges and payout systems, and a payment to several addresses
in the wallet. Nobody has to upgrade; pool operators and exchanges benefit from it.

- JSON-RPC: `submitshares` hands share records to the node, so a pool no longer needs a peer session with its own
  node to relay the chain shares of its miners (`spec/pool.md` section 7).
- JSON-RPC: `getblock` with `true` gives the block as an object (height, confirmations, whether it is on the best
  chain, and its transactions with their outputs), and `gettransaction` gives one transaction, read in the block of
  the height it is given or in the mempool. The node still keeps no index of transactions.
- JSON-RPC: `getutxos` takes a limit and the outpoint to start after, so a lock with more than 20,000 unspent
  outputs is read by pages. Without a limit the method answers as in 1.0.2.
- `neb-node --trust-loopback`: a loopback address is never banned. A pool whose software sends its node an invalid
  record loses its session and connects again, where it was cut off for 24 hours. Off by default.
- `neb-wallet send` takes several `ADDRESS AMOUNT` pairs and pays them with one transaction, up to 254 recipients.
- `neb-wallet send`: the fee is the size of the transaction in bytes when that is above the fixed fee of 0.0001 NEB.
  A payment that needed more than about 74 inputs was refused by the node ("fee below 1 base unit per byte").

## [1.0.2] - 2026-11-19

An optional release: same protocol, network, genesis and dataset file as 1.0.1, and the nodes and miners of both
versions work together. It adds a kernel to the GPU miner.

- `neb-miner-gpu --gpu-kernel table`: a kernel for the convolutional networks that reads a table of per-sample
  gradients. For each model the miner evaluates every sample of the dataset once on the CPU and uploads one row per
  sample; a device then builds an attempt from 16 rows. It gives the tickets of the reference, which still checks
  every share and solution before it is reported. A device that fails the self-test of the kernel, or disagrees
  with the reference during the run, goes back to the kernel of 1.0.1. Never the default.
- `neb-miner-gpu --selftest --dataset FILE` runs both kernels of the convolutional networks.
- JSON-RPC: `getutxos` returns the unspent outputs of one lock on the best chain, so a wallet reads its balance and
  builds a payment without reading the chain block by block. A lock with more than 20,000 unspent outputs is an
  error. The consensus rules, the protocol version and the genesis are unchanged.
- Whitepaper: Assumption A (section 6.1) is stated on the ticket, and the section describes the table of per-sample
  gradients that brings an attempt down to 16 row reads, a sum and a hash. Sections 3, 8 and 9 are updated to match.
  No rule changes.

## [1.0.1] - 2026-11-17

The main network starts with another dataset in slot 1, so it has another genesis block. **Every node and every
miner of 1.0.0 must be replaced, with the new dataset file, before the start on 2026-11-18 at 14:00 UTC**; a node of
1.0.0 cannot connect to this network. The start time, the target, the beacon and the carry-over are unchanged.

- Main network: slot 1 trains the KEP1 network on signals found in the light curves of the TESS mission, labelled
  eclipse or junk (Astronet-Triage-v2, Tey et al. 2023, CC BY 4.0), in place of the RET2 network on the retinal
  scans. Slot 0, the Kepler transit signals, is unchanged. Both slots run the same network, so a miner has the same
  attempt rate on every block. In simulation the TESS model reaches about 95% after 3,000 blocks (1,860 test
  signals, 86.6% of them junk; 9 eclipses in 10 found).
- Dataset file of the main network: 16,087,037 bytes instead of 222,320,684, two `NMD2` datasets of architecture 1.
  `tools/make_mainnet_datasets.py` builds it from the file of the test network and from the public record of the
  TESS signals, and writes the test signals kept out of the chain.
- Genesis: the beacon tag is `neb-mainnet-beacon-2`, which gives a new genesis id and a new network magic
  (`docs/mainnet.md`). The coinbase of the genesis block pays the same reward, project share and carry-over lists as
  in 1.0.0.
- A node of 1.0.1 refuses the dataset file of 1.0.0 and a data directory made by 1.0.0, and says so. That directory
  holds no block: delete it. The wallet file is not in it and works as before.
- The test network (`--testnet`) and its dataset file are unchanged.

## [1.0.0] - 2026-11-16

This release was replaced by 1.0.1 before the start of the network: its genesis block had no successor.

The main network. Without an option the software runs it; the test network of protocol 3 stays behind `--testnet`.
The network starts on 2026-11-18 at 14:00 UTC, the time of its genesis block: block 1 cannot be dated earlier, and
a node refuses a block dated more than 120 s ahead of its clock. `docs/mainnet.md` lists the launch values.

- Main network (`docs/mainnet.md`): protocol 3, two convolutional models from block 0 (KEP1 on the Kepler transit
  signals, RET2 on the retinal scans), no dataset upgrade, its own tags, genesis, magic and dataset file (two
  datasets, 222,320,684 bytes). The default data directory is `neb-data`.
- Consensus: the models of a genesis block follow the network of the dataset of each slot (`spec/shares.md` section
  11.2). Chains whose slots start with the one-hidden-layer network, the test network included, are unchanged.
- Carry-over (`docs/carry-over.md`): the genesis block pays each lock a tenth of its balance on each of the two
  test networks, as coinbase outputs of height 0. the example `carry_over` of `neb-node` rebuilds the list of the second
  test network from its chain, and `tools/carry_check.py` computes both lists a second time with its own parser.
- `--testnet` for `neb-node --run`, `--genesis` and `--model-accuracy`. A test network node needs this option and
  the dataset file of release 0.3.1.
- Address exchange (`spec/node.md` section 8.4): a node asks the peers it dialled for addresses with `getaddr`,
  proves an address with a handshake before it gives it to others, keeps its book in `peers.txt` and keeps two
  connections to it besides the relays. A node that listens on 127.0.0.1 announces port 0 and is not offered.
- `tools/make_mainnet_datasets.py` builds the dataset file of the main network from the file of the test network.
- The wallet file is the same on both networks; the comment line of a new file no longer names the test network.
- Work service: a node whose genesis time is ahead of its clock gives no job until a block 1 can be accepted, and
  says how long is left. A miner connected earlier waits and receives its first job then.
- `neb-miner-cpu --engine table`: an engine for the convolutional networks that evaluates every sample of the
  dataset once per model state, then builds an attempt from 16 rows of that table. It gives the tickets of the
  reference, which still checks every share and solution before it is reported. Never the default.

## [0.3.1] - 2026-11-12

Same protocol, network, genesis and dataset file as 0.3.0, with the same rules at block 8800: nodes of both versions
work together. Only the console of `neb-node --run` changes, so the upgrade is optional.

- The height on the status line follows the best chain. It only moved when the work service of the node connected a
  block, so a node that synced from its peers kept showing its height at start.
- The node prints a line before it replays its stored chain at start, and the time the replay took. The replay checks
  every block and every share again and can take minutes, with nothing else on the screen.

## [0.3.0] - 2026-11-11

Same network, genesis and chain. From block 8800 the two models of the chain are small convolutional networks
(spec/nebhash-v2.md), each starting from a new model at its block (spec/shares.md section 11.7): KEP1 for the Kepler
transit signals from block 8800, RET2 for the retinal scans from block 8801, on scans of 64 x 64 labelled normal or
abnormal. **Every node and every miner must upgrade, with the new dataset file, before block 8800.** Up to block 8799
this release accepts the same blocks as 0.2.1.

- nebhash v2: the C reference `pow/ref/nebhash2.c` (one implementation for both architectures, read from a table of
  layers), the test vectors `vectors/nebhash2.txt` and `vectors/nebhash2_s1.txt`, and an independent Python oracle
  that replays them (`pow/oracle/nebhash2_oracle.py`).
- Dataset upgrades at a height (spec/shares.md section 11.7) and v2 datasets (`NMD2`, with the architecture id). The
  dataset file holds four datasets (259,897,080 bytes); the release also has it compressed as `neb-datasets.zip`.
- The CPU miner runs the v2 networks with the reference code; the AVX-512 engine stays for the v1 network.
- The GPU miner has a CUDA kernel for the v2 networks (`pow/gpu/nebhash2_cuda.cu`), with a self-test against the
  reference at start. The default GPU generations also include 9.0.
- A miner whose dataset file or release does not know the network of a job waits instead of mining it.

## [0.2.1] - 2026-11-07

Same protocol (3), network and genesis as 0.2.0: nodes of both versions work together. Operators of a node that serves
miners should upgrade.

- Work service: when no job went to the miners for 30 s, the node sends them a fresh job on the same tip. A job holds
  2^32 nonces, which a fast GPU tries in about two minutes; a miner that had tried them all stopped until the next
  block. Job pushes are serialized, so a job on an older tip never reaches the miners after one on a newer tip.

## [0.2.0] - 2026-11-07

Protocol 3 and a new test network. The chain trains one model per dataset: two 784-128-10 models on public scientific
data, the transit signals of the Kepler DR24 TCE table (NASA Exoplanet Archive) and the retinal OCT scans of OCTMNIST
(MedMNIST v2, CC BY 4.0). A block of height h trains slot h mod 2, and its shares train the model they were mined on
(spec/shares.md section 11.2). The network, the step rule and the engines are unchanged; with one dataset the rule is
that of protocol 2.

- The dataset file is a set of datasets (`NMDS`, spec/shares.md section 11.6), `neb-datasets.bin`, built by
  `tools/make_datasets.py`; the miners keep one dataset per slot, and the `model` frame carries the slot.
- `getmodel` takes a slot and returns the slot and the dataset name; `getinfo` gives the hash of the whole state and the
  number of slots; `--model-accuracy` reports each slot.
- New genesis, network magic and default data directory (`neb-testnet2-data`); nodes of protocol 2 do not connect.
- The independent Python oracle replays chains of several models, and `pow/tests/chains/pools-testnet2.txt` is a
  recorded two-model chain that it and the node replay to the same hashes.

## [0.1.1] - 2026-11-07

Same protocol (2), network and genesis as 0.1.0: nodes of both versions work together. Node operators should upgrade.

- Peers: a write to a peer that takes more than 10 s ends its session, a session that has not completed the handshake
  within 10 s is closed, inbound sessions leave two slots for outbound ones and are limited to two per address, and the
  listener keeps accepting after a transient error. A new block's job goes to the miners before the block is announced.
- Mempool: a transaction pays at least one base unit per byte (the wallet's fee covers transactions of up to 10,000
  bytes); a full pool keeps the transactions of the highest fee rates, and templates take transactions by fee rate.
  Removing the transactions of a connected block takes time proportional to its inputs.

## [0.1.0] - 2026-11-05

First release: node, wallet, CPU miner and CUDA miner, protocol 2, test network with genesis
`7c43ea053c733620043a1aa67d54e66eb41d4ac0c6d5a53f3a29176ace454cd2` and two public relays. The coins of the genesis miner
output can never be spent, and a node gives work to miners only when it is started with `--miner-lock`. The wallet file is
encrypted with a password (Argon2id and XChaCha20-Poly1305).

[0.2.0]: https://github.com/SilasEigen/neb/releases/tag/v0.2.0
[0.1.1]: https://github.com/SilasEigen/neb/releases/tag/v0.1.1
[0.1.0]: https://github.com/SilasEigen/neb/releases/tag/v0.1.0
