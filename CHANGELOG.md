# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The protocol version and the genesis of the network are listed separately from
the software version: a node checks them, not the software version.

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
