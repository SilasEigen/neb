# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The protocol version and the genesis of the network are listed separately from
the software version: a node checks them, not the software version.

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

[0.1.1]: https://github.com/SilasEigen/neb/releases/tag/v0.1.1
[0.1.0]: https://github.com/SilasEigen/neb/releases/tag/v0.1.0
