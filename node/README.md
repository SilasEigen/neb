# NEB node

Rust implementation of the NEB node, wallet and miners. The normative text is `spec/node.md`; the proof of work is a training-share attempt, specified in `spec/shares.md` (sections 5 and 11). The workspace version is 0.1.0.

| Crate | Content |
|---|---|
| `neb-codec` | Canonical encoding: little-endian integers, shortest-form variable-length integers, bounded reads (spec/node.md section 3). |
| `neb-block` | Block header and compact target (section 4.1). |
| `neb-difficulty` | ASERT difficulty rule, aserti3-2d (section 6.2). |
| `neb-pow` | The C reference from `pow/ref`, compiled in at build time. Its `shares` module wraps the training-share step (`pow/ref/shares.c`, `spec/shares.md`): the model, the ticket with reusable buffers, the dataset files, and the replay of `vectors/shares.txt` for `--shares-selftest`. |
| `neb-chain` | Blocks with their share lists, transactions, locks and witnesses, the UTXO set, the conservation invariant, the model of each block (`model.rs`) and reorganisations. |
| `neb-work` | Work service: wire frames including `model` and `share` (section 7.1), mempool, block templates, jobs, reported shares and submission rebuilding. |
| `neb-store` | Append-only block store: records with a magic, a length and a BLAKE3 checksum, fsync per record, torn-tail truncation, and the connect/disconnect index. |
| `neb-json` | JSON reader and writer for the RPC and the wallet. |
| `neb-node` | The `neb-node` binary: network parameters of the testnet, peers with sequential block sync, work service, loopback JSON-RPC, daemon. |
| `neb-cpuminer` | The `neb-miner-cpu` binary: share attempts on the model of the chain, over the work service. Its session (frames, current job, counters, the blocks found) and its terminal dashboard are shared with the GPU miner. |
| `neb-gpuminer` | The `neb-miner-gpu` binary: the same attempts on CUDA devices (`pow/gpu`), checked on the CPU before they are sent, and the readings of the card for the dashboard. Built with `--features cuda`; without it the crate has no binary. |
| `neb-metalminer` | The `neb-miner-metal` binary: the same attempts on an Apple GPU with Metal (`crates/neb-metalminer/src/neb_share.metal`), checked with the reference before they are sent. macOS on Apple silicon only; on other platforms the crate builds and its entry points say why they cannot run. |
| `neb-wallet` | The `neb-wallet` binary: keys, bech32m `neb1` addresses, coin selection and Ed25519 signing through the JSON-RPC of the node. |

## Build and test

From this directory:

```
cargo build --release --locked
cargo test --locked
```

A C compiler is required: `neb-pow` compiles `pow/ref`, which is the only copy of the proof-of-work and training-share code. The binaries are in `target/release`. `cargo build --release --locked -p neb-gpuminer --features cuda` also builds `neb-miner-gpu` (CUDA toolkit required, see `docs/testnet.md`). `cargo build --release --locked -p neb-metalminer` builds `neb-miner-metal` on macOS; no Xcode is needed, because the Metal kernel is compiled by the device at start.

The built-in checks run without a network: `neb-node --selftest`, `--shares-selftest`, `--chain-selftest` and `--work-selftest`.

`bash tools/release-build.sh` from the repository root builds the release binaries with source paths mapped to fixed prefixes. On Linux, `neb-node`, `neb-wallet` and `neb-miner-cpu` are byte-identical between two builds of the same commit; the Windows binaries and `neb-miner-gpu` are not yet.

## Run

`neb-node --run --dataset FILE` starts the node. It derives the genesis block from the public beacon string of the network (`docs/testnet.md`), stores blocks in an append-only log (directory `neb-testnet-data` by default, `--datadir` to change it), serves peers on port 3440, the work service on 3441 and JSON-RPC on loopback port 8432. `FILE` is the dataset file made by `tools/make_mnist_dataset.py`; the node refuses a file with another hash. A miner connects to the work service, for example `neb-miner-cpu --node 127.0.0.1:3441 --dataset FILE`. Options, ports and wallet commands are in `docs/testnet.md`.

## Validators

`validators/asert_reference.py` is an independent Python implementation of the ASERT rule. It generates the vector files in `vectors/`, which the `neb-difficulty` tests replay.
