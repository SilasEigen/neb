# Chain fixtures

Recorded chains that the tests and the independent oracle replay block by block (format `neb-chain-fixture 2`,
documented in `node/crates/neb-chain/src/fixture.rs`). A replay checks the proof of work of every block, every share
of every list, the model step, the weights and residual hashes and the losses of the bounds of the step.

A share is a record of 40 bytes, `body_root (32) || LE32(time) || LE32(nonce)`, bound to the parent of the round and not
to the header of the block that wins it (`spec/shares.md` section 11.2). The fixtures with several pools show it: at each
height every pool has its own coinbase, so its own `body_root`, and the list of the next block holds the records of the
headers of all of them, whichever pool wins the block.

| file | blocks | steps | shares | dataset | made by |
|---|---|---|---|---|---|
| `synthetic-stress.txt` | 41 | 40 | 3351 | synthetic (spec/shares.md section 9) | `cargo run --release -p neb-chain --example chain_fixture -- gen-stress OUT` (seed `0x5EED0001`) |
| `pools-synthetic.txt` | 40 | 39 | 1285 | synthetic | `... chain_fixture -- gen-pools OUT pools-synthetic 40 4` (4 pools) |
| `pools-mnist.txt` | 60 | 59 | 2000 | network dataset (MNIST) | `... chain_fixture -- gen-pools OUT pools-mnist 60 3 DATASET_FILE HASH` (3 pools) |

`synthetic-stress` has steps of 1 to 256 shares chosen around the thresholds of the residual bound (16 shares in layer 2,
128 in layer 1); it runs in CI. `pools-synthetic` runs in CI too. `pools-mnist` needs the network dataset file (set
`NEB_DATASET`); it ends on `weights_hash c9817031...` and `residual_hash f28e2498...`.

## Replay

```
cargo run --release -p neb-chain --example chain_fixture -- replay pow/tests/chains/synthetic-stress.txt
cargo run --release -p neb-chain --example chain_fixture -- replay pow/tests/chains/pools-mnist.txt --dataset mnist-q8.bin
python3 -B -I pow/oracle/chain_oracle.py pow/tests/chains/pools-synthetic.txt
python3 -B -I pow/oracle/chain_oracle.py pow/tests/chains/pools-mnist.txt --dataset mnist-q8.bin
NEB_DATASET=mnist-q8.bin NEB_REQUIRE_DATASET=1 cargo test --release -p neb-chain --test chain_fixtures
```

The Rust replay uses the functions of the node (`verify_block`, the C reference). The oracle (`pow/oracle/chain_oracle.py`)
shares no code with them. Both give the same hashes and the same losses on all three chains. A list is in increasing
order of its records as byte strings, not as numbers: `LE32(256)` is `00 01 00 00` and sorts before `LE32(1)`, which is
`01 00 00 00`.

## Stand-in bodies

The chains were mined with a stand-in body: one coinbase `"sim-coinbase" || LE32(height)` (followed by `LE32(pool)` for a
pool above 0), whose `body_root` is `H("body", LE32(1), leaf, shares_root)`. The proof of work and the model step depend
on the body only through the `body_root` field of the header, so the fixtures record headers and lists, not bodies. For
each block a fixture records the 80-byte header, which carries the block id, the pool that won, the share list, the
weights and residual hashes, and the losses of the step (cut residual entries, clipped weight entries, sum of the cut
residuals, sum of the cut weight charge). The losses were computed by a separate implementation of the step and are
recomputed by the replay.

## SHA-256 of the fixtures

| file | SHA-256 |
|---|---|
| `synthetic-stress.txt` | `bd587e6b856fe94a78ed1b0bd284ec822349f7e63b8a87f3143a1d03810ce3fc` |
| `pools-synthetic.txt` | `b135c1c303314b6a57097ad535d1e98d33d6931d52e369831f0718f243b51b5f` |
| `pools-mnist.txt` | `67b579ad2da24c650b95c8d719d6ff2471a931deeca2f9053291f3fec9fedb81` |
