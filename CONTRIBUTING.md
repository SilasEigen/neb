# Contributing

Small, focused pull requests are welcome. Participation follows the [code of conduct](CODE_OF_CONDUCT.md).

## Before a pull request

From the repository root:

```bash
(cd node && cargo fmt --all --check && cargo clippy --locked --workspace --all-targets -- -D warnings)
(cd node && cargo test --locked)
bash pow/tests/check_spec_constants.sh     # if you touch the proof of work or the model step
bash pow/tests/check_share_vectors.sh      # C reference and Python oracle against vectors/shares*.txt
```

CI runs these checks, the Python oracle, the recorded chains and the container builds on every push. The scripts need gcc,
Python 3 and numpy.

## Rules

- Everything in the repository is in English.
- One canonical encoding for every object. Write the test first.
- Do not change a constant, a clamp, a shift or a check to make a test pass. The constants live in `spec/shares.md` and `pow/ref/shares.h`, and `pow/tests/check_spec_constants.sh` compares them.
- The vector files are pinned by SHA-256 in `spec/shares.md`. Regenerating them is a consensus-relevant decision.
- Every `unsafe` block carries a `// SAFETY:` comment; clippy enforces it.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org): `type(scope): subject`, in English and the imperative, at most 72 characters. One logical change per commit.

## Further checks

```bash
node/target/release/neb-node --selftest                                            # also --shares-selftest, --chain-selftest, --work-selftest
bash pow/tests/check_avx512.sh                                                     # AVX-512 engine, with two negative controls (skips without AVX-512)
bash pow/tests/check_shares_tc.sh                                                  # CUDA kernels K0, K1, K2 (Windows: pow\tests\check_shares_tc.bat)
python3 -B -I pow/oracle/chain_oracle.py pow/tests/chains/synthetic-stress.txt     # a recorded chain replayed by the independent oracle
(cd node && cargo test --locked --features cuda -p neb-gpuminer)                   # with a CUDA device
```

`REAL=mnist-q8.bin MODELS="model1.bin model2.bin"` (and `NEB_DATASET=mnist-q8.bin` for `cargo test`) add the network
dataset and trained models to these checks. `pow/tests/validate_all.sh` runs every check in turn, and
`pow/bench/run_bench.sh` measures the engines.

## Repository layout

| Path | Contents |
|---|---|
| `spec/` | Normative texts: `shares.md` (proof of work and model step), `node.md` (blocks, transactions, chain rules, network) |
| `docs/` | Whitepaper, network guide, Docker guide, mining engines, experiments |
| `pow/ref` | C reference of the training share, compiled into the node |
| `pow/oracle` | Independent Python implementation written from the specification |
| `pow/cpu`, `pow/gpu` | AVX-512 and CUDA engines of the share step (mining only, off by default) |
| `pow/tests` | Constant and vector checks, vector generator, differential tests, recorded chains |
| `pow/bench` | Throughput benchmark and recorded CPU runs |
| `vectors/` | Share test vectors, pinned by SHA-256 in `spec/shares.md` |
| `node/crates` | Rust workspace: codec, header, ASERT, proof check and model, chain state, work service, store, JSON, wallet, miners, engines, node |
| `node/validators`, `node/vectors` | Independent Python implementation of the ASERT rule and its vectors |
| `Dockerfile`, `Dockerfile.cuda`, `compose.yaml`, `docker/` | Container images and Compose file |
| `tools/` | Dataset tool, training simulations, release and HiveOS packaging scripts |

## Releases

`tools/release-build.sh` builds the release binaries with source paths mapped to fixed prefixes; on Linux, `neb-node`,
`neb-wallet` and `neb-miner-cpu` are byte-identical between two builds of the same commit. Add an entry to `CHANGELOG.md`
for every release.
