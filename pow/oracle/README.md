# Python oracle

`shares_oracle.py` is a second implementation of the training shares in Python, written from `spec/shares.md` only. It shares no code with the C reference (`pow/ref`) or the CUDA kernel (`pow/gpu`), so agreement between them checks both the specification and the implementations.

| File | Content |
|---|---|
| `shares_oracle.py` | The share, the model step and the state hashes of the specification, the self test and the vector replay. |
| `primitives.py` | ChaCha8 and BLAKE3 (one chunk, and the chunk tree for any length). |
| `blake3_full.py` | A second BLAKE3 for any length; `--selftest` uses it to cross-check `primitives.py`. |

## Usage

Python 3 and numpy are required. From the repository root:

```
python3 -I pow/oracle/shares_oracle.py --selftest
python3 -I pow/oracle/shares_oracle.py --check vectors/shares.txt
python3 -I pow/oracle/shares_oracle.py --check vectors/shares_s1.txt --su1 9 --su2 6
```

`--check FILE` replays a vector file and compares every field. `--su1` and `--su2` set the update shifts (defaults 15 and 12) for the test-only variant of the vectors (`spec/shares.md` section 10.1). The exit code is 0 when everything matches, 1 on a mismatch or a malformed file, and 2 on a usage error.

`pow/tests/check_share_vectors.sh` regenerates both vector files with the C reference, compares them byte for byte, and replays them with this oracle.
