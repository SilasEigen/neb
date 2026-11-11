# NEB node and chain: normative specification

This document defines the blocks, transactions, consensus rules, work service and peer-to-peer protocol of the NEB chain. The proof of work is a training-share attempt, defined in `spec/shares.md` (sections 5 and 11). The Rust node in `node/` implements this text. The coin name and ticker are NEB; the address prefix is `neb`.

## 1. Scope and principles

The node is an independent implementation, not a fork of another node. BLAKE3 and Ed25519 come from libraries; what a block, a transaction or a valid chain is, is defined in this document, never by a library.

1. The money layer is plain: UTXO accounting, Ed25519 signatures, BLAKE3 hashing. There is no virtual machine, no contract, no staking and no privacy feature.
2. The mining layer is where the chain differs: a work service built into the node (section 7), a model that every block trains (`spec/shares.md` section 11), validation ordered by cost with a per-peer budget of ticket evaluations (section 8), and launch rules enforced by consensus (section 6.8).
3. Every object has one canonical encoding. A decoder rejects every non-canonical form, so an object has exactly one valid byte string.
4. Validation is ordered from cheap to expensive. The proof of work needs the model of the parent, which depends on the bodies of the ancestors, so it is checked when the block is connected, after the cheap header checks and before the transactions (sections 6.1 and 8.1).

## 2. Constants

| name | value | section |
|---|---|---|
| header size | 80 bytes | 4.1 |
| base units per coin | 10^8 | 3 |
| target spacing | 60 s | 6.2 |
| ASERT half-life | 600 s (10 blocks) | 6.2 |
| median time window | 11 blocks | 6.1 |
| future time limit | 120 s | 6.1 |
| initial reward | 20 coins | 6.3 |
| halving interval | 450 000 blocks | 6.3 |
| project share | 5 % of the reward | 6.3 |
| coinbase maturity | 100 blocks | 5.3 |
| maximum reorganisation | 60 blocks | 6.5 |
| block size | at most 1 000 000 bytes | 5.4 |
| transaction size | at most 100 000 bytes | 5.4 |
| inputs, outputs per transaction | at most 255 each | 5.4 |
| multisig keys | at most 15 | 5.2 |
| share records per block (40 bytes each) | at most 256 | 4.2 |
| share target | block target times 2^5 | `spec/shares.md` 11.2 |
| model states kept | 100 levels | `spec/shares.md` 11.4 |

## 3. Primitives and encoding

- Hash: BLAKE3 with a 32-byte output. `H("tag", a, b, ...)` is BLAKE3 over the ASCII bytes of the tag followed by the listed parts, with no separator. The tags are `txid`, `sighash`, `multisig`, `leaf`, `node` and `body`. The block id, the lock of a single key, the inner hash of a Merkle leaf and the message checksum are plain BLAKE3.
- Signature: Ed25519 (RFC 8032) over a 32-byte message. A 64-byte signature `R || S` is valid for a public key `A` when all of these hold: `S` is below the group order `l`; `A` decodes to a point of the curve; and the compressed encoding of the point `[S]B - [k]A`, with `k = SHA-512(R || A || message) mod l`, equals the 32 bytes `R`. This is the cofactorless equation. No other condition is applied: points of small order are not rejected.
- Integers are little-endian. A variable-length integer (varint) is one byte for values below 0xFD, and otherwise the prefix 0xFD with a u16, 0xFE with a u32 or 0xFF with a u64; only the shortest form is valid.
- Amounts are `u64` in base units of 10^-8 coin. Sums of amounts are computed in 128 bits and cannot overflow.
- Every read is bounded by the remaining input, and a decoding must consume its input exactly.

## 4. Block

### 4.1 Header (80 bytes, fixed by the proof of work)

The proof of work takes exactly these 80 bytes, with the nonce in bytes 76 to 79, and uses `key = BLAKE3(header)` as the block id (`spec/shares.md` section 11.1).

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | version | u32, must be 1 |
| 4 | 32 | parent | block id of the parent; all zero in the genesis block only |
| 36 | 32 | body_root | commitment to the transactions and the share list (section 4.2) |
| 68 | 4 | time | u32, seconds since the Unix epoch |
| 72 | 4 | bits | target in the compact form below |
| 76 | 4 | nonce | u32 |

Compact target: `bits = (shift << 24) | mantissa`, with `shift` in 0..232 and `mantissa` in 0x800000..0xFFFFFF; target = mantissa << shift. Only the normalised form is valid, so each target has one encoding, and there is no sign bit. A block is valid for its target when `ticket <= target`, both read as 256-bit little-endian integers, where the ticket is the one of `spec/shares.md` section 5.4 evaluated on the model of the parent (section 6.1).

The 32-bit nonce gives 2^32 attempts per header. More work comes from the extranonce in the coinbase (section 5.3), which changes `body_root`.

Block id: `id = BLAKE3(header)`.

### 4.2 Body

`body = count (varint) || tx[0] || ... || tx[count-1] || k (varint) || record[0] || ... || record[k-1]`, with `count` at least 1. The first transaction is the coinbase and no other is. The records are the share list of the block, 40 bytes each (`body_root (32) || LE32(time) || LE32(nonce)`): `k` is at most 256 and the records are strictly increasing as byte strings (`spec/shares.md` section 11.2). A decoder rejects any other form.

`body_root = H("body", LE32(count), merkle_root, shares_root)`, where the tree is built over the leaves `H("leaf", BLAKE3(full transaction bytes, witness included))`, a node is `H("node", left, right)`, and an odd node at a level is promoted unchanged, never duplicated, and `shares_root = BLAKE3(record[0] || ... || record[k-1])` (BLAKE3 of the empty input when `k = 0`). The count and the share list are part of the commitment, so two different transaction lists or share lists cannot give the same root.

## 5. Transactions

### 5.1 Format

```
tx      := version (u8) || n_in (varint) || input* || n_out (varint) || output*
input   := prev_txid (32) || prev_index (u32) || witness_len (varint) || witness
output  := amount (u64) || lock_type (u8) || lock_data (32)
```

`txid = H("txid", serialisation without the witness fields)`; that serialisation leaves out `witness_len` and `witness` of every input. The witnesses are committed by the block (section 4.2) but not by the txid, so a third party cannot change a txid by re-encoding a signature or choosing another subset of multisig signatures. The coinbase is the exception: its witness is committed by its txid as well (`H("txid", full serialisation)`), because it carries no signature and its witness is what makes every coinbase txid unique (section 5.3).

### 5.2 Locks

- `lock_type 1`, single key: `lock_data = BLAKE3(public key)`; the witness is the public key (32) || the signature (64), 96 bytes in all.
- `lock_type 2`, multisig m-of-n with 1 <= m <= n <= 15: `lock_data = H("multisig", m, n, public keys in order)`, with `m` and `n` as one byte each. The witness is `m (u8) || n (u8) || n public keys (32 each) || m signature entries`, where an entry is `key index (u8) || signature (64)`. The parameters and keys must reproduce `lock_data`, the key indexes must be below `n` and strictly increasing, and each signature must be valid for the key it names. Used for the project share output.
- Any other lock type is invalid.

Signature message: `sighash = H("sighash", chain_id, txid, input_index, amount, lock_type, lock_data)`, where `input_index` is a u32, `amount` is a u64 (both little-endian) and the last three fields are those of the spent output, so a signature is bound to one chain, to one input and to the exact output it spends. `chain_id` is the genesis block id.

Addresses: bech32m with the human-readable prefix `neb`. The data is the 32-byte `lock_data` of a type 1 lock, with no witness-version byte, so an address looks like `neb1...`.

### 5.3 Coinbase

- One input with `prev_txid` all zero and `prev_index` 0xFFFFFFFF; a transaction of this shape is a coinbase. The witness is not interpreted by block validation. The node builds it as `height (u32) || extranonce` (16 bytes by default); the height makes every coinbase txid unique.
- Outputs, each with an amount above zero and a lock of type 1 or 2: the miner output, then the project-share output when the share of section 6.3 is above zero. The project-share output has exactly the amount of the share and the lock fixed by the network parameters (a type 2 lock); when the share is zero the coinbase has one output.
- The sum of the outputs may be less than reward plus fees (the difference is destroyed) but never more.
- A coinbase output created at height `c` can be spent by a transaction in a block of height `c + 100` or later.

### 5.4 Limits

A serialised block (header and body) is at most 1 000 000 bytes. An ordinary transaction has version 1, at most 100 000 bytes, 1 to 255 inputs and 1 to 255 outputs, outputs with an amount above zero and a lock of type 1 or 2, and a total output at most equal to the total input (the difference is the fee).

## 6. Consensus rules

### 6.1 Block validity

A block with parent P at height h = P.height + 1 is valid when all of these hold. The checks are grouped as follows: the header alone first (80 bytes, cheap to receive), then the shape of the body, then the proof of work and the shares, which need the models of the parent and the grandparent, and last the transactions.

Header:

1. The version is 1.
2. `parent` is a known block. A block with an unknown parent is not stored.
3. The models of the parent and of the grandparent are kept (`spec/shares.md` section 11.4). Otherwise the block is refused as an orphan.
4. `bits` equals the value computed from the chain by the difficulty rule (6.2).
5. `time` is greater than the median of the last 11 block times (the parent and its ten predecessors, or all of them when fewer exist; the median of `n` sorted times is the element at index `floor(n / 2)`, counting from 0), and at most 120 s ahead of the node's clock.

Body:

6. The body decodes canonically and `body_root` matches.
7. The proof of work is valid: `ticket(M(P), BLAKE3(header)) <= target`, where `M(P)` is the model of the parent (`spec/shares.md` section 11.1). This is the expensive check: it is one ticket evaluation, and it comes before the share list and the transactions.
8. The share list is valid (`spec/shares.md` section 11.2): at most 256 records, strictly increasing, none at height 1, each one a valid share of the round of P (a header whose parent is the parent of P) on the model of the parent of P. The model of the block is the result of the step (`spec/shares.md` section 11.2).
9. The size is within the limits (5.4), the coinbase rules hold (5.3) and every transaction is valid against the UTXO set, signatures included (6.4).
10. The coinbase pays at most the reward of h plus the fees of the block, with the project share (6.3).

A block that extends the best chain is connected when steps 1 to 10 hold. A block on a side branch passes steps 1 to 8 when it is stored; steps 9 and 10 are checked when its branch replaces the best chain (6.5). The limit on the depth of a reorganisation (6.5) applies at that point and is not a step of this list.

A header is checked by steps 1 to 5 alone before its body is known. The model of every block follows from the genesis block and the bodies, so a node connects blocks in order, and a block whose parent state it does not hold is refused as an orphan (section 8.2).

### 6.2 Difficulty

ASERT (absolutely scheduled exponentially rising targets, the aserti3-2d algorithm) with a target spacing of 60 s and a half-life of 600 s (10 blocks), anchored at the genesis block: the anchor target is the `bits` of the genesis block and the time base is its timestamp. It uses integer arithmetic only, with the exponent, shift and factor steps of the reference form of the algorithm. The genesis block has no parent, so for the block whose parent is at height `ph` with timestamp `tp` the drift is `(tp - genesis_time) - 60 * ph`: the reference formula with `height_diff + 1` replaced by `height_diff`, because the genesis block consumed no schedule slot.
The result clamps to the pow limit and to the floor of the compact target form (`0x00800000`); a target below the floor clamps up. The pow limit is a network parameter: a compact target not below the genesis target. The product of the anchor target and the factor is computed in 320 bits, so the arithmetic is exact for every compact target; below 2^239 it equals the reference arithmetic. `node/validators/asert_reference.py` reduces at 256 bits and gives identical results for every pow limit below 2^239; the vectors of section 10 do not distinguish the two.
Test vectors: section 10.

### 6.3 Reward

- Reward at height h: `(20 * 10^8) >> floor(h / 450000)` base units, that is 20 coins at the start, divided by 2 (an integer shift on the base units) every 450 000 blocks (312.5 days at 60 s per block). The sum of all rewards is 18 million coins less 5 850 000 base units. There is no premine.
- Project share: `floor(reward * 5 / 100)` on the base units, paid in a mandatory second coinbase output to the multisig lock fixed by the network parameters; the miner gets the rest of the reward and all the fees. The share is exactly 1 coin per block through the first era. It is paid at every height at which the formula gives an amount above zero.

### 6.4 Transaction validity

Every input names an output of the UTXO set as it stands before the block, so an output created in a block cannot be spent in the same block; the output is unspent and mature (5.3); each lock is satisfied by its witness; no input is spent twice in the same block; the sum of inputs is at least the sum of outputs. An output that would duplicate an existing outpoint makes the block invalid.

### 6.5 Chain selection and the reorganisation limit

The best chain is chosen by cumulative work, never by height. The work of one block is `floor(2^256 / (target + 1))` at the block's `bits`, the expected number of hash attempts it costs; the cumulative work of a block is the sum of the work of every block from the genesis to it, genesis included. A competing branch replaces the best chain only when its cumulative work is strictly greater and the replacement removes at most 60 blocks (see below); equal work keeps the current tip. Each block tip carries its cumulative work, the index persists it with the connect events (section 9), and a node recomputes it from the block headers when the index is read or rebuilt, so a damaged index cannot change the choice. A block counts toward the work of a branch only after its proof of work and its share list were checked on the models of its ancestors (6.1); the transactions of a side branch are checked when the branch replaces the best chain, and if one of its blocks is invalid the node keeps its best chain and its model. The model of the replacing best chain is the one a replay from the genesis block gives (`spec/shares.md` section 11.4).

The depth limit is applied in `accept_block_from` (`node/crates/neb-node/src/node.rs`) after a side-branch block is checked and stored, when its cumulative work exceeds that of the best chain: the depth is the height of the tip minus the height of the fork point. A branch deeper than 60 blocks stays a side branch, whatever its work, and `ChainState::reorg` (`node/crates/neb-chain/src/state.rs`) enforces the same bound. After a partition longer than 60 blocks, nodes of one side do not switch to the chain of the other.

### 6.6 Genesis

The genesis block is fixed by the network parameters and is not checked by the rules of 6.1, its proof of work included. Its header has version 1, an all-zero parent, the launch time as `time`, the genesis target as `bits` (the anchor of 6.2) and nonce 0. Its only transaction is a coinbase that pays the reward of height 0 as in 6.3.

The coinbase has two outputs: the reward of height 0 less the project share, with a type 1 lock whose lock data is 32 zero bytes, which no key can unlock (no public key is known whose BLAKE3 hash is zero), and the project share, locked to the project-share lock (6.3).

The witness of the genesis coinbase is `height (u32, 0) || extranonce (8 bytes) || miner data (32 bytes)`, 44 bytes. The extranonce is the first 8 bytes of `BLAKE3("neb-testnet-extranonce" || beacon)`. The miner data is `BLAKE3("neb-testnet-beacon-3" || beacon)`. The beacon is the hash of a block of a public proof-of-work chain other than NEB, written as 64 lower-case hexadecimal characters (the ASCII bytes are hashed). The project-share lock is the 1-of-1 multisig lock of the project public key, a 32-byte constant of the network parameters (`25dadba82b1372fd76c1901524cc742224661d6fc5f0ba1b4e95f526a3aa72d3` on the testnet); the matching secret key is not derived from the beacon and is not published. The network magic (8.2) is derived from the genesis id with the tag `neb-testnet-magic`. `neb-node --genesis` prints the beacon, the genesis id, the magic, the genesis miner lock, the project public key and the project lock.

The genesis id, which is the chain id, depends on the beacon. Block 1 has the genesis id as its parent, the proof of work key is `BLAKE3(header)` and the initial model is derived from the genesis id. When the beacon block is mined after the rule is announced, no valid block 1 can exist before it. The external chain is used only as a public, unpredictable timestamp; the node has no other dependency on it. The beacon of the testnet is the hash of the first block of that external chain, which already exists, so it gives no protection against a pre-mined start; the property holds for a network whose beacon is chosen after the announcement.

The values of the network (beacon, genesis time and target, pow limit, project-share lock, dataset hash, ports) are listed in `docs/testnet.md`.

### 6.7 Conservation invariant

The chain state keeps three running sums: the amounts of all unspent outputs `U`, the fees paid by the transactions of all connected blocks `F`, and the amounts of all coinbase outputs `C`. After every connected or disconnected block `U + F = C`; if they differ, connecting or disconnecting fails with an error. A fee leaves the UTXO set when it is paid and re-enters it only when a coinbase claims it. The check runs in the node on every block.

### 6.8 Launch rules

The launch rules are all consensus rules.

| Threat | Rule | Section |
|---|---|---|
| Pre-mining: a party mines blocks before the public launch | The genesis block commits to the hash of a block of an external public chain (a beacon chosen after the announcement) and block 1 names the genesis id as its parent | 6.6 |
| A burst of blocks at launch because the starting difficulty is too low | ASERT is anchored at the genesis block with a half-life of 10 minutes and the target never exceeds the pow limit | 6.2 |
| Preparing blocks in advance with chosen timestamps | A timestamp is above the median of the last 11 blocks and at most 120 s ahead of the node's clock | 6.1 |
| A private chain released later to rewrite history | A node refuses any reorganisation deeper than 60 blocks (one hour), whatever the work of the competing chain | 6.5 |
| Reorganisations while the network is small | Coinbase outputs mature after 100 blocks | 5.3 |

## 7. Mining interface

### 7.1 Work service

The node serves miners on a loopback TCP port. Binary frames, little-endian: `length (u32) || type (u8) || payload`, where `length` counts the type byte and the payload and is between 1 and 2 MiB (2 097 152 bytes). A length outside this range or a type outside 1 to 7 closes the session. A Hello, Share or Submit payload that does not decode is dropped without an answer. The node binds the service to 127.0.0.1 and opens no authentication.

| Type | Name | Direction | Payload |
|---|---|---|---|
| 1 | Hello | miner to node | `version (u16, 1) || name (varint length, UTF-8 bytes)` |
| 2 | Job | node to miner | see below |
| 4 | Submit | miner to node | `job id (u64) || extranonce (varint length, bytes) || time (u32) || nonce (u32)` |
| 5 | Result | node to miner | `accepted (u8, 1 or 0) || code (u8)` |
| 6 | Share | miner to node | `job id (u64) || nonce (u32)` |
| 7 | Model | node to miner | `slot (u8) || weights_hash (32) || packed weights` (50 816 bytes, or 13 000 for KEP1 and 11 140 for RET2, `spec/nebhash-v2.md` section 6); the first byte is the index of the dataset of the job in the network file (`spec/shares.md` sections 11.5 and 11.7) |

A Job carries, in this order: `id (u64)`, `clean (u8, 0 or 1)`, `version (u32)`, `parent (32)`, `time_start (u32)`, `time_window (u32)`, `bits (u32)`, `target (32)`, `share_target (32)`, `coinbase_prefix (varint length, bytes)`, `coinbase_suffix (varint length, bytes)`, `extranonce_size (u8, 1 to 32)`, `branch (varint count of at most 4096, then 32 bytes each)`, `tx_count (u32, at least 1)`, `shares_root (32)`, `height (u32)`, `expiry (u32)`.

- The coinbase is `coinbase_prefix || extranonce || coinbase_suffix`, with `extranonce_size` bytes of extranonce. Its leaf is `H("leaf", BLAKE3(coinbase))`; the miner folds the leaf with the branch, `h = H("node", h, s)` for each entry `s` in order, and `body_root = H("body", LE32(tx_count), h, shares_root)`. The header is `version || parent || body_root || time || bits || nonce`.
- `target` is the target of `bits` and `share_target` is the share target of `bits` (`spec/shares.md` section 11.2). The ticket of an attempt is evaluated on the model of the last Model frame and the dataset it names; the node sends that frame before every Job (`spec/shares.md` section 11.5). `time_start` is the later of the node's clock and the parent time plus one, and, for a Job on the same parent as the newest Job, of that Job's `time_start` plus one, the last bound kept at most 60 s ahead of the clock, `time_window` is 3600 and `expiry` is the creation time of the job plus 300 s. A miner rolls the nonce, then the extranonce, then the time in `[time_start, time_start + time_window]`, and stops using the job at `expiry`, the Unix time after which the job is stale. A share is bound to the parent and not to the header (`spec/shares.md` section 11.2), so a miner that changes the extranonce or the time loses no share; only a nonce tried on the reference header of the job (`spec/shares.md` section 11.5) can be reported with a Share frame. The node sets `clean` to 1 in every Job.
- The node sends a Model frame and a Job, in one write, when a miner connects (without waiting for a Hello), to every miner when the best block changes (whatever the origin of the block) and again when no job went to the miners for 30 s (local policy: the new Job has a later `time_start`, so a miner that tried all 2^32 nonces of a Job gets new ones), and to a miner whose submission was rejected. It keeps the 64 newest jobs; a Submit or Share for an older job is unknown. The Job pays the reward and the fees, less the project share, to the lock chosen by the operator; its transactions are taken from the mempool, highest fee first, within 989 248 bytes (the block limit less 10 752). The coinbase always has the two outputs of 5.3. The operator chooses the lock with `--miner-lock`; a node without it sends no job and closes every miner connection.
- A Share frame reports a nonce of the reference header of a job whose ticket meets the share target; the node makes the share record of it (`spec/shares.md` section 11.2), checks it (one ticket evaluation, not counted in the budget of 8.1), keeps it for the share list of a later block, relays it to its peers with `shares` (8.2) and does not answer (`spec/shares.md` section 11.5).
- A solo miner needs nothing else: the node rebuilds the block from a Submit, runs the checks of 6.1 and the body checks, connects it and relays it. Hello is informational.

Result codes:

| Code | Meaning |
|---|---|
| 0 | accepted |
| 1 | bad submission: unknown job, extranonce length different from the job, time outside the window, or a coinbase that does not decode |
| 2 | bad header: a header check of 6.1 failed, or the proof of work or a share of the block is invalid |
| 3 | bad state: the block was not connected (it does not extend or replace the best chain, or its body is invalid) |

### 7.2 JSON-RPC

The node serves JSON-RPC over HTTP POST on a loopback port. The request is a JSON object with `method` and `params`; a body nested deeper than 64 arrays or objects is a parse error. Methods:

| Method | Parameters | Result |
|---|---|---|
| `getblockcount` | none | height of the best chain (the genesis block has height 0) |
| `getbestblockhash` | none | block id of the tip, 64 hexadecimal characters |
| `getblockhash` | height (integer, 0 to the tip) | block id at that height |
| `getblock` | block id (64 hexadecimal characters) | the block of 4.1 and 4.2 as hexadecimal; any known block, side branches included |
| `getrawmempool` | none | array of txids |
| `getpeerinfo` | none | array of objects: `id`, `addr`, `inbound`, `version`, `subver`, `height`, `handshake`, `connected` (seconds) |
| `getinfo` | none | object: `version`, `protocolversion`, `network`, `testnet`, `blocks`, `bestblockhash`, `chainid`, `connections`, `miners`, `bits`, `weightshash` (the hash of the state of the tip, `spec/shares.md` section 11.2), `slots`, `workport`, `p2pport` |
| `getmodel` | none, a slot number, `"genesis"`, or a slot number and `"genesis"` | object: `height`, `blockhash`, `slot`, `slots`, `dataset` (the name of the slot's dataset), `weightshash` and `weights`, the packed weights of the model of that slot (default 0) in the state of the tip (or in the initial state) as hex, 50,816 bytes, or 13,000 for KEP1 and 11,140 for RET2 (`spec/shares.md` section 7, `spec/nebhash-v2.md` section 6); with `"genesis"` the model is the genesis model of the slot, before any upgrade; residuals are not included |
| `sendrawtransaction` | transaction as hexadecimal | txid; validated as in 6.4 and announced to the peers |

Usage is in `docs/testnet.md`.

## 8. Peer-to-peer network and validation cost

### 8.1 Validation cost

Rejecting a bad block costs the receiver ticket evaluations, while producing it costs the sender nothing. One evaluation (`spec/shares.md` section 5) costs about 0.5 to 1 ms of one CPU core, depending on the CPU. A block needs one evaluation for its proof and one for each share of its list, so 33 for a block with 32 shares and at most 257. The proof of work and the shares need the models of the parent and the grandparent, so the header alone cannot be checked (6.1). So:

1. The cheap checks come first (6.1 header steps 1, 2, 4 and 5, then the body shape of step 6 and the share list shape): version, known parent, expected `bits`, time window, canonical body and list. An attacker must then also name a known parent, guess the right `bits` and a plausible time. The ticket evaluations come last, in the order of 6.1: the proof of the block, then the shares.
2. Each peer session has a budget of ticket evaluations per second (200 by default, `--pow-budget`), counted as a token bucket that holds one second of tokens. A block from a peer takes one token for its proof and one for each share, only when it reaches the proof check; a `shares` message takes one token for each record that is not yet in the share store; a session without a token waits.
3. A peer that sends a block that passes the cheap checks and has an invalid proof of work or an invalid share is disconnected, and its IP address is banned for 24 hours (the ban list is kept in memory). A block whose share list is malformed (more than 256 records, or records that are not strictly increasing, or a share at height 1) is rejected as an invalid proof in the same way. A block message whose list does not decode closes the session. An honest node never does that: a stale or competing block still has a valid proof and valid shares. Orphan and duplicate blocks are not punished. Blocks from the local work service are neither limited nor banned.
4. A side-branch block is checked like any other, on the models of its ancestors, when it arrives; the node keeps the models of the side-branch blocks it has checked for the last 100 levels (`spec/shares.md` section 11.4), so a chain of competing blocks costs the sender real work for each block. At most 64 side-branch models are kept (the lowest blocks go first).

### 8.2 Messages

Framing: `magic (4) || command (12, ASCII, zero padded) || length (u32) || checksum (4) || payload`. The magic is the first four bytes of `BLAKE3("neb-testnet-magic" || chain_id)`. The checksum is the first four bytes of `BLAKE3(payload)`. The maximum payload is 2 MiB. A message with a wrong magic, a wrong checksum or too large a payload closes the session, and so does a message that does not decode.

| Command | Payload |
|---|---|
| `version` | `protocol (u32, 3) || height (u32) || nonce (u64) || genesis id (32) || port (u16) || user agent (varint length of at most 256, UTF-8 bytes)` |
| `verack` | empty |
| `ping`, `pong` | 8 bytes; `pong` repeats the payload of the `ping` |
| `getaddr` | empty; answered with `addr` |
| `addr` | `count (varint) || entries` (section 11) |
| `getheaders`, `getblocks` | `count (varint, at most 64) || block ids (32 each)`, the locator: ids of the best chain of the sender, newest first (the last 10 heights one by one, then gaps of 2, 4, 8, ... blocks, then the genesis block) |
| `headers` | `count (varint, at most 2000) || headers (80 bytes each)`; a node answers `getheaders` and ignores a `headers` message |
| `inv`, `getdata`, `notfound` | `count (varint, at most 5000) || entries`, an entry being `kind (u8, 1 block or 2 transaction) || id (32)` |
| `block` | the 80-byte header followed by the body of 4.2 |
| `shares` | `parent id (32) || count (varint, at most 256) || records (40 bytes each, strictly increasing)`: share records whose share header has this parent (`spec/shares.md` section 11.2) |
| `tx` | a transaction as in 5.1 |

- A session that receives nothing for 60 s sends a `ping` with an 8-byte payload of zeros. A session that then receives no message of any kind during the next 60 s is closed. A read that times out after the first byte of a message header or inside a payload also closes the session, without a ban.
- Handshake: each side sends `version` when the connection opens. A node closes the session when the genesis id or the protocol differs from its own or when the nonce is its own. It answers a valid `version` with `verack`. After `verack` it sends `getblocks`. Before the handshake only `version`, `verack`, `ping` and `pong` are processed.
- Sequential synchronisation: the proof of a block needs the models built by the bodies of its ancestors, so a node does not fetch headers ahead of bodies. `getblocks` is answered with an `inv` of up to 500 block ids that follow the first locator id found on the best chain (from the genesis block when none is found). The node asks with `getdata` for the first 16 blocks of the `inv` that it does not have, connects them in order, checking each one as in 6.1 and 8.1, and sends `getblocks` again with the updated locator when the last of them was checked. `getheaders` is answered with up to 2000 headers that follow the first locator id found on the best chain, for tools that read the chain.
- A block whose parent is unknown, or whose parent model the node does not hold, is refused as an orphan; the node keeps no orphan pool, and answers an unsolicited orphan with `getblocks`.
- Shares: a node that keeps a new valid share record (found by its miners or received) sends it to its peers with `shares`, once per record and per peer, and only for a parent that is the tip or the parent of the tip (`spec/shares.md` section 11.5). A `shares` message for another parent is ignored. A record whose ticket is above the share target is invalid: the session is closed and the IP address banned for 24 hours as for an invalid block (8.1). A record already in the store costs no evaluation and is not relayed.
- Relay: a best block that changed and a transaction that arrived are announced to the other peers with `inv`. A transaction received with `tx` is validated as in 6.4 against the confirmed UTXO set, so a transaction that spends an unconfirmed output is refused, and a transaction that spends an output already spent by a pooled transaction is refused.
- Unknown commands are ignored. `addr` messages are ignored.

### 8.3 Limits

| Item | Limit |
|---|---|
| peers | 8, inbound and outbound together |
| message payload | 2 MiB |
| headers in one message | 2000 |
| entries in `inv`, `getdata`, `notfound` | 5000 |
| ids in an answer to `getblocks` | 500 |
| blocks asked for with one `getdata` while synchronising | 16 |
| side-branch models kept | 64 |
| locator | 64 ids |
| user agent | 256 bytes |
| mempool | 300 MiB of serialised transactions |

### 8.4 Peers

Peers are chosen by the operator: the node dials the peers it is given, as `HOST:PORT` where a name is resolved at every dial, and accepts inbound connections up to the limit (8 by default, `--max-peers`). Without any peer it dials the default seeds of the network, a local policy. It learns no address from the network.

## 9. Storage

Consensus does not depend on how a node stores its data. The node keeps every valid block it has accepted, side branches included, in an append-only log of checksummed records, and the connect and disconnect events of the best chain in an append-only index. The index carries the cumulative work of each connected tip (6.5). On start the node replays the best chain to rebuild the UTXO set and the models (`spec/shares.md` section 11.4), checking each block as in 6.1; if the index is missing or damaged it is rebuilt from the log, choosing the tip with the most cumulative work among the blocks whose ancestry reaches the genesis block. A log that holds a block without such an ancestry is refused when the node opens it. A torn record at the end of a file is discarded. A reorganisation never needs undo data for more than 60 blocks (6.5).

## 10. Test vectors

The share step, from which the ticket of the proof of work comes, has the vector files of `spec/shares.md` section 10. The difficulty rule has two files in `node/vectors/`, pinned by SHA-256 (the files have LF line endings):

```
d87476698100451db7519b3c11c5a4e7d256c7ba5756239bf71dec5a47288555  node/vectors/asert_reference_runs.txt
c17efcbddc34ccea2182bc69e1f43b19f237a824c9aa083fb2cc7df87b4d020a  node/vectors/neb_asert_s0.txt
```

`asert_reference_runs.txt` holds the eight deterministic reference runs of the algorithm; `neb_asert_s0.txt` holds the seven runs with the NEB parameters. Two independent implementations agree on both files: `node/validators/asert_reference.py` (Python), which generates them, and the `neb-difficulty` crate (Rust), whose tests replay them.

## 11. Reserved

- Header `version` other than 1, ordinary transaction `version` other than 1, and lock types other than 1 and 2: a block or transaction that uses one is invalid.
- Work service frame type 3, with a 32-byte target as payload: a node ignores it.
- The entries of `addr`: a node sends `addr` with a count of 0 and ignores the entries it receives.
