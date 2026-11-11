# NEB pool protocol, version 1

## 1. Scope

This document specifies the protocol between a mining pool and the miners that work for it, and how the pool uses
its own node. It changes nothing in the consensus rules (`spec/node.md`, `spec/shares.md`) and needs no change in the
node: the pool takes its jobs from the work service of `spec/node.md` section 7.1 and hands what its miners find to
the node through that service and the peer-to-peer `shares` message of section 8.2.

The algorithm is nebhash: an attempt is one training step of the chain's model (`spec/shares.md` sections 5 and
11.1). Three thresholds apply to the ticket of an attempt, from the hardest to the easiest:

| Name | Threshold | What the pool does with it |
|---|---|---|
| solution | ticket at most the block target | submits the block to its node |
| chain share | ticket at most the share target (32 times the block target) | sends the share record to its node, which lists it in a later block and relays it; it trains the model |
| pool share | ticket at most the pool target of the miner | counts it for the payout of the miner |

The pool target is never below the share target, so every chain share is also a pool share and reaches the pool.

## 2. Data flow

```
node (miner lock = the pool's lock)
  | Model and Job frames (spec/node.md 7.1)
  v
pool --- mining.set_model, mining.notify, mining.set_target ---> miners
pool <-- mining.submit (extranonce2, time, nonce) -------------- miners
  | solution:    Submit frame to the node (7.1)
  | chain share: shares message to the node (8.2), over a peer session with it
  | pool share:  accounting
```

The block reward goes to the lock the node was started with (`--miner-lock`), the lock of the pool. The pool pays its
miners with ordinary transactions once the coinbase is spendable, 100 blocks later (`spec/node.md` section 5.3). The
project share stays in the coinbase as for a solo miner.

## 3. Transport and encoding

- TCP. Each message is one JSON object on one line, UTF-8, ended by `\n`. A line is at most 262 144 bytes; the
  longest is `mining.set_model`, about 101 700 bytes for the 784-128-10 network.
- Requests carry `id` (a number or a string), `method` and `params` (an array). Responses carry the same `id`,
  `result` and `error` (`null` or `[code, "message"]`). Notifications are requests with `id` set to `null`.
- Byte strings are lower-case hexadecimal, in the byte order of the NEB encodings: no byte of a hash, a target or a
  coinbase is reversed. `u32` values (`version`, `time`, `bits`, `nonce`, `height`, ...) are JSON numbers.
- Targets are 32 bytes, read as 256-bit little-endian integers, like the tickets they are compared with.

## 4. Messages

| Method | Direction | Params | Result |
|---|---|---|---|
| `mining.subscribe` | miner to pool | `[agent, "neb-pool/1"]` | `{"session", "extranonce1", "extranonce2_size", "chain_id", "dataset_hash"}` |
| `mining.authorize` | miner to pool | `[login, password]` | `true` |
| `mining.set_target` | pool to miner | `[target]` | notification |
| `mining.set_model` | pool to miner | `[slot, weights_hash, weights]` | notification |
| `mining.notify` | pool to miner | `[job]` | notification |
| `mining.submit` | miner to pool | `[login, job_id, extranonce2, time, nonce]` | `true` |

- `mining.subscribe`: `agent` names the miner software and its version. The result gives a session id (hex), the
  `extranonce1` of the session (hex, 4 bytes), the length of the `extranonce2` the miner rolls (bytes), the chain id
  (the genesis block id) and the BLAKE3 hash of the dataset file of the network (`spec/shares.md` section 11.6). A
  miner whose dataset file has another hash stops. The pool sends `mining.set_target`, `mining.set_model` and
  `mining.notify` once the miner is authorized.
- `mining.authorize`: `login` is a `neb1` address (`spec/node.md` section 5.2), optionally followed by `.` and a worker
  name of at most 32 characters `[A-Za-z0-9_-]`. The pool pays that address. The password is not used by this
  protocol; a pool may read options from it. A session may authorize several logins: a miner software with a dev fee
  authorizes its own login as well and submits that part of its attempts under it.
- `mining.set_target`: the pool target that applies to the jobs sent after it.
- `mining.set_model`: the weights of a model, as in the Model frame of `spec/node.md` section 7.1: `slot` (number,
  the index of the dataset of the jobs in the network file, which also names the network to run), `weights_hash`
  (32 bytes) and `weights` (the packed weights: 50 816 bytes for the 784-128-10 network, 13 000 for KEP1 and 11 140
  for RET2 of `spec/nebhash-v2.md`, the networks of the testnet from block 8800). The miner checks that
  `weights_hash` is the weights hash of `weights` (`spec/shares.md` section 7) and keeps the model under that hash.
  The pool sends it before a `mining.notify` that names a weights hash the session has not received.
- `mining.notify`: `job` is an object with the fields `job_id` (string), `clean` (boolean), `version`, `parent`,
  `time_start`, `time_window`, `bits`, `target`, `share_target`, `coinbase_prefix`, `coinbase_suffix`, `branch` (array
  of 32-byte values), `tx_count`, `shares_root`, `height`, `slot`, `weights_hash` and `expiry`. They are the fields of
  the Job frame of `spec/node.md` section 7.1 except `extranonce_size` (given at subscription), with the slot and the
  weights hash of its model. With `clean` set, the miner drops the older jobs at once.
- `mining.submit`: an attempt whose ticket is at most the pool target: the job, the `extranonce2` (hex, of the
  length given at subscription), the time and the nonce. `login` is one of the logins of the session.

## 5. The header of an attempt

For a job and an attempt (`extranonce2`, `time`, `nonce`) of a session:

```
coinbase  = coinbase_prefix || extranonce1 || extranonce2 || coinbase_suffix
h         = H("leaf", BLAKE3(coinbase)); for each s in branch, in order: h = H("node", h, s)
body_root = H("body", LE32(tx_count), h, shares_root)
header    = LE32(version) || parent || body_root || LE32(time) || LE32(bits) || LE32(nonce)      (80 bytes)
key       = BLAKE3(header)
ticket    = the ticket of spec/shares.md section 5.4 (spec/nebhash-v2.md for KEP1 and RET2), with the key, the
            model named by weights_hash and the dataset that the slot of the job names
```

`H` is the tagged hash of `spec/node.md` section 3. This is the header of `spec/node.md` section 7.1 with the extranonce
`extranonce1 || extranonce2`.

## 6. Miner rules

1. Roll the nonce, then `extranonce2`, then the time within `[time_start, time_start + time_window]`.
2. Evaluate every attempt on the model of the job's `weights_hash` and the dataset that its `slot` names, with the
   network of that dataset. A job whose model the miner does not hold, or whose dataset or network it does not
   know, is not mined.
3. Submit every attempt whose ticket is at most the pool target. A solution and a chain share are pool shares too;
   the miner does not need to tell them apart.
4. Stop using a job at its `expiry` (Unix time) and when a `clean` job arrives.

## 7. Pool rules

1. **Jobs.** The pool is a miner of its own node (`spec/node.md` section 7.1) and makes one job from each Job frame,
   with the slot and weights hash of the Model frame before it. Its `job_id` values are unique in the session, and
   `clean` is true when the parent differs from that of the previous job.
2. **Extranonce.** Each session gets a distinct 4-byte `extranonce1`; `extranonce2_size` is the `extranonce_size` of
   the node's job less 4. A node job with an `extranonce_size` below 8 is not used.
3. **Pool target.** The pool target of a session is never below the share target of the jobs it receives, so no
   chain share is lost. A pool should adjust it so that a miner sends about one pool share every 10 seconds.
4. **Checks of a submit**, in this order: the session is subscribed and the login authorized; the job is known and
   not past its expiry; `extranonce2` has the right length; the time is in the window of the job; the header was not
   submitted before (two jobs on the same parent can give the same header, so duplicates are found on the header, not
   on the job); the ticket, recomputed by the pool, is at most the pool target. A submit that fails gets the error of
   section 8 and counts for nothing.
5. **Solutions.** An attempt whose ticket is at most the block target is sent to the node with a Submit frame (the
   job id of the node's job, the extranonce `extranonce1 || extranonce2`, the time, the nonce).
6. **Chain shares.** For an attempt whose ticket is at most the share target, the pool builds the share record
   `body_root || LE32(time) || LE32(nonce)` (`spec/shares.md` section 11.2) and sends it to its node in a `shares`
   message whose parent is the `parent` of the job, over a peer session it holds with the node (`spec/node.md`
   section 8.2: the `version` handshake with protocol 3 and the genesis id, then `pong` to every `ping`; the pool may
   ignore the other messages, and the session takes one of the node's peer slots). The pool forwards every chain share
   it receives: these records are the training steps of the chain, and a node lists the records of every miner whose
   shares reach it. A share header may carry any coinbase, time and nonce (`spec/shares.md` section 11.2), so a pool
   attempt with any `extranonce2` is a valid share; the Share frame of the work service cannot carry it, since it
   names only a nonce of the reference header. The node checks each record again and bans for 24 hours the address
   of a session that sends an invalid one, so the pool sends only records it checked.
7. **Accounting** of pool shares and the payout scheme are the pool's choice and outside this protocol.

## 8. Errors

| Code | Meaning |
|---|---|
| 20 | other error |
| 21 | job not found or past its expiry |
| 22 | duplicate attempt |
| 23 | ticket above the pool target |
| 24 | login not authorized, or not a valid `neb1` address |
| 25 | session not subscribed |
| 26 | time outside the window of the job |
| 27 | `extranonce2` of the wrong length |

## 9. Costs and limits

- A ticket costs 0.5 to 1 ms of one CPU core (`spec/node.md` section 8.1), and the pool computes one for each
  submit. Its pool target bounds the rate of submits, and a pool should limit each session to a few submits per
  second and close a session that sends tickets above its target.
- A Model frame changes with every block, so a session receives about 100 KB per minute of `mining.set_model`.
- The miner and the pool keep the dataset file of the network (`docs/testnet.md`); the miner checks its hash at
  subscription.
- A miner that withholds its solutions loses the pool's reward and the pool loses blocks, as in other proof-of-work
  pools; the pool sees it as a miner whose pool shares never lead to blocks.

## 10. Example

```
-> {"id":1,"method":"mining.subscribe","params":["exampleminer/1.0","neb-pool/1"]}
<- {"id":1,"result":{"session":"5f0c","extranonce1":"0a1b2c3d","extranonce2_size":12,"chain_id":"8271952aba2ac5340c1dfe884caa1f2c6ee765487fa2117ddcd90710ce87248d","dataset_hash":"296a174edbf67339fa0af341f063154292bba224b3d2cacf5b18b31c70da80eb"},"error":null}
-> {"id":2,"method":"mining.authorize","params":["neb1...addr.rig1",""]}
<- {"id":2,"result":true,"error":null}
<- {"id":null,"method":"mining.set_target","params":["..."]}
<- {"id":null,"method":"mining.set_model","params":[0,"d8d4...","8f03..."]}
<- {"id":null,"method":"mining.notify","params":[{"job_id":"1a","clean":true,"version":1,"parent":"32d6...","time_start":1794150000,"time_window":3600,"bits":3318676480,"target":"...","share_target":"...","coinbase_prefix":"...","coinbase_suffix":"...","branch":[],"tx_count":1,"shares_root":"...","height":2454,"slot":0,"weights_hash":"d8d4...","expiry":1794150300}]}
-> {"id":3,"method":"mining.submit","params":["neb1...addr.rig1","1a","000000000000000000000001",1794150012,30817}
<- {"id":3,"result":true,"error":null}
```

Hex values are shortened with `...` here; on the wire every value has its full length.
