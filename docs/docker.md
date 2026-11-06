# Docker

Container images of the NEB node, the CPU miner and the GPU miner, and a Compose file that runs a node with a miner on one host. The images are built from this repository; they hold no key and no wallet.

| Image | Build | Contents |
|---|---|---|
| node | `docker build --target node -t neb-node .` | `neb-node` and `neb-wallet` |
| miner-cpu | `docker build --target miner-cpu -t neb-miner-cpu .` | `neb-miner-cpu` |
| miner-gpu | `docker build -f Dockerfile.cuda -t neb-miner-gpu .` | `neb-miner-gpu` (NVIDIA, compute capability 7.5 or later) |

The base images are pinned by digest, and Dependabot proposes updates. The processes run as the unprivileged user `neb` (uid 10001).

## Quick start

You need Docker with Compose v2 and an address that receives the rewards. Create one with `neb-wallet new` (from a release archive, or `docker run --rm -it -v "$PWD":/w -w /w neb-node:local neb-wallet --wallet wallet.txt new` once the image is built) and keep `wallet.txt` outside the containers.

```bash
export NEB_MINER_ADDRESS=neb1...          # the address printed by neb-wallet (or put it in a .env file next to compose.yaml)
docker compose up -d                      # a node and a CPU miner
docker compose --profile gpu up -d        # also the GPU miner
docker compose logs -f node miner
```

On first start the node downloads the dataset file (47 MB) from the release page, checks it against the hash pinned in the network parameters, and replays its chain. The node is healthy when its JSON-RPC answers; the miners start after that.

The work service of the node (port 3441) listens on the loopback interface of its container, as it does on a host. The miners therefore share the network namespace of the node (`network_mode: "service:node"` in `compose.yaml`) and reach it at `127.0.0.1:3441`. Only the peer port 3440 is published.

## Configuration of the node image

| Variable | Meaning |
|---|---|
| `NEB_MINER_ADDRESS` | `neb1...` address that receives the rewards of the jobs of the node. `NEB_MINER_LOCK` (64 hex characters, as printed by `neb-wallet lock`) does the same. Without either, the node refuses miners; a relay needs neither. |
| `NEB_SEEDS` | Comma-separated `HOST:PORT` peers that the node keeps dialled. |
| `NEB_MAX_PEERS` | Peer sessions, inbound and outbound together (1 to 1024). |
| `NEB_LISTEN` | Peer bind address, `0.0.0.0` by default. |
| `NEB_DATASET` | Path of the dataset file, `/data/neb-datasets.bin` by default. |
| `NEB_DATASET_URL` | Where to download the dataset file when it is missing. The node refuses a file with another hash. |
| `NEB_EXTRA_ARGS` | More arguments for `neb-node`, split on spaces. |

Any command other than the default `run` is executed as given:

```bash
docker run --rm neb-node:local neb-node --genesis
docker run --rm neb-node:local neb-node --chain-selftest
```

The miner images take the options of the miners as arguments. The default is `--node 127.0.0.1:3441 --dataset /data/neb-datasets.bin --plain`; add `--engine avx512` to the CPU miner or `--gpu-kernel k2` to the GPU miner for the optional engines (`docs/testnet.md`).

## Data

The volume `neb-data` holds the dataset file (`/data/neb-datasets.bin`) and the chain (`/data/chain`). The miners mount it read-only. The chain is rebuilt by replaying the blocks if the volume is lost.

## Without Compose

```bash
docker volume create neb-data
docker run -d --name neb-node --init -p 3440:3440 -v neb-data:/data \
  -e NEB_MINER_ADDRESS=neb1... -e NEB_DATASET_URL=https://github.com/SilasEigen/neb/releases/download/v0.2.0/neb-datasets.bin \
  neb-node:local
docker run -d --name neb-miner --init --network container:neb-node -v neb-data:/data:ro neb-miner-cpu:local
```

## Relay

A relay is a node that accepts many inbound peers (`docs/testnet.md`, "Running a public relay"). With the node image:

```bash
docker run -d --name neb-relay --init -p 3440:3440 -v neb-data:/data \
  -e NEB_MAX_PEERS=256 -e NEB_SEEDS=other-relay.example.org:3440 -e NEB_DATASET_URL=... neb-node:local
```

Open TCP 3440 inbound. Keep the work service and the JSON-RPC unpublished.

## GPU

The GPU image needs an NVIDIA driver of the 580 series or later on the host and the NVIDIA Container Toolkit (`docker run --gpus all`, or the `gpu` profile of `compose.yaml`). The CUDA runtime is linked into `neb-miner-gpu`, so the image carries no CUDA toolkit.

```bash
docker run --rm --gpus all neb-miner-gpu:local --selftest
```

## Hardening

The services in `compose.yaml` run with a read-only root file system, no Linux capability, `no-new-privileges` and an init process. The node image has a health check on the JSON-RPC port. The images contain no secret: the miner address is public, and the wallet stays outside.
