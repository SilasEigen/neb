# syntax=docker/dockerfile:1.7
#
# Container images of the NEB node and of the CPU miner.
#
#   docker build --target node -t neb-node .
#   docker build --target miner-cpu -t neb-miner-cpu .
#
# The GPU miner is in Dockerfile.cuda. Usage and configuration: docs/docker.md.

ARG RUST_IMAGE=rust:1.96-bookworm@sha256:a339861ae23e9abb272cea45dfafde21760d2ce6577a70f8a926153677902663
ARG RUNTIME_IMAGE=debian:bookworm-slim@sha256:99c0b25c46d0214d5a2091092e7908907dcaecd6a962cde0ac700acee222b6f3

# Build: the release profile of the workspace, built as the release archives are (tools/release-build.sh).
FROM ${RUST_IMAGE} AS build
WORKDIR /src
COPY tools/release-build.sh tools/release-build.sh
COPY vectors vectors
COPY pow pow
COPY node node
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/node/target \
    bash tools/release-build.sh \
 && install -d /out \
 && cp node/target/release/neb-node node/target/release/neb-wallet node/target/release/neb-miner-cpu /out/

# Runtime base: no toolchain, an unprivileged user and the data directory.
FROM ${RUNTIME_IMAGE} AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 neb \
 && useradd --system --uid 10001 --gid neb --home-dir /data --shell /usr/sbin/nologin neb \
 && install -d -o neb -g neb /data
LABEL org.opencontainers.image.source="https://github.com/SilasEigen/neb" \
      org.opencontainers.image.licenses="Apache-2.0"
USER neb:neb
WORKDIR /data

# Node and wallet. The node listens for peers on 3440; the work service (3441) and the JSON-RPC (8432) stay on the
# loopback interface of the container, so a miner reaches them from the same network namespace.
FROM runtime AS node
LABEL org.opencontainers.image.title="neb-node"
COPY --from=build /out/neb-node /out/neb-wallet /usr/local/bin/
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/docker-entrypoint.sh
ENV NEB_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 3440
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD ["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/8432"]
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["run"]

# CPU miner. It reads the dataset from /data and talks to the work service of a node.
FROM runtime AS miner-cpu
LABEL org.opencontainers.image.title="neb-miner-cpu"
COPY --from=build /out/neb-miner-cpu /usr/local/bin/
ENTRYPOINT ["neb-miner-cpu"]
CMD ["--node", "127.0.0.1:3441", "--dataset", "/data/neb-datasets.bin", "--plain"]
