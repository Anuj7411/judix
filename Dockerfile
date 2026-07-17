# Multi-stage build. Compiles on Linux in the host's cloud builder, so the local
# Windows toolchain is irrelevant to deploys. Produces a tiny runtime image.

FROM rust:1-slim-bookworm AS builder
WORKDIR /app

COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates
COPY demos ./demos
COPY web ./web

# `-j 2` caps parallel codegen. Free-tier builders are memory-constrained, and cargo
# defaults to one job per core — enough concurrent rustc processes to get the build
# OOM-killed (exit 137) on a machine with more cores than RAM. Slower, but it finishes.
# `--locked` builds exactly Cargo.lock: no surprise version drift between local and deploy.
RUN cargo build --release --locked -p judix-server -j 2

# rustls means no OpenSSL: the runtime needs only ca-certificates for root trust.
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/judix-server /usr/local/bin/judix-server
# Hosts inject $PORT (Render sets it); the server reads $PORT at runtime, so the
# injected value overrides this default. 8000 is only used for local runs.
ENV PORT=8000
EXPOSE 8000
CMD ["judix-server"]
