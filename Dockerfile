# Multi-stage build. Compiles on Linux in the host's cloud builder, so the local
# Windows toolchain is irrelevant to deploys. Produces a tiny runtime image.

FROM rust:1-slim-bookworm AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates
COPY demos ./demos
COPY web ./web
RUN cargo build --release -p judix-server

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/judix-server /usr/local/bin/judix-server
# Hosts inject $PORT (Render sets it); the server reads $PORT at runtime, so the
# injected value overrides this default. 8000 is only used for local runs.
ENV PORT=8000
EXPOSE 8000
CMD ["judix-server"]
