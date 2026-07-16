# Multi-stage build. Compiles on Linux in the host's cloud builder, so the local
# Windows toolchain is irrelevant to deploys. Produces a tiny runtime image.

FROM rust:1-slim-bookworm AS builder
WORKDIR /app

# Build only what the server needs (deterministic engine + axum). The optional
# `model` feature (reqwest/moka) stays off until the model layer ships.
COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates
COPY demos ./demos
RUN cargo build --release -p judix-server

FROM debian:bookworm-slim
# ca-certificates so outbound HTTPS to the model API (Groq) works.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/judix-server /usr/local/bin/judix-server
# Hosts inject $PORT (Render etc.); HF Spaces routes to app_port 7860. The server
# reads $PORT at runtime, so an injected PORT overrides this default.
ENV PORT=7860
EXPOSE 7860
CMD ["judix-server"]
