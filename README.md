---
title: Judix
emoji: 🦀
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Judix

**Real-time, per-turn evaluation for AI agents & RAG. A deterministic Rust engine scores; an AI model only explains.**

> Observability tells you what your agent *did*. Judix tells you, on every step and in real time, whether it was any *good* — because a **deterministic Rust engine** computes real numeric scores with **zero LLM calls**, and an AI model is used only to explain the "why."

Built solo for the **OpenAI × NamasteDev Codex Hackathon** (Rust-only).

---

## Why this isn't a wrapper

An eval tool that just asks a model "rate this 1–10" *is* a wrapper. Judix is not:

1. **The scorer is a deterministic Rust engine**, not a model. Tool-call correctness (F1), loop/repeat detection, and schema validation are real computation with **no network, no randomness, no clock** — same input, same output, every time. The AI model is a *narrator*, not the judge.
2. **Real-time, per-turn.** Incumbents (LangSmith, Braintrust) run offline on sampled traces; they can't catch the meaning of a turn *while it's happening*. Judix is designed to score every production turn cheaply.
3. **The method is ours.** Claim decomposition + evidence grounding for RAG faithfulness is implemented here; the model only fills in the per-claim verdicts and one-line reasons.

Backed by data: LangChain's *State of Agent Engineering 2026* (1,340 practitioners) — **89% have observability, only 52% run evals**, and online/real-time evals just **37.3%**; quality is the **#1 production barrier**.

---

## What it scores

**Agent Trajectory Scorer** — per step:
- `tool_call_correctness` — deterministic F1 (`0.5·set_f1 + 0.5·bag_f1`) vs. a golden tool list, with JSON-schema arg validation.
- `loop_free` — deterministic sliding-window repeat detection.
- `step_relevance`, `goal_drift` — model-explained (score + confidence + reason).

**RAG Faithfulness Scorer** — per (question, contexts, answer) triple:
- `faithfulness` — model decomposes the answer into atomic claims, verifies each against the contexts; Judix computes `supported/total` deterministically and maps unsupported claims to red-highlighted answer spans.
- `answer_relevancy`, `context_precision`, `context_recall` — model-explained.

Every metric is 0–100. Deterministic metrics also emit `pass` + the raw value; model metrics emit `confidence` + `reason`. Scores roll up into a composite **Step Quality**, a headline **Run Quality**, and a **RAG Quality** — with hard caps (a looping step or an ungrounded answer is forced red) and a derived color band (≥80 green, 50–79 amber, <50 red).

---

## Status

| Phase | State |
|---|---|
| Deterministic engine (F1, loop detection, schema validation, composite scoring + caps) | ✅ built, **13 unit tests passing** |
| `GET /health` server | ✅ |
| Model-explanation layer (`step_relevance`, `goal_drift`, RAG claim verify) | 🚧 in progress |
| SSE live streaming + web playground | 🚧 |
| Deploy (Docker → Koyeb) | 🚧 |

---

## Quickstart

Requires Rust (stable). From the repo root:

```bash
# Run the deterministic engine over a demo trace (no model, no API key, $0):
cargo run -p judix-cli -- demos/wrong_tool.json   # catches a wrong tool call + a loop, instantly
cargo run -p judix-cli -- demos/clean.json        # a clean run scores green

# Run the engine test suite:
cargo test -p judix-core

# Run the server:
cargo run -p judix-server        # then: curl localhost:8000/health
```

### Demos (`demos/`)
- `clean.json` — books a table correctly, avoiding downtown → Run Quality ~100 (green).
- `wrong_tool.json` — searches downtown when told to avoid it, then loops → tool-call FAIL + loop steps red, **100% deterministic, $0**. The money demo.
- `rag_hallucination.json` — context says 14-day returns, answer claims "30 days" → the unsupported span is flagged.

---

## Architecture

Cargo workspace:
- `judix-core` — the deterministic engine (`deterministic.rs`), composite scoring (`scoring.rs`), types (`types.rs`), and (behind the `model` feature) the model client + cache.
- `judix-server` — axum HTTP server (health, scoring routes, SSE), binds `$PORT`.
- `judix-cli` — score a trace from the terminal, deterministic-only.

Deployed as a Docker image (builds on Linux) to any container host. The AI model is any OpenAI-compatible chat API, configured entirely via env (`JUDIX_API_KEY`, `JUDIX_BASE_URL`, `JUDIX_MODEL_FAST`, `JUDIX_MODEL_STRONG`) — the deterministic engine needs no key and runs at $0.

## License

MIT
