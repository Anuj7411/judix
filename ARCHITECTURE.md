# Judix: Architecture & Design Brief

> A complete specification of the Judix system and the design problem for its web playground.
> Everything in this document is verified against the source at `C:\Projects\judix` and against
> live responses from `https://judix-8piu.onrender.com`, not from memory or from the README.
>
> **The only file to be designed is `web/index.html`.** Every Rust file is frozen.

---

## 1. What Judix is

**Judix is real-time, per-turn evaluation for AI agents and RAG.**

> Observability tells you what your agent *did*. Judix tells you, per step and in real time,
> whether it was any *good*, because a **deterministic Rust engine** computes the scores with
> **zero LLM calls**, and an AI model is used *only to explain the why*.

Built solo in Rust for the **OpenAI x NamasteDev Codex Hackathon** (Rust-only). Deadline
19 July 2026, 23:59 IST. The engine is complete, tested, and live in production. Only the UI
is being worked on.

- **Live:** https://judix-8piu.onrender.com
- **Repo:** github.com/Anuj7411/judix
- **Version:** 0.1.0, MIT, Rust edition 2021

### The thesis the design has to sell

**This is not an LLM wrapper.** An eval tool that asks a model to "rate this 1 to 10" *is* a
wrapper. Judix is not, for three reasons, and the interface has to make the first one unmissable:

1. **The scorer is a deterministic Rust engine, not a model.** Tool-call correctness (F1),
   loop detection, and JSON-schema validation are real computation with no network, no
   randomness, and no clock. Same input, same output, every time, for $0. The model is a
   *narrator*, never the judge.
2. **Real-time and per-turn.** Incumbents (LangSmith, Braintrust) run offline on sampled
   traces. They cannot catch the meaning of a turn while it is happening.
3. **The method is original.** Claim decomposition plus evidence grounding for RAG
   faithfulness is implemented here. The model only fills in per-claim verdicts and one-line
   reasons.

Supporting data point: LangChain's *State of Agent Engineering 2026* (1,340 practitioners)
found **89% have observability but only 52% run evals**, and online/real-time evals sit at
**37.3%**. Quality is the **number one** production barrier.

### Audience

Hackathon judges. Engineers. They arrive cold, give the page **30 to 60 seconds**, and leave.
They have already seen a dozen dark-mode Tailwind demos today. Data density is a feature, not
a cost, because the audience reads instruments for a living. The page is a **precision
instrument**, not a SaaS landing page.

---

## 1.5 Direction from here: live automatic per-turn evaluation and the Eval Flywheel

> Recorded from `JUDIX_update_spec.md`. This is the direction Judix takes from this
> hackathon build forward. The deterministic Rust engine, its scoring, its caps, and its
> numbers stay exactly as documented in sections 3 and 4. Everything below is added around
> that frozen core. House style for this section: no em dashes.

### 1.5.1 What changes, in one paragraph

Judix stops presenting hand-written JSON as the way to use it. The real product is live and
automatic: a developer wires Judix into their agent or RAG app one time, with a single hook,
and from then on every turn and every answer is scored live, per turn, automatically, with
no JSON and no manual work for anyone. The end user does nothing. On top of that, Judix
gains one standout feature, the **Eval Flywheel**, which turns every failure it catches in
production into a reusable regression test, so the offline eval suite writes itself. The
manual paste option stays, but only as an optional path for people who want it, never as
the main story.

### 1.5.2 Honest positioning (so the copy stays true)

- Judix scores whether an agent step or a RAG answer was actually right, live, per turn.
- The core scorer is a deterministic Rust engine, not a model and not a trained classifier.
  Exact, reproducible, zero cost, and explainable to the arithmetic. A model is used only
  to narrate the why, and only for the subjective checks.
- Real-time per-turn classifiers now exist elsewhere (Galileo Luna-2, Morph Reflex). Judix
  is **the deterministic, open-source, zero-cost take**. The structural failures (wrong
  tool, loop, contradiction) are caught by exact computation, not guessed by a paid model.
  Do not claim to be the only one doing per-turn. Claim to be the deterministic and free
  one.

### 1.5.3 The two front doors

Judix has two ways in, and each scores two things (agents and RAG).

**Front door A: the Playground** (try it, in the browser). Reorganized as a clear flow, not
a single dense screen.

1. **Entry chooser.** A short, engaging screen asking "What do you want to evaluate?" with
   two clear choices: *Score an AI agent* and *Score a RAG answer*. First thing a judge
   sees after opening the playground, so it has to orient in five seconds.
2. **Agent section.** On one side, a live example that runs and is scored turn by turn in
   front of the viewer. On the other side, a clearly visible *Score your own agent* option
   plus a short three-step how-to. The manual paste path lives here, secondary.
3. **RAG section.** A live example of a RAG answer being checked claim by claim against its
   sources, the contradicting claim flagged in place. A clearly visible *Score your own RAG
   answer* option and the same short how-to. Manual paste path here too, secondary.

**Front door B: the API and Service page** (use it in your app). A short docs-style page
that shows how a developer wires Judix into their own app so it runs live and automatically
in production. Covers the one-time hook, the request and response, and the action Judix
returns.

### 1.5.4 How live automatic scoring works (the hook)

Heart of the change. Follows the inline guardrail pattern that Galileo and Morph use,
because it is the credible, recognized approach.

- The developer adds one hook, one time: an inline call at the point where a step
  completes or an answer is produced, or a thin callback that does it for them. Configured
  by env vars for the API key and an optional self-hosted URL, exactly like the incumbents.
- After that, every turn and every answer is sent to Judix automatically. No JSON authored
  by a human. The hook produces the data under the hood.
- Judix returns fast. The deterministic verdict is ready in about a millisecond, so the app
  can act on it inline, before the reply is sent. The model narration streams in afterward
  and never blocks.
- Judix returns an **`action`** field, following the guardrail pattern: **`pass`**,
  **`flag`**, or **`block`**, derived from the score and the hard caps. This lets the app
  actually stop a bad turn, not just log it. The `block` decision fires on the
  deterministic verdict, which is the speed advantage made real.

**Agent side.** After each step, the hook sends the trace so far. Judix scores tool-call
correctness, loop, relevance, and drift, and returns the action. If run quality drops or a
critical cap fires, the app can stop the run before the reply is sent.

**RAG side.** When an answer is produced, the hook sends the question, the retrieved
sources, and the answer. Judix decomposes the answer into claims, grounds each against the
sources, flags any that contradict, caps the score, and returns the action. A hallucination
can be blocked or corrected before the user reads it.

**Framing.** The one-time hook is the commodity part that every tool has. What Judix does
once the data arrives (deterministic exact scoring at zero cost with severity caps) is the
part to show first.

### 1.5.5 The Eval Flywheel (the standout feature)

Because Judix scores every turn live, it already knows exactly which turns failed. The
flywheel turns that into value automatically.

- When Judix catches a failure live, it can save that exact case as a regression test: the
  agent run with its goal, steps, and expected tools, or the RAG answer with its question,
  sources, and the contradicting claim, along with the verdict as the baseline.
- Saved cases collect into a test suite that grows on its own from real production
  failures, instead of being hand-written and going stale.
- **Rerun any time.** Replaying a saved case through the deterministic engine confirms
  whether a fix still holds, so a change that reintroduces an old failure is caught before
  it ships.
- **Export as JSON** so it can run in CI. That is the bridge from Judix's live world to
  the offline suite everyone else lives in.

**One-line value.** Everyone else's eval suite goes stale because a human has to write the
tests. Judix watches production live, and every failure it catches becomes a regression
test automatically. The eval suite writes itself, for both agents and RAG.

**Honesty note for the docs.** Adding a trace to a dataset exists in other tools as a
manual flow. What is fresh here is that it is automatic, driven by deterministic live
per-turn scoring, covers both agents and RAG, and is open source and free. Frame it that
way, do not claim invention.

### 1.5.6 The deterministic corrective hint (light complement)

When Judix catches a failure, it can also return a rule-based corrective hint, not a model
retry.

- Agent, on a loop: `"you already made this exact call twice, break the loop and try a
  different tool."`
- RAG, on a contradiction: `"remove the unsupported claim: 30 days, the source says 14."`

These hints are deterministic and cheap, so they help without leaning on a model. A full
model-driven self-correction and retry loop is explicitly a **roadmap** item, not part of
this update, because it is model-heavy and would dilute the deterministic core.

### 1.5.7 The manual paste, kept as an optional path

The paste-your-own-trace box stays available in both the agent and the RAG sections,
clearly labeled as the manual option for people who want to check a single case by hand.
It is not the hero and not the default. The live automatic hook is the main story. The
manual box is a convenience.

### 1.5.8 What stays frozen (do not touch)

- The deterministic Rust engine: tool-call F1, loop detection, argument validation, the
  faithfulness ratio.
- The composite scoring, the weights, and the hard caps (step **49**, run **59**, RAG
  **49**).
- The three demos and their real numbers.
- The existing endpoints. New capabilities are added around them; the core is not
  rewritten.

The only thin additions permitted inside the Rust core:

- Adding the **`action`** field (`pass` / `flag` / `block`) to the scored report.
- The **save-as-test** capture endpoint that persists a failing case as JSON, so the
  flywheel has somewhere to write.

### 1.5.9 Roadmap (noted, not in this update)

- A full model-driven self-correction and retry loop with a correction budget and a human
  escalation path.
- Auto-instrumentation across frameworks, so even the one-line hook becomes zero lines for
  supported stacks.
- Drift tracking over time and a dashboard of failure patterns.

### 1.5.10 Guardrails for the build that follows this spec

- **Strong, effective, smooth.** The live hook and the flywheel must feel instant and
  reliable, the deterministic verdict is the thing that must never lag, and the demo must
  run live without a fumble.
- Keep the framing and structure in this section exactly: two front doors, the playground
  chooser plus agent and RAG sections with a live example and an optional manual box each,
  the API page, the `action` field, the flywheel for both, the deterministic hint, and the
  manual check as optional.
- Do not overclaim. Deterministic, exact, free, open source. Not the only one doing
  per-turn.
- Do not touch the frozen Rust core except to add the `action` field and the save-as-test
  capture, which are thin additions, not rewrites.

---

## 2. System architecture

Cargo workspace, three crates:

| Crate | Role |
| --- | --- |
| `judix-core` | The deterministic engine (`deterministic.rs`), composite scoring (`scoring.rs`), types (`types.rs`), and behind the optional `model` feature the model client (`model.rs`) and response cache (`cache.rs`). **The `model` feature is off by default**, so the hero engine compiles with zero network dependencies. |
| `judix-server` | axum HTTP server. Binds `$PORT` (default 8000). Embeds the playground and all three demo fixtures at compile time. |
| `judix-cli` | Score a trace from the terminal, deterministic only, no key. |

### Request flow

```
POST /score/agent
  |- deterministic engine   (always, $0, no network)  -> tool_call_correctness, loop_free
  |- model layer (optional) -> step_relevance, goal_drift   [fast provider]
  |    |- throttled?        -> fail over to strong provider
  |    |- confidence < 0.6? -> escalate to strong provider, mark low_confidence
  |- composite scoring      -> weighted average + hard caps -> band
```

### Deployment

Docker image to **Render free tier** (0.1 CPU, which matters: it is why cold paths cost
seconds). `rustls` rather than `native-tls`, so no OpenSSL in the image. A GitHub Actions
keep-warm ping hits `/health` every 10 minutes so judges never meet a cold start.
`[profile.release]` uses `opt-level = 1` deliberately: wall-clock is dominated by ~1.5s network
round-trips to model providers, the engine is microseconds either way, and level 1 cuts compile
memory and time on a constrained builder.

**Deploys are not automatic.** The Render webhook is dead. After merging, a human must click
**Render -> judix -> Manual Deploy -> Deploy latest commit**. A push is not live.
`/health` returns the running `commit` (from `RENDER_GIT_COMMIT`) precisely because a stale
deploy silently served a build ~10 commits behind for a full day.

---

## 3. The deterministic engine (the hero)

`crates/judix-core/src/deterministic.rs`. No model, no network, no clock, no randomness.
**16 unit tests passing** in `crates/judix-core/tests/engine_tests.rs`.

### 3.1 Tool-call correctness

```
final_f1 = 0.5 * set_f1 + 0.5 * bag_f1
score    = final_f1 * 100, minus 20 if any arg fails schema validation (floor 0)
pass     = args_ok AND final_f1 >= 0.8
raw_value = final_f1
```

- **set_f1**: F1 over the *unique* tool names called vs `expected_tools`.
- **bag_f1**: F1 over the *multiset* (counts), so calling one tool three times is penalized.
- Edge cases: both sides empty is a perfect 1.0; exactly one side empty is 0.0.
- Returns an **`na`** metric when the trace has no `expected_tools` (no golden label).
- Arg validation runs each call's `args` against `tool_schemas[name]` via the `jsonschema`
  crate. A *malformed schema* is skipped rather than punishing the agent for a bad label.
  Validation errors are attached as a `reason` string: `"Invalid tool args: {name} (step {i}): ..."`.

> **Design-critical:** `tool_call_correctness` is a **run-level** metric. It is computed once
> per trace and **cloned onto every tool-call step** (`scoring.rs:113` and `:125`). This is why
> `wrong_tool` shows F1 `41.67` identically on steps 1, 2, and 3. It is **one verdict about the
> whole run**, not three per-step judgments. The current UI renders it as a per-step chip, which
> visually triple-counts a single fact. The redesign must present it as run-level.

### 3.2 Loop / repeat detection

```
LOOP_WINDOW = 5   (sliding window of the most recent tool-call hashes)
hash        = "{tool_name}::{canonical_json(args)}"
repeat_count = number of identical hashes already in the window
score        = max(0, 100 - repeat_count * 25)
pass         = (repeat_count == 0)
raw_value    = repeat_count
```

`canonical_json` sorts object keys recursively, so `{a:1,b:2}` and `{b:2,a:1}` hash identically.
Only tool-call steps get this metric; LLM steps get `None`.

On `wrong_tool` this produces the signature progression **100 -> 75 -> 50** across the three
identical calls. That decay is the most legible piece of evidence on the page and deserves
visual weight.

### 3.3 Faithfulness ratio (the deterministic part of RAG)

```
faithfulness = supported / total * 100     (0 claims => 100, vacuously faithful)
```

Computed in Rust from the model's per-claim verdicts. The *ratio* is deterministic; the
*verdicts* are not. This is why RAG has no `deterministic_share`.

### 3.4 Span location

`find_span(answer, claim_text)` does a **verbatim** `str::find`. It returns `None` when the text
is not found character-for-character. See the hazard in section 9.2.

---

## 4. Composite scoring and the hard caps

`crates/judix-core/src/scoring.rs`.

### 4.1 Weights

| Agent step metric | Weight | | RAG metric | Weight |
| --- | --- | --- | --- | --- |
| `tool_call_correctness` | 0.30 | | `faithfulness` | 0.40 |
| `step_relevance` | 0.30 | | `answer_relevancy` | 0.25 |
| `goal_drift` | 0.25 | | `context_precision` | 0.20 |
| `loop_free` | 0.15 | | `context_recall` | 0.15 |

Weights are **renormalized over whatever is present**, so `na` metrics do not drag a score down;
they simply leave the average. Accumulation is in `f64` then clamped to 0..100, because f32
renormalization produced `100.00001` and a public API advertising 0 to 100 must never emit that.

- **step_quality** = renormalized weighted average of that step's non-`na` metrics.
- **run_quality** = plain mean of all non-`na` `step_quality` values.
- **rag_quality** = renormalized weighted average of the four RAG metrics.

### 4.2 The hard caps (severity beats averages)

A weighted average dilutes: three harmless-correct claims outvote one catastrophically wrong
one. So critical failures **override** the mean rather than being averaged away.

| Condition | Effect | Where |
| --- | --- | --- |
| `loop_free` fails on a step | that **step** capped at **49** (forced red) | `score_step` |
| any critical fail in the run (`loop_free` OR `tool_call_correctness` `pass == false`) | **run** capped at **59** | `score_agent` |
| `faithfulness < 50` | **RAG** capped at **49** | `score_rag` |
| **any claim contradicts the context** | **RAG** capped at **49** | `score_rag` |

> **This is the most important logic in the product and it is currently invisible.** Live RAG
> returns `faithfulness: 75`, `answer_relevancy: 100`, `context_precision: 33`,
> `context_recall: 0`, and a final `rag_quality: 49.0`. A judge reads that as broken arithmetic
> unless the UI states: **"capped at 49: a claim contradicts the context."** Rendering the cap,
> and naming which rule fired, converts an apparent bug into the severity-beats-averages
> argument. Design this. It is a headline moment, not a footnote.

### 4.3 Bands

Purely visual, derived from any 0-100 score:

| Band | Range |
| --- | --- |
| `green` | >= 80 |
| `amber` | 50 to 79 |
| `red` | < 50 |

The cap values are chosen to land exactly inside a band: 49 is red, 59 is amber.

### 4.4 deterministic_share

```
deterministic_share = deterministic_metric_count / total_non_na_metric_count
```

Present on **`AgentReport` only**. It is `0.375` on both agent demos, because a 5-step trace
yields 8 non-`na` metrics of which 3 are deterministic. LLM-only steps carry no deterministic
metric at all.

> **Honesty note for the design.** 37.5% is a minority of the metric *count*, so the framing
> "38% scored by Rust" is true but undersells. The stronger and equally honest framing is that
> **the deterministic metrics are the ones that caught the failure**: the F1 fail and the loop
> are what cap the run at 59 and force the red band. The model only narrated. Do not inflate the
> number, but do give the claim its correct weight.

---

## 5. The model layer (the narrator)

`crates/judix-core/src/model.rs`. Optional, behind the `model` feature. Any **OpenAI-compatible**
chat endpoint. Two independent providers.

| Role | Base URL | Key | Model |
| --- | --- | --- | --- |
| fast | `JUDIX_BASE_URL` | `JUDIX_API_KEY` | `JUDIX_MODEL_FAST` (default `gemini-flash-latest`) |
| strong | `JUDIX_STRONG_BASE_URL` (falls back to fast) | `JUDIX_STRONG_API_KEY` (falls back to fast) | `JUDIX_MODEL_STRONG` (default `llama-3.3-70b-versatile`) |

Production wiring: fast = Gemini (`gemini-3.1-flash-lite`, confirmed by live `/health`),
strong = Groq. If `JUDIX_API_KEY` is unset the entire layer is disabled and every model metric
comes back `na`. The app still runs keyless on the deterministic engine.

### 5.1 Model checks

| Check | Applies to | Returns |
| --- | --- | --- |
| `step_relevance` | each agent step | `{score, confidence, reason}` |
| `goal_drift` | each agent step | `{score, confidence, reason}` |
| `answer_relevancy` | RAG triple | `{score, confidence, reason}` |
| `context_precision` | RAG triple | `{score, confidence, reason}` |
| `context_recall` | RAG triple | `{score, confidence, reason}` |
| `rag_decompose` | RAG answer | `{claims:[{id, text, quote}]}` |
| `rag_verify` | RAG claims | `{results:[{id, status, context_index, reason}]}` where status is `supported` \| `contradicted` \| `unsupported` |

The relevance prompt explicitly instructs: *"HEAVILY penalize any step that violates an explicit
constraint in the goal."* That is why the downtown violation scores 0.

### 5.2 Reliability machinery

- **Failover, not backoff.** The two providers hold independent quotas, so a 429 on one is a
  reason to *use the other*. `chat()` fails over instantly and only backs off when both are busy.
  `MAX_RETRIES = 4`, `MAX_BACKOFF_MS = 12_000`.
- **Circuit breaker.** A throttled provider is skipped for `PROVIDER_COOLDOWN_SECS = 45`.
  Without it, a provider out of quota *for the day* still got tried first on every request,
  burning the full retry ladder: measured at **42s per request** once Gemini's daily quota ran out.
- **Concurrency cap.** `MAX_CONCURRENCY = 8` in-flight model calls. A 5-step trace fires 10 calls,
  so a whole trace goes in about 2 waves.
- **Escalation.** If the fast model reports `confidence < JUDIX_ESCALATE_BELOW` (**default 0.6**),
  the check is re-run on the strong model and the result is marked **`low_confidence: true`**.
  The default is 0.6 and not 0.5 on purpose: Gemini floors its uncertainty at *exactly* 0.5 and
  never dips below, so a literal `< 0.5` never fires and the strong model would never be consulted.
- **Response cache.** `SHA-256(check + model + normalized_input)`, whitespace-normalized, 1000
  entries, TTL **21600s (6h)**. A cache hit costs **$0** and returns `cost_usd: 0.0`. The TTL is
  eviction policy, not correctness: a different model or input is a different key, so a stale
  answer is impossible.
- **Denial-of-wallet bound.** `JUDIX_MAX_MODEL_STEPS` (**default 40**) caps how many steps of one
  request get model checks. The layer fires 2 calls per step with no natural ceiling and a minimal
  step is ~30 bytes, so ~60k steps fit inside axum's 2MB body limit, which would be ~120k model
  calls from one unauthenticated request. Steps past the cap still get **full deterministic
  scoring** and return `na` model metrics carrying the reason
  `"step beyond the first 40 - model checks capped per request (JUDIX_MAX_MODEL_STEPS); deterministic scoring still applied"`.
- **Cost accounting.** `model_cost_usd` is computed from real token usage at list price:
  Gemini Flash `(0.075, 0.30)` and Llama-3.3-70b `(0.59, 0.79)` USD per 1M tokens (input, output).
  Unknown models rate 0. Free-tier use bills $0, but the notional cost is exposed so the UI can
  say "this would cost $X at list price, you paid $0".

### 5.3 Rate limiting

`DEFAULT_RATE_LIMIT_PER_MIN = 20` scoring requests per client IP, fixed 60s window, tunable via
`JUDIX_RATE_LIMIT_PER_MIN`. **Only the scoring routes are limited.** `/health`, `/`, and `/demo/*`
are open, because the keep-warm pinger hits `/health` and a 429 there would let the service sleep.

Client IP comes from `CF-Connecting-IP` first (written by Cloudflare, so trustworthy), then
`X-Forwarded-For` as a fallback, then the literal string `"direct"` for local dev. Header order is
a security decision: `X-Forwarded-For`'s first entry is client-supplied and trusting it first would
let an attacker forge a fresh identity per request.

Exceeding it returns **429** with `Retry-After: 60` and:

```json
{ "error": "rate_limited",
  "message": "More than 20 scoring requests in a minute. Scoring spends model calls on a free tier, so it's capped per IP. Retry in 60s - /demo/* and /health are not limited." }
```

### 5.4 Prewarm

At boot, `prewarm()` scores all three fixtures in the background, fire-and-forget, with an 8s gap
between them (warming all three back-to-back burst ~25 requests and **rate-limited itself**, so the
RAG demo lost and never warmed). RAG retries once after a 16s pause. Measured: a warm RAG score is
~0.5s, but the very first one took **13.7s in production** vs 2.24s locally, because of the 0.1 CPU
free tier plus two sequential model waves. A judge's first click is exactly that cold path. After
prewarm, the first click is **0.21s**.

**Consequence for the design:** because prewarm populates the cache, live demo responses come back
with `latency_ms: 0` and `model_cost_usd: 0.0`. See hazard 9.1.

---

## 6. API contract

| Method | Path | Body | Returns | Rate limited |
| --- | --- | --- | --- | --- |
| `GET` | `/` | | the playground HTML | no |
| `GET` | `/health` | | `{ok, service, version, model_layer, model_fast, model_pool, commit}` | no |
| `GET` | `/api` | | machine-readable endpoint list | no |
| `GET` | `/demo/{id}` | | fixture JSON, or 404 `{error, valid:[...]}` | no |
| `POST` | `/score/agent` | `AgentTrace` | `AgentReport` | yes |
| `POST` | `/score/rag` | `RagTriple` | `RagReport`, or **501** | yes |
| `POST` | `/score/agent/stream` | `AgentTrace` | **SSE** | yes |
| `POST` | `/score/rag/stream` | `RagTriple` | **SSE** | yes |

Demo ids: `clean` | `wrong_tool` | `rag_hallucination`.

`POST /score/agent` works with **no key** (deterministic only; model metrics come back `na`).
`POST /score/rag` **requires** the model layer and returns `501` with
`{status: "model_required", message: "..."}` without it. A model failure on RAG returns **500**
`{error: "model error: ..."}`.

### 6.1 Input types

```jsonc
// AgentTrace
{ "goal": "string",
  "steps": [{ "kind": "llm" | "tool_call", "name"?: "string", "args"?: {}, "result"?: "string", "content"?: "string" }],
  "expected_tools"?: ["string"],
  "tool_schemas"?: { "tool_name": { /* JSON Schema */ } } }

// RagTriple
{ "question": "string", "contexts": ["string"], "answer": "string" }
```

A step counts as a tool call only when `kind == "tool_call"` **and** `name` is present.
Unknown fields (the fixtures carry `type`, `id`, `description`) are ignored by serde, so a demo
fixture can be POSTed verbatim.

### 6.2 Output types

```jsonc
// AgentReport
{ "run_quality": 17.25,          // 0-100 headline
  "band": "red",                 // green | amber | red
  "deterministic_share": 0.375,  // agent only; RAG has NO such field
  "latency_ms": 1211,
  "model_cost_usd": 0.000251,
  "steps": [{
    "index": 1,                  // 0-based
    "label": "search_restaurants",   // step.name, else "{kind} step" e.g. "llm step"
    "step_quality": 32.5,
    "band": "red",
    "na": false,                 // true = no computable metrics; exclude from the run mean
    "metrics": [{
      "name": "tool_call_correctness", // | loop_free | step_relevance | goal_drift
      "score": 41.67,
      "band": "red",
      "source": "deterministic",  // "deterministic" = Rust/$0 | "model" = the why
      "pass": false,              // deterministic only, omitted otherwise
      "raw_value": 0.4166,        // deterministic only (F1, or loop repeat count)
      "confidence": 0.9,          // model only
      "reason": "...",            // model only (also set on deterministic arg-validation failures)
      "na": false,
      "low_confidence": false     // true = escalated to the strong model
    }]
  }] }

// RagReport
{ "rag_quality": 49.0,
  "band": "red",
  "latency_ms": 211,
  "model_cost_usd": 0.00073,
  "metrics": [ /* faithfulness | answer_relevancy | context_precision | context_recall */ ],
  "unsupported_spans": [{
    "start": 0, "end": 48,       // char offsets INTO the answer string
    "text": "Customers have 30 days from the date of delivery to return an item.",
    "supported": false,
    "contradicted": true         // true = actively CONFLICTS with context (hallucination)
  }] }
```

Optional fields use `skip_serializing_if = "Option::is_none"`, so `pass`, `raw_value`,
`confidence`, and `reason` are **absent**, not null, when they do not apply. `na` and
`low_confidence` are always present.

### 6.3 SSE streaming (already built, already live)

> The README's Status table calls SSE "next". **It is done and deployed.** The routes exist at
> `main.rs:134-135`. The current UI does not use them. **The redesign must.**

Event protocol, one JSON object per `data:`:

| Event | Payload | When |
| --- | --- | --- |
| `deterministic` | a full `AgentReport` from the engine alone | **~1ms**, render immediately |
| `metric` | `{step_index, metric}` (agent) or `{metric}` (RAG) | as each model check lands |
| `claims` | `{claims:[{start, end, text}]}`, RAG only, **no verdicts yet** | after decomposition |
| `done` | the final report, weights and hard caps applied | when every check is in |
| `error` | `{message}` | terminal |

**Measured on `wrong_tool` (5 steps, 10 model calls): first paint 133ms vs 3534ms for the whole
run. A real score on screen 26x sooner.**

Rules the client must respect:

- **Browsers cannot POST with `EventSource`.** Consume with `fetch()` plus a `ReadableStream`
  reader. `new EventSource(...)` will not work.
- Model metrics arrive in **completion order, not step order**. The fastest explanations land
  first. `step_index` tells you where each belongs.
- The `deterministic` event is a *complete report* with caps already applied to what it knows.
  `done` is the *recomposed verdict* including model metrics, so numbers legitimately change
  between the two. Design that transition; do not hide it.
- Keyless mode emits `deterministic` then immediately `done` with the same payload.
- RAG `claims` deliberately carry **no** `supported`/`contradicted` flag, because emitting a
  claim early with `supported: false` would paint it red before it had been checked.

> **This is the product thesis as an animation.** The $0 Rust verdict paints in ~133ms; the
> model's narration fills in over the next ~3.4s. A judge *watches* the deterministic engine
> beat the model. Nothing else on the page will communicate the pitch as fast. This should be
> the signature moment of the design, and it is the one place motion is unambiguously earning
> its keep (it explains causality and ordering, per the motion rules in section 12.6).

---

## 7. The three demos and the story each tells

All numbers below are **live, verified** responses, not estimates.

### 7.1 `clean` -> **97.9 GREEN**

Goal: *"Book a dinner table for 2 on Friday evening, avoiding downtown."*
Expected tools: `search_restaurants`, `check_availability`, `book_table`. The agent searches
midtown, checks availability at Olive & Vine, books it at 19:30, confirms. F1 = 1.0 (pass),
no repeats, all steps relevant. `deterministic_share: 0.375`. The calm baseline that proves the
engine is not just a red-light generator.

### 7.2 `wrong_tool` -> **17.25 RED**. The money demo.

Goal says *avoid downtown*. The agent announces it will search downtown, calls
`search_restaurants({area:"downtown", party_size:2, day:"Friday"})` **three identical times**,
gets nothing each time, and never books. Then says *"Let me try downtown again."*

| Step | index | label | deterministic | model |
| --- | --- | --- | --- | --- |
| 0 | 0 | `llm step` | none | relevance 0, drift 0 |
| 1 | 1 | `search_restaurants` | F1 **41.67 FAIL**, loop **100 pass** | relevance 0, drift 20 |
| 2 | 2 | `search_restaurants` | F1 **41.67 FAIL**, loop **75 FAIL** (repeat x1) | relevance 0, drift 20 |
| 3 | 3 | `search_restaurants` | F1 **41.67 FAIL**, loop **50 FAIL** (repeat x2) | relevance 0, drift 20 |
| 4 | 4 | `llm step` | none | relevance 0, drift 0 |

step_quality: `0.0, 32.5, 28.75, 25.0, 0.0`. Mean = 17.25. Critical fail present, so the run is
also capped at 59 (the mean is already lower, so the cap does not bind here, but the *reason*
still holds and should be shown as "critical fail" state).

The F1 arithmetic, worth showing because it is checkable:
predicted = `[search_restaurants x3]`, expected = 3 distinct tools.
`set_f1 = 2*(1*(1/3))/(1+1/3) = 0.5`. `bag_f1 = 1/3`. `final = 0.5*0.5 + 0.5*0.333 = 0.4167`.

The model then explains: *"The agent explicitly violated the constraint to avoid downtown."*

**Rust caught the wrong tool and the loop instantly, at $0. The model only said why.** This is
the moment that wins the hackathon. Design for it.

### 7.3 `rag_hallucination` -> **49.0 RED**

Question: *"How many days do I have to return an item for a full refund?"*
Contexts say returns are allowed within **14 days**, items must be unused and in original
packaging, refunds issue within 5 business days.
Answer: *"You have 30 days from delivery to return an item for a full refund, as long as it is
unused and in its original packaging."*

Live metrics: `faithfulness 75` (amber, reason: *"3/4 claims grounded, 1 CONTRADICTS the context
(hallucination)"*), `answer_relevancy 100`, `context_precision 33`, `context_recall 0`
(**`low_confidence: true`**, so it was escalated to the strong model).

Weighted: `(75*0.40 + 100*0.25 + 33*0.20 + 0*0.15) = 61.6`. **Capped to 49** because a claim
contradicts. Band red.

The four spans tile the whole answer:

| start | end | text (the model's *paraphrase*) | supported | contradicted |
| --- | --- | --- | --- | --- |
| 0 | 48 | "Customers have 30 days from the date of delivery to return an item." | false | **true** |
| 49 | 66 | "Returns are eligible for a full refund." | true | false |
| 68 | 91 | "Items must be unused to be eligible for a return." | true | false |
| 92 | 121 | "Items must be in their original packaging to be eligible for a return." | true | false |

**Render the answer text with the contradicted span highlighted in place**, using `start`/`end`.
Seeing "30 days" go red *inside the sentence*, against a context that says 14, is far more
visceral than a list of claims. Supported claims should read calm, not decorated.

---

## 8. Every state the design must handle

Test these; do not assume.

| State | Trigger | Requirement |
| --- | --- | --- |
| **Idle / empty** | first load | Must look intentional and complete, not like a form waiting to be filled. This is the judge's first impression. |
| **Streaming, deterministic in** | ~133ms | Real scores on screen. Never a dead spinner. |
| **Streaming, model pending** | 133ms to ~3.5s | Per-metric slots that are visibly *awaiting*, not broken or empty. |
| **Streaming, recomposed** | `done` | Numbers change from the deterministic report. Show the change honestly. |
| **Cached / instant** | prewarmed demos | `latency_ms: 0`, `model_cost_usd: 0.0`. See 9.1. |
| **Cold start** | first hit after sleep | Up to ~13.7s historically, 0.21s after prewarm. Must degrade gracefully. |
| **`na` metric** | no key, or step past `MAX_MODEL_STEPS` | Deterministic-only mode must look **complete and deliberate**, not broken. Carry the `reason` when present. |
| **`na` step** | step with no computable metrics | Excluded from the run mean. Render as skipped, not as zero. |
| **`low_confidence: true`** | escalated to strong model | Show honestly. It is a feature: the system knew it was unsure and got a second opinion. |
| **501 model_required** | RAG with no key | `{status:"model_required", message}`. Explain, do not error out. |
| **500 model error** | provider failure | `{error: "model error: ..."}` |
| **429 rate limited** | >20 scoring req/min/IP | `{error:"rate_limited", message}` plus `Retry-After: 60`. **Not in the original brief; it is real and must be handled.** |
| **404 unknown demo** | bad id | `{error:"unknown demo id", valid:[...]}` |
| **Invalid pasted JSON** | user paste | Specific parse error, in the interface's voice. |
| **Empty input** | Score clicked with nothing | Guidance, not a crash. |
| **Very long trace** | 20+ steps | Must stay legible. Steps past 40 get `na` model metrics with an explanatory reason. |
| **Zero-length span** | quote not found verbatim | See 9.2. |

---

## 9. Data-shape hazards (found by reading source and live payloads)

These are the traps that will silently produce a wrong or empty UI.

### 9.1 `latency_ms: 0` and `model_cost_usd: 0.0` are normal, not bugs

Prewarm fills the cache, so live demo responses report zero latency and zero cost. The current UI
prints `$0.0000 · 0 ms`, which reads to a judge as **fake or broken**. The honest presentation:
show the real client-measured round-trip, label a cached response as cached, and present `$0` as
the genuine claim it is rather than a suspicious `0.0000`.

### 9.2 RAG span `text` is NOT a substring of the answer

`ClaimItem` carries **two** fields: `text` (the model's normalized claim) and `quote` (a verbatim
substring). `find_span` runs on the **quote**; `text` is a paraphrase. Live proof: span 1's `text`
is *"Customers have 30 days from the date of delivery to return an item."* while the answer reads
*"You have 30 days from delivery to return an item..."*. The offsets are correct; the text is not
present in the answer.

**Therefore: slice the answer by `start`/`end`. Never match by `text`.** Matching by string finds
nothing and renders no highlight at all.

**And:** when the model's quote is not verbatim, `find_span` returns `None` and the span becomes
**`start: 0, end: 0`** (`model.rs:982-983`). A naive offset walk will emit a zero-length highlight
or corrupt the render. Sort spans by `start`, skip any where `start == end`, and handle overlaps
defensively. Unlocated claims should be listed separately rather than highlighted.

### 9.3 `unsupported_spans` also contains **supported** spans

The field name misleads. Three of the four live spans are `supported: true`. Treat it as
"all claims". Three states exist, not two: **supported**, **unsupported** (no evidence), and
**contradicted** (actively conflicts). The current UI collapses the last two into one red, losing
the distinction that drives the cap.

### 9.4 `tool_call_correctness` is run-level, cloned per step

See 3.1. Do not present one verdict as three.

### 9.5 Model `reason` strings contain em-dashes

`compose_faithfulness` emits `"3/4 claims grounded — 1 CONTRADICTS the context (hallucination)"`,
and model-authored reasons may contain them too. Rust is frozen, so **strip or replace em-dashes
at the display layer** in JS. This is a presentation transform, not an API change.

### 9.6 `latency_ms` on the streaming path

`deterministic` reports elapsed at ~1ms; `done` reports total elapsed. They are different numbers
for the same run by design.

---

## 10. Hard constraints (violating these breaks the build)

1. **`web/index.html` must remain ONE self-contained HTML file.** It is embedded into the Rust
   binary at compile time via `include_str!("../../../web/index.html")` at `crates/judix-server/src/main.rs:20`.
2. **No npm, no bundler, no build step.** The Docker image only runs `cargo build`. No `import` of
   local files. Do not add a `package.json`.
3. **No Rust changes.** The API is frozen and working. If the design needs an API change, stop and
   say so.
4. **There is no static file route.** The server serves `/` as HTML and nothing else. There is no
   `ServeDir`. **Fonts and icons therefore cannot be self-hosted** without a Rust change. They must
   come from a CDN with `preconnect` and `font-display: swap`, or be system fonts. This overrides
   the usual "never `<link>` Google Fonts, always self-host" rule; self-hosting is not reachable here.
5. External CDNs are acceptable. Inline `<style>` and `<script>` are fine.
6. Keep the three demo buttons, the paste-your-own-JSON box, and the `GitHub` link.
7. Editing `index.html` triggers a `judix-server` recompile, because of `include_str!`. It touches
   no Rust source, so it cannot conflict with parallel Rust work, but the binary does rebuild.

---

## 11. What is wrong with the current UI

Diagnosed against the design rules in section 12.

### 11.1 The hierarchy is inverted

Ranked by what a judge actually needs:

1. **The verdict**: `run_quality` plus band.
2. **The evidence for why**: the F1 fail and the loop count, tied to the actual
   `"area": "downtown"` args that violate the goal.
3. **The claim**: computed by Rust, $0, no model.
4. **The model's narration**: secondary, supporting.
5. Demo triggers.
6. The paste box.
7. Brand and repo link.

Today the **paste box (rank 6) is the largest element above the results**, and **rank 2 does not
exist on the page at all**. The UI never shows the goal, never shows the tool args, so the money
demo displays a red number with no visible crime. A judge cannot see that the agent searched
downtown when told not to. **Fixing this ordering matters more than any palette decision.**

### 11.2 Named anti-pattern violations

| Violation | Where | Rule |
| --- | --- | --- |
| **Side-stripe border** | `border-l-2 border-indigo-500 pl-3` on the tagline | Absolute ban. Never intentional. |
| **AI purple/blue** | `indigo-600`/`indigo-500` on `slate` throughout | The single most-generated palette. Named ban. |
| **Emoji as icon** | the `🦀` in the header | Emoji discouraged; use a real icon family or a wordmark. |
| **Em-dashes** | throughout the copy and in `reason` strings | Zero tolerance. The number one AI copy tell. |
| **Legend instead of structure** | "▮ computed by Rust / ▮ model explanation" | The brief and the design rules agree independently: the distinction must be **structural**, not a legend. Delete the legend, move the distinction into the layout. |
| **Uniform card treatment** | every block is `rounded-xl` + border + tint | Flattens to dead-uniform blur under a squint test. Emphasize by de-emphasizing instead. |
| **Flat type scale** | system stack, few sizes, weak contrast | Typography should carry the identity. |
| **Blocking endpoints** | `/score/agent` not `/score/agent/stream` | Leaves the single best proof of the thesis unused. |

---

## 12. The design brief

### 12.1 Design Read

> A data-dense evaluation instrument for hackathon judges (engineers, 30 to 60 seconds,
> evaluating rather than browsing), in a test-report / measurement language, leaning on native
> CSS tokens and mono-led typography.

This is an **application surface**, not a landing page. Landing-page and moodboard playbooks do
not apply. Density is correct here; whitespace would be friction.

### 12.2 Intensity dials

| Dial | Value | Reason |
| --- | --- | --- |
| `DESIGN_VARIANCE` | **5** | It is an app. Symmetric structure aids comparison between steps. |
| `MOTION_INTENSITY` | **3** | Motion only where it explains. The streaming reveal is the one earned exception. |
| `VISUAL_DENSITY` | **8** | Judges are engineers; data density is a feature. Cockpit, not gallery. |

### 12.3 The two feelings, and the business goal

Feelings: **earned confidence** and **precision**. Business goal: a judge concludes within 30
seconds that this is real computation and not a model wrapper, and remembers it tomorrow.

### 12.4 Typography

Do not use Inter, Roboto, Poppins, or Montserrat by reflex. Do not use Playfair, Fraunces, or
Instrument Serif as the reflex "premium serif". Pick a display face with a point of view and a
quiet body face, paired on a real contrast axis, maximum 3 families.

Mandatory regardless of face: **tabular numerals** (`font-variant-numeric: tabular-nums`) on every
score, because numbers change in place during streaming and must not jitter. `slashed-zero` in mono
contexts. Headings tight (`letter-spacing: -0.02em` to `-0.04em`, line-height 1.1 to 1.25), body
1.5 to 1.6, dense UI 1.4. Prose capped at 65 to 75ch. `text-wrap: balance` on headings,
`text-wrap: pretty` on paragraphs.

> **Watch out:** "technical / dark / terminal + mono" is the *second-order* reflex for a Rust dev
> tool. Avoiding SaaS-cream and landing on terminal-dark-with-JetBrains-Mono is the trap one tier
> deeper. If someone could guess the aesthetic from "Rust eval engine" alone, rework it.

### 12.5 Color

**The bands already carry meaning, so they should be the only saturated color on the page.**
Everything else earns restraint. Build in **OKLCH**. Tint the neutral ramp slightly toward the
chosen hue (chroma 0.004 to 0.02) rather than using dead gray. Never `#000` or `#fff`. One accent,
locked. Give green/amber/red matched lightness and chroma so they read as one family rather than
stock Bootstrap.

Banned by reflex: AI indigo/violet glow (what the page uses today), beige-and-brass, gradient text,
neon outer glows, decorative glassmorphism.

Contrast is non-negotiable: body >= 4.5:1, large text >= 3:1, placeholders >= 4.5:1, focus rings
and UI boundaries >= 3:1. The page must be legible on a **laptop projector in a bright room**.

### 12.6 Motion

Every animation must hold one of four jobs: **explain**, **confirm**, **direct**, or **express**.
If it holds none, it does not exist. Animate `transform` and `opacity` only. Exits faster than
entrances. Nothing linear for UI movement. Everything interruptible. `prefers-reduced-motion`
replaces movement with fast opacity changes, never with nothing.

The **streaming reveal is the signature**: it explains causality (Rust first, model after) and it
is the pitch. Keep everything else quiet so the signature can be loud. Two specific candidates that
genuinely explain rather than decorate:

- A score **settling** into place as `deterministic` is superseded by `done`.
- The loop repeats **stacking** visibly as 100 -> 75 -> 50, which is the actual algorithm made visible.

Buttons get `:active { transform: scale(0.97) }`. Every interactive element needs all eight states:
default, hover, focus, active, disabled, loading, error, success. Focus via `:focus-visible`, never
`outline: none` without replacement. Touch targets >= 44px.

### 12.7 Copy

Zero em-dashes, in the page and in anything rendered from the API (strip them in JS, per 9.5).
No buzzwords: streamline, empower, supercharge, seamless, world-class, unleash, elevate, next-gen,
revolutionize, robust, cutting-edge, game-changer. No "not just X, it's Y" or "X theater"
constructions. No fake-perfect numbers; the real ones (17.25, 41.67, 0.375, 133ms vs 3534ms) are
better than anything invented. Button labels are verb plus object. Errors say what happened and how
to fix it. Specific beats clever.

### 12.8 Icons

One family only, from a CDN (Phosphor is the recommended default given the constraint in 10.4).
One stroke weight globally, sizes from a small set on a consistent grid, nothing below 12px. No
hand-rolled SVG paths. No emoji. Consistent fill logic (for example filled = active).

### 12.9 What "done" looks like

Every element of the pre-flight matrix passes, the hierarchy in 11.1 is served, and the page could
sit beside a product you respect and read as a peer, differing only on purpose.

---

## 13. Verification (do not skip)

```bash
cd /c/Projects/judix
export PATH="$HOME/.cargo/bin:/c/Users/ojhaa/winlibs/mingw64/bin:$PATH"   # required: system MinGW is broken
cargo run -p judix-server        # http://localhost:8000
```

Deterministic scoring works with no API key. Model metrics need the env vars from section 5.

1. Open in a real browser. Click all three demo buttons. Confirm the real numbers from section 7
   render: **97.9 green**, **17.25 red**, **49.0 red**. Check the console for errors.
2. Test the streaming path specifically: the deterministic report must paint before the model
   metrics land.
3. Test keyless mode (unset `JUDIX_API_KEY`) so `na` handling is proven, not assumed.
4. Check mobile and a 1280px laptop viewport.
5. Confirm `cargo build -p judix-server` still succeeds, since the HTML compiles into the binary.
6. Screenshots of all three demo states before calling it done.

Reference commands:

```bash
curl -s https://judix-8piu.onrender.com/health
curl -s https://judix-8piu.onrender.com/demo/wrong_tool \
  | curl -s -X POST https://judix-8piu.onrender.com/score/agent \
         -H 'content-type: application/json' --data-binary @-
curl -N -X POST http://localhost:8000/score/agent/stream \
  -H 'content-type: application/json' --data-binary @demos/wrong_tool.json
```

**Deploy is manual.** Render -> judix -> Manual Deploy -> Deploy latest commit. Verify the
`commit` field in `/health` afterwards to prove the deploy actually landed.
