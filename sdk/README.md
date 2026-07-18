# Judix SDK

Score every agent turn and every RAG answer live, automatically. One hook, and from then
on every step and every answer is graded by the deterministic Judix engine as it runs. The
verdict lands in about a millisecond, so you can act on it before the reply is sent.

Single file, standard library only. Nothing to install: copy `judix.py` and import it.

```
JUDIX_URL       base URL, default https://judix-8piu.onrender.com
JUDIX_API_KEY   optional; agent scoring works without it (deterministic, $0)
```

## Agent: score every turn live

```python
from judix import JudixCallback

judge = JudixCallback(goal="Book a table for 2 on Friday, avoiding downtown",
                      expected_tools=["search_restaurants", "check_availability", "book_table"])

# plug into any framework that calls on_tool_start / on_tool_end / on_llm_end
agent.run(task, callbacks=[judge])

if judge.action == "block":
    stop()          # act before the reply is sent
```

Or drive it directly, one call per step:

```python
judge = JudixCallback(goal, expected_tools)
judge.record("llm", content="I'll search downtown")
report = judge.record("tool_call", name="search_restaurants",
                      args={"area": "downtown"}, result="no availability")
print(report.quality, report.band, report.action)   # e.g. 59.0 amber block
```

## RAG: check every answer live

```python
from judix import JudixRagCallback

judge = JudixRagCallback()
report = judge.check(question, contexts, answer)     # runs the moment the answer exists

if judge.action == "block":
    correct_or_stop()                                # catch the hallucination first
```

## The `action` field

Derived on the client from the score and the engine's hard caps. No model, no guessing.

| action | when |
| --- | --- |
| `block` | run_quality below 50, or a critical cap fired: a loop, a failed tool-call check, or a RAG claim that contradicts the source |
| `flag`  | run_quality 50 to 79 (amber) |
| `pass`  | run_quality 80 or above, and nothing failed |

`report.reason_line()` gives one short sentence, for example `blocked: the agent is looping`.

## The report

`score_agent(...)` and `score_rag(...)` return a `Report` with `quality`, `band`, `action`,
`deterministic_share`, `model_cost_usd`, per-step `metrics` (each tagged `deterministic` or
`model`), and RAG `spans`. Model reason strings have em and en dashes stripped.

## Runnable examples

```bash
python examples/agent_live.py   # deterministic loop catch, per turn, $0
python examples/rag_live.py     # hallucination caught claim by claim (needs the model layer)
```

`agent_live.py` scores the wrong_tool run one turn at a time and blocks at turn 3 the moment
the loop is detected, in Rust, at $0, then prints the run-level tool-call verdict (F1 41.67).
`rag_live.py` grades the rag_hallucination answer and blocks on the contradicted 30-days claim
against a source that says 14.

The core scorer is a deterministic Rust engine. This SDK is a thin client over its streaming
endpoints and computes only the pass/flag/block decision from what the engine returns.
