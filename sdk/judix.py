"""Judix SDK: score every agent turn and every RAG answer live, automatically.

A thin, dependency-free client over the Judix streaming endpoints. Wire it in once and
every turn and every answer is scored live. The deterministic verdict lands in about a
millisecond, so you can act on it before the reply is sent; the model narration streams
in afterward and never blocks.

The scorer is a deterministic Rust engine (tool-call F1, loop detection, schema checks,
RAG claim grounding). This SDK computes nothing itself except the pass/flag/block
`action`, which is derived from the score and the hard caps the engine already returns.

Config (env):
  JUDIX_URL       base URL, default the live service
  JUDIX_API_KEY   optional. Agent scoring works without it (deterministic only).

Stdlib only, so there is nothing to pip install. Copy this file and import it.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

DEFAULT_URL = "https://judix-8piu.onrender.com"


def _strip_dashes(s: Any) -> Any:
    # No em or en dashes in anything we surface, including model reason strings.
    if isinstance(s, str):
        for d in ("\u2014", "\u2013"):
            s = s.replace(" " + d + " ", ", ").replace(d, ", ")
    return s


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

@dataclass
class Metric:
    name: str
    score: float
    source: str            # "deterministic" or "model"
    band: str = ""
    passed: Optional[bool] = None
    reason: str = ""
    na: bool = False
    low_confidence: bool = False


@dataclass
class Step:
    index: int
    label: str
    step_quality: float
    band: str
    na: bool
    metrics: List[Metric] = field(default_factory=list)


@dataclass
class Report:
    kind: str                       # "agent" or "rag"
    quality: float                  # run_quality (agent) or rag_quality (rag)
    band: str                       # green | amber | red
    action: str                     # pass | flag | block
    steps: List[Step] = field(default_factory=list)
    metrics: List[Metric] = field(default_factory=list)   # RAG top-level metrics
    spans: List[Dict[str, Any]] = field(default_factory=list)  # RAG claim spans
    deterministic_share: Optional[float] = None
    model_cost_usd: float = 0.0
    latency_ms: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    def reason_line(self) -> str:
        """One short human line explaining the action."""
        if self.action == "block":
            for s in self.spans:
                if s.get("contradicted"):
                    return "blocked: a claim contradicts the source"
            for st in self.steps:
                for m in st.metrics:
                    if m.name == "loop_free" and m.passed is False:
                        return "blocked: the agent is looping"
                    if m.name == "tool_call_correctness" and m.passed is False:
                        return "blocked: the wrong tool was called"
            return "blocked: quality below 50"
        if self.action == "flag":
            return "flagged: quality in the amber band"
        return "pass"


# --------------------------------------------------------------------------- #
# The action rule (1.4). Computed here, never in Rust.
# --------------------------------------------------------------------------- #

def _critical_cap_fired(report: Dict[str, Any], kind: str) -> bool:
    if kind == "rag":
        return any(s.get("contradicted") for s in report.get("unsupported_spans", []))
    for st in report.get("steps", []):
        for m in st.get("metrics", []):
            if m.get("name") in ("loop_free", "tool_call_correctness") and m.get("pass") is False:
                return True
    return False


def _has_signal(report: Dict[str, Any], kind: str) -> bool:
    """Was anything actually scored? A trace with only planning steps and no tool calls
    (keyless) has no computable metric yet, so its 0 quality is 'not scored', not a
    failure. Block must mean a detected problem, never merely 'nothing scored yet'."""
    if kind == "rag":
        return bool(report.get("unsupported_spans")) or any(not m.get("na") for m in report.get("metrics", []))
    return any(not m.get("na") for st in report.get("steps", []) for m in st.get("metrics", []))


def compute_action(report: Dict[str, Any], kind: str) -> str:
    quality = report.get("run_quality", report.get("rag_quality", 0.0))
    if _critical_cap_fired(report, kind):
        return "block"
    if not _has_signal(report, kind):
        return "pass"          # nothing computed yet, nothing to block on
    if quality < 50:
        return "block"
    if quality < 80:
        return "flag"
    return "pass"


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class Judix:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = (url or os.environ.get("JUDIX_URL") or DEFAULT_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("JUDIX_API_KEY")

    def _stream(self, path: str, payload: Dict[str, Any]):
        """Yield (event, data) tuples from a Judix SSE endpoint."""
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            event, data = "message", ""
            for raw in resp:
                line = raw.decode("utf-8").rstrip("\n")
                if line == "":
                    if data:
                        try:
                            yield event, json.loads(data)
                        except json.JSONDecodeError:
                            pass
                    event, data = "message", ""
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data += line[5:].strip()

    def _collect_agent(self, payload: Dict[str, Any]) -> Report:
        deterministic: Dict[str, Any] = {}
        model_metrics: Dict[int, List[Dict[str, Any]]] = {}
        final: Dict[str, Any] = {}
        for event, d in self._stream("/score/agent/stream", payload):
            if event == "deterministic":
                deterministic = d
            elif event == "metric":
                model_metrics.setdefault(d.get("step_index", -1), []).append(d.get("metric", {}))
            elif event == "done":
                final = d
            elif event == "error":
                raise RuntimeError(_strip_dashes(d.get("message", "scoring error")))
        report = final or deterministic
        return _build_report(report, "agent")

    def _collect_rag(self, payload: Dict[str, Any]) -> Report:
        final: Dict[str, Any] = {}
        for event, d in self._stream("/score/rag/stream", payload):
            if event == "done":
                final = d
            elif event == "error":
                if d.get("status") == "model_required":
                    raise RuntimeError("RAG scoring needs the model layer. Set JUDIX_API_KEY.")
                raise RuntimeError(_strip_dashes(d.get("message", "scoring error")))
        if not final:
            raise RuntimeError("RAG scoring returned no result")
        return _build_report(final, "rag")

    # ---- public core calls ------------------------------------------------ #

    def score_agent(self, goal: str, steps: List[Dict[str, Any]],
                    expected_tools: Optional[List[str]] = None,
                    tool_schemas: Optional[Dict[str, Any]] = None) -> Report:
        payload: Dict[str, Any] = {"goal": goal, "steps": steps}
        if expected_tools is not None:
            payload["expected_tools"] = expected_tools
        if tool_schemas is not None:
            payload["tool_schemas"] = tool_schemas
        return self._collect_agent(payload)

    def score_rag(self, question: str, contexts: List[str], answer: str) -> Report:
        return self._collect_rag({"question": question, "contexts": contexts, "answer": answer})


def _build_report(report: Dict[str, Any], kind: str) -> Report:
    steps = []
    for st in report.get("steps", []):
        metrics = [Metric(
            name=m.get("name", ""), score=float(m.get("score", 0.0)), source=m.get("source", ""),
            band=m.get("band", ""), passed=m.get("pass"), reason=_strip_dashes(m.get("reason", "")),
            na=bool(m.get("na", False)), low_confidence=bool(m.get("low_confidence", False)),
        ) for m in st.get("metrics", [])]
        steps.append(Step(
            index=st.get("index", 0), label=st.get("label", ""),
            step_quality=float(st.get("step_quality", 0.0)), band=st.get("band", ""),
            na=bool(st.get("na", False)), metrics=metrics,
        ))
    top = [Metric(
        name=m.get("name", ""), score=float(m.get("score", 0.0)), source=m.get("source", ""),
        band=m.get("band", ""), reason=_strip_dashes(m.get("reason", "")),
        na=bool(m.get("na", False)), low_confidence=bool(m.get("low_confidence", False)),
    ) for m in report.get("metrics", [])]
    return Report(
        kind=kind,
        quality=float(report.get("run_quality", report.get("rag_quality", 0.0))),
        band=report.get("band", ""),
        action=compute_action(report, kind),
        steps=steps, metrics=top, spans=report.get("unsupported_spans", []),
        deterministic_share=report.get("deterministic_share"),
        model_cost_usd=float(report.get("model_cost_usd", 0.0)),
        latency_ms=int(report.get("latency_ms", 0)),
        raw=report,
    )


# --------------------------------------------------------------------------- #
# Ergonomic hooks: one-line live scoring
# --------------------------------------------------------------------------- #

_default = Judix()


def score_agent(goal, steps, expected_tools=None, tool_schemas=None) -> Report:
    return _default.score_agent(goal, steps, expected_tools, tool_schemas)


def score_rag(question, contexts, answer) -> Report:
    return _default.score_rag(question, contexts, answer)


class JudixCallback:
    """LangChain-style callback. Collects each step and scores the run live.

    Plugs into any framework that calls `on_tool_start`, `on_tool_end`, `on_llm_end`.
    Also usable directly: call `.record(kind, name, args, result)` per step and read
    `.action` / `.report` after each one.
    """

    def __init__(self, goal: str, expected_tools: Optional[List[str]] = None,
                 tool_schemas: Optional[Dict[str, Any]] = None,
                 client: Optional[Judix] = None, score_each_step: bool = True):
        self.goal = goal
        self.expected_tools = expected_tools
        self.tool_schemas = tool_schemas
        self.client = client or _default
        self.score_each_step = score_each_step
        self.steps: List[Dict[str, Any]] = []
        self.report: Optional[Report] = None
        self._pending_tool: Optional[Dict[str, Any]] = None

    # direct API
    def record(self, kind: str, name: Optional[str] = None,
               args: Optional[Dict[str, Any]] = None,
               result: Optional[str] = None, content: Optional[str] = None) -> Report:
        step: Dict[str, Any] = {"kind": kind}
        if name is not None:
            step["name"] = name
        if args is not None:
            step["args"] = args
        if result is not None:
            step["result"] = result
        if content is not None:
            step["content"] = content
        self.steps.append(step)
        if self.score_each_step:
            self.report = self.client.score_agent(self.goal, self.steps, self.expected_tools, self.tool_schemas)
        return self.report

    def score_now(self) -> Report:
        self.report = self.client.score_agent(self.goal, self.steps, self.expected_tools, self.tool_schemas)
        return self.report

    @property
    def action(self) -> str:
        return self.report.action if self.report else "pass"

    # LangChain-shaped hooks
    def on_llm_end(self, response=None, **kw):
        text = ""
        try:
            text = response.generations[0][0].text  # LangChain LLMResult shape
        except Exception:
            text = kw.get("content") or (str(response) if response is not None else "")
        self.record("llm", content=text)

    def on_tool_start(self, serialized=None, input_str=None, **kw):
        name = (serialized or {}).get("name") if isinstance(serialized, dict) else None
        args = kw.get("args")
        if args is None and input_str is not None:
            try:
                args = json.loads(input_str)
            except Exception:
                args = {"input": input_str}
        self._pending_tool = {"name": name or kw.get("name", "tool"), "args": args or {}}

    def on_tool_end(self, output=None, **kw):
        pend = self._pending_tool or {"name": kw.get("name", "tool"), "args": kw.get("args", {})}
        self._pending_tool = None
        self.record("tool_call", name=pend["name"], args=pend["args"],
                    result=str(output) if output is not None else None)


class JudixRagCallback:
    """Scores a RAG answer live once the chain produces it against its contexts."""

    def __init__(self, client: Optional[Judix] = None):
        self.client = client or _default
        self.report: Optional[Report] = None

    def check(self, question: str, contexts: List[str], answer: str) -> Report:
        self.report = self.client.score_rag(question, contexts, answer)
        return self.report

    # convenience for chains that pass these through
    def on_chain_end(self, outputs=None, **kw):
        if not isinstance(outputs, dict):
            return
        q = outputs.get("question") or kw.get("question")
        ctx = outputs.get("contexts") or kw.get("contexts")
        ans = outputs.get("answer") or outputs.get("result") or kw.get("answer")
        if q and ctx and ans:
            self.check(q, list(ctx), ans)

    @property
    def action(self) -> str:
        return self.report.action if self.report else "pass"


def watch(goal: str, expected_tools: Optional[List[str]] = None,
          tool_schemas: Optional[Dict[str, Any]] = None):
    """Decorator: wrap an agent function that returns a list of step dicts. The run is
    scored when the function finishes and the Report is attached at `.judix_report`.
    """
    def deco(fn: Callable[..., List[Dict[str, Any]]]):
        def wrapped(*a, **kw):
            steps = fn(*a, **kw)
            report = score_agent(goal, steps or [], expected_tools, tool_schemas)
            wrapped.judix_report = report  # type: ignore[attr-defined]
            return report
        wrapped.judix_report = None  # type: ignore[attr-defined]
        return wrapped
    return deco
