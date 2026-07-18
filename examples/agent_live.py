"""Live per-turn agent scoring, the wrong_tool run.

A tiny five-step agent. After each step the trace so far is scored live through Judix.
Watch run_quality drop and the action flip to block the moment the loop is caught, before
the final reply is ever sent. Runs against the deterministic engine at $0 (no API key
needed); if JUDIX_API_KEY is set, model reasons stream in too.

Run:  python examples/agent_live.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))
from judix import JudixCallback  # noqa: E402

GOAL = "Book a dinner table for 2 on Friday, avoiding downtown."
EXPECTED = ["search_restaurants", "check_availability", "book_table"]

# What the (bad) agent does, one step at a time. It searches downtown, three identical
# times, then gives up. The correct move was to search a different area and book.
TURNS = [
    ("llm", None, None, "I'll look for restaurants downtown."),
    ("tool_call", "search_restaurants", {"area": "downtown", "party_size": 2, "day": "Friday"}, "No availability found downtown for Friday."),
    ("tool_call", "search_restaurants", {"area": "downtown", "party_size": 2, "day": "Friday"}, "No availability found downtown for Friday."),
    ("tool_call", "search_restaurants", {"area": "downtown", "party_size": 2, "day": "Friday"}, "No availability found downtown for Friday."),
    ("llm", None, None, "Still nothing downtown. Let me try downtown again."),
]

BAR = {"pass": "PASS ", "flag": "FLAG ", "block": "BLOCK"}


def main():
    # Per turn, we score the trace so far. tool_call_correctness is a whole-run verdict
    # (it needs the full expected-tool list to mean anything), so it is not part of the
    # per-turn signal. The genuine per-turn deterministic catch is the loop: an identical
    # call repeated. We leave expected_tools off during the run, then apply it once at the
    # end for the run-level tool-call verdict.
    judge = JudixCallback(GOAL, expected_tools=None)
    print(f"goal: {GOAL}")
    print(f"scoring every turn live through {judge.client.url}\n")
    print(f"{'turn':<5}{'what the agent did':<48}{'run':>7}  {'band':<6} action")
    print("-" * 78)

    blocked_at = None
    for i, (kind, name, args, text) in enumerate(TURNS, start=1):
        report = judge.record(kind, name=name, args=args, result=text if kind == "tool_call" else None, content=text if kind == "llm" else None)
        what = f"{name}({args['area']})" if kind == "tool_call" else f'llm: "{text[:34]}"'
        print(f"{i:<5}{what:<48}{report.quality:>7.2f}  {report.band:<6} {BAR[report.action]}")
        if report.action == "block":
            blocked_at = i
            print("-" * 78)
            print(f"{'':5}caught at turn {i}: {report.reason_line()}. Stop before the reply is sent.")
            print(f"{'':5}the loop was detected in Rust, no model call, at $0.")
            break

    # Run-level verdict for the full run (all three identical searches), with the golden
    # tool list. This is the whole-run tool-call F1, the canonical 41.67.
    from judix import score_agent
    full_steps = [
        {"kind": k, **({"name": n} if n else {}), **({"args": a} if a else {}),
         **({"result": t} if k == "tool_call" else {"content": t})}
        for (k, n, a, t) in TURNS
    ]
    final = score_agent(GOAL, full_steps, expected_tools=EXPECTED)
    f1 = next((m for st in final.steps for m in st.metrics if m.name == "tool_call_correctness" and not m.na), None)
    if f1:
        print(f"\nrun-level tool-call verdict (full run): F1 {f1.score:.2f} "
              f"{'FAIL' if f1.passed is False else 'pass'}, expected 3 distinct tools, only "
              f"search_restaurants was ever called")
    if blocked_at is None:
        print("\nrun finished without a block")


if __name__ == "__main__":
    main()
