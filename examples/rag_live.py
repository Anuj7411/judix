"""Live RAG answer check, the rag_hallucination case.

A RAG chain produces an answer from retrieved sources. Judix decomposes it into claims,
grounds each against the sources, and flags the one that contradicts, before the user
reads it. RAG grounding needs the model layer, so set JUDIX_API_KEY (a free Gemini key)
or point JUDIX_URL at the live service, which already has keys.

Run:  python examples/rag_live.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdk"))
from judix import JudixRagCallback  # noqa: E402

QUESTION = "How many days do I have to return an item for a full refund?"
CONTEXTS = [
    "Our return policy allows returns within 14 days of delivery for a full refund.",
    "Items must be unused and in their original packaging to qualify for a refund.",
    "Refunds are issued to the original payment method within 5 business days of us receiving the returned item.",
]
# The answer the RAG chain produced. It says 30 days. The source says 14.
ANSWER = ("You have 30 days from delivery to return an item for a full refund, "
          "as long as it is unused and in its original packaging.")

BAR = {"pass": "PASS", "flag": "FLAG", "block": "BLOCK"}


def main():
    judge = JudixRagCallback()
    print(f"question: {QUESTION}")
    print(f"answer:   {ANSWER}")
    print(f"checking live through {judge.client.url}\n")

    report = judge.check(QUESTION, CONTEXTS, ANSWER)

    print(f"rag_quality {report.quality:.1f}  {report.band}  action {BAR[report.action]}")
    print(f"{report.reason_line()}\n")

    print("claims, grounded against the sources:")
    ans = ANSWER
    for s in sorted(report.spans, key=lambda x: x.get("start", 0)):
        state = "CONTRADICTED" if s.get("contradicted") else ("grounded" if s.get("supported") else "unsupported")
        quote = ans[s.get("start", 0):s.get("end", 0)] or s.get("text", "")
        print(f"  [{state:<12}] {quote}")

    print("\nmetrics:")
    for m in report.metrics:
        lc = "  (escalated to strong model)" if m.low_confidence else ""
        print(f"  {m.name:<20} {m.score:>5.0f}{lc}")

    if report.action == "block":
        print("\nBlock or correct this answer before the user reads it.")


if __name__ == "__main__":
    main()
