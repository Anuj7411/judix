//! Judix CLI — score an agent trace from the terminal with the **deterministic engine
//! only** (no model, no key, $0). This is the offline half of the product: tool-call
//! correctness (F1), loop detection, and schema validation, exactly as the server
//! computes them before any model call.
//!
//! RAG scoring is model-powered (claim decomposition + grounding) and has no
//! deterministic component, so it lives on the server (`POST /score/rag`) — the CLI
//! detects a RAG triple and says so rather than failing on a missing field.
//!
//! Usage: `judix <path-to-trace.json>`

use judix_core::scoring::score_agent;
use judix_core::types::AgentTrace;
use serde_json::Value;

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: judix <trace.json>   (an agent trace: {{goal, steps, ...}})");
            std::process::exit(2);
        }
    };

    let raw = match std::fs::read_to_string(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error reading {path}: {e}");
            std::process::exit(1);
        }
    };

    // Parse loosely first so we can branch on shape and give a useful message, instead
    // of deserializing straight into AgentTrace and dying on "missing field `goal`" when
    // handed a RAG triple.
    let value: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {path} is not valid JSON: {e}");
            std::process::exit(1);
        }
    };

    // A RAG triple ({question, contexts, answer}) has no deterministic metrics — every
    // RAG score comes from the model layer — so the CLI honestly can't score it. Point
    // the user at the server rather than erroring on the shape mismatch.
    let is_rag = value.get("answer").is_some() && value.get("contexts").is_some()
        || value.get("type").and_then(Value::as_str) == Some("rag");
    if is_rag {
        eprintln!(
            "This looks like a RAG triple. RAG faithfulness is model-powered and has no\n\
             deterministic score, so it runs on the server, not the CLI:\n\
             \n\
             \x20   curl -X POST http://localhost:8000/score/rag \\\n\
             \x20        -H 'content-type: application/json' --data-binary @{path}\n"
        );
        std::process::exit(2);
    }

    let trace: AgentTrace = match serde_json::from_value(value) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "error: expected an agent trace (an object with `goal` and `steps`): {e}"
            );
            std::process::exit(1);
        }
    };

    // Deterministic-only run: no model metrics supplied.
    let report = score_agent(&trace, &[], 0, 0.0);
    match serde_json::to_string_pretty(&report) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("error serializing report: {e}");
            std::process::exit(1);
        }
    }
}
