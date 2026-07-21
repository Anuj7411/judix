//! Judix CLI — score an agent trace from the terminal with the **deterministic engine
//! only** (no model, no key, $0). This is the offline half of the product: tool-call
//! correctness (F1), loop detection, and schema validation, exactly as the server
//! computes them before any model call.
//!
//! RAG scoring is model-powered (claim decomposition + grounding) and has no
//! deterministic component, so it lives on the server (`POST /score/rag`) — the CLI
//! detects a RAG triple and says so rather than failing on a missing field.
//!
//! Input comes from a file path, from `-`, or from a pipe on stdin, so a fresh
//! `cargo install` can score a hosted demo with no local files:
//!
//!   curl -s https://judix-8piu.onrender.com/demo/wrong_tool | judix

use judix_core::scoring::score_agent;
use judix_core::types::AgentTrace;
use serde_json::Value;
use std::io::{IsTerminal, Read};

const HELP: &str = "\
judix — score an agent trace with the deterministic engine (no model, no key, $0).

usage:
  judix <trace.json>        score a trace file
  judix -                   score a trace read from stdin
  <producer> | judix        score a trace piped in

try it right now, no local files needed:
  curl -s https://judix-8piu.onrender.com/demo/wrong_tool | judix

a trace is JSON shaped like { \"goal\": \"...\", \"steps\": [ ... ] }. It prints a
report with tool-call F1, loop detection, per-step and run quality, and the color
band. RAG triples ({question, contexts, answer}) are model-scored on the server.";

fn read_stdin() -> String {
    let mut buf = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
        eprintln!("error reading stdin: {e}");
        std::process::exit(1);
    }
    buf
}

fn main() {
    // Source resolution: an explicit path, `-` for stdin, or a bare invocation that
    // reads stdin when something is piped in and otherwise prints help. This is what
    // lets `curl .../demo/wrong_tool | judix` work straight after a global install.
    let arg = std::env::args().nth(1);
    let (raw, source) = match arg.as_deref() {
        Some("-h") | Some("--help") => {
            println!("{HELP}");
            return;
        }
        Some("-") => (read_stdin(), "-".to_string()),
        Some(path) => match std::fs::read_to_string(path) {
            Ok(r) => (r, path.to_string()),
            Err(e) => {
                eprintln!("error reading {path}: {e}\n\n{HELP}");
                std::process::exit(1);
            }
        },
        None => {
            if std::io::stdin().is_terminal() {
                // No file and nothing piped in — the user just typed `judix`. Show how.
                eprintln!("{HELP}");
                std::process::exit(2);
            }
            (read_stdin(), "-".to_string())
        }
    };

    // Empty input (a bare `judix` with nothing piped, or `judix < /dev/null`) should
    // teach, not emit a cryptic "EOF while parsing" JSON error.
    if raw.trim().is_empty() {
        eprintln!("no trace to score.\n\n{HELP}");
        std::process::exit(2);
    }

    // Parse loosely first so we can branch on shape and give a useful message, instead
    // of deserializing straight into AgentTrace and dying on "missing field `goal`" when
    // handed a RAG triple.
    let value: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: input ({source}) is not valid JSON: {e}");
            std::process::exit(1);
        }
    };

    // A RAG triple ({question, contexts, answer}) has no deterministic metrics — every
    // RAG score comes from the model layer — so the CLI honestly can't score it. Point
    // the user at the server rather than erroring on the shape mismatch.
    let is_rag = value.get("answer").is_some() && value.get("contexts").is_some()
        || value.get("type").and_then(Value::as_str) == Some("rag");
    if is_rag {
        let data_ref = if source == "-" { "@-".to_string() } else { format!("@{source}") };
        eprintln!(
            "This looks like a RAG triple. RAG faithfulness is model-powered and has no\n\
             deterministic score, so it runs on the server, not the CLI:\n\
             \n\
             \x20   curl -X POST https://judix-8piu.onrender.com/score/rag \\\n\
             \x20        -H 'content-type: application/json' --data-binary {data_ref}\n"
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
