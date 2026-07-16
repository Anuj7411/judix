//! Judix CLI — score an agent trace from the terminal using the deterministic
//! engine only (no model, no key). Day-1 minimal form; a fuller `clap` interface
//! with RAG support lands in Day 3 (§11.9).
//!
//! Usage: `judix <path-to-agent-trace.json>`

use judix_core::scoring::score_agent;
use judix_core::types::AgentTrace;

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: judix <agent-trace.json>");
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

    let trace: AgentTrace = match serde_json::from_str(&raw) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error parsing agent trace: {e}");
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
