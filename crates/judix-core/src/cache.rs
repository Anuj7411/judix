use moka::future::Cache;
use sha2::{Digest, Sha256};
use std::time::Duration;

/// Response cache for model calls (§8). Keyed by SHA-256 of
/// `(check_name + model + normalized_input)` with a 1-hour TTL, so identical
/// checks on identical inputs never pay twice. A cache hit costs $0.
#[derive(Clone)]
pub struct ModelCache {
    inner: Cache<String, String>,
}

impl ModelCache {
    pub fn new(max_entries: u64, ttl_secs: u64) -> Self {
        Self {
            inner: Cache::builder()
                .max_capacity(max_entries)
                .time_to_live(Duration::from_secs(ttl_secs))
                .build(),
        }
    }

    /// Collapse runs of whitespace so trivially-different formatting of the same
    /// logical input still hits the cache.
    fn normalize(input: &str) -> String {
        input.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn key(check: &str, model: &str, input: &str) -> String {
        let mut h = Sha256::new();
        h.update(check.as_bytes());
        h.update(b"|");
        h.update(model.as_bytes());
        h.update(b"|");
        h.update(Self::normalize(input).as_bytes());
        format!("{:x}", h.finalize())
    }

    pub async fn get(&self, check: &str, model: &str, input: &str) -> Option<String> {
        self.inner.get(&Self::key(check, model, input)).await
    }

    pub async fn insert(&self, check: &str, model: &str, input: &str, response: String) {
        self.inner
            .insert(Self::key(check, model, input), response)
            .await;
    }
}
