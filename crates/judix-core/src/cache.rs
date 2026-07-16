use moka::future::Cache;
use sha2::{Digest, Sha256};
use std::time::Duration;

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

    fn key(model: &str, prompt: &str) -> String {
        let mut h = Sha256::new();
        h.update(model.as_bytes());
        h.update(b"|");
        h.update(prompt.as_bytes());
        format!("{:x}", h.finalize())
    }

    pub async fn get(&self, model: &str, prompt: &str) -> Option<String> {
        self.inner.get(&Self::key(model, prompt)).await
    }

    pub async fn insert(&self, model: &str, prompt: &str, response: String) {
        self.inner.insert(Self::key(model, prompt), response).await;
    }
}
