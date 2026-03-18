use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::chunker::DocumentOutput;

/// Trait for caching extraction results.
pub trait Cache: Send + Sync {
    fn get(&self, key: &str) -> Option<DocumentOutput>;
    fn put(&self, key: String, value: DocumentOutput);
}

#[derive(Debug)]
struct CacheEntry {
    value: Arc<DocumentOutput>,
    last_touched: AtomicU64,
}

/// Simple in-memory cache with LRU-ish eviction.
pub struct InMemoryCache {
    store: RwLock<HashMap<String, Arc<CacheEntry>>>,
    max_entries: usize,
    clock: AtomicU64,
}

impl InMemoryCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            max_entries,
            clock: AtomicU64::new(0),
        }
    }

    fn next_tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed) + 1
    }

    fn read_store(&self) -> std::sync::RwLockReadGuard<'_, HashMap<String, Arc<CacheEntry>>> {
        match self.store.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn write_store(&self) -> std::sync::RwLockWriteGuard<'_, HashMap<String, Arc<CacheEntry>>> {
        match self.store.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    pub fn len(&self) -> usize {
        self.read_store().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Cache for InMemoryCache {
    fn get(&self, key: &str) -> Option<DocumentOutput> {
        if self.max_entries == 0 {
            return None;
        }

        let entry = {
            let store = self.read_store();
            store.get(key).cloned()
        }?;

        entry
            .last_touched
            .store(self.next_tick(), Ordering::Relaxed);
        Some(entry.value.as_ref().clone())
    }

    fn put(&self, key: String, value: DocumentOutput) {
        if self.max_entries == 0 {
            return;
        }

        let tick = self.next_tick();
        let entry = Arc::new(CacheEntry {
            value: Arc::new(value),
            last_touched: AtomicU64::new(tick),
        });

        let mut store = self.write_store();
        store.insert(key, entry);

        while store.len() > self.max_entries {
            let oldest_key = store
                .iter()
                .min_by_key(|(_, entry)| entry.last_touched.load(Ordering::Relaxed))
                .map(|(key, _)| key.clone());

            if let Some(oldest_key) = oldest_key {
                store.remove(&oldest_key);
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cache, InMemoryCache};
    use crate::chunker::{Chunk, DocumentMetadata, DocumentOutput, PipelineDiagnostics};

    fn sample_output(title: &str, text: &str) -> DocumentOutput {
        DocumentOutput {
            title: Some(title.to_string()),
            canonical_url: None,
            markdown: text.to_string(),
            chunks: vec![Chunk {
                id: "c01".to_string(),
                text: text.to_string(),
                section: None,
                page_start: None,
                page_end: None,
                char_count: text.len(),
                token_estimate: text.len() / 4,
            }],
            metadata: DocumentMetadata {
                page_count: None,
                word_count: text.split_whitespace().count(),
                char_count: text.len(),
            },
            diagnostics: PipelineDiagnostics {
                pipeline_used: "test".to_string(),
                ocr_used: false,
                render_used: false,
                fallback_used: false,
                fallback_reason: None,
                text_quality_score: None,
                latency_ms: 1,
            },
            image_manifest: None,
        }
    }

    #[test]
    fn returns_cached_value() {
        let cache = InMemoryCache::new(2);
        cache.put("doc".to_string(), sample_output("one", "hello world"));

        let cached = cache.get("doc");
        assert!(cached.is_some());
        assert_eq!(cached.and_then(|item| item.title), Some("one".to_string()));
    }

    #[test]
    fn evicts_least_recently_used_entry() {
        let cache = InMemoryCache::new(2);
        cache.put("a".to_string(), sample_output("a", "first"));
        cache.put("b".to_string(), sample_output("b", "second"));

        let cached_a = cache.get("a");
        assert!(cached_a.is_some());

        cache.put("c".to_string(), sample_output("c", "third"));

        assert!(
            cache.get("a").is_some(),
            "recently touched entry should stay cached"
        );
        assert!(
            cache.get("b").is_none(),
            "least recently used entry should be evicted"
        );
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn zero_capacity_disables_storage() {
        let cache = InMemoryCache::new(0);
        cache.put("doc".to_string(), sample_output("one", "hello world"));

        assert!(cache.is_empty());
        assert!(cache.get("doc").is_none());
    }
}
