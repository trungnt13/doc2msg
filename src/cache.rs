use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::chunker::DocumentOutput;

/// Default maximum total byte size for the cache (512 MB).
const DEFAULT_MAX_BYTES: usize = 512 * 1024 * 1024;

/// Trait for caching extraction results.
///
/// `get` returns `Arc<DocumentOutput>` to avoid deep-cloning cached values on
/// the hot path. Callers that only need a reference (e.g. JSON serialization)
/// pay zero allocation cost.
pub trait Cache: Send + Sync {
    fn get(&self, key: &str) -> Option<Arc<DocumentOutput>>;
    fn put(&self, key: String, value: DocumentOutput);
}

#[derive(Debug)]
struct CacheEntry {
    value: Arc<DocumentOutput>,
    last_touched: AtomicU64,
    entry_size: usize,
}

/// Simple in-memory cache with LRU-ish eviction bounded by both entry count
/// and total byte size.
pub struct InMemoryCache {
    store: RwLock<HashMap<String, Arc<CacheEntry>>>,
    max_entries: usize,
    max_bytes: usize,
    current_bytes: AtomicU64,
    clock: AtomicU64,
}

impl InMemoryCache {
    pub fn new(max_entries: usize) -> Self {
        Self::with_limits(max_entries, DEFAULT_MAX_BYTES)
    }

    pub fn with_limits(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            max_entries,
            max_bytes,
            current_bytes: AtomicU64::new(0),
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

    pub fn current_bytes(&self) -> u64 {
        self.current_bytes.load(Ordering::Relaxed)
    }

    /// Evict the single least-recently-used entry. Returns `true` if an entry
    /// was removed.
    fn evict_one(store: &mut HashMap<String, Arc<CacheEntry>>, current_bytes: &AtomicU64) -> bool {
        let oldest_key = store
            .iter()
            .min_by_key(|(_, entry)| entry.last_touched.load(Ordering::Relaxed))
            .map(|(key, _)| key.clone());

        if let Some(oldest_key) = oldest_key {
            if let Some(removed) = store.remove(&oldest_key) {
                current_bytes.fetch_sub(removed.entry_size as u64, Ordering::Relaxed);
            }
            true
        } else {
            false
        }
    }
}

impl Cache for InMemoryCache {
    fn get(&self, key: &str) -> Option<Arc<DocumentOutput>> {
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
        Some(Arc::clone(&entry.value))
    }

    fn put(&self, key: String, value: DocumentOutput) {
        if self.max_entries == 0 {
            return;
        }

        let entry_size = value.estimated_size() + key.len();
        let tick = self.next_tick();
        let entry = Arc::new(CacheEntry {
            value: Arc::new(value),
            last_touched: AtomicU64::new(tick),
            entry_size,
        });

        let mut store = self.write_store();

        // If replacing an existing key, subtract old size first.
        if let Some(old) = store.remove(&key) {
            self.current_bytes
                .fetch_sub(old.entry_size as u64, Ordering::Relaxed);
        }

        self.current_bytes
            .fetch_add(entry_size as u64, Ordering::Relaxed);
        store.insert(key, entry);

        // Evict until both entry-count and byte-size limits are satisfied.
        while store.len() > self.max_entries
            || self.current_bytes.load(Ordering::Relaxed) > self.max_bytes as u64
        {
            if !Self::evict_one(&mut store, &self.current_bytes) {
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
        assert_eq!(
            cached.map(|item| item.title.clone()).flatten(),
            Some("one".to_string())
        );
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

    #[test]
    fn evicts_when_byte_limit_exceeded() {
        let big_text = "x".repeat(1000);
        let entry_size = sample_output("t", &big_text).estimated_size() + "key0".len();

        // Allow room for exactly 2 entries by byte budget but generous entry count.
        let max_bytes = entry_size * 2;
        let cache = InMemoryCache::with_limits(100, max_bytes);

        cache.put("k1".to_string(), sample_output("t1", &big_text));
        cache.put("k2".to_string(), sample_output("t2", &big_text));
        assert_eq!(cache.len(), 2);

        // Third insert should trigger byte-based eviction.
        cache.put("k3".to_string(), sample_output("t3", &big_text));
        assert!(
            cache.len() <= 2,
            "Expected at most 2 entries after byte eviction, got {}",
            cache.len()
        );
        assert!(
            cache.current_bytes() <= max_bytes as u64,
            "current_bytes {} exceeds max_bytes {}",
            cache.current_bytes(),
            max_bytes
        );
    }

    #[test]
    fn current_bytes_tracks_insertions_and_evictions() {
        let cache = InMemoryCache::with_limits(2, usize::MAX);
        assert_eq!(cache.current_bytes(), 0);

        cache.put("a".to_string(), sample_output("a", "hello"));
        let after_one = cache.current_bytes();
        assert!(after_one > 0);

        cache.put("b".to_string(), sample_output("b", "world"));
        let after_two = cache.current_bytes();
        assert!(after_two > after_one);

        // Third insert evicts one entry; should have exactly 2 entries worth,
        // not three.
        cache.put("c".to_string(), sample_output("c", "third"));
        assert_eq!(cache.len(), 2);
        assert!(
            cache.current_bytes() < after_one + after_two,
            "bytes should not accumulate beyond two entries"
        );
    }

    #[test]
    fn replacing_existing_key_updates_byte_count() {
        let cache = InMemoryCache::with_limits(10, usize::MAX);

        cache.put("k".to_string(), sample_output("small", "hi"));
        let bytes_small = cache.current_bytes();

        let big_text = "x".repeat(5000);
        cache.put("k".to_string(), sample_output("big", &big_text));
        let bytes_big = cache.current_bytes();

        assert!(
            bytes_big > bytes_small,
            "Replacing with larger value should increase bytes"
        );
        assert_eq!(cache.len(), 1, "Should still be a single entry");
    }

    /// Intent: Overwriting an existing cache key updates the value and adjusts byte tracking.
    #[test]
    fn overwrite_existing_key_updates_value() {
        let cache = InMemoryCache::new(10);
        cache.put("key".to_string(), sample_output("first", "hello"));
        cache.put("key".to_string(), sample_output("second", "world"));

        let result = cache.get("key");
        assert!(result.is_some());
        assert_eq!(result.unwrap().title, Some("second".to_string()),);
        assert_eq!(cache.len(), 1, "overwrite should not increase entry count");
    }

    /// Intent: Cache correctly handles concurrent reads and writes without data corruption.
    #[test]
    fn concurrent_access_no_corruption() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(InMemoryCache::new(100));
        let mut handles = vec![];

        // Spawn writers
        for i in 0..10 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..50 {
                    let key = format!("key-{i}-{j}");
                    cache.put(key, sample_output("title", &format!("text-{i}-{j}")));
                }
            }));
        }

        // Spawn readers
        for i in 0..10 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..50 {
                    let _ = cache.get(&format!("key-{i}-{j}"));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should not panic or corrupt — just verify it's accessible
        assert!(cache.len() <= 100, "cache should respect entry limit");
    }
}
