use std::collections::HashMap;

/// N-gram cache for assisted drafting.
#[derive(Debug, Clone)]
pub struct NgramCache {
    /// N-gram to next token mapping: (context_hash) -> [(token_id, count)].
    cache: HashMap<u64, Vec<(usize, usize)>>,
    /// N-gram size.
    n: usize,
    /// Maximum cache entries.
    max_entries: usize,
}

impl NgramCache {
    /// Create a new N-gram cache.
    #[inline(always)]
    pub fn new(n: usize, max_entries: usize) -> Self {
        Self {
            cache: HashMap::new(),
            n,
            max_entries,
        }
    }

    /// Hash a token sequence.
    #[inline(always)]
    fn hash_context(&self, tokens: &[usize]) -> u64 {
        let mut hash = 0u64;
        for (i, &token) in tokens.iter().enumerate() {
            hash = hash.wrapping_mul(31).wrapping_add(token as u64);
            hash = hash.wrapping_add((i as u64).wrapping_mul(17));
        }
        hash
    }

    /// Update cache with observed tokens.
    #[inline(always)]
    pub fn update(&mut self, tokens: &[usize]) {
        if tokens.len() < self.n {
            return;
        }

        for i in 0..=tokens.len() - self.n {
            let context = &tokens[i..i + self.n - 1];
            let next_token = tokens[i + self.n - 1];
            let hash = self.hash_context(context);

            let entry = self.cache.entry(hash).or_insert_with(Vec::new);

            // Update count for this token
            if let Some(pos) = entry.iter().position(|(t, _)| *t == next_token) {
                entry[pos].1 += 1;
            } else {
                entry.push((next_token, 1));
            }

            // Keep top entries by count
            entry.sort_by(|a, b| b.1.cmp(&a.1));
            entry.truncate(16);
        }

        // Evict if too large
        if self.cache.len() > self.max_entries {
            let keys_to_remove: Vec<_> = self.cache.keys().take(self.max_entries / 4).cloned().collect();
            for key in keys_to_remove {
                self.cache.remove(&key);
            }
        }
    }

    /// Get predicted next tokens for a context.
    #[inline(always)]
    pub fn predict(&self, context: &[usize], k: usize) -> Vec<usize> {
        if context.len() < self.n - 1 {
            return Vec::new();
        }

        let recent_context = &context[context.len() - (self.n - 1)..];
        let hash = self.hash_context(recent_context);

        match self.cache.get(&hash) {
            Some(predictions) => predictions.iter().take(k).map(|(t, _)| *t).collect(),
            None => Vec::new(),
        }
    }

    /// Get cache size.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Clear the cache.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
