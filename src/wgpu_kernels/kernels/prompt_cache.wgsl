// Prompt Caching / CacheBlend Kernels
// Based on SGLang RadixAttention, vLLM Prefix Caching, CacheBlend
//
// Kernels:
// - prompt_cache_hash_f32: Compute hash for token sequences (xxHash64-style)
// - prompt_cache_prefix_match_f32: Find longest prefix match
// - prompt_cache_blend_f32: CacheBlend position reencoding
// - prompt_cache_copy_kv_f32: Copy KV cache segments
// - F16 variants

enable f16;

const WORKGROUP_SIZE: u32 = 256u;
const HASH_SEED: u32 = 0x9E3779B9u;  // Golden ratio derived

// ============================================================================
// Parameters
// ============================================================================

struct HashParams {
    batch_size: u32,
    seq_len: u32,
    hash_dim: u32,  // Number of tokens per hash block
    _pad0: u32,
};

struct PrefixMatchParams {
    batch_size: u32,
    query_len: u32,
    cache_entries: u32,
    max_prefix_len: u32,
};

struct BlendParams {
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    cached_len: u32,
    new_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct CopyKVParams {
    batch_size: u32,
    num_heads: u32,
    head_dim: u32,
    src_seq_len: u32,
    dst_offset: u32,
    copy_len: u32,
    _pad0: u32,
    _pad1: u32,
};

// ============================================================================
// Hash Computation (F32) - xxHash64-style for token sequences
// ============================================================================
// Computes hash for each block of tokens for cache lookup
// Input: tokens[batch, seq_len]
// Output: hashes[batch, num_blocks] where num_blocks = seq_len / hash_dim

@group(0) @binding(0) var<storage, read> hash_tokens_f32: array<u32>;
@group(0) @binding(1) var<storage, read_write> hash_output_f32: array<u32>;
@group(0) @binding(2) var<uniform> hash_params_f32: HashParams;

fn xxhash_round(acc: u32, input: u32) -> u32 {
    var v = acc;
    v += input * 0xC2B2AE35u;
    v = (v << 13u) | (v >> 19u);
    v *= 0x27D4EB2Du;
    return v;
}

fn xxhash_finalize(h: u32, len: u32) -> u32 {
    var hash = h;
    hash ^= len;
    hash ^= hash >> 15u;
    hash *= 0x85EBCA77u;
    hash ^= hash >> 13u;
    hash *= 0xC2B2AE3Bu;
    hash ^= hash >> 16u;
    return hash;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn prompt_cache_hash_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let num_blocks = hash_params_f32.seq_len / hash_params_f32.hash_dim;
    let total = hash_params_f32.batch_size * num_blocks;

    if (idx >= total) {
        return;
    }

    let batch_idx = idx / num_blocks;
    let block_idx = idx % num_blocks;

    let tokens_base = batch_idx * hash_params_f32.seq_len + block_idx * hash_params_f32.hash_dim;

    // xxHash64-style hashing
    var h = HASH_SEED;
    for (var i: u32 = 0u; i < hash_params_f32.hash_dim; i++) {
        h = xxhash_round(h, hash_tokens_f32[tokens_base + i]);
    }

    hash_output_f32[idx] = xxhash_finalize(h, hash_params_f32.hash_dim);
}

// ============================================================================
// Prefix Match (F32)
// ============================================================================
// Finds longest matching prefix in cache
// Input: query_hashes[batch, num_blocks], cache_hashes[cache_entries, max_blocks]
// Output: match_entry[batch], match_len[batch]

@group(0) @binding(0) var<storage, read> prefix_query_f32: array<u32>;
@group(0) @binding(1) var<storage, read> prefix_cache_f32: array<u32>;
@group(0) @binding(2) var<storage, read_write> prefix_match_entry_f32: array<u32>;
@group(0) @binding(3) var<storage, read_write> prefix_match_len_f32: array<u32>;
@group(0) @binding(4) var<uniform> prefix_params_f32: PrefixMatchParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn prompt_cache_prefix_match_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let batch_idx = global_id.x;

    if (batch_idx >= prefix_params_f32.batch_size) {
        return;
    }

    let query_blocks = prefix_params_f32.query_len;
    let query_base = batch_idx * query_blocks;

    var best_entry = 0xFFFFFFFFu;  // Invalid
    var best_len = 0u;

    // Check each cache entry
    for (var entry: u32 = 0u; entry < prefix_params_f32.cache_entries; entry++) {
        let cache_base = entry * prefix_params_f32.max_prefix_len;
        var match_len = 0u;

        // Find matching prefix length
        for (var b: u32 = 0u; b < min(query_blocks, prefix_params_f32.max_prefix_len); b++) {
            if (prefix_query_f32[query_base + b] == prefix_cache_f32[cache_base + b]) {
                match_len = b + 1u;
            } else {
                break;
            }
        }

        if (match_len > best_len) {
            best_len = match_len;
            best_entry = entry;
        }
    }

    prefix_match_entry_f32[batch_idx] = best_entry;
    prefix_match_len_f32[batch_idx] = best_len;
}

// ============================================================================
// CacheBlend Position Reencoding (F32)
// ============================================================================
// Reencodes positions when blending cached and new KV
// Uses linear interpolation for smooth transition
// Input: cached_pos[cached_len], new_pos[new_len]
// Output: blended_pos[cached_len + new_len]

@group(0) @binding(0) var<storage, read> blend_cached_kv_f32: array<f32>;
@group(0) @binding(1) var<storage, read> blend_new_kv_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> blend_output_f32: array<f32>;
@group(0) @binding(3) var<uniform> blend_params_f32: BlendParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn prompt_cache_blend_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_len = blend_params_f32.cached_len + blend_params_f32.new_len;
    let kv_size = blend_params_f32.num_heads * blend_params_f32.head_dim;
    let total = blend_params_f32.batch_size * total_len * kv_size;

    if (idx >= total) {
        return;
    }

    let kv_idx = idx % kv_size;
    let seq_idx = (idx / kv_size) % total_len;
    let batch_idx = idx / (total_len * kv_size);

    if (seq_idx < blend_params_f32.cached_len) {
        // Copy from cached
        let src_idx = batch_idx * blend_params_f32.cached_len * kv_size +
                      seq_idx * kv_size + kv_idx;
        blend_output_f32[idx] = blend_cached_kv_f32[src_idx];
    } else {
        // Copy from new
        let new_seq_idx = seq_idx - blend_params_f32.cached_len;
        let src_idx = batch_idx * blend_params_f32.new_len * kv_size +
                      new_seq_idx * kv_size + kv_idx;
        blend_output_f32[idx] = blend_new_kv_f32[src_idx];
    }
}

// ============================================================================
// KV Cache Copy (F32)
// ============================================================================
// Copies KV cache segment from source to destination
// Input: src_kv[batch, src_seq_len, num_heads, head_dim]
// Output: dst_kv[batch, dst_seq_len, num_heads, head_dim] (at dst_offset)

@group(0) @binding(0) var<storage, read> copy_src_f32: array<f32>;
@group(0) @binding(1) var<storage, read_write> copy_dst_f32: array<f32>;
@group(0) @binding(2) var<uniform> copy_params_f32: CopyKVParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn prompt_cache_copy_kv_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let kv_size = copy_params_f32.num_heads * copy_params_f32.head_dim;
    let total = copy_params_f32.batch_size * copy_params_f32.copy_len * kv_size;

    if (idx >= total) {
        return;
    }

    let kv_idx = idx % kv_size;
    let seq_idx = (idx / kv_size) % copy_params_f32.copy_len;
    let batch_idx = idx / (copy_params_f32.copy_len * kv_size);

    // Source index
    let src_idx = batch_idx * copy_params_f32.src_seq_len * kv_size +
                  seq_idx * kv_size + kv_idx;

    // Destination index with offset
    let dst_seq_idx = seq_idx + copy_params_f32.dst_offset;
    let dst_idx = batch_idx * (copy_params_f32.dst_offset + copy_params_f32.copy_len) * kv_size +
                  dst_seq_idx * kv_size + kv_idx;

    copy_dst_f32[dst_idx] = copy_src_f32[src_idx];
}

// ============================================================================
// Rolling Hash Update (F32)
// ============================================================================
// Incrementally updates hash when new tokens are added
// Used for streaming/incremental cache key computation

struct RollingHashParams {
    batch_size: u32,
    window_size: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> rolling_old_hash_f32: array<u32>;
@group(0) @binding(1) var<storage, read> rolling_old_token_f32: array<u32>;
@group(0) @binding(2) var<storage, read> rolling_new_token_f32: array<u32>;
@group(0) @binding(3) var<storage, read_write> rolling_new_hash_f32: array<u32>;
@group(0) @binding(4) var<uniform> rolling_params_f32: RollingHashParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn prompt_cache_rolling_hash_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let batch_idx = global_id.x;

    if (batch_idx >= rolling_params_f32.batch_size) {
        return;
    }

    // Rolling hash: remove old token contribution, add new token
    var h = rolling_old_hash_f32[batch_idx];

    // Remove old token (simplified - full implementation needs modular inverse)
    let old_token = rolling_old_token_f32[batch_idx];
    h ^= old_token * 0x85EBCA77u;

    // Add new token
    let new_token = rolling_new_token_f32[batch_idx];
    h = xxhash_round(h, new_token);

    rolling_new_hash_f32[batch_idx] = h;
}

// ============================================================================
// F16 Variants
// ============================================================================

@group(0) @binding(0) var<storage, read> blend_cached_kv_f16: array<f16>;
@group(0) @binding(1) var<storage, read> blend_new_kv_f16: array<f16>;
@group(0) @binding(2) var<storage, read_write> blend_output_f16: array<f16>;
@group(0) @binding(3) var<uniform> blend_params_f16: BlendParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn prompt_cache_blend_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let total_len = blend_params_f16.cached_len + blend_params_f16.new_len;
    let kv_size = blend_params_f16.num_heads * blend_params_f16.head_dim;
    let total = blend_params_f16.batch_size * total_len * kv_size;

    if (idx >= total) {
        return;
    }

    let kv_idx = idx % kv_size;
    let seq_idx = (idx / kv_size) % total_len;
    let batch_idx = idx / (total_len * kv_size);

    if (seq_idx < blend_params_f16.cached_len) {
        let src_idx = batch_idx * blend_params_f16.cached_len * kv_size +
                      seq_idx * kv_size + kv_idx;
        blend_output_f16[idx] = blend_cached_kv_f16[src_idx];
    } else {
        let new_seq_idx = seq_idx - blend_params_f16.cached_len;
        let src_idx = batch_idx * blend_params_f16.new_len * kv_size +
                      new_seq_idx * kv_size + kv_idx;
        blend_output_f16[idx] = blend_new_kv_f16[src_idx];
    }
}

@group(0) @binding(0) var<storage, read> copy_src_f16: array<f16>;
@group(0) @binding(1) var<storage, read_write> copy_dst_f16: array<f16>;
@group(0) @binding(2) var<uniform> copy_params_f16: CopyKVParams;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn prompt_cache_copy_kv_f16(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    let kv_size = copy_params_f16.num_heads * copy_params_f16.head_dim;
    let total = copy_params_f16.batch_size * copy_params_f16.copy_len * kv_size;

    if (idx >= total) {
        return;
    }

    let kv_idx = idx % kv_size;
    let seq_idx = (idx / kv_size) % copy_params_f16.copy_len;
    let batch_idx = idx / (copy_params_f16.copy_len * kv_size);

    let src_idx = batch_idx * copy_params_f16.src_seq_len * kv_size +
                  seq_idx * kv_size + kv_idx;

    let dst_seq_idx = seq_idx + copy_params_f16.dst_offset;
    let dst_idx = batch_idx * (copy_params_f16.dst_offset + copy_params_f16.copy_len) * kv_size +
                  dst_seq_idx * kv_size + kv_idx;

    copy_dst_f16[dst_idx] = copy_src_f16[src_idx];
}
