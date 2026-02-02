#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

#define GLLM_DEVICE __device__ __forceinline__

struct EmbeddingConfig { uint32_t vocab_size, hidden_size, seq_len, stride; };
struct LinearConfig { uint32_t m, n, k, input_stride, weight_stride, output_stride, use_bias; };
struct AddConfig { uint32_t numel; };
struct RmsNormConfig { uint32_t hidden_size, stride; float eps; uint32_t seq_len; };
struct SiluConfig { uint32_t numel; };
struct SwiGluConfig { uint32_t numel; };
struct RopeConfig { uint32_t seq_len, head_dim, rotary_dim; float base, scale; uint32_t interleaved, position_stride, precompute_max_seq_len; };
struct FusedQkvRopeConfig { uint32_t batch, seq_len, num_heads, head_dim, rotary_dim, input_stride, qkv_stride; float base, scale; uint32_t interleaved, precompute_max_seq_len; };
struct FlashAttnConfig { uint32_t batch, num_heads, head_dim, q_seq_len, kv_seq_len, q_stride, kv_stride, o_stride, causal; float scale; uint32_t q_pos_offset; };
struct FlashAttnPagedConfig { uint32_t batch, num_heads, head_dim, q_seq_len, kv_seq_len, q_stride, kv_stride, o_stride, causal; float scale; uint32_t q_pos_offset, page_size, pages_per_layer; };
struct SamplingKernelConfig { uint32_t vocab_size, top_k; float top_p, temperature; uint32_t stride, batch; };
struct QuantizedConfig { uint32_t m, n, k, input_stride, weight_stride, output_stride; float scale; };

GLLM_DEVICE float sigmoidf_fast(float x) { return 1.0f / (1.0f + expf(-x)); }
GLLM_DEVICE float silu(float x) { return x * sigmoidf_fast(x); }
GLLM_DEVICE float dotf(const float* a, const float* b, uint32_t n) { float sum = 0.0f; for (uint32_t i = 0; i < n; ++i) sum = fmaf(a[i], b[i], sum); return sum; }
GLLM_DEVICE float rope_inv_freq(uint32_t idx, uint32_t rotary_dim, float base) { return powf(base, -2.0f * ((float)idx) / (float)rotary_dim); }
GLLM_DEVICE void rope_apply(float* x0, float* x1, float c, float s) { float v0 = *x0, v1 = *x1; *x0 = v0 * c - v1 * s; *x1 = v0 * s + v1 * c; }
GLLM_DEVICE uint32_t xorshift32(uint32_t x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
GLLM_DEVICE float rand_uniform(uint64_t seed) { uint32_t v = (uint32_t)(seed ^ (seed >> 32)); v = xorshift32(v); return (v + 1.0f) * 2.3283064365386963e-10f; }
GLLM_DEVICE int8_t int4_to_i8(uint8_t nibble) { nibble &= 0x0f; return (nibble & 0x08) ? (int8_t)(nibble | 0xf0) : (int8_t)nibble; }

template<int BITS>
struct QuantizedTraits;

template<>
struct QuantizedTraits<8> {
    static constexpr int VALUES_PER_BYTE = 1;
    GLLM_DEVICE static float decode(uint8_t packed, int /*index*/) {
        return static_cast<float>(static_cast<int8_t>(packed));
    }
};

template<>
struct QuantizedTraits<4> {
    static constexpr int VALUES_PER_BYTE = 2;
    GLLM_DEVICE static float decode(uint8_t packed, int index) {
        uint8_t nibble = (index & 1) ? (packed >> 4) : (packed & 0x0f);
        return static_cast<float>(int4_to_i8(nibble));
    }
};

template<>
struct QuantizedTraits<2> {
    static constexpr int VALUES_PER_BYTE = 4;
    GLLM_DEVICE static float decode(uint8_t packed, int index) {
        uint8_t pair = (packed >> ((index & 3) * 2)) & 0x03;
        int8_t value = (pair & 0x02) ? static_cast<int8_t>(pair) - 4 : static_cast<int8_t>(pair);
        return static_cast<float>(value);
    }
};

template<>
struct QuantizedTraits<1> {
    static constexpr int VALUES_PER_BYTE = 8;
    GLLM_DEVICE static float decode(uint8_t packed, int index) {
        uint8_t bit = (packed >> (index & 7)) & 0x01;
        return bit ? 1.0f : -1.0f;
    }
};

extern "C" __global__ void embedding_lookup(const uint32_t* tokens, const float* embedding, float* output, EmbeddingConfig params) {
    uint32_t total = params.seq_len * params.hidden_size, idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < total; i += stride) {
        uint32_t row = i / params.hidden_size, col = i - row * params.hidden_size, token = tokens[row];
        float val = token < params.vocab_size ? embedding[token * params.stride + col] : 0.0f;
        output[row * params.stride + col] = val;
    }
}

extern "C" __global__ void linear(const float* input, const float* weight, float* output, const float* bias, LinearConfig params) {
    uint32_t total = params.m * params.n, idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < total; i += stride) {
        uint32_t row = i / params.n, col = i - row * params.n;
        float sum = dotf(input + row * params.input_stride, weight + col * params.weight_stride, params.k);
        if (bias && params.use_bias) sum += bias[col];
        output[row * params.output_stride + col] = sum;
    }
}

template<int BITS>
__global__ void quantized_mm(const float* input, const uint8_t* weight, float* output, QuantizedConfig params) {
    constexpr int VPB = QuantizedTraits<BITS>::VALUES_PER_BYTE;
    uint32_t total = params.m * params.n, idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < total; i += stride) {
        uint32_t row = i / params.n, col = i - row * params.n;
        const float* in_row = input + row * params.input_stride;
        const uint8_t* w_row = weight + col * params.weight_stride;
        float sum = 0.0f;
        for (uint32_t kk = 0; kk < params.k; ++kk) {
            uint8_t packed = w_row[kk / VPB];
            float w = QuantizedTraits<BITS>::decode(packed, (int)kk) * params.scale;
            sum = fmaf(in_row[kk], w, sum);
        }
        output[row * params.output_stride + col] = sum;
    }
}

template __global__ void quantized_mm<1>(const float*, const uint8_t*, float*, QuantizedConfig);
template __global__ void quantized_mm<2>(const float*, const uint8_t*, float*, QuantizedConfig);
template __global__ void quantized_mm<4>(const float*, const uint8_t*, float*, QuantizedConfig);
template __global__ void quantized_mm<8>(const float*, const uint8_t*, float*, QuantizedConfig);

extern "C" __global__ void add(const float* lhs, const float* rhs, float* output, AddConfig params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < params.numel; i += stride) output[i] = lhs[i] + rhs[i];
}

extern "C" __global__ void rms_norm(const float* input, const float* weight, float* output, RmsNormConfig params) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= params.seq_len || params.hidden_size == 0) return;
    const float* in_row = input + row * params.stride; float* out_row = output + row * params.stride;
    float sum = 0.0f; for (uint32_t i = 0; i < params.hidden_size; ++i) { float v = in_row[i]; sum += v * v; }
    float inv = rsqrtf(sum / (float)params.hidden_size + params.eps);
    for (uint32_t i = 0; i < params.hidden_size; ++i) out_row[i] = in_row[i] * inv * weight[i];
}

extern "C" __global__ void silu(const float* input, float* output, SiluConfig params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < params.numel; i += stride) output[i] = silu(input[i]);
}

extern "C" __global__ void fused_gate_up_silu(const float* gate, const float* up, float* output, SwiGluConfig params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    for (uint32_t i = idx; i < params.numel; i += stride) output[i] = gate[i] * silu(up[i]);
}

extern "C" __global__ void rope(float* q, float* k, const int32_t* positions, const float* cos_table, const float* sin_table, RopeConfig params) {
    if (params.head_dim == 0 || params.seq_len == 0 || params.position_stride == 0) return;
    uint32_t rotary = params.rotary_dim > params.head_dim ? params.head_dim : params.rotary_dim; if (rotary == 0) return;
    uint32_t total = params.seq_len * params.position_stride, idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    bool inter = params.interleaved != 0; uint32_t half = rotary >> 1;
    bool use_table = cos_table && sin_table && params.precompute_max_seq_len > 0;
    for (uint32_t i = idx; i < total; i += stride) {
        uint32_t token = i / params.position_stride, offset = i - token * params.position_stride, dim = offset % params.head_dim;
        if (dim >= rotary) continue;
        uint32_t pair = dim, pair_idx = 0;
        if (inter) { if (dim & 1) continue; pair = dim + 1; if (pair >= rotary) continue; pair_idx = dim >> 1; }
        else { if (dim >= half) continue; pair = dim + half; if (pair >= rotary) continue; pair_idx = dim; }
        uint32_t head = offset / params.head_dim, base = token * params.position_stride + head * params.head_dim;
        int32_t pos = positions ? positions[token] : (int32_t)token;
        float c = 0.0f, s = 0.0f;
        if (use_table && pos >= 0 && (uint32_t)pos < params.precompute_max_seq_len) {
            uint32_t table_idx = (uint32_t)pos * half + pair_idx;
            c = cos_table[table_idx];
            s = sin_table[table_idx];
        } else {
            float angle = (float)pos * params.scale * rope_inv_freq(pair_idx, rotary, params.base);
            c = cosf(angle);
            s = sinf(angle);
        }
        float* q_row = q + base; float* k_row = k + base;
        float q0 = q_row[dim], q1 = q_row[pair], k0 = k_row[dim], k1 = k_row[pair];
        rope_apply(&q0, &q1, c, s); rope_apply(&k0, &k1, c, s);
        q_row[dim] = q0; q_row[pair] = q1; k_row[dim] = k0; k_row[pair] = k1;
    }
}

extern "C" __global__ void fused_qkv_rope(const float* input, const float* weight, float* qkv_out, const float* bias, const int32_t* positions, const float* cos_table, const float* sin_table, FusedQkvRopeConfig params) {
    if (params.head_dim == 0 || params.num_heads == 0 || params.seq_len == 0) return;
    uint32_t q_out = params.num_heads * params.head_dim; if (params.qkv_stride < q_out) return;
    uint32_t kv_out = (params.qkv_stride - q_out) / 2, kv_heads = kv_out / params.head_dim;
    uint32_t rotary = params.rotary_dim > params.head_dim ? params.head_dim : params.rotary_dim;
    uint32_t total = params.batch * params.seq_len * q_out, idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    bool inter = params.interleaved != 0; uint32_t half = rotary >> 1;
    bool use_table = cos_table && sin_table && params.precompute_max_seq_len > 0;
    for (uint32_t i = idx; i < total; i += stride) {
        uint32_t tmp = i, dim = tmp % params.head_dim; tmp /= params.head_dim;
        uint32_t head = tmp % params.num_heads, token = tmp / params.num_heads;
        uint32_t batch = params.seq_len ? token / params.seq_len : 0, pos_idx = params.seq_len ? token - batch * params.seq_len : 0;
        const float* in_row = input + (batch * params.seq_len + pos_idx) * params.input_stride;
        uint32_t out_base = (batch * params.seq_len + pos_idx) * params.qkv_stride;
        if (head < kv_heads) {
            uint32_t v_row = q_out + kv_out + head * params.head_dim + dim;
            float val = dotf(in_row, weight + v_row * params.input_stride, params.input_stride); if (bias) val += bias[v_row];
            qkv_out[out_base + v_row] = val;
        }
        if (rotary == 0 || dim >= rotary) {
            uint32_t q_row = head * params.head_dim + dim;
            float qv = dotf(in_row, weight + q_row * params.input_stride, params.input_stride); if (bias) qv += bias[q_row];
            qkv_out[out_base + q_row] = qv;
            if (head < kv_heads) {
                uint32_t k_row = q_out + head * params.head_dim + dim;
                float kv = dotf(in_row, weight + k_row * params.input_stride, params.input_stride); if (bias) kv += bias[k_row];
                qkv_out[out_base + k_row] = kv;
            }
            continue;
        }
        uint32_t pair = dim, pair_idx = 0;
        if (inter) { if (dim & 1) continue; pair = dim + 1; if (pair >= rotary) continue; pair_idx = dim >> 1; }
        else { if (dim >= half) continue; pair = dim + half; if (pair >= rotary) continue; pair_idx = dim; }
        uint32_t q_row0 = head * params.head_dim + dim, q_row1 = head * params.head_dim + pair;
        float q0 = dotf(in_row, weight + q_row0 * params.input_stride, params.input_stride);
        float q1 = dotf(in_row, weight + q_row1 * params.input_stride, params.input_stride);
        if (bias) { q0 += bias[q_row0]; q1 += bias[q_row1]; }
        float k0 = 0.0f, k1 = 0.0f;
        if (head < kv_heads) {
            uint32_t k_row0 = q_out + head * params.head_dim + dim, k_row1 = q_out + head * params.head_dim + pair;
            k0 = dotf(in_row, weight + k_row0 * params.input_stride, params.input_stride);
            k1 = dotf(in_row, weight + k_row1 * params.input_stride, params.input_stride);
            if (bias) { k0 += bias[k_row0]; k1 += bias[k_row1]; }
        }
        int32_t pos = positions ? positions[batch * params.seq_len + pos_idx] : (int32_t)pos_idx;
        float c = 0.0f, s = 0.0f;
        if (use_table && pos >= 0 && (uint32_t)pos < params.precompute_max_seq_len) {
            uint32_t table_idx = (uint32_t)pos * half + pair_idx;
            c = cos_table[table_idx];
            s = sin_table[table_idx];
        } else {
            float angle = (float)pos * params.scale * rope_inv_freq(pair_idx, rotary, params.base);
            c = cosf(angle);
            s = sinf(angle);
        }
        rope_apply(&q0, &q1, c, s); qkv_out[out_base + q_row0] = q0; qkv_out[out_base + q_row1] = q1;
        if (head < kv_heads) {
            rope_apply(&k0, &k1, c, s);
            uint32_t k_row0 = q_out + head * params.head_dim + dim, k_row1 = q_out + head * params.head_dim + pair;
            qkv_out[out_base + k_row0] = k0; qkv_out[out_base + k_row1] = k1;
        }
    }
}

extern "C" __global__ void flash_attention(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ output, float* __restrict__ lse, const float* __restrict__ alibi_slopes, FlashAttnConfig params) {
    if (params.head_dim == 0 || params.num_heads == 0 || params.q_seq_len == 0) return;
    uint32_t total = params.batch * params.num_heads * params.q_seq_len;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    uint32_t q_stride = params.q_stride, kv_stride = params.kv_stride, o_stride = params.o_stride, head_dim = params.head_dim;
    float scale = params.scale;
    for (uint32_t i = idx; i < total; i += stride) {
        uint32_t tmp = i, head = tmp % params.num_heads; tmp /= params.num_heads;
        uint32_t q_pos = tmp % params.q_seq_len, batch = tmp / params.q_seq_len;
        int32_t q_pos_abs = (int32_t)q_pos + (int32_t)params.q_pos_offset;
        uint32_t kv_heads = kv_stride / head_dim; if (kv_heads == 0) continue;
        uint32_t group = params.num_heads / kv_heads; if (group == 0) group = 1;
        uint32_t kv_head = head / group; if (kv_head >= kv_heads) kv_head = kv_heads - 1;
        const float* q_ptr = q + (batch * params.q_seq_len + q_pos) * q_stride + head * head_dim;
        float* o_ptr = output + (batch * params.q_seq_len + q_pos) * o_stride + head * head_dim;
        float slope = alibi_slopes ? alibi_slopes[head] : 0.0f;
        uint32_t kv_limit = params.kv_seq_len;
        if (params.causal && params.kv_seq_len >= params.q_seq_len) { uint32_t offset = params.kv_seq_len - params.q_seq_len, max_pos = offset + q_pos; if (max_pos + 1 < kv_limit) kv_limit = max_pos + 1; }
        if (kv_limit == 0) { for (uint32_t d = 0; d < head_dim; ++d) o_ptr[d] = 0.0f; if (lse) lse[i] = -INFINITY; continue; }
        if (head_dim <= 256) {
            float acc[256]; for (uint32_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
            float m = -INFINITY, l = 0.0f;
            for (uint32_t kv_pos = 0; kv_pos < kv_limit; ++kv_pos) {
                const float* k_ptr = k + (batch * params.kv_seq_len + kv_pos) * kv_stride + kv_head * head_dim;
                const float* v_ptr = v + (batch * params.kv_seq_len + kv_pos) * kv_stride + kv_head * head_dim;
                float bias = slope * ((int32_t)kv_pos - q_pos_abs);
                float score = dotf(q_ptr, k_ptr, head_dim) * scale + bias;
                float m_new = score > m ? score : m;
                float exp_m = expf(m - m_new);
                float exp_s = expf(score - m_new);
                l = l * exp_m + exp_s;
                for (uint32_t d = 0; d < head_dim; ++d) acc[d] = acc[d] * exp_m + exp_s * v_ptr[d];
                m = m_new;
            }
            float inv = l > 0.0f ? 1.0f / l : 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) o_ptr[d] = acc[d] * inv;
            if (lse) lse[i] = l > 0.0f ? (m + logf(l)) : -INFINITY;
        } else {
            float max_score = -INFINITY, denom = 0.0f;
            for (uint32_t kv_pos = 0; kv_pos < kv_limit; ++kv_pos) {
                const float* k_ptr = k + (batch * params.kv_seq_len + kv_pos) * kv_stride + kv_head * head_dim;
                float bias = slope * ((int32_t)kv_pos - q_pos_abs);
                float score = dotf(q_ptr, k_ptr, head_dim) * scale + bias; if (score > max_score) max_score = score;
            }
            for (uint32_t kv_pos = 0; kv_pos < kv_limit; ++kv_pos) {
                const float* k_ptr = k + (batch * params.kv_seq_len + kv_pos) * kv_stride + kv_head * head_dim;
                float bias = slope * ((int32_t)kv_pos - q_pos_abs);
                float score = dotf(q_ptr, k_ptr, head_dim) * scale + bias; denom += expf(score - max_score);
            }
            float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
            for (uint32_t base = 0; base < head_dim; base += 128) {
                uint32_t chunk = head_dim - base; if (chunk > 128) chunk = 128;
                float acc[128]; for (uint32_t d = 0; d < chunk; ++d) acc[d] = 0.0f;
                for (uint32_t kv_pos = 0; kv_pos < kv_limit; ++kv_pos) {
                    const float* k_ptr = k + (batch * params.kv_seq_len + kv_pos) * kv_stride + kv_head * head_dim;
                    const float* v_ptr = v + (batch * params.kv_seq_len + kv_pos) * kv_stride + kv_head * head_dim;
                    float bias = slope * ((int32_t)kv_pos - q_pos_abs);
                    float score = dotf(q_ptr, k_ptr, head_dim) * scale + bias, w = expf(score - max_score);
                    for (uint32_t d = 0; d < chunk; ++d) acc[d] += w * v_ptr[base + d];
                }
                for (uint32_t d = 0; d < chunk; ++d) o_ptr[base + d] = acc[d] * inv;
            }
            if (lse) lse[i] = max_score + logf(denom);
        }
    }
}

extern "C" __global__ void flash_attention_paged(const float* __restrict__ q, const uint64_t* __restrict__ page_table, float* __restrict__ output, float* __restrict__ lse, const float* __restrict__ alibi_slopes, FlashAttnPagedConfig params) {
    if (params.head_dim == 0 || params.num_heads == 0 || params.q_seq_len == 0 || params.page_size == 0) return;
    uint32_t total = params.batch * params.num_heads * params.q_seq_len;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x, stride = blockDim.x * gridDim.x;
    uint32_t q_stride = params.q_stride, kv_stride = params.kv_stride, o_stride = params.o_stride, head_dim = params.head_dim;
    float scale = params.scale;
    for (uint32_t i = idx; i < total; i += stride) {
        uint32_t tmp = i, head = tmp % params.num_heads; tmp /= params.num_heads;
        uint32_t q_pos = tmp % params.q_seq_len, batch = tmp / params.q_seq_len;
        int32_t q_pos_abs = (int32_t)q_pos + (int32_t)params.q_pos_offset;
        uint32_t kv_heads = kv_stride / head_dim; if (kv_heads == 0) continue;
        uint32_t group = params.num_heads / kv_heads; if (group == 0) group = 1;
        uint32_t kv_head = head / group; if (kv_head >= kv_heads) kv_head = kv_heads - 1;
        const float* q_ptr = q + (batch * params.q_seq_len + q_pos) * q_stride + head * head_dim;
        float* o_ptr = output + (batch * params.q_seq_len + q_pos) * o_stride + head * head_dim;
        float slope = alibi_slopes ? alibi_slopes[head] : 0.0f;
        uint32_t kv_limit = params.kv_seq_len;
        if (params.causal && params.kv_seq_len >= params.q_seq_len) { uint32_t offset = params.kv_seq_len - params.q_seq_len, max_pos = offset + q_pos; if (max_pos + 1 < kv_limit) kv_limit = max_pos + 1; }
        if (kv_limit == 0) { for (uint32_t d = 0; d < head_dim; ++d) o_ptr[d] = 0.0f; if (lse) lse[i] = -INFINITY; continue; }
        if (head_dim <= 256) {
            float acc[256]; for (uint32_t d = 0; d < head_dim; ++d) acc[d] = 0.0f;
            float m = -INFINITY, l = 0.0f;
            for (uint32_t kv_pos = 0; kv_pos < kv_limit; ++kv_pos) {
                uint32_t page = kv_pos / params.page_size;
                uint32_t offset = kv_pos - page * params.page_size;
                if (page >= params.pages_per_layer) continue;
                const float* base = (const float*)page_table[page];
                const float* k_ptr = base + offset * kv_stride + kv_head * head_dim;
                const float* v_ptr = base + params.page_size * kv_stride + offset * kv_stride + kv_head * head_dim;
                float bias = slope * ((int32_t)kv_pos - q_pos_abs);
                float score = dotf(q_ptr, k_ptr, head_dim) * scale + bias;
                float m_new = score > m ? score : m;
                float exp_m = expf(m - m_new);
                float exp_s = expf(score - m_new);
                l = l * exp_m + exp_s;
                for (uint32_t d = 0; d < head_dim; ++d) acc[d] = acc[d] * exp_m + exp_s * v_ptr[d];
                m = m_new;
            }
            float inv = l > 0.0f ? 1.0f / l : 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) o_ptr[d] = acc[d] * inv;
            if (lse) lse[i] = l > 0.0f ? (m + logf(l)) : -INFINITY;
        } else {
            float max_score = -INFINITY, denom = 0.0f;
            for (uint32_t kv_pos = 0; kv_pos < kv_limit; ++kv_pos) {
                uint32_t page = kv_pos / params.page_size;
                uint32_t offset = kv_pos - page * params.page_size;
                if (page >= params.pages_per_layer) continue;
                const float* base = (const float*)page_table[page];
                const float* k_ptr = base + offset * kv_stride + kv_head * head_dim;
                float bias = slope * ((int32_t)kv_pos - q_pos_abs);
                float score = dotf(q_ptr, k_ptr, head_dim) * scale + bias; if (score > max_score) max_score = score;
            }
            for (uint32_t kv_pos = 0; kv_pos < kv_limit; ++kv_pos) {
                uint32_t page = kv_pos / params.page_size;
                uint32_t offset = kv_pos - page * params.page_size;
                if (page >= params.pages_per_layer) continue;
                const float* base = (const float*)page_table[page];
                const float* k_ptr = base + offset * kv_stride + kv_head * head_dim;
                float bias = slope * ((int32_t)kv_pos - q_pos_abs);
                float score = dotf(q_ptr, k_ptr, head_dim) * scale + bias; denom += expf(score - max_score);
            }
            float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
            for (uint32_t base = 0; base < head_dim; base += 128) {
                uint32_t chunk = head_dim - base; if (chunk > 128) chunk = 128;
                float acc[128]; for (uint32_t d = 0; d < chunk; ++d) acc[d] = 0.0f;
                for (uint32_t kv_pos = 0; kv_pos < kv_limit; ++kv_pos) {
                    uint32_t page = kv_pos / params.page_size;
                    uint32_t offset = kv_pos - page * params.page_size;
                    if (page >= params.pages_per_layer) continue;
                    const float* base_ptr = (const float*)page_table[page];
                    const float* k_ptr = base_ptr + offset * kv_stride + kv_head * head_dim;
                    const float* v_ptr = base_ptr + params.page_size * kv_stride + offset * kv_stride + kv_head * head_dim;
                    float bias = slope * ((int32_t)kv_pos - q_pos_abs);
                    float score = dotf(q_ptr, k_ptr, head_dim) * scale + bias, w = expf(score - max_score);
                    for (uint32_t d = 0; d < chunk; ++d) acc[d] += w * v_ptr[base + d];
                }
                for (uint32_t d = 0; d < chunk; ++d) o_ptr[base + d] = acc[d] * inv;
            }
            if (lse) lse[i] = max_score + logf(denom);
        }
    }
}

extern "C" __global__ void sample_from_logits(const float* logits, uint32_t* output, SamplingKernelConfig params) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= params.batch) return;
    if (params.vocab_size == 0) { output[idx] = 0; return; }
    float temp = params.temperature > 0.0f ? params.temperature : 1.0f;
    float top_p = params.top_p; if (!(top_p > 0.0f) || top_p > 1.0f) top_p = 1.0f;
    uint32_t base = idx * params.stride, top_k = params.top_k;
    if (top_k == 0) top_k = top_p < 1.0f ? params.vocab_size : 1;
    if (top_k <= 1) {
        float best = logits[base]; uint32_t best_idx = 0;
        for (uint32_t i = 1; i < params.vocab_size; ++i) { float v = logits[base + i]; if (v > best) { best = v; best_idx = i; } }
        output[idx] = best_idx; return;
    }
    if (top_k > params.vocab_size) top_k = params.vocab_size; if (top_k > 128) top_k = 128;
    float top_vals[128]; uint32_t top_idx[128];
    for (uint32_t i = 0; i < top_k; ++i) { top_vals[i] = -INFINITY; top_idx[i] = 0; }
    for (uint32_t i = 0; i < params.vocab_size; ++i) {
        float v = logits[base + i] / temp; if (v <= top_vals[top_k - 1]) continue;
        uint32_t pos = top_k - 1; while (pos > 0 && v > top_vals[pos - 1]) { top_vals[pos] = top_vals[pos - 1]; top_idx[pos] = top_idx[pos - 1]; --pos; }
        top_vals[pos] = v; top_idx[pos] = i;
    }
    float max_val = top_vals[0], sum = 0.0f;
    float exp_vals[128];
    for (uint32_t i = 0; i < top_k; ++i) { float e = expf(top_vals[i] - max_val); exp_vals[i] = e; sum += e; }
    uint32_t cutoff = top_k;
    if (top_p < 1.0f && sum > 0.0f) {
        float acc_p = 0.0f;
        for (uint32_t i = 0; i < top_k; ++i) { acc_p += exp_vals[i]; if (acc_p / sum >= top_p) { cutoff = i + 1; break; } }
    }
    float subset_sum = 0.0f; for (uint32_t i = 0; i < cutoff; ++i) subset_sum += exp_vals[i];
    if (subset_sum <= 0.0f) { output[idx] = top_idx[0]; return; }
    float r = rand_uniform(clock64()) * subset_sum, acc = 0.0f; uint32_t chosen = top_idx[0];
    for (uint32_t i = 0; i < cutoff; ++i) { acc += exp_vals[i]; if (r <= acc) { chosen = top_idx[i]; break; } }
    output[idx] = chosen;
}
