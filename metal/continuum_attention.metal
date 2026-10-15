
#include <metal_stdlib>
using namespace metal;

// Same analytic coefficient expressions as the original PyTorch reference.
// Series is used solely near the removable singularity at delta == 0.
inline float segment_L(float t) {
    if (t < 0.1f) {
        float t2 = t*t;
        float t3 = t2*t;
        float t4 = t2*t2;
        float t5 = t4*t;
        float t6 = t3*t3;
        return 1.0f - 0.5f*t + t2/6.0f - t3/24.0f
             + t4/120.0f - t5/720.0f + t6/5040.0f;
    }
    return (1.0f - exp(-t)) / t;
}
inline float segment_mu(float d) {
    float t = abs(d);
    if (t < 0.1f) {
        float d2 = d*d;
        return 0.5f + d/12.0f - d*d2/720.0f + d2*d2*d/30240.0f;
    }
    float pos = 1.0f / (1.0f-exp(-t)) - 1.0f/t;
    return d >= 0.0f ? pos : 1.0f-pos;
}
inline float segment_mass(float a, float b, float rowmax) {
    return exp(max(a,b)-rowmax)*segment_L(abs(b-a));
}

// Group-wide (up to 8 SIMD groups) reduction for 256 threads.
// All threads MUST participate in identical barrier sequences.
inline float reduce_max256(float x, uint lane, uint simdlane, uint simdidx,
                           threadgroup float *scratch) {
    float w = simd_max(x);
    if (simdlane == 0u) scratch[simdidx] = w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float g = simdidx == 0u && simdlane < 8u ? scratch[simdlane] : -INFINITY;
    g = simd_max(g);
    if (lane == 0u) scratch[0] = g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}
inline float reduce_sum256(float x, uint lane, uint simdlane, uint simdidx,
                           threadgroup float *scratch) {
    float w = simd_sum(x);
    if (simdlane == 0u) scratch[simdidx] = w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float g = simdidx == 0u && simdlane < 8u ? scratch[simdlane] : 0.0f;
    g = simd_sum(g);
    if (lane == 0u) scratch[0] = g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

// Exactly one kernel dispatch produces normalized weights. Parallel lanes
// compute key endpoint contributions; two reductions compute rowmax and Z.
kernel void acma_weights(
    const device float* scores [[buffer(0)]],
    device float* weights [[buffer(1)]],
    device float* row_stats [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint simdlane [[thread_index_in_simdgroup]],
    uint simdidx [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint query = row % T;
    uint offset = row * T;
    float sv = lane <= query ? scores[offset+lane] : -INFINITY;
    float m = reduce_max256(sv, lane, simdlane, simdidx, scratch);
    float w = 0.0f;
    if (lane <= query) {
        float s = scores[offset+lane];
        if (lane == 0u) w += 0.5f * exp(s-m);
        if (lane == query) w += 0.5f * exp(s-m);
        if (lane > 0u) {
            float a = scores[offset+lane-1u];
            float r = segment_mass(a,s,m);
            w += r * segment_mu(s-a);
        }
        if (lane < query) {
            float b = scores[offset+lane+1u];
            float r = segment_mass(s,b,m);
            w += r * (1.0f-segment_mu(b-s));
        }
    }
    float Z = reduce_sum256(w, lane, simdlane, simdidx, scratch);
    if (lane == 0u) {
        row_stats[2u*row] = m;
        row_stats[2u*row+1u] = 1.0f / Z;
    }
    if (lane < T) weights[offset+lane] = lane <= query ? w/Z : 0.0f;
}

// A row maximum over up to 1024 causal scores. Each of 256 threads
// compares four positions (lane + 0/256/512/768), then uses one parallel
// SIMD-group reduction. No dynamic token scan and no full causal mask tensor.
kernel void acma_long_rowmax(
    const device float* scores [[buffer(0)]],
    device float* row_max [[buffer(1)]],
    constant uint& T [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint simdlane [[thread_index_in_simdgroup]],
    uint simdidx [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint query = row % T;
    uint offset = row * T;
    uint k0 = lane;
    uint k1 = lane+256u;
    uint k2 = lane+512u;
    uint k3 = lane+768u;
    float v0 = (k0<=query && k0<T) ? scores[offset+k0] : -INFINITY;
    float v1 = (k1<=query && k1<T) ? scores[offset+k1] : -INFINITY;
    float v2 = (k2<=query && k2<T) ? scores[offset+k2] : -INFINITY;
    float v3 = (k3<=query && k3<T) ? scores[offset+k3] : -INFINITY;
    float m = reduce_max256(max(max(v0,v1),max(v2,v3)),
                            lane,simdlane,simdidx,scratch);
    if (lane==0u) row_max[row]=m;
}

// For longer contexts, process all (row,key) pairs in independent Metal
// threads. Each 256-lane threadgroup is a tile in the flattened row/key
// plane. Row maxima are computed by torch.amax upstream. This avoids
// hardware-dependent threadgroup reductions and the T<=256 restriction.
// The normalizer is a separate row-wise torch.sum (one tensor operation).
// Every key weight includes both adjacent segment endpoints, and both caps.
kernel void acma_long_raw_weights(
    const device float* scores [[buffer(0)]],
    const device float* row_max [[buffer(1)]],
    device float* raw_weights [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows*T) return;
    uint row = index/T;
    uint key = index%T;
    uint query = row%T;
    if (key > query) {raw_weights[index] = 0.0f; return;}
    uint offset = row*T;
    float s = scores[index];
    float m = row_max[row];
    float w = 0.0f;
    if (key == 0u) w += 0.5f * exp(s-m);
    if (key == query) w += 0.5f * exp(s-m);
    if (key > 0u) {
        float a = scores[index-1u];
        float r = segment_mass(a,s,m);
        w += r * segment_mu(s-a);
    }
    if (key < query) {
        float b = scores[index+1u];
        float r = segment_mass(s,b,m);
        w += r * (1.0f-segment_mu(b-s));
    }
    raw_weights[index] = w;
}

// TRUE fused normalization + value aggregation. Each lane owns a causal
// segment; coefficients are computed ONCE. The original endpoint caps are
// included. There is no W[T,T] allocation. Head dimension D <= 128.
// Value channels are reduced independently across cooperating SIMD lanes.
kernel void acma_direct(
    const device float* scores [[buffer(0)]],
    const device float* values [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& T [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint simdlane [[thread_index_in_simdgroup]],
    uint simdidx [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    uint query = row % T;
    uint bh = row / T;
    uint offset = row * T;
    uint value_offset = bh * T * D;
    float score = lane <= query ? scores[offset+lane] : -INFINITY;
    float m = reduce_max256(score, lane, simdlane, simdidx, scratch);
    float mass = 0.0f;
    float mu = 0.5f;
    if (lane > 0u && lane <= query) {
        float a = scores[offset+lane-1u];
        float b = scores[offset+lane];
        mass = segment_mass(a,b,m);
        mu = segment_mu(b-a);
    }
    float cap0 = lane == 0u ? 0.5f * exp(scores[offset]-m) : 0.0f;
    float capi = lane == query ? 0.5f * exp(scores[offset+query]-m) : 0.0f;
    float Z = reduce_sum256(mass+cap0+capi, lane, simdlane, simdidx, scratch);
    // All threads execute the same number of barriers, independent of D.
    for (uint d = 0u; d < D; ++d) {
        float part = 0.0f;
        if (lane > 0u && lane <= query) {
            float vl = values[value_offset+(lane-1u)*D+d];
            float vr = values[value_offset+lane*D+d];
            part = mass * ((1.0f-mu)*vl + mu*vr);
        }
        if (lane == 0u) part += cap0 * values[value_offset+d];
        if (lane == query) part += capi * values[value_offset+query*D+d];
        float numerator = reduce_sum256(part, lane, simdlane, simdidx, scratch);
        if (lane == 0u) output[row*D+d] = numerator/Z;
    }
}

// Exact score gradient of the continuous-memory weights. The three moments
// integrate 1, u, u^2 against exp(d*u-max(d,0)) over u in [0,1].
// For small |d|, these are Taylor expansions of closed-form integrals,
// NOT quadrature or an alternative attention operator.
inline float3 segment_moments(float d) {
    float t = abs(d);
    if (t < 1.0f) {
        // Horner expansions through degree 9; uniform float32 accuracy.
        float m0 = (1.000000000000000e+00f + t*(-5.000000000000000e-01f + t*(1.666666666666667e-01f + t*(-4.166666666666666e-02f + t*(8.333333333333333e-03f + t*(-1.388888888888889e-03f + t*(1.984126984126984e-04f + t*(-2.480158730158730e-05f + t*(2.755731922398589e-06f + t*-2.755731922398589e-07f)))))))));
        float m1;
        float m2;
        if (d >= 0.0f) {
            m1 = (5.000000000000000e-01f + t*(-1.666666666666667e-01f + t*(4.166666666666666e-02f + t*(-8.333333333333333e-03f + t*(1.388888888888889e-03f + t*(-1.984126984126984e-04f + t*(2.480158730158730e-05f + t*(-2.755731922398589e-06f + t*(2.755731922398589e-07f + t*-2.505210838544172e-08f)))))))));
            m2 = (3.333333333333333e-01f + t*(-8.333333333333333e-02f + t*(1.666666666666667e-02f + t*(-2.777777777777778e-03f + t*(3.968253968253968e-04f + t*(-4.960317460317460e-05f + t*(5.511463844797179e-06f + t*(-5.511463844797178e-07f + t*(5.010421677088344e-08f + t*-4.175351397573620e-09f)))))))));
        } else {
            m1 = (5.000000000000000e-01f + t*(-3.333333333333333e-01f + t*(1.250000000000000e-01f + t*(-3.333333333333333e-02f + t*(6.944444444444444e-03f + t*(-1.190476190476191e-03f + t*(1.736111111111111e-04f + t*(-2.204585537918871e-05f + t*(2.480158730158730e-06f + t*-2.505210838544172e-07f)))))))));
            m2 = (3.333333333333333e-01f + t*(-2.500000000000000e-01f + t*(1.000000000000000e-01f + t*(-2.777777777777778e-02f + t*(5.952380952380952e-03f + t*(-1.041666666666667e-03f + t*(1.543209876543210e-04f + t*(-1.984126984126984e-05f + t*(2.254689754689755e-06f + t*-2.296443268665491e-07f)))))))));
        }
        return float3(m0,m1,m2);
    }
    float x = exp(-t);
    float inv = 1.0f/t;
    float inv2 = inv*inv;
    float m0 = (1.0f-x)*inv;
    if (d > 0.0f) {
        return float3(m0, (t-1.0f+x)*inv2,
                      (t*t-2.0f*t+2.0f-2.0f*x)*inv2*inv);
    }
    return float3(m0, (1.0f-(t+1.0f)*x)*inv2,
                  (2.0f-(t*t+2.0f*t+2.0f)*x)*inv2*inv);
}

// A separate elementwise backward kernel avoids multi-SIMD scratch reductions.
// The forward kernel has already computed the exact (row max, inverse Z),
// so these row statistics are cached in a tiny [rows,2] buffer. The only
// other row statistic, dot(grad_weights, weights), is calculated with PyTorch
// reduction and passed as a [rows] tensor. One Metal lane computes one
// independent score derivative, with no barriers, atomics or token loops.
kernel void acma_scores_backward(
    const device float* scores [[buffer(0)]],
    const device float* grad_weights [[buffer(1)]],
    const device float* row_stats [[buffer(2)]],
    const device float* row_means [[buffer(3)]],
    device float* grad_scores [[buffer(4)]],
    constant uint& T [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows*T) return;
    uint row = index/T;
    uint lane = index%T;
    uint query = row%T;
    if (lane > query) {grad_scores[index] = 0.0f; return;}

    uint offset=row*T;
    float s = scores[index];
    float m = row_stats[2u*row];
    float invZ = row_stats[2u*row+1u];
    float mean = row_means[row];
    float g = grad_weights[index]-mean;

    float result=0.0f;
    if (lane == 0u) result += 0.5f*exp(s-m)*g;
    if (lane == query) result += 0.5f*exp(s-m)*g;
    if (lane > 0u) {
        float a = scores[index-1u];
        float3 j = segment_moments(s-a);
        float e = exp(max(a,s)-m);
        float gl = grad_weights[index-1u]-mean;
        result += e*(gl*(j.y-j.z)+g*j.z);
    }
    if (lane < query) {
        float b = scores[index+1u];
        float3 j = segment_moments(b-s);
        float e = exp(max(b,s)-m);
        float gr = grad_weights[index+1u]-mean;
        result += e*(g*(j.x-2.0f*j.y+j.z)+gr*(j.y-j.z));
    }
    grad_scores[index]=result*invZ;
}
