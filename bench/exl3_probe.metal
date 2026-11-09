// Experimental mul1 EXL3 operators. Format reference: ExLlamaV3 tests/testlib/trellis.py.
// Control uses the published tile order; optimized variants transpose only the
// outer tile axes at load. Every packed ring stays byte-identical. No dense copy.
#include <metal_stdlib>
using namespace metal;

inline uint exl3_index(uint r, uint c) {
    const uint t = (c & 7u) * 4u + ((r & 7u) >> 1);
    const uint j = (r & 1u) + (r >> 3) * 2u + (c >> 3) * 4u;
    return t * 8u + j;
}

inline float exl3_value(device const uint* tile, uint r, uint c, uint bits2) {
    const uint i = exl3_index(r, c), words = bits2 * 4u, ring = words * 32u;
    const uint end = (i + 1u) * (bits2 / 2u) + (bits2 & 1u) * ((i + 1u) / 2u);
    const uint start = (end + ring - 16u) % ring;
    const uint at = start / 32u, shift = start & 31u;
    uint window = tile[at] << shift;
    if (shift) window |= tile[(at + 1u) % words] >> (32u - shift);
    const uint v = (window >> 16u) * 0x83DCD12Du;
    const uint bytesum = (v & 255u) + ((v >> 8u) & 255u) + ((v >> 16u) & 255u) + (v >> 24u);
    const float inv = float(as_type<half>(ushort(0x1EEE)));
    const float bias = float(as_type<half>(ushort(0xC931)));
    // mul1 specifies one fp16 rounding after the fused affine codebook operation.
    return float(half(fma(float(1024u + bytesum), inv, bias)));
}

kernel void exl3_input(device float* out, device const half* x, device const half* scale,
                       constant long& width, uint2 group [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float h[128];
    const ulong column = ulong(group.x) * 128u + tid;
    const ulong at = ulong(group.y) * ulong(width) + column;
    h[tid] = float(x[at]) * float(scale[column]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1; step < 128; step <<= 1) {
        const float a = h[tid], b = h[tid ^ step];
        const float v = (tid & step) ? b - a : a + b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[at] = h[tid] * 0.08838834764831844f;
}

kernel void exl3_output(device half* out, device const float* x, device const half* scale,
                        constant long& width, uint2 group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float h[128];
    const ulong column = ulong(group.x) * 128u + tid;
    const ulong at = ulong(group.y) * ulong(width) + column;
    h[tid] = x[at];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 1; step < 128; step <<= 1) {
        const float a = h[tid], b = h[tid ^ step];
        const float v = (tid & step) ? b - a : a + b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        h[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[at] = half(h[tid] * 0.08838834764831844f * float(scale[column]));
}

// One group per output column and request row; arbitrary K divisible by 128,
// including the 17408-wide FFN down projection. FP32 accumulation throughout.
kernel void exl3_matvec(device float* out, device const float* x,
                        device const uint* packed, constant long& K, constant long& N,
                        constant long& bits2, uint3 group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    const ulong column = group.y, row = group.z, nt = ulong(N) / 16u;
    const ulong tile_words = ulong(bits2) * 4u;
    float acc = 0;
    for (ulong k = tid; k < ulong(K); k += 256u) {
        const ulong tile = ((k / 16u) * nt + column / 16u) * tile_words;
        const float w = exl3_value(packed + tile, uint(k & 15u), uint(column & 15u), uint(bits2));
        acc = fma(x[row * ulong(K) + k], w, acc);
    }
    const float subtotal = simd_sum(acc);
    if (lane == 0) partial[sid] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        const float total = simd_sum(lane < 8u ? partial[lane] : 0.0f);
        if (lane == 0) out[row * ulong(N) + column] = total;
    }
}

// Lossless [N/16,K/16,words] tile order brings each output block's K sweep
// together. The bit-window position is invariant as k advances by 256.
// Arithmetic and lookup variants isolate codebook cost from the layout change.
template <bool UseLut>
kernel void exl3_nmajor(device float* out, device const float* x,
                        device const uint* packed, device const half* lut,
                        constant long& K, constant long& N, constant long& bits2,
                        uint3 group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint sid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    const ulong column = group.y, row = group.z, kt = ulong(K) / 16u;
    const uint words = uint(bits2) * 4u, ring = words * 32u;
    const uint i = exl3_index(tid & 15u, uint(column & 15u));
    const uint end = (i + 1u) * (uint(bits2) / 2u) + (uint(bits2) & 1u) * ((i + 1u) / 2u);
    const uint start = (end + ring - 16u) % ring;
    const uint at = start / 32u, next = at + 1u == words ? 0u : at + 1u, shift = start & 31u;
    const ulong base = (column / 16u) * kt * ulong(words);
    const float inv = float(as_type<half>(ushort(0x1EEE)));
    const float bias = float(as_type<half>(ushort(0xC931)));
    float acc = 0;
    for (ulong k = tid; k < ulong(K); k += 256u) {
        const ulong tile = base + (k / 16u) * ulong(words);
        uint window = packed[tile + at] << shift;
        if (shift) window |= packed[tile + next] >> (32u - shift);
        const uint code = window >> 16u;
        float w;
        if (UseLut) {
            w = float(lut[code]);
        } else {
            const uint v = code * 0x83DCD12Du;
            const uint bytesum = (v & 255u) + ((v >> 8u) & 255u) + ((v >> 16u) & 255u) + (v >> 24u);
            w = float(half(fma(float(1024u + bytesum), inv, bias)));
        }
        acc = fma(x[row * ulong(K) + k], w, acc);
    }
    const float subtotal = simd_sum(acc);
    if (lane == 0) partial[sid] = subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        const float total = simd_sum(lane < 8u ? partial[lane] : 0.0f);
        if (lane == 0) out[row * ulong(N) + column] = total;
    }
}

#define EXL3_NMAJOR(LOOKUP, NAME) \
    template [[host_name(NAME)]] kernel void exl3_nmajor<LOOKUP>( \
        device float*, device const float*, device const uint*, device const half*, \
        constant long&, constant long&, constant long&, uint3, uint, uint, uint);
EXL3_NMAJOR(false, "exl3_nmajor_arithmetic")
EXL3_NMAJOR(true, "exl3_nmajor_lut")

// Prefill fallback: reconstruct only the rotated half matrix per call, then use
// MPS GEMM. Hadamards and scales act on activations, not on the dense weights.
kernel void exl3_unpack(device half* out, device const uint* packed,
                        constant long& K, constant long& N, constant long& bits2,
                        uint at [[thread_position_in_grid]]) {
    const ulong count = ulong(K) * ulong(N);
    if (ulong(at) >= count) return;
    const ulong r = ulong(at) / ulong(N), c = ulong(at) % ulong(N);
    const ulong tile = ((r / 16u) * (ulong(N) / 16u) + c / 16u) * ulong(bits2) * 4u;
    out[at] = half(exl3_value(packed + tile, uint(r & 15u), uint(c & 15u), uint(bits2)));
}
