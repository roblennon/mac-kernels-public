// Experimental M=1 kernel. Python prepends the production METAL_COMMON helpers.
// One 256-thread group per output; coalesced char4 reads, fp32 reduction.
// Normalize and round each decoded weight exactly as i8_decode4 does. Round the
// sum to TC, then use production exponent restoration and output rounding.
template <typename TC, typename TO>
kernel void small_m_i8(device TO* out,
                       device const TC* x,
                       device const int* row_exp,
                       device const char4* weight,
                       device const float* scale,
                       constant long& K,
                       constant long& scale_stride,
                       uint2 pos [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint sid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    const ulong row = pos.y, quads = ulong(K) / 4;
    int exponent;
    const float mant = split_pow2(scale[row * ulong(scale_stride)], 7, exponent);
    float acc = 0.0f;
    for (ulong q = tid; q < quads; q += 256) {
        const float4 w = float4(vec<TC, 4>(float4(weight[row * quads + q]) * mant));
        const float4 a = float4(*(device const vec<TC, 4>*)(x + q * 4));
        acc += dot(a, w);
    }
    const float local = simd_sum(acc);
    if (lane == 0) partial[sid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        const float total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
        if (lane == 0) {
            const uint bits = as_type<uint>(float(TC(total)));
            store_b(out, row, ldexp_bits(bits, row_exp[0] + exponent));
        }
    }
}
#define QMV(TC, TO, NAME) \
    template [[host_name("small_m_i8_" NAME)]] kernel void small_m_i8<TC, TO>( \
        device TO*, device const TC*, device const int*, device const char4*, \
        device const float*, constant long&, constant long&, uint2, uint, uint, uint);
QMV(half, ushort, "f16_bf16")
QMV(half, half, "f16_f16")
QMV(half, float, "f16_f32")
QMV(float, ushort, "f32_bf16")
QMV(float, half, "f32_f16")
QMV(float, float, "f32_f32")
