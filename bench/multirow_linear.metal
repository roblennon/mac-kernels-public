// Multi-row linear for small row counts: y[M,N] = x[M,K] @ W[N,K]^T (+ bias[N]).
//
// W stays in nn.Linear's [N, K] row-major layout. Each simdgroup owns some weight
// rows and streams them along K exactly once; every 16-byte weight chunk a lane
// loads is applied to all M input rows before the next one is loaded, so the
// weight crosses the memory bus once per call regardless of M. Accumulation is
// fp32; the output is rounded once to the input format.
//
// Four kernel families, chosen per shape and row count by multirow_linear.py:
//   mrl_dev  per-lane dot products; x read from device memory (cache-resident, <= 256 KB)
//   mrl_tg   the same, with a K tile of x staged in threadgroup memory
//   mrl_mma  simdgroup 8x8 matrix multiply-accumulate on tiles staged in threadgroup memory
//   mrl_mmr  simdgroup 8x8 matrix multiply-accumulate fed straight from device memory
// The per-lane families are fastest at 1 to 4 rows. At 8 or more rows they are bound
// by x reads (each x element is reused only across one lane's R weight rows), and the
// matrix families, which reuse each x fragment across 8 weight rows, take over.
//
// Shared template parameters:
//   C     element codec (F16, F16H or BF16; all 16-bit, loaded 8 at a time as uint4)
//   M     input rows (compile time, so the accumulators stay in registers)
//   R/NS  weight rows (dev, tg) or 8-row weight tiles (mma, mmr) per simdgroup
//   U     K steps per loop iteration
//
// Host contract (checked in multirow_linear.py): x, W, y contiguous; K % 8 == 0;
// x and W 16-byte aligned. Rows past N are clamped for loads and never written.
#include <metal_stdlib>
using namespace metal;

struct F16 {
    typedef float FT;  // fragment operand type for the mmr variant
    static inline float4 lo(uint4 q) { return float4(float2(as_type<half2>(q.x)), float2(as_type<half2>(q.y))); }
    static inline float4 hi(uint4 q) { return float4(float2(as_type<half2>(q.z)), float2(as_type<half2>(q.w))); }
    static inline float4 four(uint2 q) { return float4(float2(as_type<half2>(q.x)), float2(as_type<half2>(q.y))); }
    static inline float4 flo(uint4 q) { return lo(q); }
    static inline float4 fhi(uint4 q) { return hi(q); }
    static inline float4 ffour(uint2 q) { return four(q); }
    static inline float one(ushort u) { return float(as_type<half>(u)); }
    static inline ushort pack(float f) { return as_type<ushort>(half(f)); }
    // Round-2 fragment helpers (mmh, mmt, mmp): w2 turns one 32-bit word (two adjacent 16-bit
    // values) into a lane's two fragment elements; x2 takes the low or high value of each of two words.
    static inline float2 w2(uint u) { return float2(as_type<half2>(u)); }
    static inline float2 x2(uint a, uint b, bool hi) {
        const half2 ha = as_type<half2>(a), hb = as_type<half2>(b);
        return float2(hi ? ha.y : ha.x, hi ? hb.y : hb.x);
    }
};

// F16 with half-precision fragment operands for the mmr variant: products of two halves
// are formed and accumulated in the fp32 accumulator fragment (checked by the suite).
struct F16H {
    typedef half FT;
    static inline float one(ushort u) { return F16::one(u); }
    static inline ushort pack(float f) { return F16::pack(f); }
    static inline half4 flo(uint4 q) { return half4(as_type<half2>(q.x), as_type<half2>(q.y)); }
    static inline half4 fhi(uint4 q) { return half4(as_type<half2>(q.z), as_type<half2>(q.w)); }
    static inline half4 ffour(uint2 q) { return half4(as_type<half2>(q.x), as_type<half2>(q.y)); }
    static inline half2 w2(uint u) { return as_type<half2>(u); }
    static inline half2 x2(uint a, uint b, bool hi) {
        const half2 ha = as_type<half2>(a), hb = as_type<half2>(b);
        return half2(hi ? ha.y : ha.x, hi ? hb.y : hb.x);
    }
};

struct BF16 {
    typedef float FT;
    static inline float2 two(uint u) { return float2(as_type<float>(u << 16), as_type<float>(u & 0xffff0000u)); }
    static inline float4 lo(uint4 q) { return float4(two(q.x), two(q.y)); }
    static inline float4 hi(uint4 q) { return float4(two(q.z), two(q.w)); }
    static inline float4 four(uint2 q) { return float4(two(q.x), two(q.y)); }
    static inline float4 flo(uint4 q) { return lo(q); }
    static inline float4 fhi(uint4 q) { return hi(q); }
    static inline float4 ffour(uint2 q) { return four(q); }
    static inline float one(ushort u) { return as_type<float>(uint(u) << 16); }
    static inline float2 w2(uint u) { return two(u); }
    static inline float2 x2(uint a, uint b, bool hi) {
        return float2(as_type<float>(hi ? (a & 0xffff0000u) : (a << 16)),
                      as_type<float>(hi ? (b & 0xffff0000u) : (b << 16)));
    }
    static inline ushort pack(float f) {
        // Round to nearest even; NaN stays NaN.
        uint u = as_type<uint>(f);
        if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40u);
        u += 0x7fffu + ((u >> 16) & 1u);
        return ushort(u >> 16);
    }
};

template <typename C, int M, int R>
inline void store_rows(thread float (&acc)[R][M], device ushort* y, device const ushort* bias,
                       uint row0, uint N, bool has_bias, uint lane) {
    for (int r = 0; r < R; ++r) {
        const uint n = row0 + r;
        const float b = (has_bias && n < N) ? C::one(bias[n]) : 0.0f;
        for (int m = 0; m < M; ++m) {
            const float s = simd_sum(acc[r][m]);
            // Spread the stores over lanes; each (r, m) is written by one lane.
            if (lane == uint(m & 31) && n < N) y[ulong(m) * N + n] = C::pack(s + b);
        }
    }
}

template <typename C, int M, int R, int U>
kernel void mrl_dev(device const uint4* x [[buffer(0)]],
                    device const uint4* w [[buffer(1)]],
                    device const ushort* bias [[buffer(2)]],
                    device ushort* y [[buffer(3)]],
                    constant long& K [[buffer(4)]],
                    constant long& N_ [[buffer(5)]],
                    constant long& has_bias [[buffer(6)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint nsg [[simdgroups_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint K8 = uint(K) / 8, N = uint(N_);
    const uint row0 = (tg * nsg + sg) * R;
    if (row0 >= N) return;  // no barriers in this variant
    device const uint4* wr[R];
    for (int r = 0; r < R; ++r) wr[r] = w + ulong(min(row0 + r, N - 1)) * K8;
    float acc[R][M];
    for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;

    for (uint k = lane; k < K8; k += 32 * U) {
        uint4 wq[U][R];
        for (int u = 0; u < U; ++u) {
            const uint kk = k + 32 * u;
            for (int r = 0; r < R; ++r) wq[u][r] = kk < K8 ? wr[r][kk] : uint4(0);
        }
        for (int u = 0; u < U; ++u) {
            const uint kk = k + 32 * u;
            if (kk >= K8) break;
            float4 wl[R], wh[R];
            for (int r = 0; r < R; ++r) { wl[r] = C::lo(wq[u][r]); wh[r] = C::hi(wq[u][r]); }
            for (int m = 0; m < M; ++m) {
                const uint4 xq = x[ulong(m) * K8 + kk];
                const float4 xl = C::lo(xq), xh = C::hi(xq);
                for (int r = 0; r < R; ++r) acc[r][m] += dot(xl, wl[r]) + dot(xh, wh[r]);
            }
        }
    }
    store_rows<C, M, R>(acc, y, bias, row0, N, has_bias != 0, lane);
}

template <typename C, int M, int R, int U>
kernel void mrl_tg(device const uint4* x [[buffer(0)]],
                   device const uint4* w [[buffer(1)]],
                   device const ushort* bias [[buffer(2)]],
                   device ushort* y [[buffer(3)]],
                   constant long& K [[buffer(4)]],
                   constant long& N_ [[buffer(5)]],
                   constant long& has_bias [[buffer(6)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint nsg [[simdgroups_per_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint TILE = 32 * U;  // uint4 per x row per tile
    threadgroup uint4 xs[M * TILE];
    const uint K8 = uint(K) / 8, N = uint(N_);
    const uint row0 = (tg * nsg + sg) * R;
    device const uint4* wr[R];
    for (int r = 0; r < R; ++r) wr[r] = w + ulong(min(row0 + r, N - 1)) * K8;
    float acc[R][M];
    for (int r = 0; r < R; ++r) for (int m = 0; m < M; ++m) acc[r][m] = 0.0f;
    const uint nthreads = nsg * 32;

    for (uint k0 = 0; k0 < K8; k0 += TILE) {
        // Issue this lane's weight loads before the barrier so they overlap the x staging.
        uint4 wq[U][R];
        for (int u = 0; u < U; ++u) {
            const uint kk = k0 + 32 * u + lane;
            for (int r = 0; r < R; ++r) wq[u][r] = kk < K8 ? wr[r][kk] : uint4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < M * TILE; i += nthreads) {
            const uint m = i / TILE, kk = k0 + i % TILE;
            xs[i] = kk < K8 ? x[ulong(m) * K8 + kk] : uint4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int u = 0; u < U; ++u) {
            float4 wl[R], wh[R];
            for (int r = 0; r < R; ++r) { wl[r] = C::lo(wq[u][r]); wh[r] = C::hi(wq[u][r]); }
            for (int m = 0; m < M; ++m) {
                const uint4 xq = xs[m * TILE + 32 * u + lane];
                const float4 xl = C::lo(xq), xh = C::hi(xq);
                for (int r = 0; r < R; ++r) acc[r][m] += dot(xl, wl[r]) + dot(xh, wh[r]);
            }
        }
    }
    store_rows<C, M, R>(acc, y, bias, row0, N, has_bias != 0, lane);
}

// Variant "mma": simdgroup 8x8 matrix multiply-accumulate in fp32. Variants dev and tg
// reuse an x element only across the R rows of one lane, so at 12 or more rows they
// are bound by x reads from cache or threadgroup memory, not by the weight stream.
// Here a threadgroup stages a [NTG x KT] weight tile and an [M x KT] x tile in
// threadgroup memory as fp32 (exact for fp16 and bf16), and each simdgroup multiplies
// 8x8 fragments, which share x across 8 weight rows and the weight across 8 x rows.
// The next weight tile is loaded into registers while the current one is computed.
// Products of two 16-bit floats are exact in fp32; accumulation is fp32.
//
//   M   input rows; padded to a multiple of 8 with zeros inside the kernel
//   S   simdgroups per threadgroup;  NS  8-row weight tiles per simdgroup
//   KT  K tile in elements (multiple of 8)
// Grid: (ceil(N / (S*NS*8)) * 32*S, splits) threads, groups of (32*S, 1). With
// splits > 1 each K range writes fp32 partial sums to `part`, and mrl_reduce adds
// them in a fixed order, adds the bias and rounds.
template <typename C, int M, int S, int NS, int KT>
kernel void mrl_mma(device const uint4* x [[buffer(0)]],
                    device const uint4* w [[buffer(1)]],
                    device const ushort* bias [[buffer(2)]],
                    device ushort* y [[buffer(3)]],
                    constant long& K [[buffer(4)]],
                    constant long& N_ [[buffer(5)]],
                    constant long& has_bias [[buffer(6)]],
                    device float* part [[buffer(7)]],
                    constant long& klen [[buffer(8)]],
                    constant long& splits [[buffer(9)]],
                    uint2 tgp [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint tid [[thread_index_in_threadgroup]]) {
    constexpr int MT = (M + 7) / 8, MP = MT * 8;
    constexpr int NTG = S * NS * 8;
    constexpr int KC = KT / 8;             // uint4 chunks per row per tile
    constexpr int LD = KT + 4;             // padded threadgroup row stride, in floats
    constexpr int WCH = NTG * KC, XCH = MP * KC, NT = S * 32;
    constexpr int WPT = (WCH + NT - 1) / NT;
    static_assert(MP * NTG <= NTG * LD, "output staging must fit in the weight tile");
    threadgroup float ws[NTG * LD];
    threadgroup float xs[MP * LD];

    const uint K8 = uint(K) / 8, N = uint(N_);
    const uint n0 = tgp.x * NTG;
    const uint kb = tgp.y * (uint(klen) / 8);
    const uint ke = min(K8, kb + uint(klen) / 8);

    simdgroup_float8x8 acc[MT][NS];
    for (int mt = 0; mt < MT; ++mt)
        for (int ns = 0; ns < NS; ++ns) acc[mt][ns] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint4 wreg[WPT];
    for (int i = 0; i < WPT; ++i) {
        const uint c = tid + i * NT, row = c / KC, cc = c % KC;
        wreg[i] = (c < uint(WCH) && n0 + row < N && kb + cc < ke) ? w[ulong(n0 + row) * K8 + kb + cc] : uint4(0);
    }
    for (uint k0 = kb; k0 < ke; k0 += KC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < WPT; ++i) {
            const uint c = tid + i * NT;
            if (c < uint(WCH)) {
                threadgroup float* dst = ws + (c / KC) * LD + (c % KC) * 8;
                *(threadgroup float4*)dst = C::lo(wreg[i]);
                *(threadgroup float4*)(dst + 4) = C::hi(wreg[i]);
            }
        }
        for (uint c = tid; c < uint(XCH); c += NT) {
            const uint m = c / KC, cc = c % KC;
            const uint4 q = (m < uint(M) && k0 + cc < ke) ? x[ulong(m) * K8 + k0 + cc] : uint4(0);
            threadgroup float* dst = xs + m * LD + cc * 8;
            *(threadgroup float4*)dst = C::lo(q);
            *(threadgroup float4*)(dst + 4) = C::hi(q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint k1 = k0 + KC;
        if (k1 < ke) {
            for (int i = 0; i < WPT; ++i) {
                const uint c = tid + i * NT, row = c / KC, cc = c % KC;
                wreg[i] = (c < uint(WCH) && n0 + row < N && k1 + cc < ke) ? w[ulong(n0 + row) * K8 + k1 + cc] : uint4(0);
            }
        }
        for (int kk = 0; kk < KT; kk += 8) {
            simdgroup_float8x8 a[MT];
            for (int mt = 0; mt < MT; ++mt) simdgroup_load(a[mt], xs + mt * 8 * LD + kk, LD);
            for (int ns = 0; ns < NS; ++ns) {
                simdgroup_float8x8 b;
                simdgroup_load(b, ws + (sg * NS + ns) * 8 * LD + kk, LD, ulong2(0, 0), true);
                for (int mt = 0; mt < MT; ++mt) simdgroup_multiply_accumulate(acc[mt][ns], a[mt], b, acc[mt][ns]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int mt = 0; mt < MT; ++mt)
        for (int ns = 0; ns < NS; ++ns)
            simdgroup_store(acc[mt][ns], ws + mt * 8 * NTG + (sg * NS + ns) * 8, NTG);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool hb = has_bias != 0, split = splits > 1;
    for (uint i = tid; i < uint(M * NTG); i += NT) {
        const uint m = i / NTG, j = i % NTG, n = n0 + j;
        if (n >= N) continue;
        const float v = ws[m * NTG + j];
        if (split) part[(ulong(tgp.y) * M + m) * N + n] = v;
        else y[ulong(m) * N + n] = C::pack(v + (hb ? C::one(bias[n]) : 0.0f));
    }
}

template <typename C>
kernel void mrl_reduce(device const float* part [[buffer(0)]],
                       device const ushort* bias [[buffer(1)]],
                       device ushort* y [[buffer(2)]],
                       constant long& MN [[buffer(3)]],
                       constant long& N [[buffer(4)]],
                       constant long& splits [[buffer(5)]],
                       constant long& has_bias [[buffer(6)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= uint(MN)) return;
    float s = 0.0f;
    for (long p = 0; p < splits; ++p) s += part[ulong(p) * ulong(MN) + i];
    if (has_bias != 0) s += C::one(bias[i % uint(N)]);
    y[i] = C::pack(s);
}

// Variant "mmr": simdgroup 8x8 multiply-accumulate fed straight from device memory into
// registers, with no threadgroup memory and no barriers. It computes y^T = W x^T one
// 8x8 fragment at a time. In an 8x8 fragment, lane l holds row fm and columns fn, fn+1
// (fm = (q & 4) + (l / 2) % 4, fn = (q & 2) * 2 + (l % 2) * 2, q = l / 4; the layout MLX's
// steel GEMM relies on, and checked here by the correctness suite). The order of k inside
// a 32-element block is permuted, identically for W and x, so that each lane's share of
// four consecutive fragments is one contiguous 16-byte load of a weight row and two
// 8-byte loads of x rows: fragment b, column c = 2p + e maps to k = 8p + 4e + b.
//   NS  8-row weight tiles per simdgroup;  U  32-element K blocks per loop step;  PF  prefetch
// Grid: (ceil(N / (S*NS*8)) * 32*S, splits) threads, groups of (32*S, 1); splits as mrl_mma.
template <int M, int NS, int U>
inline void mmr_load(thread uint4 (&wq)[U][NS], thread uint2 (&xq)[U][(M + 7) / 8][2],
                     thread device const uint4* (&wrow)[NS], thread device const ushort* (&xrow)[(M + 7) / 8][2],
                     thread bool (&xok)[(M + 7) / 8][2], uint k, uint ke, uint wp, uint xp) {
    for (int u = 0; u < U; ++u) {
        const uint kw = k + 32 * u + wp, kx = k + 32 * u + xp;
        for (int ns = 0; ns < NS; ++ns) wq[u][ns] = kw < ke ? wrow[ns][kw / 8] : uint4(0);
        for (int mt = 0; mt < (M + 7) / 8; ++mt)
            for (int j = 0; j < 2; ++j)
                xq[u][mt][j] = (xok[mt][j] && kx < ke) ? *(device const uint2*)(xrow[mt][j] + kx) : uint2(0);
    }
}

template <typename C, int M, int NS, int U, bool PF>
kernel void mrl_mmr(device const ushort* x [[buffer(0)]],
                    device const uint4* w [[buffer(1)]],
                    device const ushort* bias [[buffer(2)]],
                    device ushort* y [[buffer(3)]],
                    constant long& K_ [[buffer(4)]],
                    constant long& N_ [[buffer(5)]],
                    constant long& has_bias [[buffer(6)]],
                    device float* part [[buffer(7)]],
                    constant long& klen [[buffer(8)]],
                    constant long& splits [[buffer(9)]],
                    uint2 tgp [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint nsg [[simdgroups_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int MT = (M + 7) / 8;
    const uint K = uint(K_), K8 = K / 8, N = uint(N_);
    const uint n0 = (tgp.x * nsg + sg) * NS * 8;
    if (n0 >= N) return;
    const uint q = lane / 4;
    const uint fm = (q & 4) + (lane / 2) % 4, fn = (q & 2) * 2 + (lane % 2) * 2;
    const uint kb = tgp.y * uint(klen), ke = min(K, kb + uint(klen));
    const uint wp = (fn / 2) * 8;                     // this lane's W offset inside a 32-block
    const uint xp = (fm / 2) * 8 + (fm % 2) * 4;      // this lane's x offset inside a 32-block

    device const uint4* wrow[NS];
    for (int ns = 0; ns < NS; ++ns) wrow[ns] = w + ulong(min(n0 + ns * 8 + fm, N - 1)) * K8;
    device const ushort* xrow[MT][2];
    bool xok[MT][2];
    for (int mt = 0; mt < MT; ++mt)
        for (int j = 0; j < 2; ++j) {
            const uint m = mt * 8 + fn + j;
            xok[mt][j] = m < uint(M);
            xrow[mt][j] = x + ulong(min(m, uint(M - 1))) * K;
        }

    simdgroup_float8x8 acc[NS][MT];
    for (int ns = 0; ns < NS; ++ns)
        for (int mt = 0; mt < MT; ++mt) acc[ns][mt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // PF: load the next step's weight and x registers before computing this step, so each
    // simdgroup keeps two steps of loads in flight.
    uint4 wq[U][NS];
    uint2 xq[U][MT][2];
    mmr_load<M, NS, U>(wq, xq, wrow, xrow, xok, kb, ke, wp, xp);
    for (uint k = kb; k < ke; k += 32 * U) {
        uint4 wn[U][NS];
        uint2 xn[U][MT][2];
        if (PF) mmr_load<M, NS, U>(wn, xn, wrow, xrow, xok, k + 32 * U, ke, wp, xp);
        for (int u = 0; u < U; ++u) {
            typedef typename C::FT T;
            vec<T, 4> wl[NS], wh[NS], x0[MT], x1[MT];
            for (int ns = 0; ns < NS; ++ns) { wl[ns] = C::flo(wq[u][ns]); wh[ns] = C::fhi(wq[u][ns]); }
            for (int mt = 0; mt < MT; ++mt) { x0[mt] = C::ffour(xq[u][mt][0]); x1[mt] = C::ffour(xq[u][mt][1]); }
            for (int b = 0; b < 4; ++b) {
                simdgroup_matrix<T, 8, 8> A[NS], B[MT];
                for (int ns = 0; ns < NS; ++ns) {
                    A[ns].thread_elements()[0] = wl[ns][b];
                    A[ns].thread_elements()[1] = wh[ns][b];
                }
                for (int mt = 0; mt < MT; ++mt) {
                    B[mt].thread_elements()[0] = x0[mt][b];
                    B[mt].thread_elements()[1] = x1[mt][b];
                }
                for (int ns = 0; ns < NS; ++ns)
                    for (int mt = 0; mt < MT; ++mt) simdgroup_multiply_accumulate(acc[ns][mt], A[ns], B[mt], acc[ns][mt]);
            }
        }
        if (PF) {
            for (int u = 0; u < U; ++u) {
                for (int ns = 0; ns < NS; ++ns) wq[u][ns] = wn[u][ns];
                for (int mt = 0; mt < MT; ++mt) { xq[u][mt][0] = xn[u][mt][0]; xq[u][mt][1] = xn[u][mt][1]; }
            }
        } else {
            mmr_load<M, NS, U>(wq, xq, wrow, xrow, xok, k + 32 * U, ke, wp, xp);
        }
    }
    const bool hb = has_bias != 0, split = splits > 1;
    for (int ns = 0; ns < NS; ++ns) {
        const uint n = n0 + ns * 8 + fm;
        if (n >= N) continue;
        const float b = (hb && !split) ? C::one(bias[n]) : 0.0f;
        for (int mt = 0; mt < MT; ++mt)
            for (int j = 0; j < 2; ++j) {
                const uint m = mt * 8 + fn + j;
                if (m >= uint(M)) continue;
                const float v = acc[ns][mt].thread_elements()[j];
                if (split) part[(ulong(tgp.y) * M + m) * N + n] = v;
                else y[ulong(m) * N + n] = C::pack(v + b);
            }
    }
}


// ---------------------------------------------------------------- round 2: 9 to 16 rows
//
// Round 1's register-fed variant (mmr) at 16 rows spends about as many issue cycles building
// fragments (16-bit to fp32 conversion, element moves, bounds-checked loads) as on the 8x8
// multiplies themselves (docs/review/2026-11-10-experiments/multirow/round2/README.md). The
// families below keep mmr's idea (each lane loads its share of a fragment straight into
// registers) and change the k order so that a lane's two elements of a weight fragment are one
// 32-bit word: fragment b, column c = 2p + e maps to k = 8p + 2b + e inside a 32-element block.
// A lane with fragment column pair (fn, fn+1) then loads one 16-byte chunk of its weight row
// and uses word b as fragment b: no conversion for fp16 (half operands, fp32 accumulators),
// two integer operations for bf16. The x fragment pairs two x rows at one k, so x costs one
// select per element (mmh, mmt) or nothing at all once x is repacked into fragment order (mmp).
//
//   mmh  register-fed; S simdgroups of a threadgroup share the same NS*8 weight rows and split K
//        between them (interleaved 32*U chunks), then add their partial tiles in threadgroup
//        memory. More parallel work per weight row without a second kernel.
//   mmp  mmh with x repacked by mrl_xpack into fragment order first (one extra small dispatch).
//   mmt  the weight staged through threadgroup memory, double-buffered, raw 16-bit, one barrier
//        per K tile; S simdgroups own different weight rows; x fragments from device memory.
// Every helper is force-inlined and every constant loop unrolled: without that, NS = 4 kept
// the accumulator and load arrays in stack memory and ran 3 times slower.
// All three: U 32-element blocks per step, optional register prefetch (PF), K split across
// threadgroups (splits > 1, with mrl_reduce), rows past N clamped and not written, x rows past
// M clamped (they only feed output columns that are never written), k outside [kb, ke) zeroed.

#define MRL_INLINE inline __attribute__((always_inline))

template <typename C, int NS, int MT>
MRL_INLINE void frag_mma(thread simdgroup_float8x8 (&acc)[NS][MT], thread const uint4 (&wq)[NS],
                     thread const uint4 (&xq)[MT][2], bool hi, bool packed) {
    typedef typename C::FT T;
    _Pragma("unroll") for (int b = 0; b < 4; ++b) {
        simdgroup_matrix<T, 8, 8> A[NS], B[MT];
        _Pragma("unroll") for (int ns = 0; ns < NS; ++ns) {
            const vec<T, 2> v = C::w2(wq[ns][b]);
            A[ns].thread_elements()[0] = v.x;
            A[ns].thread_elements()[1] = v.y;
        }
        _Pragma("unroll") for (int mt = 0; mt < MT; ++mt) {
            const vec<T, 2> v = packed ? C::w2(xq[mt][0][b]) : C::x2(xq[mt][0][b], xq[mt][1][b], hi);
            B[mt].thread_elements()[0] = v.x;
            B[mt].thread_elements()[1] = v.y;
        }
        _Pragma("unroll") for (int ns = 0; ns < NS; ++ns)
            _Pragma("unroll") for (int mt = 0; mt < MT; ++mt)
                simdgroup_multiply_accumulate(acc[ns][mt], A[ns], B[mt], acc[ns][mt]);
    }
}

// Load U 32-element blocks starting at uint4 index k8 (a multiple of 4). wrow/xrow already
// include this lane's offset (wo, xo, in uint4); with `check`, chunks at or past ke8 are zero.
template <int NS, int MT, int U, bool PK>
MRL_INLINE void frag_load(thread uint4 (&wq)[U][NS], thread uint4 (&xq)[U][MT][2],
                      thread device const uint4* const (&wrow)[NS], thread device const uint4* const (&xrow)[MT][2],
                      uint k8, uint ke8, uint wo, uint xo, bool check) {
    _Pragma("unroll") for (int u = 0; u < U; ++u) {
        const uint kk = k8 + 4 * u;
        const bool wok = !check || kk + wo < ke8, xok = !check || kk + (PK ? 0 : xo) < ke8;
        _Pragma("unroll") for (int ns = 0; ns < NS; ++ns) wq[u][ns] = wok ? wrow[ns][kk] : uint4(0);
        _Pragma("unroll") for (int mt = 0; mt < MT; ++mt) {
            if (PK) {
                xq[u][mt][0] = xok ? xrow[mt][0][(kk / 4) * MT * 32] : uint4(0);
            } else {
                _Pragma("unroll") for (int j = 0; j < 2; ++j) xq[u][mt][j] = xok ? xrow[mt][j][kk] : uint4(0);
            }
        }
    }
}

template <typename C, int M, int NS>
MRL_INLINE void store_frags(thread simdgroup_float8x8 (&acc)[NS][(M + 7) / 8], device ushort* y,
                        device const ushort* bias, device float* part, uint n0, uint N, bool hb,
                        bool split, uint split_index, uint fm, uint fn) {
    _Pragma("unroll") for (int ns = 0; ns < NS; ++ns) {
        const uint n = n0 + ns * 8 + fm;
        if (n >= N) continue;
        const float b = (hb && !split) ? C::one(bias[n]) : 0.0f;
        _Pragma("unroll") for (int mt = 0; mt < (M + 7) / 8; ++mt)
            _Pragma("unroll") for (int j = 0; j < 2; ++j) {
                const uint m = mt * 8 + fn + j;
                if (m >= uint(M)) continue;
                const float v = acc[ns][mt].thread_elements()[j];
                if (split) part[(ulong(split_index) * M + m) * N + n] = v;
                else y[ulong(m) * N + n] = C::pack(v + b);
            }
    }
}

template <typename C, int M, int NS, int U, int S, bool PF, bool PK>
kernel void mrl_mmh(device const uint4* x [[buffer(0)]],
                    device const uint4* w [[buffer(1)]],
                    device const ushort* bias [[buffer(2)]],
                    device ushort* y [[buffer(3)]],
                    constant long& K_ [[buffer(4)]],
                    constant long& N_ [[buffer(5)]],
                    constant long& has_bias [[buffer(6)]],
                    device float* part [[buffer(7)]],
                    constant long& klen [[buffer(8)]],
                    constant long& splits [[buffer(9)]],
                    uint2 tgp [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int MT = (M + 7) / 8, NR = NS * 8, MP = MT * 8;
    constexpr uint STEP8 = 4 * U * S;
    threadgroup float red[S > 1 ? S * NR * MP : 1];
    const uint K = uint(K_), K8 = K / 8, N = uint(N_);
    const uint n0 = tgp.x * NR;
    const uint q = lane / 4;
    const uint fm = (q & 4) + (lane / 2) % 4, fn = (q & 2) * 2 + (lane % 2) * 2;
    const uint wo = fn / 2, xo = fm / 2;
    const bool hi = (fm & 1) != 0;
    const uint kb8 = tgp.y * uint(klen) / 8, ke8 = min(K8, kb8 + uint(klen) / 8);

    device const uint4* wrow[NS];
    _Pragma("unroll") for (int ns = 0; ns < NS; ++ns) wrow[ns] = w + ulong(min(n0 + ns * 8 + fm, N - 1)) * K8 + wo;
    device const uint4* xrow[MT][2];
    _Pragma("unroll") for (int mt = 0; mt < MT; ++mt)
        _Pragma("unroll") for (int j = 0; j < 2; ++j)
            xrow[mt][j] = PK ? x + mt * 32 + lane : x + ulong(min(uint(mt * 8 + fn + j), uint(M - 1))) * K8 + xo;

    simdgroup_float8x8 acc[NS][MT];
    _Pragma("unroll") for (int ns = 0; ns < NS; ++ns)
        _Pragma("unroll") for (int mt = 0; mt < MT; ++mt) acc[ns][mt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint k8 = kb8 + sg * 4 * U;
    uint4 wq[U][NS], xq[U][MT][2];
    if (PF) {
        bool full = k8 + 4 * U <= ke8;
        if (full) frag_load<NS, MT, U, PK>(wq, xq, wrow, xrow, k8, ke8, wo, xo, false);
        while (full) {
            const uint nk = k8 + STEP8;
            const bool nfull = nk + 4 * U <= ke8;
            uint4 wn[U][NS], xn[U][MT][2];
            if (nfull) frag_load<NS, MT, U, PK>(wn, xn, wrow, xrow, nk, ke8, wo, xo, false);
            _Pragma("unroll") for (int u = 0; u < U; ++u) frag_mma<C, NS, MT>(acc, wq[u], xq[u], hi, PK);
            if (nfull) {
                _Pragma("unroll") for (int u = 0; u < U; ++u) {
                    _Pragma("unroll") for (int ns = 0; ns < NS; ++ns) wq[u][ns] = wn[u][ns];
                    _Pragma("unroll") for (int mt = 0; mt < MT; ++mt) { xq[u][mt][0] = xn[u][mt][0]; xq[u][mt][1] = xn[u][mt][1]; }
                }
            }
            k8 = nk;
            full = nfull;
        }
    } else {
        for (; k8 + 4 * U <= ke8; k8 += STEP8) {
            frag_load<NS, MT, U, PK>(wq, xq, wrow, xrow, k8, ke8, wo, xo, false);
            _Pragma("unroll") for (int u = 0; u < U; ++u) frag_mma<C, NS, MT>(acc, wq[u], xq[u], hi, PK);
        }
    }
    if (k8 < ke8) {  // at most one partial chunk per simdgroup
        frag_load<NS, MT, U, PK>(wq, xq, wrow, xrow, k8, ke8, wo, xo, true);
        _Pragma("unroll") for (int u = 0; u < U; ++u) frag_mma<C, NS, MT>(acc, wq[u], xq[u], hi, PK);
    }

    const bool hb = has_bias != 0, split = splits > 1;
    if (S == 1) {
        store_frags<C, M, NS>(acc, y, bias, part, n0, N, hb, split, tgp.y, fm, fn);
        return;
    }
    _Pragma("unroll") for (int ns = 0; ns < NS; ++ns)
        _Pragma("unroll") for (int mt = 0; mt < MT; ++mt)
            simdgroup_store(acc[ns][mt], red + (sg * NR + ns * 8) * MP + mt * 8, MP);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = sg * 32 + lane; i < uint(M * NR); i += S * 32) {
        const uint m = i / NR, nl = i % NR, n = n0 + nl;
        if (n >= N) continue;
        float v = 0.0f;
        _Pragma("unroll") for (int s2 = 0; s2 < S; ++s2) v += red[(s2 * NR + nl) * MP + m];
        if (split) part[(ulong(tgp.y) * M + m) * N + n] = v;
        else y[ulong(m) * N + n] = C::pack(v + (hb ? C::one(bias[n]) : 0.0f));
    }
}

// x[M, K] into fragment order for mmp: for each 32-element block, row tile mt and lane, one
// uint4 whose word b holds (x[mt*8+fn][k], x[mt*8+fn+1][k]) at k = block + 8*(fm/2) + 2b + fm%2.
// Rows past M and k past K are zero.
template <int M>
kernel void mrl_xpack(device const ushort* x [[buffer(0)]],
                      device uint4* xp [[buffer(1)]],
                      constant long& K_ [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    constexpr int MT = (M + 7) / 8;
    const uint K = uint(K_), nblk = (K + 31) / 32;
    if (i >= nblk * MT * 32) return;
    const uint lane = i % 32, mt = (i / 32) % MT, blk = i / (32 * MT);
    const uint q = lane / 4;
    const uint fm = (q & 4) + (lane / 2) % 4, fn = (q & 2) * 2 + (lane % 2) * 2;
    const uint m0 = mt * 8 + fn, m1 = m0 + 1;
    const uint kbase = blk * 32 + 8 * (fm / 2) + (fm & 1);
    uint4 out;
    _Pragma("unroll") for (int b = 0; b < 4; ++b) {
        const uint k = kbase + 2 * b;
        const uint lo = (m0 < uint(M) && k < K) ? uint(x[ulong(m0) * K + k]) : 0u;
        const uint hi = (m1 < uint(M) && k < K) ? uint(x[ulong(m1) * K + k]) : 0u;
        out[b] = lo | (hi << 16);
    }
    xp[i] = out;
}

template <typename C, int M, int NS, int U, int S>
kernel void mrl_mmt(device const uint4* x [[buffer(0)]],
                    device const uint4* w [[buffer(1)]],
                    device const ushort* bias [[buffer(2)]],
                    device ushort* y [[buffer(3)]],
                    constant long& K_ [[buffer(4)]],
                    constant long& N_ [[buffer(5)]],
                    constant long& has_bias [[buffer(6)]],
                    device float* part [[buffer(7)]],
                    constant long& klen [[buffer(8)]],
                    constant long& splits [[buffer(9)]],
                    uint2 tgp [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int MT = (M + 7) / 8, NTG = S * NS * 8;
    constexpr int KC = 4 * U, LDC = KC + 1;          // uint4 per row per tile; padded row stride
    constexpr int CH = NTG * KC, NT = S * 32, WPT = (CH + NT - 1) / NT;
    threadgroup uint4 ws[2 * NTG * LDC];
    const uint K = uint(K_), K8 = K / 8, N = uint(N_);
    const uint n0 = tgp.x * NTG, tid = sg * 32 + lane;
    const uint q = lane / 4;
    const uint fm = (q & 4) + (lane / 2) % 4, fn = (q & 2) * 2 + (lane % 2) * 2;
    const uint wo = fn / 2, xo = fm / 2;
    const bool hi = (fm & 1) != 0;
    const uint kb8 = tgp.y * uint(klen) / 8, ke8 = min(K8, kb8 + uint(klen) / 8);
    const uint ntiles = (ke8 - kb8 + KC - 1) / KC;

    device const uint4* xrow[MT][2];
    _Pragma("unroll") for (int mt = 0; mt < MT; ++mt)
        _Pragma("unroll") for (int j = 0; j < 2; ++j) xrow[mt][j] = x + ulong(min(uint(mt * 8 + fn + j), uint(M - 1))) * K8 + xo;
    simdgroup_float8x8 acc[NS][MT];
    _Pragma("unroll") for (int ns = 0; ns < NS; ++ns)
        _Pragma("unroll") for (int mt = 0; mt < MT; ++mt) acc[ns][mt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint4 reg[WPT];
    _Pragma("unroll") for (int i = 0; i < WPT; ++i) {
        const uint c = tid + i * NT, row = c / KC, cc = c % KC, k8 = kb8 + cc;
        reg[i] = (c < uint(CH) && k8 < ke8) ? w[ulong(min(n0 + row, N - 1)) * K8 + k8] : uint4(0);
    }
    _Pragma("unroll") for (int i = 0; i < WPT; ++i) {
        const uint c = tid + i * NT;
        if (c < uint(CH)) ws[(c / KC) * LDC + c % KC] = reg[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ntiles > 1)
        _Pragma("unroll") for (int i = 0; i < WPT; ++i) {
            const uint c = tid + i * NT, row = c / KC, cc = c % KC, k8 = kb8 + KC + cc;
            reg[i] = (c < uint(CH) && k8 < ke8) ? w[ulong(min(n0 + row, N - 1)) * K8 + k8] : uint4(0);
        }
    for (uint t = 0; t < ntiles; ++t) {
        threadgroup const uint4* buf = ws + (t & 1) * NTG * LDC + (sg * NS * 8 + fm) * LDC + wo;
        const uint k8 = kb8 + t * KC;
        _Pragma("unroll") for (int u = 0; u < U; ++u) {
            uint4 wq[NS], xq[MT][2];
            const bool xok = k8 + 4 * u + xo < ke8;
            _Pragma("unroll") for (int ns = 0; ns < NS; ++ns) wq[ns] = buf[ns * 8 * LDC + 4 * u];
            _Pragma("unroll") for (int mt = 0; mt < MT; ++mt)
                _Pragma("unroll") for (int j = 0; j < 2; ++j) xq[mt][j] = xok ? xrow[mt][j][k8 + 4 * u] : uint4(0);
            frag_mma<C, NS, MT>(acc, wq, xq, hi, false);
        }
        if (t + 1 < ntiles) {
            threadgroup uint4* nb = ws + ((t + 1) & 1) * NTG * LDC;
            _Pragma("unroll") for (int i = 0; i < WPT; ++i) {
                const uint c = tid + i * NT;
                if (c < uint(CH)) nb[(c / KC) * LDC + c % KC] = reg[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t + 2 < ntiles)
            _Pragma("unroll") for (int i = 0; i < WPT; ++i) {
                const uint c = tid + i * NT, row = c / KC, cc = c % KC, k8n = kb8 + (t + 2) * KC + cc;
                reg[i] = (c < uint(CH) && k8n < ke8) ? w[ulong(min(n0 + row, N - 1)) * K8 + k8n] : uint4(0);
            }
    }
    const uint nsg0 = n0 + sg * NS * 8;
    if (nsg0 >= N) return;
    store_frags<C, M, NS>(acc, y, bias, part, nsg0, N, has_bias != 0, splits > 1, tgp.y, fm, fn);
}
