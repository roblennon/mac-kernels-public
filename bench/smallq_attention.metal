// Small-query attention over a long fp16 KV cache: T = 1 to 16 new query rows, grouped-query
// heads, causal inside the new block. Experiment E of the mps-decode thread.
//
//   out[b, h, t, :] = softmax_n(scale * q[b, h, t, :] . k[b, h / G, n, :]) @ v[b, h / G, n, :]
//   over n = 0 .. pos + t, with G = Hq / Hkv.
//
// Two passes, flash-decoding style. Pass 1 (sqa_pass1) splits the visible keys 0 .. pos + T - 1
// into NS contiguous chunks, one threadgroup per (chunk, kv head, row group, batch). A row group
// is up to 8 * RT query rows: the (query head sharing this kv head, new token) pairs, flattened
// as r = g * T + t. The threadgroup walks its chunk in sub-blocks of BK keys. Each key and value
// row of a sub-block is loaded from device memory once per threadgroup, straight into registers
// and simdgroup 8x8 fragments, and applied to every query row in the group, so the cache crosses
// the memory bus once per kv head and row group. Per sub-block:
//   A. scores S^T = K Q^T into threadgroup memory as [row][key]. By default (QREG false) the
//      simdgroups split the 8-key tiles and each tile meets every 8-row query tile, with the Q^T
//      fragments read from threadgroup memory; with QREG each simdgroup instead keeps one row
//      tile's Q^T fragments in registers. The head dimension is permuted (below) for K and Q
//      alike, so a lane's share of four consecutive K fragments is one 16-byte load.
//   B. online softmax, one simdgroup per row, fp32, in the exp2 domain: reference max m, running
//      sum l. m moves only when a block's max exceeds it by more than SLACK (then the
//      accumulators are rescaled by alpha = exp2(m_old - m_new)); otherwise p = exp2(s - m) stays
//      below 2^SLACK. Masked keys are -inf; a row whose keys are all masked so far keeps
//      m = -inf, alpha = 1, p = 0, so a fully masked block contributes nothing and no NaN forms.
//   C. O = diag(alpha) O + P V (the diag product only when some row moved its max), simdgroups
//      split the head dimension (32 columns each), so each value row is loaded once per
//      threadgroup.
// Pass 1 writes the unnormalised O (fp32), m and l per (row, chunk). Pass 2 (sqa_reduce)
// combines the chunks with weights exp2(m_j - max m) and divides by the combined l.
//
// Keys at or beyond pos + T are never read: the key index is clamped to pos + T - 1 before the
// load and the score is masked, so whatever the cache holds past the valid entries (stale
// values, NaN) cannot reach the output. Accumulation is fp32 throughout; the output is rounded
// once to fp16.
//
// Pass 1 has two entry points around one body (sqa_pass1_body). sqa_pass1 takes pos as a host
// value and the host sizes the chunks from pos + T. sqa_pass1_dev reads pos from a device buffer
// (one value per batch element, or one shared) and the host sizes the chunks from the cache
// length Lmax, so the launch does not depend on pos: chunks that start at or past pos + T run no
// sub-block and the reduce gives them weight 0. That entry is the one a compiled model uses.
//
// Fragment layout (Apple GPU, the layout MLX's steel GEMM relies on): in an 8x8 fragment, lane
// l holds row fm and columns fn, fn + 1, with q = l / 4, fm = (q & 4) + (l / 2) % 4,
// fn = (q & 2) * 2 + (l % 2) * 2.
// Head-dimension permutation inside each 32-element block j, used for the contraction in pass
// A (applied identically to K and Q) and for the output columns in pass C (undone on the
// store): fragment step b (0..3), column c maps to d = 32 j + (c / 2) * 8 + 2 b + (c & 1).
// A lane's elements for steps b = 0..3 are then the four 32-bit words of the 16-byte chunk at
// d = 32 j + (fn / 2) * 8.
//
// Template parameters: QT    operand type of the score product (float, or half with fp32
//                            accumulation; products of two halves are exact in fp32)
//                      D     head dim (64 or 128); S = D / 32 simdgroups per threadgroup
//                      RT    8-row query tiles per threadgroup (1, 2 or 4, dividing S)
//                      BK    keys per sub-block (a multiple of 32 and of 8 * S)
//                      QREG  pass A work split (above)
// Host contract (checked in smallq_attention.py): fp16 tensors on one device, last dim
// contiguous, K and V row strides and offsets multiples of 8 elements (16-byte loads),
// pos + T <= Lmax (sqa_pass1_dev clamps pos to 0 .. Lmax - T on the GPU), Hq % Hkv == 0.
#include <metal_stdlib>
using namespace metal;

// K for pass A (this simdgroup's key tiles, all head-dim words) and V for pass C (every key tile,
// this simdgroup's 32 columns) of the sub-block starting at key k0, into registers.
template <int S, int KTS, int KT>
inline void load_kv(thread uint4 (&kw)[KTS][S], thread uint4 (&vw)[KT], device const half* kbase,
                    device const half* vbase, long ksl, long vsl, int k0, int kgA, int KG, int Lk,
                    uint fm, uint fn, uint sg) {
    _Pragma("unroll") for (int u = 0; u < KTS; ++u) {
        const int n = min(k0 + (kgA + u * KG) * 8 + int(fm), Lk - 1);
        device const uint4* krow = (device const uint4*)(kbase + long(n) * ksl) + fn / 2;
        _Pragma("unroll") for (int j = 0; j < S; ++j) kw[u][j] = krow[4 * j];
    }
    _Pragma("unroll") for (int kk = 0; kk < KT; ++kk) {
        const int n = min(k0 + kk * 8 + int(fm), Lk - 1);
        vw[kk] = *((device const uint4*)(vbase + long(n) * vsl) + 4 * sg + fn / 2);
    }
}

constant constexpr float SLACK = 8.0f;  // log2 units: probabilities stay below 2^8 before rescaling

// Pass 1 arguments that do not depend on how pos reaches the kernel.
struct SqaArgs {
    long qsb, qsh, qst, ksb, ksh, ksl, vsb, vsh, vsl;
    int T, G, pos, chunk, NS, RG, shift;
    float scale;
};

// The body of pass 1, shared by the host-position entry (sqa_pass1) and the device-position entry
// (sqa_pass1_dev). Threadgroup memory is declared by the entries (Metal allows it only at kernel
// scope) and passed in. A chunk that starts at or beyond the valid keys pos + T runs no sub-block
// and writes m = -inf, l = 0, O = 0, which the reduce weights by 0.
template <typename QT, int D, int RT, int BK, bool QREG>
inline void sqa_pass1_body(device const half* q, device const half* k, device const half* v,
                           device float* part_o, device float* part_ml, const SqaArgs A,
                           uint3 tgp, uint3 ntg, uint sg, uint lane, uint tid,
                           threadgroup QT* qs, threadgroup float* ss, threadgroup float* alpha,
                           threadgroup int* rescaled) {
    constexpr int S = D / 32;          // simdgroups; simdgroup sg owns output columns 32 sg .. 32 sg + 31
    constexpr int NT = 32 * S;
    constexpr int RP = 8 * RT;         // padded query rows per threadgroup
    constexpr int KT = BK / 8;         // 8-key tiles per sub-block
    // Pass A work split. QREG: simdgroup sg takes query row tile sg % RT and every KG-th 8-key
    // tile from sg / RT, with its Q^T fragments held in registers for the whole chunk. Otherwise
    // simdgroup sg takes every S-th key tile from sg against all RT row tiles, reading the Q^T
    // fragments from threadgroup memory (fewer registers; measured faster at every RT).
    constexpr int KG = QREG ? S / RT : S;
    constexpr int KTS = KT / KG;       // key tiles per simdgroup in pass A
    constexpr int DS = D / 8;          // fragment steps along the head dim
    constexpr int RPS = RP / S;        // softmax rows per simdgroup
    constexpr int KPL = BK / 32;       // keys per lane in the softmax
    constexpr int LDQ = D + 4;         // padded row strides: column loads of 8 rows spread over banks
    constexpr int LDS = BK + 4;
    static_assert(D % 32 == 0 && BK % 32 == 0 && S % RT == 0 && KT % KG == 0 && RP % S == 0,
                  "bad configuration");

    const long qsb = A.qsb, qsh = A.qsh, qst = A.qst, ksb = A.ksb, ksh = A.ksh, ksl = A.ksl;
    const long vsb = A.vsb, vsh = A.vsh, vsl = A.vsl;
    const float scale = A.scale;
    const int T = A.T, G = A.G, pos = A.pos, RG = A.RG, NS = A.NS;
    const int R = G * T;                          // query rows per kv head
    const int split = int(tgp.x);
    const int kvh = int(tgp.y) / RG, rg = int(tgp.y) % RG;
    const int b = int(tgp.z);
    const long Hq = long(ntg.y / uint(RG)) * G;
    const int r0 = rg * RP;
    const int Lk = pos + T;                       // valid keys
    const int ks = split * A.chunk;
    const int ke = min(ks + A.chunk, Lk);

    // Stage Q: qs[r][8 * step + c] = Q[row r][d(step, c)], zero for padding rows.
    for (int e = int(tid); e < RP * D; e += NT) {
        const int r = e / D, cc = e % D;
        const int step = cc / 8, c = cc % 8, j = step / 4, bb = step % 4;
        const int d = 32 * j + (c / 2) * 8 + 2 * bb + (c & 1);
        const int rr = r0 + r;
        float val = 0.0f;
        if (rr < R) {
            const int g = rr / T, t = rr % T;
            const long hq = long(kvh) * G + g;
            val = float(q[long(b) * qsb + hq * qsh + long(t) * qst + d]);
        }
        qs[r * LDQ + cc] = QT(val);
    }

    const uint qd = lane / 4;
    const uint fm = (qd & 4) + (lane / 2) % 4, fn = (qd & 2) * 2 + (lane % 2) * 2;
    device const half* kbase = k + long(b) * ksb + long(kvh) * ksh;
    device const half* vbase = v + long(b) * vsb + long(kvh) * vsh;

    // Softmax state for the rows this simdgroup owns (rows sg * RPS + i), uniform across lanes.
    float m_run[RPS], l_run[RPS];
    int lim[RPS];
    _Pragma("unroll") for (int i = 0; i < RPS; ++i) {
        m_run[i] = -INFINITY;
        l_run[i] = 0.0f;
        const int rr = r0 + int(sg) * RPS + i;
        lim[i] = min(pos + (rr % T) + A.shift, Lk - 1);   // last visible key of this row
    }
    const float sl2 = scale * M_LOG2E_F;

    simdgroup_matrix<float, 8, 8> oacc[RT][4];
    _Pragma("unroll") for (int rt = 0; rt < RT; ++rt)
        _Pragma("unroll") for (int c = 0; c < 4; ++c) oacc[rt][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Pass A: this simdgroup takes row tile rtA (QREG) or all row tiles, and every KG-th key tile
    // from kgA. With QREG its Q^T fragments are loaded into registers once, for the whole chunk.
    const int rtA = QREG ? int(sg) % RT : 0, kgA = QREG ? int(sg) / RT : int(sg);
    constexpr int QF = QREG ? DS : 1;
    simdgroup_matrix<QT, 8, 8> qf[QF];
    if (QREG) {
        _Pragma("unroll") for (int st = 0; st < QF; ++st)
            simdgroup_load(qf[st], qs + rtA * 8 * LDQ + 8 * st, LDQ, ulong2(0, 0), true);
    }

    for (int k0 = ks; k0 < ke; k0 += BK) {
        // This sub-block's K and V, all loads issued together. (Loading the next sub-block's
        // before computing this one was measured: no gain.)
        uint4 kw[KTS][S], vw[KT];
        load_kv<S, KTS, KT>(kw, vw, kbase, vbase, ksl, vsl, k0, kgA, KG, Lk, fm, fn, sg);

        // A. Scores S^T (keys x rows) for this simdgroup's key tiles, stored transposed as
        //    ss[row][key]. The independent accumulation chains are interleaved.
        if (QREG) {
            simdgroup_matrix<float, 8, 8> acc[KTS];
            _Pragma("unroll") for (int u = 0; u < KTS; ++u) acc[u] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            _Pragma("unroll") for (int j = 0; j < S; ++j) {
                _Pragma("unroll") for (int bb = 0; bb < 4; ++bb) {
                    _Pragma("unroll") for (int u = 0; u < KTS; ++u) {
                        const half2 h = as_type<half2>(kw[u][j][bb]);
                        simdgroup_matrix<QT, 8, 8> a;
                        a.thread_elements()[0] = QT(h.x);
                        a.thread_elements()[1] = QT(h.y);
                        simdgroup_multiply_accumulate(acc[u], a, qf[(4 * j + bb) % QF], acc[u]);
                    }
                }
            }
            _Pragma("unroll") for (int u = 0; u < KTS; ++u)
                simdgroup_store(acc[u], ss + rtA * 8 * LDS + (kgA + u * KG) * 8, LDS, ulong2(0, 0), true);
        } else {
            _Pragma("unroll") for (int u = 0; u < KTS; ++u) {
                simdgroup_matrix<float, 8, 8> acc[RT];
                _Pragma("unroll") for (int rt = 0; rt < RT; ++rt) acc[rt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                _Pragma("unroll") for (int j = 0; j < S; ++j) {
                    _Pragma("unroll") for (int bb = 0; bb < 4; ++bb) {
                        const half2 h = as_type<half2>(kw[u][j][bb]);
                        simdgroup_matrix<QT, 8, 8> a;
                        a.thread_elements()[0] = QT(h.x);
                        a.thread_elements()[1] = QT(h.y);
                        _Pragma("unroll") for (int rt = 0; rt < RT; ++rt) {
                            simdgroup_matrix<QT, 8, 8> bq;
                            simdgroup_load(bq, qs + rt * 8 * LDQ + 8 * (4 * j + bb), LDQ, ulong2(0, 0), true);
                            simdgroup_multiply_accumulate(acc[rt], a, bq, acc[rt]);
                        }
                    }
                }
                _Pragma("unroll") for (int rt = 0; rt < RT; ++rt)
                    simdgroup_store(acc[rt], ss + rt * 8 * LDS + (kgA + u * KG) * 8, LDS, ulong2(0, 0), true);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // B. Online softmax per row; probabilities replace the scores in place. The reference
        //    max m moves only when a block's max exceeds it by more than SLACK (log2 units), so
        //    p <= 2^SLACK and the accumulators are rescaled rarely; m, l and O stay consistent.
        bool moved = false;
        _Pragma("unroll") for (int i = 0; i < RPS; ++i) {
            const int r = int(sg) * RPS + i;
            float sv[KPL];
            float mx = -INFINITY;
            _Pragma("unroll") for (int e = 0; e < KPL; ++e) {
                const int kk = int(lane) + 32 * e;
                const float s = ss[r * LDS + kk] * sl2;
                sv[e] = (k0 + kk <= lim[i]) ? s : -INFINITY;
                mx = max(mx, sv[e]);
            }
            mx = simd_max(mx);
            const float mold = m_run[i];
            const float mnew = mx > mold + SLACK ? mx : mold;   // also takes the first finite max
            float a = 1.0f, psum = 0.0f;
            if (mnew != mold) {
                a = exp2(mold - mnew);                  // 0 when mold is -inf
                moved = true;
            }
            if (mnew != -INFINITY) {
                _Pragma("unroll") for (int e = 0; e < KPL; ++e) {
                    const float p = exp2(sv[e] - mnew); // 0 for masked keys
                    ss[r * LDS + int(lane) + 32 * e] = p;
                    psum += p;
                }
            } else {
                _Pragma("unroll") for (int e = 0; e < KPL; ++e) ss[r * LDS + int(lane) + 32 * e] = 0.0f;
            }
            psum = simd_sum(psum);
            l_run[i] = l_run[i] * a + psum;
            m_run[i] = mnew;
            if (lane == 0) alpha[r] = a;
        }
        if (lane == 0) rescaled[sg] = moved ? 1 : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // C. Rescale and accumulate P V for this simdgroup's 32 output columns.
        //    The rescale is a product with diag(alpha), skipped when no row moved its max: scaling
        //    the accumulators' elements in place through thread_elements() was measured 1.7x slower.
        bool any = false;
        _Pragma("unroll") for (int i = 0; i < S; ++i) any = any || rescaled[i] != 0;
        if (any) _Pragma("unroll") for (int rt = 0; rt < RT; ++rt) {
            const float a = alpha[rt * 8 + fm];
            simdgroup_matrix<float, 8, 8> dg;
            dg.thread_elements()[0] = fm == fn ? a : 0.0f;
            dg.thread_elements()[1] = fm == fn + 1 ? a : 0.0f;
            _Pragma("unroll") for (int c = 0; c < 4; ++c) simdgroup_multiply(oacc[rt][c], dg, oacc[rt][c]);
        }
        _Pragma("unroll") for (int kk = 0; kk < KT; ++kk) {
            simdgroup_matrix<float, 8, 8> vb[4];
            _Pragma("unroll") for (int c = 0; c < 4; ++c) {
                const half2 h = as_type<half2>(vw[kk][c]);
                vb[c].thread_elements()[0] = float(h.x);
                vb[c].thread_elements()[1] = float(h.y);
            }
            _Pragma("unroll") for (int rt = 0; rt < RT; ++rt) {
                simdgroup_matrix<float, 8, 8> pa;
                simdgroup_load(pa, ss + rt * 8 * LDS + kk * 8, LDS);
                _Pragma("unroll") for (int c = 0; c < 4; ++c) simdgroup_multiply_accumulate(oacc[rt][c], pa, vb[c], oacc[rt][c]);
            }
        }
        // Pass C reads the probabilities in ss; the next sub-block's pass A overwrites them.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Partials: part_o[b, hq, t, split, d] (fp32), part_ml[b, hq, t, split, {m, l}].
    _Pragma("unroll") for (int rt = 0; rt < RT; ++rt) {
        const int rr = r0 + rt * 8 + int(fm);
        if (rr >= R) continue;
        const long hq = long(kvh) * G + rr / T;
        const long base = (((long(b) * Hq + hq) * T + rr % T) * NS + split) * D + 32 * sg + (fn / 2) * 8;
        _Pragma("unroll") for (int c = 0; c < 4; ++c)
            *(device float2*)(part_o + base + 2 * c) =
                float2(oacc[rt][c].thread_elements()[0], oacc[rt][c].thread_elements()[1]);
    }
    if (lane == 0) {
        _Pragma("unroll") for (int i = 0; i < RPS; ++i) {
            const int rr = r0 + int(sg) * RPS + i;
            if (rr >= R) continue;
            const long hq = long(kvh) * G + rr / T;
            const long idx = ((long(b) * Hq + hq) * T + rr % T) * NS + split;
            *(device float2*)(part_ml + 2 * idx) = float2(m_run[i], l_run[i]);
        }
    }
}

// Host-position entry: pos is a host value; the host sizes the chunks from pos + T.
template <typename QT, int D, int RT, int BK, bool QREG>
kernel void sqa_pass1(device const half* q [[buffer(0)]],
                      device const half* k [[buffer(1)]],
                      device const half* v [[buffer(2)]],
                      device float* part_o [[buffer(3)]],
                      device float* part_ml [[buffer(4)]],
                      constant long& qsb [[buffer(5)]],
                      constant long& qsh [[buffer(6)]],
                      constant long& qst [[buffer(7)]],
                      constant long& ksb [[buffer(8)]],
                      constant long& ksh [[buffer(9)]],
                      constant long& ksl [[buffer(10)]],
                      constant long& vsb [[buffer(11)]],
                      constant long& vsh [[buffer(12)]],
                      constant long& vsl [[buffer(13)]],
                      constant long& T_ [[buffer(14)]],
                      constant long& G_ [[buffer(15)]],
                      constant long& pos_ [[buffer(16)]],
                      constant long& chunk_ [[buffer(17)]],
                      constant long& NS_ [[buffer(18)]],
                      constant long& RG_ [[buffer(19)]],
                      constant long& shift_ [[buffer(20)]],
                      constant float& scale [[buffer(21)]],
                      uint3 tgp [[threadgroup_position_in_grid]],
                      uint3 ntg [[threadgroups_per_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint tid [[thread_index_in_threadgroup]]) {
    threadgroup QT qs[8 * RT * (D + 4)];
    threadgroup float ss[8 * RT * (BK + 4)];
    threadgroup float alpha[8 * RT];
    threadgroup int rescaled[D / 32];
    const SqaArgs A = {qsb, qsh, qst, ksb, ksh, ksl, vsb, vsh, vsl, int(T_), int(G_), int(pos_),
                       int(chunk_), int(NS_), int(RG_), int(shift_), scale};
    sqa_pass1_body<QT, D, RT, BK, QREG>(q, k, v, part_o, part_ml, A, tgp, ntg, sg, lane, tid,
                                        qs, ss, alpha, rescaled);
}

// Device-position entry: pos is read on the GPU, posd[b * pstride] for batch element b (pstride 0
// shares one value; transformers' position_ids[:, 0]), so nothing waits on the host. The host
// sizes the chunks from Lmax, so the grid does not depend on pos; chunks past pos + T do no work.
// pos is clamped to 0 .. Lmax - T, so a bad value can give a wrong answer but never reads outside
// the cache.
template <typename QT, int D, int RT, int BK, bool QREG>
kernel void sqa_pass1_dev(device const half* q [[buffer(0)]],
                          device const half* k [[buffer(1)]],
                          device const half* v [[buffer(2)]],
                          device float* part_o [[buffer(3)]],
                          device float* part_ml [[buffer(4)]],
                          constant long& qsb [[buffer(5)]],
                          constant long& qsh [[buffer(6)]],
                          constant long& qst [[buffer(7)]],
                          constant long& ksb [[buffer(8)]],
                          constant long& ksh [[buffer(9)]],
                          constant long& ksl [[buffer(10)]],
                          constant long& vsb [[buffer(11)]],
                          constant long& vsh [[buffer(12)]],
                          constant long& vsl [[buffer(13)]],
                          constant long& T_ [[buffer(14)]],
                          constant long& G_ [[buffer(15)]],
                          device const long* posd [[buffer(16)]],
                          constant long& chunk_ [[buffer(17)]],
                          constant long& NS_ [[buffer(18)]],
                          constant long& RG_ [[buffer(19)]],
                          constant long& shift_ [[buffer(20)]],
                          constant float& scale [[buffer(21)]],
                          constant long& pstride [[buffer(22)]],
                          constant long& Lmax_ [[buffer(23)]],
                          uint3 tgp [[threadgroup_position_in_grid]],
                          uint3 ntg [[threadgroups_per_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup QT qs[8 * RT * (D + 4)];
    threadgroup float ss[8 * RT * (BK + 4)];
    threadgroup float alpha[8 * RT];
    threadgroup int rescaled[D / 32];
    const long p = posd[long(tgp.z) * pstride], hi = Lmax_ - T_;
    const long pc = p < 0 ? 0 : (p > hi ? hi : p);
    const SqaArgs A = {qsb, qsh, qst, ksb, ksh, ksl, vsb, vsh, vsl, int(T_), int(G_), int(pc),
                       int(chunk_), int(NS_), int(RG_), int(shift_), scale};
    sqa_pass1_body<QT, D, RT, BK, QREG>(q, k, v, part_o, part_ml, A, tgp, ntg, sg, lane, tid,
                                        qs, ss, alpha, rescaled);
}

// Pass 2: one threadgroup of D / 4 threads per output row (b, hq, t); thread i owns columns
// 4 i .. 4 i + 3. Chunks whose row saw no visible key (m = -inf) get weight 0.
template <int D>
kernel void sqa_reduce(device const float* part_o [[buffer(0)]],
                       device const float* part_ml [[buffer(1)]],
                       device half* out [[buffer(2)]],
                       constant long& NS_ [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint i [[thread_index_in_threadgroup]]) {
    const int NS = int(NS_);
    device const float2* ml = (device const float2*)part_ml + long(row) * NS;
    float M = -INFINITY;
    for (int s = 0; s < NS; ++s) M = max(M, ml[s].x);
    float L = 0.0f;
    float4 acc = 0.0f;
    device const float4* po = (device const float4*)(part_o + long(row) * NS * D) + i;
    for (int s = 0; s < NS; ++s) {
        const float2 e = ml[s];
        const float w = (e.x == -INFINITY) ? 0.0f : exp2(e.x - M);
        L += w * e.y;
        acc += w * po[s * (D / 4)];
    }
    *((device half4*)(out + long(row) * D) + i) = half4(acc / L);
}
