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

// Row-vector path (sqa_vec): for few rows, R = G * T up to 8, where the 8-row simdgroup tiles of
// pass 1 would be mostly padding. No simdgroup matrices, no threadgroup memory, no barriers. Each
// simdgroup takes one row group (RV of the R rows) of one key chunk. A threadgroup holds RGT = A.RG
// row groups times nsg / RGT chunks, so the row groups of a chunk read the same keys at about the
// same time, mostly from the core's cache; with more row groups than RGT, RGY threadgroups along
// grid y share a (kv head, chunk) and each takes RGT of them. A simdgroup works as
// 32 / LPK slots of LPK lanes; a slot takes one key at a time and its LPK lanes split the head dim,
// lane `sub` owning the W-element runs at c * W * LPK + W * sub (W = min(E, 8) with E = D / LPK
// elements per lane, c < E / W), so each load instruction of a slot reads W * LPK * 2 contiguous
// bytes of a K or V row. Per step a simdgroup takes U consecutive keys per slot, loaded together.
// The score for each of the R rows is a dot over the lane's elements and log2(LPK) xor shuffles
// inside the slot. Q is held in registers, pre-multiplied by scale * log2(e). Each slot keeps its
// own online softmax per row, with the same lazy reference max as pass 1 (it moves only when a
// step's max exceeds it by more than SLACK); the slots are merged by shuffles at the end, and the
// chunk's (O, m, l) go to the same partial buffers, so the same reduce combines the chunks.
// Template RV = rows per group (compiled per RV); rows past R in the last group are computed as
// fully masked and not written.
template <int W> struct VecWord;
template <> struct VecWord<8> { typedef uint4 type; };
template <> struct VecWord<4> { typedef uint2 type; };

// One step's K and V words for this lane: U keys per slot from key n0, rows clamped to Lk - 1.
template <int U, int C, int W, int LPK, int NSLOT, typename word>
inline void vec_load(thread word (&kw)[U][C], thread word (&vw)[U][C], device const half* kbase,
                     device const half* vbase, long ksl, long vsl, int n0, int slot, int Lk) {
    _Pragma("unroll") for (int u = 0; u < U; ++u) {
        const int n = min(n0 + NSLOT * u + slot, Lk - 1);
        _Pragma("unroll") for (int c = 0; c < C; ++c) {
            kw[u][c] = *(device const word*)(kbase + long(n) * ksl + c * W * LPK);
            vw[u][c] = *(device const word*)(vbase + long(n) * vsl + c * W * LPK);
        }
    }
}

template <int D, int RV, int U, int LPK>
inline void sqa_vec_body(device const half* q, device const half* k, device const half* v,
                         device float* part_o, device float* part_ml, const SqaArgs A,
                         uint3 tgp, uint3 ntg, uint sg, uint nsg, uint lane) {
    constexpr int E = D / LPK;         // head-dim elements per lane
    constexpr int W = E < 8 ? E : 8;   // elements per load
    constexpr int C = E / W;           // loads per lane per K or V row
    constexpr int NSLOT = 32 / LPK;
    typedef typename VecWord<W>::type word;
    static_assert(D % LPK == 0 && E >= 4 && E % W == 0 && 32 % LPK == 0 && RV >= 1 && U >= 1,
                  "bad configuration");
    const int slot = int(lane) / LPK, sub = int(lane) % LPK;
    const int T = A.T, G = A.G, pos = A.pos, NS = A.NS;
    const int R = G * T;
    const int RGT = A.RG, RGY = (R + RV * RGT - 1) / (RV * RGT);
    const int split = int(tgp.x) * (int(nsg) / RGT) + int(sg) / RGT;
    const int r0 = ((int(tgp.y) % RGY) * RGT + int(sg) % RGT) * RV;
    const int kvh = int(tgp.y) / RGY, b = int(tgp.z);
    const long Hq = long(ntg.y / uint(RGY)) * G;
    const int Lk = pos + T;
    const int ks = split * A.chunk;
    const int ke = min(ks + A.chunk, Lk);
    const float sl2 = A.scale * M_LOG2E_F;

    float qv[RV][E];
    int lim[RV];
    _Pragma("unroll") for (int r = 0; r < RV; ++r) {
        const int rr = min(r0 + r, R - 1);
        const int g = rr / T, t = rr % T;
        device const half* qr = q + long(b) * A.qsb + (long(kvh) * G + g) * A.qsh + long(t) * A.qst;
        _Pragma("unroll") for (int c = 0; c < C; ++c)
            _Pragma("unroll") for (int e = 0; e < W; ++e)
                qv[r][W * c + e] = float(qr[c * W * LPK + W * sub + e]) * sl2;
        lim[r] = r0 + r < R ? min(pos + t + A.shift, Lk - 1) : -1;   // padding rows see no key
    }
    device const half* kbase = k + long(b) * A.ksb + long(kvh) * A.ksh + W * sub;
    device const half* vbase = v + long(b) * A.vsb + long(kvh) * A.vsh + W * sub;

    float m[RV], l[RV], o[RV][E];
    _Pragma("unroll") for (int r = 0; r < RV; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        _Pragma("unroll") for (int e = 0; e < E; ++e) o[r][e] = 0.0f;
    }

    // (Loading the next step's K and V before this step's arithmetic was measured: no gain at
    // T = 1, slower at T = 2 to 8.)
    for (int n0 = ks; n0 < ke; n0 += NSLOT * U) {
        word kw[U][C], vw[U][C];
        vec_load<U, C, W, LPK, NSLOT>(kw, vw, kbase, vbase, A.ksl, A.vsl, n0, slot, Lk);
        float s[U][RV];
        _Pragma("unroll") for (int u = 0; u < U; ++u) {
            float kf[E];
            _Pragma("unroll") for (int c = 0; c < C; ++c)
                _Pragma("unroll") for (int i = 0; i < W / 2; ++i) {
                    const half2 h = as_type<half2>(kw[u][c][i]);
                    kf[W * c + 2 * i] = float(h.x);
                    kf[W * c + 2 * i + 1] = float(h.y);
                }
            const int n = n0 + NSLOT * u + slot;
            _Pragma("unroll") for (int r = 0; r < RV; ++r) {
                float d = 0.0f;
                _Pragma("unroll") for (int e = 0; e < E; ++e) d = fma(kf[e], qv[r][e], d);
                _Pragma("unroll") for (ushort x = 1; x < LPK; x *= 2) d += simd_shuffle_xor(d, x);
                s[u][r] = n <= lim[r] ? d : -INFINITY;
            }
        }
        // Lazy reference max per slot and row; a step whose keys are all masked leaves it alone.
        _Pragma("unroll") for (int r = 0; r < RV; ++r) {
            float mx = s[0][r];
            _Pragma("unroll") for (int u = 1; u < U; ++u) mx = max(mx, s[u][r]);
            if (mx > m[r] + SLACK) {
                const float a = exp2(m[r] - mx);       // 0 while m is -inf
                l[r] *= a;
                _Pragma("unroll") for (int e = 0; e < E; ++e) o[r][e] *= a;
                m[r] = mx;
            }
        }
        _Pragma("unroll") for (int u = 0; u < U; ++u) {
            float vf[E];
            _Pragma("unroll") for (int c = 0; c < C; ++c)
                _Pragma("unroll") for (int i = 0; i < W / 2; ++i) {
                    const half2 h = as_type<half2>(vw[u][c][i]);
                    vf[W * c + 2 * i] = float(h.x);
                    vf[W * c + 2 * i + 1] = float(h.y);
                }
            _Pragma("unroll") for (int r = 0; r < RV; ++r) {
                const float p = s[u][r] == -INFINITY ? 0.0f : exp2(s[u][r] - m[r]);
                l[r] += p;
                _Pragma("unroll") for (int e = 0; e < E; ++e) o[r][e] = fma(p, vf[e], o[r][e]);
            }
        }
    }

    // Merge the slots (lanes LPK apart, then 2 LPK, ...); every slot ends with the merged state.
    _Pragma("unroll") for (int r = 0; r < RV; ++r) {
        _Pragma("unroll") for (ushort x = LPK; x < 32; x *= 2) {
            const float mo = simd_shuffle_xor(m[r], x), lo = simd_shuffle_xor(l[r], x);
            const float M = max(m[r], mo);
            const float w = m[r] == -INFINITY ? 0.0f : exp2(m[r] - M);
            const float wo = mo == -INFINITY ? 0.0f : exp2(mo - M);
            l[r] = l[r] * w + lo * wo;
            _Pragma("unroll") for (int e = 0; e < E; ++e) {
                const float oo = simd_shuffle_xor(o[r][e], x);
                o[r][e] = o[r][e] * w + oo * wo;
            }
            m[r] = M;
        }
    }
    if (slot == 0) {
        _Pragma("unroll") for (int r = 0; r < RV; ++r) {
            const int rr = r0 + r;
            if (rr >= R) continue;
            const long hq = long(kvh) * G + rr / T;
            const long idx = ((long(b) * Hq + hq) * T + rr % T) * NS + split;
            device float* po = part_o + idx * D + W * sub;
            _Pragma("unroll") for (int c = 0; c < C; ++c)
                _Pragma("unroll") for (int e = 0; e < W; e += 4)
                    *(device float4*)(po + c * W * LPK + e) =
                        float4(o[r][W * c + e], o[r][W * c + e + 1], o[r][W * c + e + 2], o[r][W * c + e + 3]);
            if (sub == 0) *(device float2*)(part_ml + 2 * idx) = float2(m[r], l[r]);
        }
    }
}

// Row-vector entries, the same buffers as pass 1 (buffer 19, RG, is the row groups per threadgroup).
template <int D, int RV, int U, int LPK>
kernel void sqa_vec(device const half* q [[buffer(0)]],
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
                    uint nsg [[simdgroups_per_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const SqaArgs A = {qsb, qsh, qst, ksb, ksh, ksl, vsb, vsh, vsl, int(T_), int(G_), int(pos_),
                       int(chunk_), int(NS_), int(RG_), int(shift_), scale};
    sqa_vec_body<D, RV, U, LPK>(q, k, v, part_o, part_ml, A, tgp, ntg, sg, nsg, lane);
}

template <int D, int RV, int U, int LPK>
kernel void sqa_vec_dev(device const half* q [[buffer(0)]],
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
                        uint nsg [[simdgroups_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const long p = posd[long(tgp.z) * pstride], hi = Lmax_ - T_;
    const long pc = p < 0 ? 0 : (p > hi ? hi : p);
    const SqaArgs A = {qsb, qsh, qst, ksb, ksh, ksl, vsb, vsh, vsl, int(T_), int(G_), int(pc),
                       int(chunk_), int(NS_), int(RG_), int(shift_), scale};
    sqa_vec_body<D, RV, U, LPK>(q, k, v, part_o, part_ml, A, tgp, ntg, sg, nsg, lane);
}

// Pass 2: one threadgroup of D / 4 threads per output row (b, hq, t); thread i owns columns
// 4 i .. 4 i + 3. Chunks whose row saw no visible key (m = -inf) get weight 0. This is the
// reduce of experiment E's configuration (Config.rs = 0): one thread walks every chunk in turn,
// so its cost grows with NS at the latency of one load per chunk.
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
    for (int s = 0; s < NS; s += 1) M = max(M, ml[s].x);
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

// Pass 2, parallel over chunks (Config.rs = RS > 0): one threadgroup of RS * D / 4 threads per
// output row; thread (i, j) = (tid % (D / 4), tid / (D / 4)) owns columns 4 i .. 4 i + 3 of the
// chunks s = j, j + RS, ..., two loads in flight per step. Every thread takes the reference max M
// over all chunks (the m values are a few hundred bytes, read from cache); the RS partial sums are
// then added through threadgroup memory. Same arithmetic as sqa_reduce up to summation order.
template <int D, int RS>
kernel void sqa_reduce_par(device const float* part_o [[buffer(0)]],
                           device const float* part_ml [[buffer(1)]],
                           device half* out [[buffer(2)]],
                           constant long& NS_ [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    constexpr int Q = D / 4;
    threadgroup float4 accs[RS > 1 ? RS - 1 : 1][Q];
    threadgroup float ls[RS > 1 ? RS - 1 : 1];
    const int NS = int(NS_);
    const int i = int(tid) % Q, j = int(tid) / Q;
    device const float2* ml = (device const float2*)part_ml + long(row) * NS;
    float M = -INFINITY;
    for (int s = 0; s < NS; ++s) M = max(M, ml[s].x);
    float L0 = 0.0f, L1 = 0.0f;
    float4 a0 = 0.0f, a1 = 0.0f;
    device const float4* po = (device const float4*)(part_o + long(row) * NS * D) + i;
    int s = j;
    for (; s + RS < NS; s += 2 * RS) {
        const float2 e0 = ml[s], e1 = ml[s + RS];
        const float4 p0 = po[s * Q], p1 = po[(s + RS) * Q];
        const float w0 = (e0.x == -INFINITY) ? 0.0f : exp2(e0.x - M);
        const float w1 = (e1.x == -INFINITY) ? 0.0f : exp2(e1.x - M);
        L0 += w0 * e0.y;
        L1 += w1 * e1.y;
        a0 += w0 * p0;
        a1 += w1 * p1;
    }
    if (s < NS) {
        const float2 e = ml[s];
        const float w = (e.x == -INFINITY) ? 0.0f : exp2(e.x - M);
        L0 += w * e.y;
        a0 += w * po[s * Q];
    }
    float4 acc = a0 + a1;
    float L = L0 + L1;
    if (RS > 1) {
        if (j > 0) {
            accs[j - 1][i] = acc;
            if (i == 0) ls[j - 1] = L;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j > 0) return;
        _Pragma("unroll") for (int jj = 0; jj < RS - 1; ++jj) {
            acc += accs[jj][i];
            L += ls[jj];
        }
    }
    *((device half4*)(out + long(row) * D) + i) = half4(acc / L);
}
