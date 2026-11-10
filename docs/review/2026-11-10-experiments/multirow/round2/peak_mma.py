"""Peak simdgroup 8x8 multiply-accumulate rate on this GPU (round 2 diagnosis), against the number
of independent accumulators and resident simdgroups per core; operands are constants in registers.
Run through the GPU lock: bench/gpu_runner.py --monitor --seconds 280 --rss-gib 10 -- <python> peak_mma.py out.json

The "fma float4" rows are a plain-FMA loop; its FLOP count is right but the loop did not reach
the ALU peak (about 3 to 4 TFLOPS), so it is not used as a ceiling."""
import json, statistics, sys, time, torch
SRC = """
#include <metal_stdlib>
using namespace metal;
template <typename T, typename A, int NA>
kernel void mpk(device float* out [[buffer(0)]], constant long& iters [[buffer(1)]],
                uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_matrix<T, 8, 8> a = make_filled_simdgroup_matrix<T, 8, 8>(T(1.0f + lane * 1e-3f));
    simdgroup_matrix<T, 8, 8> b = make_filled_simdgroup_matrix<T, 8, 8>(T(1.0f - lane * 1e-3f));
    simdgroup_matrix<A, 8, 8> acc[NA];
    for (int i = 0; i < NA; ++i) acc[i] = make_filled_simdgroup_matrix<A, 8, 8>(A(0.0f));
    for (long it = 0; it < iters; ++it)
        for (int i = 0; i < NA; ++i) simdgroup_multiply_accumulate(acc[i], a, b, acc[i]);
    float s = 0.0f;
    for (int i = 0; i < NA; ++i) s += float(acc[i].thread_elements()[0]);
    out[gid] = s;
}
template <int NA>
kernel void fpk(device float* out [[buffer(0)]], constant long& iters [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
    float4 a = float4(1.0f + gid * 1e-9f), b = float4(0.999f);
    float4 acc[NA];
    for (int i = 0; i < NA; ++i) acc[i] = float4(i);
    for (long it = 0; it < iters; ++it)
        for (int i = 0; i < NA; ++i) acc[i] = fma(acc[i], b, a);
    float4 s = 0.0f;
    for (int i = 0; i < NA; ++i) s += acc[i];
    out[gid] = s.x + s.y + s.z + s.w;
}
"""
rows = []
def run(fn, groups, iters, flop_per):
    out = torch.empty(groups * 128, device="mps")
    fn(out, iters, threads=groups * 128, group_size=128); torch.mps.synchronize()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); fn(out, iters, threads=groups * 128, group_size=128); torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    t = statistics.median(ts)
    return t, flop_per * groups / t / 1e12
for T, A in (("float", "float"), ("half", "float"), ("half", "half")):
    for NA in (2, 4, 8):
        lib = torch.mps.compile_shader(SRC + f'\ntemplate [[host_name("k")]] kernel void mpk<{T}, {A}, {NA}>(device float*, constant long&, uint, uint);\n')
        for spc in (8, 16, 24, 32):
            groups = 32 * spc // 4
            iters = 40_000_000 // (groups * NA) * 8
            t, tf = run(lib.k, groups, iters, 4 * iters * NA * 1024)
            rows.append({"kind": f"mma {T}x{T}+={A}", "acc": NA, "sg_per_core": spc, "ms": t * 1e3, "TFLOPS": tf})
            print(json.dumps(rows[-1]), flush=True)
for NA in (4, 8):
    lib = torch.mps.compile_shader(SRC + f'\ntemplate [[host_name("k")]] kernel void fpk<{NA}>(device float*, constant long&, uint);\n')
    for spc in (8, 16, 32):
        groups = 32 * spc // 4
        iters = 300_000_000 // (groups * NA)
        t, tf = run(lib.k, groups, iters, 128 * iters * NA * 8)
        rows.append({"kind": "fma float4", "acc": NA, "sg_per_core": spc, "ms": t * 1e3, "TFLOPS": tf})
        print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
