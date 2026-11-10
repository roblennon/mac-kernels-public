"""A fused rotary positional embedding for MPS (rotate-half convention, as Llama and Qwen use),
one Metal kernel per tensor in place of about five elementwise kernels. fp16 only.
Computes x*cos + rotate_half(x)*sin in float32 per element and rounds once.

apply_rotary_pos_emb(q, k, cos, sin) matches transformers' signature for Qwen3: q, k are
[B, H, T, D] views of [B, T, H, D] buffers (as the attention module produces them); cos, sin [B, T, D].
"""
import torch

SRC = r"""
#include <metal_stdlib>
using namespace metal;
kernel void rope_f16(device half* out [[buffer(0)]],
                     device const half* x [[buffer(1)]],
                     device const half* cs [[buffer(2)]],
                     device const half* sn [[buffer(3)]],
                     constant long& T [[buffer(4)]],
                     constant long& H [[buffer(5)]],
                     constant long& D [[buffer(6)]],
                     constant long& total [[buffer(7)]],
                     constant long& cos_batch [[buffer(8)]],
                     uint idx [[thread_position_in_grid]]) {
    if (long(idx) >= total) return;
    const long d = long(idx) % D;
    const long bt = long(idx) / (D * H);          // x and out are [B, T, H, D]; bt = b*T + t
    const long b = bt / T, t = bt % T;
    const long cb = cos_batch == 1 ? 0 : b;       // cos and sin are [B, T, D] or [1, T, D] broadcast over the batch
    const long half_d = D / 2;
    const float xv = float(x[idx]);
    const float other = d < half_d ? -float(x[idx + half_d]) : float(x[idx - half_d]);
    const long ci = (cb * T + t) * D + d;
    out[idx] = half(xv * float(cs[ci]) + other * float(sn[ci]));
}
"""
_lib = None

def _kernel():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(SRC)
    return _lib

def rope_one(x, cos, sin):
    # x: [B, H, T, D] view whose transpose(1, 2) is a contiguous [B, T, H, D] buffer
    xb = x.transpose(1, 2)
    if not xb.is_contiguous() or x.dtype != torch.float16 or x.shape[-1] % 2:
        raise ValueError("rope_one needs fp16, even head_dim, and a [B,T,H,D]-contiguous buffer")
    B, T, H, D = xb.shape
    if cos.dim() != 3 or cos.shape[1:] != (T, D) or cos.shape[0] not in (1, B) or sin.shape != cos.shape:
        raise ValueError(f"cos/sin must be [B or 1, T, D]; got {tuple(cos.shape)} for x {tuple(xb.shape)}")
    out = torch.empty_like(xb)
    total = out.numel()
    _kernel().rope_f16(out, xb, cos.contiguous(), sin.contiguous(), T, H, D, total, cos.shape[0],
                       threads=total, group_size=256)
    return out.transpose(1, 2)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    if q.dtype != torch.float16 or q.device.type != "mps" or unsqueeze_dim != 1:
        raise ValueError("fused rope: fp16 on MPS with unsqueeze_dim=1 only")
    return rope_one(q, cos, sin), rope_one(k, cos, sin)

def reference(x, cos, sin):
    x, c, s = x.cpu().double(), cos.cpu().double().unsqueeze(1), sin.cpu().double().unsqueeze(1)
    h = x.shape[-1] // 2
    rot = torch.cat((-x[..., h:], x[..., :h]), -1)
    return x * c + rot * s

if __name__ == "__main__":
    torch.manual_seed(0)
    worst = {}
    for (B, T, Hq, Hk, D) in [(1, 1, 16, 8, 128), (4, 1, 16, 8, 128), (16, 1, 16, 8, 128), (2, 37, 16, 8, 128), (1, 5, 3, 1, 64)]:
        qb = torch.randn(B, T, Hq, D, device="mps", dtype=torch.float16)
        kb = torch.randn(B, T, Hk, D, device="mps", dtype=torch.float16)
        q, k = qb.transpose(1, 2), kb.transpose(1, 2)
        for cb in (B, 1):
          ang = torch.rand(cb, T, D // 2) * 50
          cos = torch.cat([ang.cos()] * 2, -1).half().to("mps"); sin = torch.cat([ang.sin()] * 2, -1).half().to("mps")
          fq, fk = apply_rotary_pos_emb(q, k, cos, sin)
          h = D // 2
          def hf(x):
              c, s = cos.unsqueeze(1), sin.unsqueeze(1)
              return x * c + torch.cat((-x[..., h:], x[..., :h]), -1) * s
          for name, got, base, x in (("q", fq, hf(q), q), ("k", fk, hf(k), k)):
              ref = reference(x, cos, sin)
              e_f = float((got.cpu().double() - ref).abs().max()); e_h = float((base.cpu().double() - ref).abs().max())
              worst[(B, T, Hq, D, cb, name)] = (e_f, e_h)
              assert got.shape == base.shape and e_f <= e_h + 1e-9, (B, T, name, e_f, e_h)
    # planted bug: wrong sign on the rotated half must be caught
    bad = rope_one(q, cos, -sin)
    assert float((bad.cpu().double() - reference(q, cos, sin)).abs().max()) > 0.1
    for k_, v in worst.items(): print(k_, "fused err %.2e  hf-fp16 err %.2e" % v)
    print("fused rope: error vs float64 no worse than Hugging Face's fp16 formula in every case; planted sign bug caught")
