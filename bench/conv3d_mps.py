"""conv3d alternatives on MPS at Wan VAE decoder shapes (fp16).

Variants: torch conv3d (baseline); channels_last_3d layout; per-time-tap conv2d (frames batched, kt calls summed);
time taps stacked on channels into one conv2d (frames batched);
explicit im2col (unfold) + fp16 GEMM chunked over output rows. Input already causally padded in time, spatial padding 1.
Env: ROWS (comma list of im2col row-chunk sizes), SHAPES as JSON list of [Cin, Cout, T_in, H, W] (kernel 3x3x3), default = Wan 2.2 decoder shapes at 1280x704.
"""
import json, os, time
import torch, torch.nn.functional as F

SHAPES = json.loads(os.environ.get("SHAPES", "[[1024,1024,3,44,80],[1024,1024,4,88,160],[512,512,6,176,320],[256,256,6,352,640]]"))
dev, dt = "mps", torch.float16

def timed(fn, reps=3):
    fn(); torch.mps.synchronize(); t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.mps.synchronize(); return out, (time.perf_counter() - t) / reps

def conv_taps(x, w, b):
    # x (1, Cin, T_in, H, W); w (Cout, Cin, kt, 3, 3). out[t] = sum_dt conv2d(x[t + dt], w[:, :, dt])
    kt = w.shape[2]; t_out = x.shape[2] - kt + 1
    xf = x[0].transpose(0, 1)  # (T_in, Cin, H, W)
    out = None
    for d in range(kt):
        y = F.conv2d(xf[d:d + t_out], w[:, :, d], b if d == 0 else None, padding=1)
        out = y if out is None else out.add_(y)
    return out.transpose(0, 1).unsqueeze(0)

def conv_stacked(x, w, b, frames=None):
    # stack the kt time taps on the channel axis: one conv2d with kt*Cin input channels, output frames batched
    kt = w.shape[2]; t_out = x.shape[2] - kt + 1
    w2 = w.permute(0, 2, 1, 3, 4).reshape(w.shape[0], kt * w.shape[1], w.shape[3], w.shape[4])
    xf = x[0].transpose(0, 1)  # (T_in, Cin, H, W)
    frames = frames or t_out
    outs = []
    for t0 in range(0, t_out, frames):
        n = min(frames, t_out - t0)
        xs = torch.cat([xf[t0 + d:t0 + d + n] for d in range(kt)], dim=1)  # (n, kt*Cin, H, W)
        outs.append(F.conv2d(xs, w2, b, padding=(w.shape[3] // 2, w.shape[4] // 2)))
    return torch.cat(outs).transpose(0, 1).unsqueeze(0)

def conv_im2col(x, w, b, rows=16):
    # unfold each (t, row-chunk) window to (Cin*27, H_chunk*W) and multiply by the flattened weight
    cout, cin, kt = w.shape[0], w.shape[1], w.shape[2]
    t_out = x.shape[2] - kt + 1; H, W = x.shape[3], x.shape[4]
    wm = w.reshape(cout, cin * kt * 9)
    xp = F.pad(x, (1, 1, 1, 1))
    out = torch.empty(1, cout, t_out, H, W, dtype=x.dtype, device=x.device)
    for t in range(t_out):
        win = xp[0, :, t:t + kt]  # (Cin, kt, H+2, W+2)
        for r in range(0, H, rows):
            rr = min(rows, H - r)
            cols = F.unfold(win[:, :, r:r + rr + 2].reshape(cin * kt, 1, rr + 2, W + 2).transpose(0, 1), 3)  # (1, Cin*kt*9, rr*W)
            out[0, :, t, r:r + rr] = (wm @ cols[0] + b[:, None]).reshape(cout, rr, W)
    return out

with torch.inference_mode():
    for cin, cout, t_in, h, w_ in SHAPES:
        g = torch.Generator(device=dev).manual_seed(0)
        x = torch.randn(1, cin, t_in, h, w_, device=dev, dtype=dt, generator=g)
        w = torch.randn(cout, cin, 3, 3, 3, device=dev, dtype=dt, generator=g) * (1 / (cin * 27) ** 0.5)
        b = torch.randn(cout, device=dev, dtype=dt, generator=g)
        ref, t_ref = timed(lambda: F.conv3d(x, w, b, padding=(0, 1, 1)))
        res = dict(shape=[cin, cout, t_in, h, w_], conv3d_s=round(t_ref, 4))
        xc, wc = x.to(memory_format=torch.channels_last_3d), w.to(memory_format=torch.channels_last_3d)
        o, t = timed(lambda: F.conv3d(xc, wc, b, padding=(0, 1, 1))); res.update(channels_last_s=round(t, 4), channels_last_err=float((o - ref).abs().max()))
        o, t = timed(lambda: conv_taps(x, w, b)); res.update(conv2d_taps_s=round(t, 4), conv2d_taps_err=float((o - ref).abs().max()))
        o, t = timed(lambda: conv_stacked(x, w, b)); res.update(stacked_conv2d_s=round(t, 4), stacked_err=float((o - ref).abs().max()))
        o, t = timed(lambda: conv_stacked(x, w, b).contiguous()); res.update(stacked_conv2d_contig_s=round(t, 4))
        o, t = timed(lambda: conv_stacked(x, w, b, frames=1)); res.update(stacked_conv2d_1frame_s=round(t, 4))
        if os.environ.get("IM2COL", "1") == "1":
            for rows in (int(r) for r in os.environ.get("ROWS", "16").split(",")):
                o, t = timed(lambda: conv_im2col(x, w, b, rows), reps=1); res.update({f"im2col_r{rows}_s": round(t, 4), "im2col_err": float((o - ref).abs().max())})
        print("RESULT " + json.dumps(res), flush=True)
        del x, w, b, ref, xc, wc, o; torch.mps.empty_cache()
