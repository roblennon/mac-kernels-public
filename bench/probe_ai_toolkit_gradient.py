"""Short diagnostic separating fp16 subnormal loss from packed weight/reference drift."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train' / 'ai-toolkit'))


def main():
    from bench.probe_text_encoder_load import require_supervisor
    require_supervisor()
    from bench.probe_ai_toolkit import candidate, relative
    from comfy_kitchen.backends.mps.autograd import int8_linear_grad_input
    from toolkit.util.convrot_quant import ConvRotInt8Quantizer
    from toolkit.util.ostris_quant import convert_linear_to_ostris

    torch.mps.set_per_process_memory_fraction(.25)
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wide', action='store_true')
    args = parser.parse_args()
    m, k, n = (256, 4096, 4096) if args.wide else (7, 256, 128)
    gen = torch.Generator().manual_seed(51)
    layer = torch.nn.Linear(k, n).half()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(n, k, generator=gen) / k ** .5)
        layer.bias.copy_(torch.randn(n, generator=gen) * .01)
    quant = ConvRotInt8Quantizer(rot_size=256)
    convert_linear_to_ostris(layer, quant)
    dense = quant.dequantize(layer).double()
    qc, sc = quant._qdata(layer).clone(), quant._scales(layer).clone()
    xc = torch.randn(m, k, generator=gen).half()
    gc = (torch.randn(m, n, generator=gen) / (m*n)).half()
    ref = gc.double() @ dense
    layer.to('mps')
    x = xc.to('mps').requires_grad_()
    layer(x).sum().backward()
    stable_original = torch.equal(qc, quant._qdata(layer).cpu()) and torch.equal(sc, quant._scales(layer).cpu())
    print(json.dumps({'original_preserves_packed_buffers': stable_original,
                      'subnormal_dy': int(((gc != 0) & (gc.abs() < 2**-14)).sum()),
                      'subnormal_dx': int(((ref != 0) & (ref.abs() < 2**-14)).sum()),
                      'fp16_rounding_floor_relative_l2': relative(ref.half(), ref)}), flush=True)
    q, s = quant._qdata(layer), quant._scales(layer)
    for name, input_dtype, out_dtype, scale in (
        ('unscaled_fp16', torch.float16, torch.float16, 1),
        ('scaled_fp16', torch.float16, torch.float16, 1024),
        ('fp16_in_fp32_out', torch.float16, torch.float32, 1),
        ('fp32_in_fp32_out', torch.float32, torch.float32, 1),
    ):
        g_cpu = (gc.float() * scale).to(input_dtype)
        g = g_cpu.to('mps')
        got = int8_linear_grad_input(g, q, s, True, 256, out_dtype=out_dtype).cpu()
        want = g_cpu.double() @ dense
        print(json.dumps({'case': name, 'relative_l2': relative(got, want),
                          'relative_to_rounded_reference': relative(got, want.to(out_dtype)),
                          'zero_count': int((got == 0).sum()),
                          'gradient_transfer_exact': torch.equal(g.cpu(), g_cpu)}), flush=True)
    with candidate(quant):
        for scale in (1, 1024):
            x.grad = None
            g_cpu = (gc.float() * scale).half()
            layer(x).backward(g_cpu.to('mps'))
            got = x.grad.cpu()
            print(json.dumps({'case': f'autograd_scale_{scale}',
                              'relative_l2': relative(got, g_cpu.double() @ dense),
                              'zero_count': int((got == 0).sum())}), flush=True)
    stable_candidate = torch.equal(qc, quant._qdata(layer).cpu()) and torch.equal(sc, quant._scales(layer).cpu())
    print(json.dumps({'candidate_preserves_packed_buffers': stable_candidate}), flush=True)


if __name__ == '__main__':
    main()
