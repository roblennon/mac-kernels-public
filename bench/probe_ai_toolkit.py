"""Bounded, opt-in trial of weight-only INT8 training at ai-toolkit's real boundary.

This changes activation numerics: the candidate removes ai-toolkit's activation
fake quantization. It is an experiment, not a transparent replacement or an
installed ai-toolkit patch. Both grad and no-grad calls use the same candidate.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

REPO = Path(__file__).resolve().parents[1]


@contextmanager
def candidate(quantizer):
    from toolkit.util.convrot_quant import ConvRotInt8Quantizer
    from comfy_kitchen.backends.mps.autograd import int8_linear_trainable

    if type(quantizer) is not ConvRotInt8Quantizer:
        raise TypeError('This trial supports the ordinary INT8 quantizer only')
    original = quantizer.forward

    def forward(module, x):
        if x.device.type != 'mps':
            return original(module, x)
        return int8_linear_trainable(x, quantizer._qdata(module), quantizer._scales(module),
                                     module.bias, True, quantizer._rot(module))

    quantizer.forward = forward
    try:
        yield
    finally:
        del quantizer.forward


def relative(actual, expected):
    return ((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30)).item()


def trial(m, k, n, dtype):
    from toolkit.util.convrot_quant import ConvRotInt8Quantizer
    from toolkit.util.ostris_quant import convert_linear_to_ostris

    generator = torch.Generator().manual_seed(51)
    linear = torch.nn.Linear(k, n, bias=True).to(dtype=dtype)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(n, k, generator=generator) / k ** .5)
        linear.bias.copy_(torch.randn(n, generator=generator) * .01)
    quantizer = ConvRotInt8Quantizer(rot_size=256)
    assert convert_linear_to_ostris(linear, quantizer)
    # This is the actual source quantizer's frozen, rotated weight in the original basis.
    dense = quantizer.dequantize(linear).double()
    bias = linear.bias.detach().double()
    x_cpu = torch.randn(m, k, generator=generator).to(dtype)
    dy_cpu = (torch.randn(m, n, generator=generator) / (m * n)).to(dtype)
    # A mean-loss gradient at the wide shape is entirely fp16-subnormal. Training
    # uses loss scaling; preserve that contract rather than demand impossible
    # unscaled fp16 precision from a correctly rounded input gradient.
    gradient_scale = 1024 if dtype == torch.float16 else 1
    reference = F.linear(x_cpu.double(), dense, bias)
    dx_reference = dy_cpu.double() @ dense
    linear.to('mps')
    x = x_cpu.to('mps').requires_grad_()
    dy = (dy_cpu.float() * gradient_scale).to(dtype).to('mps')
    # ai-toolkit's original MPS training path includes activation fake quantization.
    original = linear(x)
    original.sum().backward()
    original_cpu = original.detach().cpu()
    x.grad = None
    del original
    saved = []

    def pack(tensor):
        saved.append({'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                      'pointer': tensor.data_ptr()})
        return tensor

    with candidate(quantizer):
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = linear(x)
        output.backward(dy)
        output_cpu = output.detach().cpu()
        dx_cpu = x.grad.detach().cpu().double() / gradient_scale
        # Hooks establish saved-tensor ownership. The separately reviewed kernel
        # supplies the remaining context-attribute/lifetime contract.
        assert len(saved) == 2
        assert {item['pointer'] for item in saved} == {
            linear.cr8_qdata.data_ptr(), quantizer._scales(linear).data_ptr()}
        with torch.no_grad():
            inference = linear(x.detach())
        assert torch.equal(output.detach(), inference)
        # Checkpoint recomputation traverses the same real OstrisLinear and candidate.
        xc = x.detach().clone().requires_grad_()
        checked = checkpoint(linear, xc, use_reentrant=False)
        checked.backward(dy)
        torch.testing.assert_close(checked, output, rtol=0, atol=0)
        torch.testing.assert_close(xc.grad, x.grad, rtol=0, atol=0)
        # An adapter feeding the frozen linear must backpropagate through its input
        # gradient, as an earlier transformer block's adapter does during training.
        down = torch.nn.Parameter(torch.randn(4, k, generator=generator).to('mps') * .01)
        up = torch.nn.Parameter(torch.zeros(k, 4, device='mps'))
        opt = torch.optim.AdamW([down, up], lr=1e-3)
        scaler = torch.amp.GradScaler('mps', init_scale=1024) if dtype == torch.float16 else None
        for step in range(2):
            adapted = x.detach() + F.linear(F.linear(x.detach(), down.to(dtype)), up.to(dtype))
            loss = linear(adapted).float().square().mean()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in (down, up))
            if step == 1:
                assert all(torch.count_nonzero(p.grad).item() > 0 for p in (down, up))
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
        assert torch.count_nonzero(up).item() > 0

    forward_error, backward_error = relative(output_cpu, reference), relative(dx_cpu, dx_reference)
    tolerance = .012 if dtype == torch.bfloat16 else .002
    assert forward_error < tolerance, (forward_error, tolerance)
    assert backward_error < tolerance, (backward_error, tolerance)
    assert torch.isfinite(output_cpu).all() and torch.isfinite(dx_cpu).all()
    return {'shape': [m, k, n], 'dtype': str(dtype), 'forward_relative_l2': forward_error,
            'input_gradient_relative_l2': backward_error,
            'input_gradient_loss_scale': gradient_scale,
            'activation_quantization_relative_l2_difference': relative(original_cpu, output_cpu),
            'checkpoint_output_and_gradient_exact': True, 'grad_and_no_grad_exact': True,
            'saved_tensors': [{k: v for k, v in item.items() if k != 'pointer'} for item in saved],
            'upstream_adapter_two_steps_finite': True, 'driver_gib': torch.mps.driver_allocated_memory() / 2**30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolkit', type=Path, default=REPO / 'train' / 'ai-toolkit')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    toolkit = args.toolkit.resolve(strict=True)
    revision = subprocess.run(['git', '-C', str(toolkit), 'rev-parse', 'HEAD'], check=True,
                              capture_output=True, text=True).stdout.strip()
    print(json.dumps({'event': 'plan', 'source_revision': revision, 'run': args.run,
                      'semantics': 'weight-only candidate; activation fake quantization removed'}), flush=True)
    if not args.run:
        return
    sys.path.insert(0, str(REPO))
    from bench.probe_text_encoder_load import require_supervisor
    require_supervisor()
    sys.path.insert(0, str(toolkit))
    torch.mps.set_per_process_memory_fraction(.25)
    torch.set_num_threads(4)
    start = time.perf_counter()
    for shape, dtype in (((7, 256, 128), torch.float16), ((7, 256, 128), torch.bfloat16),
                         ((256, 4096, 4096), torch.float16)):
        result = trial(*shape, dtype)
        print(json.dumps({'event': 'trial', **result}), flush=True)
        torch.mps.synchronize()
        torch.mps.empty_cache()
    print(json.dumps({'event': 'complete', 'seconds': time.perf_counter() - start}), flush=True)


if __name__ == '__main__':
    main()
