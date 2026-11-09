"""Trainer surgery uses real packed layouts; only the GPU kernel boundary is replaced."""
from types import SimpleNamespace
import sys

import torch
import torch.nn.functional as F

from test_training_recovery import trainer  # noqa: F401


def test_mixed_base_routes_packed_layers_and_trains_lora(trainer, monkeypatch):
    from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor, TensorWiseINT8Layout
    from comfy_kitchen.backends.eager import w4a8_int8 as eager
    from comfy_kitchen.backends.mps import autograd, packed_autograd
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    monkeypatch.setitem(sys.modules, 'comfy.quant_ops', SimpleNamespace(
        QuantizedTensor=QuantizedTensor, TensorWiseINT8Layout=TensorWiseINT8Layout,
        AsymW4A8Int8Layout=AsymW4A8Int8Layout,
    ))
    calls = []

    def packed_cpu(x, q, sr, sc, cb=None, corr=None, bias=None, group_size=16, convrot_groupsize=256):
        calls.append(('packed', q.shape[1], group_size, cb is not None, corr is not None))
        w = eager.dequantize_w4a8_int8_weight(q, sr, sc, cb, corr, group_size,
                                             convrot_groupsize, torch.float32)
        return F.linear(x, w, bias)

    def int8_cpu(x, q, s, bias=None, convrot=False, convrot_groupsize=256):
        calls.append(('int8', convrot, convrot_groupsize))
        w = q.float() * s[:, None]
        if convrot:
            w = (w.reshape(-1, 1, 256) @ _build_hadamard(256)).reshape_as(w)
        return F.linear(x, w, bias)

    monkeypatch.setattr(packed_autograd, 'w4a8_linear_trainable', packed_cpu)
    monkeypatch.setattr(autograd, 'int8_linear_trainable', int8_cpu)
    gen = torch.Generator().manual_seed(72)

    class FrozenLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.in_features = self.out_features = 256
            self.register_buffer('weight', weight)
            self.register_buffer('bias', torch.zeros(256))

        def forward(self, x):
            w = self.weight.dequantize() if isinstance(self.weight, QuantizedTensor) else self.weight
            return F.linear(x, w, self.bias)

    def packed_weight(bits, asymmetric=False):
        gs = 32 if bits == 6 else 16
        q = torch.randint(-128, 128, (256, 256 * bits // 8), dtype=torch.int8, generator=gen)
        p = AsymW4A8Int8Layout.Params(
            scale=torch.ones(256, 256 // gs), s_channel=torch.full((256,), .01),
            codebook=torch.linspace(-7, 7, 16) if asymmetric else None,
            correction=torch.randn(256 // gs, 256, generator=gen) * .001 if asymmetric else None,
            group_size=gs, convrot_groupsize=256, orig_dtype=torch.float32, orig_shape=(256, 256),
        )
        return QuantizedTensor(q, 'AsymW4A8Int8Layout', p)

    q = torch.randint(-127, 128, (256, 256), dtype=torch.int8, generator=gen)
    p = TensorWiseINT8Layout.Params(scale=torch.full((256,), .001), orig_dtype=torch.float32,
                                    orig_shape=(256, 256), convrot=True, convrot_groupsize=256)
    weights = [packed_weight(4, True), packed_weight(6),
               QuantizedTensor(q, 'TensorWiseINT8Layout', p), torch.randn(256, 256, generator=gen) * .01]
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList()
    for weight in weights:
        block = torch.nn.Module()
        block.linear = FrozenLinear(weight)
        model.blocks.append(block)
    model.extra = FrozenLinear(packed_weight(4))  # Outside LoRA targets; still needs input backward.
    spec = {'blocks': 'blocks', 'targets': ('linear',)}
    loras, count = trainer.patch_model(model, spec, rank=4, seed=2, device='cpu')
    assert count == 4 and set(loras) == {f'blocks.{i}.linear' for i in range(4)}
    x = torch.randn(2, 256, generator=gen, requires_grad=True)
    for block in model.blocks:
        x = block.linear(x).tanh()
    loss = model.extra(x).square().mean()
    loss.backward()
    assert calls == [('packed', 128, 16, True, True), ('packed', 192, 32, False, False),
                     ('int8', True, 256), ('packed', 128, 16, False, False)]
    for lora in loras.values():
        assert lora.up.grad is not None and torch.isfinite(lora.up.grad).all()
        assert lora.up.grad.abs().max() > 0
    opt = torch.optim.AdamW([p for lora in loras.values() for p in lora.parameters()], lr=.001)
    opt.step()
    assert all(lora.up.abs().max() > 0 for lora in loras.values())
