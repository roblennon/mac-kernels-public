"""Multi-row linear kernel (bench/multirow_linear.py): every variant family against a float64
reference, with F.linear's own error as the yardstick, plus a planted bug the check must catch.
Needs an Apple GPU; run through bench/gpu_runner.py like any other GPU process."""
import math

import pytest
import torch
import torch.nn.functional as F

from bench import multirow_linear as mrl

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

FAMILIES = [mrl.Config("dev", 2, 1, 4), mrl.Config("tg", 1, 2, 8), mrl.Config("mma", 1, 1, 4, 4),
            mrl.Config("mmr", 2, 1, 4, 3, 1), mrl.Config("mmr", 2, 1, 4, 1, 0, 1)]
UNIT = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}


def normalised_error(y, x, w, b):
    xd, wd = x.double(), w.double()
    ref, scale = xd @ wd.t(), xd.abs() @ wd.abs().t()
    if b is not None:
        ref, scale = ref + b.double(), scale + b.double().abs()
    return ((y.cpu().double() - ref).abs() / scale).max().item()


def within_linear_error(x, w, b, cfg, lib=None):
    K = x.shape[1]
    gx, gw = x.to("mps"), w.to("mps")
    gb = None if b is None else b.to("mps")
    ours = normalised_error(mrl.multirow_linear(gx, gw, gb, config=cfg, lib=lib), x, w, b)
    lin = normalised_error(F.linear(gx, gw, gb), x, w, b)
    return ours <= lin + UNIT[x.dtype] + K * 2.0 ** -24


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cfg", FAMILIES, ids=lambda c: c.name())
@pytest.mark.parametrize("M", [1, 2, 7, 16, 17, 32])
def test_matches_linear_error_with_bias_odd_n_and_partial_tile(dtype, cfg, M):
    # Arrange: K = 2056 is not a multiple of any K tile; N = 1027 is odd.
    g = torch.Generator().manual_seed(M)
    x = torch.randn(M, 2056, generator=g).to(dtype)
    w = (torch.randn(1027, 2056, generator=g) / math.sqrt(2056)).to(dtype)
    b = torch.randn(1027, generator=g).to(dtype)
    if not cfg.fits(M):
        pytest.skip("configuration exceeds threadgroup memory at this M")
    # Act / Assert
    assert within_linear_error(x, w, b, cfg)
    assert within_linear_error(x, w, None, cfg)


def test_planted_bug_is_caught():
    # Arrange: skip the first 256-element K tile in the device-x variant.
    src = mrl.SOURCE_PATH.read_text()
    old, new = "for (uint k = lane; k < K8;", "for (uint k = lane + 32; k < K8;"
    assert src.count(old) == 1
    lib = mrl._Library(src.replace(old, new))
    x = torch.randn(8, 2056).half()
    w = (torch.randn(1027, 2056) / 45).half()
    # Act / Assert
    assert not within_linear_error(x, w, None, mrl.Config("dev", 2, 1, 4), lib=lib)


def test_unsupported_inputs_are_refused():
    w = torch.randn(64, 60, device="mps", dtype=torch.float16)
    assert not mrl.supports(torch.randn(4, 60, device="mps", dtype=torch.float16), w)  # K % 8
    w = torch.randn(64, 64, device="mps", dtype=torch.float16)
    assert not mrl.supports(torch.randn(64, 4, device="mps", dtype=torch.float16).t(), w)  # strided
    assert not mrl.supports(torch.randn(33, 64, device="mps", dtype=torch.float16), w)  # M > 32
    with pytest.raises(ValueError):
        mrl.multirow_linear(torch.randn(4, 64, device="mps", dtype=torch.float32), w)
