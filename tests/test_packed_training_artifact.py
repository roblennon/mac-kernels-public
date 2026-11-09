"""Tiny real packed weights prove the preparation/training artifact boundary on CPU."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', [[], ['--prepare-only', '--prepared', 'unused.pt']])
def test_probe_requires_exactly_one_mode_before_startup(tmp_path, mode):
    result = subprocess.run([sys.executable, str(ROOT / 'bench/probe_packed_training.py'),
                             '--bits', '4', '--out', str(tmp_path / 'run'), *mode],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert '--prepare-only' in result.stderr and '--prepared' in result.stderr
    assert not (tmp_path / 'run').exists()


def test_real_packed_artifact_and_cpu_preparation(tmp_path):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(tmp_path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PACKED_ARTIFACT_CPU_OK' in result.stdout
    assert 'PREPARATION_WITHOUT_MPS_OK' in result.stdout


def run_fixture(directory):
    import dataclasses
    from types import SimpleNamespace
    from unittest.mock import patch
    sys.path[:0] = [str(ROOT), str(ROOT / 'train')]
    from bench.probe_text_encoder_load import bootstrap_cpu
    bootstrap_cpu()
    import comfy.model_management
    import torch
    from comfy.quant_ops import AsymW4A8Int8Layout, QuantizedTensor
    from bench import probe_packed_training as probe
    import train_lora as trainer

    directory = Path(directory)
    source = torch.randn((8, 256), generator=torch.Generator().manual_seed(7), dtype=torch.bfloat16) * .02
    for bits in (4, 6):
        weight = QuantizedTensor.from_float(source, 'AsymW4A8Int8Layout', bits=bits,
                                           group_size=16 if bits == 4 else 32, convrot_groupsize=256,
                                           codebook=bits == 4)
        decoded = weight.dequantize()
        relative = float((decoded.float() - source.float()).norm() / source.float().norm())
        record = probe.record_packed(weight, relative, probe.tensor_digest(decoded))
        assert set(record['params']) == {f.name for f in dataclasses.fields(AsymW4A8Int8Layout.Params)}
        payload = dict(version=1, bits=bits, block=1, base={'sha256': 'fixture-base', 'bytes': 8},
                       codec=probe.codec_fingerprints(), layers={'layer': record})
        path = directory / f'w{bits}.pt'
        manifest = probe.save_prepared(path, payload)
        assert manifest['artifact'] == probe.fingerprint(path)
        assert manifest['serialized_dequantization_verified'] is True
        # A training-process load must never re-quantize or fully dequantize a weight.
        with patch.object(QuantizedTensor, 'from_float', side_effect=AssertionError('quantized while loading')), \
                patch.object(QuantizedTensor, 'dequantize', side_effect=AssertionError('decoded while loading')):
            loaded, provenance = probe.load_prepared(path, bits, ['layer'])
            rebuilt = probe.restore_packed(loaded['layers']['layer'], bits)
        assert provenance == manifest
        torch.testing.assert_close(rebuilt.dequantize(), decoded, rtol=0, atol=0)
        assert loaded['layers']['layer']['conversion_relative_l2'] == relative
        assert probe.tensor_digest(rebuilt.dequantize()) == record['dequantized_sha256']
        with pytest.raises(ValueError, match='does not match'):
            probe.load_prepared(path, 6 if bits == 4 else 4, ['layer'])
        with pytest.raises(ValueError, match='does not match'):
            probe.load_prepared(path, bits, ['another-layer'])
        bad = {**record, 'params': dict(record['params'])}
        del bad['params']['transposed']
        with pytest.raises(ValueError, match='layout fields'):
            probe.restore_packed(bad, bits)
        bad = {**record, 'params': {**record['params'], 'convrot_groupsize': 128}}
        with pytest.raises(ValueError, match='layout parameters'):
            probe.restore_packed(bad, bits)
        bad = {**record, 'qdata': record['qdata'].clone()}
        bad['qdata'][0, 0] ^= 1
        with pytest.raises(ValueError, match='tensor bytes changed'):
            probe.restore_packed(bad, bits)
        # Even valid packed storage cannot publish readiness with a false decoded digest.
        invalid_path = directory / f'invalid-w{bits}.pt'
        invalid = {**payload, 'layers': {'layer': {**record, 'dequantized_sha256': 'incorrect'}}}
        with pytest.raises(ValueError, match='dequantized weight changed'):
            probe.save_prepared(invalid_path, invalid)
        assert not Path(str(invalid_path) + '.json').exists()
        with patch.object(probe, 'codec_fingerprints', return_value={}):
            with pytest.raises(ValueError, match='does not match'):
                probe.load_prepared(path, bits, ['layer'])
        # Reject a damaged artifact using its file digest before attempting deserialization.
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with patch.object(torch, 'load', side_effect=AssertionError('deserialized corrupt bytes')):
            with pytest.raises(ValueError, match='SHA256'):
                probe.load_prepared(path, bits, ['layer'])
    print('PACKED_ARTIFACT_CPU_OK')

    # Exercise the actual --prepare-only branch with tiny model-loading boundaries.
    # Quantization, serialization, rehydration, verification and provenance remain real.
    modules = {name: torch.nn.Linear(256, 8, bias=False, dtype=torch.bfloat16)
               for name in trainer.ARCHS['qwen21']['targets']}
    for module in modules.values():
        module.weight = torch.nn.Parameter(source.clone(), requires_grad=False)
    block = SimpleNamespace(get_submodule=lambda name: modules[name])
    model = SimpleNamespace(mmap_offload=True,
                            model=SimpleNamespace(diffusion_model=SimpleNamespace(transformer_blocks=[None, block])))
    class Loader:
        def load_unet(self, *args, **kwargs):
            return (model,)
    base = directory / 'base.safetensors'
    base.write_bytes(b'base fingerprint fixture')
    out = directory / 'prepared'
    with patch.dict(trainer.cb.N, {'UNETLoader': Loader}), \
            patch.object(trainer.cb.folder_paths, 'get_full_path_or_raise', return_value=str(base)), \
            patch('bench.probe_text_encoder_load.require_supervisor'), \
            patch.object(torch.mps, 'set_per_process_memory_fraction', side_effect=AssertionError('MPS allocator touched')), \
            patch.object(torch.mps, 'driver_allocated_memory', side_effect=AssertionError('MPS queried')), \
            patch.object(sys, 'argv', ['probe', '--prepare-only', '--bits', '4', '--out', str(out)]):
        probe.main()
    payload, provenance = probe.load_prepared(out / 'packed-block.pt', 4, trainer.ARCHS['qwen21']['targets'])
    assert provenance['base'] == probe.fingerprint(base)
    assert len(payload['layers']) == 6
    assert comfy.model_management.get_torch_device().type == 'cpu'
    print('PREPARATION_WITHOUT_MPS_OK')


if __name__ == '__main__':
    run_fixture(sys.argv[1])
