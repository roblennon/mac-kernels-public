"""CPU persistence proof; substitute only the expensive VAE boundary."""
from pathlib import Path
import subprocess
import sys


def test_saved_wan_bytes_picker_cache_and_selected_decoder(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import sys, os
from pathlib import Path
root, tmp = map(Path, sys.argv[1:])
sys.path[:0] = [str(root), str(root/'vendor/ComfyUI')]
sys.argv = ['cpu-proof', '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch, folder_paths, nodes
from safetensors.torch import save_file
from bench.wan_latents import load_saved_latent
from bench.wan_latent_nodes import BenchWanSavedLatent, BenchWanDecode, saved_files
folder_paths.set_output_directory(str(tmp/'comfy'))
os.environ['MK_ROOT'] = str(tmp/'headless')
source = {'samples': torch.arange(192, dtype=torch.float32).reshape(1, 48, 2, 1, 2)}
saver = nodes.SaveLatent()
a = saver.save(source, 'latents/wan/proof')
b = saver.save(source, 'latents/wan/proof')
assert a['ui']['latents'] != b['ui']['latents']
choices = sorted(saved_files())
assert len(choices) == 2
loader = BenchWanSavedLatent()
loaded = loader.load(choices[0])[0]
assert torch.equal(loaded['samples'], source['samples'])
first_hash = loader.IS_CHANGED(choices[0])
file = saved_files()[choices[0]]
saved = file.read_bytes()
headless = tmp/'headless/out/latents'
headless.mkdir(parents=True)
torch.save(source['samples'], headless/'wan5b-proof.pt')
assert torch.equal(loader.load('Headless/wan5b-proof.pt')[0]['samples'], source['samples'])
_, receipt, geometry = load_saved_latent(file)
assert geometry == (5, 16, 32, 3) and receipt['finite']
changed = source['samples'] + 1
save_file({'latent_tensor': changed, 'latent_format_version_0': torch.tensor([])}, file)
assert loader.IS_CHANGED(choices[0]) != first_hash
assert torch.equal(loader.load(choices[0])[0]['samples'], changed)
file.write_bytes(saved)
assert loader.IS_CHANGED(choices[0]) == first_hash
for bad in ('../wan5b-proof.pt', '/tmp/latent.pt'):
    try: loader.load(bad)
    except ValueError: pass
    else: raise AssertionError('unlisted files must fail')
save_file({'latent_tensor': source['samples']}, tmp/'unversioned.latent')
try: load_saved_latent(tmp/'unversioned.latent')
except ValueError as error: assert 'unscaled' in str(error)
else: raise AssertionError('unversioned latent must fail')
torch.save(torch.full((1, 48, 1, 1, 1), float('nan')), headless/'wan5b-bad.pt')
try: loader.load('Headless/wan5b-bad.pt')
except ValueError as error: assert 'finite' in str(error)
else: raise AssertionError('nonfinite latent must fail')
# VAE loading/decoding is the GPU boundary; the shared saver/reader above is real.
from types import SimpleNamespace
from unittest.mock import patch
calls = []
def load_vae(self, name):
    calls.append(name)
    return (SimpleNamespace(latent_channels=16 if name in ('lighttaew2_1.safetensors', 'wan_2.1_vae.safetensors') else 48),)
def decode(self, vae, sample):
    t = sample['samples']
    scale = 16 if t.shape[1] == 48 else 8
    return (torch.zeros(t.shape[0]*(4*t.shape[2]-3), t.shape[3]*scale, t.shape[4]*scale, 3),)
with patch.object(nodes.VAELoader, 'load_vae', load_vae), patch.object(nodes.VAEDecode, 'decode', decode):
    for mode, name in [('Draft (tiny)', 'lighttaew2_2.safetensors'), ('Final (full)', 'wan2.2_vae.safetensors')]:
        before = len(calls)
        images = BenchWanDecode().decode(loaded, mode)[0]
        assert calls[before:] == [name] and tuple(images.shape) == geometry
    assert torch.equal(loaded['samples'], source['samples'])
    before = len(calls)
    BenchWanDecode().decode(loaded, 'Final (full)', vae=SimpleNamespace(latent_channels=48))
    assert len(calls) == before  # Reuse the full conditioning VAE, no second load.
    try: BenchWanDecode().decode(loaded, 'Final (full)', vae=SimpleNamespace(latent_channels=16))
    except ValueError as error: assert 'channels' in str(error)
    else: raise AssertionError('linked decoder mismatch must fail')
    assert len(calls) == before
    BenchWanDecode().decode(loaded, 'Draft (tiny)', vae=SimpleNamespace(latent_channels=48))
    assert calls[before:] == ['lighttaew2_2.safetensors']
    sixteen = {'samples': torch.zeros(1, 16, 2, 1, 2)}
    for mode, name in [('Draft (tiny)', 'lighttaew2_1.safetensors'), ('Final (full)', 'wan_2.1_vae.safetensors')]:
        before = len(calls)
        images = BenchWanDecode().decode(sixteen, mode)[0]
        assert calls[before:] == [name] and tuple(images.shape) == (5, 8, 16, 3)
with patch.object(nodes.VAELoader, 'load_vae', lambda *a: (SimpleNamespace(latent_channels=16),)):
    try: BenchWanDecode().decode(loaded, 'Final (full)')
    except ValueError as error: assert 'channels' in str(error)
    else: raise AssertionError('decoder channel mismatch must fail')
with patch.object(nodes.VAELoader, 'load_vae', load_vae), patch.object(nodes.VAEDecode, 'decode', lambda *a: (torch.zeros(1, 8, 8, 3),)):
    try: BenchWanDecode().decode(loaded, 'Final (full)')
    except RuntimeError as error: assert 'dimensions' in str(error)
    else: raise AssertionError('decoder geometry mismatch must fail')
assert file.read_bytes() == saved
print('SAVED_WAN_CPU_PROOF_OK')
'''
    result = subprocess.run([sys.executable, '-c', script, str(root), str(tmp_path)], cwd=root, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'SAVED_WAN_CPU_PROOF_OK' in result.stdout
