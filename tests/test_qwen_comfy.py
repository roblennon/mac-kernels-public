"""Encoder ownership must end before diffusion admission, including failures."""
import sys
from types import SimpleNamespace
import weakref
import pytest
from bench import qwen_comfy, qwen_memory


EDIT_IMAGES = {'image_1': 'source', 'image_2': 'ref'}
VAE = SimpleNamespace(patcher='vae-patcher')
PREPARE = {
    'text': (lambda: qwen_comfy.BenchQwenPrepare().prepare('unet', 'clip', 'prompt', 'negative'),
             ('negative', None, 1024, None), ('model', ['positive'], ['negative'])),
    'edit': (lambda: qwen_comfy.BenchQwenEditPrepare().prepare('unet', 'clip', VAE, 'source', 'prompt', 768, image_2='ref'),
             ('', VAE, 768, EDIT_IMAGES), ('model', ['positive'], {'samples': 'latent'})),
}


@pytest.mark.parametrize('kind', PREPARE)
@pytest.mark.parametrize('retain', [False, True])
def test_encoder_released_before_diffusion_load(monkeypatch, retain, kind):
    prepare, encoded, returned = PREPARE[kind]
    refs, held, calls = [], [], []
    class Asset:
        pass
    def load_clip(*args):
        clip = Asset()
        clip.cond_stage_model = Asset()
        clip.patcher = Asset()
        refs.extend(weakref.ref(v) for v in (clip, clip.cond_stage_model, clip.patcher))
        if retain:
            held.append(clip)
        return (clip,)
    def encode(clip, prompt, negative_prompt, vae=None, resolution=1024, images=None):
        calls.append(('encode', prompt, (negative_prompt, vae, resolution, images)))
        return SimpleNamespace(result=(['positive'], ['negative'], {'samples': 'latent'}))
    def load_unet(name):
        assert all(ref() is None for ref in refs)
        calls.append(('load', name))
        return 'model'
    mm = SimpleNamespace(unload_all_models=lambda: None, soft_empty_cache=lambda: None,
        unload_model_and_clones=lambda patcher, **kw: calls.append(('unload', kw) if patcher != VAE.patcher else ('unload vae', kw)),
        cleanup_models_gc=lambda: None)
    monkeypatch.setattr(qwen_memory, 'load_bounded', load_unet)
    monkeypatch.setitem(sys.modules, 'nodes', SimpleNamespace(
        CLIPLoader=lambda: SimpleNamespace(load_clip=load_clip),
        UNETLoader=lambda: SimpleNamespace(load_unet=load_unet)))
    monkeypatch.setitem(sys.modules, 'comfy', SimpleNamespace(model_management=mm))
    monkeypatch.setitem(sys.modules, 'comfy.model_management', mm)
    monkeypatch.setitem(sys.modules, 'comfy_extras', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'comfy_extras.nodes_qwen', SimpleNamespace(
        TextEncodeQwenImage21=SimpleNamespace(execute=encode)))
    if retain:
        with pytest.raises(RuntimeError, match='retained its encoder'):
            prepare()
        assert not any(call[0] == 'load' for call in calls)
    else:
        assert prepare() == returned
        # The edit path also moves the VAE off the GPU before diffusion loads; decode reloads it.
        released = [('unload vae', {'all_devices': True})] if kind == 'edit' else []
        assert calls == [('encode', 'prompt', encoded), ('unload', {'all_devices': True}), *released,
                         ('load', 'unet')]


def test_edit_refuses_a_gap_in_reference_images():
    # ComfyUI numbers connected images consecutively, so image_3 without image_2 would be <image2>.
    with pytest.raises(ValueError, match='image_2'):
        qwen_comfy.BenchQwenEditPrepare().prepare('unet', 'clip', VAE, 'source', 'prompt', 1024, image_3='ref')
