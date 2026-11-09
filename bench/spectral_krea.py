"""Opt-in Krea Euler experiment using the pinned MIT LC123 spectral helper.

The helper is installed separately; importing this module performs no downloads
or custom-node registration. This experiment has no Apple quality/speed claim.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

REVISION = 'e73bdd3aca64c207c821d188ff17356f0963f205'
LICENSE_SHA256 = 'ef7be470ba8c778bba6ce112901a17fdfbf8991328cf94687d79c302a4e0fc68'
SOURCE_SHA256 = '28da73c45b5e6bfee531ffe02bfd8acfb81e1f9f4850630c189fe8de4f33d74a'
SOURCE = Path(__file__).resolve().parents[1] / 'vendor/LC123-spectral/lc_speed_boost.py'
_HELPER = None


def prepare():
    """Acquire the exact small helper, preserving any existing changed source."""
    if SOURCE.exists():
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise RuntimeError('Existing spectral helper differs from the pinned source')
        verify_install()
        return
    endpoint = f'repos/lonecatone23/ComfyUI_LC123_nodes/contents/lc_speed_boost.py?ref={REVISION}'
    source = subprocess.check_output(['gh', 'api', endpoint, '-H', 'Accept: application/vnd.github.raw+json'])
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError('Downloaded spectral helper differs from the pinned source')
    license_text = subprocess.check_output(['gh', 'api', f'repos/lonecatone23/ComfyUI_LC123_nodes/contents/LICENSE?ref={REVISION}',
                                            '-H', 'Accept: application/vnd.github.raw+json'])
    if hashlib.sha256(license_text).hexdigest() != LICENSE_SHA256:
        raise RuntimeError('Downloaded spectral license differs from the pinned source')
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    with SOURCE.with_name('LICENSE').open('xb') as file:
        file.write(license_text)
    with SOURCE.open('xb') as file:
        file.write(source)


def verify_install():
    for path, expected in ((SOURCE, SOURCE_SHA256), (SOURCE.with_name('LICENSE'), LICENSE_SHA256)):
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pinned spectral source/license is missing or changed')


def helper():
    global _HELPER
    if _HELPER is None:
        verify_install()
        spec = importlib.util.spec_from_file_location('mac_kernels_lc_spectral', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _HELPER = module
    return _HELPER


def wrap_euler(sampler, patcher, positive, negative, *, enabled=False, grow_at_step=4, report=None,
               prepared_reference=None):
    """One outer KSAMPLER invocation; the inner Euler integrates each size stage."""
    if not enabled:
        return sampler
    verify_install()
    import comfy.model_base
    import comfy.model_sampling
    from comfy.k_diffusion.sampling import sample_euler
    from comfy.samplers import KSAMPLER

    if not isinstance(patcher.model, comfy.model_base.Krea2):
        raise ValueError('Spectral experiment requires Krea2')
    sampling = patcher.get_model_object('model_sampling')
    if (not isinstance(sampling, comfy.model_sampling.CONST)
            or not isinstance(sampling, comfy.model_sampling.ModelSamplingFlux)
            or getattr(sampling, 'noise_scale', 1.0) != 1.0):
        raise ValueError('Spectral experiment requires CONST flow sampling')
    if sampler.sampler_function is not sample_euler or sampler.extra_options or sampler.inpaint_options:
        raise ValueError('Spectral experiment requires unmodified deterministic Euler')
    if not 1 <= grow_at_step <= 6:
        raise ValueError('Grow step must leave at least two full-size steps')
    _validate_reference(patcher, positive, negative, prepared_reference)
    return KSAMPLER(_sample, extra_options={'inner': sampler, 'grow_step': grow_at_step,
        'report': report, 'patcher': patcher, 'positive': positive, 'negative': negative,
        'prepared_reference': prepared_reference})


def _validate_reference(patcher, positive, negative, prepared):
    excluded = {'control', 'gligen', 'concat_latent_image', 'mask', 'area', 'hooks'}
    if prepared is None:
        excluded.update({'reference_latents', 'reference_latents_method'})
    for conditioning in (positive, negative):
        if any(any(meta.get(key) is not None for key in excluded) for _, meta in conditioning):
            raise ValueError('Spectral experiment excludes control, reference, masked and area conditioning')
    if prepared is None:
        if patcher.model_options != {'transformer_options': {}}:
            raise ValueError('Spectral experiment excludes unowned model options')
        return
    if __package__:
        from .krea_reference import validate_spectral_reference
    else:
        from krea_reference import validate_spectral_reference
    validate_spectral_reference(patcher, positive, negative, prepared)


def _same_options(actual, expected):
    """Preserve callable/tensor identity through Comfy's structural option clone."""
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(_same_options(actual[k], v) for k, v in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (type(actual) is type(expected) and len(actual) == len(expected)
                and all(_same_options(a, b) for a, b in zip(actual, expected)))
    if expected is None or type(expected) in (str, int, float, bool):
        return type(actual) is type(expected) and actual == expected
    return actual is expected


def _sample(model, x, sigmas, extra_args=None, callback=None, disable=None, *, inner, grow_step,
            report, patcher, positive, negative, prepared_reference):
    extra_args = dict(extra_args or {})
    actual_patcher = helper()._patcher(model)
    if actual_patcher is not patcher or getattr(model.inner_model, 'cfg', None) != 1.0:
        raise ValueError('Spectral experiment requires the admitted Krea model at CFG 1')
    _validate_reference(patcher, positive, negative, prepared_reference)
    options = extra_args.get('model_options')
    if options is None:
        if prepared_reference is not None:
            raise ValueError('Spectral reference requires its bound execution options')
    else:
        if not isinstance(options, dict) or not isinstance(options.get('transformer_options'), dict):
            raise ValueError('Spectral execution model options are invalid')
        transformer = dict(options['transformer_options'])
        sample_sigmas = transformer.pop('sample_sigmas', None)
        if sample_sigmas is not sigmas:
            raise ValueError('Spectral execution sample_sigmas differs from the original schedule')
        # prepare_model_patcher creates empty wrapper/callback containers even
        # without hooks. They convey no behavior; any nonempty addition refuses.
        for key in ('wrappers', 'callbacks'):
            if key not in patcher.model_options['transformer_options'] and transformer.get(key) == {}:
                transformer.pop(key)
        bound = {**options, 'transformer_options': transformer}
        expected_load = {key: patcher.model_options['transformer_options'].get(key, {})
                         for key in ('wrappers', 'callbacks')}
        if 'to_load_options' in bound and _same_options(bound['to_load_options'], expected_load):
            bound.pop('to_load_options')
        if not _same_options(bound, patcher.model_options):
            raise ValueError('Spectral execution options differ from the admitted model')
    if extra_args.get('denoise_mask') is not None:
        raise ValueError('Spectral experiment excludes denoise masks')
    latent = getattr(model, 'latent_image', None)
    import torch
    if (x.dtype != torch.float32 or sigmas.dtype != torch.float32
            or sigmas.ndim != 1 or sigmas.device != x.device):
        raise ValueError('Spectral experiment requires prepared FP32 latents and device-matched sigmas')
    image_shape = x.ndim == 4 or (x.ndim == 5 and x.shape[2] == 1)
    if not image_shape or x.shape[0] != 1 or any(n < 8 or n % 4 for n in x.shape[-2:]):
        raise ValueError('Spectral experiment requires one image with spatial dimensions divisible by four')
    if latent is None or latent.shape != x.shape or torch_count_nonzero(latent):
        raise ValueError('Spectral experiment requires an empty initial latent')
    schedule = sigmas.detach().cpu().tolist()
    if (len(schedule) != 9 or not all(0 <= t <= 1 for t in schedule)
            or schedule[0] < 0.9 or schedule[-1] != 0
            or any(a <= b for a, b in zip(schedule, schedule[1:]))):
        raise ValueError('Spectral experiment requires eight descending full-denoise flow steps')
    h, w = x.shape[-2:]
    if prepared_reference is not None and tuple(prepared_reference.output_size) != (w * 8, h * 8):
        raise ValueError('Spectral reference output size differs from the admitted full latent')
    t = schedule[grow_step]
    t2 = 2 * t / (1 + t)
    event = {'source_sha256': SOURCE_SHA256, 'sigmas': schedule, 'grow_step': grow_step,
             'small_shape': [*x.shape[:-2], h // 2, w // 2], 'full_shape': list(x.shape),
             'grow_sigma': t, 'resumed_sigma': t2, 'remaining_sigmas': [t2, *schedule[grow_step + 1:]],
             'expansion_seed': (int(extra_args.get('seed', 0)) + 7) & 0xFFFFFFFF}
    if report is not None:
        report(event)
    return helper()._sampler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                             lc_inner=inner, lc_auto=False, lc_step=grow_step)


def torch_count_nonzero(value):
    import torch
    return bool(torch.count_nonzero(value).item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', required=True)
    parser.parse_args()
    prepare()
    print(json.dumps({'source': str(SOURCE), 'revision': REVISION, 'sha256': SOURCE_SHA256}))
