# Local LoRA training

This is a headless local trainer, separate from the Comfy generation GUIs. [train_lora.py](../train/train_lora.py) supports `krea2` and `qwen21`, batch one with frozen quantized bases and trainable plain LoRA factors. It does not train Identity Edit, video models or a RAW-to-Turbo distillation automatically.

## Prepare a run

The patched ComfyUI and Kitchen checkouts and model assets must already exist; see [runtime ownership](../CONTRIBUTING.md). [setup.sh](../train/setup.sh) creates `train/.venv` with Python 3.12 and the trainer's [locked dependencies](../train/requirements.lock.txt), then path-links Kitchen. It does not acquire runtime forks or checkpoints. Keep inference and training environments separate.

Set `MK_ROOT` for model/output storage before starting the process. Choose a supported architecture, its registered `--unet` filename, a dataset directory and a unique output directory. Krea defaults to the RAW INT8 base; Qwen needs an explicit matching Qwen checkpoint selection. Encoder/VAE names follow the trainer's architecture registry. Existing workbench support for another checkpoint does not itself establish training compatibility.

The dataset directory contains images and `dataset.json`:

```json
{"images": [{"file": "portrait.png", "caption": "rwbflat style, a portrait"}]}
```

Image filenames are relative to that directory. Captions become cached conditioning. Square `--res` is supported; `--buckets` accepts width/height pairs such as `512x768,768x512`, with positive multiples of 32. The deterministic crop/bucket geometry becomes part of cache/run identity.

Save defaults in a JSON config and supply `--config` to the trainer. Keys use argparse destination names (`cache_file`, `save_every`, `stop_after`), not dashed CLI spellings. Explicit CLI options override JSON values. Config-supplied dataset/output/cache paths are relative to the config file; CLI paths are relative to the invocation directory. Unknown keys, duplicate keys and invalid values are refused. Do not treat defaults as a tuned recipe for a new dataset.

## Run and recover

Launch `train/.venv/bin/python train/train_lora.py` with those selections under [gpu_runner.py](../bench/gpu_runner.py) monitoring, a finite deadline and the single memory policy. The older `with_gpu.sh` wrapper serializes ownership alone; it does not enable monitoring. The terminal/operator can perform setup and launch on your behalf.

The trainer caches text embeddings and VAE latents, then releases those preparation models before the training phase. `--cache-file` selects an existing cache and refuses to rebuild or replace it. Cache identity includes source/configuration/geometry; don't rename incompatible data into place.

`--steps` is the total target. `--stop-after` bounds updates in this session and saves resumable state. `--state-every` controls periodic optimizer/RNG recovery saves, separate from exported adapter checkpoints. `--resume` requires `OUT/state.pt` and matching base, configuration and cache identities. Preserve that state and its sources; an adapter safetensors alone is not a resumable optimizer checkpoint.

Previews default off. Opt-in preview controls use the actual Comfy sampler with raw-base conditioning and admitted CPU VAE decode, preserving training RNG/mode. They cost time and memory and are not the accelerated Turbo inference recipe. Start with a bounded session rather than launching an experiment matrix.

## Evaluate what was learned

Use [eval_lora.py](../train/eval_lora.py) with `--column base` or `--column adapter`, exactly one prompt and one selected LoRA per guarded process. The base column still takes the LoRA selection for matched artifact naming but does not apply it. Hold checkpoint, prompt, seed, conditioning and schedule constant, inspect finite final pixels, then assemble the comparison. Historical `eval_m1.sh` and `m2_*.sh` recipes and `--column all` are not the current bounded evaluation procedure.

A changed palette or low training loss does not prove learned style. Qwen's same-subject trigger-present/absent × adapter-present/absent comparison remains an explicit follow-up. The bucket/resume/preview pilot proves its four-update integration scope, not convergence or general quality.

For baked Krea inference checkpoints, use [Workbench export](how-to-krea-workbench.md): merge compatible plain LoRAs into an already compatible floating Turbo source and quantize. Export does not confer Turbo behavior on RAW. Training/configuration/backward/export contract tests are mapped in the [contributor guide](../CONTRIBUTING.md#verification-by-boundary).

## Helper scripts

These sit beside the trainer in `train/` and were used for the experiments in [findings](findings.md):

- `make_dataset.py`, `make_dataset_qwen.py`: build synthetic style datasets from Krea 2 Turbo INT8 or Qwen-Image 2.1 INT8 with the Viggle 6-step LoRA.
- `plot_loss.py`: draw a loss curve PNG from a run's `log.jsonl`.
- `compare_loras.py`: compare what two LoRAs learned, layer by layer, on the CPU.
- `hue_stats.py`: mean hue and saturation of an image's coloured pixels, used to check palette shifts.
- `diag_vae_nan.py`: a one-off diagnosis of which Qwen 2.1 VAE encodes at 1024px went non-finite.
