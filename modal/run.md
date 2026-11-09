# Running the Modal scaffold

The `modal` CLI (1.6.3) is installed with pipx. Check auth first; set it up only if the check fails.

```
modal token info            # succeeds when ~/.modal.toml holds a live token
modal setup                 # only if the above fails; opens a browser and writes ~/.modal.toml
```

Run from the repo root. Nothing under `vendor/` is mounted; the container has only the pinned packages in `modal_app.py`.

```
modal run modal/modal_app.py::gguf_dequant_check                 # CPU, under a minute, cents
modal run modal/modal_app.py::cuda_gemm_bench_t4                 # one GPU, a few minutes, cents
modal run modal/modal_app.py --gpus T4,A10G,A100                 # CPU check plus three GPUs in parallel
modal run modal/modal_app.py --skip-gpu
modal run modal/modal_app.py --skip-cpu --gpus A10G > modal/results/<date>-cuda-gemm-anchor.json
```

Notes

- GPU runs need a payment method on the workspace even inside the $30/month Starter allowance (added 2026-11-09).
- Modal's internal pip mirror served only torch 2.15.0 on 2026-11-09; older pins fail the image build.
- Timings exclude image pull and container start; each function warms up on the device before timing, and reports the median of `n_iters` CUDA-event timings, matching `bench/bench_int8_linear.py`.
- GPU types are separate functions (`cuda_gemm_bench_t4`, `_a10g`, `_l4`, `_a100`). On 2026-11-09 L4 sat in the scheduler queue for 20 minutes with no capacity, so it is not in the default list. Do not add `region=` or `nonpreemptible=`; both multiply cost.
- Functions return dicts only. Returning a torch tensor fails locally with `DeserializationError` because the pipx `modal` env has no torch.
- Anything longer than a laptop sleep: add `--detach`.
