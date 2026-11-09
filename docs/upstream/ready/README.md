# Ready to post, October 13, 2026

Six items: branches rebased onto today's upstream heads, suites rerun, bodies trimmed, no em-dashes, no agent narration. Posted October 13, 2026: ComfyUI [#16996](https://github.com/Comfy-Org/ComfyUI/pull/16996) (fp16 VAE), [#16997](https://github.com/Comfy-Org/ComfyUI/pull/16997) (MPS OOM), [#16998](https://github.com/Comfy-Org/ComfyUI/pull/16998) (WAN Diffusers LoRAs), the [#16574 comment](https://github.com/Comfy-Org/ComfyUI/pull/16574#issuecomment-6149479608), PyTorch [#200782](https://github.com/pytorch/pytorch/issues/200782) and comfy-kitchen [#249](https://github.com/Comfy-Org/comfy-kitchen/pull/249). Agent-Provenance trailers kept. Each file holds the exact title and body. Branches are already on the public forks `roblennon/ComfyUI` and `roblennon/comfy-kitchen` (created today).

| Item | File | Head | Proof on today's base |
|---|---|---|---|
| CU-02 fp16 VAE | [comfyui-pr-mps-fp16-vae.md](comfyui-pr-mps-fp16-vae.md) | `mps-fp16-vae` `751a75c0` on master `8010b48d` | 201 passed, 14 skipped; ruff clean |
| CU-12 MPS OOM | [comfyui-pr-mps-oom-recovery.md](comfyui-pr-mps-oom-recovery.md) | `mps-oom-recovery` `861b6169` on master `8010b48d` | 202 passed, 14 skipped; ruff clean |
| CU-16 WAN Diffusers LoRAs | [comfyui-pr-wan-diffusers-lora-keys.md](comfyui-pr-wan-diffusers-lora-keys.md) | `wan-diffusers-lora-keys` `f647d9ed` on master `8010b48d` | 199 passed, 14 skipped; ruff clean |
| CU-04 UniPC comment | [comfyui-comment-pr16574.md](comfyui-comment-pr16574.md) | `unipc-pr16574-mps-evidence` `fec872c0` on PR head `045def13` | October 9 MPS/CPU evidence; #16574 still open, untouched since September 26 |
| PT-01 PyTorch issue | [pytorch-issue-mps-wide-linear.md](pytorch-issue-mps-wide-linear.md) | none (issue) | 2.14.1 and nightly 2.16 runs, October 9; full tables inline |
| CK-01 Kitchen backend | [kitchen-pr-mps-backend.md](kitchen-pr-mps-backend.md) | `mps-backend` `650dd6d` on main `13d3468` | 1082 passed, 3161 skipped under the GPU guard; ruff clean |

## Destination rules checked today

- ComfyUI: no PR template for ordinary PRs. CLA Assistant asks the PR author to comment `I have read and agree to the Contributor License Agreement` after opening; the allowlist already names `claude` and `ampagent`, so agent-touched commits are expected there. `check-ai-co-authors` rejects only `Co-authored-by` trailers naming agents; our commits carry none. The wiki asks first-time contributors to start with small PRs, which is why the five larger ComfyUI branches wait.
- comfy-kitchen: Apache-2.0, DCO sign-off on every commit (all eight carry it), same CLA bot. The maintainers merge large backend ports from outside (Ascend, HIP).
- PyTorch: bug-report form, labels `module: mps` and `module: performance`. No duplicate found in a refreshed search.

## Posting (done)

Open the three ComfyUI PRs from the fork against `master`, the Kitchen PR against `main`, the comment on #16574, and the PyTorch issue. With `gh`, the body file is the Markdown under `## Body` in each file (`sed -n '/^## Body/,$p' FILE | tail -n +2`); the title is the line under `## Title`. After each PR opens, the CLA comment is the author's own action.

Order that costs nothing: fp16 VAE, OOM, WAN LoRAs, UniPC comment, PyTorch issue, then Kitchen.

## Review round 1, October 13

CodeRabbit approved #16998 and asked for five small changes on the others, all valid and all applied the same day: a device-resolution test on #16996 (`c6c7240b`); on #16997, the CPU flag restored after import and the `AcceleratorError` case made conditional (`15ca1448`); on comfy-kitchen #249, `lora_kwargs` coerces the scale to a Python float with a test, and the README matrix gains an `mps` column plus the W4A8 rows. Suites rerun on each head; Kitchen 1083 passed under the GPU guard. Replies posted on each thread. No human maintainer comment yet.
