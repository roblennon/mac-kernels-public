# Upstream series refresh verification

2026-10-04 UTC, CPU-only packaging pass. Source implementations had already received their own reviews and integrated checks; this pass checks the separated branches. It does not claim the complete project or server memory acceptance is finished.

## Scope and undo

Refresh the upstream-ready branches and their descriptions without public submission. Existing private branch history is preserved; the backend and attention fixes append to it. New independent and dependent series live under `upstream/*` in the two preparation worktrees. At packaging time these follow-up refs were unpublished. They have since been pushed privately; preserve that history and use follow-up commits for corrections. No primary source tree, model, training state, public remote or CLA was changed by this lane.

## New CPU evidence

| Branch | Focused result | Boundary |
|---|---|---|
| Kitchen backend | 2 pass | Real CPU allocations/GEMMs; substituted Metal boundary proves old chunk temporaries die before the next decode |
| Kitchen training | 48 pass, 57 skip | 39 attention-gradient CPU cases, 7 packed-wrapper contracts, 2 lifetime cases; device backward cases skipped |
| Kitchen block-formats | 2 pass, 150 skip | Lifetime checks pass; all format GPU cases collected and skipped |
| Kitchen GGUF | 2 pass, 177 skip | Root development environment includes `gguf`; all GGUF GPU cases collected and skipped |
| ComfyUI attention | 71 pass, 7 skip | Real CPU SDPA broadcast/mask/chunk oracle; MPS default/upcast cases skipped |
| ComfyUI clone-accounting | 4 pass | Original list representation, real clone and memory-size operations |
| ComfyUI adapter-types | 18 pass | Four accounting cases with real `LoraSide` plus fourteen scale-range admission cases |
| ComfyUI block-formats | 21 pass | Device capability and encoder dtype/memory admission policies |
| ComfyUI direct-load | 22 pass, 2 skip | CPU direct-option boundaries, node compatibility and thin-Wan reconstruction; direct-MPS byte/storage cases skipped |
| ComfyUI OOM recovery | 10 pass | Real VAE retry path, substituted failing decoder boundary |
| ComfyUI encoder ownership | 21 pass | Nine Wan ownership cases plus twelve small real Krea conditioning/storage/LoRA/file-lifecycle cases |
| ComfyUI cache ownership | 53 pass | Six real cache/encoder lifecycle cases plus async provider, execution-list and reentry checks |
| ComfyUI mapped-offload candidate | 40 pass, 2 skip | 17 mapping/offload contracts, 3 storage cases, plus accounting, scale-range and loader checks; not additive independent coverage |

CPU runs explicitly forced ComfyUI's CPU mode. When collecting device cases, MPS availability was disabled after PyTorch Dynamo initialization. No MPS tensors, shaders or model generation were run by this pass. Kitchen used the local branch checkout; ComfyUI imported the installed integrated Kitchen package for CPU-only tests. The exact dependency-pair GPU checks below were subsequently run by the parent.

The accounting tests exposed a stale list fixture after the additional-adapter merge. Commit `245bc0f0` changes only the two fixture assignments to `LoraSide`; all original byte-count, clone, mutation and unpatch assertions remain. The primary tree received the equivalent fix as `2caf86ad`. This CPU suite was not part of the earlier 113-case integrated GPU run.

The old split sidepath suite also required DoRA, small diffs and model strength to merge. Commit `d8674953` removes those obsolete fallback cases because the new adapter suite verifies their exact side-path equivalence; set/function fallback assertions remain. No tolerance or production behavior was relaxed.

The direct-loader implementation previously had only an external integration test. Commit `cb7f20e7` packages it into the upstream tree and adds CPU non-regression cases. Block-format commit `354d0f5` removes one pre-existing blank line at EOF so the branch diff passes whitespace checks.

## Source and merge checks

Every prepared branch individually merges via `git merge-tree --write-tree` onto the fetched public heads: Kitchen `be003b7`, ComfyUI `f1072eb0`. This includes the unchanged original branches. Changed Python source is compiled without importing it, and each branch diff passes whitespace checks. These are static checks, not rebased runtime evidence. All implementation commits retain their human author; ComfyUI's prepared ranges contain no co-author trailers.

ComfyUI adapter-types plus block-formats merge cleanly. Kitchen training plus block-formats merge cleanly. Kitchen GGUF plus block-formats conflict in `comfy_kitchen/backends/mps/__init__.py` where both add operation registrations. The integrated working branch already preserves both sets, but these separate review branches intentionally retain their focused histories. Resolve and test that combination when stacking for publication.

Implementation cherry-picks preserve source behavior. Test packaging separates unrelated VAE tests from the ComfyUI format policy suite and makes the direct-option suite independent of the separate mapped-offload feature. No implementation conflict was hand-rewritten.

Ruff is absent from the available environments, so this refresh does not claim a new Ruff pass. Historical lint results remain labeled as historical.

## Coordinated GPU checks

A separate resource supervisor owns GPU scheduling. Use the root development environment for Kitchen, because it carries `gguf`; use the training environment for ComfyUI's current safetensors loader. Set the import path to the preparation checkouts and verify the imported module paths before running. Serialize branch switches and tests; do not switch a worktree while a test process uses it.

| Kitchen tip | ComfyUI tip, if needed | Focused suite |
|---|---|---|
| `upstream/mps-backend` | none | `tests/test_mps_linear_lifetime.py`, `tests/test_mps_int8_linear.py`, `tests/test_mps_w4a8.py` |
| `upstream/mps-training` | none | `tests/test_mps_autograd.py`, `tests/test_mps_attention_backward.py`, `tests/test_mps_packed_autograd.py` |
| `upstream/mps-block-formats` | none | `tests/test_mps_block_formats.py` |
| `upstream/mps-gguf` | none | `tests/test_mps_gguf.py` |
| `upstream/mps-backend` | `upstream/mps-pytorch-attention` | `tests-unit/comfy_test/sdpa_bounded_test.py`, `tests-unit/comfy_test/mps_attention_default_test.py` |
| `upstream/mps-backend` | `upstream/mps-lora-types` | `tests-unit/comfy_test/lora_sidepath_mps_test.py`, `tests-unit/comfy_test/lora_sidepath_types_mps_test.py`, accounting and scale-range files |
| `upstream/mps-block-formats` | `upstream/mps-block-formats` | `tests-unit/comfy_test/mps_weight_only_test.py`, `tests-unit/comfy_test/mps_policy_test.py` |
| `upstream/mps-backend` | `upstream/mps-direct-load` | `tests-unit/comfy_test/mps_pread_load_test.py` |

Mapped-offload additionally needs the integrated representative generation/offload workload and memory-pressure evidence. Its CPU checks cannot replace that acceptance. Packed full-model training acceptance and the trainer's bucket/preview pilots are separate parent-owned work, not part of this upstream packaging proof.

All eight rows ran against the listed branch pairs in separate processes on MPS. Backend: 324 passed; training: 103; block formats: 150; GGUF: 177; attention: 78; adapter types/accounting/range: 130; format policy: 31; direct loader: 4. This is 997 passing test executions across overlapping suites, not 997 independent model scenarios. Every process printed the imported Kitchen and ComfyUI checkout paths. The supervisor completed normally in 30.38 seconds with 0.77 GiB peak process RSS and no swap increase. Exact tested commit hashes are in `logs/upstream-gpu-matrix.json`; complete output is `logs/upstream-gpu-matrix.log`.

These results establish the focused branch GPU contracts. They do not establish G1 persistent-server acceptance or public submission. Mapped-offload remains a candidate until its separate full-model lifecycle and server measurements are resolved.

## Contributor trailer normalization

The ComfyUI CI script initially rejected the adapter-types source commit's AI co-author trailer. Only the two unpublished dependent series were rebuilt to remove that one trailer. Human author, `Agent-Provenance`, source commit references and DCO sign-offs remain. At normalization time, both trees were byte-identical to their pre-normalization versions, preserved under `codex/u2-backup-lora-types-20261004` and `codex/u2-backup-mapped-offload-20261004`. Existing private-published history was not rewritten. The mapped-offload branch has since gained the default-off follow-up described below. The repository's actual `check-ai-co-authors.sh` passed on all 14 ComfyUI branches prepared at that checkpoint.

## Direct-loader dependency

Commit `b0680dea` raises ComfyUI's declared safetensors floor from 0.4.2 to the tested 0.8.0, and mapped-offload merges that dependency at `375bbc12`. Installed distribution metadata reports official Hugging Face safetensors 0.8.0, installed by uv without a direct-source override; its `safe_open` signature exposes the keyword-only `backend`. The [official PyPI 0.8.0 release](https://pypi.org/project/safetensors/0.8.0/), published 2026-06-09, requires Python >=3.10, matching ComfyUI's own Python floor. No package was installed or changed in the environment. The earliest release supporting pread was not investigated or claimed.

## Independent packaging review

An independent review covered the manual test adaptations and dependency contract (run `20261004T023948942892Z-90bcae4b8227`) and the format-policy test split (run `20261004T023948831055Z-38bb10458d18`). Both completed their required coverage with no actionable findings. Optional whole-file context remained limited, while required changed hunks and support were complete. Production implementation provenance was supplied explicitly; this was not a new review of the complete kernel or generation design.

The docs review (`20261004T023948812498Z-9759b456e8bb`) found two S3 issues: undefined lettered dependency shorthand and an insufficiently explicit pending GPU statement in the format follow-up. The sidepath dependency was named explicitly, and the format description at that time stated that its separated-branch GPU suite had not run. That description now reports the subsequent 31 passing MPS checks.

The already-reviewed Krea encoder follow-up `607b2ccd` was appended unchanged as `a135aafc` after review dispatch. Its twelve CPU tests pass on this branch alongside the nine Wan tests, and the refreshed tip still merges cleanly onto the fetched upstream. No heavy model job or GPU work was run.

The focused docs recheck (`20261004T024157190872Z-8473794327ce`) completed required coverage with no actionable findings after both wording fixes and the Krea/dependency updates. That earlier source audit covered 31 distinct Kitchen and 47 distinct ComfyUI changed Python files; all compiled, merged individually onto fetched upstream and passed whitespace checks. The additional cache and mapped-offload checks are recorded below. Private publication was still pending at that review; the publication checkpoint below records the subsequent push. The coordinated branch GPU results above supersede the earlier pending GPU status.


## Encoder cache follow-up packaging

`upstream/mps-cache-ownership` (`274356f1`) is an independent cherry-pick of reviewed source `c75c2013` onto `65787d66`. Its stable patch ID is `8655faecfb6b222961357bacb8c3ec74c3812c92`, identical to the source; no implementation or test adaptation was needed. It merges cleanly onto fetched upstream `f1072eb0` and alongside encoder-ownership `a135aafc`. These are merge checks, not execution of a rebased tree.

The separated checkout passed **53 CPU tests**, with ComfyUI CPU mode explicitly selected before model-management import. The run printed and checked the preparation checkout's source paths. Kitchen came from the installed integrated development checkout; the storage-introspection API used by this branch already exists in the pre-MPS Kitchen base `5eeefcf`, so the new backend series is not a dependency. Six tests cover real mapped CLIP weights, shared clones, pending consumers, physical quantized storage, a warm executor and actual RAM remeasurement after shared-entry eviction. Existing cache-provider, execution-list and reentry suites supply the remaining cases. The primary integrated source had separately passed 92 CPU checks. No GPU tensors or full models were run for this packaging pass.

The implementation's final independent review (`20261004T025910197053Z-f6566f35cfa8`) completed with no findings or coverage gaps. The review's earlier estimated-freed-bytes finding is fixed in this exact patch: cache eviction no longer pretends shared live storage has been reclaimed. The branch passes whitespace, Python compilation and the repository's contributor-trailer check, with original human author, source reference, provenance and DCO preserved. Full server pressure acceptance remains unproven by these CPU tests.


## Mapped-offload default-off follow-up

The mapped-offload candidate now ends at `b0d3ac5b`, an unchanged cherry-pick of independently reviewed `f290a26f` after `375bbc12`. Ordinary diffusion-file loading leaves mapping retention off. Explicit `model_options["mmap_offload"] = True` retains the experimental path, including clone and cached reconstruction behavior. The exact dependencies remain adapter-types `d8674953` plus direct-loader `b0680dea`; it does not depend on cache-ownership or encoder-ownership. This correction preserves the candidate mechanism without advertising it as accepted generation-memory policy.

The source change passed independent review (`20261004T030440902508Z-6ea869631e37`). The prepared tip merges cleanly onto fetched upstream `f1072eb0` and passes the contributor-trailer and whitespace checks. Its focused CPU run passed **40 tests**, with two direct-MPS cases skipped. CPU mode was explicit, MPS availability was disabled for collection, and module paths confirmed the prepared ComfyUI checkout; the installed integrated Kitchen supplied existing CPU layout contracts. The two changed Python files compile. Its stable patch ID `0744122d6614ef810d083ed9e2861ec56adc8c02` matches the source exactly. The earlier GPU matrix did not exercise mapped-offload and is not presented as validating this policy.

The cache/default-off packaging descriptions received a final independent review (`20261004T030828570794Z-aa1e96e6e8b9`) with complete coverage and no findings. Earlier checks identified stale GPU-pending statements and an outdated branch count; the record now distinguishes historical snapshots from the then-current 14-branch inventory. No branch was pushed or public PR opened by this follow-up.


## Quantized encoder opt-in follow-up

At this opt-in packaging checkpoint, `upstream/mps-block-formats` ended at `5c75872e`, an unchanged cherry-pick of reviewed source `05510e7f`; the later FP32 follow-up below advances it to `ca51eaed`. Automatic quantized-encoder promotion requires exact `COMFY_MPS_TEXT_ENCODER=1`; explicit caller devices retain precedence. This gate preserves the existing admission calculation, including its known lack of allocator-fraction and current-driver-allocation awareness. Complete quantized encoder conditioning was unqualified at this checkpoint. A later fully offloaded H3 encoder diagnostic passes on integrated source; whole-encoder GPU residency and H3 API/video acceptance remain unqualified.

The prepared checkout passed **25 CPU tests** from the format/admission and new opt-in suites. CPU mode was selected before import, MPS availability disabled, and imported paths confirmed the preparation ComfyUI checkout and installed integrated Kitchen. Eight new cases cover disabled values and real CLIP construction with explicit CPU/MPS load-device choices while weights stay on CPU. The integrated source had passed 59 adjacent CPU checks and failed three new guard cases before the change. Independent review (`20261004T031541653739Z-055f0f57f507`) completed with no findings or coverage gaps.

Stable patch ID `78170c978536fd254b1f1285a327e46722e4fbfd` matches the source. The branch merges cleanly onto fetched upstream `f1072eb0`, passes whitespace and contributor-trailer checks, and all three changed Python files compile. No GPU run, full model load, private push or public PR was performed in this packaging pass.


## Explicit direct-file and node option packaging

Reviewed source `de4ff712` is an unchanged cherry-pick at `53b4ec3d` on `upstream/mps-direct-load`, following the existing safetensors dependency commit `b0680dea`. Both implementation commits have stable patch ID `909c73fb34d9d1f80f5bdf491cf39bfc3e317143`. The option stays independent of mapped-offload, the new Kitchen backend and the other ComfyUI series; those remain relevant when composing the full accelerated generation stack, not prerequisites for this loader change.

The first separated CPU run exposed three test-only `AttributeError` failures because this independent branch has no `mmap_offload` property. Test-only commit `1f77853d` makes the portable assertions accept an absent property as disabled and still reject `True`; no production code or retention behavior was added. Its independent review (`20261004T032014819504Z-a81c6630d2d0`) completed with full coverage and no actionable findings. The final focused run passed **22 tests**, skipping the two direct-MPS cases. CPU mode and imported preparation-checkout paths were explicitly verified; the installed integrated Kitchen provided existing CPU layouts. No GPU tensors, full model reads or generation were used in this packaging lane.

The prepared branch merges cleanly onto fetched upstream `f1072eb0`, passes whitespace and contributor-trailer checks, and preserves the human author, source reference, agent provenance and DCO sign-off. Source review `20261004T031412186097Z-34b82da334c0` completed with one accepted S3 limitation: system-based eviction may leave idle GPU models in place, after which the driver-budget check refuses loading. This conservative refusal avoids introducing broad eviction that could worsen CPU pressure. The explicit node remained experimental pending persistent-server acceptance at this packaging checkpoint. Later parent-owned integrated production-node and bounded Qwen API runs pass as recorded below; the option remains experimental and default-off. The earlier four-case GPU matrix result establishes only the pre-existing low-level loader contract.


### Integrated production-node follow-up

A separate run exercised `UNETLoader.load_unet(..., mps_direct_load=True)` without a `utils.load_torch_file` hook. Using the same cached CPU conditioning, the 512px Qwen Viggle portrait completed in **33.358 seconds**, with finite pixels, **195/195** patched layers reached and **192/192** quantized side applications. Peak process-group RSS was **9.860 GiB**, MPS driver allocation **8.97 GiB**, and swap remained **1.95074 GiB**. The result is in `logs/qwen-production-direct-node-proof.log`. It verifies the integrated production-node path; it does not include fresh encoding or execute this separated upstream checkout. The native server's `/object_info` also exposed the optional input. Full Qwen server generation and repeated-server acceptance remained pending at this node-proof checkpoint. The later bounded three-request API session passes on integrated source: 512×512, explicit native direct INT8 loading, fresh CPU encoding, 25 Euler/simple steps, CFG 1 and full VAE, with no LoRA and enhancer off. The unchanged launcher uses paired 0.5/0.4 watermarks and `--cache-ram 18`; the whole 332.100-second session completes with flat swap and distinct inspected seed/prompt outputs. Every request re-encodes and samples; no resident warm-model speedup, ordinary-loader qualification or execution of the separated branch follows.

## Private publication checkpoint

On October 4 UTC, both private remotes were fetched and received all 14 prepared ComfyUI branches and all four prepared Kitchen branches without force. This includes the subsequent encoder gate, cache-ownership and direct-option follow-ups recorded above. At that checkpoint the integrated ComfyUI working branch was pushed through `05510e7f`; Kitchen and GGUF working branches were already pushed through `56e8b96` and `7156e0a`. No public remote was written and no PR or CLA response was submitted. Earlier statements that a particular packaging pass did not push describe that pass, not current publication status.


## Temporal VAE padding follow-up

`upstream/mps-vae-temporal-padding` (`0252e040`) is an independent cherry-pick of reviewed source `9d9b4232` onto `65787d66`. Both commits have stable patch ID `8741b915452360f0767b65f74062dc2dc4a52bfa`; no implementation or test adaptation was needed. The branch merges cleanly onto freshly fetched upstream `f1072eb0`, and its two changed Python files compile and pass whitespace and contributor-trailer checks. Human author, source reference, agent provenance and DCO sign-off are preserved. The parent subsequently pushed this fifteenth ComfyUI branch and integrated source to the private repository. The private refs are verified at `0252e040` and `9d9b4232`, respectively; all 15 prepared ComfyUI branches are now privately published.

The prepared checkout passed **8 CPU tests**, with its 10 MPS cases deselected. CPU mode was explicit, MPS availability was disabled after Dynamo initialization, and imported source paths confirmed the independent ComfyUI checkout and installed integrated Kitchen. No GPU work was run in this packaging lane. The parent subsequently ran all **18 CPU/MPS cases** against this exact independent branch, with both imported checkout paths printed and the ComfyUI path asserted. The run completed normally in **2.121 seconds** with no swap growth; evidence is `logs/t1-avgdown-upstream-regression.log`. These are the same contracts also tested on the integrated source, not additive scenario coverage.

The exact large-image defect was reproduced with two failing FP16/FP32 tests before the source fix; the integrated source then passed all **18 CPU/MPS tests**. Unhooked native 1024px and 512px image roundtrips now agree with CPU references. The 1024px FP16 latent is identical to the temporary candidate; the native 512px FP32 latent differs from CPU FP32 by relative L2 `1.88e-6`. Old 512px cached latents were also affected, so the 512px result is a corrected-reference check, not an unchanged-output claim. The focused [PR description](comfyui-mps-vae-temporal-padding.md) records quality, runtime and resource measurements.

Independent review `20261004T041706106674Z-0a145e37301c` completed with full coverage and no actionable findings. The patch addresses the demonstrated `AvgDown3D` use of the existing [PyTorch temporal-padding defect](https://github.com/pytorch/pytorch/issues/194922); it does not claim a new framework fix or repair existing caches. No public remote, PR or CLA response was submitted.


## H3 FP32 force-cast follow-up

`upstream/mps-block-formats` now ends at **`ca51eaed`**, appended after its privately published `5c75872e` head. It is an unchanged cherry-pick of independently reviewed integrated source **`6ad852be`**; both have stable patch ID `087299be5c429f373c34651c36a90722925fb993`. The human author, source reference, agent provenance and DCO sign-off are preserved. The branch merges cleanly onto freshly fetched ComfyUI `f1072eb0`; three changed Python files compile, and whitespace and contributor-trailer checks pass. This packaging work made no source adaptations.

The separated checkout passed **29 CPU tests**, explicitly in CPU mode with MPS unavailable for collection; its **16 MPS tests were skipped**. A subsequent run covered exactly those 16 MPS weight-only cases against prepared ComfyUI `ca51eaed` and its declared Kitchen block-format dependency `354d0f5`: **16 passed, zero skips/failures**, with actual import paths and branch hashes asserted. The supervisor exited normally in **4.302 seconds**, at **0.323 GiB sampled peak group RSS** and flat **1.677363 GiB swap** (`logs/h3-upstream-force-cast-mps.log`). Integrated source `6ad852be` separately passed all **28 targeted CPU/MPS regressions**, zero skips. The counts overlap and are not additional independent scenario coverage.

Both `6ad852be` and `ca51eaed` were pushed privately; both tracking refs match. The global pre-push hook used the wrong Python environment and failed collection, so the push used the established vendor bypass backed by these explicit integrated/separated checks. This is not a claim that the global hook passed. No public remote, PR or CLA response was submitted.

The production fix preserves native packed NVFP4 under CLIP's FP32 force-cast path. Full-encoder acceptance was still open at this packaging checkpoint: an earlier full 388-token encode returned finite conditioning and all 351 expected native operations, then failed the helper's invalid positive-residency assertion. After correcting only that validator, the final retry stopped for system memory pressure at **122.875 seconds**, with **337 native returns** (336 NVFP4 plus one INT8 embedding), **3.560 GiB sampled peak group RSS**, and flat swap. The separately retained disk-error attempt also executed native layers before ENOSPC disrupted the supervisor; its filename's “prelaunch” label is misleading. Attempt evidence. No guard-complete full-encoder or H3 template result follows from the passing focused suite.

Independent review of these preparation notes and the H3 outcome records completed with all six required coverage atoms and no actionable findings (`20261004T065553648609Z-4016c1318081`). One optional surrounding-source view was limited; the changed claims and complete compact evidence were covered. Fourteen recorded source-file hashes, 64 document links and whitespace checks pass.


A subsequent quiet-host run of the unchanged integrated source and frozen encoder recipe now provides the complete guarded result: **80.073 seconds**, **3.546 GiB** sampled peak group RSS, flat **0.569521 GiB** swap, all **350 NVFP4 plus one INT8 embedding** operations, finite CPU conditioning and zero persistent MPS weights within the inclusive 0–1 GiB cap. The actual initial free-memory reading was **83%**. Earlier failure records remain preserved. This closes bounded F1 full-encoder execution; selected-layer numerical comparison remains distinct, and H3 API/video and ordinary CPU-encoder acceptance remain open. This result does not change the prepared-branch identity or constitute public submission. Complete evidence and report/log hashes.
