# ComfyUI follow-up PR descriptions

Public PRs have not been opened. Exact branch dependencies and publication status are in the [series index](README.md); current checks are in [refresh-verification.md](refresh-verification.md).

## Clone accounting

Branch `upstream/mps-lora-accounting`, head `98956074`, based on `upstream/mps-lora-sidepath`.

**Title:** Count resident LoRA side factors once across model clones

Cloning a model with resident side factors could copy a total size that already included them, then add them again when reporting the clone's size. Clone bookkeeping now preserves the base model size while the size query counts the current shared resident factors. Four CPU cases exercise explicit and inferred base sizes, factors attached before and after cloning, multiple clone generations, factor replacement and unpatch.

Source commit: `a531016e`. This branch uses the original list representation from the initial sidepath PR; the adapter-types follow-up changes the test fixture to the real `LoraSide` container without changing its accounting assertions.

## Adapter types

Branch `upstream/mps-lora-types`, head `d8674953`, based on clone-accounting.

**Title:** Extend the MPS LoRA side path to additional adapter types without requantizing weights

LoKr, LoHa, supported LoCon mids and conv-shaped 1x1 factors, small diffs, DoRA and `strength_model` now have side representations. DoRA and model strength become channel scales on the base and earlier terms, preserving `calculate_weight`'s order. Supported adapters keep the packed base unchanged; unsupported shapes, patch functions, set patches and over-budget cases retain the merge fallback.

Scale folding checks representability before admission. A fold that would overflow or erase a nonzero representable value falls back to the existing merge instead of silently accepting a changed effective weight. Tests compare against the real fp32 merge, verify unchanged packed bytes and fallback behavior, and cover both row and input-axis scaling. Four clone-accounting cases and fourteen scale-range cases pass on CPU. The coordinated separated-branch MPS run also passed 130 adapter, sidepath, accounting and scale-range test executions; exact dependencies and limits are in the refresh record.

Implementation sources: `29eeb20e`, `907a73ed`. Packaging commits `245bc0f0` and `d8674953` update the accounting fixture and remove three obsolete assertions that newly supported types must merge. Their stronger exact-equivalence coverage lives in `lora_sidepath_types_mps_test.py`; set/function fallback checks remain in the initial suite.

## Formats and text encoders

Branch `upstream/mps-block-formats`, head `ca51eaed`, based on the initial LoRA sidepath. Requires Kitchen block-formats as well as backend.

**Title:** Use MPS weight-only block formats with experimental opt-in quantized text encoding

ComfyUI enables NVFP4, MXFP8 and ConvRot W4A4 when Kitchen advertises their MPS kernels. These weight-only operations preserve float activations, including when a checkpoint requests full-precision matrix multiplication. Ordinary and hook-scheduled text encoding use the same scoped dispatch context and restore the previous layer settings afterward.

Encoder promotion requires explicit `COMFY_MPS_TEXT_ENCODER=1` while whole-encoder GPU residency remains unqualified. The existing policy applies only to quantized formats with native kernels and accounts for execution dtype and dense weights against recommended working-set and system free-memory headroom. It does not account for the configured allocator fraction or current driver allocation. Explicit caller device selection retains precedence. Unset, disabled or other environment values preserve the ordinary CPU placement; unsupported formats and unquantized encoders retain their previous behavior.

The opt-in follow-up passes 25 CPU policy and real CLIP-constructor checks on the separated branch; it is unchanged from independently reviewed source `05510e7f`. The earlier coordinated MPS run passed 31 format/policy checks against the specified Kitchen block-format branch, before this qualification gate. Those focused contracts do not establish whole-encoder GPU acceptance. Earlier source commits: `d1c94797`, `88dbeb97`, `9686445b`, `fa7bb09d`. The source policy file also contained unrelated VAE tests; the split branch keeps those on the existing fp16-VAE series.

The appended FP32 text-encoding fix (`ca51eaed`, unchanged from reviewed `6ad852be`) keeps packed NVFP4 dispatch when the real CLIP path forces FP32 compute. It applies only inside the supported MPS weight-only text context, with FP32 nongradient input and no weight/bias functions. Other formats, devices, dtypes, gradients and ordinary force-cast behavior keep their prior paths; nested and exceptional exits restore the context.

This follow-up passes 29 CPU checks on the separated checkout, with 16 MPS cases skipped in that CPU run. Those exact **16 MPS cases** then ran **with zero skips**, using Kitchen block-formats `354d0f5`; both actual import paths and branch identities were asserted. The monitored run completed normally in 4.302 seconds with flat swap. The integrated source separately passed all 28 targeted CPU/MPS regressions. These are focused overlapping contracts, not additive full-model coverage.

The integrated source now completes a guarded 388-token H3 encoder diagnostic: all 350 NVFP4 layers plus the INT8 embedding execute once, producing finite CPU conditioning with zero persistent MPS weights within the inclusive 0–1 GiB cap. The quiet-host run exits normally in 80.073 seconds with flat swap. This closes bounded F1 encoder execution; numerical reference comparison remains at selected-layer scope. It does not qualify whole-encoder GPU residency, ordinary CPU defaults or H3 video/API templates, and it is not full-model execution of the separated branch. Earlier invalid-residency-assertion and pressure failures remain preserved.

## Direct loading

Branch `refresh/mps-direct-load` in `vendor/ComfyUI-upstream`, head `8ebf0c5c`: the five earlier commits cherry-picked onto upstream `08ff3c11` (October 9) plus one fix commit. Supersedes `upstream/mps-direct-load` (`1f77853d`). Independent. The diffusion-file/node option remains **experimental and default-off**. Integrated Qwen production-node and bounded three-request API proofs (October 4) pass; other formats, ordinary loading and general server qualification remain separate.

**Title:** Offer explicit direct MPS loading for INT8 ConvRot diffusion models

Direct MPS safetensors loading uses the pread backend for weight tensors and a scoped CPU reader for `.comfy_quant` JSON-byte tensors. Quantized module loading can then read its metadata without copying the device weight back through CPU. The readers close together, and the disable-mmap path preserves the intended ownership of CPU metadata. The `backend` parameter and the pread backend first shipped in safetensors 0.8.0 (June 9, 2026), which is why the branch raises the floor from `>=0.4.2` to `>=0.8.0`.

The file API now accepts `model_options={"mps_direct_load": True}`, and the existing UNETLoader exposes an optional advanced boolean with the same name. Both default off; existing graph inputs and two-argument node calls retain ordinary loading. The option requires static MPS loading with CPU offload and supported INT8 ConvRot metadata. It rejects float8/NVFP4 storage, other quantization formats and unsupported device/operation combinations before device reads. BF16, F16 and F32 unquantized storage is admitted. Preflight opens only storage descriptors and bounded JSON metadata, then closes its CPU reader. Loading proceeds only with enough system RAM and driver headroom; PyTorch's active-allocation counter cannot account for all native pread buffers.

After a direct load the patcher is registered through `load_models_gpu(force_full_load=True)`, the same thing the checkpoint loader does when it loads directly to GPU. Without that, the weights sat in device memory unregistered, so `free_memory` could not offload them and a second loader's admission check failed against idle residency.

Driver-only refusal is deliberately conservative. The existing eviction pass responds to system RAM, so idle device models may remain when driver headroom is insufficient. The loader reports that shortage rather than forcing every model into CPU memory. Clone reconstruction preserves the explicit option. This branch does not introduce or depend on mapped-offload retention.

**Refresh evidence (October 9, M1 Max 32 GiB, Torch 2.14.1, safetensors 0.8.0).** Upstream `08ff3c11` still has no direct-to-MPS path: `load_diffusion_model` always reads to CPU, and on the real Wan 2.2 5B INT8 ConvRot file (4.77 GiB) the ordinary path takes 0.1 s to map plus 16.8 s in `load_models_gpu` to reach MPS (driver 5.11 GiB). On the same file, `safe_open(device="mps")` with the mmap backend takes 10.0 s against 1.8 s with pread, at the same 4.77 GiB driver allocation and 9.22 GiB peak RSS; pread is the faster path, not the only one. Refusals on real files all happen before any device read, with driver allocation still 0.00 GiB afterwards: the NVFP4 Krea file, fp8 dtype, a CPU load device, a `.pt` path, dynamic loading, and a driver shortfall. With no option the loader passes `device=None` and all 300 quantized weights land on CPU. The full real-file direct load was refused by its own admission check on this host (needs 9.72 GiB headroom, 8.63 GiB free, swap at 10 GiB) and was not forced. On real MPS the thin ConvRot fixture loads through the actual loader, stays resident on MPS, is registered, and evicts to CPU through `free_memory`. Upstream overlap: open PR #15438 (August 9) rewrites the `--disable-mmap` branch of `load_torch_file` into a non-mmap reader, so whichever lands second needs a rebase; nothing upstream loads diffusion weights directly to MPS. Issue #16729 (closed October 2) concerns the aimdo mmap loader, which this branch does not touch.

**Review (October 9).** Independent review of the rebased branch: one blocking finding (unregistered residency, above), two should-fix (F16 storage was rejected although fp16 compute is accepted; a real MiniMax H3 INT8 ConvRot export carries 102 F16 tensors and was refused; and the reconstruction test proved option forwarding only). All three are fixed in `8ebf0c5c` with tests that failed first; the re-review of the fix found nothing. Full `tests-unit/comfy_test` on CPU: 215 passed, 17 skipped. The two loader files on real MPS under the GPU guard: 27 passed, zero skips. Probe outputs, review transcripts and the probe scripts for all three refreshed branches are in `logs/2026-10-09-mps-ownership-refresh-probes.log` (local, untracked).

**Notes (not part of the PR).**
- The full real-file direct load was not rerun on this branch; the only real-file proof is the October 4 integrated Qwen session, which ran integrated source, not this branch.
- Load-time registration runs the memory manager's own `free_memory` for the model's size and may evict other resident models before sampling, as checkpoint direct-to-GPU already does. Not measured in a multi-model workflow.
- The `safetensors>=0.8.0` floor applies to every user for an opt-in feature; upstream is at `>=0.4.2`. A runtime check with a clear error is the alternative if maintainers object.
- Whether a platform-specific boolean belongs on the core `UNETLoader` is untested with maintainers.
- The admission margin (20% plus 4 GiB) is untuned.
- Earlier text below this point describes the pre-refresh branch and is kept for history.

The separated branch passed **22 CPU checks**, with two MPS loader cases skipped. These cover reader ownership, format and independent memory-budget refusals, node compatibility and a real thin-Wan constructor/clone/reconstruction path whose device-read boundary is simulated. The earlier coordinated MPS run passed four lower-level loader cases at `b0680dea`; it did not test this new node option. Implementation source `de4ff712` was cherry-picked unchanged as `53b4ec3d`; test-only commit `1f77853d` makes three assertions tolerate the absence of the separate mapped-offload feature while still rejecting enabled retention. Source review completed with the accepted conservative driver-refusal limitation.

The earlier utility implementation is `f077b3b9`, with portable regression packaging `cb7f20e7`. Commit `b0680dea` declares `safetensors>=0.8.0`, matching the tested pread API. This is the verified runtime floor, not a claim about the first version to support pread. A subsequent run exercised the actual production UNETLoader option without a utility hook: a 512px Qwen Viggle portrait completed in 33.358 seconds with finite output, all 195 patched layers reached and all 192 quantized side paths applied. It replayed the same cached CPU conditioning; it did not include fresh text encoding. Peak process-group RSS was 9.860 GiB, driver allocation 8.97 GiB and swap stayed at 1.95074 GiB. The native server exposed the optional input through `/object_info`, but no complete Qwen server generation was established by that schema check. Evidence: `logs/qwen-production-direct-node-proof.log` in the integration repository. This run used the integrated primary checkout, not the separated review branch. A later integrated Qwen API session completes three sequential 512×512 requests with initial/changed-seed/changed-prompt outputs independently inspected. Each request uses fresh CPU encoding, native direct INT8 loading, 25 Euler/simple steps, CFG 1 and full-VAE decoding, with enhancer off and no LoRA. The unchanged launcher uses paired 0.5/0.4 watermarks and `--cache-ram 18`; the complete session takes 332.100 seconds with normal pressure and flat swap. All three requests re-encode and sample, so no resident warm-model speedup follows. This proves that explicit bounded profile, not the separated branch in isolation, larger defaults or the ordinary diffusion loader.

## VAE recovery

Branch `refresh/mps-oom-recovery` in `vendor/ComfyUI-upstream`, head `eb6acbd0`, one commit on upstream `08ff3c11` (October 9). Supersedes `upstream/mps-oom-recovery` (`e07e1dcc`). Independent.

**Title:** Recognize MPS allocator out-of-memory errors in `is_oom`

On Torch versions that have `torch.AcceleratorError`, `is_oom` checks the "out of memory" text only on that type. The MPS allocator still raises a plain `RuntimeError` ("MPS backend out of memory …"; confirmed on Torch 2.14.1, October 9). So on Apple GPUs every OOM fallback treats it as an ordinary error: tiled VAE decode/encode, attention and upscale tiling, SeedVR, mesh post-processing, and the executor's unload-all-models response. On older Torch, where `ACCELERATOR_ERROR` falls back to `RuntimeError`, the substring check already matched, so this restores earlier behavior.

`is_oom` now also accepts a `RuntimeError` whose message starts with the MPS allocator prefix. Typed CUDA and PyTorch OOM handling is unchanged. A message that merely contains the prefix, or another exception type with the same text, still propagates.

Ten CPU tests drive the real VAE decode path with a failing backend boundary. An MPS OOM on the whole 320-wide latent recovers through four smaller 3D tiles with exact output; a failure during the tiled retry propagates without a loop; unrelated errors propagate untouched. Three of them fail on unpatched upstream. The test builds a `VAE` without a checkpoint, so it is coupled to `VAE` attributes; maintainers may prefer only the `is_oom` cases. Full `tests-unit/comfy_test`: 201 passed, 14 skipped.

Review (October 9): an independent review of the refreshed commit found nothing blocking. Its findings (CPU-only import order, a recovery test that never actually split into tiles) are fixed. Real hardware OOM recovery inside a full workflow was not rerun.

## Encoder ownership

Branch `refresh/mps-encoder-ownership` in `vendor/ComfyUI-upstream`, head `702eaec4`: the two earlier commits cherry-picked onto upstream `08ff3c11` (October 9) plus one fix commit. Supersedes `upstream/mps-encoder-ownership` (`a135aafc`). Independent. Upstream's only change to `sd1_clip.py` since the old base (`get_dynamic_vram__units`) is unrelated; the cherry-pick was clean.

**Title:** Transfer owned Wan and Krea text-encoder tensors without an extra model-sized copy

Owned Wan encoder state can transfer into the destination module directly when the loader's ownership contract permits it. The loading path retains copy behavior where source storage is not owned and keeps the source dictionary's lifetime explicit. This reduces avoidable encoder-loading overlap without treating borrowed checkpoint storage as disposable.

The Krea follow-up extends exact CPU storage assignment only to ordinary bf16 Qwen3VL-4B files loaded for Krea. Borrowed state still copies, other formats and placement retain their prior behavior, and dynamic assignment keeps its existing contract. Two conditions were added on refresh: an owned load copies when any two target tensors share bytes (PyTorch checkpoints preserve such aliases, and assigning them would couple parameters that the copying loader keeps independent), and ownership permission is scoped to one load (every load sets the module flag explicitly and restores it afterwards, and the constructor's grant is consumed by the call it covers, so a borrowed load entered through a shared clone or by re-entering the same CLIP never inherits it). Source commits: `90aff867`, `607b2ccd`.

**Refresh evidence (October 9, M1 Max, real MPS host).** Upstream places text encoders on CPU on MPS by default: `vram_state=SHARED`, `text_encoder_device()` is CPU, and the initial device is CPU for both the 11.4 GiB fp16 umt5-xxl and the 8.9 GiB bf16 Qwen3VL-4B. So the owned path applies to the default placement, not only to CLIPLoader's `device=cpu`. On the refreshed branch the real files assign every transformer tensor to the loader's storage: Wan 242/242 and Krea 713/713, each loading in 0.4 s at 0.52 GiB peak RSS. On upstream master the real Krea file shares 0/713, takes 38.0 s, and peaks at 9.66 GiB RSS, which is the anonymous copy of the whole encoder. These are storage-identity counts and process RSS at load time, not allocator counters; the branch's pages are file-backed and fault in when the encoder runs, and steady-state RSS after a real encode was not measured. With `--disable-mmap` the branch still assigns 242/242 (the loader's own copies), at 56.0 s and 6.39 GiB; the saving there is one copy, not the mapping. After the CLIP and its clone are dropped, the parameters are collected. On the thin fixtures, 11 of the 15 branch tests fail on unpatched master. No upstream PR or issue in the last three months covers ordinary encoder ownership; upstream's own assignment path is dynamic-VRAM only.

**Review (October 9).** Independent review of the rebased branch: one blocking finding (aliased keys) and one should-fix (nested borrowed loads inherit permission), both fixed in `702eaec4` with tests that failed first; the re-review caught that the constructor's instance flag still leaked into a same-instance nested load, which is also fixed and tested. Two further should-fix items were not changed: the two "dynamic" tests only exercise the pre-existing `can_assign_sd` path (kept as contract guards), and `--disable-mmap` and final-owner release had no coverage (added). Full `tests-unit/comfy_test` on CPU: 218 passed, 14 skipped. Evidence: `logs/2026-10-09-mps-ownership-refresh-probes.log` (local, untracked).

**Notes (not part of the PR).**
- The memory claim is "no anonymous model-sized copy, and a 38 s load becomes 0.4 s." Steady-state RSS after encoding is unmeasured.
- Assigning mmap-backed (private, copy-on-write) storage as parameters is something upstream does only under dynamic VRAM; maintainers may want the non-dynamic path to stay copying.
- Release is proven at the Parameter level; the mapping's own release is not separately shown.
- The Wan gate applies to any WAN-type file in MPS mode; in practice only fp16 files assign, through `owned_state_dict_dtypes`.

## Encoder cache ownership

Branch `refresh/mps-cache-ownership` in `vendor/ComfyUI-upstream`, head `bc58dad1`: the earlier commit cherry-picked onto upstream `08ff3c11` (October 9) plus one fix commit. Supersedes `upstream/mps-cache-ownership` (`274356f1`). Independent of the MPS backend branches, and not MPS-specific: it changes cache sizing and the pressure recheck on every platform, and the PR text should say so.

**Title:** Release unused text-encoder cache ownership under RAM pressure

Ordinary CLIP outputs previously appeared too small for the existing large-entry cache pass. They now expose their physical parameter and buffer storage through the cache's existing sizing hook, including packed quantized storage without dequantization and, since the refresh, the original weights a patcher keeps in its backup while a LoRA is applied. Dynamic patchers keep their previous accounting. Pending graph consumers and shared CLIP clones still own their weights normally.

**Refresh evidence (October 9, CPU).** On upstream `08ff3c11` a 4 MiB CLIP entry sizes as 0 bytes, so the large-entry pass never releases it; on the branch it sizes as 4,194,304 bytes. On upstream, evicting the cache in the middle of a warm prompt leaves the CLIP weight alive until the prompt ends, because the executor's gather kept every cached entry; on the branch it is freed. Five of the six original branch tests fail on unpatched master. The pressure-recheck test passes on master only because the sizing hook is absent there; it guards the recheck only together with the hook. Upstream's `execution.py` changed since the old base (media enrichment, `NoAssets`, validation paths) in unrelated regions. Issue #16885 (open) is about the same RAM-pressure cache but a different defect (loop children evicted between prompts); no equivalent fix was found.

**Review (October 9).** Independent review of the rebased branch: nothing blocking. One should-fix, patcher backups not counted (a patched encoder owns twice its parameter bytes and could fall under the 512 MiB threshold), is fixed in `bc58dad1` with a test that failed first; the re-review found nothing. A second should-fix, that the fixture bypasses production CLIP loading and the dynamic test is trivial, was not changed. Full `tests-unit/comfy_test` on CPU: 198 passed, 14 skipped. Evidence: `logs/2026-10-09-mps-ownership-refresh-probes.log` (local, untracked). The "92 CPU tests" in the cherry-picked commit message refers to an integrated run and should not be quoted.

**Notes (not part of the PR).**
- The fresh RAM reading after eviction can be pessimistic when freed memory has not yet returned to the OS, which would free pins more eagerly than before. Not measured.
- Real pin release and persistent-server pressure acceptance remain unproven; the executor tests mock `free_pins` and the memory reading.
- The cache tests build a `CLIP` around `nn.Linear` with `assign=True`, not through production loading.

Warm-prompt cache lookups collect only cached node IDs, avoiding an unused request-long snapshot of model outputs. After evicting large entries, the executor re-measures available RAM before deciding whether to release pins: evicted entries can still share live storage, so estimated bytes are not proof of reclaimed RAM. Headroom, thresholds, async provider calls and graph-link behavior stay unchanged.

Start review in `execution.py` for the two ownership decisions, then `CLIP._comfy_cache_tensors` and the six real lifecycle tests. The separated branch passed 53 CPU tests covering those cases plus cache providers, execution lists and reentry. The patch is identical to the independently reviewed source commit `c75c2013`; it does not depend on encoder-loading, mapped-offload or MPS Kitchen branches. It uses the existing Kitchen tensor-layout API and introduces no package requirement change. Full server memory acceptance remains separate.

## Mapped offload candidate

Branch `upstream/mps-mapped-offload`, head `b0d3ac5b`, based on adapter-types with direct-load merged as an explicit second dependency. **Candidate: generation-memory acceptance is still pending.**

**Title:** Restore owned mapped quantized weights when offloading MPS diffusion models

The diffusion loader marks eligible mapped checkpoint storage. Ordinary file loads leave this experimental retention policy disabled; callers must explicitly set `model_options["mmap_offload"] = True` to enable it. Before an unpatched quantized layer moves from CPU to MPS, the patcher retains its reclaimable CPU mapping; offload restores that mapping instead of allocating an anonymous CPU copy of the device weight. LoRA side factors use the same ownership path, clones retain the policy, and direct-device loading does not masquerade as CPU mmap storage.

The policy is limited to the diffusion-loader opt-in, static patchers, CPU offload and genuinely mapped quantized weights. Training, dynamic patchers, borrowed/non-mapped storage and incompatible paths retain the previous behavior. The CPU contract suite covers actual storage annotations, clone and reload behavior, disabled mmap, patching and exclusion cases. Source commits: `b470a560` and the reviewed default-off follow-up `f290a26f`. The opt-in survives clones and cached reconstruction.

Passing CPU lifetime tests does not establish a safe long-running server memory envelope. Keep this branch a candidate until the coordinated generation workload verifies pressure and offload behavior. No claim is made that this fixes the training pilot pressure stops; the parent investigation observed pressure stops with mapped retention disabled as well.
