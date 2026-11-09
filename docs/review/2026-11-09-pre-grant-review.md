# Pre-grant review: mac-kernels-public

Reviewed 2026-11-09. Inputs were 9 reviewers, 8 web-research lanes, a two-vote adversarial check on every finding, and a Modal scaffold. All line numbers refer to the public checkout.

## 1. Verdict

1. The kernel-level evidence holds up. The range-safety design, the bit-exact GGUF decode and the oracle-driven tests all survived adversarial checking. The findings log is unusually honest.
2. The README headline table is the weak point. Two rows rest on a noisy or stale number, one row has no public evidence, and one sentence overclaims what was verified.
3. A reviewer cannot reproduce anything today. The pinned commits in CONTRIBUTING.md:33-36 exist only on private remotes.
4. The repo has no research question, contributions list, related-work section, limitations section or roadmap. A grant panel will look for all five.
5. Before sharing: fix the README table and its wording, push the pinned branches to the public forks, port the EXL3 evidence, refresh the upstream status, and add the five framing sections.

## 2. Must-fix before sharing

**M1. Reproducibility (blocker).** The SHAs pinned at CONTRIBUTING.md:33-35 do not exist on the public forks, and GitHub returns "No commit found" for each. Those commits are ComfyUI e11d922, comfy-kitchen 9825e98 and ck-gguf da65b20, on branches mps-int8-weight-only, mps-weight-only-backend and mps-gguf. They live only on roblennon/mac-kernels-comfyui and roblennon/mac-kernels-comfy-kitchen. The vendor HEADs have also moved past the pins (0416252, b5dd528). The GGUF decode kernel is in the comfy-kitchen mps-gguf branch, not in the node fork. `bench_gguf_linear.py --no-stock` avoids the node fork entirely, so publishing mps-gguf is what unblocks the bit-exact GGUF claim. **Fix:**
- Push the three integration branches to the public forks.
- Update the pins.
- Publish the GGUF node fork.
- Add docs/REPRODUCING.md with a model-free kernel tier (bench_int8_linear.py, bench_gguf_linear.py --no-stock) and a full-model tier. bench_int8_backward.py needs the Krea checkpoint, so it belongs in the full-model tier.
- Add a root dependency lock.

**M2. EXL3 row has no public evidence (blocker).** README.md:19 reads "27B ... 2.4-4.4x slower". The ratio is real: it re-derives to 2.42-4.43x against Nano GGUF. But it lives only in the private file mac-kernels/docs/reviews/2026-10-25-exl3-prototype.md and its .json, where the model is orcarouter/OrcaSAQ-2-27B rev 443cd64a. The public repo has only the GGUF side (bench/gguf_op_probe.md:75-95). The scope note is at bench/exl3_probe.py:241, not :162. **Fix:** port the review and its JSON into docs/research and link the row to it. Add the caveats: complete-call wall time, FP16 vs F32 I/O, and different model lineages.

**M3. Krea 1024 row (README.md:16).** There are four problems:
- The stock "35 s" is a single run taken during a download. The log flags it for a rerun (findings.md:36, :45). Its step times are 29.3-49.9 s, median 36.9 s, and the text-encoder load took 277 s against 37 s in the other runs.
- The 16.5 s is an older number. Later runs measure 14.91 s (findings.md:331) and 15.02-15.06 s (:295).
- The row mixes the kernel with SDPA attention, and stock+SDPA was never measured. Kernel alone is about 24.5 s/step.
- The "21.7 -> 15.0 GB" figures are driver_allocated_memory snapshots taken after sampling, not peaks (bench/run_krea2.py:164, :172).

**Fix:** lead with Krea 512², 8.7 -> 3.7 s/step (2.35x, findings.md:36, :83, :106). That pair has no noise caveat and is the headline of PR #249. Add Qwen-Image 2.1 1024², 15.1 -> 8.8 s/step at 9.1 GB (findings.md:95-96). Label memory as "after sampling". Drop or caveat the 1024 row until the four-way rerun (E1) exists.

**M4. INT8 overhead is stale (README.md:15).** "4-9%" comes from the table before hardening (findings.md:14-22). Later tables give 3.3-4.5% (findings.md:76-81) and 1.4-2.1% (kitchen-pr-mps-backend.md:38-41). **Fix:** "within about 2-5% of a resident fp16 GEMM at half the memory." The "about 2x faster than dense fp16" at K=16384 holds only against F.linear's slow transposed path. Against a bare GEMM the kernel is still about 4% slower. Make the same correction in kitchen-pr-mps-backend.md:74 and comfy-kitchen-mps-backend.md:110, :128, which are live in PR #249. The dequantize-per-call ratio should read 1.2-2.0x.

**M5. Verification overclaim (README.md:21).** "Every number is checked against an independent float32 or float64 reference" is false for timings, memory and end-to-end images ("Images match visually; no NaNs", findings.md:32). **Fix:** "Every kernel output is checked against a float32/float64 reference; end-to-end runs are checked for NaNs and compared to stock by PSNR/SSIM where noted; timings are single-machine measurements."

**M6. PyTorch figure (README.md:31).** 8.7x is the fp32 ceiling. In fp16 the ratio is 3.6x at K=32768 (pytorch-issue-mps-wide-linear.md:53, :57). **Fix:** give both dtypes and the date ("torch 2.14.1 / nightly 2.16, Oct 9"). The issue is filed as #200782, so update pytorch-mps-wide-linear-issue.md:3, which still says "Not filed".

**M7. GGUF docs are stale.**
- README.md:18 and gguf-on-metal.md:7, :56, :63, :99-100, :127 say ten types with IQ uncovered. The kernel and tests now include IQ4_NL and IQ4_XS (gguf.py:46-47; test_mps_gguf.py:28). Run those bit-exact tests before writing "twelve".
- The "Measured" section (gguf-on-metal.md:121-123) is an empty placeholder. Fill it from findings.md:546 (Qwen Q4_K_M: 39.6 vs 66.4 s, 1.68x) and :560 (Wan 5B: 145.9 vs 200.7 s, 1.38x).
- Section 5 (:104, :110, :116) still describes per-layer cached LoRA factors. The fork now builds factors on each call in FP32 (commits 2f71d58, bef90ac) and sends explicit dequant_dtype/patch_dtype settings to the stock path (7156e0a). The node-test count is 25, not 17.
- Do not attribute the 27 dB Metal-vs-stock gap to fp16 scales alone. The note at :71 says the bf16 cast dominates, and the LoRA path also differs between the two runs.

**M8. "MPS has no _int_mm" is out of date.** PyTorch 2.15.0 (released 2026-10-28) ships a native MPS `_int_mm` ([#193153](https://github.com/pytorch/pytorch/pull/193153), [v2.15.0 notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0)). On M1-M4 with macOS 15 it is a generic tiled kernel, not a matrix-unit path. **Fix:** qualify findings.md:24, comfy-kitchen-mps-backend.md:39, :126, comfyui-mps-kitchen-formats.md:11 and survey.md:31 as "torch <= 2.14". ComfyUI [#16451](https://github.com/Comfy-Org/ComfyUI/pull/16451) would turn this path on for torch >= 2.15, which changes the baseline that PR (e) argues against. See experiment E2.

**M9. Upstream status is stale (README.md:23-33).** #249 now conflicts with main. A competing lane exists (#270, #317). LXD-8's coordination question on #249 (Oct 22) is unanswered. See section 6. **Fix:** date the paragraph, use the replacement wording in section 6, and reply to LXD-8.

**M10. Grant framing is missing.** None of these exist yet:
- A research question.
- A numbered contributions list. The training result is absent from the results table.
- A related-work section. The survey notes speak to the author in second person.
- A consolidated limitations section.
- A roadmap with aims, hardware budget and success criteria.

Seed material: survey.md:49-55 (Top 5 with effort estimates), lora-training note :77-79 (milestones), and that note's :67 (cross-precision LoRA transfer is an open question). State plainly that Modal cannot test Metal, so the budget needs Apple hardware.

## 3. Should-fix

- **Variance.** Most numbers are one median from one process, with n=3-10 (bench_int8_linear.py:14-19 and similar). The dense fp16 baseline drifts 2-4% across tables, which is the same size as the overhead claim. Reuse the interleaved-rounds pattern already in repro_mps_wide_linear.py. Run 3+ fresh processes, report median/IQR plus a bootstrap CI, and publish the raw per-iteration JSONL (logs/ is gitignored).
- **Host state.** Record `pmset -g therm`, powermetrics, load average, swap and the top processes in every RESULT line. gpu_runner.py:178-196 tracks only memory pressure and swap.
- **External baselines.** The repo has no mflux, stable-diffusion.cpp or Draw Things run on the M1 Max, and MLX is not installed (CONTRIBUTING.md:38). See experiment E5.
- **Quality metrics.** Results use one prompt and one seed per model, with PSNR/SSIM and visual inspection. Spectral percentages are quoted to five significant figures from 2-3 runs (findings.md:869, :874, :880). Round them, and add LPIPS/CLIP over a prompt set.
- **fp16 precision wording.** The overflow proof is sound, but findings.md:58 says "Exact" and kitchen-pr-mps-backend.md:28 says "changes no mantissa ... exact at subnormals". The scaling is exact. The fp16 cast after it keeps full precision only down to about rowmax*2^-14. For bf16 inputs the practical effect is negligible. Also state that the GEMM output carries 11 bits, and add an explicit K=16384 fp16 per-element test. The existing K=1024 oracle test already rules out naive fp16 accumulation.
- **Training evidence.**
  - Milestone criterion (1), a one-block LoRA gradient against an fp32 reference, was never reported.
  - The packed W4/W6 backward is checked numerically only at n=67, K<=768. The real-Qwen probe checks only that gradients are finite and nonzero.
  - "Validation loss" is computed on training images 0 and 7 (train_lora.py:573-584). Rename it to "fixed training-set loss".
  - The "2-8% of dense backward" claim fails for attn.wk, which is 39% slower (findings.md:385).
  - The INT8-vs-bf16 speed gap is confounded by bf16 emulation on M1.
- **findings.md register.** Lines 609-842 read as agent-orchestration receipts: tool-call counts and interpreter violations at :806, :808, :823, :838. Lines 880-884 have missing spaces ("the512px", "Root422pass"). Move the process metadata out and add a curated docs/RESULTS.md.
- **Internal inconsistency.** findings.md:100 says "Apple GPUs emulate bf16 convolutions", while comfyui-mps-fp16-vae.md:39 dropped that claim as unproven. Scope it to M1.
- **Tests.** .repo-check line 1 runs test_lora_sidepath*.py (115 tests) on the GPU without bench/with_gpu.sh, while CONTRIBUTING.md:67-68 requires the lock. Gate them on RUN_MPS_TESTS. CONTRIBUTING "Run the tests" should name the vendor kernel-numerics test files and the fork commit.
- **Presentation.** Move the *.command launchers out of the root, and put evidence ahead of workbench guides in docs/README.md:5-10.
- **Minor items:** see Appendix B.

## 4. Refuted or downgraded (considered, do not act on as stated)

- E9 "nobody knows the H3/Wan14B working set; profile it on CUDA": refuted. Peaks are measured (findings.md:56, :121, :152, :470-478). Bounded loaders closed G2 for short renders (how-to-video-workbench.md:5). CUDA peaks cannot see macOS page-cache pressure.
- E11 "conv3d still at ~2 TFLOPS": refuted. That was the baseline before the fix. The shipped stacked-conv2d path runs at about 10.5 effective TFLOPS, which is at or above the GEMM ceiling.
- "PyTorch's Nov 3-4 Metal matmul kernels make the 8.7x stale": refuted. `metal_mm_out` has no caller, and Linear.mm is unchanged. Dating the figure is still worthwhile.
- "Maintainers see #270 as the active mps lane": unsupported. No maintainer has touched #249, #270 or #317.
- "#317 contradicts the weight-only rationale": softened. #317 falls back to weight-only `_weight_int8pack_mm`.
- E12 "1024 training is recompute-bound": overstated. The chunked backward is 1.7x faster than torch's path, and the step is dominated by the linears. The packed layer-wide exponent limit was scanned locally across 1,149 shipped layers: the maximum per-channel scale spread is 2^7.4, far below the fp16 2^14 cliff, so the limit is inert.
- E2 (CUDA baselines) and E4 (other Apple chips): real gaps, downgraded from blocker to major. The MPS ratios are internally anchored.
- "Workbench material dominates the README": overstated. The README leads with research. Workbench tests are about a third of the suite, not most of it.
- "test_f1 is the only accumulation hint": wrong. The K=1024 oracle test is stronger.

## 5. Positioning

**Where this sits.** In November 2026, almost every new Apple quantized-diffusion speedup targets M5 Neural Accelerators and macOS 26.2+ MPP paths:
- [ComfyUI-AppleSilicon-FP8 v1.5.0](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8/releases/tag/v1.5.0), 2026-11-06: W8A8 is M5-only, capped at 2048 tokens. Krea 2 1024² goes from 41.8 to 18.2 s on M5 Max.
- [Draw Things 8-bit S / MQA](https://releases.drawthings.ai/p/metal-quantized-attention-pulling), 2026-04-01: M5 only.
- [MLX #4802](https://github.com/ml-explore/mlx/pull/4802), 2026-10-30: about 1.2x at diffusion-sized M, NAX only.
- [h3.c](https://github.com/antirez/h3.c): int8 is M5-only, with BF16 on M3 and older.
- [h3-metal](https://github.com/zuiho-kai/h3-metal), 2026-10-28: M5 Max.

On M1-M4, independent work agrees with this repo that activation quantization does not pay:
- [FastMetal-QAD](https://haoailab.com/blogs/fastmetal/), 2026-08-10, built a W8A8 kernel and dropped it.
- [lilting.ch on an M1 Max](https://lilting.ch/en/articles/anima-int8convrot-mps-metal-mlx-m1-max), 2026-08-04, found fp16 beat every int8 path it tried. This repo's kernel beats each of those paths.
- [unsloth #13344](https://github.com/unslothai/unsloth/pull/13344), 2026-10-15, found `_weight_int8pack_mm` unreliable across chips.

**Closest competitors.**
- [Siliconed](https://github.com/gwenn-ha-dev/Siliconed), Oct 2026: a standalone Swift/Metal app that reads the same Comfy-Org ConvRot files and streams weights. It runs Krea 1024 at about 18 s/step in 2.2 GB on an M1 Pro. The repo runs 14.9-15.3 s/step at 15 GB on an M1 Max.
- [Argus minimax-h3-mac](https://github.com/Argus-AiTeam/minimax-h3-mac): runs H3 1344x768x124 at about 601 s/NFE on a 24 GB M4 Pro by streaming blocks. The repo runs 1420-1940 s/step, memory-bound.

Both show that memory headroom, not packed decode, is the next lever.

**Comparable numbers.**
- Wan 5B 1280x704: this repo 37.4 s/step at 9.1 GB on an M1 Max. FastMetal reports about 33 s/step at 9.3 GB on an M4 Max, and an independent reproduction of FastMetal measured 122 s denoise against the published 98.5 s ([FastVideo #2016](https://github.com/hao-ai-lab/FastVideo/issues/2016)).
- MLX on an M3 Max needs 56 GB for the same model ([rotecodefraktion](https://www.rotecodefraktion.de/en/blog/lokale-ki-video-wan-mac/), 2026-10-10).
- Stock ComfyUI GGUF Wan 14B on an M1 Max runs about 240 s/step at 33 frames ([lilting.ch](https://lilting.ch/en/articles/ltx2-wan22-mac-local-video-gen), 2026-05-20). This repo runs 216 s/step at 81 frames.
- No one else publishes per-step 4/6-bit H3 numbers on Apple GPUs.

**Academic anchors.**
- [ConvRot](https://arxiv.org/abs/2512.03673) (2025-12) defines the format. It evaluates W4A4/W8A8 on CUDA and argues weight-only is insufficient. This repo inverts that premise for hardware without integer units.
- [Mary & Mikaitis, SISC 2025](https://perso.lip6.fr/Theo.Mary/doc/NarrowRange.pdf) is the theory behind power-of-two range scaling. Cite it.
- [Ideogram INT8 kernel](https://arxiv.org/abs/2606.14598) (2026-06) is the CUDA counterpart to "fuse decode into the GEMM". [Ideogram GGUF](https://arxiv.org/abs/2606.12280) (2026-06) is the only paper on GGUF quality for DiTs.
- [FourTune](https://arxiv.org/abs/2607.05711) (2026-07) is the CUDA frontier for LoRA on quantized DiTs. [IntLoRA](https://arxiv.org/abs/2410.21759) is the merge-based alternative to the side path.
- [Ochiai, M3 Ultra](https://arxiv.org/abs/2605.16259) (2026) found "no speedup from quantization". This repo adds that quantization buys memory, and that the stock path is what makes it slow.
- [torchao #4695](https://github.com/pytorch/ao/issues/4695): an HF maintainer found weight-only INT8 beats ConvRot W8A8 on quality.

**Draft contribution statement.**
> Apple GPUs before M5 have large unified memory but no low-precision matrix units. Quantized diffusion checkpoints published for CUDA either fail to load on them or are dequantized on every call. This work keeps INT8-ConvRot, W4A8, W6A8 and GGUF block weights packed and resident, and decodes them in one Metal pass that feeds the stock fp16 GEMM. Exact power-of-two row normalization makes fp16 range-safe for activations that otherwise overflow (MiniMax H3, |x|≈16,000). On an M1 Max the INT8 linear runs within about 2-5% of a resident fp16 GEMM at half the memory, and 1.2-2.0x faster than per-call dequantization. GGUF decode is bit-exact against the reference for ten block types. End to end: Krea 2 Turbo 512² drops from 8.7 to 3.7 s/step, Qwen-Image 2.1 1024² from 15.1 to 8.8 s/step, and GGUF Q4_K_M samples 1.4-1.7x faster than ComfyUI-GGUF. A 6-bit H3 video model with no stock Apple path runs at 6.5 s/step. The same design gives a frozen-weight backward that saves only codes and scales, within 5-8% of dense fp16 on the large layers. LoRA training on an INT8 base then fits in 9.5 GiB instead of 15.6. An unmerged LoRA side path is 70x more accurate than requantized merging (0.04% vs 2.8%). The core changes are submitted to comfy-kitchen, ComfyUI and PyTorch. All results come from one M1 Max with 32 GB. Extending them across Apple generations and memory sizes is the first proposed aim.

Before using this statement, confirm Krea 2's parameter count. Change "ten" to "twelve" only after the IQ4 tests pass.

## 6. Upstream status (as of 2026-11-09)

| Item | State | Action |
|---|---|---|
| comfy-kitchen #249 (mps backend) | Open, **conflicting** with main. Head 43fe176 (Oct 13) is 26 behind main 7d6a22ae. Build Wheels CI awaits maintainer approval. comfyanonymous requested; no human review. | Rebase. Expect conflicts in `__init__.py`, constraints.py, w4a8_int8.py and the README matrix. Reply to LXD-8. |
| comfy-kitchen #270/#297/#275 (LXD-8) | Open, mergeable, stacked behind #298. Same registration block as #249. No maintainer review. | Coordinate. These are complementary modules. |
| comfy-kitchen #317 (TheApeMachine) | Draft, Nov 7. A separate `metal` backend: W8A8 via `_int_mm` with a W8A16 fallback, NVFP4/SVDQuant/AWQ. Unmeasured. | Comment: pre-M5 should route weight-only and M5+ to MPP. Ask what hardware it was tested on. |
| ComfyUI #16996 / #16997 / #16998 | Open, behind master, CodeRabbit-approved. All CI blocked at action_required. Last activity Oct 13. Master has not absorbed any of them. | One polite ping or rebase. |
| PyTorch #200782 | Open, triaged. kinetrix (M5) offered to profile on Oct 20; no reply yet. | Answer kinetrix. Post 2.15.0 numbers plus a K=32769 row. |
| ComfyUI #16574 (UniPC) | The PR author (chelsealong) asked for a separate test PR on Oct 13; no maintainer has commented. Branch fec872c0 exists; no PR opened. | Open it. This is the cheapest upstream win available. |
| Overlaps to log | ComfyUI #16451 (INT8 on MPS for torch >= 2.15), #17596, #15902 (H3 VAE pad), #17446 (rotate input), #16890, #16433 (Qwen VAE encode bug, likely the same pad root cause). city96 ComfyUI-GGUF has had no commit since 2026-01-12. | Add these to contribution-review-2026-10-08.md:94-99. Comment on #16433. Re-decide the GGUF destination. |

Replacement for README.md:26: "Posted October 13, 2026. As of November 9, none of the five has had a human maintainer review. comfy-kitchen #249 needs a rebase and now shares the mps lane with #270 and the draft #317. ComfyUI #16996-16998 are bot-approved, with CI awaiting maintainer approval."

## 7. Experiments to run next (ranked)

Modal status: authenticated and verified on 2026-11-09, CPU and GPU (payment method added). Scaffold and first results are in [modal/](../../modal/README.md). E4 below is done; see the findings entry for 2026-11-09.

| # | Experiment | Where | Closest script | Cost |
|---|---|---|---|---|
| E1 | Clean four-way Krea run (stock, stock+SDPA, metal, metal+SDPA) at 512 and 1024. Fresh process each, 5 repeats, randomized order. | M1 Max | bench/run_krea2.py | ~2 GPU-h |
| E2 | torch 2.15.0 `_int_mm` vs weight-only kernel vs fp16 F.linear. No public M1-M4 number exists, so the repo can be first. | M1 Max | bench/bench_ck01_int8.py | hours |
| E3 | Quality vs bitwidth against bf16. Qwen bf16 runs on the Mac. The H3 and Wan14B bf16 references need an A100-80GB. Add LPIPS/CLIP. | Split: Mac + Modal | run_h3.py, run_qwen21.py, img_similarity.py | ~$15-25 Modal |
| E4 | CUDA GEMM anchor at the same shapes (T4 then L4) | Modal | modal_app.py::cuda_gemm_bench | <$1 |
| E5 | mflux (q8, bf16) and stable-diffusion.cpp (same Qwen Q4_K_M) on matched recipes | M1 Max | new | ~3 h |
| E6 | Gradient-fidelity probe, one block then 2-4 blocks: dA/dB cosine and rel-L2 for INT8/W6/W4 against fp32-dequant and bf16 | Kernel arm on Mac, fp64 arm on Modal CPU | bench_int8_backward.py, probe_packed_training.py | 1-2 days |
| E7 | Packed W4/W6 backward at real Qwen K (3072/12288) against fp64 | M1 Max | extend bench_int8_backward.py | hours |
| E8 | Wide-K control: add a device flag and run on CUDA (expect ~1.0x). On the Mac, rerun on 2.15 plus K=32769. | Modal + Mac | bench/repro_mps_wide_linear.py (MPS hardcoded at :32, :35, :53, :91) | <$1 |
| E9 | Per-layer weight error across formats (INT8, W4A8, W6A8, Q4_K) on real checkpoints | Modal CPU | comfy-kitchen eager quantizers, wan_int8_convert.py | ~$2 |
| E10 | Perceptual metrics for the 4-step, Spectral and FastWan speedups on a prompt set | Generate on Mac, score on Modal | qwen_accel_compare.py, spectral_krea.py | 1 day |
| E11 | Block streaming for H3 at 1344x768x124. Target: beat Argus's 601 s/NFE. | M1 Max | run_h3.py, h3_memory.py | days |
| E12 | Re-verify upstream verdicts on torch 2.15 and the 2.16 nightly (attention, conv3d, Wan VAE padding) | M1 Max | attn_study.py, sdpa_mps.py, conv3d_mps.py | hours |
| E13 | EasyCache on 8-step Krea and Qwen Turbo, to close the question with data | M1 Max | run_krea2.py | 20 min |

## 8. Grant fits (top 3)

1. **[Open Intelligence Foundation](https://openintelligencefoundation.ai/)**, rolling, up to $10k in cash, compute or hardware, reviewed within days, no reporting. It has already funded MLX-for-Apple and GGUF-quantization builders. Ask for M-series hardware with more memory, or for compute credits.
2. **[Emergent Ventures](https://mercatus.tfaforms.net/5099527)**, rolling. A 1,500-word proposal that must be human-written; AI-written proposals are rejected. Rob has to write it himself.
3. **[Mozilla Human-Scale AI](https://gfmd.info/fundings/human-scale-ai-call-for-proposals/)**, **deadline 2026-11-20**, info session 2026-11-12. $300k for 12 months. Needs a US 501(c)(3) fiscal sponsor and evidence of deployment with users. This is a stretch: decide by Nov 12.

Not a fit now:
- NLnet Restack (deadline 2027-01-03) excludes generative AI under 1M users.
- NSF PESOSE (2027-03-02) is for organizations only.
- Sovereign Tech Fund has a €50k minimum and wants adopted base technology.
- Modal academic credits appear to be waitlist-only; join the waitlist anyway.

## 9. Gaps the sweep could not fill

- Whether the IQ4_NL/IQ4_XS bit-exact tests pass. They were not run.
- Whether Apple's fp16 GEMM accumulates in fp32. This is not documented anywhere and was not measured.
- No published `_int_mm` timing on M1-M4 with torch 2.15.
- No Draw Things per-step numbers for Krea 2, Qwen, Wan or H3.
- No independent M1 Max numbers for Krea 2 or Qwen.
- Siliconed's Metal ConvRot decode and the "INT4 ConvRot" format Radiant Canvas ships were not inspected.
- The Mozilla call page returned 403, so the scoring criteria are unverified.
- The status of Prime Intellect's open compute grants is unknown.
- Whether a payment card is on the Modal workspace.
- The maximum CPU and memory per Modal container.
- No step-time breakdown of attention backward inside the 36 s Qwen 1024 training step. The bench numbers exist only in private logs.

---

## Appendix A: Supporting numbers for positioning

| Project | Chip | Model / shape | s/step | Memory | Date |
|---|---|---|---|---|---|
| This repo | M1 Max 32 GB | Wan 2.2 5B 1280x704x81 INT8 | 37.4 | 9.1 GB | Oct 2026 |
| FastMetal-QAD | M4 Max 36 GB | Wan 2.2 5B 1280x704x81 INT8 g64 | ~33 (98.5 s / 3 steps) | 9.34 GiB | 2026-08-10 |
| mlx-video | M3 Max 64 GB | Wan 5B 1280x704x41 bf16 | 22.6 | 56.4 GB | 2026-10-10 |
| This repo | M1 Max | Wan 14B 832x480x81 INT8 | 216 | 21.2 GB | Oct 2026 |
| ComfyUI-GGUF (stock) | M1 Max 64 GB | Wan 14B 832x480x33 Q4_K_S | ~240 | n/a | 2026-05-20 |
| This repo | M1 Max | Qwen-Image 2.1 1024² INT8 | 8.8 | 9.1 GB | Oct 2026 |
| mflux 8-bit | M4 Max 64 GB | Qwen-Image 2.1 1024² | 6.0 | ~30 GB | 2026 |
| This repo | M1 Max | Krea 2 Turbo 1024² INT8 | ~15 | 15.0 GB (after sampling) | Oct 2026 |
| Siliconed | M1 Pro 16 GB | Krea 2 Turbo 1024² ConvRot | ~18 | 2.2 GB | Oct 2026 |
| Argus | M4 Pro 24 GB | H3 1344x768x124 bf16, streamed | 601 /NFE | 15.8 GB | 2026 |
| This repo | M1 Max | H3 1344x768x124 W4A8 | 1420-1940 | 30.1 GB, swapping | Oct 2026 |

## Appendix B: Minor fixes

- bench/exl3_probe.py:26 hardcodes the GPU lock path. Read MK_GPU_LOCK like the other scripts.
- bench/queue.sh:6-7 and bench/template_session.sh:5 redirect into logs/ without creating it. Add `mkdir -p`.
- Video Workbench.command has no .venv guard. Copy the one from the Krea launcher.
- bench/gpu_runner.py:196 reads `/private/var/vm`, which breaks off macOS.
- LICENSE:190 still has the placeholder copyright line. Add a NOTICE naming the holder and year.
- findings.md mixes two heading styles and contains 14 em-dashes.
- bench/README.md should name exl3_reference.LICENSE and say that bench_ck01_int8.py supersedes bench_int8_linear.py.
- Move the train/m2_*.sh launchers to train/historical/.
- Document the 13 Krea2Edit test skips and how to obtain that runtime.