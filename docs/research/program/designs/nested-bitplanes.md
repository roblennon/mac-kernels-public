# Nested bit-planes as precision on demand (merged cluster)

Status: parked, 2026-11-09. A hostile review found the headline experiment gives no benefit on this hardware and is largely published. The one open part needs a different design and a prior-art search first. The search gate is still open: on 2026-11-09 every search query was refused again because the shared budget ran out. The record is at the end.

Merges "Precision as the thing that gets paged", "Bit-plane streaming", "Nested-bit extrapolation" and the bit-truncation guidance idea. "Staircase bit-planes" and "Plane-on-demand MLP reads" need their own cards; they are not designed here.

## The original question

Can one 8-bit weight file for a diffusion transformer serve as several models, read to 4, 5, 6 or 8 of its top bit-planes, with the sampler choosing the planes per step? Later, could the OS drop the low planes under memory pressure instead of swapping?

## Why it is parked

The per-step schedule changes nothing a Mac user can feel. The card itself predicted near-zero speed gain at 512 px, because these GEMMs are not limited by weight reads (not checked). A 4-then-8 schedule still needs all 8 planes resident for the late steps, so it saves no memory either. A yes would show only that early steps tolerate coarse weights. TMPDiff already reports per-step precision schedules on FLUX.1-dev and Sana. Nesting int4 inside the top bits of int8 is Matryoshka Quantization's idea, and MatGPTQ and Any-Precision LLM have the format and kernels for LLMs.

The weight-error table would not test anything: truncated uniform codes are a coarser uniform grid, so the predicted curve is arithmetic.

The image step does not fit the lab's limits as written. The repo records Qwen-Image 2.1 INT8 at 9.1 GB for 1024 px inference (docs/RESULTS.md) and a 9.44 to 9.46 GiB driver peak for 512 px training (docs/findings.md). The INT8 file alone is 7.26 GB. The 8 GiB process limit is very likely broken before VAE, activations, the scratch buffer and LPIPS are counted.

What is left open is one systems idea: keep the low bit-planes as clean, file-backed pages on unified memory, so the OS can drop them under pressure and the runtime falls back to fewer planes instead of stalling. No paper read so far touches it. But a novelty search has not run, the first slice that tests it is a paging probe rather than a research result, and the fallback precision would probably cost too much image quality without MatQuant-style joint training. Each of those three is a reason to park, not drop. The search gate is still open: a second attempt on 2026-11-09 had all five queries refused (see "Search gate").

## What would unpark it

All three, in order.

1. A real prior-art search, by a person or an agent with search budget. Queries to record with dates: diffusion plus Matryoshka or nested quantization (2025 to 2026); timestep-aware or timestep-dependent bit-width for diffusion (a line of work since about 2023, not checked this pass); any-precision diffusion; MSB truncation or precision fallback under memory pressure; mmap or page-cache eviction of quantized weights on Apple silicon or other unified memory. If anything already does residency-driven precision fallback, drop the card.

2. The residency probe below comes back with evicted pages that the GPU can skip.

3. A fallback precision that holds image quality. Static 4-plane truncation was predicted to fail (R above 0.5, my guess). A 6+2 split would fall back to about 3.5% weight error at a 25% byte saving (extrapolated from the table, not measured). If neither split gives a fallback worth having without joint training, the card stays parked behind a MatQuant-style master, which needs calibration on a 7B DiT and probably cloud time.

## Design for the unparked version

### Step 1: residency probe (fits one slice)

Prediction, written now: a no-copy Metal buffer over a mapped file stays clean and file-backed while idle, and the OS drops its pages under a ballast process. Reading an evicted region from a compute kernel faults it back in from SSD and stalls; it does not fail or return stale data. I do not know whether Metal wires the pages while a command buffer holds them. That is the main thing the probe decides.

1b-i. Map a 1 GiB file page-aligned. Wrap it with `newBufferWithBytesNoCopy`. Read it from a compute kernel. Check with `vmmap` that the region stays file-backed and clean.

1b-ii. Start a ballast process that allocates and touches memory until free memory is low. Call `mincore` on the mapped region. Record the resident fraction before and after.

1b-iii. Time one GPU read of a fully resident region, then of a region `mincore` reports as absent. Repeat each five times, arms interleaved, with the ballast held steady. The difference is the cost of one fault-back, which is what a residency-aware policy avoids.

Self-comparison: two resident reads must time within the resident spread. Planted effect: evict a known region with `madvise(MADV_DONTNEED)` on a clean mapping (or `msync` plus `MADV_FREE`, whichever works on this macOS; not checked), and `mincore` must report exactly that region absent.

Budget: one process per arm, each well under 300 s. Under 5 GPU-min in all (not checked). 1 GiB of disk, deleted after. The ballast must stay inside the 8 GiB process limit, so it may take several ballast processes; record each one's peak.

What a no means: if Metal wires the pages, or `mincore` cannot see the eviction, the systems idea is dead on this platform and the card is dropped.

### Step 2: residency-aware fallback (several slices, needs Rob's approval)

Baseline: a mapped, clean INT8 file (it is already evictable; it does not swap). A plane-separated file with the high planes in one region and the low planes in another. Before each step, `mincore` the low-plane region of each layer. Where low pages are absent, the kernel reads only the high planes for that layer. Compare wall time and image quality per image against the mapped INT8 baseline and the W4A8 file under the same ballast. The model must fit the 8 GiB process limit (see below).

### Model that fits

Qwen-Image 2.1 does not fit 8 GiB. Wan 2.1 T2V 1.3B appears in the repo's weight-error results (306 layers, 1.42B parameters), so its INT8 file would be under 2 GB. Whether it is on the model drive, and whether a one-frame Wan run is a fair image model, are not checked. Krea 2 Turbo INT8 (13.5 GB) does not fit. A smaller DiT is the default for any image arm; Qwen comes back only with mmap'd block streaming, which does not exist yet.

### Image arms, if the fallback question is reached

These fix the statistics the review attacked.

- 16 prompts by 1 seed, because prompts carry the variance. Held-out prompts and more seeds come in a later slice before anything is called a result.
- One decisive comparison, committed in advance: the fallback arm against the full 8-plane arm, with R (mean same-seed LPIPS against the 8-plane image, divided by mean LPIPS between two seeds of the 8-plane model). Threshold: R below 0.3 counts as a usable fallback. R of 0.3 or above, or an interval that crosses 0.3, counts as no usable fallback. There is no mixed reading.
- Planted effect on R: a fake arm in which half the pairs use the other seed's 8-plane image and half use the same image. Its R is close to 0.5 by construction. The pipeline must read 0.5 inside its interval.
- Self-comparison: the masking wrapper at 8 planes goes through the scratch-buffer path with an identity mask, not a special case, and must give bit-identical pixels.
- Masking: codes q become (q AND NOT(2^d − 1)) + offset, with d = 8 − planes, offset = 2^(d−1) for d ≥ 1 and offset = 0 for d = 0.
- Interleaving matters for the timings only. The image metric is deterministic.
- Record the checkpoint file hash and source before running. Confirm captured text conditioning exists for the prompts, or budget the text-encoder pass in its own process.
- LPIPS AlexNet weights: size not checked; write it down against the 15 GB disk floor before download.

Per process: count load time plus images, and keep it under 300 s. Write the number of processes and total GPU minutes into the slice plan before running.

### Weight table, demoted to an instrument check

The table at 3 to 8 planes, with and without offset, only confirms the analytic curve and the code. One arm in it is not known in advance: direct p-bit rounding of the bf16 weights with the same row scale. The gap between direct rounding and truncated codes is the real cost of nesting. Prediction: near zero, double-rounding error only. If that gap is large, nesting itself is the problem and the card should point at joint training.

Quality per bit is stated in advance: nested INT8 loses to W4A8 (4.51 bits, 7.307% median, from modal/results/2026-11-09-weight-error.json). A fair quality-per-bit control would be a nested format with W4A8's group-16 scales. That is not part of this card.

## What is already known

No search query ran this pass, for this card's author or its reviewer; the shared web-search budget was used up both times. Only direct fetches of known arXiv IDs were made.

- [Matryoshka Quantization](https://arxiv.org/abs/2502.06786) (Nair, Datta, Dean, Jain, Kusupati), v1 2025-02-10, v3 2025-03-03, fetched by the reviewer. Int4 and int2 nested in the most significant bits of int8; one model trained to serve several precisions. The origin of nested MSB slicing.
- [TMPDiff](https://arxiv.org/html/2603.14062), 2026-03-14, read in full. Per-step precision schedules (16-bit and W4A4) on FLUX.1-dev, Sana and others, using two stored weight sets. Names nested quantization and kernel-level switching as future work.
- [MatGPTQ](https://arxiv.org/html/2602.03537v2), 2026-02-03 (v2 2026-09-30), read except the last 2.5k characters. Nested 8-bit master, top-r-bit reads, recentering offset, bit-plane buffers, CUDA kernels. LLMs only.
- [Any-Precision LLM](https://arxiv.org/html/2402.10517), ICML 2024. Bit-plane overlay with bit-plane kernels.
- [PagedWeight](https://arxiv.org/html/2607.16184), 2026-07-17, read in full. Lowers and raises bit-planes per expert under KV-cache pressure in GPU serving of MoE LLMs. The closest work to the open idea; it does not use OS pages or unified memory.
- [DP-LLM](https://arxiv.org/abs/2508.06041), 2025-08-08, abstract only. Per-layer, per-step precision on an overlaid model.
- [PMPD](https://arxiv.org/abs/2410.13461), 2024-10-17, abstract only. Precision lowered during LLM decoding.
- [LLM in a flash](https://arxiv.org/abs/2312.11514), 2023-12-12, abstract only. Streams weights from flash at a fixed precision.
- [TDQ](https://arxiv.org/abs/2306.02316), 2023-06-04, abstract only. Uses per-timestep quantization intervals. The bit-width does not change.
- [Autoguidance](https://arxiv.org/abs/2406.02507), 2024-06. One line says a lower-bit copy did not make a useful guide. The card's guidance and extrapolation side arms are parked with it.

## Claims about larger models

Only the residency result is at stake, and it is a property of macOS, Metal and one chip. The truncation error curve is true by construction and carries no evidence. Any early-step tolerance result would be a property of one checkpoint and one sampler and would not carry without retesting.

## Search gate, 2026-11-09

Queries run: 0. Refused: 5 of 5. The shared budget ran out again on the first call. Refused: Matryoshka or nested quantization for diffusion transformers; timestep-aware bit-width for diffusion; any-precision diffusion bit-planes; precision fallback under memory pressure; mmap or page-cache eviction of quantized weights on Apple silicon. My ten extra queries were not tried. Routing searches through arXiv, Semantic Scholar or GitHub search via fetch would sidestep the limit, so I did not.

I fetched two known papers directly, abstract pages only: [LLM in a flash](https://arxiv.org/abs/2312.11514) (2023-12-12, v3 2024-07-30) and [TDQ](https://arxiv.org/abs/2306.02316) (2023-06-04). Both are now listed under "What is already known". LLM in a flash is the nearest Apple-side prior art for weights that are not kept in memory. It streams weights at a fixed precision, and its abstract does not mention OS pages.

Still open: all five owed searches and the ten extra queries. Neither fetch changes the picture or the design. Unpark condition 1 is not met.

## Review record

Objections from the hostile review, and what was done.

- Fatal, the schedule gives no user benefit and is largely published (TMPDiff): agreed. The schedule is dropped as a headline; the card is parked; the open part is reframed as residency-driven fallback.
- Must-fix, the weight table is arithmetic: agreed. Relabelled as an instrument check; a direct p-bit rounding arm added to measure the real cost of nesting.
- Must-fix, the W4A8 control cannot embarrass anything: agreed. The loss on quality per bit is stated in advance; the fair control (nested with group-16 scales) is named and left out of scope.
- Must-fix, the static 6-plane control has no counterpart in the 4+4 build: agreed. Removed; the decisive comparison is now fallback against full 8 planes.
- Must-fix, step 1c breaks the 8 GiB and 300 s limits: agreed, given the 9.1 GB and 9.44 to 9.46 GiB peaks in the repo. Qwen image arms removed from step 1; a smaller DiT (Wan 2.1 1.3B, availability not checked) is the default; per-process budgeting is required in the slice plan.
- Must-fix, step 3 baseline and residency design are wrong: agreed. Baseline is now a mapped clean INT8 file; `mincore` residency check and a fallback policy added; the probe is extended to measure the cost of reading an evicted region.
- Must-fix, statistics cannot meet the decision rule: agreed. 16 prompts by 1 seed, one decisive comparison, one threshold, and an explicit reading for every outcome.
- Should-fix, analysis-level planted effect is plumbing: agreed. Replaced with a planted effect on R (half the pairs swapped, R near 0.5).
- Should-fix, prior art thin, MatQuant missing: agreed. MatQuant cited as the origin; a real search is the first unpark condition. My own two searches also did not run (budget used up).
- Should-fix, masking formula wrong at d = 0, self-check trivially passable: agreed. Offset is 0 at d = 0; the self-check goes through the scratch-buffer path.
- Should-fix, Turbo checkpoint provenance and captured conditioning not checked: agreed. Hash, source and conditioning are preconditions in the design.
- Minor, interleaving does nothing for a deterministic metric: agreed. Kept for timings only.
- Minor, LPIPS weight size not checked: still not checked; it must be written down before download.
- Minor, the larger-model claim leaned on a rule true by construction: agreed. Rewritten to say only the residency result is at stake.
- Reviewer verdict "park": agreed.

Reviewer's scooping search: none of the six planned searches ran because the budget was used up. Besides the topics listed in unpark condition 1, they covered progressive-precision sampling on FLUX in 2026 and switchable-precision image generators. The reviewer fetched MatQuant directly. Two things look open but have not been checked: a nested DiT file with an in-kernel plane schedule, and low planes kept as clean file-backed pages on unified memory.
