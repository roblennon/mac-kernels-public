# Gap research for the pre-grant review (2026-11-09)

> Superseded in part on 2026-11-09 by [2026-11-09-funder-research.md](2026-11-09-funder-research.md).
> Mozilla's pages were later read directly: six evaluation criteria are published, and a fiscal
> sponsor is recommended, not required. The Mozilla statements below are kept as the record of
> what was known before that read.

This page answers the open questions in Section 9 of
[2026-11-09-pre-grant-review.md](2026-11-09-pre-grant-review.md) by web research only.

**Nothing on this page was measured by this repo.** Every figure is copied from a public source
read on 2026-11-09. The source ledger, with each URL, source date and figure, is
[2026-11-09-experiments/gap-research-sources.json](2026-11-09-experiments/gap-research-sources.json).
"Unconfirmed" means the primary source could not be read or does not say.

## What needs a person

1. **Mozilla's own pages could not be read.** They return HTTP 403 to the fetch tool. Everything
   in section (a) comes from two third-party summaries. Before the application goes in, someone
   should open the call page and the FAQ in a browser and check the fiscal sponsor wording and
   the info session link.
2. **The review says Mozilla needs a US 501(c)(3) fiscal sponsor. That is unconfirmed.** GFMD
   says a selected team must be able to receive funding from a US-based 501(c)(3), and does not
   mention fiscal sponsors. Mozilla Foundation is itself a US 501(c)(3), so the grantor may
   already satisfy that wording. The Granted blog lists "individual" and fiscal sponsorship as
   separate applicant types. Neither source settles whether an individual needs a sponsor.
3. **The review's Siliconed figure (about 18 s per step in 2.2 GB on M1 Pro) was not found in
   Siliconed's current docs.** The docs give whole-render times instead. See section (b).

## (a) Mozilla Human-Scale AI call

**Established (from third-party summaries, not from Mozilla's pages):**

- Deadline: Friday 2026-11-20, 11:59 pm PST.
- Award: 300,000 USD over 12 months starting April 2027, with up to 200,000 USD more over two
  further years, so up to 500,000 USD in total. Seven projects are funded.
- Eligibility: the applicant must be legally able to receive grant funds from Mozilla Foundation,
  and the work must have a charitable purpose. Nonprofit status is not required. Applications are
  in English. GFMD's wording is that a selected team "must be able to receive funding from a
  U.S.-based 501(c)(3)". GFMD does not mention fiscal sponsors.
- Applicant types on the form: individual, nonprofit, fiscal sponsorship, for-profit or other
  organisation, governmental unit, other.
- Stage: projects should already be working and deployed with their intended users, even as a
  small pilot. The form offers four stages (concept only, working prototype, deployed to fewer
  than 100 users, deployed to 100 or more). Concept-only is out of scope.
- The form asks for the largest parameter count of any core model, on a dropdown that runs from
  "10 billion or fewer" to more than 40 billion, and whether the project depends on a frontier
  model API. Some answers trigger an out-of-focus notice.
- Three required essays, one each on the themes Small, Intentional and Tailored.
- Process: an initial project proposal is the main review. A shortlist is invited to a full
  application.
- Scoring: no rubric or weights are published. The summaries say reviewers look for early
  product and community fit, and for meaningful technical, product or sustainability work still
  to do. Open sharing is favored but not required.
- Info session: 2026-11-12 at 5 pm UTC (12 pm EST, 9 am PST). Registration was listed as coming
  soon, with no link.
- Application form: https://mozillafoundation.tfaforms.net/f/humanscaleai

**Sources:**

- GFMD listing, https://gfmd.info/fundings/human-scale-ai-call-for-proposals/ (page date not
  stated).
- Granted blog, https://grantedai.com/blog/mozilla-human-scale-ai-incubator-500000-november-20-2026
  (2026-11-05).
- Mozilla call page and FAQ under
  https://www.mozillafoundation.org/en/what-we-do/grantmaking/incubator/human-scale-ai/
  (HTTP 403, not read). A search snippet of the FAQ says Mozilla often recommends fiscal
  sponsorship for teams that are not tax-exempt.

**Unknown or unconfirmed:**

- Whether a fiscal sponsor is required for an individual applicant. The GFMD wording about
  receiving funds from a US 501(c)(3) is ambiguous, because Mozilla Foundation is itself that
  organisation.
- The scoring rubric. None is published.
- The info session registration link, a contact address, and the list of ineligible countries.
- How the repo's stage would be judged. It has a public release but no documented user count.

## (b) Siliconed and Radiant Canvas

### Siliconed

**Established (read from source at head 2ca21fe, pushed 2026-11-08):**

- Apache-2.0, Swift and Metal, requires macOS 26. Created 2026-10-09.
- It has three ways to run a quantized linear layer, and picks one per machine by measuring:
  1. **Widened (the default on older chips).** Weights are decoded to fp32 on the CPU into a
     reserve, with the ConvRot Hadamard rotation applied to the integers first. The product then
     runs in fp32 through MPS.
  2. **Native int8.** The int8 weights are used as published. A Metal 4 `matmul2d` multiplies
     float activations by int8 weights. The ConvRot rotation moves to the activation side, and
     the per-column scale is applied in a separate pass afterwards.
  3. **W8A8.** The activation is rotated and quantized to int8 per row, the product is int8 by
     int8 into int32, then scaled. This matches ComfyUI's `int8_linear`. It is used only where
     the machine's own measurement says it is faster, which in practice means M5.
- Its own source comment (`TensorGEMM.swift`) says `matmul2d` loses on M1 Pro: 2.36 TFLOP/s
  against 3.46 for MPS. On M5 the same comment gives 6.0 TFLOP/s for bf16 by bf16 `matmul2d`
  against 2.9 for MPS fp32. The M5 pair compares two different number formats, and the comment
  does not name the formats for the M1 Pro pair.
- The arithmetic is fp32, not fp16.
- On 16 GB Macs it streams weights from disk as needed. It trades speed for low memory.
- It has its own "SQ" format: 4 or 5 bit codes in groups of 16 with one group byte, one fp32
  scale per row, Hadamard blocks of 256, at 4.5 to 5.5 bits per weight. It decodes without loss
  to int8 and then follows the int8 route. It also reads GGUF.
- Published timings are whole renders, not per step. On a Mac mini M5 Pro 64 GB at 1024 square
  with 8 evaluations (app 0.21): Krea 2 Turbo 32.0 s compact and 45.4 s standard, with peaks of
  17.1 and 32.0 GiB. Qwen-Image-2.1 Turbo 21.3 and 29.2 s.
- On an M1 Pro (app 0.14, dated 2026-10-23, Standard, 1024 square, Faster Rendering on): Krea 2
  Turbo took 135.4 to 137.4 s per render with the Accelerators switch on, and 144.6 to 144.9 s
  with it off. On M1 Pro that switch mainly means fp16 products on the Neural Engine, because
  the same doc says the M1 Pro GPU is not faster in bf16. Off means fp32 throughout.
- These are not full-size 8-evaluation renders. Faster Rendering runs the first evaluations on a
  half-size grid, 4 of 8 for Krea 2, and the doc says a half-size evaluation costs about a
  quarter of a full one. Do not divide these times by 8 and compare them with this repo's
  full-size per-step numbers.

**How it differs from this repo's kernel:**

- This repo keeps the packed weights resident on the GPU and decodes them inside one Metal pass
  that feeds the stock fp16 GEMM. Siliconed, on pre-M5 chips, decodes to fp32 on the CPU ahead
  of the product.
- Siliconed's integer paths depend on Metal 4 `matmul2d` and only win on M5. This repo's target
  is the M1 to M4 range, where that primitive runs on shader cores.
- Siliconed is a standalone Swift app and engine. This repo plugs into PyTorch and ComfyUI.
- Siliconed runs W8A8 on some machines. This repo is weight-only, with fp16 activations.

**Source:** https://github.com/gwenn-ha-dev/Siliconed (files listed in the ledger).

**Unknown or unconfirmed:**

- The review's figure of about 18 s per step in 2.2 GB on M1 Pro. The current docs do not state
  it in that form. It should be re-sourced or replaced with the render times above.
- No head-to-head run of Siliconed against this repo on the same machine exists.

### Radiant Canvas

**Established:**

- Closed-source Mac App Store app, formerly Krealize, for Apple silicon on macOS 26.2 or later.
- It ships Krea 2 Turbo as "INT4 ConvRot" (15.5 GB) and Q8 (22.2 GB).
- Its page lists timings on an M5 Max 48 GB at 1024 square: Krea 2 INT4 15.1 s fast and 20.2 s
  standard, Krea 2 Q8 20.5 and 26.8 s.
- The developer's August post describes the engine as a C++ port of the mflux Krea 2 pipeline on
  the MLX C++ API, with quantized layers calling MLX `quantized_matmul`. "Fast" mode renders at
  90 percent size and resizes, and turbo reuses a prediction to cut 8 denoiser evaluations to 6.
  That post does not mention ConvRot or INT4.
- "INT4 ConvRot" is also the name of a community checkpoint format, stored as W4A4. A Forge Neo
  extension exists to run those checkpoints with 8-bit activations instead.

**Sources:**

- Product page, https://radiantbeargames.com/radiant-canvas (no page date, app v1.30).
- Developer post,
  https://dev.to/a_sallay/i-made-a-native-krea-2-app-for-apple-silicon-macs-3d9n (2026-08-17).
- Review, https://www.promptquorum.com/power-local-llm/radiant-canvas (updated 2026-09-25).
- https://github.com/eduardoabreu81/forge-neo-int4-w4a8 (created 2026-10-27).
- ConvRot paper, https://arxiv.org/abs/2512.03673 (2025-12-03).

**Unknown or unconfirmed:**

- What "INT4 ConvRot" is inside the app. There is no code and no format note. It is unconfirmed
  whether the app runs W4A4, W4A8, or decodes to float, and whether it kept the MLX engine.
- No Radiant Canvas timing on any M1 to M4 machine was found.

**How it differs from this repo:** it is an MLX app with no published kernel, benchmarked only
on M5. Nothing public says it does a fused packed decode.

## (c) Published per-step numbers near M1 Max

**Established:**

| Model and setup | Machine | Figure | Source date |
| --- | --- | --- | --- |
| Krea 2 Turbo bf16, ComfyUI, torch 2.10.0, 1024, 8 steps | M1 Max 64 GB | about 212 s warm, 278.8 s cold, about 31 GB resident | not captured |
| Qwen-Image 2.1 bf16, diffusers on MPS, 832x1216, 40 steps | M1 Max 64 GB | 15.12 s per step, 614.71 s total, about 45.8 GB peak | 2026-09-21 |
| Qwen-Image-2512 8-bit on MLX | "M-series", chip not named | about 8.5 s per step | not captured |
| FLUX.2 Klein 4B, mflux 0.17.5, 1024, 4 steps | M1 Max 64 GB | 31.7 s total (iris.c 41.5 s) | not captured |

- The same Krea 2 article reports that fp8 fails on MPS, and that raw bf16 at 52 steps took
  47 min 2 s and produced a black image.
- Draw Things published a four-machine benchmark on 2026-10-15 covering Krea 2 Turbo, Qwen
  Image 2.1 Turbo and MiniMax H3 against mflux, mlx-serve and vpipe. The machines are M3 Ultra,
  M4 Pro, M5 Ultra and M6. The text gives ratios only: Draw Things is 1.14x to 1.97x the fastest
  competitor on M4 Pro and 1.11x to 1.43x on M3 Ultra. Absolute seconds are in chart images that
  could not be read.

**Sources:**

- https://lilting.ch/en/articles/krea-2-m1-max-comfyui
- https://lilting.ch/en/articles/qwen-image-2-1-m1-max-local
- https://huggingface.co/mlx-community/Qwen-Image-2512-8bit
- https://lilting.ch/en/articles/flux2-klein-4b-mflux-iris-m1-max
- https://releases.drawthings.ai/p/harvest-season-new-hardware-new-inference

**Unknown or unconfirmed:**

- No published Draw Things per-step number for any model on M1 or M2 was found. The Draw Things
  charts for M3 Ultra and M4 Pro need a person to read them.
- No stable-diffusion.cpp Metal per-step number for Qwen or Wan on Apple silicon was found.
- No Wan 2.x number on M1 Max was found.
- No independent INT8 or GGUF Krea 2 number on M1 Max was found. The one independent Krea 2
  figure on M1 Max is bf16 on a 64 GB machine, which does not fit in this repo's 32 GB.

## (d) Compute grants and credits

**Established:**

- **Prime Intellect:** the homepage lists no general compute grant. The old Fast Compute Grants
  page returns HTTP 404. What is visible is a bounty-style environments program, a partner
  benchmark grant, and an ICLR free-compute page that was not opened.
- **Modal startup credits:** require venture funding (any amount from a partner fund, or over
  1M USD from any fund), are for new Modal customers, and are one-time. The page does not state
  the amount.
- **Modal academic credits:** the page says up to 10,000 USD. It gives no eligibility rules and
  no application process, only a sign-up link to hear about grant programs.

**Sources:** https://www.primeintellect.ai/ , https://www.primeintellect.ai/blog/fast-compute-grants
(404), https://modal.com/startups , https://modal.com/academics (all read 2026-11-09).

**Unknown or unconfirmed:**

- Whether Prime Intellect has any open compute grant. None was found, which is not proof there
  is none.
- Modal's startup credit amount, and whether an independent, unfunded, non-academic project
  qualifies for anything. On the published rules it does not qualify for startup credits.

## (e) Modal CPU and memory limits per container

**Established:**

- Defaults are 0.125 CPU cores and 128 MiB of memory. The default soft CPU limit is 16 cores
  above the request.
- Modal enforces a maximum when a Function or Sandbox is created, and rejects larger requests
  with an `InvalidError`.
- The per-container disk write quota defaults to 512 GiB. A larger disk can be requested with
  `ephemeral_disk`, up to 3.0 TiB. The page gives no default for `ephemeral_disk` itself.

**Source:** https://modal.com/docs/guide/resources (read 2026-11-09).

**Unknown:** Modal does not publish the numeric CPU or memory maximum. It could be found at no
cost by requesting an oversized function and reading the error. That was not done, because this
lane is web-only.

## (f) torch `_int_mm` on MPS and release notes

**Established:**

- torch 2.15.0 was published on 2026-10-28. There is no 2.16 release yet.
- The 2.15.0 notes add a native MPS `torch._int_mm` (PR #193153). Other MPS items: macOS 15 is
  now the minimum, fp8 dtype support and fp8 inputs to `scaled_mm`, native `grouped_mm`,
  `torch.mps.Stream`, more ops moved to native Metal kernels, an attention output-shape fix, and
  errors raised on command buffer failure.
- PR #193153 uses a Metal 4 `matmul2d` fast path for int8 inputs and a tiled Metal kernel
  otherwise. Its only published timing is on an Apple M4 Pro with macOS 26.6: a median of
  33.140 ms at (10032, 3072, 3072) and 133.193 ms at (10032, 3072, 12288). It gives no fp16
  baseline.
- This repo's wide-K issue (PyTorch #200782) is open with one comment, last updated 2026-10-20.
- A related upstream PR to raise on 32-bit index overflow in several MPS ops (#203937) was
  closed on 2026-11-07. A September paper (arXiv 2609.22991) documents silent wrong results in
  MPS `bmm` and `matmul` at very large element counts across torch 2.4.1 to 2.14.0.

**Sources:** https://github.com/pytorch/pytorch/releases ,
https://github.com/pytorch/pytorch/pull/193153 (merged 2026-08-14),
https://github.com/pytorch/pytorch/issues/200782 ,
https://github.com/pytorch/pytorch/pull/203937 , https://arxiv.org/abs/2609.22991 .

**What this means for the upstream verdicts:**

- The published `_int_mm` timing has no float baseline, so it does not by itself say whether
  integer GEMM beats fp16 on shader cores. This repo's own M1 Max measurement
  ([2026-11-09-experiments/int-mm-2.15.json](2026-11-09-experiments/int-mm-2.15.json)) stays the
  only evidence for that verdict on M1.
- Nothing in the 2.15.0 notes addresses the wide-K case.

**Unknown or unconfirmed:**

- No published `_int_mm` MPS timing on M1, M2 or M3 was found.
- A third-party summary says 2.16 will move attention fully off MPSGraph. That is unconfirmed
  and was not checked against the PyTorch repository.

## (g) New work to cite in the positioning section (about the last 3 months)

**Established, in rough order of relevance:**

- **Siliconed** (2026-10-09 onward). The closest neighbour. Cite it as the fp32-widen and Metal 4
  approach, and note its own finding that `matmul2d` loses on M1 Pro.
- **PyTorch 2.15 native `_int_mm` and fp8 on MPS** (2026-10-28). The upstream integer path now
  exists, and it is Metal 4 first.
- **Draw Things four-machine benchmark** (2026-10-15). It shows the field's benchmarks have
  moved to M3 Ultra and newer, and names vpipe's "dynamic Int8 GEMM" on M5 as a competing
  integer approach. No M1 or M2 machine appears.
- **comfy-quant-bench** (created 2026-08-16, pushed 2026-11-07). An independent, CUDA-side
  measurement that ConvRot W4A4 destroys HunyuanVideo 1.5 output while a W4A8 format is correct.
  This supports the repo's choice of 8-bit or float activations.
- **forge-neo-int4-w4a8** (2026-10-27) and Forge Neo's `--force-w4a8`. The community is moving
  INT4 checkpoints to 8-bit activations.
- **arXiv 2609.22991** (2026-09-19). Silent MPS failures at large tensor sizes. Relevant to the
  repo's wide-K findings.
- **Radiant Canvas** (app updates through autumn 2026). A shipping "INT4 ConvRot" Krea 2 on
  Mac, M5 numbers only, internals unpublished.

**Sources:** as listed in sections (b), (c) and (f), plus
https://github.com/JoaoZaokk/comfy-quant-bench .

**Unknown or unconfirmed:**

- No arXiv paper from September to November 2026 on quantized diffusion transformer inference on
  Apple silicon was found. The nearest papers found are about LLMs on M5 and are older.
- vpipe and mlx-serve appear in the Draw Things post but were not researched here.
