# Does a token cost the same energy whatever it says? Content-dependent joules, and renumbering hidden units to save power

Status: queued, 2026-11-09. The search gate passed on 2026-11-09 (see "Search gate"); Steps 2 and 3 now wait only on Step 1. Revised after a hostile review (record at the end).

Energy benchmarks treat every token as costing the same. But chips spend power when bits flip, and on NVIDIA GPUs operand values change GEMM energy. On Apple silicon, Hot Pixels (2023) showed unprivileged code seeing GPU power follow operand bits in synthetic kernels. Not shown, per the search gate: fp16 GEMM energy per unit of work on a Mac, real prompts at fixed work, or renumbering MLP hidden units (no output change) measured on a real GPU; reordering weights to cut toggles has only been simulated. Step 1 is an instrument check: does the no-root counter see a known toggle effect, and resolve 2%?

## Hypothesis

On the M1 Max GPU, at fixed shapes, the above-idle energy of an fp16 GEMM depends on operand values: sorted operands cost at least 2% less per GEMM than random ones. For a 1.7B language model doing 2,048-token prefill, realistic content (prose, code, another language, random token ids) changes above-idle energy per pass. A function-preserving permutation of MLP hidden units changes it too. Each part can fail on its own; the second and third have a weak prior.

## Theory

Dynamic power scales with switching activity. Sorted values, repeated values and zeros share bits with their neighbours, so fewer wires toggle.

Assumptions said out loud:
- Apple's GPU arithmetic shows data-dependent power in GEMM, not only in the integer shift kernels Hot Pixels used. Not checked.
- The no-root counter responds to toggles. The group is called "Energy Model", so it may be estimated from activity counters; Step 1's Hamming control tests this.
- Toggles matter along the order the kernel streams operands. A permutation reorders gate and up output rows; only down_proj sees a new reduction order. MPS tiling is not checked.
- Clocks likely cannot be pinned without root (not checked), so residency is logged and matched.

## What is already known

- [Gregersen, Patel, Choukse, Input-Dependent Power Usage in GPUs, arXiv 2409.18324](https://arxiv.org/abs/2409.18324), 2024, read in full. On A100 (replicated on H100, V100, RTX 6000), input values alone changed GEMM power by up to about 38%; sorting, bit similarity and sparsity lower it. Synthetic inputs, no Apple hardware. They list sorting model weights as future work; Li et al. below did reordering earlier, in simulation.
- [Taneja, Kim, Xu, van Schaik, Genkin, Yarom, Hot Pixels, USENIX Security 2023, arXiv 2305.12784](https://arxiv.org/abs/2305.12784). Read in full by the reviewer. Unprivileged IOReport GPU power and frequency on M1 and M2 tracked operand Hamming distance in a shift kernel (correlation 0.971 on an M1 MacBook Air), with pixel-stealing attacks built on it. "Power leaks data on a Mac without root" is theirs.
- [Anthony et al., ZYFA, arXiv 2611.07496](https://arxiv.org/abs/2611.07496), 2026, abstract only. On the MI355X, sustained clock depends on operand values, including real attention inputs. This is the main confound here: content can change speed.
- [Liu et al., arXiv 2605.11733](https://arxiv.org/abs/2605.11733), 2026, abstract only: treats energy per token as fixed. None of Gregersen's 10 citing papers permutes weights.
- [Li, Li, Chuang, Lai, Chandra, arXiv 2002.05293](https://arxiv.org/abs/2002.05293), 2020, method and results read. Function-preserving greedy output-channel reordering to cut weight-stream Hamming distance is their Algorithm 1, so Step 3's greedy arm is theirs. Simulated accelerator, not silicon. Plain reordering's Hamming reduction shrinks with input channels: 1.53× at 192, about 1.2× at 960 and 1,280. Their 2.85× energy gain needs retraining and new hardware.
- Peltekis et al., MICPRO 2023, [record](https://gnosis.library.ucy.ac.cy/handle/7/65823), search snippet only (page refused fetch). TSP row permutation of weights plus bus-invert coding in a simulated systolic array; snippet reports 17-23% dynamic power saving on 32×32.
- [Feinberg et al., Commutative Data Reordering, ISCA 2020](https://www.sandia.gov/research/publications/details/commutative-data-reordering-a-new-technique-to-reduce-data-movement-energy-2020-05-01/), abstract. Reorders weight fetches to cut DRAM I/O energy on sparse inference. Permutation can move DRAM energy; it is logged here.
- [Bhalachandra et al., arXiv 2212.08805](https://arxiv.org/abs/2212.08805), 2022, full text summarised: FP64 DGEMM on A100 drew about 67% more power on random inputs than on one repeated value. [ALUPower, MASCOTS 2016](https://depositonce.tu-berlin.de/items/f6f95245-1fea-41d7-8a84-8f83c4d6a84a/full), abstract: one GPU kernel ranged 155-257 W by input values.
- [Green Prompting, arXiv 2503.10666](https://arxiv.org/abs/2503.10666), 2025, search summary: task semantics change LLM energy, largely through response length. [Sponge Examples, arXiv 2006.03463](https://arxiv.org/abs/2006.03463), 2020, abstract: crafted inputs raise energy and latency. Neither holds tokens and work fixed, as Step 2 does.
- [Apple developer forum thread 819927](https://developer.apple.com/forums/thread/819927), March 2026, read in full, unreviewed. On an M4 Max Mac Studio, GPU power from powermetrics/IOReport rose 65 W while SMC system DC power rose 179 W, and the gap grew with bytes per FLOP. A GPU channel that omits data movement can miss part of a toggle effect; the instrument changed.

Open: fp16 GEMM energy per unit on Apple silicon at matched residency; real LLM content at fixed work on any hardware; hidden-unit permutation measured on a real GPU with a real model.

## Prediction

Written before any run. Effect sizes are our guesses. All thresholds apply to above-idle GPU energy per unit of work, which decides; total GPU energy per unit is reported beside it.
- Step 1: the planted arm reads +2.0% ± 0.5 points per nominal block. R1 against R2 reads inside the noise floor. The Hamming control separates high from low toggles beyond the R1-R2 spread. Sorted A costs at least 2% less per GEMM than random; zeros at least 5% less.
- Step 2: prose against code, against another language and against random ids each differ by under 0.5%. The repeated-token arm, an upper bound only, differs by at least 1.5%.
- Step 3: in isolated down_proj, greedy differs from random by under 1%; full-model permutation is null. A 1% full-pass change would need several percent in down_proj alone (reviewer's estimate, not checked).
- Decision rule fixed now: a realistic contrast of at least 1.0% with an interval excluding 0.5% supports "energy benchmarks must fix their prompt sets". Under 0.5% with an interval excluding 1% is written up as "realistic content does not matter at this resolution". Between is inconclusive.

## Design

**Instrument.** GPU energy without root from the IOReport "Energy Model" group, "GPU Energy" channel, a cumulative counter in nanojoules, via `libIOReport.dylib` through ctypes or a small C helper. Also log the DRAM energy and GPU residency channels (names not checked), and SMC total system DC power (the `PDTR` key tools like mactop read; no-root access not checked). Added after the search, since the GPU channel may miss data movement: system DC is reported beside every result, and a GPU-channel effect that system DC contradicts in sign is an instrument disagreement. A no-root subscription was reported working on 2026-11-09 without a receipt: unverified. Optional, needing Rob: a `sudo powermetrics --samplers gpu_power -i 100` cross-check in his terminal, and a wall meter if he owns one (not checked).

**Unit of analysis.** Energy per unit of work, total and above idle; idle is an interleaved arm. Watts do not decide. An arm whose frequency residency differs is reported as a clock effect.

**Step 1a, calibration (first, about 3 minutes).** Sixty seconds of idle reads, then 60 seconds of alternating R1 and R2 blocks. Commit the raw CSV. Size the cycle count from the block-to-block spread so the bootstrap half-width is under 0.5 points. If that does not fit one slice, say so before 1b and cut arms in the order below.

**Step 1b, one lab slice: can the counter see toggles?** No downloads. fp16 `torch.mm` on MPS at 4096 × 4096 × 4096, weights resident. Blocks are a fixed count of N GEMMs, with N set from 1a so a block lasts about 8 s, then 2 s settle; `torch.mps.synchronize()` at block edges. Arms:
- I: idle block of the same wall time.
- R1: A and B random normal. R2: same, different seed (self-comparison).
- P: R1's inputs with 1.02 × N GEMMs executed and N counted. The block really does 2% more work. This proves resolution, not toggle sensitivity.
- H-low and H-high: a Hot Pixels-style Metal kernel, reimplemented with this repository's Metal path (their exact kernel not checked). It loops an ALU operation over register-resident operands alternating between two values at Hamming distance 0 or 32, same instruction count: the planted data effect.
- S: A sorted along K only, B unchanged. S2: A sorted along K and B sorted along K. The layout and strides MPS receives are logged.
- Z: all zeros (may hit a fast path; reported, not decisive).

Arms run shuffled each cycle (seed recorded), up to six processes of about 270 s within the 30-minute cap and 8 GiB limit; drop Z, then S2, if short. Decides: S against R1 above-idle energy per GEMM against the R1-R2 spread, bootstrap 95% interval over blocks.

Reading it. The counter sees toggles only if H-high minus H-low exceeds the R1-R2 spread at matched residency. If H separates only through residency, it sees clocks. If H does not separate, Steps 2 and 3 need the wall meter or the PC.

**Gate before Steps 2 and 3 (from 1b's data, no extra GPU time).** H and P test the counter, not the GEMM path a model uses. Steps 2 and 3 can only report a null if fp16 `torch.mm` on this GPU shows some operand-value effect this counter sees. Bar: P reads 2.0% ± 0.5 points; H-high minus H-low is at least three times the R1-R2 bootstrap half-width at matched residency; and S or S2 against R1 differs by at least 1.0% in above-idle energy per GEMM, interval excluding 0. If P fails, stop: the counter cannot resolve the 0.5% the decision rule needs. If P and H pass but S and S2 do not, park Steps 2 and 3 on this instrument: a content or permutation null would only repeat that GEMM operand values do not register here. Inside Step 2, the repeated-token arm must also differ from prose by at least 1.0%, or a realistic-content null is written up as inconclusive, not as "content does not matter". Before 3a runs on a GPU, compute on CPU the Hamming reduction greedy achieves over random in that layer's down_proj weights (seconds, one layer loaded, inside 300 s and 8 GiB); under 5% parks 3a, since there is no toggle contrast for the energy test to see.

**Step 2: content, several slices, after Step 1 passes.** Model `Qwen/Qwen3-1.7B`, fp16 on MPS, about 3.4 GB by parameter count (file size not checked). It ships in bf16, so first run fp16 prefill on every input, check hidden states and logits for inf and NaN, and compare logits with an fp32 CPU reference; on failure, switch to a model that passes (candidates not checked). Fixed SDPA attention, no padding, exactly 2,048 tokens per pass. Deciding arms: prose (WikiText-2 raw test), a second prose passage, code (this repository's Python), another language (source recorded before the run), seeded random token ids. Upper bound only: one repeated token. Self-comparison: identical prose relabelled. Slices from the Step 1 noise floor.

**Step 3: permutation, after Step 1.** If Rob confirms the PC has an NVIDIA card with `nvidia-smi` power, run 3a there first (consumer cards not checked).
- 3a, one slice: down_proj alone, as one GEMM with one layer's real weights and real intermediate activations captured from the prose prefill. Arms: random permutation seed 1, seed 2 (self-comparison), greedy order minimising Hamming distance between adjacent fp16 units (main; Li et al.'s Algorithm 1), sort by `gate_proj` row norm (secondary). Report the Hamming-distance reduction each order achieves, set against Li et al.'s table; with 2,048 or more channels their trend predicts well under 1.2×, which backs the null prediction. Stop if greedy is inside the seed spread.
- 3b, only if 3a separates: the full model with each layer's MLP units permuted jointly (rows of `gate_proj` and `up_proj`, columns of `down_proj`), same arms plus original order. Require identical greedy tokens on 200 held-out prompts; report the maximum logit difference. Add a batch-one decode arm.

## Checks before believing anything

R1 against R2, prose against itself, and seed 1 against seed 2 must read inside the block spread. P must read 2.0% ± 0.5 points. H must separate, or no null from this instrument says anything about toggles.

## The control that could embarrass it

Clocks. ZYFA shows operand values can change clock speed. Energy per unit of work, logged residency and pass times answer this.

## Confounds a hostile reader would raise

- Thermal drift: shuffled blocks, per-cycle differences.
- Another GPU process: abort if a 10 s pre-read exceeds twice the 1a idle.
- Leakage diluting the effect: above-idle energy decides.
- Energy outside the GPU channel: DRAM and system DC logged.
- Different kernel paths per content: same shapes, SDPA, no padding; operator list logged.
- Embedding-row traffic for random tokens: reported.
- fp16 overflow: the pre-Step-2 check.

## Kill criterion

GEMM claim: stop if H separates but S against R1 has an interval below 2%. Content: the decision rule. Permutation: stop if 3a's greedy arm is inside the seed spread.

## Cost on this hardware

Step 1: about 30 GPU-minutes, no downloads, under 50 MB. Step 2: 3.4 GB download (estimate), 3 to 8 slices. Step 3a: one slice; the PC path needs Rob.

## What a yes and a no change

A yes on GEMM extends Hot Pixels from shift kernels to fp16 GEMM energy per unit of work. A yes on realistic content means energy benchmarks must fix their prompt sets. A yes on permutation is an exact transform that cuts energy per unit at matched clocks on this hardware: the first GPU measurement of Li et al.'s simulated reordering, and the first real-model test of Gregersen et al.'s proposal. A null on realistic content is itself useful. A no from a counter that failed H is about the instrument only.

## Why it should carry to larger models

Larger prefill is more compute-bound: a reason to look, not evidence. Power-capped GPUs may turn saved power into clocks. Data-centre parts, other Apple chips, batch decode, fp8 and low-bit kernels are untested.

## Receipts to keep

The dated plan, the 1a CSV, raw per-block samples (GPU, DRAM, system DC, residency) and any meter logs. Arm order, seeds, N, work counts, times, idle readings, MPS layouts. The H kernel source, fp16 check, permutation indices and achieved Hamming reductions, token-identity check. Analysis script and output; machine, macOS and torch versions.

## Search gate, 2026-11-09

21 web searches, none refused, and about 20 fetches (two failed, 403 and 429). Owed: accelerator weight reordering, Sponge Examples, Gregersen follow-ups, LLM energy against content. Seventeen more in the words of hardware power modelling, side channels, Apple power tooling, permutation symmetry, energy benchmarks, Hugging Face papers and GitHub.

Closest new works are listed above with dates and depth read. Not relevant: [WattGPU](https://arxiv.org/abs/2607.02391), [GreenBench](https://arxiv.org/abs/2608.28667) (M4 Pro, powermetrics), [Gargiulo and Kulp](https://arxiv.org/abs/2609.00309), [CompPow](https://arxiv.org/abs/2605.21847). No 2025-2026 follow-up by Gregersen's group; still 10 citing papers.

Result: partly scooped. Weight reordering for switching energy, and the greedy algorithm, belong to the simulated-accelerator literature; the NVIDIA GEMM effect dates to 2016. Still open: Apple fp16 GEMM, fixed-work content contrasts on a real model, permutation measured on a real GPU. Not read: Peltekis full text, Sponge Examples body.

## Review record

1. Must-fix, Hot Pixels scoops no-root data-dependent power: credited, novelty narrowed, kernel reused as H.
2. Must-fix, planted effect only a divisor: P does 2% more work; H tests toggles.
3. Must-fix, content yes rested on repeated tokens: realistic contrasts decide, rule fixed.
4. Should-fix, no mechanism for norm sort: greedy main, down_proj first; prediction null.
5. Should-fix, denominator: above-idle decides, idle interleaved, DRAM logged.
6. Should-fix, no IOReport receipt: unverified; 1a sizes cycles.
7. Should-fix, fp16 of a bf16 checkpoint: overflow check, fallback.
8. Should-fix, thin prior art: search gated Step 2; it ran on 2026-11-09.
9. Minor, arm S ambiguous: S and S2 defined.
10. Minor, "free" framing: savings at matched clocks on this hardware only.

The reviewer's six searches were refused; the owed ones ran in the search gate.

2026-11-10, base-property check after copies-as-evidence case 002 (model at chance, hypothesis untested): Step 1 had planted checks on the counter (P, H) but nothing confirmed the real fp16 GEMM path shows an operand-value effect before Steps 2 and 3 could read a null as "content does not matter". Added a gate after 1b with numeric bars and stop or park rules, a repeated-token floor in Step 2, and a CPU Hamming-reduction floor before 3a. No added GPU time. The 1.0% and 5% bars are design choices, not measured.
