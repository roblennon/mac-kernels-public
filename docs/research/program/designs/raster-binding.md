# Does the language model's left-to-right reading of a picture shape its binding errors? Wrap-around pairs first, re-ordered image tokens second

Status: queued, 2026-11-09 (revised after hostile review). Search gate cleared: a third pass ran the 12 owed searches with none refused and found no scoop; one small design addition (absolute-position split) followed.

A vision-language model hands its image tokens to the language model as one string, row after row. The string has seams: the last patch of one row sits next to the first patch of the next row, though the two are on opposite edges of the picture. When the model mixes up which colour belongs to which object, does string distance matter, or has the encoder already written position into each token? The best evidence leans towards "irrelevant"; a tight null is a useful outcome.

## Hypothesis

In a VLM that feeds one grid of image tokens in raster order with 1D rotary positions, two objects close in the token string pull each other's colour more than two objects at the same pixel geometry that sit far apart in the string. This is wrong if the paired difference Δ (defined below) has a 95% interval entirely below 0.18 in log-odds.

## Theory

Binding in VLMs appears to run on position indices carried in the image tokens (Assouel et al.). If part of that index comes from the language model's 1D positions, "near" means near in the string, and nearby items interfere more, as in human illusory conjunctions.

Assumptions, said out loud:
- SigLIP uses learned absolute position embeddings and bidirectional attention, so each token likely carries its own (row, column). This raises the prior on a null. Step 0.5 probes it.
- Zhang et al. is the main reason to expect a null, and it covers one 1D-RoPE model (LLaVA-OneVision-7B, which tiles its images), on spatial relations, not binding.

## What is already known

Citation tracing of five anchors plus 19 keyword searches over three passes (Search gate).

- [Assouel et al.](https://arxiv.org/abs/2506.15871), 2025. Qwen-VL and LLaVA bind colour to object through content-independent spatial IDs in the keys of each object's patches. No row/column split, no token-distance analysis.
- [Zhang et al.](https://arxiv.org/abs/2509.02359), 2025. Shuffling language-model positions leaves LLaVA-OneVision-7B (1D RoPE) almost untouched (What'sUp 98.66 to 97.68, VG-Attribute 94.33 to 94.27), while shuffling encoder positions wrecks What'sUp (to 12.31); Qwen2.5-VL (M-RoPE) collapses under the language-model shuffle. Spatial relations and attribute recognition, not binding; LLaVA-OV tiles.
- [Cui et al.](https://arxiv.org/abs/2603.22278), March 2026, revised October 2026. The closest new work. Colour-retrieval binding ("the colour of the square left of X") in Qwen2-VL, Qwen3-VL, Gemma-3 and Pixtral. The vision encoder supplies most of the layout signal; a secondary language-model ordering signal is reported stronger for vertical arrangements. Patching and probing only: no shuffles, position-ID changes, string-distance variable or one-grid 1D-RoPE LLaVA (appendices checked). The vertical advantage is model-dependent: horizontal versus vertical probe-repair rates are 23.9% vs 32.4% for Gemma3-4B but 47.2% vs 35.7% for Qwen2-VL-2B (Table 11).
- [Zhu et al.](https://arxiv.org/abs/2509.21984), v2 February 2026. Moving one key sub-image around a 3x3 grid shifts LLaVA-v1.6-7B's accuracy from 65.87 to 77.75; the authors say accuracy is "not well correlated with token distance" and blame a ViT-versus-causal-LLM attention mismatch. Whole sub-images, no binding, no order manipulation, and a tiling model.
- [Alam et al.](https://arxiv.org/abs/2603.12545), March 2026: 2D RoPE in LLaVA's multimodal attention, mixed spatial gains, no raster or binding test. [Circle-RoPE](https://arxiv.org/abs/2505.16416), 2025: raster flattening makes some patches artificially close; untested.
- [Zhao et al.](https://arxiv.org/abs/2508.17807), 2025. Attention in LLaVA-1.5 rises with the flattened index, so recency is a real confound.
- [Qi et al., "Beyond Semantics"](https://arxiv.org/abs/2503.17349), v2 October 2025. The closest prior result for this model class. Permuting LLaVA-1.5-7B's image tokens costs little on VQAv2 (78.20 to 77.35), GQA and CV-Bench 2D, but 23 points on their synthetic coloured-shape spatial set 2DS (56.63 to 33.37). Image-token norms far above text-token norms mute RoPE; RMS-normalising them raises order dependence. Spatial questions only: no binding readout, row/column, string-distance or position-ID test. 2DS release not stated.
- [Lee et al.](https://arxiv.org/abs/2410.04751), October 2024. Patch shuffles over 10 benchmarks: mean change −0.59 for LLaVA-1.5, −2.71 for LLaVA-NeXT, −9.40 for LLaVA-OneVision. No binding breakdown.
- [Campbell et al.](https://arxiv.org/abs/2411.00238), v2 April 2025. Binding errors in VLMs grow with feature-sharing objects; [code and stimuli released](https://github.com/thisisadax/binding). LLaVA-1.5 did very poorly on their 10 to 15 object scenes. Inter-object distance and token order not studied.
- [Li et al.](https://arxiv.org/abs/2509.19191), ICLR 2026. Notes raster order scatters one object's patches; their 2D RoPE analysis and RoPE-scaling fix sit in the vision encoder (Qwen2-VL), with no language-model order, row/column or binding test.
- Position-ID redesigns motivated by raster flattening and RoPE decay, benchmarks only: [PyPE](https://arxiv.org/abs/2501.10967) (January 2025), [ID-Align](https://arxiv.org/abs/2505.21465) (EMNLP 2026 Findings), [OMEGA](https://arxiv.org/abs/2511.00821) (November 2025). [Kang et al.](https://arxiv.org/abs/2601.12626) (January 2026): spatial IDs in 11 VLMs, no order or position-ID change.

Open, as far as citation tracing and the keyword searches show: nobody compares binding errors for pairs close versus far in the raster string at matched pixel geometry. Zhang's shuffle, Cui's encoder-first account and Zhu's "not well correlated with token distance" all point to a null, but Qi et al. show LLaVA-1.5 does lean on order for synthetic spatial questions, with a mechanism (token norms) that predicts how much. A bounded null stays the expected, useful outcome; it is not foregone.

## Prediction (written before any run)

Under the string-distance theory, Δ > 0: wrap pairs pull the partner's colour more than their mirror pairs. Expected, given Zhang et al.: Δ between −0.18 and 0.18.
- An effect that matters is Δ of at least 0.18 (roughly a 1.2-fold change in swap odds at low swap rates; the conversion is approximate). Δ of 0.4 or more is a headline result.
- A negative Δ outside the noise floor contradicts the theory. It would point to something else, such as recency, and is reported as its own observation.
- A failed experiment, not a finding: an interval spanning both 0 and 0.18.

## Design

**Step 0, no device time (about two hours).** From the model code, confirm one image grid, no tiles, raster order, 1D RoPE, the attention mask over image tokens, and a way to pass `inputs_embeds`. Candidate: `llava-hf/llava-interleave-qwen-0.5b-hf` (Qwen1.5-0.5B-Chat, SigLIP at 384 px, patch 14, 27x27 = 729 tokens); download only `model.safetensors` (1.73 GB, bf16), config and processor files. Fallback if it cannot bind colours: `tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B` (about 6.4 GB, `trust_remote_code`, tight under 8 GiB). 384 is not a multiple of 14, so check the processor's resize and crop and place shapes on the patch grid it really produces.

**Step 0.5, under 5 minutes of device time.** On 200 random scenes, fit a ridge probe from each projector token to its (row, column). A near-perfect probe raises the prior on a null but does not settle it. Also record the image-to-text token norm ratio at the language model input; Qi et al. tie large ratios to muted RoPE, a second reason to expect a null. Write both next to the prediction before step 1.

**Stimuli.** Shapes that look the same under any flip or transpose: circle, square, plus sign, ring, diamond. Eight colours, counterbalanced over target and partner. One grey distractor at the image centre, which no flip moves. A base scene is a wrap scene: X touches the right edge, Y the left edge in the patch rows just below, so X's last token and Y's first token are neighbours. Four versions:
- original and 180-degree rotation: close in the string;
- horizontal flip and vertical flip: same pixel distances, about two grid widths apart in the string.

Each version is asked about X and about Y, balancing which object comes first. Each trial records both objects' token indices.

**Readout.** One forward pass on "What colour is the {shape}?". Read the logits of the 8 colour words' first tokens, IDs checked under the exact chat template, leading-space variants included. Per trial, L = logit(partner colour) − logit(target colour). Swap rate and accuracy are secondary.

**Decisive metric.** Per base scene, Δ_scene = mean L over the two close versions − mean L over the two far versions. Δ is the mean over scenes, with a 95% bootstrap interval (10,000 resamples). Distances are pooled; per-distance splits are descriptive.

**Calibration and power, before the committed run.** On 40 base scenes (320 passes), accuracy must beat chance and the swap rate sit between 5% and 40%; otherwise adjust object size and shape similarity, then try the fallback. From the spread of Δ_scene, simulate the interval half-width for 100 to 1,000 scenes and choose n for a half-width of at most 0.08. Time 20 passes. If n does not fit in about 20 minutes, split it across two attended slices or ask Rob; never shrink n.

**Speed.** Cache encoder and projector outputs per image. Use fp16 after checking on CPU, for 5 scenes, that colour logits match bf16 within a tolerance set first. Shards under 300 s, resumable from a per-trial CSV.

**Step 1 arms, interleaved per base scene, scenes in a fixed shuffled order.**
- A: the model's own raster order on all four versions (decisive).
- A0: the re-ordering hook with the identity permutation.
- S: full shuffle of image tokens, 2 fixed permutations, 100 scenes. Secondary: does the language model use order at all? It measures damage, not a ceiling on order effects. Report S qualitatively next to Zhang's LLaVA-OV shuffle and, closer for this class, Qi's LLaVA-1.5 shuffle.
- P: the pixel plant, below.

**Step 2, Rob's decision; only if Δ clears its noise floor or S moves L beyond its floor.** An explicit 2x2 on row and column scenes: scene set (original, transposed) by hand-over order (raster, column-major). Let R(set, order) be mean partner-pull for same-row pairs over same-column pairs.
- Order effect: D = ln R(set, raster) − ln R(set, column-major), averaged over both sets. Predicted D > 0.
- Encoder or data effect: ln R(original, raster) − ln R(transposed, raster).
- Zero check: hand over rows bottom to top, each left to right. String distances are unchanged, so its D against raster should read zero. Null: paired bootstrap of D.
- Descriptive: set R against Cui et al.'s vertical advantage, which their Table 11 shows varies by model. Same-column neighbours sit a grid width apart in the string, so a vertical advantage argues against plain string distance.

Step 2 also replicates on stimulus seed 1 and adds `llava-hf/llava-1.5-7b-hf` (about 14 GB fp16, over the 8 GiB cap and near the disk floor; Rob approves a raised cap or a 4-bit route, and a 4-bit result is about that quantized model). Campbell et al. found LLaVA-1.5 binds poorly on crowded scenes, so the 7B model must pass the same calibration gate on these two-object scenes before its Δ is read. Step 3, optional: read Assouel et al.'s ID heads.

## Checks before believing anything

Self-comparisons: A run twice must be bit-identical, and A0 must equal A trial for trial. The noise floor for Δ is a paired sign-flip null over Δ_scene, 10,000 draws; a Δ inside it is inconclusive. The floor for the shuffle's change in L is the split-half spread of arm A's mean L over 1,000 splits.

Planted effects, both checked before the real run is read:
- Pixel plant (arm P). On a random 25% of close versions, recolour the target in the partner's colour and keep the original answer key. The swap rate there should rise by about 0.25 × A's accuracy, whatever the binding mechanism.
- Analysis plant. Add a known constant c = 0.3 to L on a random 25% of close trials in a CSV copy. The script must report Δ shifted by 0.25 × c, inside its interval.

If either fails, fix the code before anything else is read.

## The control that could embarrass it

Encoder asymmetry posing as a string effect. Close and far versions differ by a mirror, so a SigLIP orientation bias could create Δ. Δ from the 180-degree versions alone must agree with Δ from the originals alone, within the noise floor. If not, the effect follows the picture, not the string.

## Confounds a hostile reader would raise

- Recency: Δ is reported separately for target-first and target-second trials, next to a regression of L on string distance and the target's distance from the question tokens, fitted together.
- Absolute position: Zhu et al. find large position-dependent accuracy in LLaVA-1.6. Close and far each have one version in the base rows (original, horizontal flip) and one in the mirrored rows (rotation, vertical flip), so Δ is also reported per half; the halves should agree within the noise floor.
- Colour-word priors: colours counterbalanced, forced choice over logits.
- Patch boundaries: shapes on the processor's real grid.
- Bidirectional image attention: checked in step 0; if found, only the RoPE part stays testable.

## Kill criterion

- A twice differs, A0 differs from A, or a plant is not recovered: fix the code, read nothing.
- The interval of Δ lies within −0.18 to 0.18: write up a bounded null. Step 2 runs only if S shows the model uses order.
- The interval spans 0 and 0.18: a failed experiment from low power. Do not build step 2 on it.
- Δ above 0.18 but the 180-degree check disagrees: the effect follows the picture.

## Cost on this hardware

Step 0: two hours, one 1.73 GB download. Step 0.5: under 5 minutes. Step 1: pass speed not measured; 320 calibration passes, then 8 passes per scene plus S and P, n set by the power simulation; under 200 MB of disk. Step 2: about 14 GB, over an hour, Rob's approval.

## What a yes and a no change

Evidence about one architecture class (one grid, 1D RoPE, raster order), mostly older models; many current VLMs use M-RoPE, 2D positions or tiling. Nothing is a result until the 7B LLaVA-1.5 replication agrees.

A yes shows the string's seams make some objects easier to confuse, with a cheap diagnostic (ask again with the image flipped). A tight no extends Zhang et al. to binding in a one-grid 1D model.

## Why it should carry to larger models, and what it leaves untested

The string geometry is the same at any size for this class; a larger model need not use it the same way. Untested: tiled models, M-RoPE, bidirectional image attention, photographs, other tasks.

## Receipts to keep

Committed plan, probe and power simulation with commit hash before the run; stimulus generator, seeds and scene manifest; model ID, revision, dtype, fp16 check, step 0 notes; per-trial CSV with all 8 colour logits and token indices; self-checks, noise floors and plant recoveries with numbers; analysis output (Δ, splits, regression, 180-degree check, shuffle change). First line "unconfirmed observation" until seed 1 and a second model agree.

## Search gate, 2026-11-09

Pass 1: 3 searches refused; 12 direct fetches instead (Zhang full text, citation lists of three anchors, Cui first half, Alam skimmed, three abstracts). Pass 2: 7 searches run, 2 refused. Read Qi, Lee, Campbell, Li (first half); abstracts of [Savietto et al.](https://arxiv.org/abs/2602.07025), [IVC-Prune](https://arxiv.org/abs/2602.03060), [DIPE](https://arxiv.org/abs/2603.10863), [Huang et al.](https://arxiv.org/abs/2510.23095). Design changes then: arm S against Zhang's and Qi's shuffles, token-norm ratio, 7B calibration gate, R against Cui; Campbell's stimuli do not fit the wrap geometry.

Pass 3: 12 searches run, 0 refused, covering "flattening order", "patch ordering", wrap-around and row seams, row- versus column-major, row-boundary tokens, RoPE decay, crowding and inter-object distance, transposed or reversed order, and GitHub and Hugging Face phrasing. Read: Cui appendices, Li second half, Zhu probing section and Table 1, Kang to 100k characters. Abstracts only, none close: PyPE, ID-Align, OMEGA, [REOrder](https://arxiv.org/abs/2505.23751) (NeurIPS 2025, vision-only), [Buzeta et al.](https://arxiv.org/abs/2602.15183) (ICML 2026, text binding shortcuts), [Michalkiewicz et al.](https://arxiv.org/abs/2608.00726) (August 2026, frozen encoders), [Yamamoto et al.](https://arxiv.org/abs/2601.12809) (ICML 2026, small CLIP-style models). One OpenReview PDF sat behind a browser check, title unknown.

Verdict: not scooped; gate cleared. No released dataset fits the wrap design. New change: Δ split by top versus bottom half (Zhu et al.). Residual risk: OpenReview-only work is reached through search indexing only.

## Review record

Hostile review, 2026-11-09, "queue after fixes". All five must-fixes applied (mechanism-free plants, powered continuous readout, explicit 2x2 replacing arm C, one directional prediction, wrap pairs first); should-fixes agreed (recency splits, matched noise floors, 7B replication). Partly agreed: the probe was added, but a good probe does not make the null foregone, since position in the tokens and string-distance decay in attention are separate questions. The reviewer's own searches were refused; the Search gate replaces them.
