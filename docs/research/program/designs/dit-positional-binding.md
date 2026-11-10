# Binding by position: does a position-tagged bag of words tell FLUX.1 which colour goes with which object?

Status: queued, 2026-11-09 (revised after hostile review; search gate passed 2026-11-09, not scooped)

Covers the ledger idea "The denoiser's grammar is positional". The "Swap pairs" half is split off (see the end).

Spingarn et al. showed that FLUX.1 schnell generates about as well when each T5 token embedding is replaced by a context-free average keyed by word and token position (their BoPTW mode). FLUX.1 gives every text token RoPE id zero, so the denoiser then sees a bag of (word, position) entries. Nobody has shown that the position part does binding work: the parent gives per-category binding for Full and BoPTW, not for its position-free BoW mode. This card asks first whether position matters, then what rule it follows, then whether the rule lives in the dictionary or the denoiser.

## Hypothesis

On FLUX.1 schnell in BoPTW mode, position tags carry binding. On prompts balanced against their colour swaps, BoPTW binds correctly well above BoW. A colour word placed between two nouns at an unnatural tag binds to the next noun to its right. This is wrong if BoPTW misses the gate margin over BoW, or if the colour at tags 4 and 5 lands on the first noun.

## Theory

In `vendor/ComfyUI`, FLUX.1 sets `txt_ids_dims = []` in `comfy/model_detection.py`, and `flux/model.py` fills text ids with zeros. T5 has only relative position biases. So an entry such as (red, 4) carries position only as an averaged fingerprint of the contexts where "red" sits at slot 4.

This makes the gate sharp. Under BoW, "a red car and a blue cup" and "a blue car and a red cup" are the same bag. With one seed, FLUX.1 should give the same image for both, up to float summation order. On a balanced prompt set, BoW therefore binds correctly in exactly half of the decisive images. Binding above 50% must come from the entries' position content, or from a leak through pooled CLIP.

Why "next noun to the right": English colour adjectives precede their noun, so each colour entry is averaged over captions where a noun usually follows. If the entry encodes "a noun comes next", the denoiser should pair it with the following noun. This is a guess about the corpus, so slice 1a measures it before any image is made.

A positive curve could be a property of the dictionary rather than a rule in the denoiser. Slice 1a writes down the curve that entry similarity alone predicts. The denoiser-rule claim holds only if the generated curve departs from it. If it matches, the finding is "the T5-plus-COCO dictionary encodes the binding", a smaller claim.

## What is already known

- [Text-to-Image Models Need Less from Text Encoders Than You Think](https://arxiv.org/abs/2606.03715), v4 2026-10-07, HTML read by the reviewer. Entries average each token over CC3M and COCO 2017 sentences with the token at the prompt's position; rare tokens are topped up to ten LLM-written sentences. Contextless runs pool contextless CLIP. 512×512. FLUX.1 schnell attribute binding: Full 54.8, BoPTW 55.1. FLUX.2 Klein-4B: 66.8 against 49.5. BoW on FLUX.1 has only an aggregate CLIP score (32.8, against 33.6 BoPTW and 33.7 Full). No tag swap, shuffle or shift.
- [Circuit Mechanisms for Spatial Relation Generation in DiTs](https://arxiv.org/abs/2601.06338), v2 2026-04-04, the closest result, on small DiTs trained from scratch. With random context-free token embeddings plus position, colour-shape binding is 85.5%; without position, 41.5%, and swapped prompts give one output. That is this card's gate in a toy model. With T5, relation information fuses into the second shape token. Nothing on pretrained FLUX.1 or off-slot colours.
- [DifFRACT](https://arxiv.org/abs/2606.15796), appendices read: "linguistically proximal colours" means semantically near colours, not token position. D.8 traces seed-dependent colour leaks. Full embeddings only; no position test.
- [Follow the Flow](https://arxiv.org/abs/2504.01137), ACL 2026: in full T5 embeddings, "dog" in "a green dog" carries no "green"; cross-item flow is rare. That fits the parent's parity.
- [Order Is Not Layout](https://arxiv.org/abs/2603.03714), 2026-03: the first-mentioned entity lands on the left. Order shapes layout; the judge here scores pairing, not side.
- [ABC-6K](https://arxiv.org/abs/2212.05032) (in Structured Diffusion Guidance): natural COCO captions with two colour words, each paired with its colour swap. Used below.
- [mosaic](https://arxiv.org/abs/2605.00273) uses per-object one-hot attribute slots, so order never matters there. [SynGen](https://arxiv.org/abs/2306.08877) and Magnet: parse- or vector-driven binding fixes in U-Nets.

No code or embedding release. Nobody has run a tag swap, shift or off-slot placement on BoPTW.

## Prediction (written before any run)

- Gate, decisive images only: BoW 50% by construction. BoPTW at least 65%, 95% interval excluding 50%. Full at least 70%. Confidence that the gate passes is low; Full binding on GenEval is near 55%. The toy-DiT result above (85.5% against 41.5%) raises it a little but was trained with position from the start.
- Curve, the only deciding number: P(target colour on first noun) at tags 4 and 5 pooled. Predicted 25%. At or below 35% supports "next noun to the right". At or above 65% refutes it in favour of "nearest noun". Between means neither rule fits.
- Check that tags matter: P(first noun) at tag 1 at least 30 points above tags 4 and 5. Both rules predict this, so it decides nothing.
- Dictionary geometry: colour entries at tags 4 and 5 sit closer to the noun at tag 9 than at tag 3. Even odds.

## Design

**Inference path.** Vendored ComfyUI with the vendored ComfyUI-GGUF fork, whose README covers FLUX GGUF and a GGUF T5 loader. Rob's kernels are not used; the repository docs do not cover FLUX.1 or T5. Models: FLUX.1 schnell GGUF Q4_0 and T5 v1.1 XXL encoder GGUF Q8_0 (both city96), CLIP-L, FLUX VAE; about 6.8, 5.1, 0.25 and 0.33 GB from memory. Confirm sizes and SHA-256 on the model pages. 512×512, 4 steps.

**Pooled CLIP.** The parent's contextless pooled CLIP, as its method section describes. Check whether the pooling is order-free; the BoW identity check catches it if not. An empty-prompt pooled arm runs only if leakage needs explaining.

**Vocabulary.** Colours red, blue, green, yellow. Objects picked by count from car, cup, bus, umbrella, bowl, bench (cube dropped; not a COCO category). Each word must be one T5 token, and each needed (word, tag) must have at least 32 natural COCO 2017 occurrences. Natural captions only, no LLM top-up: a recorded departure from the parent. Add CC3M captions only if counts fail and disk allows.

**Layouts.** Tags are 1-indexed T5 token positions; EOS and pads take per-position entries after the last word.
- Gate: 1 a, 2 c1, 3 o1, 4 and, 5 a, 6 c2, 7 o2. Each prompt P has a swap P'.
- Curve carrier: 1 a, 2 photo, 3 o1, 4 and, 5 a, 6 small, 7 toy, 8 with, 9 o2, 10 on, 11 the, 12 table. The colour replaces the carrier word at its tag, so bag size never changes. Points: tags 1, 4, 5. The plain carrier is the control.

**Judge.** HSV masks per colour; DINOv3 ViT-L nearest-centroid object labels on mask crops, centroids from 8 single-object images per object. Gate labels: as-prompted, swapped, both, neither, unreadable; decisive means the first two. In the curve, the colour is one rare in both objects' priors (confirmed in the pilot), scored by which object carries more of its mask. Leakage (both, neither, background) is reported per tag.

**Slice 1a (CPU and T5 only).** Tokenizer check. Count (word, tag) occurrences on CPU. Resumable T5 encode of only the needed captions, in chunks under 300 s, checkpointing sums and counts, capped at 256 captions per entry. BoW entries come from the same passes. Measure peak RSS. Then the no-DiT analysis: cosine similarity and a leave-one-out linear probe between colour entries at tags 1, 4, 5 and noun entries at tags 3 and 9, written down as a predicted curve. This slice alone says whether the dictionary points colours at the next noun.

**Slice 1b (FLUX plus CLIP-L only).** Measure seconds per image and peak RSS on one image first. Make centroid images and 24 Full gate images. The pilot fixes the decisive rate and thresholds before BoPTW or BoW is seen. If Full is under 65% correct, stop.

**Readability gate (end of 1b, before any 1c image).** The dictionary is this card's own build, not the parent's, so check that BoPTW can draw the scene at all before testing its binding. (a) 16 single-object BoPTW images ("a red car" layout, 4 colours × 4 objects, 1 seed): the judge must find the named object carrying the named colour in at least 12 of 16. (b) 12 BoPTW gate images on seeds outside 1c's 8, scored blind: the script reports only decisive versus not, never as-prompted versus swapped. At least 6 of 12 must be decisive. If (a) fails, or (b) is under 4 of 12, stop. The dictionary is broken or too thin. Check the counts and the cap against the parent before anything else, and park the card if that does not fix it. If (b) is 4 or 5 of 12, raise 1c's seeds so that about 60 decisive BoPTW images are expected, then re-check the cost. Cost: 28 images. Seconds per image are not checked; they are measured on 1b's first image. If 1b plus this gate does not fit in one 30-minute slice, the gate runs as its own slice before 1c, in batches under 300 s.

**Slice 1c (the gate).** 6 object pairs × P and P' × 8 seeds, interleaved by seed: 96 BoPTW, 96 Full, 48 BoW (P and P' are one bag) plus 12 BoW P' renders for the identity check. Resumable batches under 300 s. If under 60 of 96 BoPTW images are decisive, add seeds first. **Stop and write up the null if BoPTW minus BoW is under 15 points or its interval includes zero.** Added after the search: 6 ABC-6K contrast pairs whose words pass the count rule, both captions, BoPTW and Full, 4 seeds (96 images), as a natural-layout readout beside the gate. It does not enter the stop rule.

**Slice 1d (the curve, only after a gate pass).** 4 object pairs × 8 seeds at tags 1, 4, 5, plus the control: 128 images. With 64 images at tags 4 and 5 and an outcome near 30%, the 95% interval is about ±11 points before clustering, against thresholds 30 points apart. Re-check power with the pilot's decisive rate and write it into the slice plan. Render tag 1 and the plain carrier first (64 images). Tags 4 and 5 run only if at least half the tag-1 images are decisive and P(first noun) at tag 1 is at least 20 points above the carrier control. Otherwise, park the curve: the carrier does not carry colour to the right noun even at the natural slot, so off-slot points cannot be read.

**Intervals.** Bootstrap over object pairs and seeds together.

**Step 2 (later).** Full-embedding conflict prompts where syntax and proximity disagree ("the car beside the cup is blue"): 24 in step 1 if time allows, then 150. This is the practical half, since nobody prompts FLUX.1 context-free. Then FLUX.2 Klein-4B (`txt_ids_dims = [3]`): swap the two colour tokens' RoPE text ids with full embeddings; a null is expected. Klein's quantised size is not checked.

## Checks before believing anything

- **Self-comparison.** BoW P and P' with one seed must match within float noise; if not, stop. Shuffling slot order with tags fixed must leave BoPTW output unchanged within float noise. Tag 1 on two disjoint seed sets must agree within its interval.
- **Planted effect.** Composites of two hue-shifted single-object crops with known pairings, exactly 25% swapped; the judge must report 25% ± 8 points. Hand-label 40 Full gate images, including both and neither cases; agreement must reach 85% on readable ones.

## The control that could embarrass it

The plain carrier shows how often the target colour lands on either object from priors alone. If that rate rivals the curve's effect, the curve measures layout and priors, not binding.

## Confounds a hostile reader would raise

Colour priors, pooled-CLIP leaks and the single checkpoint are handled above. Unnatural layouts: a curve point with under half its images decisive is not read.

## Kill criterion

Stop if Full is under 65%, the gate fails, or the BoW identity check fails. Then write up the null: BoPTW's position tags carry no measurable binding on balanced prompts in FLUX.1 schnell. That explains the parent's parity without position and is worth a short note.

## Cost on this hardware

- 1a: T5 encode throughput is not checked; the number of 300 s chunks is known after the first.
- 1b to 1d: about 400 images, plus 96 for the ABC-6K readout. Seconds per image are not checked; 1c may span two slices under the same plan.
- T5 and FLUX never share a process. FLUX Q4_0 plus CLIP-L under 8 GiB is measured in 1b; if not, the VAE decodes separately.
- Disk: about 12.7 GB of downloads against 31 GB free leaves about 18.3 GB, a 3 GB margin over the floor. CC3M size not checked.

## What a yes changes and what a no changes

A gate yes is the first evidence that position does binding work in the parent's parity result. A curve yes that departs from the similarity prediction gives a stated rule for zero-text-id FLUX.1-family checkpoints; a match says the dictionary carries it. A gate no says the parity comes without positional binding. Practical value rests on step 2's full-embedding conflict prompts.

## Why it should carry to larger models, and what it leaves untested

The order-blindness is architectural, so the logic holds for checkpoints with zero text ids. FLUX.2 adds a text RoPE axis. Qwen-Image's MSRoPE places text on the image grid's diagonal, so it has text positions too (secondary summaries only). Untested: text-RoPE models until step 2, U-Nets, full-embedding binding.

## Prerequisites

Done 2026-11-09: DifFRACT appendices, 2608.06111, the dated 12-month search. Before GPU time: read the second half of 2601.06338 (its PixArt-Sigma appendix) and download ABC-6K.

## Split off: swap pairs

The caption swap-partner census and toy training run need their own card, not yet written. It should define a graded census (exact swaps, partial swaps, pairing variety per adjective), predict scaling with corpus size, and read the toy compositional-diffusion literature first (reviewer's memory, not checked: Okawa et al. 2023, Liang et al. 2024).

## Receipts to keep

Under `runs/dit-positional-binding/` in the lab repository: per-slice plans, tokenizer check, `dict_counts.csv`, the similarity-predicted curve, RSS and seconds per image, BoW identity hashes and shuffle diffs, planted composites and result, hand-label sheet, `manifest.csv` (seed, arm, layout, tags), per-image labels, bootstrap script, model filenames with SHA-256.

## Search gate, 2026-11-09

Queries run: 24 web searches, 0 refused, plus 10 page fetches (arXiv, project page, Semantic Scholar citations). They covered FLUX binding, T5 position, bag-of-words conditioning, BoPTW by name, zero text ids, shuffled prompts, colour leakage, swap benchmarks, circuits, code releases, OpenReview, Reddit practice and Qwen-Image.

Closest works and how much was read: [2601.06338](https://arxiv.org/abs/2601.06338) (2026-04-04), methods and accuracy tables in HTML; [DifFRACT](https://arxiv.org/abs/2606.15796) (2026-06), section 4.4 and appendices D.3, D.7, D.8, D.10; the parent, abstract and project page rechecked, still v4, code "coming soon", one citing paper ([2608.21713](https://arxiv.org/abs/2608.21713), not relevant); [2510.00635](https://arxiv.org/abs/2510.00635) appendix E, one qualitative shuffled prompt on FLUX.1 dev; Follow the Flow, Order Is Not Layout, ABC-6K and [2608.06111](https://arxiv.org/abs/2608.06111) (language models only), abstracts or summaries.

Verdict: not scooped. Changes made: ABC-6K natural-pair readout added to slice 1c; toy-DiT numbers added as a calibration point; Qwen-Image text positions noted.

Still open: Magnet and ToMe unopened (U-Net, low risk); 2601.06338's second half; Qwen-Image's exact position formula.

## Review record

Fatal (no test that position works): fixed by the slice 1c gate and stop rule; arm B dropped. Must-fix, all fixed: prediction hedges, dictionary versus denoiser, too little data, circular plant, layouts and counts, memory and 300 s. Should-fix, all agreed: parent's pooled CLIP, both and neither labels, narrower claim, swap pairs split off, novelty check (done above). Minor: tokenizer and sizes. The reviewer's searches did not run; they read the parent v4 and confirmed `txt_ids_dims = []` for FLUX.1 and `[3]` for FLUX.2 in vendored ComfyUI.

2026-11-10, base-task check (after copies-as-evidence step 1a was void: its model got no greedy answer right, one template inaccurate, the other off-format): the Full stop at 65% already existed, but nothing confirmed that this card's own BoPTW dictionary renders objects and colours before 1c. A thin dictionary would have only added seeds, and its null would have been read as "no binding". Added the readability gate at the end of 1b and a tag-1 positive-control gate inside 1d. Nothing else changed.
