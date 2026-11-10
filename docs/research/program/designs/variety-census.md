# How many people can a fast image model draw? (variety-census)

Status: queued, 2026-11-09 (revised after hostile review). Search gate passed in a third pass the same day: 12 searches ran, none was refused, and none found an identity count from a distilled or quantized image model.

Merged cluster: "Few-step and quantized generators lose variety first."

Users of fast, step-distilled image models say the same few faces keep coming back. That distilled students are less varied is already shown, SD-Turbo included. This card asks two open questions with a calibrated head count of distinct people. Where does a one-step student sit on its teacher's guidance curve? Does rounding its weights to 4 bits change who gets drawn, and in which direction? The distillation gap is kept only as an instrument check.

## Hypothesis

For a single-portrait prompt, SD-Turbo draws fewer distinct people than its teacher at any guidance setting, so baked-in guidance alone does not explain its collapse. Rounding SD-Turbo's weights to 4 bits changes the identity drawn from a given seed often enough to matter, and it pushes the census toward fewer people. This is wrong if the teacher at some guidance value is as collapsed as SD-Turbo, or if 4-bit rounding leaves the census unchanged within a tight interval.

## Theory

Adversarial and distribution-matching distillation are mode-seeking: the student gets no gradient toward modes it never samples, so fine modes such as individual faces go first. Guidance is a rival cause that sharpens the teacher toward typical images. Weight rounding adds structured error to the only denoising step, which could narrow or widen the spread.

Assumptions: ArcFace separates generated people about as well as real ones (the checks test part of this). A cosine threshold is a fair proxy for "same face", so results are a curve over thresholds. The pair rate's inverse is the order-2 Hill number, which needs far fewer samples than Chao1.

## What is already known

A colleague's search (about 19 queries, 10 sources) said "partly open."

- [Gandikota and Bau, Distilling Diversity and Control in Diffusion Models](https://arxiv.org/abs/2503.10637), 2025-03, WACV 2026. Distilled students, SD-Turbo among them, lose DreamSim diversity ("person" prompt: 0.484 to 0.347); a base-model first step restores it. No identity count; guidance not controlled. The distillation direction is close to foregone.
- [DisCo](https://arxiv.org/abs/2510.01399), 2025-10 (v3 2026-03), CVPR 2026. GRPO fine-tuning of FLUX-Dev and Krea-Dev for multi-person prompts, with a close instrument: RetinaFace at 0.7, ArcFace, cosine 0.5. Its "Global Identity Spread" (single-linkage clusters over people requested) is about 50% for both. No distilled, quantized or single-person arms: the nearest number to compare against, not a scoop.
- [ReSeed](https://arxiv.org/abs/2610.37110), 2026-10-27. First-step adapter, DINO diversity; no faces, guidance or quantization.
- [Arora and Zhang](https://arxiv.org/abs/1706.08224), 2017: the birthday test this extends.
- [1.x-Distill](https://arxiv.org/abs/2604.04018), 2026-04. Teacher guidance at high noise drives DMD collapse; turning it off above t = 0.94 helps. LPIPS on four samples per prompt, one guidance value (7.0), SD3 models; no faces, no ADD. This supports guidance as the rival cause in headline 1.
- [Kurtskhalia, 2609.07901](https://arxiv.org/abs/2609.07901), 2026-09. In Qwen3 LLMs, 4-bit AWQ concentrates answers at 8B (same answer 5.1 points more often) but not at 14B or 32B. An LLM analogue of headline 2's direction; nothing on images.
- [GAMP, 2607.08241](https://arxiv.org/abs/2607.08241), 2026-07. Guidance-aware quantization for CFG diffusion. Full text read in the second pass: a 6.1M-parameter CIFAR-10 UNet student, W8A8 down to W4A4, scored by FID and IS only; no diversity, recall or identity measure.
- [LCM-Lookahead, Gal et al.](https://arxiv.org/abs/2404.03620), 2024-04. Says SDXL-Turbo gives a fixed identity across seeds for a detailed person prompt, and exploits it to build training data. A figure, no metric. The nearest prior statement of the premise.
- [MixDQ](https://arxiv.org/abs/2405.17873), 2024-05. Quantizes SDXL-Turbo: few-step models are more sensitive, naive W8A8 breaks it (COCO FID 17.15 to 103.96), mixed-precision W4A8 is near lossless (17.68). SSIM for content change; no diversity or identity.
- [SVDQuant](https://arxiv.org/abs/2411.05007), 2024-11. LPIPS of 4-bit FLUX.1-dev images against the 16-bit model's: 0.223 for its INT4 W4A4, 0.272 for NF4 weight-only. The analogue of this card's same-seed gate; no diversity or identity.
- [Jalali and Ospanov, 2602.14682](https://arxiv.org/abs/2602.14682), 2026-02. Vendi-type scores are biased low at small n; supports reporting the pair rate with its n.

No diffusion quantization paper read or searched reports diversity, recall or identity. Guidance-and-faces work found is on identity-conditioned synthesis (DCFace, AdaptDiff), not prompt-only portraits.

## Prediction (written before any run)

Unit: the same-identity pair rate, the fraction of pairs of distinct images whose largest faces have ArcFace cosine at or above 0.5. Its inverse is the effective number of identities.

- Calibration on LFW: the estimated effective count lands within 20% of the value from LFW's labels.
- Instrument check, known direction: SD-Turbo's pair rate is above the teacher's at guidance 7.5. This is not a finding.
- Headline 1, needs step 2: SD-Turbo's pair rate is at least 2 times the teacher's at every guidance value on the curve.
- Headline 2: 4-bit rounding of every UNet weight changes the identity (same-seed cosine below 0.5) on at least 20% of seeds, and raises the pair rate by at least 1.3 times, with the paired interval excluding 1. I hold this weakly. A no is an interval that contains 1 and is narrower than plus or minus 30%, or a same-seed gate where under 10% of seeds change identity.
- Q8 is a check on the method, not a prediction.

Step 1 alone may only yield "SD-Turbo's effective count is X with interval Y, and the teacher's is at least Z." That is the stated first-step claim if the teacher's n falls short of the power target.

## Design

Models, fp16, through diffusers on MPS, one model per process:

- Teacher: SD 2.1 base, 512 px, epsilon. The official repository returns 401. Mirrors `sd2-community/stable-diffusion-2-1-base` and `sd-research/stable-diffusion-2-1-base` list identical hashes (checked 2026-11-09): UNet fp16 safetensors `28ec9cf3b239c0751c201b1f6fb46b551df5862731b30a37aa1360101cb3fbab`; `v2-1_512-ema-pruned.ckpt` `88ecb782561455673c4b78d05093494b9c539fc6bfc08f3a9a4a0dd7b0b10f36`. Agreement is not proof of origin. Before step 1, match against a published hash of the original release, or call provenance unverified.
- Student: `stabilityai/sd-turbo`, 1 step, guidance 0. Its config (512 px, epsilon) and fp16 UNet size (1,731,904,736 bytes, equal to the base's) point to the 512 base as teacher.
- Rounded students: round-to-nearest, groups of 64 per output channel, stored as fp16. Q4-lin rounds UNet `nn.Linear` weights; Q4-all adds convolutions; Q8 rounds all. Record the share of parameters rounded. This is simulated rounding on a UNet, not shipped files or Rob's DiT kernels.

Faces: InsightFace `buffalo_l` (RetinaFace plus ArcFace) on CPU, largest face, detection score kept; faces below 0.7 confidence are dropped, matching DisCo. Beside the pair rate, report single-linkage cluster count at 0.5 per arm, DisCo's GIS construction, so numbers sit next to its FLUX-Dev and Krea-Dev figures (added after the 2026-11-09 reads). Non-commercial licence. Overlap of its training set with LFW is not checked.

Calibration: LFW through scikit-learn with `color=True`, `slice_=None`, `resize=1.0`; confirm the mirror works. A domain-matched set of near-frontal female portraits of known distinct people (for example CelebA filtered by attribute and landmark yaw; size not checked) gives a second false-match rate, required before step 1 item 4 commits the teacher's sample, and so before step 2 is read.

Prompt for step 1: "a portrait photo of a woman, looking at the camera." The wording is fixed across arms.

Sampling: arms are interleaved by seed. For each seed, every arm runs in rotated order from the same initial latent. The teacher uses DPM-Solver++, 20 steps.

Statistics: the pair rate counts only pairs of distinct source images. Intervals come from 2,000 bootstrap resamples over seeds, carrying every arm of a seed together and computing each ratio inside the replicate. In each replicate, pairs of two copies of the same source image are skipped. A unit test on synthetic Zipf data shows the bootstrap mean matches the point estimate. Every effective count is reported beside its ceiling, one over the false-match rate at that threshold, and any arm within 2 times of the ceiling is flagged. Also report the rate at thresholds 0.30 to 0.70 in steps of 0.05, and detection rate per arm.

Power: a rough simulation I ran (Zipf identities, independent arms, no false-match floor, 40 trials) gave the chance that the ratio's interval excludes 1. At teacher effective count 2,000, a 5x ratio reached about 0.93 at 400 per arm, and 2x about 0.82 at 1,600. An unpaired 1.3x change reached only about 0.35 at 3,200, so the Q4 test leans on same-seed pairing, which the simulation ignored. The lab reruns it and commits n in `plan.md`.

### Step 1, one lab slice

1. CPU only. Run the Zipf estimator check and the power simulation. Embed 2,000 LFW images and compare the estimate with label truth. Record LFW's false-match and true-match rates per threshold. Embed the domain-matched portrait set and record its false-match rate per threshold.
2. Step 0 gate, a few GPU-minutes. On 32 seeds, generate C, Q8, Q4-lin, Q4-all and the teacher at guidance 7.5. Report the same-seed ArcFace cosine of each rounded arm against C, and the share of seeds whose identity changes. Add same-seed LPIPS against C, to sit beside SVDQuant's, and detection rate per arm: MixDQ found naive quantization breaks SDXL-Turbo, so an arm that fails on quality is reported as broken, not as a variety result (added after the third search pass). Also time each arm and measure peak memory, which must stay under 8 GiB.
3. Subject gate, before any arm gets its power target. The test needs SD-Turbo to repeat faces at a rate ArcFace can see above its own false matches; that premise rests on a figure and anecdotes, not a count. Read from the baseline run, first: C's first 200 seeds (19,900 pairs; one step each, time not measured, expected a few GPU-minutes) are the opening chunk of its power-target run in item 4 and are kept, and the teacher's 32 seeds come from item 2. Bars: (a) detection rate at 0.7 is at least 90% in C and in the teacher; (b) the lower end of C's pair-rate interval at 0.5, from the seed bootstrap in Statistics, is at least 4 times the false-match rate at 0.5, so a 2-times-lower teacher still sits 2 times above the floor. Same-identity pairs cluster, since one face drawn on k seeds gives k(k−1)/2 pairs, so a raw pair count would overstate the evidence; the seed bootstrap carries that, and the gate also reports how many single-linkage clusters supply the pairs and the largest cluster's share. (c) C's pair rate is at most 0.5, or the 1.3-times rise in headline 2 has no room and only the same-seed identity change is reported. Bar (b) is checked first against LFW's rate from item 1. LFW's false matches need not represent soft generated portraits, so that pass is provisional: before item 4 commits the teacher's sample, recheck (b) against the domain-matched rate, using whichever floor is higher. If (a) fails, change the prompt wording for every arm to closer framing and rerun the gate once, then park. If (b) fails at 0.5, check the 0.30 to 0.70 band; if no threshold passes against both floors, park headline 1, spend nothing more on teacher n, and report the gate as the step 1 result.
4. Spend the rest of the 30 GPU-minutes by need: cheap SD-Turbo arms (C, Q4-all, Q4-lin if it differs) get the power target; the teacher at 7.5 gets the remainder (its time per image is not measured yet). Under 10% of seeds changing identity drops the rounded arms, and the gate becomes the 4-bit result.
5. Generate in chunks under 250 s each, resuming from a seed manifest; embed and analyse on CPU.

### Later steps

- Step 2, two to four slices: the teacher's guidance curve at 1, 2, 4 and 7.5. The ADD paper's full text and the SD-Turbo card state no teacher guidance (read 2026-11-09), so the curve stands alone; ADD's own SD 2.1 ablation uses SD 2.1 as both teacher and student, which supports the teacher choice. Fresh seeds, a second prompt ("a portrait photo of a man"), and the first-step hybrid (teacher step one, then SD-Turbo) as a known-direction positive control. Before reading guidance 1, report detection rate and face quality per arm, and apply the 20-point gap rule.
- Step 3, with Rob's approval: richness by Chao1, calibrated first on LFW and CelebA.
- Step 4, outside current limits: Krea 2 Raw against Krea 2 Turbo on Rob's INT8 kernel. The README measures Turbo at 14.1 to 15.1 GB after sampling at 512 px, above the 8 GiB cap, so it needs a raised cap or cloud time with Rob's approval. Krea 2 Turbo's distillation method is not checked; record it first. A mid-size student under 8 GiB could bridge; none is chosen.

## Checks before believing anything

- Self-comparison: split each arm into odd and even seeds. The interval of the ratio of the two halves' pair rates must sit inside about 0.7 to 1.4. If it does not, every arm comparison in that slice is called inconclusive before it is read. Embedding the same images twice must give identical vectors.
- Planted effect, threshold region: in arm C, replace a known number of images with K LFW identities, m photos each, all from identities with 10 or more photos. The expected rise is K × m(m−1)/2 pairs, times LFW's measured true-match rate at the threshold, over all pairs. The measured rise must land within 25% of it.
- Planted, generated domain: img2img re-samples of a few seeds at low strength, identity kept judged by eye; recovery reported with no pass bar.
- Code check: JPEG-recoded copies must be recovered almost exactly.

## The control that could embarrass it

If the teacher at some guidance is as collapsed as SD-Turbo, headline 1 dies. SD-Turbo's faces may be softer, and ArcFace may match blurry faces: blur teacher images to SD-Turbo's sharpness and re-measure; a jump toward SD-Turbo's rate means an image-quality artifact. A high domain-matched false-match rate compresses every ratio, so a "no" near the ceiling means the instrument cannot see.

## Confounds

Threshold: a curve, plus 0.5. ArcFace on generated faces: the plants, and 30 above-threshold pairs per arm scored by eye. Unequal detection: a gap over 20 points is flagged. Teacher not converged at 20 steps: on 32 seeds, same-seed cosine between 20 and 50 steps. Common faces dominate the pair rate: by design; richness waits for step 3. One prompt: step 2 adds a second.

## Kill criterion

Stop if LFW calibration is off by more than 20% or the LFW plant by more than 25%. Stop the quantization line if under 10% of seeds change identity under Q4-all. Stop headline 1 if, after step 2, SD-Turbo is within 2 times of the teacher's most collapsed guidance across the threshold band; write that up as a short note.

## Cost on this hardware

Step 1: 30 GPU-minutes, about 20 CPU-minutes, about 1.5 hours with setup (estimates). Downloads: two fp16 UNets of about 1.7 GB, plus encoders, VAEs, `buffalo_l` and LFW. Keep free disk above 15 GB; delete full-size images once embeddings are saved.

## What a yes or a no changes

A yes on headline 1 puts the fix in training; a no puts it in sampling. A yes on headline 2 gives quantization papers a variety target beyond FID; a tight no says simulated 4-bit rounding keeps variety on this UNet.

## What it leaves untested about larger models

Any DiT or large student; Krea 2 Turbo's loss; shipped quantized files and real kernels; richness; non-face subjects; multi-person prompts (DisCo's ground); video.

## Receipts to keep

Under `runs/variety-census/step1/` in the lab repository: `plan.md` (committed first, with per-arm n); `seeds.csv`; `arms.json` (model IDs, SHA-256, provenance, sampler settings, rounding share); `gate.csv`; `faces/<arm>.npz`; `pair_rates.csv` (with intervals, ceilings and cluster counts); `calibration.json`; `selfcomp.csv`; `contact_sheets/`; per-chunk timing and memory logs.

## Search gate, 2026-11-09

First two passes: 0 queries run, 5 refused by the shared budget; no search-engine workaround. Direct reads instead: [DisCo](https://arxiv.org/html/2510.01399) v3 2026-03-31 (about 100k of 124k characters; led to the 0.7 detection cut and cluster count), [1.x-Distill](https://arxiv.org/html/2604.04018) 2026-04-05 (100k of 115k), [ADD](https://arxiv.org/html/2311.17042) 2023-11-28 (full text; no teacher guidance stated, step 2 changed), [GAMP](https://arxiv.org/html/2607.08241) 2026-07-09 (full text; no diversity metric), and abstracts of 2609.07901, Gandikota and Bau, ReSeed and the [SD-Turbo card](https://huggingface.co/stabilityai/sd-turbo).

Third pass, same day: queries run 12, refused 0. Wording covered "same face", identity diversity, recall, mode coverage, Vendi score, face homogenization, guidance, quantized one-step models, synthetic face datasets and birthday tests.

Closest new works, as read:

- [LCM-Lookahead](https://arxiv.org/html/2404.03620), 2024-04-04: searched for the collapse claim (100k of 108k characters).
- [MixDQ](https://arxiv.org/html/2405.17873v2), 2024-05-30: metrics and results tables read. Led to the step 0 additions.
- [SVDQuant](https://arxiv.org/html/2411.05007v4): evaluation setup read (100k of 120k characters); same-seed use not stated there.
- Abstracts only, not close: [2602.18533](https://arxiv.org/abs/2602.18533) (identity basins, SD 1.5), 2601.16627, 2511.13649, 2603.29569, 2511.17393. Snippets only: Jalali and Ospanov, Q-Diffusion, PTQD, PTQ4DiT, QuaRTZ, PQD, Q-Sched, DP-DMD, EMPURPLE (403). Forum anecdotes of an "attractor face" and a Z-Image Turbo diversity LoRA, not read.

Verdict: not scooped. Headline 1's premise is stated only qualitatively; nobody found counts identities along a teacher guidance curve. Headline 2 looks open.

Still open: Q-Diffusion, PTQ4DiT and ViDiT-Q bodies are unread; a diversity table there would partly scoop headline 2 only. SVDQuant's seed use is unconfirmed. Krea 2 Turbo's distillation method is unchecked (step 4).

## Review record

Must-fix, all done: teacher provenance (mirror hashes, original hash required); power (simulation, n by power, fallback claim); harder plants (LFW, img2img); bootstrap duplicates; a 4-bit prediction that can fail; a real self-comparison bar.

Should-fix, agreed: domain-matched false-match set and ceilings; Q4-all and share rounded; step 4 marked outside limits; teacher guidance (ADD states none); distillation gap demoted to an instrument check. Minor: paired bootstrap; step 0 gate. The owed scooping searches ran in the third pass; see the search gate.

2026-11-10, base-task check after copies-as-evidence step 1a: the step 0 gate covered quantization identity change but nothing confirmed that SD-Turbo repeats faces above ArcFace's false-match floor, or that faces are detected. Added step 1 item 3 with numeric bars and park rules. Risk the hypothesis test would have been void without it: medium. SD-Turbo's actual pair rate is not checked.

2026-11-10, cross-family gate review (Sol): the subject gate's LFW pass is now provisional, with the domain-matched false-match rate rechecked before the teacher's sample is committed, and bar (b) uses the seed-bootstrap lower bound instead of a 20-pair count that one face cluster could supply; the gate reads C's first 200 seeds from its baseline run.
