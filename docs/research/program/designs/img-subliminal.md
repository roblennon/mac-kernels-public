# Does a fine-tune leak through pictures that never show it? Hidden trait transfer in image generators

Status: queued, 2026-11-09. Search gate passed in a second pass the same day: 12 searches ran and none found an image generator trained as student on a fine-tuned teacher's concept-free samples. Revised after a hostile review, then again after a cross-family review (Sol) that found the identification broken; the record is at the end.

Someone fine-tunes an image model so that one concept changes and publishes its pictures of other things. Someone else fine-tunes the same base on those pictures. Does the copy pick up the concept? In language models this is subliminal learning. No paper found tests it in an image generator. The claim is bounded to that narrow case: a content-filtered dataset made with a fine-tuned checkpoint, used to fine-tune the same base. Pipelines that train a different or smaller student are out of scope. Every number is a prediction or estimate unless marked measured.

## What "hidden" means here

Images have no channel unrelated to content by construction. A teacher's other pictures will differ from the base's in colour, texture and statistics, and a student can learn that in the ordinary way. So "subliminal" is not the claim. The claim has two operational parts.

1. Init-matched transfer, identified by a crossed design. Two bases, A and A', trained from different seeds. Each gets its own teacher with the same concept fine-tune, tA and tA'. Each teacher's carriers train a student from both bases. The claim is an interaction: students pick up the concept more from the teacher that shares their init than from the teacher that does not. Comparing one teacher's carriers across two students cannot do this, because two generators can differ in sensitivity to the same visible shift. In the crossed design a student that is simply more sensitive gains from both teachers alike, and a carrier set that is simply more visible helps both students alike. Only a match between student and teacher init produces the interaction.
2. Detectability units. Each carrier set gets a detectability score D, measured under both bases: how well the base's own denoising loss, plus a pixel-statistics probe, tells it from that base's own carriers. Transfer is reported against D as seen by the student's own base. A decoy teacher that changes a different concept must not move the concept readout.
3. Sensitivity calibration. Each student init is also trained on base carriers with a planted visible shift and with real teacher samples mixed in. If A and A' differ in sensitivity on those, the crossed arms are reported both raw and per unit of each student's measured sensitivity.

The readout that scores the concept is never used to filter or match carriers.

What stays unidentified. Any init-matched signal is still carried in the pixels. A positive interaction shows a shift that a matched student reads and a mismatched one, calibrated for sensitivity, does not: invisible to the mismatched model and to D, not invisible in principle. The card claims init-matched transfer not explained by generic sensitivity or by D, and no more.

## Hypothesis

In a small class-conditional diffusion model, students trained by ordinary denoising loss on a teacher's concept-free samples pick up the concept more when they share the teacher's init. In the crossed design the interaction I, defined in Step 1, is positive, and the decoy stays inside its control's interval. The hypothesis is wrong if, at the largest carrier count, I stays inside its seed confidence interval.

## Theory

A fine-tune changes the weights a little everywhere, so the teacher's output on other classes shifts slightly. A same-init student fitting those outputs gets gradients partly along the teacher's weight change, because both compute in the same coordinates. A different-init student should move less along the concept. Finished images carry one noisy sample each, so signal-to-noise, not direction, is the open question; 2610.04907 predicts hard targets need far more carriers than soft ones.

## What is already known

Searched earlier on 2026-11-09: arXiv listings for "subliminal" (about 50 papers), Hugging Face papers, LessWrong, GitHub, the web. Mostly abstracts.

- [Cloud et al. (2507.14805)](https://arxiv.org/abs/2507.14805): the original result, including an MNIST classifier case; shared base required.
- [Subliminal Learning in VLMs (2611.08538)](https://arxiv.org/abs/2611.08538) (read in full; teachers emit numbers) and [Near-Duplicate Image Selection Transfers Preferences in VLMs (2610.27943)](https://arxiv.org/abs/2610.27943) (abstract; a same-base curator picks among near-duplicate photos and the student learns the preference): the closest image-side work. No student generates images.
- [Subliminal Learning Is Steering Vector Distillation (2606.00995)](https://arxiv.org/abs/2606.00995): transfer is a steering vector, and adaptive optimizers are necessary in language models.
- [Subliminal Learning is Non-Semantic Distillation (2608.05734)](https://arxiv.org/abs/2608.05734): weight noise raises transfer.
- [Subliminal Learning is a LoRA Artifact (2606.00831)](https://arxiv.org/abs/2606.00831) and [Regularization Amplifies Subliminal Learning (2611.10397)](https://arxiv.org/abs/2611.10397) report weak transfer under full fine-tuning. [Tracing Subliminal Learning Through Pretraining (2610.30884)](https://arxiv.org/abs/2610.30884) and [Subliminal Steering (2604.25783)](https://arxiv.org/abs/2604.25783) report the opposite. A live dispute.
- [Learning Through Noise (2605.23645)](https://arxiv.org/abs/2605.23645): on MNIST a shared initialization is not needed, so the crossed arms are a real test.
- [DriftScope (2607.00183)](https://arxiv.org/abs/2607.00183): a diffusion bleed instrument, not prior art.
- Radioactive watermarks are the nearest generator-student result. [Meintz et al. (2506.23731)](https://arxiv.org/abs/2506.23731) report sampler and decoder watermarks falling to about 1 TPR at 1% FPR in an SD 2.1 student; [HMark (2512.00094)](https://arxiv.org/abs/2512.00094) claims an h-space mark survives a LoRA student. Those marks are not a UNet weight change, so they bound this card without answering it.
- [Data-Chain Backdoor (2512.15769)](https://arxiv.org/abs/2512.15769): a backdoored generator's samples pass a pixel-visible trigger to downstream models, apparently classifiers.
- [2608.26958](https://arxiv.org/abs/2608.26958): more carriers recover the trait better, and off-target traits can grow too. [2608.20442](https://arxiv.org/abs/2608.20442): the Adam first moment carries the perturbation, including in MNIST CNNs. A [NeurIPS 2026 VLA paper](https://neurips.cc/virtual/2026/164357) finds transfer through continuous action trajectories.

Open: no paper found trains an image generator on a fine-tuned teacher's concept-free samples. Not scooped at the date of the search.

## Step 1: toy diffusion test on the Mac (the core of this card)

**Models.** A small class-conditional DDPM UNet on MNIST, trained from scratch with two seeds: base A and base A'. Epsilon prediction, one fixed sampler (DDIM, 50 steps) and one guidance scale for every sample in the card. Train in resumable chunks so no process passes 300 s.

**Teachers.** Fine-tune base A so that the label "3" draws an 8 (teacher tA), and base A' with the identical recipe (teacher tA'). LoRA on the UNet with Adam. Before any student runs, the two teachers' 3-to-8 scores must agree within 0.1; if not, extend the weaker teacher's training and record the steps.

**Decoy teacher.** Fine-tune base A so that "4" draws a 9. Same recipe. Its carriers are used only for the specificity control.

**Carriers.** Samples of classes 0, 1, 2, 5, 6, 7 from tA, tA', the decoy, base A and base A', with shared noise seeds within each base. Drop any that a filter classifier (a different architecture from the readout, trained separately) rates as 3, 4, 8 or 9 above a threshold set on real digits.

**Readout.** A held-out MNIST classifier, used for nothing else. Score = fraction of 1,000 label-3 samples it calls 8. For a student trained on a teacher's carriers, T = (student score − its control's score) / (that teacher's score − that teacher's base score). The control is the same student init trained on carriers from the teacher's base, which also subtracts drift from training on another model's samples.

**Arms.** Students are LoRAs with the teachers' rank, Adam with fresh optimizer state (never a teacher's, per 2608.20442), identical steps and data order, interleaved by seed. Every student is also scored on the 4-to-9 readout at every count, since 2608.26958 finds off-target traits can grow with data.
- Crossed core, four arms: A on tA carriers, A' on tA' carriers (matched); A on tA' carriers, A' on tA carriers (mismatched).
- Controls, four arms: A and A' each on base-A carriers and on base-A' carriers.
- Decoy: base A on decoy carriers, scored on the 3-to-8 readout.
- Positive arm: base A matches tA's epsilon on carrier classes. It shows the fine-tuning path can move the concept at all.
- Planted in data, both student inits: own-base carriers with 1%, 5% and 20% real label-3 samples from the matched teacher mixed in.
- Planted visible shift, both student inits: details in the appendix.

The test statistic is the interaction I = ½[(T(A←tA) + T(A'←tA')) − (T(A←tA') + T(A'←tA))]. The one-teacher gap T(A←tA) − T(A'←tA) is reported but is not the result.

Carrier counts: 1,000, 5,000 and 20,000. At least 10 seeds per arm; the final count comes from the noise floor measured in the first slice, chosen so the seed confidence interval on I is no wider than ±0.025. The interval on I is a seed-level bootstrap over the crossed arms and their controls together.

**Detectability.** Per carrier set and per base: that base's denoising-loss gap against fresh carriers from the teacher's base at fixed timesteps and noises (effect size), and a linear probe on pixel means, covariances and histograms (AUC). Report each crossed arm's T against D under the student's own base. If the matched arms also have higher student-side D, the deciding number is the matched term in a regression of T on student-side D.

**First slice (fits 30 minutes, 300 s processes, 8 GiB).** Train base A in chunks, train teacher tA, and write a measured table: seconds per training step, seconds per 1,000 samples, peak process memory, load time. Run the zero check and the scorer planted check. Nothing crossed is run in the first slice. Base A', tA', the decoy and the crossed arms follow. The crossed design adds a teacher, two arms, two controls and a second set of planted arms, so the earlier estimate of roughly 180 student runs no longer holds; it is recomputed from the measured table before they start.

## Predictions, written before any run

- Zero: two control seeds and base-against-itself read T inside ±0.02.
- Positive arm: T ≥ 0.2 at 5,000 carriers.
- Planted in data: T rises with the mixed fraction, in order, at all three fractions.
- Matched arms at 20,000 carriers: T between 0.02 and 0.10. At 1,000 carriers, inside noise.
- I above its seed confidence interval at 20,000 carriers. Low confidence; 2605.23645 points the other way.
- A and A' agree within a factor of 1.5 on planted sensitivity. Not checked; this is a guess.
- Decoy inside the control's interval.
- I = 0.05 outside its interval is the smallest effect that matters.

## Checks before believing anything

- Zero: base against itself with two sample seeds; two control seeds against each other. Both inside ±0.02.
- Planted effect in the scorer: score the teacher LoRA at scales 0 to 1 in steps of 0.05. A LoRA's effect is nonlinear in scale, so this is a curve, not a line. Every student's score is also reported as the equivalent scale read off that curve.
- Planted effect in the data: for both student inits, the arms with mixed-in teacher samples must recover the 1%, 5%, 20% ordering before any crossed arm is read. If either init's 5% arm reads inside noise, the pipeline cannot see the predicted effect and the crossed arms are not read.
- Symmetry: the two matched arms should agree in sign. If only one of them carries the interaction, it is reported as one-sided and not as a result.

## The control that could embarrass it

The mismatched arms transfer as much as the matched ones, so I sits at zero: the images carry the concept visibly enough for any student, and init-matched transfer fails. Or the matched arms win but only where student-side D is also higher, so visibility explains it. Or decoy carriers also move the 3-to-8 readout: generic drift from generated data.

## Confounds a hostile reader would raise

- "The carriers showed 8s." Separate filter classifier, manual look at the 50 carriers the readout rates most 8-like, and the readout never used for filtering.
- "Training on any generated data drifts the model." Per-arm controls and the decoy.
- "A' is just a worse learner," or "tA's carriers are just more visible." Each cancels in I; planted sensitivity and two-base D measure them directly.
- "The match is still visible change, just visible to the matched model." Conceded above: the claim is init-matched transfer, not invisibility.
- "One concept." Nothing is a result until the decoy recipe, as a second concept with its own crossed teachers and readout, repeats it.
- "MNIST says nothing about Stable Diffusion." Agreed; Step 1 only settles whether the channel exists in a generator at all.

## Kill criterion

The positive arm inside the control's interval: the pipeline is broken; fix before anything else. The positive arm works and I stays inside the interval at all counts: no init-matched channel through finished images at this scale, and the negative is written up. A positive one-teacher gap with I at zero is written up as ordinary generator-to-generator transfer. SD 1.5 is not touched.

## Step 2: Stable Diffusion 1.5, only if Step 1 is positive

Gate, checked 2026-11-09: `stable-diffusion-v1-5/stable-diffusion-v1-5` is live (an unaffiliated mirror, CreativeML OpenRAIL-M); fp16 UNet 1.72 GB, text encoder and VAE sizes not checked. First deliverable: the measured cost table, including whether an fp16 training step fits 8 GiB and load fits 300 s. Go/no-go, written now: proceed only if the 2,000-carrier arms fit in at most 20 attended slices.

On one base the crossed design is not available, so SD 1.5 alone claims only generator-to-generator transfer after content filtering; hidden against visible is not identified, and the write-up says so. Design: one teacher (a frog-shaped red teapot LoRA, 30 hand-curated base images captioned "a teapot"), a student against a same-init control at 500, 2,000 and the largest affordable count. Detectability describes the carriers without identifying the channel: per-channel VAE latent means and covariances, colour histograms and base denoising loss, with a planted-in-data curve of 1%, 5%, 20% real teacher teapots. The teapot filter is a fixed threshold on a calibrated teapot-against-no-teapot score, set on hand-labelled images. Readout: DINOv3 cosine to the teacher's teapot centroid, plus CLIP text-image score, plus a greyscale repeat to check the gap is not colour alone. At least 5 control seeds, with a seed-level interval and a per-image bootstrap.

A second base needs Rob's approval for cloud spend. SD 2.1 base is the candidate, but `stabilityai/stable-diffusion-2-1-base` returned HTTP 401 to an anonymous fetch on 2026-11-09, so it may be gated or withdrawn. The crossed design there needs the same teapot LoRA trained on SD 2.1 as a second teacher, with each teacher's own teacher-minus-base gap as its denominator. Without a second base, the init-matched result stays toy-only.

## Follow-up cards, triggered only by a positive core

The bits axis, the base-only auditor, the dispute arms (plain SGD and full fine-tune, which 2606.00995 suggests weaken transfer), crossed SD 2.1, a Qwen-Image 2.1 INT8 repeat with the repo trainer, and the permutation-aligned stranger.

## Cost on this hardware

Disk free was 33 GB on 2026-11-09; keep above 15 GB. No toy or SD speed is measured yet.
- Step 1: CPU or MPS, MNIST 11 MB, under 1 GB of samples, student-run count to be recomputed for the crossed design (not checked); slices set by the measured table.
- Step 2: about 2 GB download (fp16 UNet 1.72 GB measured on the hub), slice count from its own table, capped by the go/no-go rule.

## What a yes and a no change

A toy yes says an init-matched channel through finished images exists in a generator, beyond generic sensitivity and beyond what D sees. An SD 1.5 yes says content filtering a same-base fine-tune's outputs does not remove what it was tuned on, without saying through which channel. How this scales with size is a hypothesis for the Qwen-Image follow-up. A no with the positive arm working says finished samples are a much narrower channel than number strings.

## What it could reject, and what it leaves untested

It can reject an init-matched, concept-specific channel through finished samples in a small diffusion model. It cannot show that a signal is undetectable in principle. Untested: large models, full fine-tunes, other families, mixed real data, video and audio.

## Receipts to keep

Dated plan; base A and A', teachers tA and tA', decoy weights and logs; teacher step counts after equalisation; carrier seeds, filter scores, dropped lists; every student's config, seed and data order; raw per-sample readout scores for all arms and checks; D and probe AUC per set under each base; planted-sensitivity curves per init; cost tables.

## Review record

- Must-fix, circular bleed control: claim is now shared-init specificity plus transfer per unit of detectability; readout kept out of filtering.
- Must-fix, runnable steps missed the image question: Step 1 is a from-scratch MNIST diffusion toy with same- and different-init students.
- Must-fix, planted check assumed linearity: dense scale grid, equivalent-scale reporting, gating planted-in-data arm.
- Must-fix, low power and undefined cross-base T: 10+ toy seeds, 5+ SD control seeds, seed interval plus bootstrap; cross-base T defined.
- Should-fix: cut to the core with follow-ups split off; relabelled twin removed; threat model narrowed; measured cost tables and a written go/no-go before Step 2; availability gate, epsilon targets, one sampler.
- Minor: 2606.00995, 2610.27943 and 2608.05734 added; calibrated teapot filter, hand-curated teacher images, greyscale readout check.

The reviewer's own six searches were refused by the budget; it read three abstracts, none with a generator student.

2026-11-09, cross-family review (Sol, [queue review](../reviews/2026-11-09-queue-review-sol.md)):
- Must-fix, identification broken (same-init beating different-init does not rule out ordinary learning, since generators differ in sensitivity to the same visible shift). Chose to add a design element, because the toy already has two bases and one more teacher makes the comparison identifying at modest cost. Added a second teacher tA' on base A', a crossed teacher-init by student-init design with the interaction I as the test statistic (generic sensitivity and generic visibility cancel in it), per-init sensitivity calibration, D under both bases with a regression check, and teacher-strength equalisation.
- Same point, the residue the design cannot remove: claim narrowed to init-matched transfer not explained by generic sensitivity or D, with a paragraph saying the signal is still in the pixels. Step 2 on SD 1.5 alone narrowed to generator-to-generator transfer, hidden against visible not identified.
- Knock-on: first slice runs no comparison; student-run count marked not recomputed; predictions, kill criterion, controls, confounds, receipts and yes/no rewritten around I. Search-gate detail moved to an appendix.

## Appendix: planted visible shift

For each student init, take that init's own base carriers and apply a fixed one-pixel stroke dilation to 5%, 20% and 50% of them. Train students as in the crossed arms. Read each student's own samples on a stroke-width statistic (mean foreground pixel count per digit on carrier classes), never the concept readout. Sensitivity is the slope of that statistic against the dilated fraction. This shift is visible and has nothing to do with 3-to-8, so a difference between inits here measures generic sensitivity to a visible change. The planted-in-data arms measure the concept-specific counterpart. Crossed results are reported raw and divided by each init's concept-specific planted sensitivity at the 5% fraction.

## Appendix: search gate, 2026-11-09

First pass: no searches (4 refused by the shared budget) and seven direct fetches, including the abstracts of 2610.27943, 2611.08538 and 2605.23645, the SD 1.5 mirror, and SD 2.1 base (HTTP 401). Second pass: 12 searches, none refused, across diffusion and text-to-image subliminal learning, trait transfer through generated images, backdoors through generated data, self-consuming loops, bias inheritance, LoRA and DreamBooth leakage, watermark radioactivity and MNIST same-init, on arXiv, OpenReview, Hugging Face, alphaXiv, LessWrong and Semantic Scholar. Depth read: 2506.23731 abstract and experiment tables; [Are Watermarks for Diffusion Models Radioactive?](https://mlanthology.org/iclrw/2025/dubinski2025iclrw-watermarks) (ICLR 2025 workshop), 2512.15769, 2608.26958, 2608.20442 and [2609.23260](https://arxiv.org/abs/2609.23260) (a kernel account where shared init makes transfer aligned; no generators) and the VLA paper, abstracts only; 2512.00094, search summary only. Still open: full texts of HMark and Data-Chain Backdoor, and whether the radioactivity student shared the watermarking model's base, which the paper's HTML does not state.
