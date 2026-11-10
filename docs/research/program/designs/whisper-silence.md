# What Whisper reads from its padding (whisper-silence)

Status: queued, 2026-11-09. Designed, not run; no code exists yet. Step 1 is 1a (pilot and checks, two slices) and 1b (the deciding run, several slices). "Step 1, exactly" is the run plan; where the rest of the card seems to differ, it governs. Merged from four ledger ideas.

Whisper pads every clip to 30 seconds and encodes the padding. Cut the padding and the decoder no longer knows when to stop. Three stories could explain this. The pad carries a clip-specific stop signal. Or the pad is an attention sink: its content does not matter, but removing its attention mass destabilises decoding. Or the pad does nothing the decoder needs. Replacing pad states, not only masking them, separates the three.

"Pad" means synthetic padding only: encoder positions whose input window lies wholly in the zeros the feature extractor appends after the recording. Silence recorded inside the clip is not pad. "Speech keys" means every non-pad position, the boundary band included (appendix A2).

## Hypothesis

For clips of 2 to 8 seconds, the presence of the pad states matters for stopping and their content does not. Replacing a clip's pad states with a fixed per-model mean (arm M) keeps the overrun rate within 2 points of baseline. Masking cross-attention to the pad (arm A2) raises it by at least 5 points.

## Theory and what is known

Reid's 2023 blog (Whisper tiny) found positions after speech attend back to its end, which predicts the first story. The sink story has a precedent in StreamingLLM. Unpadded failure and pad masking are known, and a cached pad tensor works in a retrained encoder (WhisperKit). Open, as far as searched: whether pad content or only presence matters on unchanged checkpoints, and whether a fixed tensor can stand in for pad states that have seen the speech. Sources: appendix A0.

Assumptions: the front end clamps the log-mel at each clip's maximum minus 8, so pad inputs differ by clip, which is why every input is peak-normalised first (Step 1). Masking renormalises the softmax, which is out of distribution, hence control C. Cosine between final-layer states is centred and descriptive only.

## Prediction (written before any run)

Committed, per model (small, turbo), on the 1b set, each scored held, wrong or undecided by the rules in "Step 1, exactly":

- Overrun levels: A0 at or below 2 percent; A2 rises at least 10 points on small and 5 on turbo; C rises at most half as much as A2.
- Equivalent to A0: M within ±2 points, with corpus WER change under 0.5 percentage points; Sw-sim within ±2; Sw-diff (5 to 8 s) within ±3, with median paired end-of-text step shift within ±2 tokens.
- Early stop: under 2 percent in A0, M, Sw-sim, Sw-diff and C. A0' is determinism only: at 50 clips its Wilson upper bound for 0/50 is 7.1 percent.

Exploratory: A3 output; centred cosine of pad against speech states by distance; log-mel maximum; pad weight at the end-of-text step.

What the outcomes mean. M and Sw-diff match baseline and A2 raises overruns more than C: the pad is a content-free sink a fixed tensor replaces on unchanged checkpoints, a short citable note, unconfirmed until step 2, saying nothing about skipping the encoder. Sw-diff raises overruns: a clip-specific stop signal, a smaller note. A2 changes nothing: a lab note. Untested: recorded silence, non-English, long-form, Distil-Whisper, other padded encoders.

## Design

- A0 baseline; A0', the same in-process, determinism only.
- M: each pad position replaced with the mean at that position over a fixed pool of clips for which it is also pad, disjoint from the pilot and 1b: one tensor per model, as a cache would be.
- Sw-sim: pad replaced from a donor 0 to 0.5 s shorter, so the donor's near-boundary pad enters but none of its speech.
- Sw-diff: donor at least 3 s shorter, so replaced positions sit 3 s or more into the donor's pad. 5 to 8 s stratum only.
- A2: cross-attention to every pad position masked.
- C: at every step, layer and head, the lowest-attended speech keys masked until the masked weight equals that row's pad weight (appendix A4).
- A3, exploratory: speech keys masked, pad kept.

M, Sw-sim, Sw-diff and A2 touch only the target's pad positions P, never the boundary band. C and A3 mask speech keys, which include the band.

C could embarrass the result: if removing the same weight from speech keys raises overruns as much as A2, the effect is generic softmax disruption. C removes at most half the speech weight, so rows with pad weight above a third are capped (call 3). C is never cut.

1a (40 pilot clips per model) settles code, timings, 1b's n and slices, C's feasibility and A2's informativeness. Hypothesis, predictions and kills are read in 1b only.

Sample size. TOST at 5 percent with 80 percent power needs n = (1.645 + 1.282)^2 d / 0.02^2, about 21,400 d for discordance d (normal approximation, my arithmetic): 600 clips cover d up to about 2.8 percent. Check 5 simulates the real power.

Later steps are in appendix A5; none runs before 1b is read.

## Step 1, exactly

This section is the plan. Commit the card before 1a; any later change is a dated amendment under "Amendments", written before the run it affects.

### Rob's calls, with defaults

1. 1b needs more 30-minute slices than this programme's four, so it belongs to the lab project. Default: run across slices, resuming through the JSONL. If Rob caps the slices, n shrinks to fit, under check 5's power rule.
2. A2 is uninformative on a model when 1a's mean per-row pad weight there is under 1 percent. That model leaves the A2 and C rows; if both models do, neither row is testable.
3. C's capping is a known limit: a control that removes less than A2 cannot embarrass it. Check 8 shows the capped share before 1b, when Rob can amend C.
4. Stated choices, not validated: the 2- and 3-point margins, the overrun-rule thresholds, M's 20-contributor floor, C's 10 percent infeasibility threshold.
5. Check 7's hand labels are Rob's, blind to arm. Default if he delegates: a fresh agent labels and Rob reviews 20 of its labels.

### Inputs

- `openai/whisper-small` at `973afd24965f72e36ca33b3055d56a652f456b4d` (80 mel bins, 12+12 layers; `model.safetensors` 966,995,080 bytes, fp32) and `openai/whisper-large-v3-turbo` at `41f01f3fe87f28c78e2fbf8b568835947dd65ed9` (128 mel bins, 32 encoder and 4 decoder layers; 1,617,824,864 bytes). Each uses its own feature extractor. Download and load per appendix A1.
- `openslr/librispeech_asr` at `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`, file `clean/test/0000.parquet` (350,452,636 bytes; columns `id`, `text`, `audio`). Test-clean has 2,620 utterances; counts per stratum are not checked.
- Software: the repository `.venv`, Python 3.12, transformers 5.18.0, torch 2.14.1, huggingface-hub 1.33.0, numpy 2.5.3, scipy 1.18.1. pyarrow, soundfile and jiwer are not installed; 1a installs current releases and records the versions in the receipt; none affects the result.

Code to write first (only `bench/gpu_runner.py` exists): `padarm` and its registration, selection, M pool, arms, overrun rules, statistics, simulation, shard driver.

### Decoding, features and metrics

Greedy, English, transcribe, no timestamps, no temperature fallback, no-speech threshold off, no conditioning on previous text, `max_new_tokens=200`. Every arm and timing: MPS, fp16, batch 1, through one registered attention function `padarm`, eager attention plus the arm's edit (appendix A1); A0 uses an empty edit. Every input (every arm, check 1's stock instance, M-pool contributors, donors, the CPU run) is decoded from FLAC to float with soundfile and peak-normalised before feature extraction: x times 10^(-1/20) over max |x|, skipped when max |x| is 0. The log-mel maximum is recorded per clip. One rule everywhere, because the front end's clamp ties pad inputs to the clip's level. Features come from the stock `WhisperFeatureExtractor` call (its torch path, CPU, fp32), then move to MPS as fp16.

The deciding metric is the overrun rate on speech. A decode overruns if it trips any of three rules:

- 200-token rule: 200 tokens generated after the four forced prompt tokens, end-of-text not counted. A text-level plant is retokenized with the model's tokenizer, no special tokens.
- Length rule: more than twice the reference word count plus 10 words.
- 5-gram rule: any 5-gram three or more times, overlapping occurrences counted (appendix A3). A clip whose reference trips it is excluded from this rule and counted.

Secondary: early stop (fewer than half the reference words); corpus WER (total errors over total reference words, so overruns do not dominate); end-of-text step.

### Decision rules

Paired overrun and early-stop differences, the C row's A2 minus C included, use Newcombe's hybrid score interval without continuity correction; single-arm rates use the Wilson score interval; WER differences, end-step shifts and the C prediction's per-clip x use a paired clip-level bootstrap, 10,000 resamples, seed 0. All per model. Equivalence and non-inferiority are supported by a 90% interval (one-sided tests at 5 percent), a rise by a 95% lower bound, and a kill needs the 95% interval wholly past the margin.

The floor is check 6's share of clips whose overrun flag differs between fresh-process A0 and A0, per model. A rise (A2 row, C row, A2 prediction) also needs its 95% lower bound above the floor. Equivalence ignores the floor, since a true null sits at or below it.

| Claim | Supported | Wrong (kill) |
|---|---|---|
| M equivalent (cache) | 90% interval of M minus A0 inside ±2 points, and 90% upper bound of the WER difference below 0.5 pp | 95% interval wholly above +2 points, or 95% WER-difference interval wholly above 0.5 pp, on either model. Wholly below −2 is reported, not a kill |
| Pad presence matters (A2) | 95% lower bound of A2 minus A0 at least 5 points and above the floor, on every informative model | 95% upper bound below 5 points on every informative model |
| No clip-specific signal (Sw-diff) | 90% interval inside ±3 points | 95% interval wholly above +3 points on either model: the stop-signal story holds |
| Sink, not generic disruption (C) | 95% lower bound of A2 minus C above zero and above the floor, on every model informative under call 2 and feasible under check 8 | 95% interval of A2 minus C wholly below zero on every such model (C disrupts more than pad removal) |

The M row is read on all targets and on targets with at least 95 percent of P replaced (check 10): supported only if supported on both, wrong if wrong on either, since unreplaced positions keep the clip's content and the restricted set drops short targets, and each can flatter the claim.

Anything else is undecided. Called inconclusive, not wrong: a rise whose lower bound clears its margin but not the floor, and an A2 minus C interval that includes zero, since both sit inside the baseline's own spread. The hypothesis holds only if the M and A2 rows are both supported, and is wrong if either is wrong. Before 1a, rerun the A7 search and log it; pause for Rob only if a paper replaces (not masks) pad states on unchanged Whisper checkpoints and reports stopping, the claim this card would add.

Prediction scoring, per model, apart from the verdict. Rises and equivalences follow the table, with A2 at 10 and 5 points and Sw-sim at ±2. Levels (baseline, early stop): held if the Wilson 95% upper bound is at most 2 percent, wrong if the lower bound exceeds it. C at most half of A2: per clip x = (C − A0) − (A2 − A0)/2 on the overrun flags; held if the bootstrap 95% upper bound of mean x is at most zero, wrong if the lower bound exceeds zero. End-step shift: median per-clip Sw-diff minus A0 where both stop; held if its bootstrap 95% interval lies inside ±2 tokens, wrong if wholly outside.

### Selection and seeds

Each generator is a `numpy.random.default_rng(k)`, created once and consumed in the order named. Draws take a prefix of `rng.permutation` over an `id`-sorted list; a donor is `rng.integers` over an `id`-sorted candidate list, with no draw for a target without candidates. Strata are [2.0, 5.0) and [5.0, 8.0) s, duration being decoded samples over 16,000; "eligible" means in a stratum. Selection is model-independent and saved before any decode.

| Seed | Use |
|---|---|
| 1 | Drawn first. Pilot: one generator, stratum [2, 5) then [5, 8); first 20 permuted eligible clips each. |
| 0 | 1b: one generator, same order; first 300 of each stratum's permuted remainder, or all if fewer. If n shrinks, keep the first n/2 per stratum. |
| 2 | Planted kind-1 subset: 8 of 40 pilot clips; a fresh generator for 1b, 30 of 600. |
| 3 | Donors: one generator, passes in this order, targets in sorted-`id` order within each: check 3, Sw-sim pilot, Sw-diff pilot, Sw-sim 1b, Sw-diff 1b. |
| 6 | Arm order: one generator per phase, clips in sorted-`id` order, each clip permuting only the arms it runs in that phase, listed as A0, A0', M, Sw-sim, Sw-diff, A2, C, A3. Same order on both models; check items run outside it. |
| 7 | M pool subset, only if the build must shrink (A2). |
| 8 | Text plants: one generator per phase, consumed as kind-2 subset, kind-3 subset from the rest, then kind-3 words clip by clip in sorted-`id` order. |
| 9 | Coverage and power simulation. |
| 10 | Check 7's random unflagged decodes, over (clip, arm) sorted by `id` then arm in the seed-6 listed order, excluding the 50 highest-ratio decodes. |

1b's smaller subsets are the first clips per stratum in seed-0 order: 25 each for A0' and A3, 50 each for fresh-process A0. Seeds 4 and 5 belong to 1c.

### Runs

1a, per model: A0, A0', M, Sw-sim, A2, C and A3 on every pilot clip, plus Sw-diff on the 5 to 8 s ones. Clip-major: encode once, arms in seed-6 order, `torch.mps.synchronize()` around each timed item, encoder and decoder seconds apart. Order: (1) hook unit test, the 8 planted kind-1 decodes, and one planted C decode (the kind-1 clip with the lowest sorted `id`, under C with end-of-text suppressed, timing only), so the worst case is timed early; (2) M pool build; (3) identity checks; (4) the arms; (5) replacement bites; (6) fresh-process A0, in a later shard, re-encoding from audio. After the slice: A0 through `padarm` on CPU in fp32, first 10 pilot clips by sorted `id`, same runner command.

1b, per model, default 600 clips (300 per stratum): A0, M, Sw-sim, A2 and C on all; Sw-diff on the 300 of 5 to 8 s; A0' and A3 on 50; fresh-process A0 on 100; planted kind-1 on 30. 3,530 decodes per model. Checks 4 and 6 are repeated on 1b's data before any arm is read.

### Budget

Lab limits: 300 s per process including load, 8 GiB, 30 GPU-minutes per slice, free disk above 15 GiB. The runner refuses when free disk falls below `--min-disk-free-gib` plus the unused swap-growth allowance (4 GiB by default; `memory_stop_reason` in `bench/gpu_runner.py`), so the working floor is about 19 GiB at admission and throughout. Check the process list before each shard, then run `python3 bench/gpu_runner.py --monitor --seconds 290 --rss-gib 8 --min-disk-free-gib 15 -- .venv/bin/python <shard script>` (the same GPU lock as `bench/with_gpu.sh`, which cannot pass `--monitor`; 290 leaves the runner's 5-second grace).

The shard reads a monotonic clock at the top of its script and adds 3 s for interpreter start. It stops taking items once elapsed time plus 1.5 times the worst item time so far for this model (from the JSONL, across shards and phases) would pass 260 s; before any item is timed, the margin is 60 s. Each item carries a wall-clock `StoppingCriteria` at its timeout (A6), measured from the item's start. An item killed by the runner's deadline writes no line and is rerun by the next shard; it is not a timeout, which would drop slow overruns preferentially. An item killed twice stops the run until the margin is re-planned by amendment. A slice is at most 6 shards. 1a is two slices, 1a-small then 1a-turbo; leftover items continue in a follow-on slice, recorded as an amendment.

### Checks, with thresholds

Checks 1 to 5 and 11 are go or no-go for 1b.

1. Identity, token for token. A0 through `padarm` equals stock `generate` from a second, eager instance in the same process on all 40 (the hook, not cross-process determinism). On the first 10 pilot clips by sorted `id`, two items must reproduce A0: the masked code path with an empty key set (the bias tensor built and added, all zeros), which A0 skips; and the clip's own pad swapped back. A0' equals A0 on all 40.
2. Hook bites. In every masked decode, logged weight on masked keys is at most 1e-6 per row, with no NaN or infinity. A unit test masking all keys but one returns that key's value. For M and the swaps, the encoder output passed to `generate` differs from A0's at the edited positions and is bit-equal elsewhere.
3. Replacement bites. The first 10 pilot clips of the 2 to 5 s stratum by sorted `id` with a donor at least 3.1 s longer (seed 3), or all if fewer, get their first 150 pad positions replaced with the donor's states there, which are donor speech. Pass: all but one transcript change.
4. Planted effect, a pipeline check. Kind 1: `SuppressTokensLogitsProcessor([50257], device="mps")` on the seed-2 subset, each a 200-token decode; Whisper's own suppress processors still apply. Kind 2 and 3 candidates are unflagged A0 transcripts of at least five words whose reference does not trip the 5-gram rule, so no plant fails its "only" test through the clip. Kind 2: on a seed-8 subset of 10 candidates (30 in 1b), append the last five normalised words twice. Kind 3: on another 10 (30), append, in seed-8 permutation order, distinct normalised words from the phase's other references (`id`-sorted) absent from the transcript, until the length rule trips, under 200 tokens. Pass: each kind-1 plant trips at least the 200-token rule (others not required: suppressed decodes may emit symbols the normaliser strips); each kind-2 plant trips the 5-gram rule only; each kind-3 plant trips the length rule only. On the pseudo-arm (A0 with kind-1 decodes swapped in), the interval code returns the planted shift, not equivalence. Natural overruns are check 7's job.
5. Interval code and power. First reproduce a published worked example of the paired score interval: Newcombe 1998 (Stat Med 17:2635) if available, else Fagerland, Lydersen and Laake's review (Stat Med, 2014), else an independent implementation; the receipt names the source and method number. Then simulate 2,000 paired datasets per cell at the planned 1b n from a four-cell multinomial: for true difference δ and discordance d, p10 = (d + δ)/2 (arm overruns, A0 does not) and p01 = (d − δ)/2. d has a floor of 3 percent so both stay positive at δ = ±2, and the A0 rate is the largest of 1a's A0 rate, 3 percent and (d + 2)/2 points, so no cell is negative.
   - Coverage: δ in {0, +2, −2}, d = the largest 1a discordance against A0 among M, Sw-sim, Sw-diff, A2 and C; A3 is reported apart. Pass: at least 88 percent for the 90% interval and 93 percent for the 95% interval at every δ; over-coverage is reported.
   - Power: δ = 0, d = the larger of M's and Sw-sim's 1a discordance; power is the share of 90% intervals inside ±2. Pass: at least 80 percent at the planned n. From 50 to 80 percent 1b runs, and an unsupported M or Sw-sim row is reported as underpowered; under 50, 1b is parked. Default, Rob's to change; n cannot simply grow, since the M pool takes the remaining short clips. Sw-diff's power at ±3 is reported, not gated.

Checks 6 to 10 feed 1b:

6. Floor: fresh-process A0 against A0, overrun discordance and token disagreement. CPU fp32 is backend sensitivity, not the floor.
7. Overrun rules. A true overrun is text not in the audio (looping, or invention past the reference), not a long correct decode. Label every flagged decode, the 50 unflagged A2, M and swap decodes with the highest hypothesis-to-reference word ratio, and 25 random unflagged ones from the rest (seed 10). Revise the rules on saved transcripts before 1b if flagged precision is under 0.9 or 3 or more of the 75 unflagged are true overruns. Recall is not estimated.
8. C feasibility: capped share per model from C's own decodes (A4); over 10 percent makes C infeasible there.
9. Pad mass: mean and median per-row pad weight in A0, overall and per layer and head; Rob's call 2 applies.
10. M coverage: per stratum, the share of each target's first 25 pad positions that M replaced, and the count of targets with at least 95 percent of P replaced.
11. Base task in reach (go or no-go; tests the model, not the instrument). Per model, from 1a's 40 A0 decodes, read before any other arm's numbers: corpus WER at most 10 percent, at most 2 of 40 decodes flagged by any overrun rule, at most 2 of 40 early stops. The bars are stated choices; published test-clean WER for these checkpoints is not checked. Check 1 cannot catch a front-end fault, since stock `generate` receives the same features; this check can. On a miss, compare the 10 CPU fp32 decodes: a pipeline fault is fixed by amendment and 1a's A0 rerun; a model that still misses is parked, since a baseline that already overruns or stops early leaves no room for M's ±2 or A2's rise to show. Both parked: 1b does not run. Cost: none beyond 1a's planned decodes; WER and flags are CPU arithmetic on saved transcripts.

### Outputs and receipts

Code goes in `bench/` (`padarm` and the shard driver); everything else under `docs/research/program/raw/whisper-silence/`. Receipts: the committed card; step 0 query; versions; clip lists, M pool, donor pairs, seeds; one fsynced JSONL line per (clip, arm), keyed by model revision, clip id and arm, with tokens, text, timings, flags and memory; attention-weight logs; final-layer states; each check's result; hand labels; intervals and simulations; memory peaks (`torch.mps.driver_allocated_memory()` and RSS after load, every 25 items, at exit); and the 1b plan note: n, simulated power, slices (projected GPU-minutes over 28).

## Cost on this hardware

M1 Max, 32 GB; free disk 32 GiB on 2026-11-09. Decodes per model: 1a 419 on GPU plus 10 on CPU, 1b 3,530, 1c 558, plus encoder passes for the M pool and donors. Seconds per decode are unmeasured; 1a sets 1b's slice count. Downloads: about 2.6 GB of models and 350 MB of LibriSpeech; about 0.9 GB of ESC-50 for 1c; 3.09 GB for 1L. Final-layer fp16 states are saved for A0 only, one per clip and model: about 3.7 GB for 1b's 600 clips across both models (600 × 1500 × (768 + 1280) × 2 bytes, my arithmetic), 0.25 GB more for the pilot.

## Amendments

None yet.

## Appendix: implementation detail

### A0. Prior work, as read

- [WhisperKit, 2507.10860](https://arxiv.org/abs/2507.10860), section 2.1.1. Self-distils large-v3-turbo's encoder with block-diagonal attention over 15-second blocks, then computes a wholly zero-padded block's output once and reuses it ("silence caching"), at +0.32 WER on test-clean. The cached block never sees the speech, and the encoder is retrained.
- [StreamingLLM, 2309.17453](https://arxiv.org/abs/2309.17453), abstract. Attention sinks in language models; background only.
- [NPUsper, 2607.01108](https://arxiv.org/abs/2607.01108), to Appendix D. Unpadded LibriSpeech hallucination: base 34.4 percent, small 24.0, medium 36.4, against 0.0 padded (Tables 1 and 3). D.4 masks pad columns with a finite -100 bias on one utterance; no replacement.
- [Moonshine, 2410.15608](https://arxiv.org/abs/2410.15608), section 2. Tiny.en, test-clean, beam 5: padded 5.21 percent WER; unpadded with the prefix of the position embedding 107.38, with the suffix 18.45.
- [Simul-Whisper, 2406.10052](https://arxiv.org/abs/2406.10052) uses the speech-to-pad frame as a boundary cue; [WhisperFlow, 2412.11272](https://arxiv.org/abs/2412.11272) trains a hush word to replace padding; [FUTO whisper-acft](https://github.com/futo-org/whisper-acft) and [whisper.cpp issue 137](https://github.com/ggerganov/whisper.cpp/issues/137) report repeats with cut audio context. None tests pad content.

Nearby: [Reid's blog](https://er537.github.io/blog/2023/09/05/whisper_interpretability.html); [Beyond Transcription, 2508.15882](https://arxiv.org/abs/2508.15882) (large-v3 repetition sits in cross-attention; layer 18 head 13 alone suppresses 78 percent); [2604.21276](https://arxiv.org/abs/2604.21276); [2609.28739](https://arxiv.org/abs/2609.28739); [2608.15940](https://arxiv.org/abs/2608.15940); [2610.22395](https://arxiv.org/abs/2610.22395); [2505.12969](https://arxiv.org/abs/2505.12969); [2606.07473](https://arxiv.org/abs/2606.07473); [2608.30927](https://arxiv.org/abs/2608.30927); [2506.15912](https://arxiv.org/abs/2506.15912); [2501.11378](https://arxiv.org/abs/2501.11378).

### A1. Download, attention backend and memory

Download with `huggingface_hub.snapshot_download(revision=..., allow_patterns=["*.json", "*.txt", "model.safetensors"])`, about 2.59 GB for both. Configs are cached locally; weights are not. Load with `dtype=torch.float16`. End-of-text is 50257 in both generation configs.

Read in transformers 5.18.0, not run: `WhisperDecoder.forward` passes `encoder_attention_mask=None`. `WhisperAttention.forward` pre-scales the query and calls the `config._attn_implementation` function with `scaling=1.0`; eager adds `attention_mask` to the scores and applies softmax in the input dtype. `generate` accepts `encoder_outputs`.

The mask trap: `masking_utils._preprocess_mask_arguments` returns no causal mask for an `_attn_implementation` missing from `AttentionMaskInterface._global_mapping`. So setup registers `AttentionInterface.register("padarm", padarm)` and `AttentionMaskInterface.register("padarm", eager_mask)`; without the second, the prompt pass loses its causal mask and check 1 fails.

On modules tagged at setup (each `model.model.decoder.layers[i].encoder_attn`), `padarm` adds the arm's bias: shape (1, 1, 1, 1500) for A2 and A3, (1, H, T, 1500) for C, built per call; -1e4 in fp16 at masked keys, 0 elsewhere: no overflow, exactly zero after softmax. No row is wholly masked. Before the bias it computes the fp32 softmax of the unmasked scores and logs pad and band weight per step, layer and head (masked arms also log residual masked weight). The encoder runs untagged; the key and value cache is unaffected.

Memory: `torch.mps.set_per_process_memory_fraction((8 GiB − RSS after load) / torch.mps.recommended_max_memory())` caps MPS allocations, which RSS may not show, so the two together stay inside 8 GiB. Two fp16 instances (about 3.2 GB) fit for check 1.

### A2. Pad positions, pools and the energy detector

Pad positions. Hop 160 samples, centred 400-sample window and a stride-2 second convolution give position j 20 ms and an input window spanning samples 320j - 520 to 320j + 520 (my arithmetic from the configs). For N samples, P = { j : 320j - 520 >= N, j < 1500 }. The boundary band is the positions outside P with 320j + 520 > N.

M pool: every test-clean utterance under 8.0 s (under 2 s included, since contributors at a target's first pad positions must be no longer than it) outside the pilot and the default 1b draw; it does not change if n shrinks. Each model encodes the pool once, in sorted-`id` order, keeping an fp32 sum and count per position over contributors for which it is pad; positions with fewer than 20 contributors stay unreplaced; the mean is cast to fp16 once. 1a prints pool size and contributors per position before any M decode. After each shard the fp32 sum and count tensors and the list of done contributor ids are written atomically (temporary file, then rename), so the build resumes across shards.

If the first shard's mean encoder seconds per clip times pool size exceeds 8 GPU-minutes, the pool shrinks to the largest that fits: every clip under 2.5 s plus a seed-7 permutation prefix of the rest, built afresh so M depends only on the saved list. If the clips under 2.5 s alone exceed 8 minutes, stop and amend.

Donors: eligible clips other than the target meeting the length rule (Sw-sim 0 to 0.5 s shorter; Sw-diff at least 3 s shorter; check 3 at least 3.1 s longer), re-encoded at each use. A target with none is dropped from that arm and logged.

Energy detector, only for the end-of-speech label and distance analysis. Frame j spans samples 320j to 320j + 320; a frame is speech if its RMS exceeds max(10th-percentile frame + 15 dB, loudest frame - 40 dB); end of speech is one past the last run of 3 or more speech frames. Validated after 1b against pyannote segmentation-3.0 (local cache, separate venv) on 50 clips: at least 90 percent within 100 ms, else Rob inspects 20 spectrograms.

### A3. Words

Normalise hypothesis and reference with transformers' `EnglishTextNormalizer` from the pinned small repo's `normalizer.json`, for both models. Words are whitespace tokens after normalisation. 5-grams are counted at every start, so "a a a a a a a" holds three occurrences of one 5-gram.

### A4. Control C, confounds and analysis

Per row (step, layer, head), with unmasked fp32 weights on C's own trajectory: pad weight p, speech weight s = 1 - p, target t = min(p, s/2). Sort speech keys by ascending weight, the closest analogue of diffuse pad weight, and mask the shortest prefix whose sum reaches t. Rows with p > 1/3 are capped. The capped share is the sum of (p - t) over the sum of p, over every row of C's decodes on that model; it is check 8's only definition, since only C's logs give t. A C overrun, with its extra softmax and sort per row, is the slowest item; 1a's planted C decode times it.

Other confounds: loudness (peak normalisation, step 3); LibriSpeech in training data (shifts levels, not paired differences); trailing recorded silence (both distances logged).

A3 is exploratory: the encoder is bidirectional, so words recovered from pad alone are words the pad copied. Centred cosine is reported by distance bin (0 to 0.5 s, 0.5 to 1, 1 to 2, 2 to 5, over 5 s).

### A5. Later steps

- **1c, non-speech, after 1b; decides nothing here.** ESC-50 at commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`, classes 20 to 29 excluded, clips with an internal digital-zero run over 100 ms dropped, trailing zeros trimmed, 4 per class (160) by seed 4, resampled to 16 kHz (`scipy.signal.resample_poly`), peak-normalised; 25 pink-noise clips (seed 5, 2 to 8 s, -45 dBFS); one digital-zero input. A0, M, A2; invented-text rate; 558 decodes per model.
- **1L, one slice.** `openai/whisper-large-v3` at `06f233fe06e710322aca913c1bc4249a0d71fce1` on 200 1b clips: A0, A2, M, Sw-diff, since turbo's 4-layer decoder may stop differently; pad mass at layer 18 head 13. If it fails, the claim is scoped to small and turbo.
- **2, truncation.** State drift for speech plus k pad frames (k in 0, 8, 25, 50, 150), then spliced cached pad states; 1,000 clips, new seeds.
- **3, CPU.** Front-end clamp scope on long recordings.
- **4, cloud only with Rob's approval.** whisper.cpp q8_0, q5_0 and q4_0 through this repo's GGUF decode on the 1c set, A0, M, A2. Drop the arm if the q4_0 to fp16 invented-text ratio is under 1.2 with an interval including 1.

### A6. Counts and timeouts

1a GPU decodes per model: 7 arms on 40 clips plus Sw-diff on 20 (300), stock-eager identity 40, fresh-process A0 40, empty-key masked path 10, swap-back 10, replacement bites 10, planted kind-1 8, planted C 1: 419, plus 10 on CPU.

In 1a, any item over 60 s is a timeout. In 1b it is the larger of 60 s and twice the longest 1a item for that model, so a real 200-token decode is never one. A planted decode timing out in 1a stops that model until the budget is amended. Any other timeout is retried once in a fresh shard; a second drops the clip from every arm, and over 1 percent dropped invalidates the run. Deadline kills follow Budget.

### A7. Search gate, 2026-11-09

Step 0 is done: twelve queries (padding sinks, `audio_ctx` repetition, training-free truncation, cached encoder output, dynamic padding, trailing silence, sliced position embeddings) plus six fetches. It missed WhisperKit, which the cross-family review supplied. Repeat one arXiv search for "padding" plus "attention sink" in Whisper on the day 1a runs. Still open: the whisper.cpp and faster-whisper trackers and OpenReview.

## Review record

- 2026-11-09: hostile review, cross-family review (Sol, `reviews/2026-11-09-queue-review-sol.md`, section 1), cold planning pass and cold read round 2; all must-fix items addressed.
- Cold read round 3, 2026-11-09: pass rules, floor, C criterion, interval levels, prediction scoring, seeds, M-pool shrink and reading, deadline kills fixed.
- Cold read round 4, 2026-11-09: peak normalisation moved into Step 1; check 5 power given a pass line; C row on Newcombe; C kill needs the interval wholly below zero, with inconclusive outcomes named; plant candidates, coverage d, early-stop arms, orderings, file layout, MPS cap and prior-work pause rule set; disk floor (about 19 GiB), check 1's masked path, kind-1 claim, state size and 1c count corrected; repetition cut.

- 2026-11-10: subject-property gate review, after copies-as-evidence 1a ran a model that got no greedy base-task answer right (one template inaccurate, the other off-format). Pad mass (call 2, check 9, 1 percent bar) and pad-content sensitivity (check 3) were already gated; base-task competence at A0 was not. Added check 11 (WER, overrun and early-stop bars on 1a's A0, go or no-go, no extra decodes). Risk the test would have been void without it: low, since Whisper on test-clean is expected to be in reach, but a front-end fault would have passed checks 1 to 5.

Not done in any pass: no command was run against a model, and every timing remains unmeasured.
