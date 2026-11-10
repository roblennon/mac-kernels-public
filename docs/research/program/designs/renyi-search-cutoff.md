# The search cutoff: which entropy sets the cost of language-model tree search (renyi-search-cutoff)

Status: queued, 2026-11-09. Rebuilt after a hostile review. Search gate passed on the second pass the same day: no scoop of the per-sentence predictor, but password research already has a sampled per-target rank estimator, now added as a second control.

Merged with "A compressor with no model in it": hashed-token recovery is the test bench, not a product claim.

When a language model guides an ordered search toward a target, how much work does the search do? The usual answer is perplexity: about 2 to the target's surprisal, divided by the pruning. Guessing theory says that holds only for targets the model finds typical. Real text is atypical for the model, and for such a target the number of sequences ranked above it follows a Renyi-type (Chernoff) count, computable from logits with no fitted parameter. Nobody seems to have checked this against measured search work on real text.

## Hypothesis

For each real-text sentence, a Chernoff estimate computed from one forward pass over that sentence (minimum over s of the sum over positions of log2 Z_t(s) − s log2 p_t(x_t), turned into a work count as in Theory) predicts the hash-averaged search work log2 W better than the perplexity rule. The exact rule is under Prediction. This is wrong if the two rules are within the noise floor of each other, or if the estimate misses by more than 3 bits.

No longer claimed: a population-mean cutoff k* on real text, which a finite corpus cannot observe at useful lengths. That X_1 exceeds cross-entropy S is near-certain by Jensen's inequality and is now only described.

## Theory

For a target x and any s in (0, 1], every sequence ranked above x has probability at least p(x), so there are at most (sum over y of p(y)^s) × p(x)^(−s) of them. At s = 1 this is 1/p(x), the perplexity rule. Smaller s helps when x is less probable than the model's typical draws: the mismatch case. A k-bit hash per (position, token) passes each wrong token with probability 2^−k and always passes the truth. So expected work for x sums over prefixes y with cost at most the truth's, each weighted by 2^(−k × positions where y differs from x).

Two ways to get the sum over y of p(y)^s for an autoregressive model:
- On-path estimate: the product over t of Z_t(s), the sum of p_t(v)^s over the vocabulary at the true path's contexts. One forward pass. It ignores the other contexts the search visits, so it is not a bound.
- Sampled value: draw y per step at temperature 1/s. The product of that proposal's normalisers along y has expectation exactly the sum over y of p(y)^s, off-path contexts included. Report its effective sample size.

The perplexity rule is the s = 1 member of the same family. The minimum over s can therefore lose only where the on-path estimate falls below the measured work. That tilts the comparison toward the estimate, which is why the hypothesis also carries an absolute-error condition.

For the model's own targets, mean work is dominated by rare targets. Draw targets at per-step temperature 2 and reweight each by p(x)/q(x) = product over t of p_t(x_t)^(1/2) Z_t(1/2). A naive sample mean is biased low; the design measures that bias on purpose.

## What is already known

Sources: [Arikan 1996](https://yoksis.bilkent.edu.tr/pdf/files/10.1109-18.481781.pdf) (read in full; guessing moments and the cutoff rate for memoryless sources), [Christiansen and Duffy 2012](https://arxiv.org/abs/1205.4135) (abstract; log guesswork obeys a large deviation principle with mean equal to Shannon entropy, moments track Renyi), [Salamatian et al. 2019](https://arxiv.org/abs/1907.00531) (read in full; i.i.d. mismatched rate only implicit, so a fixed-s closed form is a bound), [PassGPT](https://arxiv.org/abs/2306.01545) (no Renyi exponent seen), and [Schaeffer et al. 2025](https://arxiv.org/abs/2502.17578) (fitted tails, not from logits). Removed: "Li et al. 2024" (Shannon's guessing game with LLMs), whose ID 2407.02446 resolves to an unrelated paper; the real one is unidentified.

Closest from the Duffy and Medard groups: [Liu, Malak and Medard 2019](https://doi.org/10.1109/ITW44776.2019.8989008) apply the Markov guesswork large deviation principle to seq2seq translation as accuracy bounds: no measured work, no per-target estimate. [Mariona et al. 2023](https://arxiv.org/abs/2305.03850) treat finite-length mismatch, not language models.

Password research already estimates per-target rank by sampling. [Dell'Amico and Filippone, CCS 2015](https://www.eurecom.edu/publication/4711/download/rs-publi-4711.pdf) (estimator section read) draw n samples from the model and count each sample more probable than the target with weight 1/(n p(y)); the estimate is unbiased for the number of strings ranked above the target. Neural meters and PassGPT reuse it. [Liu, Blocki and Bai, IEEE S&P 2023](https://par.nsf.gov/biblio/10505354-confident-monte-carlo-rigorous-analysis-guessing-curves-probabilistic-password-models) (abstract) show it can be badly wrong for some targets and give confidence bounds. The ML memorisation literature uses rank by perplexity too ([Carlini et al. 2019, Secret Sharer](https://arxiv.org/abs/1802.08232), search snippet only), but over a small, uniform canary space. None of these predicts rank from one forward pass, uses Renyi or Chernoff terms, or measures hashed search work.

Adjacent, abstracts only: [Cao, Valiant and Liang, Nov 2025](https://arxiv.org/abs/2511.11966) on model entropy over generations versus log loss on human text, the gap behind X_1 − S; [a May 2026 power-distribution paper](https://arxiv.org/abs/2605.04542): local approximations miss sequence-level power without suffix information, which is secondary observation 3.

Open: a per-sentence, logit-only predictor of search work on real text, checked against measured work. Twelve keyword searches across information theory, LLM, password and inversion wording found nothing that does it.

## Prediction

Written before any run. One rule decides; the secondary observations below cannot be promoted later.

The deciding rule (step 2): on real text at n = 16, in both GPT-2 and SmolLM2-135M, the perplexity rule's median absolute error in log2 W minus the on-path estimate's is at least 1 bit. The bootstrap 90 percent interval over sentences must clear the seed-to-seed noise floor. The estimate's own median absolute error must be at most 3 bits. If the interval overlaps the floor, the result is inconclusive, not a no.

Secondary observations, in order of precedence, none of which can override the rule:
1. Markov arm: the importance-sampled mean recovers the exact mean within its interval; the naive mean falls below it by the amount the exact calculation predicts.
2. The perplexity rule's per-sentence error grows with the sentence's surprisal minus the model's entropy along the same path.
3. On-path and sampled values of the sum of p(y)^(1/2) agree within 1 bit at n = 16. This is the off-path question.
4. Descriptive: X_1 − S and the matched rate M per model.
5. Per-sentence predicted success rate under a node budget against the measured rate.
6. Sampled control (added 2026-11-09): median absolute error of the Monte Carlo estimate below, against the on-path estimate's.

## Design

Models: [GPT-2](https://huggingface.co/openai-community/gpt2), [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B). Download sizes from the first draft (548 MB, 269 MB, 988 MB) were not checked live. Text: 200 arXiv abstracts from October 2026, after every model's training cutoff, fetched with the arXiv API. Keep the ID list. No truncation arms in steps 1 and 2.

### Step 1: feasibility gate, one slice, no language-model search

It already reports the naive-mean bias against exact values and whether off-path contexts change the sum of p(y)^(1/2) (secondary 1 and 3).

Markov arm (numpy). V = 64, rows Dirichlet(0.3), n in {8, 16, 24}.
- Exact expected work by dynamic programming over (target, prefix) state pairs with a discretised cost difference. Halving the bucket width checks discretisation; brute force checks a 4-state chain.
- Matched arm: targets and search from chain P. Mismatched arm: targets from Q, search with P = Q tempered to T = 0.7, then Dirichlet-perturbed.
- Per arm and k: naive mean over 2,000 targets, importance-sampled mean over 2,000 tilted targets, exact value.
- Planted checks: importance sampling recovers the exact value; the naive mean shows the predicted low bias; measurements follow the exact matched-to-mismatched shift.
- Reference, not a planted check: a binary i.i.d. case with per-position hashes, its work exponent against Tavakoli 2026 (per-bit slope only; the hash is not a linear syndrome).

Language-model statistics, one process per model:
- Real text, 16k tokens: surprisal and log2 Z_t(s) for s on a grid of 20 values in (0, 1], at T in {0.7, 1.0, 1.3}. Run the forward pass in chunks of positions, keep float32 logits for one chunk at a time, and accumulate in float64.
- Tempered samples at T = 2, a few hundred sequences of length 16 per model: the product of normalisers, its effective sample size, and the sampled sum of p(y)^(1/2) at n = 8 and 16.
- Sampling runs in its own process per model. Whether it fits 300 s on CPU is not measured; if not, use MPS or fewer samples.

Outputs: per-sentence estimates and predicted log2 W for each k, the predicted work of tilted model targets, and effective sample sizes.

Gate to step 2, written now: the Markov planted checks pass, and for each model at n = 16 at least 80 percent of real-text sentences have a k with predicted work between 2^4 and a provisional budget of 2^16 nodes (revised by the step-2 probe). If not, cut to n = 8 and say so; if that fails, park. The model-target arm goes forward only if its effective sample size is at least 10 percent of samples at n = 16.

Room-for-effect gate, added 2026-11-10. The deciding rule needs the two predictions to differ: if the perplexity rule and the on-path estimate predict within d bits of each other on a sentence, their errors differ by at most d bits there, so a small gap forces a no without testing anything. From the step-1 statistics already computed (no extra passes), per model at n = 16 and each sentence's chosen k, take |predicted log2 W (on-path) − predicted log2 W (perplexity)|. Bar: median at least 2 bits and at least 60 percent of sentences at 1 bit or more. If a model fails, cut to n = 8 and recheck; if it still fails, park the card and report the gap as a descriptive finding (perplexity and the estimate agree on this text), not as a no. In the step-2 throughput probe, also require at most 20 percent of the 10 sentences censored at the budget and a seed-pair spread (median |log2 W, seeds 0–1 minus seeds 2–3|) of at most 0.5 bit; otherwise raise the budget or seed count within the slice, or cut to n = 8, before the full run. Cost: nothing beyond step 1 and the probe already planned.

### Step 2: search on real text, GPT-2 and SmolLM2 first

W is the number of prefixes with cost at most the truth's cost that pass the hash. It does not depend on expansion order, so batching is safe. Cap W at the node budget and record censoring.

- Throughput probe first: one model, one k, 10 sentences, two seeds. It sets the budget and slice count. Target 100 sentences per model at n = 16, never under 50. Whether that decides is not computed; report the interval.
- Each sentence's k comes from its step-1 prediction. Four hash seeds per sentence; measured value is log2 of the seed mean.
- Models alternate between processes; sentence order is shuffled with seed 0.
- The sampled control on the same sentences, k and seeds, in its own process.
- Model-sampled targets (if gated in): importance-sampled mean W at n = 8 and 16, against the sampled bound.
- Temperature is a secondary observation, run only in cells where |X_1 − M| is above twice the self-comparison floor from step 1.

### Step 3: one real task, abbreviation expansion

The constraint is the word-initial letters plus the word count. Use the per-sentence estimate restricted to consistent tokens. Decide in advance: absolute error of log2 W with a 3-bit tolerance, plus the partial correlation with log2 W controlling for sentence surprisal. Truncation enters only here: the statistic is computed over surviving positions, and the true token's miss rate is a separate failure probability.

### Step 4: needs Rob's approval

Qwen3.5-4B statistics, and a check against the Schaeffer et al. tails (whether their data are public is not checked).

## Checks before believing anything

- Self-comparison: hash seeds 0 and 1 against seeds 2 and 3 on the same sentences give the noise floor for log2 W. Two disjoint halves of the text give the floor for each statistic. Rerunning one cell must give identical W counts; if the MPS backend breaks that, use CPU float32.
- Planted effect: the Markov matched and mismatched arms, which have exact answers (step 1). Under importance sampling, the measured mean must recover the exact one within its interval.

## The control that could embarrass it

The perplexity rule. If it predicts per-sentence work as well as the Chernoff estimate, perplexity suffices.

A second control, from password research: the Dell'Amico and Filippone estimator, adapted. Per sentence, draw 256 sequences at temperature 1/s for that sentence's minimising s; at each prefix length, count samples whose cost is at most the truth's, weighted by importance weight, by 1/(n p(y)), and by 2^(−k × positions differing from the truth). It is unbiased for expected W but costs hundreds of forward passes. If it beats the one-pass estimate beyond the noise floor, the write-up calls the Chernoff estimate a cheap approximation. Report effective sample size. The deciding rule is unchanged.

## Confounds a hostile reader would raise

Memorised text shrinks ranks: post-cutoff abstracts. Tokenizers differ: conclusions are per model. Unused vocabulary slots inflate Z: report Z with and without them. Choosing k from the prediction: k is fixed before the search, and all rules are scored at the same k.

## Kill criterion

Park after step 1 if the gate fails at n = 8. Stop after step 2 if the perplexity rule is within the noise floor of the estimate in both models, or if the estimate misses by more than 3 bits while the Markov checks pass. A harness that fails the Markov check twice parks the card.

## Cost on this hardware

- Step 1: under 2 GB downloads, disk under 1 GB, CPU, one slice. Qwen logits over 16k positions are about 9 GB as float32, hence chunking. Per-process time not measured.
- Steps 2 and 3: slices set by the throughput probe.
- Step 4: Qwen3.5-4B (9.3 GB on disk) exceeds the 8 GiB process limit; it needs a quantized copy or an exception, and Rob's sign-off.

## What a yes or a no changes

A yes gives a one-pass, per-target estimate of search work that beats perplexity on atypical text, readable as success probability under a node budget. A no says perplexity suffices. A random hash is unlike real verifiers, whose errors correlate with the model's, so task claims wait for step 3.

## Carry to larger models, and what is left untested

Untested: models above 0.5B, natural verifiers, pruned beam search, lengths above 16, population means on real text. A finding stays unconfirmed until it recurs on fresh text and seeds.

## Receipts to keep

Committed plan, arXiv IDs, model revisions, per-position .npz statistics, Markov exact values, importance weights, raw W per sentence and seed with censoring flags, sampled-control estimates, check results, script versions.

## Search gate, 2026-11-09

First pass: 6 searches refused, 0 ran; about 20 citation fetches instead (forward citations of Christiansen and Duffy 2012, Liu, Malak and Medard 2019, and 2407.02446). It found, abstracts only: [Tavakoli, June 2026](https://arxiv.org/abs/2607.00205) (exact guesswork exponent inside a syndrome coset, matched and i.i.d.; now a step-1 reference), [Cohen, June 2026](https://arxiv.org/abs/2606.24537) (importance sampling via a guesswork rank exponent, i.i.d.) and [Christiansen et al. 2013](https://arxiv.org/abs/1301.6356).

Second pass: 12 searches ran, 0 refused. Wording covered guesswork and Renyi with LLM logits, best-first node counts, the guessing game, password and privacy guess numbers, hash recovery and model inversion, GRAND with a language prior, tempered sequence sampling, and checksum enumeration. New closest works and how much was read are in "What is already known". Judged distant: LLM inversion (Morris et al. ICLR 2024, SIPIT 2025), GRAND, best-k decoding, and [arXiv 2505.02072](https://arxiv.org/abs/2505.02072) on the guessing game, which names a Zouhar et al. 2024 paper that may be the lost "Li et al." reference. Not checked.

Still open: that Zouhar et al. 2024 paper; forward citations of Salamatian 2019 and Arikan 1996; whether any neural password meter reports per-target Monte Carlo error against true rank on long strings.

## Review record

Fixed after the hostile review: naive means missing the targets that set X_1 (per-sentence claim, importance sampling, exact Markov reference, gate on n); a partly foregone result (X_1 − S descriptive, absolute-error condition); several deciding rules (one rule, inconclusive band); no mismatched ground truth (mismatched Markov arm). Six should-fix and minor points were all taken: 20-point s grid, truncation only in step 3, temperature gated on |X_1 − M|, chunked logits with a throughput probe, partial correlation, payoff as success probability under a budget.

Zipf toys showed fixing s = 1/2 loses under 0.1 bit; naive means at n = 24 would need on the order of 2^37 targets. Scooping: see Search gate.

2026-11-10, base-task check (after copies-as-evidence step 1a ran at chance): the step-1 gate checked that predicted work fits the budget but not that the two rules predict differently enough for the 1-bit deciding rule to be passable. Added a room-for-effect gate on the predicted gap, plus censoring and seed-spread bars on the throughput probe. Whether real text clears the gap bar is not checked.
