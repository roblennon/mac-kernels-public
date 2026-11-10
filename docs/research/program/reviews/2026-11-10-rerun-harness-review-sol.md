**Not yet.** The timing design is substantially better, but the correctness gate and one planted control need fixing before this run.

**1) Blocking defects**

- **Failed checks do not stop the run.** [rerun_all.sh:15](../../../../bench/rerun_all.sh:15) prints the child’s exit status, then returns the successful `echo` status. Neither correctness script exits unsuccessfully for a failed verdict, and nobody checks that ordinary controls pass and planted controls fail. [rerun_analyze.py:35](../../../../bench/rerun_analyze.py:35) also accepts missing processes. A crash, failed equivalence check, or missing control can therefore produce a partial summary followed by “ALL DONE.” Require expected JSON verdicts and complete results before accepting timings.

- **The `skip_reset` control overruns its cache.** [check_spec_rollback.py:40](../../../../bench/check_spec_rollback.py:40) allocates `P+82` slots. Without resets, the schedule writes 108 tokens; zero-based round 18 reaches `P+83`. Transformers writes using that live counter, so this control eventually crashes before writing its verdict. It must stop after detecting corruption or have capacity for the faulty execution.

- **Acceptance lengths are proposed, not forced or enforced.** [check_spec_rollback.py:68](../../../../bench/check_spec_rollback.py:68) drafts from the one-row reference, but numerical differences can shorten actual acceptance. The verdict at [line 91](../../../../bench/check_spec_rollback.py:91) ignores planned-versus-actual acceptance and required coverage. It can pass without exercising all lengths or intended transitions.

**2) Nonblocking weaknesses**

- **The “95%” intervals overstate certainty.** [rerun_analyze.py:11](../../../../bench/rerun_analyze.py:11) bootstraps four process medians. With four distinct observations, its endpoints reduce to the observed minimum and maximum; that range covers a continuous population median only **87.5%** of the time under independent sampling. Verification A/A cells each have just one process and receive zero-width intervals. Report these as descriptive ranges, not dependable 95% coverage.

- **Rollback tolerance can absorb shared faults.** [check_spec_rollback.py:59](../../../../bench/check_spec_rollback.py:59) calibrates from every third plain position, using the candidate path itself, without an upper bound or reference-finiteness check. This is a relative smoke test. The `0.5` margin allowance also permits unequal greedy tokens. I found no static evidence that the `0.1` floor necessarily rejects legitimate fp16 results.

- **The numerical acceptance rule supports aggregate comparability.** [check_decode_equivalence.py:60](../../../../bench/check_decode_equivalence.py:60) genuinely enforces the declared rule, but can accept two high-margin token errors: sparse errors barely affect vocabulary-wide mean/p99.9, and maximum error is unjudged. Candidate margins are not recorded. Keep the verdict narrower than greedy equivalence; correctness contexts remain only 32 tokens.

- **Rotation balances position, not every carryover effect.** [rerun_all.sh:30](../../../../bench/rerun_all.sh:30) balances average arm positions, but compilation heat, allocator state and nonlinear drift remain. Small gains need the A/A evidence. Also, [rerun_analyze.py:53](../../../../bench/rerun_analyze.py:53) ignores repeat-token consistency and EOS flags, so its whole-request table alone cannot establish completed, matching answers.

**3) Checked and sound**

- The ordinary compiled speculative baseline avoids the earlier redundant rollback writes.
- Rollback position arithmetic, rejected-slot boundaries and in-place counter resets match installed Transformers 5.18 behavior.
- The visible-sentinel control survives into the next one-token call. Five-token calls overwrite previous rejected sentinels, so their presence alone is not evidence of visibility testing.
- Kernel routing matches the labels; batch 1 bypasses it. Verification timing synchronizes correctly and resets/refills outside timed samples.
- Ratios are formed within processes before aggregation; timed samples are retained without selecting the fastest pass.
- CPU tokenization puts the renamed code near **852 tokens**, so the 1,024-token cap is sufficient for the requested source.

Read-only review; no GPU work or installations.
