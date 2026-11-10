# How do Apple's stock GEMMs add? A summation-order and accumulator census on the M1 Max

Status: queued, 2026-11-09. Search gate passed (Appendix C). Revised after a hostile review and four cold reads the same day. Designed, not run. Merges "Exact integers inside float GEMMs" and "What fp16 arithmetic means on each stack"; the divergence law gets its own card, not yet written.

Published models of matrix arithmetic cover NVIDIA and AMD, not Apple. This card asks, on the M1 Max: when PyTorch MPS, MLX and Accelerate compute a fp32 or fp16 GEMM, in what order do they add, is the accumulator inside a block wider than fp32, and how do they round? A measured description of Apple GPU GEMM arithmetic is the part with citation value.

## Hypothesis

A small set of adversarial integer-valued probes identifies, for each cell, one class of models from a written list, and that class predicts held-out probes it was not fitted on. A model is an outer summation structure, an inner step, an accumulator that is fp32 or wider than fp32 inside a block, and a rounding mode. A cell is (backend route, version, dtype, M, K, N, call form). The hypothesis fails for a cell if no listed model matches its fit probes, or if no member of the fitted class matches every held-out probe.

Step 1 can only say "wider than fp32": wide enough to hold 2^24 + 1 (sliding) and 2^24 + 3 (rounding) exactly, so at least 25 significand bits. How much wider is a step 2 question.

## Theory

fp32 holds every integer up to 2^24 exactly, so integer-valued probes make each rounding event visible as a small integer error. Every product in the probes is exactly representable in fp32 by construction (the generator asserts it), so every rounding event is in a sum. Worked values below were simulated in numpy at K = 64. "Wider" means exact inside a block, rounded to fp32 when blocks combine.

**Layout.** out[i, j] = sum over k of x[i, k] · w[j, k]. One operand carries the probes, one per row; every element of the other is a constant c (1 for fp32; 4096 for fp16, with probe values divided by 4096, so the products are the same). The probe operand is whichever of x and w has more rows (x on a tie). Shapes are kept exactly, so M = N = 1 holds one probe per call. Outputs of the fp32-or-wider models are integers of size at most 2048, exact in fp16 too.

No probe reads one property alone, so the classifier compares every output with every candidate's simulation; the rules below only show that the families separate candidates.

- **Rounding.** Terms 2^24, 3, −2^24 at positions 0, 1, 2; exact sum 3. fp32 round-to-nearest-even (RNE) gives 4 for sequential, pairwise and blocks of 8; round-toward-zero (RZ) gives 2. Wider-in-block gives 3, and so does fp32 with 2 strided lanes, where positions 0 and 2 share a lane and cancel. The mirrored row gives −4, −2 and −3. The triple repeats at other offsets and spreads, which separates alignments that confuse one triple.
- **Sliding.** 2^24 at 0, 1 at 1, a second 1 at d, −2^24 at K − 1; exact sum 2. fp32 sequential gives 0 for every d. fp32 contiguous blocks of b give 0 until d shares the last block with −2^24, then 1, because that block sums to 1 − 2^24 exactly: blocks of 8 give 0 for d = 2 to 55 and 1 for 56 to 62, and pairwise gives 0 up to 31, then 1. Wider-in-block with blocks of b gives 2 while d < b, 0 in a middle block and 1 in the last: blocks of 8 give 2 for d = 2 to 7, 0 for 8 to 55, 1 for 56 to 62; blocks of 32 give 2 up to 31, then 1. Strided lanes alias: 4 lanes cycle 0, 1, 0, 2 with d. So the first step reads b, and the cycle reads a lane count.
- **Tree (FPRev-style pairs).** 2^24 at i, −2^24 at j, ones at all remaining positions when K ≤ 2049 (up to 2047 ones) and at 2000 of them above that. The output depends on how many ones join a partial sum near 2^24 before the pair cancels, and on how those sums round: a lone 1 next to 2^24 is lost, a sum of ones of 2 or more can survive, and 2^24 + 3 rounds to 2^24 + 4 under RNE. Where the tree joins i and j therefore shows in the output.

Assumptions: MPS picks kernels by shape and call form (the earlier probe saw order change with both), so results are per cell, never pooled. Whether torch 2.15.0 changed the MPS float GEMM path is not checked. numpy 2.5.3 and torch 2.14.1 both report Accelerate as their BLAS (checked 2026-11-09), but torch also reports `USE_EIGEN_FOR_BLAS=ON`, and the routes differ at the vector shapes: numpy uses sdot only when M = N = 1 and sgemv when one of them is 1, while `F.linear` goes through addmv/gemv or dot. So the two CPU routes count as one library only where their outputs agree.

## What is already known

From this repo, measured on this M1 Max:
- [docs/findings.md](../../../findings.md), "2026-11-09: MPS fp16 GEMM accumulates in fp32" (torch 2.14.1): partial sums are held in fp32, not wider, at some stage; a wider accumulator inside a block is not ruled out. Of 66 structured results, 19 matched sequential order (at (64,1024,64), (512,512,512), (512,4096,512), and one call form at (3,1023,5)), 28 matched only a lane, pairwise or block order, and 19 matched no simulated order.
- Same file, "2026-11-09: torch 2.15.0 `_int_mm` on MPS, measured": bit-exact against int64, 42.1 ms against 39.4 ms for fp16 `F.linear` at 4608 × 6144 × 6144. This removed the card's earlier headline (a chunked exact fp32 route); a small census checks exactness only for sanity.

Read by link: [FFLAS-FFPACK, arXiv cs/0601133](https://arxiv.org/abs/cs/0601133); [Khattak and Mikaitis, arXiv 2512.07004](https://arxiv.org/abs/2512.07004) (tensor-core models, V100 to B200, no Apple); abstract only, [MMA-Sim, arXiv 2511.10909](https://arxiv.org/abs/2511.10909) and [FPRev, arXiv 2411.00442](https://arxiv.org/abs/2411.00442). The Metal spec allows RNE or RZ and denormal flushing, with no API to select or query either (Appendix C).

**MLX is a partial known-answer control.** Its source at v0.32.4 fixes, per cell, the kernel path, partition sizes and outer structure (Appendix B). It does not state the order, width or rounding inside `simdgroup_multiply_accumulate`, the 8-wide step of the steel and split-K paths, so that step is measured, not checked. MPS may use the same instruction; not checked.

## Prediction (written before any run)

Fairly confident: RNE on every backend, though the spec permits RZ; bit-identical runs; 0 census mismatches. Weak priors: denormals flushed on the GPU arms and kept on the CPU; no wider in-block accumulator on MPS fp32 (about even odds); MPS blocked, with a block size that changes with shape and call form; MLX a different model from MPS, with path, partitions and outer structure as its source says; Accelerate one blocked order that does not change with M, the two routes agreeing except perhaps at M = 1 or N = 1; torch 2.14.1 and 2.15.0 identical on MPS; most MPS cells identified. Revised before any run: a draft prior of "not sequential at K ≥ 1024" contradicted the repo's fp16 finding above, so sequential order is expected among the fitting candidates at its three shapes.

## Design

Step 0 (desk) passed 2026-11-09. Step 1 (one lab slice) is "Step 1, exactly", which wins over any other section. Step 2 (another slice) re-runs on new seeds and positions so the observation can become a result; new candidates enter here, and a wider accumulator gets a Khattak-Mikaitis-style truncation-window probe. Step 3 (needs Rob): run the probe on the 2017 PC's GPU in CUDA fp32; any cloud run needs Rob's approval.

## Step 1, exactly

**Code, committed before any GPU time.**
- `bench/numerics_probes.py`, numpy only: probe generators, candidate simulators, classifier, comparator. `rnd`, `seq_sum`, `simulate` and `ORDERS` move here from `bench/probe_fp16_accumulation.py`, which imports them back (it imports torch, and the MLX venv has none).
- `bench/numerics_probe_set.npz`: the whole probe set in compact form (per probe: K, nonzero positions and values, fit or held-out, call placement), generated once on the CPU. Why: regenerating inside a GPU process would redraw tree ones-sets, and each redraw needs every fp32-or-wider candidate's prediction, too slow for a 240 s deadline.
- `bench/probe_numerics_torch.py --backend mps|cpu` and `bench/probe_numerics_mlx.py`: thin entry scripts that expand the `.npz` with numpy only, call the backend, and append outputs to JSON per shape, flushed, so a deadline kill keeps finished shapes. They simulate nothing.
- `bench/numerics_cells.json`: cells, seeds, the frozen candidate list and MLX source reading (Appendix B), and the sha256 of the `.npz`. Every candidate's predictions are generated on the CPU before GPU time.
- CPU tests in `tests/`: every worked value in Theory and Appendix A, the planted checks, the range checks.

**Environments.** All Python 3.12 with numpy 2.5.3. E1, the repo venv: torch 2.14.1. E2: torch 2.15.0 (cp312 wheel about 139 MB). E3: `mlx==0.32.4` and `mlx-metal==0.32.4` (wheel 78.1 MB), no torch. E2 and E3 live outside the repository; receipts record versions, not paths. Free disk was 33 GiB on 2026-11-09; stop installing if it would fall below 20 GiB (the runner refuses admission below about 19). MLX version is the researcher's call; default 0.32.4, source read at the same tag. MLX's peak-memory call moved between versions (`mx.get_peak_memory` against `mx.metal.get_peak_memory`); use whichever 0.32.4 provides, checked at install.

**Shapes**, fifteen (M, K, N): (1,16,1), (4,64,4), (3,1023,5), (64,1024,64), (512,512,512), (512,4096,512), (1,16384,1), (1,16384,512), (512,16384,1), (512,16384,512), (8,32768,8), (64,16,64), (64,64,64), (64,256,64), (64,4096,64). Nine distinct K: 16, 64, 256, 512, 1023, 1024, 4096, 16384, 32768.

**Arms and cells.** MPS on E1 and E2, fp32 and fp16, call forms `F.linear(x, w)`, `x @ wt` contiguous, `x @ w.t()`: 180 cells. MLX on E3, fp32 and fp16, `x @ wt` contiguous and `x @ w.T`: 60 cells. CPU on E1, fp32 only, numpy `x @ wt` and torch `F.linear`: 30 cells, each route classified on its own and pooled per shape only where outputs agree bit for bit. No CPU fp16: neither route sends fp16 to Accelerate's GEMM.

**Probes, seeds and packing.** Generated at each K by Appendix A, with every draw from `numpy.random.default_rng([seed, K])`, so every cell at one K sees the same probes. Seed 0 draws fit tree pairs and their ones-sets; seed 1 draws held-out tree pairs (disjoint from fit) and their ones-sets, and splits rounding and sliding probes 70/30; seed 2 draws the census; seed 3 drives the planted shuffler; seed 4 shuffles probe order. The 70/30 split takes floor(0.3 n) probes as held-out and the rest as fit, counting mirrored rounding pairs as one unit so each pair stays on one side (K = 16: 2 of 8 rounding pairs and 3 of 13 sliding probes held out). Per cell: 64 fit and 48 held-out tree probes. (Researcher defaults, changeable before the freeze.)

At every shape, fit and held-out probes form one list, shuffled with seed 4, that fills call rows in order, call after call, so held-out probes share calls of the same shape and form (a separate call could hit another kernel). Spare rows, in the last call or where a call has more rows than probes, cycle the list from its start; the repeats feed the position check. Denormal probes run in their own calls (Appendix A).

**Process order.** Eight processes: P1 MPS-E1, P2 MPS-E2, P3 MLX-E3, P4 MLX-E3, P5 MPS-E2, P6 MPS-E1 (ABCCBA, so drift with time or heat shows between repeats), then P7 CPU (numpy, then torch) and P8 CPU (torch, then numpy). Inside each: comparator check, the full probe set twice (passes A and B), denormal probes, fp32 census. Inside a pass, shapes run by ascending K (ties in the listed order), then dtype, then call form, so a deadline kill loses the largest shapes first. Classification runs afterwards on the CPU: it fits on fit probes, writes the fitted class to a file, and only then reads held-out outputs.

**Per-process budget.** Every process, GPU or CPU, runs as `python3 bench/gpu_runner.py --monitor --seconds 240 --rss-gib 8 --min-disk-free-gib 15 -- <venv python> <script> ...`, inside the lab's 300 s per-process limit. Not `bench/with_gpu.sh`, which sets no deadline. Six GPU processes at 240 s cap GPU time at 24 minutes, inside the 30 GPU-minute slice. RSS may undercount unified memory, so each GPU process records `torch.mps.driver_allocated_memory()` or MLX's peak memory after every shape and exits past 6 GiB. This set's time is not estimated (the earlier probe took 24.4 s), so before P1 a timed dry run of the P1 entry script runs one pass of the five shapes with K ≥ 16384 under `--seconds 120` (2 of the 6 spare GPU minutes; its outputs are not classified). If the dry run projects a full process past 200 s, each GPU process is split in two by K (up to 4096; 16384 and above) at 120 s each, which keeps the 24-minute cap. A second timed dry run, of the P3 entry script on E3 under the same 120 s guard, feeds the subject gate below (2 more spare minutes). Prediction and classification (about 200 candidates with RZ, flush and denormal variants, K up to 32768) run as one runner process per K each, 18 in all, since one process for all K could pass 300 s.

**Checks before GPU time, on the CPU.** Any miss stops the run.
- Range checks and worked values pass (Appendix A, Theory).
- Planted backends: fp32 sequential RNE, fp32 sequential RZ, fp32 pairwise RNE, wider-in-block blocks of 16 RNE, an MLX-steel-like running accumulator (8-term chunks summed exactly, rounded, then added, RNE), and a shuffler that draws a new order every call. Each is written independently of `simulate()`, with explicit `np.float32` scalar arithmetic and a hand-written RZ, and is fed the packed operands for all 15 shapes, fp32 and fp16 (the /4096 scaling path), through the same pack, shuffle and unpack code as the entry scripts. Why: a planted backend that calls the classifier's own simulator is identified by construction, so the check would test only packing. The first five must come out "identified" at every K with the planted candidate among the identified members; at K = 16, blocks of 16 sharing a class with larger blocks and an exact sum rounded once is a pass, since a class is all a black box can tell apart. The shuffler must fail self-comparison and, sent past that stop on purpose, must not come out "identified".
- Planted accumulation error, fp32: products 2^24 − 1 at 0, 4 at 1, −2^24 at K − 1, zeros elsewhere; exact sum 3. The partial sum 2^24 + 3 is not representable. Predicted output: sequential RNE 4, sequential RZ 2, pairwise RNE 4, wider blocks of 16 RNE 3 at K = 16 and 4 at K > 16, the running accumulator 4 at every K (its first chunk rounds 2^24 + 3 to 2^24 + 4). Pass: each planted backend's output equals its prediction at every K.

**Subject gate, before P1.** The planted checks test the instrument. This gate tests the backends: the hypothesis can only be tested where a backend's outputs on the probes are not simply the exact sum. A backend that accumulates exactly across all of K returns the exact sum on every rounding, sliding and tree probe; every such candidate then lands in one large class, the cell counts as "identified", and a "yes" says nothing about order. Measured on the two dry runs (MPS on E1, MLX on E3, five shapes with K ≥ 16384, both dtypes, every call form) and on one CPU runner process of the same five shapes (numpy and torch routes, fp32, `--seconds 120`): for each cell, the share of fit probes whose output differs from the generator's exact sum, and whether two calls agree bit for bit. Only fit-probe outputs are compared with exact sums; no candidate is simulated against them and no held-out output is read, so nothing is fitted before the freeze. Bar, per arm (backend × dtype): every dry-run cell that completes has at least 5 percent of its fit probes off the exact sum, and at least one cell completes deterministically. An arm that fails is "exact on the probe set" (or "not deterministic"), reported as a finding; its cells leave the run and the kill line's denominator. If both MPS arms fail, the hypothesis cannot be tested on MPS by these probes: stop and park the card before P1. Prior evidence, MPS fp16 on torch 2.14.1 only: 62 of 66 structured results in the repo's fp16 finding were not exact. MPS fp32, MLX and the CPU routes are not checked. MPS on E2 is not piloted; E1 stands in. Cost: 2 GPU minutes beyond the existing dry run and up to 2 CPU minutes, inside the slice.

**Checks during and after the run.**
- Comparator: flipping the low bit of one copied output gives exactly 1 mismatch, at the right index.
- Self-comparison: passes A and B, and repeat processes, agree bit for bit. A failing cell is "not deterministic", a finding, not classified; a backend failing in every cell stops the run.
- Position: every copy of one probe (spare-row repeats and the copies along the constant operand) must agree. Otherwise the cell is "position-dependent", reported and not classified, since no single output can stand for the probe.
- Incomplete: a cell whose pass A, pass B or repeat process was cut by a deadline is "incomplete", not "not deterministic", and is not classified.
- Identification, per cell. A class is a set of candidates whose simulations agree on every fit probe of that cell; held-out, denormal and census probes do not, or held-out probes could never refute. Candidates matching the observed fit outputs agree with each other, so at most one class matches; several plausible models show as class size, which the report gives. Outcomes:
  - "identified": a class matches every fit probe, and at least one member matches every held-out probe. Those are the identified members; the report counts the members held-out probes removed. The class is on file before held-out outputs are read, and one miss refutes a member because the simulators are exact.
  - "fit but not predictive": a class matches every fit probe and no member matches every held-out probe. Counts against the hypothesis.
  - "no listed model": no class matches every fit probe. Counts against the hypothesis.
  For each identified cell the report also gives the nearest rival class and how many held-out probes separate it from the identified members.
- MLX against its source. A cell's source set is every candidate with the outer structure its source reading gives (path, partition sizes, running accumulator or source-written model), under any inner step, width and rounding. Pass: the identified members include a source-set candidate. Steel and split-K cells that are "no listed model" or "fit but not predictive" are findings about the 8-wide step, not mismatches, because their source set spans every listed inner step. The 16 `dot_product`, `gemv` and `gemv_t` cells ((1,16,1), (1,16384,1), (1,16384,512), (512,16384,1)) have source-written candidates, so any such outcome there is a mismatch: it points to a wrong source reading or simulator. An identified cell whose members include no source-set candidate is also a mismatch. "Not deterministic", "position-dependent" and "incomplete" are reported, never mismatches. Classification runs after all eight processes, so a mismatch does not stop anything; it triggers the re-freeze.
- Re-freeze, at most once, logged: fix the source reading, simulator or classifier, re-freeze the cells file with new seeds for every cell, and re-run all eight processes as a second slice (step 1b) with the same guards, since one slice cannot hold both. No cell is rescued by adding a candidate after its outputs are read.
- Denormals, identified cells only: the identified members are simulated under the eight on/off combinations of input, product and partial-sum flushing (Appendix A). A property is "yes" or "no" when every combination consistent with the outputs agrees, "not separable" when they disagree, "inconsistent" when none fits. fp16 cells report inputs only.
- Census, fp32 only: codes drawn uniformly from the integers −127 to 127, K in {256, 1024}, shape (320, K, 320), in every process. GPU arms use their first call form (`F.linear` on MPS, `x @ wt` on MLX); each CPU process censuses both routes. Worst case 1024 · 127² = 16,516,096, below 2^24. Pass: 0 mismatches against int64.

**Outputs and receipts.** JSON per process under `docs/review/<run date>-experiments/cross-stack-numerics/`, written per shape: chip, macOS and package versions, reported BLAS, seeds, the cells-file commit and `.npz` sha256, check results, peak memory, free disk, wall time. Per call it stores one output per probe (its first copy), the count and indices of copies that disagree with it, and the sha256 of the full output buffer. Full M × N outputs would be roughly 60M values, 400 to 600 MB of JSON. One classification JSON: per cell, the outcome, class and its size, identified members, nearest rival and separation count, MLX source check and denormal readout. Under 50 MB. Every write-up opens "unconfirmed observation" until step 2 repeats it.

## Controls and confounds

Against a chance fit: held-out probes, a frozen list, independent planted backends and MLX's source-known outer structure. If 2.14.1 and 2.15.0 differ, no claim is made about "MPS" without a version. The probes are not realistic data; they separate models, and the earlier random-data results stand beside them. Load-dependent kernel choice shows in self-comparison and the ABCCBA repeats.

## Kill criterion

An MPS cell is conclusive when classified or "position-dependent" (which counts as not identified). "Not deterministic" and "incomplete" cells are inconclusive and leave both the count and its denominator: by the frame's rule, a difference inside the baseline's own spread is inconclusive, and a cell that disagrees with itself has nothing fixed to identify.
- Kill: fewer than half of the conclusive MPS cells are "identified". Report "not identifiable by black-box probes" and stop. With all 180 conclusive this is the original line of 90.
- Inconclusive: fewer than 90 MPS cells are conclusive. No pass or kill; the report names the lost cells, and a re-run is the researcher's call.

Raising the line, and the 90-cell floor (a default), are the researcher's call before the freeze.

## Cost on this hardware

Step 1: at most 28 guarded GPU minutes (24 plus two dry runs), 8 CPU minutes for P7 and P8, 2 for the subject gate's CPU process, plus up to 72 for per-K prediction and classification (18 processes at 240 s at most), and one more slice if the re-freeze fires; half a day to a day of coding; downloads of about 139 MB (torch 2.15.0) and 78 MB (mlx-metal) plus MLX. Step 2: one slice. Step 3: minutes on the PC; any cloud run needs Rob.

## What a yes changes and what a no changes

A yes is the first measured description of Apple GPU GEMM arithmetic the search found, and tells anyone comparing Mac and CUDA outputs which differences are order and which are bugs. A no says reproducibility work on Apple must pin its own kernels. No verifiable-inference claim: `_int_mm` is already exact for linears, and whole models fail on attention, softmax and norms.

## Why it should carry to larger models, and what it leaves untested

Order belongs to the kernel at a shape, so it carries to any model calling that kernel there. Untested: M2 and later (M5 has NAX paths; M3 and later also take MLX's `gemv_wide`), other macOS versions, bf16, unprobed shapes, the width of any wider accumulator.

## Review record

**Earlier passes, 2026-11-09.** A hostile review removed the exact-path headline (`_int_mm` is already exact at 1.07 to 1.08x fp16), cut the census to a sanity check, pinned torch, narrowed width to the in-block question, added the search gate, planted backends and a 2^24-crossing row, reused the existing probe and split out the divergence law (Yuan et al. arithmetic not rechecked). A cold-planning review and cold reads 2 and 3, which simulated the classifier and read MLX v0.32.4, fixed probe placement, seeds, per-K generators, fp16 probes, CPU routes, process order, runner limits and the kill line; replaced a 5-point held-out margin, which failed true models on perfect data, with exact matching; formed classes over fit probes only; corrected MLX routing (no `gemv_wide`, more split-K shapes, steel as one running accumulator, `gemv` from source); and made the re-freeze its own slice.

**Cold read round 4, 2026-11-09.** The reader judged step 1 runnable cold; all points taken. MLX reading corrected: at (1,16384,512) the call forms take different kernels, and (512,16384,1) runs `gemv` with bn = 8. The "ambiguous" outcome was removed (it cannot occur). The 16 source-written MLX cells now re-freeze on a miss. Planted backends are independent of the simulator and cover the fp16 path. Added a committed probe file, per-K CPU processes, an output format that fits 50 MB, "position-dependent" and "incomplete" outcomes, a timed dry run, the inconclusive outcome in the kill rule, and defaults for butterfly lanes, split-K chunk alignment, the 70/30 rounding and denormal row fill. Fixed the numpy sdot/sgemv statement, the tree ones count, and the MPS prior that contradicted the repo's fp16 finding.

**Subject-gate review, 2026-11-10.** Prompted by copies-as-evidence step 1a, where every measurement check passed but the model got no greedy base-task answer right (one template inaccurate, the other off-format). This card had planted checks on the instrument but nothing confirming the backends' outputs are not simply exact sums, which would make every cell "identified" in one uninformative class. Added a subject gate before P1 (5 percent of fit probes off the exact sum per dry-run cell, per arm; failing arms leave the run; both MPS arms failing parks the card), an MLX dry run and a CPU pilot; GPU cap 26 to 28 minutes. Void risk without it judged medium: the repo's MPS fp16 evidence argues against exactness there, but MPS fp32, MLX and CPU are not checked. Nothing else changed.

## Appendix A. Probe generators at any K

Positions are 0-based; values are products. fp16 operands (value / 4096: 4096, 2^-12, 3 · 2^-12) are exact normal fp16 numbers. The generator checks that every exact sum, and every prediction of the fp32-or-wider candidates, lies in [−2048, 2048]. The fp16-product and fp16-accumulator candidates are exempt (2^24 overflows fp16); the comparator treats NaN as equal to NaN, as `same()` does in the old script.

- **Rounding.** Offsets a = 2^j − 1 for j = 0, 1, 2, … while a ≤ K − 3 (so a = 0 once). At each a: the adjacent triple (2^24, 3, −2^24) at a, a + 1, a + 2, and the spread triple at a, a + 1, K − 1, dropped where it equals the adjacent one; each also mirrored. K = 16 gives 16 probes.
- **Sliding.** 2^24 at 0, 1 at 1, 1 at d, −2^24 at K − 1. For K ≤ 66, every d from 2 to K − 2 (13 probes at K = 16). For larger K, d = 2^j − 1, 2^j and 2^j + 1 for each j, plus 48 values of d evenly spaced from 2 to K − 2 and rounded; keep 2 ≤ d ≤ K − 2, deduplicated.
- **Tree pairs.** Ordered pairs (i, j), i ≠ j, drawn uniformly without repeats, with seed 0 (fit) or 1 (held-out); held-out pairs exclude fit pairs. Fit pairs always include, for each 2^j with 1 ≤ 2^j ≤ K − 1, the straddling pairs (2^j − 1, 2^j) and (2^j, 2^j − 1); the rest are drawn. Ones at all remaining positions (not alternating ones) when K ≤ 2049; otherwise at 2000 positions drawn per probe, uniformly without replacement from the rest, and zeros elsewhere. If any fp32-or-wider prediction for a probe leaves [−2048, 2048] (RNE ties near 2^24 can round upward), that probe's ones-set is redrawn from the next draws of the same stream, and the redraw is recorded. 2000, not 2047, keeps redraws rare.
- **Denormals, fp32**, each in its own call at the cell's shape and form, with its own c, the probe repeated in every row. Inputs: 16 terms of 2^-140 at positions 0 to 15, zeros elsewhere, with c = 2^30; exact 2^-106, flushed inputs give 0. Products: c = 2^-70, one product of 2^-120 at position 0, then 15 products of 2^-140 at positions 1 to 15; exact 2^-120 + 15 · 2^-140, which fits every K ≥ 16 and is exactly representable; flushed products give 2^-120. Partial sums: c = 2^-70, products 2^-120 and −(2^-120 − 2^-130) at positions 0 and 1, then 2^-120 at K − 1; exact 2^-120 + 2^-130. The normal term first keeps a sequential running sum out of the subnormal range; blocked orders may not, hence the joint readout.
- **Denormals, fp16**: 16 inputs of 2^-20 (fp16 subnormal) at positions 0 to 15 with c = 4096, repeated in every row; exact 2^-4, flushed inputs give 0. Products of fp16 values are at least 2^-48 in size, so they never reach the fp32 subnormal range.

## Appendix B. Candidate list and MLX source reading

Outer structures:
- Sequential; pairwise.
- Strided lanes, 2, 4, 8, 16 or 32, each lane summed sequentially, lanes combined sequentially.
- Butterfly lanes (a generic family for MPS): 2, 4, 8, 16 or 32 strided lanes, powers of two because `simd_shuffle_down` halves; lanes combined by a halving shuffle-down tree; 1, 2, 4 or 8 simdgroups, each owning a contiguous range of K, their results summed sequentially. Researcher's default. Why it matters: the list is frozen, so this choice decides which MPS cells can be identified and feeds the kill line.
- Contiguous blocks of 8, 16, 32, 64, 128, 256 or 1024. Each block forms its own sum, its 8-term steps combining sequentially into it; block sums combine sequentially.
- Running accumulator: one accumulator across all of K, updated every 8 terms, with no block sum. The accumulator is either fused into the step (accumulator plus 8 products summed exactly, rounded once) or added after it (the 8-term chunk formed by the inner step, then added). (Adding it first, then products one by one, is sequential.) Accumulator 1 and a chunk of 2^24 + 1 give 2^24 + 2 fused and 2^24 added after, under RNE.
- Split-K: partitions, each a running accumulator, whose results combine sequentially. For MPS, 2, 4, 8, 16 or 32 equal parts (any remainder in the last); for each MLX split-K cell, also the partition sizes the source gives, including an unequal last partition. 8-term chunks align to each partition's start, so a partition whose length is not a multiple of 8 (K = 1023 in 32 parts of 31) ends in a short chunk. Researcher's default; the alternative, alignment to global k, is a different candidate set.
- Source-written: `dot_product`, `gemv` and `gemv_t`, written from the v0.32.4 kernel source during the source reading, with every parameter (lane chunk, lane stride, simdgroup count and whether it splits K or the output, shuffle and threadgroup reduction order) recorded per cell and call form. Where the source leaves an order open, each reading is its own candidate.

Inner 8-term step, for blocks, running accumulators and split-K: sequential, pairwise, or exact then rounded once (a guess at what `simdgroup_multiply_accumulate` does).

Width inside a block or lane: fp32, or wider (exact, rounded to fp32 when combined). For running accumulators the fused placement plays this role.

Rounding: RNE or RZ, applied to every fp32 operation of the model and to the final cast of an fp16 output. Why one mode throughout: the search gate reads the mode as set per GPU family, not per instruction.

fp16 cells add the earlier probe's fp16-product and fp16-accumulator candidates, RNE only, as in the old script.

**MLX source reading at v0.32.4 on the M1 Max** (device class ending in 's', architecture generation 13; round-2 to round-4 readers, re-checked before the freeze). The two call forms take the same path at every shape except (1,16384,512).
- `dot_product`: (1,16,1), (1,16384,1).
- (1,16384,512): `gemv()` picks its kernel from transpose_mat = !transpose_b. `x @ w.T` runs `gemv` with bm = 1 and bn = 8 (K ≥ 16 · out_vector_len), bn splitting K. `x @ wt` contiguous runs `gemv_t` with sm = 8, sn = 4, bn = 4, tn = 4 (out_vector_len 512 ≥ 512), bn splitting the output, not K.
- (512,16384,1): `gemv` with bm = 1 and bn = 8 for both call forms (mat = x, transpose_a false, out_vector_len 512, K ≥ 8192).
- Regular steel, a running accumulator: (4,64,4), (64,16,64), (64,64,64). Its bk = 16 is a tile-load size, not an arithmetic block.
- Split-K (taken when the 16-row tile count _tm · _tn ≤ 2048, _tk = K / 16 ≥ 8 and K ≥ max(M, N)): (3,1023,5) with 31 partitions of 16 terms and a last of 527 (`steel_gemm_splitk.h` lines 140 to 153, leftover_bk = 15); (64,256,64) 4 partitions; (64,1024,64) 16; (64,4096,64) 32; (8,32768,8) 32; (512,512,512) 2; (512,4096,512) 2; (512,16384,512) 4.
- `gemv_wide` never runs here: `gemv_wide_config` returns no config below generation 15 and needs an fp16 or bf16 output.

The list is frozen before the run; new candidates go to step 2 with new seeds, except through the logged re-freeze.

## Appendix C. Search gate, 2026-11-09

24 queries (23 web, one arXiv API listing; Semantic Scholar refused one with HTTP 429), the step-0 terms and then other communities' words. Closest works:
- [Rigel, arXiv 2606.12765](https://arxiv.org/abs/2606.12765), abstract: M4 Max `matmul2d`, accumulator at least fp32; no order, rounding or M1 data.
- [Apple Developer Forums thread 844598](https://developer.apple.com/forums/thread/844598), read in full: MSL 8.1 and 8.2 permit both rounding modes and denormal flushing, undocumented per GPU family, not queryable.
- [MLX issue #3953](https://github.com/ml-explore/mlx/issues/3953), read in full: fp32 matmul on M5 Max rounds to the fp16 grid on the NAX path.
- [Niimi, arXiv 2609.22991](https://arxiv.org/abs/2609.22991), abstract: MPS `bmm` indexing bugs, not rounding.
- No Apple GPU coverage, abstract level: [Hawkeye, arXiv 2603.20421](https://arxiv.org/abs/2603.20421); MMA-Sim; FPRev; [Open-1B, arXiv 2609.17380](https://arxiv.org/abs/2609.17380); arXiv 2606.22283 (Neural Engine). MPS figures a search summary attributed to [arXiv 2606.16059](https://arxiv.org/abs/2606.16059) are not in its abstract and are not cited.

Still open: Rigel's full text; MPSGraph headers and PyTorch `LinearAlgebra.mm` for fast-math settings; a Semantic Scholar pass; non-arXiv 2026 workshop papers.
