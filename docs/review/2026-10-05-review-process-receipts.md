# Review-process receipts, 2026-10-05

These four paragraphs were moved verbatim out of [docs/findings.md](../findings.md) on 2026-11-09. They record how the independent source reviews of 2026-10-05 were run (durations, tool-call counts, restriction violations), not what the hardware measured. The findings entries keep the measurements and a pointer here.

<a id="h3-smaller-review"></a>

## Review before the smaller H3 attempt (findings entry "smaller H3 still stops for pressure; WAN v8 review is invalid")

This H3 attempt followed an actual 773.965-second independent review completed at 20:49:56 UTC with a fresh `NO_FINDINGS` verdict. The parent accepted its correctness verdict with qualified procedural scope: two read-only `awk` text counters violated the literal interpreter restriction, without candidate-code execution, mutation or payload/GPU access. The review’s claim of no interpreter use was rejected. That bounded acceptance does not change the later failed GPU outcome or imply that other reviews passed.

<a id="wan-v8-first-review"></a>

## First WAN v8 review, invalid (same findings entry)

The separate WAN v8 512×288×33 candidate passes 56 parent CPU tests in 4.571 seconds with its 2^26 attention cap. Its first independent review is invalid for activation: it was stopped after 211.404 seconds, exited −15 and emitted no final verdict. Actual tool use wrote `/tmp/vf.txt` and `/tmp/vb.txt` and invoked a prohibited Python interpreter. Source and input checks after the stop preserve 334 source pins, 377 evidence pins, five failure pins, six model-stat identities, 715 copied text files and the 717-file inventory. No GPU execution or activation occurred.

<a id="wan-v8-replacement-review"></a>

## WAN v8 replacement review (findings entry "bounded WAN v8 still stops for pressure after a compliant replacement review")

The separate replacement review completes in 388.396 seconds at 21:13:37 UTC with exit 0 and `NO BLOCKING FINDINGS`. All 46 actual tool calls are reads inside the copied input tree, with no dynamically loaded instruction files. Parent adjudication retains accurate partial coverage, corrects the optional claim that CPU test names are not pinned and an incorrect file attribution, and defers inherited direct-test-run ergonomics because the frozen discovery command executes all 56 tests. The original invalid review remains invalid and preserved; this compliant replacement source review does not qualify the failed GPU result.

<a id="h3-task-vm-review"></a>

## Review of the H3 task-VM observation (findings entry "H3 task-VM observation stops for pressure without a completed forward")

The actual independent source review completes in 509.298 seconds with no blocking findings. Its 49 tool calls are reads, with 34 complete files and seven excerpts; no extra instruction files or tool errors are recorded. Parent adjudication preserves partial coverage and two nonblocking limitations: the cumulative telemetry-cost budget can stop a sufficiently slow observation, and an oversized supplemental diagnostic may be omitted after the workload is already stopping. Neither limitation weakens the primary guard or accepts an incomplete run. All owned processes and their group are gone, and the GPU lock is idle at parent closeout.
