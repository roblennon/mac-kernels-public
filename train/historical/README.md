# Historical training launchers

These are the fixed launchers for the Milestone 2 runs recorded in [docs/findings.md](../../docs/findings.md) (October 2026): the Qwen-Image 2.1 512px INT8 against bf16 comparison, the 1024px runs, the profiling passes and their evaluation. They are kept so those entries can be traced to the exact command lines. They expect the datasets and checkpoints of that time under `$MK_ROOT` and are not a starting point for new runs; use [docs/how-to-train.md](../../docs/how-to-train.md) for that.

Each script resolves `train/` as the parent of this directory, so it still uses `train/.venv` and `train/train_lora.py`. Run from the repository root, for example `ARM=int8 bench/with_gpu.sh train/historical/m2_ab512.sh`. `m2_qwen1024_guarded.sh` takes its own GPU lock and is run directly.
