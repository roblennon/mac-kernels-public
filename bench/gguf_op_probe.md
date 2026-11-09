# Installed ggml actual-weight operator probe

`gguf_op_probe.cpp` isolates one actual quantized matrix from a local GGUF. It opens the model read-only, parses metadata without allocating model tensors, and reads only the selected tensor payload. The file remains immutable. The selected payload is SHA-256 checked before and after execution; file size, inode, modification time and change time must remain stable. This is not a full-model hash check; retain the model acquisition/qualification receipt separately.

CPU compilation, metadata admission, and the selected actual-weight GPU projection cases below passed on October 25, 2026. This qualifies this bounded operator baseline, not full-model generation. All execution with `--run-metal` must be submitted through the existing serialized GPU supervisor, after its current owner exits. No backend is loaded without that flag.

## Build and interface

The following is a reusable build procedure, not a GPU launch:

```sh
GGML_PREFIX="$(brew --prefix ggml)"
c++ -std=c++17 -O2 -Wall -Wextra -Wno-deprecated-declarations \
  -I"$GGML_PREFIX/include" bench/gguf_op_probe.cpp \
  -L"$GGML_PREFIX/lib" -lggml -lggml-base \
  -Wl,-rpath,"$GGML_PREFIX/lib" -o /tmp/gguf_op_probe
```

Options: `--model FILE --tensor NAME --m M --iters N --evict-mib 0|128`. M is 1–512, iterations 1–100, with three additional warmups. Eviction defaults to zero. The default mode reads metadata only. GPU execution additionally requires `--run-metal --metal-lib FILE`, with the explicit installed plugin normally at `$GGML_PREFIX/libexec/libggml-metal.so`. That plugin exports a backend registration interface; there is no directly linked `ggml_backend_metal_init` symbol in this installation's `libggml`.

The code uses the public installed gguf/ggml headers. It requires a quantized 2D tensor, K/N no larger than 32768, selected payload at most 256 MiB, and an in-bounds byte span. Unknown or unsupported operations fail; there is no CPU execution fallback. Output is JSONL on stdout, diagnostic library logs/errors on stderr. Exit 0 after a result means all stated checks passed.

## Actual Orca mappings

| EXL3 name | GGUF tensor | K | N | Installed GGUF type |
|---|---|---:|---:|---|
| layer 0 `linear_attn.in_proj_z` | `blk.0.attn_gate.weight` | 5120 | 6144 | IQ4_XS |
| layer 0 `mlp.up_proj` | `blk.0.ffn_up.weight` | 5120 | 17408 | IQ4_XS |
| layer 0 `mlp.gate_proj` | `blk.0.ffn_gate.weight` | 5120 | 17408 | IQ4_XS |
| layer 0 `mlp.down_proj` | `blk.0.ffn_down.weight` | 17408 | 5120 | IQ4_XS |

All four mappings were independently verified with this binary's header-only mode against both installed models. Nano has identical K/N geometry but different actual formats:

| GGUF tensor | Nano type | Nano packed bytes | Orca packed bytes |
|---|---|---:|---:|
| `blk.0.attn_gate.weight` | IQ3_S | 13,516,800 | 16,711,680 |
| `blk.0.ffn_up.weight` | IQ3_S | 38,297,600 | 47,349,760 |
| `blk.0.ffn_gate.weight` | IQ2_S | 28,549,120 | 47,349,760 |
| `blk.0.ffn_down.weight` | IQ3_S | 38,297,600 | 47,349,760 |

The models have different declared source checkpoints, so these are matching operator geometries, not identical weights or a quantization-quality comparison. Existing full-model identity receipts record Orca SHA-256 `4ab6bdf8a9008869abf1630fb92d04b5439805f96b56692962fa240b1d439977` (15,676,553,472 bytes) and Nano SHA-256 `6d383a195b96d5a77c186f9ff8b71ab2356c7ea6e0b446992244371b8c75a21b` (12,993,304,352 bytes). These are previous acquisition/qualification hashes, not full-file rehashes by this probe. Carry the model receipt into each trial alongside the new per-tensor hash.

## Input, reference and timing contract

Input is contiguous F32 [M,K], with flat element `i` equal to `float((17*i + 13) % 257 - 128) / 128`. Every input value is exactly representable in FP16, permitting matched-value FP16 comparisons. The graph is a single installed `ggml_mul_mat` over resident actual packed weights, producing F32 [M,N].

The CPU reference uses ggml's CPU block dequantizer followed by explicit double-accumulating scalar dot products. It shares the format definition with the runtime but does not call the Metal decoder or CPU quantized matmul. It checks up to 64 evenly spaced output channels and eight evenly spaced input rows, including both endpoints. Every output is checked finite on every warmup and timed trial, with output pre-poisoned to NaN. Relative RMS must be below 0.003; maximum absolute error divided by reference maximum magnitude must be below 0.004. The reference is against this checkpoint's decoded values, not its original BF16 source.

Each measured call includes graph dispatch and backend synchronization. Loading, compilation, resident buffer allocation, uploads, output poisoning, downloads and correctness checks are excluded. Output and graph buffers are reused. The emitted result contains every measured duration, median, worst reference errors, selected-payload hash and allocated buffer bytes. This measures the complete single projection operation, not cold startup or end-to-end inference.

With `--evict-mib 128`, a separate Metal SCALE graph reads a distinct 128 MiB F32 input and writes a distinct 128 MiB output immediately before each matrix warmup/timed call. Input is initialized once; its output is not aliased with the input or model graph. The sweep and its synchronization are outside the matrix timer. Endpoint checks confirm the sweep's result after the trial sequence. Two graph metadata arenas and the additional 256 MiB device allocation are included in the existing context/buffer management; no 128 MiB host copy is retained. The result records both byte counts and treatment. This is an eviction treatment, **not proof of a perfectly cold cache**. Run both settings because an isolated projection can fit in system-level cache while a full decoder traverses many layers.

For an EXL3 comparison, use identical input values and M/K/N. Include its input/output Hadamards, scale handling and synchronization. Either reuse its output buffers too or report its allocation overhead separately; a fresh-output Torch call versus this reused-output call is not an allocation-matched comparison. Keep compilation outside both timings and compare several interleaved blocks. Metal F32 input/output here versus a different EXL3 output dtype must be disclosed.

## Installed library identity at build

Homebrew ggml stable 0.26.0, installed as a bottle. The existing Homebrew llama-server links the same ggml/base libraries. This does not imply its old build-number commit is the ggml source revision. `otool -L` establishes linkage; the installation receipt contains no source Git commit.

SHA-256 of resolved installed binaries on October 25:

| File under ggml 0.26.0 | SHA-256 |
|---|---|
| `lib/libggml-base.0.26.0.dylib` | `5b58f3bf245e261da898ee867eeebc8a5fd79b46528ad05b20f0225df64536e2` |
| `lib/libggml.0.26.0.dylib` | `a23fcdbe8cc5d65c6a7f504d0cf4642ce3ddf98c1ae03f9fb87a67592e892231` |
| `libexec/libggml-metal.so` | `2282accab16d665c402d68ed96303af473e9fef57a2eb3662ac61a7784b5c6cb` |

The probe reports the loaded base-library path and requested plugin path. Preserve/recheck these hashes in trial receipts: Homebrew `opt` links can move after upgrades. No runtime installation, rebuild or global configuration change is part of this probe.

## Actual-weight GPU qualification, October 25

The installed plugin registers as `MTL`; the probe admits exactly `MTL` or `Metal` and records the observed name. The first attempt stopped before projection execution because admission expected only `Metal`. Correcting that name check did not change operation support or numerical tolerances.

M1 Max 32 GB, installed ggml/Metal 0.26.0 with all three library hashes above rechecked before and after the FFN sequence. Each row is one process/run with three warmups and ten timed calls. Every call includes actual quantized-weight decoding, matrix execution and synchronization. The 128 MiB read/write eviction treatment is outside the timer.

| Actual projection | M | Nano median ms | Orca median ms |
|---|---:|---:|---:|
| FFN up | 1 | 0.6359 | 0.5956 |
| FFN up | 4 | 1.6096 | 1.5298 |
| FFN gate | 1 | 0.6425 | 0.5839 |
| FFN gate | 4 | 1.4274 | 1.5444 |
| FFN down | 1 | 0.7146 | 0.6162 |
| FFN down | 4 | 1.4656 | 1.7212 |

All twelve FFN cases passed. Worst sampled CPU-reference relative RMS was 1.15e-7; worst scaled maximum error was 1.31e-7. All output elements were finite after NaN poisoning in every warmup and timed call; source stat and selected tensor hashes remained unchanged. GGML reported 297,074,688–316,145,664 bytes of allocated graph buffers, including both eviction buffers.

Each case ran under the shared 18 GiB RSS / 4 GiB swap-growth guard with a 120-second deadline. All guards exited successfully, with no sampled warning pressure or swap growth. Each process completed in about 0.36–0.39 seconds, shorter than the periodic resource sampler interval: its sampled RSS peak is **not** a meaningful measurement of the true peak. The graph buffer figures above are allocation evidence, not whole-process peak RSS.

Earlier `attn_gate` cases passed both with and without the eviction treatment:

| Model | M | No eviction, median ms | 128 MiB treatment, median ms |
|---|---:|---:|---:|
| Nano | 1 | 0.3951 | 0.3963 |
| Nano | 4 | 0.7655 | 0.7543 |
| Orca | 1 | 0.3766 | 0.3679 |
| Orca | 4 | 0.6858 | 0.6864 |

These are separate short runs, not repeated interleaved experiments or full-model speed results. A smaller weight encoding did not consistently produce a faster operator, so use the actual measured per-format baselines when evaluating EXL3. The models' decoded weights differ; numerical correctness is checked against each model's own CPU decode.

Private evidence: `.local/exl3-run/gguf-ffn-receipt.json` contains all durations, source/payload/library identities and guard summaries; individual logs follow `gguf-{nano,orca}-ffn_{up,gate,down}-m{1,4}-e128.log`. Earlier projection logs end in `-v2.log`. FFN trial source SHA-256: `9b9988d250e3ab3b0a0999bc7f873e8145b107e5aa49f6f87f37433a3af2fc09`; executable SHA-256: `fa19adfabf4a69f3b3ca8625c18bb7dbce2965681b6ec7dba84c6861319e5fe2`.
