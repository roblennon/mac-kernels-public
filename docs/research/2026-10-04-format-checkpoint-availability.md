# F1 checkpoint availability — October 4, 2026 UTC

**Both missing native formats have available Krea 2 Turbo checkpoints.** The earlier local inventory correctly found no local MXFP8 or ConvRot W4A4 file; that is not an online availability gap. The smallest sufficient next proof is the existing useful Krea recipe once for each format, with actual native dispatch attribution, finite inspected output and the unchanged resource guards. Availability and compatible headers do not establish a working model.

The search used current publisher model APIs and model cards, followed by exact HTTP-range reads from two pinned revisions. No complete checkpoint payload was downloaded in this research pass. The [machine-readable evidence](2026-10-04-format-checkpoint-availability.json) preserves repository revisions, API file records, expected payload hashes, header hashes, shapes, storage counts and all distinct quantization configurations. Complete-file hashes come from publisher LFS metadata and must be independently checked after download.

Subsequent parent-owned execution verified both complete payload hashes and passed tiny synthetic MPS loader cases. The first full MXFP8 replay stopped during model admission before a forward; W4A4 model execution remains pending. The research-only findings below retain their original scope.

## Selected native-format proofs

| Candidate | Verified layout and present loader support | Resource consequence |
|---|---|---|
| [Comfy-Org Krea 2 Turbo MXFP8](https://huggingface.co/Comfy-Org/Krea-2), repository updated September 23, 2026; 13,532,318,080 bytes (12.60 GiB) | Pinned header contains 256 MXFP8 layers in `_quantization_metadata`, native F8_E4M3 weights and U8 block scales. Existing `convert_old_quants` converts this metadata for `MixedPrecisionOps`; the absence of `.comfy_quant` tensors is expected. Mixed full-precision flags need actual route attribution during proof. | Packed storage alone exceeds the current 0.5 MPS allocator allowance, about 10.67 GiB. A full direct-MPS load is unsuitable. Use ordinary CPU-mapped construction and the separately reviewed partial-residency diagnostic, with fresh-process conditioning. |
| [ariaotp Krea 2 Turbo mixed INT4 ConvRot](https://huggingface.co/ariaotp/INT4-ConvRot-W4A4-ComfyUI), repository updated July 21, 2026; 9,234,026,328 bytes (8.60 GiB) | All 224 `.comfy_quant` bodies were read: 128 `convrot_w4a4` and 96 `int8_tensorwise`. W4 configuration uses nested `params`, ConvRot group 256, quantization group 64 and `linear_dtype=int4`; packed I8 weights are N×K/2 with F32 row scales. Existing `MixedPrecisionOps` accepts this contract. | The file also has ordinary F32/BF16 tensors. Keep conservative original storage accounting until the actual loaded representation is measured. Partial residency is the proposed bounded route for both this and MXFP8. |

The MXFP8 header was 95,456 bytes; the mixed W4A4 header was 87,840 bytes. All W4A4/INT8 configuration bodies occupy one 20,224-byte range at the end of that file. Every fetched range returned HTTP 206 with the expected length. The largest individual stored tensors are 452,984,832 bytes and 905,969,664 bytes respectively; these are storage sizes, not execution-memory bounds.

The mixed checkpoint proves that an actual model executes the W4A4 route if counters and output pass. It does not become an all-W4A4 checkpoint by nomenclature. Its 96 INT8 layers must be separately counted. No conversion, new production loader, changed default routing or larger memory cap is required merely to prepare these candidates.

The current diagnostic owner is preparing ordinary CPU-mapped loading plus a 4 GiB diffusion residency budget for both native and stock comparisons. That preparation and any eventual results are separate from this availability finding. The previous successful Krea NVFP4 capture cannot simply be relabeled for a different model: each candidate must have its own identity-valid conditioning capture or an explicitly validated reuse contract.

## Other architecture evidence and search boundary

The original F1 acceptance contract requires bounded real-model proof for each advertised format and the encoder route. It does not require the Cartesian product of four architectures and every quantization format. Krea supplies compatible candidates for the two remaining format proofs. This bounded search therefore stops without claiming exhaustive absence elsewhere.

| Architecture/source checked | Finding and limit |
|---|---|
| Krea, [JoaoZaokk all-W4A4 alternative](https://huggingface.co/JoaoZaokk/Krea-2-Turbo-W4A4-ConvRot), repository updated September 14, 2026 | Publisher lists a smaller 8,057,499,632-byte native all-W4A4 checkpoint, but its API reports automatic gating. No access gate was accepted and no header was fetched. The ungated mixed candidate avoids depending on this alternative. |
| H3, [joeygambino native FL2VA repository](https://huggingface.co/joeygambino/MiniMax-H3-comfy-native-fl2va), updated August 15, 2026 | API lists a native MXFP8 checkpoint of 21,560,298,928 bytes. Publisher says W4A4 is not published in that repository because of quality loss. Neither statement proves compatibility or universal W4A4 absence; no header was fetched. This larger candidate is unnecessary for the smallest F1 proof. |
| H3, [abenliao MXFP8 transformer](https://huggingface.co/abenliao/minimax-h3-transformer-mxfp8) | Publisher describes Diffusers/cuBLASLt scale tiling. A matching format label does not establish the native Comfy checkpoint contract. It is not a verified drop-in candidate. |
| Qwen, [official Qwen Image 2.1 repository](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) and [ModelsLab W4A4](https://huggingface.co/ModelsLab/Qwen-Image-2.1-W4A4-int4) | No matching MXFP8/W4A4 filenames were found in the bounded official-repository snapshot. ModelsLab describes SVDQuant/GPTQ with a rank-128 low-rank branch and Nunchaku packing; that is not the plain ConvRot W4A4 loader contract. No compatible Qwen checkpoint was established by this search. |
| Wan, [official repackaged repository](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) and [INCModel AutoRound MXFP8](https://huggingface.co/INCModel/Wan2.2-T2V-A14B-Diffusers-MXFP8-AutoRound) | No matching filenames were found in the bounded official snapshot. The AutoRound artifact is a sharded Diffusers transformer with its own quantization configuration; native Comfy compatibility was not established. |
| Wan, [JoaoZaokk TI2V ConvRot experiments](https://huggingface.co/JoaoZaokk/Wan2.2-TI2V-5B-W4A8-ConvRot) | Publisher includes W4A4 as a measured negative quality result. This is evidence against assuming format-name equivalence or useful output, not proof that W4A4 files do not exist. |

API update timestamps describe repository revisions, not an independently established publication date for each weight file. Alternative model-card claims above are the authors’ descriptions, not local execution evidence. The selected candidates are the only two with header-level compatibility verification in this pass.

## Quantized-encoder boundary

The planned complete H3 API graph uses the actual 14.61 GiB `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` encoder through `CLIPLoader(type=minmax, device=default)` and requires fresh conditioning. Its 4 GiB partial-residency helper applies to diffusion; it does not move the encoder to MPS. The paired 0.5/0.4 watermarks and absence of `COMFY_MPS_TEXT_ENCODER=1` retain the current CPU encoder policy.

Consequently, a completed H3 graph can establish whole quantized-checkpoint conditioning through the actual API route, with actual placement recorded. It cannot establish native GPU execution of the complete NVFP4 encoder. The two real H3 encoder-layer proofs separately establish native layout, AWQ scaling, numerical behavior and offload/reload for those layers.

The full H3 encoder's packed weights alone exceed the current GPU cap; its existing admission estimate is 18.301 GiB and does not account for that allocator fraction. The explicit opt-in gate must remain intact. A whole-encoder GPU claim requires a separately proved bounded placement/lifecycle method and actual completed route counts. CPU success, synthetic matrices, selected real layers and Krea's unquantized BF16 encoder do not substitute for that claim.
