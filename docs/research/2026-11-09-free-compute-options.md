# Free compute for open-weight models

2026-11-09. Four research agents surveyed where open-weight LLMs and diffusion models can run for free beyond Kaggle. The lanes were notebook GPUs, hosted inference APIs, grant and credit programs, and unconventional routes. They favored sources from the last six months. Most figures come from aggregator blogs and provider pages read on the day. Very little was confirmed on a provider's own pricing page, and none of the Reddit or Hacker News threads could be retrieved. Treat every limit below as a lead to recheck before planning a run.

**Correction, 2026-11-10.** This survey repeated the common "about 30 GPU hours a week" figure for Kaggle. The decode thread then measured something nearer 6 hours a week, and later marked the allowance unverified because Kaggle's API reported inconsistent figures (commits `042e533` and `e792454`). Check kaggle.com/settings before planning.

## The short answer

For this project, compute grants matter more than free tiers. Inference.net grants (up to $10k, with LLM inference performance as a named focus) and Google's TPU Research Cloud (no institution required, research must be shared openly) were the two programs open to an independent developer. Applications for both were drafted on 2026-11-09 and kept outside this repository. Neither is submitted.

Ranked for a 14B–30B quantized LLM or a video diffusion model that needs 24 GB or more:

1. **AMD Developer Cloud:** a one-time $100 credit, about 50 hours on an MI300X with 192 GB. The only free route to a card that runs WAN 14B without offload. Needs a card on file. ROCm, not CUDA.
2. **Kaggle:** two T4s, 32 GB total split across two cards, with a weekly cap (see the correction above). Fits a Q4 30B LLM split across both cards. Video needs heavy quantization or offload.
3. **Modal Starter:** $30 of compute each month, about 40 hours on an L4 (24 GB) or about 6 hours on an 80 GB A100 (our estimate). Code-first serverless, not a notebook.
4. **Lightning AI:** since September 2026, a one-time grant of up to 30 credits (about 50 GPU hours on interruptible machines), including a 48 GB L40S. No longer monthly.
5. **Hugging Face ZeroGPU:** 48–96 GB Blackwell slices, but 5 minutes a day on a free account and Gradio Spaces only. Usable in bursts through existing WAN and FLUX Spaces with `gradio_client` and an HF token.

## Notebook and dev-environment GPUs

| Platform | Hardware | Free quota | Catches |
| --- | --- | --- | --- |
| Kaggle | 2× T4 16 GB, or P100; TPU v3-8 | Weekly GPU and TPU caps, 12 h sessions (9 h TPU) | Cap unverified (see correction). Phone verification. |
| Google Colab free | T4 16 GB, not guaranteed | Unpublished, 12 h max session | FAQ bans remote desktops and web UIs used mainly for content generation, which rules out ComfyUI. |
| Lightning AI | T4, L4, A10G, L40S | One-time up to 30 credits, 12-month expiry | Phone required; one GPU per Studio; 4 h restarts. |
| HF ZeroGPU | RTX Pro 6000 Blackwell, 48 GB slice (96 GB `xlarge`) | 5 min/day free, 40 min/day PRO | Demo host for Gradio Spaces, 60 s per call, no `torch.compile`. |
| Modal Starter | T4 to B300 | $30/month | Serverless code, not a notebook. |
| AMD Developer Cloud | MI300X 192 GB | $100 one-time | Card on file; approval can take 3 business days. |
| ModelScope / Tencent Cloud Studio / Baidu AI Studio | A10 24 GB / T4 / V100 | Reported free hours | China accounts and real-name verification; terms unverified. |
| Intel Tiber AI Cloud | Gaudi, Data Center GPU Max | Free notebooks claimed | Unverified. oneAPI / XPU, not CUDA. |

Dead or not useful: SageMaker Studio Lab is closed to new users. Paperspace's free tier rarely has a GPU. GCP's $300 trial excludes GPUs and TPUs. Azure's $200 trial GPU quota is commonly refused. Oracle Always Free has no GPU. Cloud Shell, Codespaces and Gitpod are CPU only.

## Hosted inference APIs for open-weight models

This lane only covered the well-known providers, and their free tiers come to thousands of requests a day at most. It's not enough for sourcing millions of tokens. That would need a wider search into small hosts, lab previews, stealth models and regional promotions, checked against live catalogs and each provider's own quota and expiry terms.

| Provider | Notable free models (2026-11-09) | Limits |
| --- | --- | --- |
| NVIDIA build.nvidia.com | Nemotron 3 Ultra 550B, Gemma 4 31B, gpt-oss-20b | 40 RPM, 10,000 requests/day per model; evaluation use only |
| OVHcloud AI Endpoints | Qwen3.5-397B, gpt-oss-120b, Llama 3.3 70B | 2 RPM per IP per model, no signup |
| OpenRouter `:free` | Gemma 4 31B and 26B-A4B, Nemotron 3 Ultra and others (12 IDs, read live) | 20 RPM; 50 requests/day, 1,000/day after buying $10 of credit |
| Groq | gpt-oss-120b, gpt-oss-20b | About 30 RPM, 1,000 requests/day; models retire often |
| Cloudflare Workers AI | Llama 3.3 70B, gpt-oss-120b, Gemma 4 26B, FLUX | 10,000 neurons/day, about 60 FLUX.1 schnell images at 1024px |
| Gonka Broker | DeepSeek-V4-Flash, MiniMax-M2.7, GLM-5.3-Flash | About 1M tokens/month, SMS verification |
| Z.ai | GLM-4.7-Flash and other Flash models | 1 concurrent request |
| Cerebras | Llama 3.3 70B | Sources disagree; unconfirmed |

On the day, OpenRouter's live model list had no free DeepSeek, Kimi or large Qwen models, though guides from July still listed them. No free video-generation API turned out to be generous. Pollinations lists WAN video models, but its free allowance was unconfirmed. The maintained list `mnfst/awesome-free-llm-apis` was the best single source. `cheahjs/free-llm-api-resources` returned 404.

## Grants and credits, by eligibility for an independent developer

| Program | What you get | Fit |
| --- | --- | --- |
| Inference.net grants | Up to $10k compute | Yes. Focus areas include inference performance optimization. |
| Google TPU Research Cloud | Free Cloud TPUs; VMs and storage billed | Likely yes. No institution requirement; publish openly and give feedback. |
| TensorWave open-source program | Up to $250k AMD GPU credits | Only for AMD/ROCm support work in open-source ML tools. |
| AMD AI Developer Program | $100 Developer Cloud credit | Yes. |
| CloudRift AI grant | $100–$1,000 credit | Yes; asks for credit when you publish. |
| Modal Starter | $30/month | Yes. Academic and startup tiers don't apply. |
| NAIRR pilot, NSF ACCESS, AWS research credits, Lambda, Nebius | Varies | No; each needs an institution or company. |
| NVIDIA Inception, Microsoft Founders Hub, Google for Startups | Credits | Only after forming a company. |
| Thinking Machines interactivity grants, a16z open-source AI grant | — | Closed, or no 2026 round found. |

Not checked: Cerebras, Groq, Intel, Stability, BFL, LAION, EleutherAI, Mozilla, OpenAI and Anthropic compute programs. (The [funder research](../review/2026-11-09-funder-research.md) covers Mozilla separately.)

## Unconventional routes

- **Oracle Always Free Ampere A1:** a 4-core, 24 GB ARM server that never sleeps. llama.cpp runs 3B–8B dense models at roughly 8–15 tok/s. Small mixture-of-experts models such as 30B-A3B should fit better than dense models of the same size, but that is untested. A1 capacity is often unavailable at signup.
- **Hugging Face free CPU Space:** 2 vCPU and 16 GB running `llama-server` gives an OpenAI-compatible URL at 2–5 tok/s for 3–8B models. Sleeps when idle.
- **ZeroGPU through `gradio_client`:** call public WAN 2.1 14B, FLUX and MusicGen Spaces as endpoints. Pass an HF token, because PRO quota through the client has had bugs.
- **Browser WebGPU (WebLLM, transformers.js):** runs on the user's own GPU. Claims of an 8B at about 41 tok/s on an M3 Max and gpt-oss-20B at about 60 tok/s on an M4 Pro are best-case numbers.
- **Coding-tool promotions (Kilo Code, OpenCode Zen, Cline):** rotate open-weight models in and out for free. Several ended in March and April 2026.
- **AI Horde and Petals:** volunteer networks with no uptime guarantee. Swarm health is unverified.
- **GPU CI (Cirun, RunsOn) and Cloud Run L4:** not free; they bill your own cloud account.

Skipped as a terms-of-service risk: multi-accounting to farm free credits, and abusing CI minutes.
