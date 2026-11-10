# Funder research: what is scored, and which directions are unclaimed

2026-11-09. Supersedes sections (a) and the Mozilla corrections in
[2026-11-09-gap-research.md](2026-11-09-gap-research.md). Eight research lanes ran, each with a
reader and an independent verifier that re-opened the sources. Tags: [R] read directly in a
browser or as raw text this day, [S] read only through a summarizing fetch, [I] inference, not
stated by any source. Claims a verifier could not support are listed as corrections, not dropped
silently.

## 1. Funders

### Mozilla Human-Scale AI Incubator [R; FAQ last updated 2026-10-15]

- Seven projects, $300k over 12 months from April 2027, up to $200k more over two years. Deadline
  2026-11-20 23:59 PST. Info session 2026-11-12 09:00 PST.
- Wants "working technologies already deployed with the people they're designed to serve, even at
  a small scale or through an early pilot."
- Not wanted: "research projects without a working technology component"; "Technologies without
  users, contributors, or another meaningful signal of early product-community fit".
- Six criteria, unweighted: Human-Scale AI (Small, Intentional, Tailored), Technical viability
  ("Is this a novel technology with real momentum?"), Problem-community clarity,
  Product-community fit, Impact and values, Sustainability potential.
- Product-community fit evidence may be "returning users, contributors, community involvement in
  shaping the project, partners who want to pilot or fund the work, or downstream adoption by
  other builders."
- Fiscal sponsorship is recommended ("Yes, we often recommend"), not required. The recipient must
  be able to receive funds from a US 501(c)(3). The eligibility guidelines PDF lists
  "Individuals. Anyone carrying out a charitable project aligned with our mission" and "Advance
  scientific research" as a charitable purpose, and says open-source licensing "isn't enough on
  its own if the underlying purpose is commercial." (PDF read by a verifier, not by the author of
  this note.)
- AI use: AI may help "brainstorm ideas, clarify language, or assist with translation"; content
  must "reflect your own original work, ideas, and voice".

Corrections to earlier repo docs: a rubric is published (six criteria); a fiscal sponsor is not a
stated requirement.

### Mozilla Fellowship 2027-2028 [R; pages, FAQ last updated 2026-10-25, full RFP PDF]

- Ten Fellowships, individuals only. Independent track: $125,000 ($100k stipend, $25k project
  budget, hardware up to $5,000 per item). Letter of Intent due 2026-12-17 23:59 Pacific. Full
  application by invitation, due 2027-03-12. Starts 2027-06-01, 12 months.
- FAQ: "Working technology and users are welcome, but not required." The programme page still
  asks for "proof you can build, people using what you built" and says "Tell us who uses it and
  how you know." The bar on users is lower than the Incubator's, not absent.
- Relevant focus areas: Hard infrastructure ("local inference hardware that don't depend on one
  vendor's roadmap"); Sovereign AI, public infrastructure and maintenance ("Communities shouldn't
  have to rent their intelligence from a distant cloud").
- Not a fit: "Is a technology looking for a problem"; "Is designed about a community rather than
  with it"; "Optimizing the status quo is somebody else's job."
- The LOI is short: project summary 200 words, one-liner 30 words, target users 50 words,
  traction to date 50 words.
- May be held alongside other funding. Applying to both Mozilla programmes is allowed.

### Open Intelligence Foundation, openintelligencefoundation.ai [R, raw HTML]

- Up to $10,000 per grant, "Larger grants available for select projects". Volunteer-run, founded
  by CRS Credit API, "not yet 501(c)(3)". Form at /request-funding.html asks for X and GitHub
  links, a description, an amount and a use of funds; "Equipment/Hardware" is an allowed use.
- Two public grantees: Prince Canuma (MLX on Apple silicon; the grant was one GPU) and Maziyar
  Panahi. Both had very large adoption numbers when chosen.
- The site was last modified 2026-06-02 and the X account has no posts. The programme may be
  dormant.
- A different organisation, openintelligence.foundation, shows no grants for individuals.

### Emergent Ventures [R for the form; S for Marginal Revolution posts]

- "Emergent Ventures no longer accepts AI-written applications. We will test all submissions and
  reject those using AI for anything other than proofreading."
- 1500 words in three parts: personal story, one consensus view, the idea and what is new about
  it. No PDFs. "NO typical grant range". The only sourced size is a 2019 median of $10k to $25k.
- Winner lists for cohorts 42 to 59 contain no GPU-kernel, quantization or Apple-hardware line.

### Others

- NLnet NGI Zero Commons Fund: final call closed 2026-06-01. Its weights (30% technical, 40%
  impact, 30% cost) are only an example of how one funder scored.
- a16z Open Source AI Grants funded SGLang and vLLM, described by throughput, and Ostris. No
  application route or amounts are published.
- FLOSS/fund excludes "Very new projects or projects with minimal usage". Astera requires
  Emeryville. Sloan and CZI do not fund individuals. Most other open programmes are CUDA compute
  credits.

### What the rubrics have in common

Every published rubric read here scores adoption and public benefit at least as heavily as
technical merit. Performance work does get funded (vLLM, SGLang, ggml) when it arrives with
adoption. No funded example of Apple-GPU kernel work was found.

## 2. Research directions, checked against prior art

| Direction | Verdict after verification |
| --- | --- |
| Custom fp16 or bf16 GEMM; attention kernels at head_dim 128; int8 GEMM on GPU before M5 | Dead ends. PyTorch maintainers (2026-10-20): large layers "already close to bandwidth-bound". PyTorch 2.16 attention on Metal is about 1.00x at the repo's shapes. |
| Neural Engine plus GPU split | Already shipped in the open on M1: Siliconed v0.14 reports Krea 2 at 135.4 to 137.4 s against 144.6 to 144.9 s on M1 Pro, 5 to 7 percent, with 16 to 26 percent less energy. Larger gains (1.59 to 1.70x, vpipe, FLUX, M4 Pro) are on newer chips. Not a novel result here. |
| 4-bit on Metal | Ordinary. mflux, Draw Things, Siliconed and vpipe ship it. |
| Streaming packed weights for video | Taken. vpipe docs/MINIMAX-H3.md: "Both widths stream, so neither has to fit in RAM", measured on a 16 GB M5 Air. vpipe also reports that width barely changes clip time because the time goes to storage reads. |
| Video or large-image numbers on pre-M5 Macs with 16 GB or less | Open. The only data point is one hobbyist M1 Air README without per-step time, peak memory or quality. |
| Perceptual quality of quantized variants on Mac | Open. No Mac engine read reports one. Academic papers report it only with simulated quantization on NVIDIA. |
| Speed, memory and fidelity benchmark across Mac tiers for diffusion | Open as far as one search can show. SiliconBench (2026-09) does this for LLM serving only. |
| LoRA training on a quantized base | The idea is prior art: OneTrainer int8 backward on CUDA, FourTune, mflux on MLX. Open: a backward that uses packed codes on Apple GPUs, and a real training run of a current large model on a 16 GB Mac. Published Mac figures for Qwen-Image 2.1 are 21.2 to 21.4 GB (mflux) and 40.5 GB (ai-toolkit) on M5 Max; this repo measured 9.46 GiB on M1 Max. That comparison crosses chips and trainers and needs a same-machine run. |
| Adapter trained on a quantized base, used on the full-precision base | Standard practice (Hugging Face FLUX QLoRA post, 2025-06-19). Open: a controlled measurement, and the INT8 ConvRot case. |
| Audit of framework defaults on consumer GPUs | The bf16 default tax is documented separately on M1 Max, NVIDIA T4 and AMD RDNA2, and no source connects them. PyTorch already runs a nightly MPS performance workflow on M2 runners, so "no monitoring exists" would be wrong. Gaps: M1-class hardware, full diffusion pipelines, ComfyUI default paths, quantized formats. Maintainers have dismissed a median of 10 runs as too noisy. |
| Wide-K matmul on M1 and M2 | An open PyTorch PR handles an out-of-bounds read with a kernel about 8x slower. The author of a candidate replacement kernel measured M4 Max only. M1 Max numbers are a contribution maintainers can use. Zero effect on Krea end to end. |

Corrections owed elsewhere in the repo: `docs/research/2026-10-02-lora-training-on-quantized-bases.md`
predates mflux's Qwen trainer (added 2026-11-06); the research statement should not imply
attention is a remaining lever on Krea.

## 3. Inference [I]

- The fundable claim is access, not speed: which models run and train on which Macs people
  already own, at what memory and what fidelity. The measured speedups are the evidence for it.
- The cheapest thing that raises the odds with every funder is adoption evidence: resolvable
  fork pins, a merged upstream PR, results from a second machine, named users.
- The Fellowship is the best structural fit. The Incubator is a long shot for a one-person
  project without users. Open Intelligence Foundation is a small, uncertain, low-effort ask.

## Decisions, 2026-11-09

- No application is submitted today, to any programme.
- The Mozilla Fellowship Letter of Intent (due 2026-12-17) is the date to plan around. The other
  options stay on record: the Incubator (due 2026-11-20, long shot without users) and Open
  Intelligence Foundation (short form, one hardware item, may be dormant).
- Before any application, widen the research: look for new findings beyond porting CUDA ideas to
  Metal, testable on one M1 Max, one 2017-era gaming PC and free cloud tiers. The idea search and
  its prior-art checks are in
  [../research/2026-11-09-research-directions.md](../research/2026-11-09-research-directions.md).

## Not read

Mozilla Fellowship LOI questions after "Primary Project Output(s)"; Emergent Ventures cohort
posts at source (summarizer only); FourTune full text; vpipe and Siliconed beyond the pages
quoted; Wayback history for any funder.
