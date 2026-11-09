# Predictions written before the 768 and 256 Krea runs

## 768, written 2026-11-09 14:25:04 PST, before any 768 run

Fit from the 512 and 1024 medians, tokens taken as proportional to pixel count (text tokens ignored), M=1 at 512:
stock = a + b*M: a = 2.10 s, b = 6.45 s (8.55 at M=1, 27.9 at M=4)
metal = c*M + q*M^2: c = 3.67 s, q = 0.088 s (3.76 at M=1, 16.1 at M=4)
Prediction at 768 (M = 2.25): stock 16.6 s/step, metal 8.70 s/step, speedup 1.91x.
Falsified if either lands more than about 5 percent away (stock outside 15.8 to 17.4, metal outside 8.3 to 9.1).
Alternative with no fixed stock cost (pure power law through the two points): stock 8.55*2.25^0.853 = 17.1, metal 3.76*2.25^1.049 = 8.80. The stock numbers separate the two models by 0.5 s; metal barely.

Outcome (e1-krea-768-check.jsonl, two repeats per cell): stock 15.89 to 16.17 s/step, kernel 8.41 to 8.57 s/step. Both inside the stated bands. Stock landed 3 to 4 percent under the 16.6 point prediction, which was fitted to stock runs from before about 14:00; stock runs after 14:00 were about 3 percent faster at 512 and 1024 as well. The power-law alternative (17.1) misses stock by 6 to 8 percent.

## 256, written 2026-11-09 14:49:11 PST, before any 256 run

Tokens M = (side/16)^2 + 46 text tokens: 302 at 256, 1070 at 512, 2350 at 768, 4142 at 1024.
Fit to SDPA runs taken after 14:00 today: stock 8.33 (512), 15.90 (768), 27.13 (1024); kernel 3.755, 8.55, 15.88.
Gap, stock minus kernel = 2.26 s + 2.17 ms per token (fitted on 512 and 1024; gives 7.36 at 768, measured 7.35).
Kernel = 0.03 s + 3.36 ms*M + 1.12e-7 s*M^2 (exact through the three points).
Prediction at 256: kernel 1.05 s/step, stock 3.97 s/step, speedup about 3.8x.
Falsified if stock lands outside 3.6 to 4.4 or kernel outside 0.9 to 1.2.
If the stock penalty had no fixed part and only scaled with tokens, stock would be about 1.7 to 2.4 s/step. That is the result this run can rule out.

Outcome (e1-krea-256-check.jsonl, two repeats per cell): stock 4.25 to 4.29 s/step, kernel 1.21 to 1.23 s/step, speedup 3.5x. Stock is inside its band. Kernel is 0.02 s above the top of its band, so the kernel half of the prediction failed narrowly: the kernel curve has a fixed cost of about 0.2 s per step that the three-point fit put at 0.03 s. The gap is 3.05 s against 2.91 predicted. The alternative with no fixed stock cost (1.7 to 2.4 s/step) is ruled out.
