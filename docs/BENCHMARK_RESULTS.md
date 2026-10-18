# Specialist versus generalist pilot benchmark

Earlier pilot (regression evidence for v4/v5): 38 Nutrition5k presence images and 12 FoodSeg103 segmentation images. Mask scores include transport-only parsing (JSON fences removed; longer contours accepted), with strict scores recorded separately. Invalid geometry/missing outputs still count as failures.

| Model | Presence macro F1 | 95% image-bootstrap interval | Rice precision / recall | Chicken precision / recall | Parsed mask IoU | Strict mask IoU | Presence median latency | API cost |
|---|---:|---|---|---|---:|---:|---:|---:|
| Local YOLO v1 | 30.8% | 21.8%–38.8% | 25% / 30% | 26% / 50% | 30.3% | 30.3% | 0.012 s | $0.00000 |
| Local YOLO v2 | 39.6% | 26.0%–49.9% | 100% / 20% | 50% / 70% | 23.3% | 23.3% | 0.016 s | $0.00000 |
| Local ensemble v3 | 37.5% | 21.6%–50.8% | 100% / 20% | 67% / 40% | 19.9% | 19.9% | 0.095 s | $0.00000 |
| Local semantic v4 | 58.5% | 31.8%–74.1% | 75% / 30% | 60% / 30% | 54.1% | 54.1% | 0.136 s | $0.00000 |
| Local semantic v5 | 58.5% | 31.8%–73.0% | 75% / 30% | 60% / 30% | 52.8% | 52.8% | 0.134 s | $0.00000 |
| Local presence specialist | 50.4% | 36.3%–60.1% | 53% / 90% | 67% / 40% | N/A | N/A | 0.029 s | $0.00000 |
| qwen/qwen3.5-9b | 85.0% | 55.9%–91.2% | 70% / 70% | 75% / 90% | 16.9% | 0.0% | 1.599 s | $0.00239 |
| openai/gpt-6-luna | 73.7% | 49.4%–85.7% | 60% / 60% | 70% / 70% | 2.1% | 1.9% | 2.053 s | $0.00460 |
| anthropic/claude-opus-5.5 | 83.0% | 53.5%–89.8% | 75% / 60% | 75% / 90% | 53.7% | 27.8% | 3.725 s | $0.25236 |
| google/gemini-4-argon | 89.1% | 59.7%–96.1% | 78% / 70% | 83% / 100% | 59.1% | 0.0% | 3.126 s | $0.19804 |

## Interpretation

These are separate tasks. Ingredient presence is scored on 38 official-test Nutrition5k images (10 rice, 2 pasta, 12 broccoli, 10 chicken positives and 12 negatives). Pixel-mask overlap is scored on 12 FoodSeg103 test images, including four other-food negatives. The benchmark is small and stratified, so it is not an estimate of natural cafeteria prevalence. The two pasta positives are insufficient for a stable pasta comparison.

The local presence specialist trained only a linear multilabel head over frozen MobileNetV3 Small features, using 713 FoodSeg103 and 226 Nutrition5k training images. It is a separate experiment, not a mask generator and not the deployed segmentation model. All local checkpoints/thresholds were selected using validation only.

Generalist models were asked for food contours through their text/vision interfaces. Parsed mask scoring strips JSON code fences and accepts longer valid contours without changing geometry; the strict original output-format score is also shown. This secondary parsing analysis was added after inspecting formatting failures, before computing accuracy, and is explicitly post-hoc. Truncation, malformed coordinates and missing outputs remain failures. Low strict-format scores are not equivalent to poor visual recognition.

Presence latency includes the measured API round trip for cloud models and local prediction for YOLO; local classifier latency includes preprocessing. Model loading is excluded except any first-prediction warmup. Runs share one laptop/network with concurrent work; this is a practical pilot, not controlled hardware benchmarking. Local API cost is zero, not a claim of zero electricity/hardware cost.

Uncertainty intervals use 1,000 image-bootstrap resamples, seed 93. They do not account for unknown source clusters, domain shift, annotation ambiguity or public-benchmark contamination. No universal claim of outperforming frontier models is justified.

## Failures and coverage

- Local YOLO v1: 38/38 presence; 12/12 parsed masks (12 strict).
- Local YOLO v2: 38/38 presence; 12/12 parsed masks (12 strict).
- Local ensemble v3: 38/38 presence; 12/12 parsed masks (12 strict).
- Local semantic v4: 38/38 presence; 12/12 parsed masks (12 strict).
- Local semantic v5: 38/38 presence; 12/12 parsed masks (12 strict).
- Local presence specialist: 38/38 presence; no mask head.
- qwen/qwen3.5-9b: 38/38 presence; 8/12 parsed masks (1 strict).
- openai/gpt-6-luna: 38/38 presence; 12/12 parsed masks (10 strict).
- anthropic/claude-opus-5.5: 37/38 presence; 6/12 parsed masks (2 strict).
- google/gemini-4-argon: 38/38 presence; 12/12 parsed masks (5 strict).

## Budget and evidence

Total build API accounting at scoring: **$1.621076 / $5**, including $0.80 in retained reservations for eight rejected OpenRouter setup calls. Model rows show returned request costs only, not those setup reservations or earlier app verification calls. All API access uses the shared persistent ledger. No paid GPU service was used.

Raw responses and native usage are in `artifacts/benchmark/responses/`; rejected setup evidence in `setup-errors/`; deterministic image selection in `manifest.json`; all per-image scores and class counts in `detailed-results.json`. Saved catalog pricing and prompt hashes support reproduction. `docs/benchmark-summary.json` is the portable summary. Public benchmark images were not added to training.

## Reproduce

```sh
.venv/bin/python ml/prepare_nutrition.py
.venv/bin/python ml/train_presence.py
.venv/bin/python ml/benchmark_frontier.py   # paid calls; cache avoids duplicates
.venv/bin/python ml/score_benchmark.py
.venv/bin/python ml/report_benchmark.py
```

Source/license and split details: [data provenance](DATA_AND_LICENSES.md). Procedure and setup corrections: [frozen protocol](BENCHMARK_PROTOCOL.md). The screenshot supplied in the request describes another task and is not evidence for this food benchmark.
