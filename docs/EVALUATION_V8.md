# V8 evaluation and hackathon selection

**Selected local model: v6.** Local and Local + OpenAI now use unchanged v6 masks. GPT-6 Luna remains the OpenAI model. V3 is retained on disk for rollback. All models remain experimental and masks require review.

## Experiment

Trained the frozen DINO dense head for 16 epochs on CPU, using hard-pixel mining and v6 distillation. Training took 110.9 seconds; epoch 5 had the lowest validation loss. Data: 1,732 training and 197 validation FoodSeg images. No new training data, paid compute, or API requests. See EVALUATION_V8_PROTOCOL.md and ml/train_hardpixels_v8.py.

The best validation-IoU diagnostic increased chicken IoU from 46.20% to 50.14% and pixel precision from 49.86% to 60.53%. But pixel recall fell from 86.31% to 74.50%. Presence recall stayed 57/68 and false-positive images stayed 32. This is rejected: cleaner masks do not justify deleting true chicken pixels. The best recall-eligible policy reached only 46.17% IoU, below unchanged v6.

Calibration therefore selects the identity policy: no v8 refinement. Regression gate equality is expected because this returns v6 exactly; it is not evidence that the trained v8 improved.

## Retained v6 regression results

| Set | Chicken IoU | Pixel recall | Presence recall | False-positive images |
|---|---:|---:|---:|---:|
| food110 | 44.37% | 77.25% | 70.97% | 16 |
| chicken32 | 68.08% | 97.20% | 93.75% | 3 |
| food188 | 31.81% | 78.72% | 77.59% | 41 |
| food83 | 37.03% | 61.18% | 69.23% | 15 |
| nutrition_new60 | unknown (no masks) | unknown | 73.33% | 4 |
| nutrition_old38 | unknown (no masks) | unknown | 70.00% | 2 |

V4 has stronger mask overlap on the small shared pilot, but loses Nutrition5k chicken recall (30% on the original 38-image set versus v6’s 70%). V7 also has mixed mask regressions. V6 is the explicit hackathon compromise, not a claim that all original production gates passed. It still has substantial false positives. No reduction in false-positive image count is claimed for this run.

## Models page

One sortable-by-metric bar graph with scores underneath, containing all 13 benchmark entries. Toggle mask IoU and food-presence F1. Every mask score uses the original same 12 FoodSeg images; every presence score uses the original same 38 Nutrition5k images. V8 is the validation-IoU-maximizing rejected diagnostic, not its unchanged-v6 fallback. It reaches 53.87% four-food pilot IoU; v4 54.12%, selected v6 52.65%, Gemini 59.09%. We have not beaten Gemini.

These are repeatedly inspected public regression samples, not an untouched final test. Cloud predictions are historical cached responses with their original prompts, including the earlier weak Luna contour prompt; this graph does not measure the current direct-OpenAI prompt. Missing/invalid masks count as failures; unsupported metrics display —. FoodSeg chicken is chicken/duck; Nutrition labels indicate ingredients, not leftover mass. Pretraining overlap is unknown.

## Cost and verification

API cost this run: **$0.00**. Shared ledger: **$1.63087759 / $5**, including retained unknown-outcome reservations. This is accounting, not a provider invoice. Unmeasured weights remain unknown.

Frontend production build passes. Models graph, metric selector, and mobile layout checked in the browser. Native local inference verified against the selected v6 components. See artifacts/local-v8-selection-smoke.json and output/playwright/models-v8-*.png.

Reproduce:

```sh
.venv/bin/python ml/train_hardpixels_v8.py
V7_MODEL=hardpixels-v8 .venv/bin/python ml/calibrate_backbone_v7.py
REFINEMENT_MODEL=hardpixels-v8 .venv/bin/python ml/evaluate_backbone_v7.py
.venv/bin/python ml/score_recent_pilot.py
```

Inference caches enforce model identity. Remove the experiment-specific probability cache before intentionally changing weights. Keep calibration and benchmark identities separate.
