# Local semantic segmentation: v4 experiment

Trained locally on **1,732 images** for 40 epochs (9.8 minutes). Architecture: DINOv2 ViT-S/14 frozen backbone + trained 1x1 dense MLP head. New pixel-annotated training data increased chicken/duck examples from 269 to 748. No paid GPU was used.

## Fresh local evaluation: 110 images

| Model | Macro IoU | Chicken IoU | Chicken precision | Chicken presence recall | Chicken pixel recall | Chicken false positives | Other-food false alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original YOLO v1 | 39.3% | 19.8% | 29.2% | 45.2% | 48.9% | 34 | 23/40 |
| Previous ensemble v3 | 48.8% | 33.4% | 48.8% | 64.5% | 77.3% | 21 | 11/40 |
| Semantic v4 | 68.2% | 46.0% | 56.8% | 67.7% | 66.4% | 16 | 9/40 |

| Food | v1 IoU | v3 IoU | v4 IoU | v1 recall | v3 recall | v4 recall | v4 pixel recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| rice | 20.5% | 46.2% | 71.0% | 65.0% | 55.0% | 90.0% | 84.7% |
| pasta | 47.3% | 38.0% | 67.7% | 75.0% | 60.0% | 90.0% | 74.3% |
| broccoli | 69.5% | 77.8% | 88.1% | 100.0% | 88.9% | 100.0% | 95.0% |
| chicken | 19.8% | 33.4% | 46.0% | 45.2% | 64.5% | 67.7% | 66.4% |

## Paired Gemini comparison: 32 new images

| Model | Macro IoU | Chicken IoU | Chicken presence recall | Chicken pixel recall |
|---|---:|---:|---:|---:|
| Original YOLO v1 | 35.2% | 17.6% | 36.4% | 33.9% |
| Previous ensemble v3 | 50.7% | 43.0% | 72.7% | 80.1% |
| Semantic v4 | 67.7% | 45.7% | 72.7% | 62.1% |
| Gemini 4 Argon | 71.7% | 56.4% | 72.7% | 81.1% |

Local minus Gemini macro-IoU difference: **-4.0 percentage points**, paired image-bootstrap 95% interval **[-12.2, +4.1]**. Gemini yielded 32/32 usable outputs. Failed geometry/transport outputs count as empty predictions.

Valid-Gemini-geometry diagnostic (32 images): local macro IoU 67.7%, Gemini 71.7%. This excludes malformed/missing outputs only; wrong or empty but valid predictions remain. It is a diagnostic, not a replacement for the full predefined comparison.

**This experiment does not establish that the local model beats Gemini.** A higher point estimate alone, if present, is insufficient when the paired interval crosses zero.

## Validation selection and recall guard

| Food | Selected source | Threshold | Minimum component area | Presence recall floor | Pixel recall floor | All constraints met |
|---|---|---:|---:|---:|---:|---|
| rice | semantic | 0.60 | 0.50% | 81.8% | 77.4% | True |
| pasta | semantic | 0.15 | 0.05% | 72.7% | 59.9% | True |
| broccoli | semantic | 0.50 | 0.50% | 95.7% | 93.7% | True |
| chicken | semantic | 0.70 | 5.00% | 61.8% | 59.7% | True |

## Training approach comparison (validation only)

| Approach | Calibrated macro IoU | Calibrated chicken IoU | All constraints passed | Selection objective |
|---|---:|---:|---|---:|
| deeplab-v4 | 46.6% | 26.0% | False | 0.3628 |
| dino-v4 | 68.6% | 47.3% | True | 0.5795 |
| dino448-v4 | 71.1% | 49.8% | True | 0.6048 |

Selected: **dino448-v4**. Validation-only: maximize 0.5 chicken IoU + 0.5 four-class macro IoU among candidates passing ALL class recall and false-positive constraints. If none pass, retain explicit experimental status and block promotion. Defined before test access.

The threshold grid and objective were frozen before test scoring. Presence recall detects whether a food was found at all; pixel recall measures how much of its annotated region was recovered. Neither quantity is food mass. Validation constraints are not a guarantee on unseen plates.

## Limits and evidence

Fresh held-out images from the publisher training pool, screened against all earlier splits. Chicken includes duck. Food presence recall and pixel recall are reported separately. Validation constraints do not guarantee recall on new images. No tuning on these results. Public-source contamination of frontier pretraining is unknown.

The 110 images were reserved from unused publisher **training** data before training. All previous holdouts remain excluded. This new local split is not the official publisher test split. Independent pHash audit found no training-to-holdout near duplicates; unknown source clustering remains possible. FoodSeg103 chicken/duck labels cannot validate chicken-only species recognition.

API accounting at v4 scoring: **$1.515430 / $5**, including retained unknown-outcome reservations. The new 32 Gemini requests cost $0.256532. Training cost no API tokens. Unmeasured food weight stays explicitly unknown.

Checkpoint SHA-256: `d43540ab54ff88403edfe2a6ebe129fbc70fa560d84dc088712e26b364d398bc`. Raw predictions, per-image counts, paired bootstrap and cost evidence are in `artifacts/evaluation-v4/` and `artifacts/benchmark-v4/`. Portable results: [evaluation-v4-summary.json](evaluation-v4-summary.json). Training configuration and sources: [frozen protocol](EVALUATION_V4_PROTOCOL.md). Earlier benchmark comparisons are now regression evidence.
