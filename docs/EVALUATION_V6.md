# V5 follow-up: chicken recall and local domain adaptation

**Active scanner: v3 remains unchanged. These are research candidates, not a claim of beating Gemini.**

## What changed

Added 300 training images from Nutrition5k and reserved 60 validation plus 60 official-test images. Together with earlier training examples, this gives 526 Nutrition images (217 chicken positives), alongside 1,732 FoodSeg103 pixel-labeled training images. Independent decoded-image audit found no new duplicates against 2,694 prior images or within the added data. Source: original Nutrition5k CC-BY-4.0, via the public food-nutrients mirror. Ingredient labels are presence supervision only, never pixel masks or leftover weights.

Trained three approaches locally on CPU in 110.7 seconds: linear DINO presence classification, nonlinear MLP presence classification, and weak-label multiple-instance learning (MIL) for the dense segmentation head with pixel-supervised replay. DINOv2 backbone remains frozen; this is not full backbone fine-tuning. FoodSeg training cached features use bilinear resizing; new Nutrition features and actual presence inference use bicubic resizing. Calibration uses actual inference features.

## Audit of the old Nutrition5k misses

Of ten ingredient-positive plates, v5 found three. Five of seven misses contained high-confidence chicken seeds but were rejected by the 4% image-area rule. Two had no seed above 0.75. The contact sheet shows diced chicken and smaller portions among the area-filter failures. Ingredient annotations do not prove all chicken is visible, particularly in mixed dishes; labels were not rewritten from visual guesses.

![All ten old chicken-positive plates](../artifacts/evaluation-v6/nutrition-chicken-audit.jpg)

## Validation-only selection

Selected rescue: `{"method": "mlp", "rule": {"confidence": 0.75, "grow_confidence": 0.1, "min_area_fraction": 0.005}, "source": "mil-v6", "threshold": 0.8}`.

Eligible policies: 0; selected policy meets validation requirements: **False**. Selected FoodSeg validation chicken IoU 46.2%, presence recall 83.8%, pixel recall 86.3%, false positives 32. New Nutrition validation recall 63.3%, false positives 3/30.

The rescue unions additional regions with v5. It cannot remove a v5 chicken pixel; the other three classes are unchanged. That guarantees recall relative to v5, not relative to v3, and may increase false positives. Thresholds and weight hashes were frozen before evaluating the new test. A failed validation policy remains explicitly ineligible; no silent constraint relaxation.

## Nutrition5k presence results

| Dataset | Model | Chicken recall | Precision | False positives |
|---|---|---:|---:|---:|
| nutrition_old38 | v3 | 40.0% (4/10) | 66.7% | 2/28 |
| nutrition_old38 | v5 | 30.0% (3/10) | 60.0% | 2/28 |
| nutrition_old38 | v6 | 70.0% (7/10) | 77.8% | 2/28 |
| nutrition_new60 | v3 | 26.7% (8/30) | 66.7% | 4/30 |
| nutrition_new60 | v5 | 40.0% (12/30) | 75.0% | 4/30 |
| nutrition_new60 | v6 | 73.3% (22/30) | 84.6% | 4/30 |
| nutrition_old38 (cached) | Gemini | 100.0% (10/10) | 83.3% | 2/28 |

Validation method comparison (each restricted to false-positive limits):

| Mask source | Classifier | Nutrition recall | Nutrition FP | FoodSeg IoU |
|---|---|---:|---:|---:|
| semantic-v5 | linear | 30.0% | 3 | 46.8% |
| semantic-v5 | mlp | 53.3% | 3 | 46.7% |
| mil-v6 | linear | 30.0% | 3 | 46.7% |
| mil-v6 | mlp | 63.3% | 3 | 46.2% |

New60 is the only fresh test in this round. It is balanced by ingredient presence and has no pixel ground truth; it does not estimate dining-hall prevalence or prove segmentation accuracy. Old38 is regression evidence and motivated the change.

## Earlier segmentation regression sets

| Dataset | Model | Chicken IoU | Presence recall | Pixel recall | False positives |
|---|---|---:|---:|---:|---:|
| food110 | v5 | 42.9% | 67.7% | 74.0% | 16 |
| food110 | v6 | 44.4% | 71.0% | 77.2% | 16 |
| chicken32 | v5 | 68.5% | 93.8% | 97.0% | 3 |
| chicken32 | v6 | 68.1% | 93.8% | 97.2% | 3 |
| food188 | v5 | 32.2% | 77.6% | 78.7% | 40 |
| food188 | v6 | 31.8% | 77.6% | 78.7% | 41 |
| food83 | v5 | 38.5% | 69.2% | 61.1% | 13 |
| food83 | v6 | 37.0% | 69.2% | 61.2% | 15 |

All these pixel-labeled sets have been used in prior evaluations. They remain regression evidence, not fresh model-superiority evidence. Chicken/duck is the FoodSeg source label.

## Economical Gemini comparison

| Cached paired set | Model | Chicken IoU | Presence recall | Pixel recall | False positives |
|---|---|---:|---:|---:|---:|
| food110 paired subset | v6 | 45.6% | 72.7% | 72.3% | 3 |
| food110 paired subset | gemini | 56.4% | 72.7% | 81.1% | 4 |
| chicken32 paired subset | v6 | 77.6% | 100.0% | 99.6% | 1 |
| chicken32 paired subset | gemini | 90.1% | 87.5% | 93.9% | 0 |

Cached Gemini outputs are reused without another request. Both systems use identical image IDs and ground truth; invalid cloud outputs would count as failures. Previously seen samples cannot establish a fresh win. Paired bootstrap intervals (2,000 resamples) are in `artifacts/evaluation-v6/results.json`. Public model pretraining overlap remains unknown.

**New API spend: $0. Total accounted: $1.621076 / $5**, including $0.80 of retained unknown-outcome reservations. Accounted costs are estimates, not provider billing. No paid GPUs.

## Deployment checks

- validation_eligible: **False**
- old_nutrition_recall_vs_v3: **True**
- new_nutrition_recall_vs_v3: **True**
- new_nutrition_fp_vs_v3: **True**
- food110_pixel_recall_vs_v3: **False**
- food110_presence_recall_vs_v3: **True**
- food110_fp_vs_v3: **True**
- fresh_segmentation_superiority_established: **False**
- chicken32_recall_vs_v3: **True**
- chicken32_pixel_recall_vs_v3: **True**
- chicken32_fp_vs_v3: **True**
- food188_recall_vs_v3: **True**
- food188_pixel_recall_vs_v3: **True**
- food188_fp_vs_v3: **True**
- food83_recall_vs_v3: **True**
- food83_pixel_recall_vs_v3: **True**
- food83_fp_vs_v3: **True**

On Food110, candidate pixel recall is 77.250% versus v3 77.294%. Even this small shortfall fails the strict no-regression check.

No candidate was promoted. No new pixel-labeled independent benchmark was collected, so a fresh segmentation win against Gemini is unestablished even if a cached metric improves. Unknown food mass remains unknown.

## Reproduction

With the existing v5 artifacts and source manifests:

```sh
.venv/bin/python ml/prepare_nutrition_v6.py
.venv/bin/python ml/audit_nutrition_v6.py
.venv/bin/python ml/audit_nutrition_v5.py
.venv/bin/python ml/train_chicken_v6.py
.venv/bin/python ml/calibrate_chicken_v6.py
.venv/bin/python ml/evaluate_chicken_v6.py
.venv/bin/python ml/report_chicken_v6.py
.venv/bin/python -m pytest -q
```

Do not tune on the reported test results and describe a rerun on the same images as fresh. New weight versions require a new evaluation/cache directory. `freeze.json` records checkpoint and policy hashes; detailed per-image predictions and the miss audit are retained.
