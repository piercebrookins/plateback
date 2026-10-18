# V7: partial backbone fine-tuning and hackathon workflow

**The default local scanner remains v3. The latest research weights are evaluated separately.**

## Local training

Updated 3,601,157 parameters: the final two DINOv2 transformer blocks, final normalization layer and dense segmentation head. Six epochs on local Apple MPS took 376.2 seconds including frozen-prefix extraction. Earlier blocks stayed frozen. Training used 1,732 FoodSeg103 pixel-labeled images and 526 Nutrition5k ingredient-labeled images; no test data entered training.

Training combines weighted CE/Tversky, an extra supervised boundary penalty, and weak ingredient-presence MIL. Backbone LR 1e-5, head LR 1e-4, seed 171, gradient clipping and cosine scheduling. The selected epoch minimizes validation loss. A second candidate averages the trained parameters 50/50 with MIL-v6 initialization to reduce drift. Both versions are calibrated on validation only.

Selected candidate: **backbone-v7**. Policy: `{"fallback": false, "mode": "replace", "rule": {"confidence": 0.05, "min_area_fraction": 0}}`.

Refinement starts from v6 detections. It can replace, intersect or union the chicken geometry, with a validation-selected empty-mask fallback. Selection requires no loss of v6 validation presence or pixel recall and no increase in false-positive images. Other food masks remain unchanged. If no eligible IoU improvement exists, the policy stays identical to v6.

## Chicken results

All sets below have been seen in previous evaluations and are regression evidence. No fresh claim of beating Gemini is supported.

| Segmentation set | Model | IoU | Presence recall | Pixel recall | False-positive images |
|---|---|---:|---:|---:|---:|
| food110 | v6 | 44.37% | 70.97% | 77.25% | 16 |
| food110 | v7 | 45.37% | 70.97% | 77.16% | 16 |
| chicken32 | v6 | 68.08% | 93.75% | 97.20% | 3 |
| chicken32 | v7 | 66.80% | 93.75% | 97.33% | 3 |
| food188 | v6 | 31.81% | 77.59% | 78.72% | 41 |
| food188 | v7 | 32.60% | 77.59% | 78.76% | 41 |
| food83 | v6 | 37.03% | 69.23% | 61.18% | 15 |
| food83 | v7 | 36.27% | 69.23% | 60.55% | 15 |

| Nutrition5k presence set | Model | Recall | False positives |
|---|---|---:|---:|
| nutrition_old38 | v6 | 70.00% (7/10) | 2 |
| nutrition_old38 | v7 | 70.00% (7/10) | 2 |
| nutrition_new60 | v6 | 73.33% (22/30) | 4 |
| nutrition_new60 | v7 | 73.33% (22/30) | 4 |

Nutrition ingredient labels are not pixel masks. FoodSeg chicken includes duck. Images may differ from real dining-hall leftovers; pretrained-model overlap is unknown.

## Cached Gemini comparison

| Paired subset | Model | Chicken IoU | Presence recall | Pixel recall |
|---|---|---:|---:|---:|
| food110 | v7 | 43.33% | 72.73% | 68.90% |
| food110 | gemini | 56.39% | 72.73% | 81.08% |
| chicken32 | v7 | 75.46% | 100.00% | 99.75% |
| chicken32 | gemini | 90.12% | 87.50% | 93.93% |

## Hackathon OpenAI path

Added **GPT-6 Luna contours · experimental**, a direct image-to-polygons route independent of local detection. Returned masks remain editable and carry no invented confidence or food weight. Added **Research masks + OpenAI review · experimental** for the validation-selected local candidate with OpenAI label review; it is opt-in and does not replace the default. A live hybrid call and isolated API replay both passed. Fragmented plates send at most eight largest regions for cloud review; remaining regions stay editable and explicitly require manual review. The existing **Local + GPT-6 Luna label review** retains default local geometry and asks OpenAI to review the labels. These are distinct workflows.

Two fixed 24-image diagnostics tested the direct route. Changing from 0–1000 JSON-mode coordinates to an explicit full-image 0–1 strict schema improved parsing and localization. This is prompt development on a reused benchmark, not independent validation.

| Direct Luna protocol | Valid responses | Chicken IoU, paired16 | Chicken recall | Chicken FP / 8 negatives |
|---|---:|---:|---:|---:|
| Initial coordinates | 23/24 | 7.08% | 62.50% | 1/8 |
| Normalized strict schema | 24/24 | 41.80% | 87.50% | 5/8 |

Direct Luna contours remain experimental because false positives and boundary errors are substantial. Use the local + OpenAI label-review workflow and manual mask correction for the demonstration. Label review does not fix missed masks. See [HACKATHON.md](HACKATHON.md).

## Budget and verification

Run ceiling: **$3**. Accounted run cost: **$0.009802**. Cumulative accounted cost: **$1.630878 / $5**. SQLite reservations enforce both limits atomically; unknown outcomes retain their reservation. No paid GPU or paid fine-tuning.

34 automated tests passed, including the independent run ceiling, strict geometry validation, direct cloud routing without local masks, and detection-preserving refinement. The frontend production build passed. Live provider responses were tested separately; an isolated API smoke replay verified editing, confirmation, original prediction preservation, unknown grams, and exclusion of demo records from real analytics.

All v6-relative regression gates passed: **False**. Failed checks:

- food110_pixel_recall
- chicken32_iou
- food83_pixel_recall
- food83_iou

No default model promotion was made. Exact checkpoint hashes, calibration searches, per-image predictions and training history remain in `artifacts/models/backbone-v7`, `artifacts/models/backbone-avg-v7`, and `artifacts/evaluation-v7`.

## Reproduce

```sh
.venv/bin/python ml/train_backbone_v7.py
.venv/bin/python ml/average_backbone_v7.py
.venv/bin/python ml/calibrate_backbone_v7.py
V7_MODEL=backbone-avg-v7 .venv/bin/python ml/calibrate_backbone_v7.py
.venv/bin/python ml/evaluate_backbone_v7.py
.venv/bin/python ml/report_backbone_v7.py
```

Cached prefixes require matching source-checkpoint and manifest identities. Use new probability-cache directories for changed weights. OpenAI benchmark caches deliberately do not retry pending or failed requests. The earlier test sets must not be described as fresh after further tuning.
