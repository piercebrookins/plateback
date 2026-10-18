# V9: spatial decoder trained from v4

**Selected for Local and Local + OpenAI:** v6 masks with v9 fixed-area spatial refinement. All v6-relative regression gates pass. Improvements are small, and this remains an experimental hackathon model.

## Training and selection

Started from Local semantic v4. Its frozen DINO backbone and original head are bit-identical in the new checkpoint. Trained a 3×3-convolution residual decoder at twice the patch resolution on 1,732 FoodSeg training images, with 197 validation images. Twenty CPU epochs took 272.9 seconds; epoch 4 had the lowest validation loss. Seed 191. Checkpoint SHA-256: `440c84ffe822f2d9286f6fe38ea097682e14b2731447ba94822cd8d87420091c`.

Mined 657 hard-negative training images using v4 chicken scores; independently checked that none is outside the training split or contains chicken/duck mask pixels. Oversampled them 2.5×, with chicken images 1.8×. These are existing licensed images, not 657 newly acquired images. Loss combines recall-weighted CE/Tversky, hard-pixel CE and v4 distillation.

Ordinary threshold refinement failed: its best chicken validation IoU was 49.73%, but pixel recall fell to 74.98% and Nutrition validation detections dropped from 19 to 17. This policy was rejected before regression evaluation.

The selected rule instead preserves each v6 mask’s exact pixel area and image-level presence. It ranks pixels using the new decoder plus a 0.05 prior for existing v6 pixels, within a dilation radius of 5% of the image’s smaller dimension. Stable sorting makes ties deterministic. No non-chicken mask changes. Validation IoU rose from 46.2036% to 46.2187%; pixel recall from 86.3083% to 86.3276%. This is a small boundary adjustment, not a major accuracy breakthrough.

## Regression evaluation

| FoodSeg set | Chicken IoU v6 → v9 | Pixel recall v6 → v9 | False-positive pixels removed | False-positive images v6 → v9 |
|---|---:|---:|---:|---:|
| food110 | 44.3657% → 44.4445% | 77.2500% → 77.3449% | 1,303 | 16 → 16 |
| chicken32 | 68.0833% → 68.1017% | 97.2038% → 97.2195% | 130 | 3 → 3 |
| food188 | 31.8145% → 31.8227% | 78.7216% → 78.7369% | 306 | 41 → 41 |
| food83 | 37.0297% → 37.0471% | 61.1779% → 61.1989% | 268 | 15 → 15 |

All image-level detection counts are unchanged. Nutrition5k chicken recall remains 22/30 (73.33%) on the newer 60-image regression set, with 4 false positives, and 7/10 (70%) on the original 38-image set, with 2 false positives. Nutrition5k does not provide pixel masks here. False-positive **pixels** improve; false-positive **images** do not. Aggregate pixel recall improves on each set; this does not imply every individual plate improved.

These are repeatedly inspected regression datasets, not a new untouched test. FoodSeg chicken includes duck, and pretrained-model overlap is unknown. The original production gates have not been re-certified by this v6-relative comparison.

## Shared benchmark and application

Models now contains 14 entries. On the shared 12-image mask pilot, v9 reaches 52.6498% four-food IoU versus v6 52.6481%, v4 54.1190%, and cached Gemini 59.0933%. The selected model preserves v6 chicken recall; v4 has higher pilot IoU but lower Nutrition5k chicken recall. We have not beaten Gemini. No cloud benchmark was rerun. Historical OpenAI results use earlier prompts and are not an evaluation of the current direct-contour prompt.

Selection is saved in artifacts/models/local-selection.json. Local and Local + GPT-6 Luna use it. V6 and all earlier checkpoints remain available. Unknown food weights remain unknown.

## Verification and cost

38 tests pass, including spatial initialization, frozen-head gradients and deterministic fixed-area redistribution. Frontend build passes. Native selected-model inference returns all four component hashes; see artifacts/local-v9-selection-smoke.json.

New API spend: **$0**. Accounted total: **$1.63087759 / $5**, including uncertain reservations; not a provider invoice. All training, calibration and new benchmark inference ran locally.

## Reproduce

```sh
.venv/bin/python ml/train_spatial_v9.py
V7_MODEL=spatial-v9 .venv/bin/python ml/calibrate_backbone_v7.py
# Preserve the initial calibration as calibration-thresholds.json before the next step.
.venv/bin/python ml/calibrate_spatial_budget_v9.py
REFINEMENT_MODEL=spatial-v9 .venv/bin/python ml/evaluate_backbone_v7.py
.venv/bin/python ml/score_recent_pilot.py
```

Full protocol: EVALUATION_V9_PROTOCOL.md. Model and probability caches are identity-checked. The threshold and fixed-area policies are selected using validation only, before this regression evaluation.
