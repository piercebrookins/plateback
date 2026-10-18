# Original pretrained models and sample-data controls

Measured locally on 2026-10-17, with no API calls or weight updates.

## Before PlateBack training

The Models page includes two original model entries alongside the trained candidates:

| Model | Four-food mask IoU | Interpretation |
| --- | ---: | --- |
| Original YOLO11n segmentation, COCO weights | 4.34% | Only broccoli overlaps the four target classes; rice, pasta and chicken are unsupported and produce no predictions. |
| Original DINOv2 encoder | Unavailable | The encoder that underlies v4/v9 has no original food classifier or segmentation decoder. An untrained random head would not be a meaningful pretrained baseline. |

Original YOLO's broccoli-only IoU is 17.35%. Presence macro F1 across the four targets is 19.23%. The evaluation uses the same 12 mask examples and 38 presence examples as the pilot, native confidence 0.25, image size 320 and CPU inference. No thresholds were fitted to these examples. This is a taxonomy-limited off-the-shelf baseline, not a controlled measure of the gain from training DINOv2.

Reproduce with `.venv/bin/python ml/score_original_baselines.py`. Full metrics, checkpoint SHA, class mapping and per-image results: `artifacts/benchmark/original-baselines.json`; per-image caches: `artifacts/benchmark/original-yolo/`.

## Why v9 is selected

V4 has higher mean mask IoU on the 12-image pilot: 54.12% versus v9's 52.65%. V9 preserves v6's chicken detection recall: 70% on the original Nutrition5k subset, versus v4's 30%. V9 uses a spatial decoder trained from v4 with v6 detection/area decisions. It is selected for this recall tradeoff, not because it wins every metric. See [v9 evaluation](EVALUATION_V9.md).

## Sample-data behavior

Settings → Load sample data controls the fictional overview plus sample records in Your scans and History. It defaults on and persists in browser local storage. Turning it off hides virtual and demo-marked scans without deleting real records.

The gallery has 1,200 read-only fictional records across 30 meal sessions, derived from the existing 240-record scenario across five weeks. Public Nutrition5k illustrative photos repeat. These are not 1,200 unique photographs or real student observations. All weights remain unknown, and no sample masks or predictions are fabricated. Virtual records are never inserted into SQLite or used in training/evaluation. Attribution and provenance are shown in Settings. The original 240-record overview remains unchanged.

Trials, repeated page labels, local-workspace badges and prominent demo banners were removed. Meal-session creation is available in Settings.

Verification: 39 Python tests passed; production frontend build passed; browser checks covered persistent toggle state, real-only filtering, pagination, search, read-only record dialog and original model rows. This change spent $0 in API calls; recorded project accounting remains approximately $2.58 of $5, including retained reservations.
