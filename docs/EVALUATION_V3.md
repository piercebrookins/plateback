# Deployed local policy: v3

V3 uses the retrained YOLO v2 for rice, pasta and broccoli, and the original YOLO for chicken with a Nutrition5k-informed presence gate. The source, confidence, minimum mask area and gate threshold were selected per class on 197 validation images. The gate is a locally trained multilabel head over frozen MobileNetV3 Small features. All training was local; no paid GPU was used.

The v2 test revealed a chicken recall tradeoff, motivating this follow-up. Its 188 images and the earlier generalist pilot are therefore exploratory regression comparisons for v3. After freezing v3, a separate 83-image subset of unused publisher test images was evaluated. Exact and perceptual near-duplicate screening excluded all prior split images.

**Final coverage limitation:** the remaining eligible subset contains 20 rice, 21 broccoli, 26 chicken/duck-positive images and 40 other-food negatives, but no pasta positives. It cannot validate pasta recall. Three-class macro IoU below excludes pasta; pasta false positives remain reported.

| Food | v1 precision | v3 precision | v1 recall | v3 recall | v1 false positives | v3 false positives | v1 pixel IoU | v3 pixel IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rice | 51.6% | 92.3% | 80.0% | 60.0% | 15 | 1 | 45.9% | 44.4% |
| broccoli | 61.8% | 90.9% | 100.0% | 95.2% | 13 | 2 | 62.2% | 82.0% |
| chicken | 36.6% | 46.4% | 57.7% | 50.0% | 26 | 15 | 32.7% | 26.9% |

Three-observed-class macro IoU: **46.9% → 51.1%**. Negative-image false alarms: **26/40 → 13/40**. Pasta false positives: 14 → 4; pasta recall remains unmeasured on this final subset.

**Interpretation:** false-positive reduction is real on this sample, but it trades away some recall. Chicken pixel overlap decreased, so this is not a general chicken-segmentation accuracy win. A reviewed workflow and manual correction remain necessary. More representative chicken masks and independently labeled post-meal cafeteria plates are still needed. The chicken/duck source proxy is unchanged.

CPU prediction median: 89.0 ms, p95 102.6 ms, excluding app contours and network. It runs two YOLO checkpoints and a compact classifier, so it is slower than v1 (13.4 ms median).

## Earlier-set regression check

On the previously evaluated 188-image set, v3 macro IoU is 46.0%; other-food false alarms are 19/70. These are regression results, not another untouched test claim.

## Reproduction and artifacts

`ml/refine_gate.py` searches validation data only. `ml/prepare_final_holdout.py` records the unused image selection in `ml/final-holdout-manifest.json`. `ml/evaluate_v3.py` calls the same ensemble predictor used by the backend. `ml/promote_v3.py` copies verified component hashes into the active model folder. Restart the backend after promotion.

Active components and selection are recorded in `docs/model-metadata.json`. Original v1, candidate v2, presence classifier, and v3 artifacts remain separately available. All weights remain local. Unmeasured food weight is always null/unknown.

```sh
.venv/bin/python ml/refine_gate.py
.venv/bin/python ml/prepare_final_holdout.py
.venv/bin/python ml/evaluate_v3.py
.venv/bin/python ml/report_v3.py
.venv/bin/python ml/promote_v3.py
```

Paired image bootstrap (1,000 resamples, seed 127): 95% percentile interval for the three-class macro-IoU change is -6.3 to 13.9 percentage points. This finite-sample interval does not account for source clustering or domain shift.
