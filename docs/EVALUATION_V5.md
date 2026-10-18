# Local model experiments: stronger masks, unresolved recall regressions

**Active scanner model: v3, unchanged. V5 is an experimental candidate.** The new chicken-focused holdout improves strongly, but earlier-set recall regressions fail the complete deployment gate. No threshold was changed after the final test.

## What changed

- Expanded pixel-annotated FoodSeg103 training from 713 to 1,732 images, including 748 chicken/duck examples (previously 269) and 400 target-negative images. All three public training shards are now used; known holdouts and perceptual near duplicates remain excluded.
- Tried full DeepLabV3/MobileNetV3 fine-tuning locally on MPS. Stopped after 14 completed epochs (24 planned), preserving best epoch 12, after the stronger DINO candidate passed validation.
- Trained a dense segmentation head over frozen Apache-2.0 DINOv2 ViT-S/14 features entirely on CPU: 40 epochs, best epoch 21, about 9.8 minutes including feature extraction. Compared 336px and 448px inference on validation; 448px won. This is head-only transfer learning, not full DINO backbone fine-tuning.
- Added confidence-based seed detection and connected boundary expansion for chicken. The trained parameters are unchanged between v4 and v5. Separate seed and boundary decisions recover more food pixels without introducing presence where no seed exists.

## Fresh chicken-focused holdout: 32 images

16 previously unused publisher-test chicken/duck images and 16 previously unused meat/fish negatives. Independent decoded-image pHash audit found no matches against 2,662 previous images, or within this new set, at Hamming distance <=6. This is a balanced, small public-data test, not natural cafeteria prevalence.

| Model | Chicken IoU | Presence precision | Presence recall | Pixel recall | False positives / 16 negatives |
|---|---:|---:|---:|---:|---:|
| Original YOLO v1 | 33.6% | 54.5% | 75.0% | 67.4% | 10 |
| Previous ensemble v3 | 39.8% | 64.3% | 56.2% | 65.0% | 5 |
| Semantic v4 candidate | 77.6% | 83.3% | 93.8% | 91.3% | 3 |
| Boundary-refined v5 | 68.5% | 83.3% | 93.8% | 97.0% | 3 |

V5 minus v3 chicken IoU: paired-bootstrap 95% interval **+9.3 to +47.6 percentage points** (2,000 image resamples, seed 151).

## Gemini comparison: same 16 images

Eight positives and eight hard negatives from the frozen new holdout. Same original prompt, full-image input, model and output budget; no best-of calls or image-specific prompt tuning. All 16 Gemini outputs were usable after the predefined transport parser.

| Model | Chicken IoU | Presence recall | Pixel recall | False positives |
|---|---:|---:|---:|---:|
| Previous ensemble v3 | 39.9% | 62.5% | 70.0% | 4 |
| Boundary-refined v5 | 78.3% | 100.0% | 99.3% | 1 |
| Gemini 4 Argon | 90.1% | 87.5% | 93.9% | 0 |

Local minus Gemini chicken IoU: 95% paired-bootstrap interval **-29.1 to -0.3 percentage points**. **The local model does not beat Gemini on segmentation IoU.** It recovers more chicken pixels and finds one additional positive image on this subset, but overcovers more non-chicken pixels. Higher recall is not equivalent to better masks.

## Why v3 remains active

The final holdout is not the only evidence. These earlier sets motivated follow-up work and are now explicitly regression checks for v5:

| Earlier 110-image set | v3 | v5 |
|---|---:|---:|
| Chicken IoU | 33.4% | 42.9% |
| Chicken presence recall | 64.5% | 67.7% |
| Chicken pixel recall | 77.3% | 74.0% |
| Chicken false-positive plates | 21 | 16 |

On the separate 38-image Nutrition5k ingredient-presence regression set, chicken recall is **30% for v5 versus 40% for v3** (only ten chicken-positive examples). Nutrition5k has ingredient labels, not segmentation masks; its weights/nutrients are not leftover measurements. Overall presence macro-F1 improves, but the chicken recall drop cannot be hidden by that average.

The 110-image regression also has a small broccoli pixel-recall decline versus original v1 (about one percentage point), while broccoli presence recall stays 100%. V5 preserves v4 policies for rice, pasta and broccoli.

**Deployment decision:** keep the existing v3 checkpoint active, retain the trained candidates and expose all benchmark results in the Models dashboard. The complete gate fails on earlier chicken recall checks. No claim of loss-free recall across datasets is made.

## Frozen policy and limits

Chicken rule: seed probability ≥0.75, seed component ≥4% of image area, connected growth at probability ≥0.1. Validation pixel-recall target was 85%, with original v1 presence recall and v3 false-positive/IoU bounds. These are uncalibrated network scores, not probability-of-correctness guarantees. The relatively large seed requirement can miss small leftovers; lowering it without new validation would revive false positives.

The source poultry category combines chicken and duck. It cannot validate chicken-only species recognition. Public-source overlap with DINO/frontier pretraining is unknown. No model was trained on the fictional dining dashboard, inferred mask labels, inferred mass, or final holdout labels. These data do not establish real post-meal cafeteria performance.

## Cost, artifacts and reproduction

**Total API accounting: $1.621076 / $5**, including $0.80 retained reservations from earlier setup failures. New calls this iteration: $0.256532 for the 32-image four-food comparison and $0.105646 for the final 16-image chicken comparison. All model training was local; no paid GPU was used. Unmeasured food weight remains null/unknown.

Candidate: `artifacts/models/semantic-v5/best.pt` plus its settings and metadata. Active model: `artifacts/models/plateback/` (v3). The DINO code dependency is pinned locally under `data/source/dinov2/`; it does not download weights during inference.

Candidate checkpoint SHA-256: `d43540ab54ff88403edfe2a6ebe129fbc70fa560d84dc088712e26b364d398bc`.

Portable evidence: [final chicken results](chicken-final-summary.json), [v4 four-food results](EVALUATION_V4.md), [all-model earlier pilot](BENCHMARK_RESULTS.md), [v5 protocol](EVALUATION_V5_PROTOCOL.md), [data and licenses](DATA_AND_LICENSES.md). Raw per-image counts, responses and all failed gates are retained in `artifacts/benchmark-chicken-final/`, `artifacts/evaluation-v5/`, and `artifacts/benchmark/`.

These commands assume the retained v1/v3 checkpoints and earlier dataset artifacts from the README build sequence. For a new workspace, reproduce that base pipeline first.

```sh
.venv/bin/python ml/download.py
.venv/bin/python ml/prepare_v4.py
.venv/bin/python ml/train_semantic.py --epochs 24 --stop-after-epochs 14
.venv/bin/python ml/train_dino.py
.venv/bin/python ml/select_semantic.py
.venv/bin/python ml/evaluate_semantic.py --evaluate-only
.venv/bin/python ml/prepare_chicken_holdout.py
.venv/bin/python ml/audit_chicken_holdout.py
.venv/bin/python ml/refine_chicken_boundary.py
.venv/bin/python ml/benchmark_chicken_final.py  # paid only for missing cache entries
.venv/bin/python ml/evaluate_v5_regression.py
.venv/bin/python ml/score_benchmark.py
.venv/bin/python ml/evaluate_chicken_final.py
.venv/bin/python ml/report_chicken_final.py
.venv/bin/python scripts/smoke_semantic_candidate.py  # isolated DB; no API calls
```
