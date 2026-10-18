# Recall-constrained local semantic segmentation

This experiment was specified before opening the new test scores. It follows the v3 chicken recall regression. Earlier benchmark results are development/regression evidence, not fresh tests.

## Data and split

- All three publisher FoodSeg103 training shards; the newly downloaded third shard is SHA-256 `7b16b566c273e049a10d651caf6d28951f6cdf730fba6596e832b451230d6ac1`.
- 1,732 training images: 388 rice, 72 pasta, 604 broccoli, 748 chicken/duck; counts overlap. Includes 400 images containing none of those four foods.
- The existing 197-image validation split stays fixed.
- 110 previously unused images from the third publisher **training** shard are reserved as a local test split before training: 20 rice, 20 pasta, 27 broccoli, 31 chicken/duck, 40 target-negative images. This is not the publisher test split.
- Every earlier validation/test image remains excluded from new training by identity and perceptual hash. `ml/independent-audit-v4.json` independently checks decoded on-disk images: no train/holdout matches at Hamming distance <=6, across 666 held-out references. Perceptual hashing is screening, not proof that all photographed meals are independent.
- New training data is pixel-annotated FoodSeg103. Nutrition5k ingredient labels are not fabricated into pixel masks. Its previously used images remain a separate regression presence test.

Taxonomy remains rice, pasta, broccoli, and FoodSeg103's **joint chicken/duck** category. This cannot establish chicken-only accuracy. Data is used under the publisher/mirror's declared Apache-2.0 license; provenance is recorded in `ml/source_manifest.json` and `docs/DATA_AND_LICENSES.md`.

## Model and training

Torchvision DeepLabV3 with a MobileNetV3 Large backbone, initialized from the official COCO/VOC-label checkpoint, not a checkpoint already trained on FoodSeg103. Five semantic outputs: background plus the four foods. Native raster masks retain holes instead of converting training targets into object polygons.

Local Apple MPS only; CPU inference. Fixed seed 141, 320-square training/inference, batch 8, 24 epochs, AdamW, cosine schedule, 0.00015 backbone / 0.0015 head learning rates. Weighted cross entropy plus Tversky loss penalizes missing food pixels more heavily (0.7 false-negative versus 0.3 false-positive weight); chicken-positive images receive 1.6x sampling weight. Mild color/brightness augmentation and horizontal flips. Pretrained batch-normalization running statistics remain frozen. All network weights otherwise train.

Best checkpoint is chosen by four-class pooled validation IoU. Post-training calibration searches each class independently over fixed pixel-confidence thresholds and component-area cutoffs. A candidate must retain both plate-presence recall and pooled pixel recall relative to **both** original v1 and deployed v3, with no more false-positive images than v1. Among eligible candidates, choose the best pooled IoU. There is no silent relaxation of a recall constraint. Baseline fallback is explicit; `constraints_satisfied` records whether it qualifies, and a failed chicken constraint blocks promotion.

Validation constraints do not guarantee equal recall on new images. Fresh test differences and uncertainty must be reported, including regressions.

## Fresh paired Gemini test

32 images are selected deterministically from the new 110-image local holdout: six unique selections per food plus eight negatives. Foods overlap, so positive counts may exceed six. Selection is frozen before candidate performance is available in `artifacts/benchmark-v4/manifest.json`.

Gemini 4 Argon through OpenRouter receives the same image and the unchanged prior polygon prompt, one request per image. Max output 2,400 tokens, low reasoning, 512px JPEG input, no best-of retries. Saved raw output is scored with the already documented transport parser: JSON fences can be stripped and contours up to 4,096 vertices are accepted. Geometry, food labels and coordinates are not repaired. Missing/invalid outputs count as failures. Strict-format scores remain available on the earlier pilot.

Primary comparison: four-class pooled pixel IoU, not JSON-format compliance. Also report class IoU, plate-presence precision/recall, pooled pixel precision/recall, false-positive images, and usable output count. Paired image bootstrap uses 2,000 resamples with seed 141. An interval crossing zero cannot establish superiority. Public benchmark contamination in frontier pretraining is unknown. These are served-food images, not a validation of post-meal dining-hall leftovers.

## Spending and sources

All paid requests use the existing persistent $5 ledger. All 32 fresh Gemini calls together cost $0.256532. Build accounting after them is $1.51542989, including old retained reservations. No paid GPUs or paid training data were used. No image-derived food mass is introduced.

- [Torchvision architecture and pretrained weight documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html)
- [FoodSeg103 dataset card and declared license](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103/blob/main/README.md)
- [Nutrition5k original source](https://github.com/google-research-datasets/Nutrition5k)

## Reproduce

```sh
.venv/bin/python ml/download.py
.venv/bin/python ml/prepare_v4.py
.venv/bin/python ml/audit_v4.py
.venv/bin/python ml/train_semantic.py
.venv/bin/python ml/train_dino.py
.venv/bin/python ml/select_semantic.py
.venv/bin/python ml/evaluate_semantic.py --evaluate-only
.venv/bin/python ml/benchmark_v4.py  # paid only for missing cached responses
.venv/bin/python ml/score_semantic_benchmark.py
```

## Second local approach, specified before test scoring

A second candidate trains a small dense MLP segmentation head over **frozen DINOv2 ViT-S/14 features**, entirely on CPU. This is head-only transfer learning, not full DINO backbone fine-tuning. DINOv2 code and weights are Apache-2.0 according to [Meta's model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md); code is pinned to commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. License/model card copies are retained locally. DINO's LVD-142M pretraining identities have not been audited against these public images; that pretraining-overlap uncertainty is explicit.

The same 1,732 training and 197 validation images are used. Features are extracted at 336px, a 384→128→5 pointwise head trains for 40 epochs on CPU, seed 142, weighted CE/Tversky, AdamW with cosine decay. Pasta images receive 3x sampling and chicken images otherwise 1.6x; feature/mask horizontal flips are paired. Head checkpoint selection uses 96px validation IoU, followed by the same full-resolution recall-constrained calibration as DeepLab.

Before test scoring, select one approach by **0.5 chicken IoU + 0.5 four-class macro IoU** on validation, among candidates passing every class's recall and false-positive constraints. If neither qualifies, label the chosen result experimental and block deployment. This architecture comparison and objective were defined while DeepLab was at epoch 7 and before DINO training results or any new test scores were inspected. Test responses remain outside all model and threshold selection.

`ml/select_semantic.py` preserves both candidates, records the decision, and places the selected candidate in `artifacts/models/semantic-v4/`. Then use `ml/evaluate_semantic.py --evaluate-only`, followed by the paired scorer. Polygon rendering unions each individual region after subtracting its own holes, so one region's hole cannot erase another region of the same food.

Before calibration, the chicken false-positive ceiling is tightened to the smaller of v1 and deployed v3 counts (32/197 validation images). Other classes retain the original v1 ceiling. Recall floors remain the maximum of both models. This prevents improving chicken recall by reverting to the original model's higher false-positive rate.

A fixed inference-resolution ablation also compares DINO at 336px and 448px using exactly the same trained parameters. This was specified at DINO head epoch 10 before full-resolution calibration or new test scoring. Both variants use the same recall constraints and validation objective; no test-time choice per image is allowed.

A secondary valid-geometry diagnostic compares local and Gemini only where Gemini returned a parseable mask. Empty-but-valid predictions remain included. This diagnostic was specified before inspecting new test scores; it separates geometry quality from unusable-output failures without replacing the full 32-image primary metric.

Validation search amendment before any test scoring: the original confidence grid stopped at 0.8 and component area at 1%. DINO chicken had high recall but exceeded the false-positive ceiling at every original setting. The grid was extended to 0.85/0.90/0.95/0.98 confidence and chicken component areas 1.5/2/3/5%; all recall and false-positive constraints remain unchanged. Component-statistic scoring is tested for exact equivalence to serving masks. The initial unsuccessful grid is retained as calibration-initial.json.

DeepLab was stopped after 14 completed epochs (24 requested), with best checkpoint epoch 12, once DINO336 passed every native-resolution validation constraint. No new test scores had been accessed. Reproduce its completed schedule with `ml/train_semantic.py --epochs 24 --stop-after-epochs 14`; the cosine schedule still spans 24 epochs. This is an early-stopped comparison, not a claim that a fully trained DeepLab could never improve.
