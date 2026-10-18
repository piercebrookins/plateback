# Model evaluation — 2026-10-17

This is a small public-data prototype benchmark, not a cafeteria deployment result. Raw predictions were scored before any user or cloud correction.

## Training and independence

Fine tuning ran locally on MPS for 266.4 seconds. Base: YOLO11n-seg, 320px input, frozen first 10 modules, batch 8, seed 42. Requested 35 epochs; early stopping ended at epoch 32, with validation-selected best weights from epoch 22. No paid GPU or training API was used.

209 training images, 52 validation images, **88 final test images**. Test uses publisher labels held out from fine tuning. One near-duplicate test image (`test_55`, visually duplicated by `train_652`) was excluded before metric computation. Source IDs, hashes, split assignments and the duplicate audit are saved in `ml/`. Test class counts can overlap.

The confidence threshold **0.2** maximized validation macro category-union IoU among 0.1–0.5. Test labels were not used to choose model weights or confidence. MPS nondeterminism warnings are retained in the training log.

## Held-out results

| Class | Reference plates | Presence precision | Presence recall | Mask IoU |
|---|---:|---:|---:|---:|
| rice | 19 | 45.5% | 78.9% | 34.9% |
| pasta | 20 | 73.9% | 85.0% | 62.7% |
| broccoli | 31 | 73.2% | 96.8% | 75.6% |
| chicken_duck_proxy | 32 | 50.0% | 78.1% | 38.3% |

Macro category IoU: **52.9%**. Pixels are unioned within each class; intersections and unions are pooled across all test images. Missed masks and false-positive masks count against the result. Presence counts a food once per plate. Empty/empty classes do not receive a perfect score.

On **20 other-food images**, the model hallucinated at least one target food on **12**. These are unfamiliar-food negatives, **not independently verified empty plates**. Empty-plate false-positive rate is unmeasured. The model should always be reviewed.

CPU prediction median **11.6 ms**, p95 **14.6 ms**, with 0 failed requests in the evaluator. This is a warmed model, excludes UI/upload time, and uses 4 PyTorch CPU threads. Separate fresh-process adapter testing took 1.32 s for the first model load/prediction and about 17–23 ms for subsequent images. Do not treat these as capture-to-confirm times.

## What remains unknown

- Chicken-specific precision/recall: source labels combine chicken and duck. The poultry row is explicitly a proxy.
- Real leftover-plate domain performance; the public dataset contains composed food photographs and web-image biases.
- Physical serving/session independence: the source lacks grouping metadata. SHA and perceptual duplicate checks cannot prove scene independence.
- Empty plates, utensils and difficult laptop lighting as separate measured strata.
- Human correction rate, capture-to-confirm time and scale accuracy on real sessions. Dashboard correction counts from software tests are not a user study.
- Gemini accuracy and live connectivity, pending the supplied key. OpenAI smoke tests establish connectivity/schema handling only.

## Artifacts

- `artifacts/models/plateback/best.pt`, metadata and selected settings.
- `runs/food-v1/results.csv`, training args, checkpoints and loss curves.
- `artifacts/evaluation/test.json`, validation results, per-image raw polygons and compressed predicted bitmap masks.
- Independent reference masks: `data/datasets/plateback/masks/test/`.
- `artifacts/evaluation/examples.jpg`: deterministic lowest, one-third, two-thirds and highest per-image mean IoU examples. Empty/empty images are excluded from example ranking; use the aggregate negative-food counts for false-positive interpretation.
- `artifacts/local-smoke.json`: fresh-process local adapter results on mixed held-out images. Topology repair of pinched contours was validated after raw bitmap evaluation; model weights and threshold were unchanged.

## Budget and provider evidence

Real OpenAI review uses `gpt-4.1-mini-2025-04-14`, full plate plus reference-region crops. `artifacts/openai-smoke.json` records 720 input tokens, 96 output tokens, **$0.0004416 estimated cost**. Further hybrid smoke usage, if present, is recorded in the persistent SQLite ledger and `docs/VERIFICATION.md`. Gemini was not called because no key was provided.

## Sources

[FoodSeg103 publisher](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1), [dataset mirror and license declaration](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103), [Ultralytics segmentation](https://docs.ultralytics.com/tasks/segment/), [OpenAI image analysis](https://developers.openai.com/api/docs/guides/images-vision), [OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [Gemini image understanding](https://ai.google.dev/gemini-api/docs/image-understanding), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing).
