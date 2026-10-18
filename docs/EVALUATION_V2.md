# Local model v2 evaluation

Both versions are evaluated on the same 188 newly reserved publisher test images. The v2 checkpoint and confidence/area policy were selected using validation only and frozen before test inference. The original 88-image test is reported separately as a regression set.

| Food | v1 IoU | v2 IoU | v1 precision | v2 precision | v1 recall | v2 recall | v1 false positives | v2 false positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Rice | 34.1% | 47.6% | 49.3% | 82.4% | 82.9% | 68.3% | 35 | 6 |
| Pasta | 44.1% | 38.3% | 42.9% | 100.0% | 69.2% | 38.5% | 24 | 0 |
| Broccoli | 62.0% | 73.5% | 71.4% | 89.6% | 93.8% | 89.6% | 18 | 5 |
| Chicken/duck proxy | 24.1% | 22.7% | 41.0% | 47.9% | 74.1% | 39.7% | 62 | 25 |

Macro category IoU: **41.1% → 45.5%**. Other-food negative images with any false target: **43/70 → 12/70**.
Paired image bootstrap (1,000 resamples, seed 77) gives a 95% percentile interval of -2.8 to 12.3 percentage points for the macro-IoU change. This reflects this finite image sample, not cafeteria deployment or source-group uncertainty.

## Protocol and interpretation

- IoU is intersection/union of all semantic pixels per category, pooled across the set. Misses and false masks both count; absent/absent categories do not inflate the score.
- Precision and recall are image/category presence metrics, not object-detection AP. False-positive counts are images with a predicted class absent from the reference.
- Ground truth is independent publisher annotation. Chicken means the source chicken/duck proxy; chicken-only accuracy remains unknown. Other-food negatives do not establish empty-plate performance.
- Policy search: confidence .15/.2/.3/.4/.5/.6/.7/.8 and minimum whole-image mask area 0/.002/.005/.01. Per class, maximize .55 × pooled IoU + .45 × presence F0.5 preferring validation recall at least .50 when feasible. The searched v2 chicken policies could not reach that target; the fallback maximized the objective and selected 38.2% validation recall. This deliberately trades some sensitivity for fewer false detections.
- Expanded validation macro IoU: 43.5% → 50.9%. Validation results are selection evidence, not unbiased generalization estimates.
- The model and API do not infer food weight. Unmeasured weight remains null/unknown.

## Frozen policy

```json
{
  "imgsz": 416,
  "confidence": 0.05,
  "per_class": {
    "rice": {
      "confidence": 0.5,
      "min_area_fraction": 0.01
    },
    "pasta": {
      "confidence": 0.5,
      "min_area_fraction": 0
    },
    "broccoli": {
      "confidence": 0.6,
      "min_area_fraction": 0
    },
    "chicken": {
      "confidence": 0.3,
      "min_area_fraction": 0
    }
  },
  "selection": "Validation only: maximize .55 category pooled IoU + .45 presence F0.5; prefer recall >= .50 when feasible in the search, otherwise maximize the objective. Class-specific thresholds; frozen before test."
}
```

## Runtime

CPU prediction-only latency (single run, 4 PyTorch threads): v1 median 15.0 ms / p95 17.3 ms; v2 median 20.8 ms / p95 27.2 ms. These exclude app contour conversion, network, and cold import costs.

## Artifacts and reproduction

`artifacts/evaluation-v2/` contains validation, calibration candidates, both fresh-test comparisons, both regression comparisons, cache identities, per-image metrics, and bootstrap results. `artifacts/models/baseline-v1/` preserves rollback weights and settings. Candidate training metadata and SHA-256 are beside `artifacts/models/candidate-v2/best.pt`.

```sh
.venv/bin/python ml/download.py
.venv/bin/python ml/prepare_v2.py
.venv/bin/python ml/train_v2.py
.venv/bin/python ml/evaluate_v2.py
.venv/bin/python ml/report_v2.py
```

Evaluation caches are bound to checkpoint hash/input size/split; a changed checkpoint requires a fresh cache directory. Reusing this test set for another model iteration makes it a regression benchmark. A future independent cafeteria set, including look-alike grains, bread, potatoes, fish and verified empty plates, is still needed.
