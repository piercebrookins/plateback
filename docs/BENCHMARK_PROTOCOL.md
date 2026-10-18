# Frozen specialist/generalist benchmark

Written before model calls and result inspection. All costs share PlateBack's persistent $5 ledger. No paid GPUs. The example screenshot motivates a hypothesis; its fraud-detection results are not evidence about food recognition.

Tasks are scored separately:

1. **Category presence:** predict rice, pasta, broccoli, and chicken from the same full image. Nutrition5k-derived official test images provide independent ingredient labels; local YOLO presence is the union of retained masks. No cloud model receives ground truth, local predictions, crops, or nutrition values. Exact ingredient mapping is recorded. Pasta salad is pasta; rice noodles are not rice; chicken sausage is excluded. This is ingredient presence, not proof that every ingredient is visually identifiable.
2. **Segmentation:** on a deterministic subset of 12 fresh FoodSeg103 test IDs, cloud models return normalized contour polygons; local models return native masks. Score category-union pixel IoU including misses and false positives. FoodSeg103 chicken/duck is explicitly a proxy. This measures the end-to-end prompted polygon interface, not a claim that these generalist APIs expose dedicated segmentation heads.

Comparators: the original local v1, validation-selected local v2, off-the-shelf Qwen3.5-9B, GPT-6 Luna, Claude Opus 5.5, and Gemini 4 Argon, using catalog-confirmed OpenRouter IDs. One prompt, one response per image/model/task; no test-time prompt tuning, best-of sampling, or silent retries. Invalid JSON, mismatched schemas, timeouts, and API failures are counted as failed examples and all-negative predictions, and their rate is reported. Stop a provider after three consecutive service errors and report missing coverage rather than replacing it silently.

Local thresholds are frozen on validation, before these comparisons. Nutrition5k test images never enter training. Public benchmarks may have appeared in generalist pretraining; contamination cannot be ruled out. Small stratified samples support a pilot comparison, not a universal frontier-superiority claim. Report quality, per-class precision/recall, false positives, measured latency, cost, and uncertainty separately. Synthetic dashboard observations are never evaluation ground truth.

Model calls resize the image to at most 512 pixels, cap output tokens, use JSON responses and disabled reasoning when supported, and retain raw response/usage evidence without keys. Pricing is taken from the saved OpenRouter catalog. Usage costs are settled from the provider's `usage.cost`; missing usage retains the pessimistic reservation. Endpoint errors do not trigger automatic retries.

## Endpoint setup correction (before scoring)

Opus 5.5 and Gemini 4 Argon rejected `reasoning.enabled=false` at the router before inference. Six image requests plus two diagnostic requests returned HTTP 400; their eight $0.10 reservations remain accounted conservatively. Raw setup evidence is retained separately. Those endpoints now use required reasoning at `effort: low` and a 2,400-token total output ceiling; Qwen and Luna use disabled reasoning and 600 presence / 1,800 segmentation tokens. Prompts, images, labels and scoring were not tuned. The difference in mandatory reasoning is reported as an endpoint configuration difference, and its cost/latency is included. Invalid model-generated JSON still counts as a scored failure; only router setup errors are separated from model accuracy.

## Secondary parser-robustness analysis

After inspecting format errors but before computing accuracy, add a clearly labeled secondary score that strips a surrounding Markdown JSON fence and accepts valid contours longer than the requested 40 vertices. Coordinates, labels and geometry are not corrected, clamped, inferred or re-prompted. Truncated JSON and malformed coordinates remain failures. Preserve the strict original-protocol score too. This post-hoc transport analysis prevents ordinary formatting quirks from being confused with visual recognition quality; it is not a new model trial or a basis for training/threshold selection. The in-app mask column uses the parser-normalized score and documents that choice.

A separate local presence specialist is also included: frozen MobileNetV3 Small ImageNet features with a locally trained four-label head, trained on 713 FoodSeg103 and 226 official-training Nutrition5k images. Checkpoint and class thresholds use only the 197 FoodSeg103 validation images. It does not produce segmentation masks and is not substituted for the mask model. No external-test labels or model scores informed its training.

The external split audit screened all 38 test images against 1,136 training/validation reference images (FoodSeg103 plus selected Nutrition5k training data), and screened within the external test subset. No pHash matches at Hamming distance ≤6 were found. This does not prove independent physical plates or eliminate generalist pretraining contamination. Details: `ml/nutrition-split-audit.json`.

## V3 exploratory follow-up

After the v2 local test exposed lost chicken recall, a validation-only policy search added an optional locally trained presence gate and chose the source checkpoint per class. V3 uses v2 for rice/pasta/broccoli and v1 for chicken gated at 0.35, with confidence 0.05 and minimum mask area 1%. Its previous 188-image and frontier-pilot comparisons are exploratory regression results. A separate unused 83-image final check was reserved; it has no eligible pasta-positive cases and is reported accordingly. No new cloud calls were made for this local follow-up.
