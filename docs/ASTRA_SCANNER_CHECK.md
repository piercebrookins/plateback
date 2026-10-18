# Astra scanner check

Astra is the default OpenAI model in Settings. Luna remains available as a cheaper option; PlateBack uses the semantic-v4 local checkpoint.

## Measured comparison

Six reused FoodSeg103 regression images were selected before Astra inference: the first three chicken-positive and first three chicken-negative examples in the existing 24-image scanner manifest. Luna's paid results were reused. Both models received the same contour prompt and image encoding, capped at 512 pixels. Luna used no reasoning and 2,000 output tokens; Astra used low reasoning and 4,000 output tokens.

| Measure | Luna | Astra |
| --- | ---: | ---: |
| Chicken mask IoU, pooled pixels | 28.68% | 79.30% |
| Chicken-positive images detected | 2/3 | 3/3 |
| Chicken detections on negative images | 1/3 | 0/3 |

Astra also predicted one broccoli region absent from the reference masks. This set has no positive rice, pasta or broccoli masks, so it does not establish performance on those foods. Six reused images cannot support a general accuracy claim. These scores appear separately from the main benchmark.

On the user's rice/protein photo, Astra outlined rice and omitted Luna's chicken labels on the other foods. On the first mixed plate, Astra returned no target masks. That eliminates unsupported pasta/broccoli labels but is not a recall success: the Nutrition5k ingredient record includes rice and chicken. No pixel ground truth exists for these two screenshots, so their mask IoU is unknown.

Eight Astra requests cost $0.23193. The comparison had a $1.20 run ceiling inside the existing $5 project ceiling. Unknown outcomes retain reservations. A subsequent live app smoke call cost $0.02608, bringing project accounting to $2.83825659. It returned rice and one chicken region on the rice/protein photo, unlike the first rice-only response. This variability is a remaining classification risk. The sample scan was confirmed to verify persistence, not to certify its labels.

## Reproduce and inspect

Run `.venv/bin/python scripts/compare_astra.py`, then `.venv/bin/python scripts/score_astra.py`. Existing results are cached; the script does not repay cached requests. The frozen manifest, raw polygons, per-class counts and visual comparison are in `artifacts/astra-comparison/`.

Official [Astra documentation](https://developers.openai.com/api/docs/models/gpt-6-astra) supports low reasoning, image input and structured output. [Standard pricing](https://developers.openai.com/api/docs/pricing) is $10 input and $50 output per million tokens; cached input is $1 and cache writes $12.50. The app reserves $0.50 for each bounded Astra request, then settles from returned usage. It sends no more than nine 512px images and permits at most 4,000 output tokens. No automatic retries or provider fallback.

The [vision guide](https://developers.openai.com/api/docs/guides/images-vision) notes errors in spatial localization. Schema-valid polygons can still be wrong. Review outlines before confirming; unmeasured weight remains unknown.

## Current food review interface

The scanner now uses the all-foods-review-1 prompt to name other foods as reviewable guesses. The scores above measure the earlier four-food prompt, not this expanded prompt. One live smoke on the user's plate returned pizza, chicken and carrots for $0.03705; these are unverified guesses, not measured accuracy. Browser checks verified grouping chicken pieces, confirmation, renaming, removal, undo and persistence. Model information and weight controls are hidden on the scanner; settings retains the model choice. Unknown food names are stored separately from the original four benchmark classes.
