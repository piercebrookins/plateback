# Frontier benchmark expansion

The displayed benchmark hides Gemini at the user’s request; all raw responses and its measured scores remain in the full results. The default view contains v4, selected v9 and all non-Gemini external models. A toggle reveals earlier local experiments. No score was deleted or changed to alter rankings.

## Added models

Fixed before calls: GPT-6.1 Sol, Claude Sonnet 5.5 and Grok 4.7. Availability, image support, pricing and reasoning requirements were taken from the [OpenRouter model catalog](https://openrouter.ai/api/v1/models), saved locally as artifacts/benchmark/openrouter-model-catalog-expansion.json.

Each model received the same original 38 Nutrition5k presence images and 12 FoodSeg103 mask images, identical task prompts and 512px JPEG inputs. Reasoning is low, with 2,400 total output tokens. Historical models retain their original configurations. This is an economical constrained pipeline comparison, not an exhaustive test of each model’s best possible settings. No best-of selection or outcome-based model exclusion.

| Model | Mask IoU | Presence macro F1 | Valid outputs | Reported cost |
|---|---:|---:|---|---:|
| openai/gpt-6.1-sol | 56.32% | 80.09% | 38/38 presence; 12/12 parsed masks | $0.092024 |
| anthropic/claude-sonnet-5.5 | 25.51% | 77.68% | 34/38 presence; 6/12 parsed masks | $0.081912 |
| x-ai/grok-4.7 | 36.09% | 84.15% | 38/38 presence; 12/12 parsed masks | $0.174148 |

Missing or invalid outputs count as failures. Masks use the same transport-only parsing as historical rows (JSON fences may be removed; no geometry repair). Presence follows the original strict JSON parser. Sonnet produced 34/38 valid presence outputs and 6/12 parsed masks; its lower score includes these format failures.

## The Opus claim

V4 mask IoU is 54.118998%, versus Opus 5.5 at 53.748052%: a +0.370947 percentage-point estimate on 12 plates. A 2,000-draw paired image bootstrap gives a 95% percentile interval of −12.784984 to +10.830714 percentage points. This interval includes zero: the pilot does not establish a reliable or general advantage. Selected v9 scores 52.649769%, below Opus; it is selected for preservation of chicken recall on the larger regression sets. New GPT-6.1 Sol exceeds v4 on this pilot and remains displayed.

## Accounting and configuration correction

The initial GPT-6.1 Sol and Grok configuration attempted to disable mandatory reasoning. Six HTTP 400 requests were explicitly rejected before provider routing (provider_name null). These are archived in expansion-rejected-configuration and excluded from model-quality scoring. The requests were corrected to low reasoning before the valid benchmark; no unknown outcome was retried. Their $0.60 reservations remain counted conservatively.

Reported cost of the three completed 50-image runs: **$0.348084**. This run’s accounted cost including retained reservations: **$0.948084**, within its immutable $1.50 ceiling. Total accounted: **$2.57896159 / $5**. These estimates are not a provider invoice.

The public pilot has been repeatedly inspected during local development; it is not a fresh test. FoodSeg labels chicken/duck jointly; Nutrition5k ingredients are a distinct presence task. Pretraining overlap is unknown. No food weights are inferred.

## Verification

38 tests pass; frontend production build passes. Default and expanded views retain Gemini only in the downloadable data. Scores and raw responses are under artifacts/benchmark/. Reproduce scoring without API calls with `.venv/bin/python ml/score_frontier_expansion.py`.
