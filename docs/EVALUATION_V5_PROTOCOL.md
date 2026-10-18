# Chicken boundary refinement: frozen final protocol

The v4 experiment improved plate-level chicken detection and mask IoU, but the fresh 110-image test exposed a decline in **chicken pixel recall** relative to v3. It also did not beat Gemini on the 32-image comparison. Those sets are therefore development/regression evidence for v5; they are not reused as fresh tests.

V5 retains the same locally trained DINOv2 dense head, same 448px inference and unchanged rice/pasta/broccoli policies. Only chicken mask postprocessing changes. Confident, sufficiently large seed regions decide whether chicken is present. Each seed then expands within a connected lower-confidence chicken region. This cannot introduce chicken on an image that has no qualifying seed. It separates presence decisions from boundary recovery instead of discarding low-confidence food edges.

## Validation-only selection

Use the same 197 validation images. Grid: seed probabilities 0.60/0.65/0.70/0.75/0.80/0.85/0.90/0.95/0.98/0.99; seed-component image fractions 0.0005/0.002/0.005/0.01/0.015/0.02/0.03/0.04/0.05; growth probabilities 0.05/0.10/0.20/0.30/0.40/0.50/0.60/0.70/0.80/0.90, never above seed probability. Minimum component size remains 20 pixels.

Hard requirements: chicken presence recall >=42/68 (original v1), pixel recall >=85% (a deliberately stronger sensitivity target), false-positive images <=32 (v3), and IoU >=v3 validation IoU. Maximize chicken IoU among eligible rules. If no rule qualifies, stop; do not silently relax a requirement. The new test set and Gemini output are excluded from selection. Fast component-statistic scoring is verified against the exact backend masks.

## New final test

Before this search, reserve 32 previously unused publisher-test FoodSeg103 images: 16 chicken/duck positives and 16 hard negatives containing steak, pork, sausage, fried meat, lamb or fish, with none of the four target foods. Seed 151; exclude all earlier dataset identities and perceptual near duplicates (Hamming distance <=6), including duplicates within the new set. Native four-class masks are retained, but chicken metrics are primary; other foods lack balanced coverage.

The paired Gemini subset is the first eight images in each frozen group (16 total). Same model, image preprocessing, prompt, output limits and transport parser as the earlier comparisons. One request per image, no retries of cached/unknown outcomes. Report both plate-presence recall and pixel recall, precision, pooled chicken IoU and false-positive plates; bootstrap paired image differences with 2,000 resamples, seed 151. Do not infer overall four-food superiority from this chicken-focused test.

All data retains FoodSeg103's declared Apache-2.0 license and chicken/duck taxonomy. No additional weight fitting, no paid GPU and no synthetic mask labels. Unknown overlap with public backbone/frontier pretraining remains a limitation. This is still public served-food imagery, not a validation of tiny real dining-hall leftovers.

All calls share the original $5 ledger. Before dispatch, 16 full $0.10 reservations plus existing accounting total less than $5. Food weight remains unknown unless measured.
