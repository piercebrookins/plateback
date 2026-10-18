# Collect a target-domain extension

The current benchmark uses independent FoodSeg103 publisher masks. For a future cafeteria benchmark, collect explicit permission to photograph plates, exclude faces and identifying backgrounds, and record a physical-serving ID plus capture-session ID before splitting.

Use four target labels (rice, pasta, broccoli, chicken) and unknown. Label contiguous visible servings rather than individual rice grains. Keep separate islands as separate components; preserve holes and do not annotate occluded food. Exclude duck from chicken unless independently identified. Have a human verify poultry identity and mask boundaries. Raw cloud suggestions are not independent test truth.

Group by serving/session and keep near-duplicate views together. Choose validation settings before looking at a final test set. Include unfamiliar foods, genuinely empty plates, utensils, touching similar-color foods, and varied laptop lighting. Export normalized full-image polygons or semantic PNG masks; keep original photos and annotation provenance.

Do not measure mass from masks. Use a tared scale and record a real net reading, or leave weight null. Track capture-to-confirm time and human correction rates separately from raw model metrics.
