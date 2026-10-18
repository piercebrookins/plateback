# V8: hard-pixel training with distillation

This experiment freezes the DINO backbone and trains its existing dense head from MIL-v6. It uses 1,732 FoodSeg training images and 197 validation images. Nutrition5k knowledge is inherited from MIL-v6 and the unchanged v6 presence rescue. No regression image is a training example. No new data or cloud labels are purchased.

The loss combines the existing recall-weighted CE/Tversky objective, 0.35 times the mean cross-entropy of the hardest 20% of pixels, and 0.5 times KL divergence from the frozen v6 head. Training uses seed 181, 16 epochs, AdamW, horizontal feature/mask flips, and chicken/negative oversampling. Checkpoint selection uses minimum validation loss. Native-resolution calibration then maximizes IoU subject to v6 chicken presence recall, pixel recall, false-positive image count, and Nutrition validation recall/false-positive constraints. If none improves IoU, use unchanged v6.

After freezing weights and policy, evaluate the four existing FoodSeg regression sets and two Nutrition5k regression sets. They are repeatedly inspected regression sets, not fresh tests. Require no reduction in chicken presence/pixel recall or IoU and no additional false-positive images across all sets before claiming an improvement. Failure means retain the prior model, rather than tune to these outcomes.

For the Models page, score recent versions on the original shared 12-image mask / 38-image presence pilot and reuse saved cloud responses. Do not compare scores from mismatched sets. These public-data results cannot establish superiority over frontier models generally. FoodSeg chicken includes duck; Nutrition5k labels indicate ingredients, not masks. Pretraining overlap remains unknown. Unmeasured grams stay unknown.

All training and inference are local. New API spending for this experiment is $0; the shared $5 ledger is retained.
