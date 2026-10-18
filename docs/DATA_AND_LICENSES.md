# Data provenance and licensing

Retrieved 2026-10-17. Local prototype; no dataset images or credentials are bundled into the frontend.

- Dataset: **FoodSeg103**, Wu, Fu, Liu, Lim, Hoi and Sun, *A Large-Scale Benchmark for Food Image Segmentation*, ACM MM 2021. [Publisher repository](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1), [paper](https://arxiv.org/abs/2105.05409), [homepage](https://xiongweiwu.github.io/foodseg103.html).
- Download mirror: [EduardoPacheco/FoodSeg103](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103). Its dataset card explicitly declares Apache-2.0; the publisher repository also declares Apache-2.0. Saved declarations and license are in `docs/licenses/`. This records the distributors' declared license; it is not an independent audit of every underlying web image's chain of rights. Do not imply that photographs are owned by PlateBack.
- Downloaded first publisher training shard (1,661 source rows) and publisher test split (the mirror names it `validation`). Only the selected subset is used. Exact download URLs and upstream LFS SHA-256 checksums are in `ml/source_manifest.json`.
- `ml/manifest.json` records source image IDs, byte SHA-256, publisher splits, selected classes and local split assignments. `ml/excluded_duplicates.json` records near-duplicates excluded across splits. `ml/split_audit.json` records perceptual screening.
- Mapping: 66 → rice, 64 → pasta, 87 → broccoli, 48 → **chicken/duck proxy**. Do not merge `noodles` into pasta or `fried meat` into chicken. The app's `chicken` field is a user-facing target class; model output from class 48 requires review and is **not a validated chicken-only classifier**.
- Reference labels are publisher annotations, independent of PlateBack's predictions and OpenAI review. Test masks preserve holes and disconnected areas. Training converts connected regions to YOLO polygons and therefore fills holes; evaluation uses original semantic masks and counts that error.
- Split procedure: seed-42 source-order shuffle, class-balanced caps, 20 other-food negative examples per source split, train-image hash modulo five for validation. Keep publisher test images out of training and model selection. Screen cross-split 63-bit perceptual hashes at Hamming distance ≤6. There are no serving/session identifiers in the mirror: grouped real-capture independence cannot be established. Public-food photographs are not independently collected cafeteria leftover plates.
- Image resize: longest side ≤640, mask nearest-neighbor; training input 320. No cloud-generated masks are used as truth.

## Model and software

[Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) distributes its code and model assets under AGPL-3.0, with commercial licensing alternatives. PlateBack's application code and fine-tuned derivative weights are provided under AGPL-3.0; preserve the license and provide corresponding source when distributing or offering a network service under that license. Base checkpoint: `yolo11n-seg.pt`, downloaded by the official Ultralytics package. Fine tuning does not remove its license obligations.

Dataset license and model license are separate. Public data and model weights remain local under ignored `data/` and `artifacts/`; share only deliberately with attribution. The original project-planning documents retain their existing authorship.

## Expanded v2 dataset

The second training shard is now included, with its upstream SHA-256 verified in `ml/source_manifest.json`. `ml/prepare_v2.py` uses seed 77, preserves old training/validation membership, and screens additions against all old splits and previously accepted images using the same perceptual distance ≤6 rule. Exact duplicates are also excluded. No publisher test images enter training or validation.

| Split | Images | Rice | Pasta | Broccoli | Chicken/duck | Other-food negatives |
|---|---:|---:|---:|---:|---:|---:|
| Training | 713 | 211 | 62 | 232 | 269 | 180 |
| Validation | 197 | 55 | 22 | 70 | 68 | 50 |
| New test | 188 | 41 | 26 | 48 | 58 | 70 |

Class columns count images and can overlap. Negatives contain none of the four source classes, but do contain other foods; they are randomly selected under a cap, not mined by model errors and not verified empty plates. New test images exclude all 88 previous test IDs and near-duplicates of reserved images. The old test set is now a regression benchmark, not an untouched holdout. `ml/manifest-v2.json` and `ml/excluded-v2.json` preserve selection and exclusion records. Local training input increases to 416 pixels; semantic masks remain at longest side ≤640. Source licensing and the chicken/duck proxy limitation above still apply.

## Nutrition5k external ingredient benchmark

The requested [mmathys/food-nutrients](https://huggingface.co/datasets/mmathys/food-nutrients) mirror contains 3,260 cafeteria plate images and ingredient annotations from [Google Research Nutrition5k](https://github.com/google-research-datasets/Nutrition5k). The mirror's generic `cc` tag is resolved through the original publisher's explicit [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) grant. Attribution: Thames et al., *Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food*, CVPR 2021. The original publisher's RGB train/test lists are downloaded from its public storage bucket; incremental views of a plate are grouped by the publisher.

The selected subset contains 226 official-training images and 38 official-test images (10 rice, 2 pasta, 12 broccoli, 10 chicken positive images, plus 12 other-food negatives; categories overlap). Selection is deterministic, seed 93. `ml/nutrition-manifest.json` records image hashes, original URLs, exact ingredient names and licenses; `ml/nutrition-label-map.json` records mapping. This dataset has no segmentation masks and therefore is not used as pixel-mask ground truth. Its nutrition and mass values are not imported into the app or used to infer leftover weight. The mirror card appears to swap mass/calorie descriptions; these fields are deliberately unused.

The test subset is external ingredient-presence evaluation, never segmentation training data. The tiny pasta-positive subset limits per-class conclusions. Served-food images are not post-meal waste observations. Fictional dining-hall records remain separate and never become benchmark labels.

The 226 official-training Nutrition5k images were also used for a separate local presence classifier: frozen Torchvision MobileNetV3 Small features plus a trained multilabel linear head. It does not create masks. Torchvision source is BSD-3-Clause; pretrained ImageNet weights come from the official PyTorch model URL recorded in metadata. The FoodSeg103-derived segmentation weights remain AGPL-3.0. Do not conflate the classifier's ingredient labels with pixel annotations.

## Semantic v4 expansion

The third FoodSeg103 publisher training shard was downloaded from the same Apache-2.0-declared mirror and verified against its published LFS SHA-256. `ml/source_manifest.json` records the URL and hash. No new dataset license is inferred from a model repository. The expanded selection contains 1,732 training images and preserves the existing validation set; 110 previously unused third-shard images form a separately reserved local test set. This local split is not the official publisher test split. See `ml/manifest-v4.json`, `ml/independent-audit-v4.json`, and [the frozen evaluation protocol](EVALUATION_V4_PROTOCOL.md).

The new architecture is Torchvision DeepLabV3 MobileNetV3 Large, with official `COCO_WITH_VOC_LABELS_V1` initialization from `https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth`. It has no FoodSeg-specific pretrained labels. Torchvision source is BSD-3-Clause; the existing local license copy remains in `docs/licenses/torchvision-BSD-3-Clause.txt`. The original YOLO checkpoint remains subject to AGPL-3.0 when used as an explicit class fallback. Training and evaluation scripts retain the chicken/duck taxonomy caveat. Nutrition5k image-level ingredient annotations were not treated as segmentation ground truth.

## DINOv2 dense-head candidate

The second v4 approach uses Meta's official `dinov2_vits14_pretrain.pth`, a frozen ViT-S/14 backbone, with a small dense mask head trained locally on FoodSeg103. Meta's [DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) declares Apache-2.0 for the models. Code is pinned to `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; local copies of its license and model card are in `docs/licenses/dinov2-Apache-2.0.txt` and `docs/licenses/dinov2-MODEL_CARD.md`. The backbone itself was pretrained externally on LVD-142M; only the dense head is trained in this project. Unknown overlap between that public pretraining corpus and FoodSeg images is disclosed. Frozen feature extraction and head fitting both run on local CPU, with no paid inference/training service. The 448px variant changes inference resolution only; it is not a separate training run.

### V5 follow-up: Nutrition5k domain adaptation

`ml/nutrition-v6-manifest.json` adds 420 decoded-image-deduplicated Nutrition5k examples from the same public mirror and original CC-BY-4.0 source: 300 training, 60 validation, and 60 official-test examples. The 226 earlier training examples remain, giving 526 Nutrition training images (217 chicken ingredient positives). The new validation/test sets each contain 30 chicken ingredient positives and 30 negatives. The independent audit compares all new images with 2,694 earlier unique images and with each other at perceptual hash distance <=6; no matches remain. Official split membership and saved-image SHA256 values are checked.

This round uses ingredient presence only. It creates no Nutrition5k pixel masks, does not use served grams as leftover mass, and does not train on either the old 38-image regression set or the new 60-image test set. The three experiments use the existing Apache-2.0 DINOv2 backbone, frozen, with locally fitted linear/MLP presence heads or a dense head with weak-label MIL and FoodSeg103 pixel replay. Chicken/duck proxy labels and unknown pretrained-data overlap remain limitations. Files and exact counts: `artifacts/evaluation-v6/independent-data-audit.json`, `artifacts/models/chicken-v6/metadata.json`.

### V7 partial backbone fine-tuning

No new dataset was introduced. The final two DINOv2 transformer blocks, final normalization and dense head were updated locally from MIL-v6, using the same protected FoodSeg103 and Nutrition5k training/validation splits. Six epochs on Apple MPS updated 3,601,157 parameters. Earlier transformer blocks remained frozen. A 50/50 parameter interpolation candidate is also retained. Source licenses, chicken/duck limitations, ingredient-only supervision and unknown pretrained-data overlap remain unchanged. No API fine-tuning or paid GPU was used.
