---
license: apache-2.0
size_categories:
- n<1K
task_categories:
- image-segmentation
task_ids:
- semantic-segmentation
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype: image
  - name: classes_on_image
    sequence: int64
  - name: id
    dtype: int64
  splits:
  - name: train
    num_bytes: 1140887299.125
    num_examples: 4983
  - name: validation
    num_bytes: 115180784.125
    num_examples: 2135
  download_size: 1254703923
  dataset_size: 1256068083.25
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
---

# Dataset Card for FoodSeg103

## Table of Contents
- [Dataset Card for FoodSeg103](#dataset-card-for-foodseg103)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Dataset Structure](#dataset-structure)
    - [Data categories](#data-categories)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Refinement process](#refinement-process)
      - [Who are the annotators?](#who-are-the-annotators)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)

## Dataset Description

- **Homepage:** [Dataset homepage](https://xiongweiwu.github.io/foodseg103.html)
- **Repository:** [FoodSeg103-Benchmark-v1](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1)
- **Paper:** [A Large-Scale Benchmark for Food Image Segmentation](https://arxiv.org/pdf/2105.05409.pdf)
- **Point of Contact:** [Not Defined]

### Dataset Summary

FoodSeg103 is a large-scale benchmark for food image segmentation. It contains 103 food categories and 7118 images with ingredient level pixel-wise annotations. The dataset is a curated sample from [Recipe1M](https://github.com/facebookresearch/inversecooking) and annotated and refined by human annotators. The dataset is split into 2 subsets: training set, validation set. The training set contains 4983 images and the validation set contains 2135 images.

### Supported Tasks and Leaderboards

No leaderboard is available for this dataset at the moment.

## Dataset Structure

### Data categories

| id | ingridient |
| --- | ---- |
| 0 | background |
| 1 | candy |
| 2 | egg tart |
| 3 | french fries |
| 4 | chocolate |
| 5 | biscuit |
| 6 | popcorn |
| 7 | pudding |
| 8 | ice cream |
| 9 | cheese butter |
| 10 | cake |
| 11 | wine |
| 12 | milkshake |
| 13 | coffee |
| 14 | juice |
| 15 | milk |
| 16 | tea |
| 17 | almond |
| 18 | red beans |
| 19 | cashew |
| 20 | dried cranberries |
| 21 | soy |
| 22 | walnut |
| 23 | peanut |
| 24 | egg |
| 25 | apple |
| 26 | date |
| 27 | apricot |
| 28 | avocado |
| 29 | banana |
| 30 | strawberry |
| 31 | cherry |
| 32 | blueberry |
| 33 | raspberry |
| 34 | mango |
| 35 | olives |
| 36 | peach |
| 37 | lemon |
| 38 | pear |
| 39 | fig |
| 40 | pineapple |
| 41 | grape |
| 42 | kiwi |
| 43 | melon |
| 44 | orange |
| 45 | watermelon |
| 46 | steak |
| 47 | pork |
| 48 | chicken duck |
| 49 | sausage |
| 50 | fried meat |
| 51 | lamb |
| 52 | sauce |
| 53 | crab |
| 54 | fish |
| 55 | shellfish |
| 56 | shrimp |
| 57 | soup |
| 58 | bread |
| 59 | corn |
| 60 | hamburg |
| 61 | pizza |
| 62 |  hanamaki baozi |
| 63 | wonton dumplings |
| 64 | pasta |
| 65 | noodles |
| 66 | rice |
| 67 | pie |
| 68 | tofu |
| 69 | eggplant |
| 70 | potato |
| 71 | garlic |
| 72 | cauliflower |
| 73 | tomato |
| 74 | kelp |
| 75 | seaweed |
| 76 | spring onion |
| 77 | rape |
| 78 | ginger |
| 79 | okra |
| 80 | lettuce |
| 81 | pumpkin |
| 82 | cucumber |
| 83 | white radish |
| 84 | carrot |
| 85 | asparagus |
| 86 | bamboo shoots |
| 87 | broccoli |
| 88 | celery stick |
| 89 | cilantro mint |
| 90 | snow peas |
| 91 |  cabbage |
| 92 | bean sprouts |
| 93 | onion |
| 94 | pepper |
| 95 | green beans |
| 96 | French beans |
| 97 | king oyster mushroom |
| 98 | shiitake |
| 99 | enoki mushroom |
| 100 | oyster mushroom |
| 101 | white button mushroom |
| 102 | salad |
| 103 | other ingredients |

### Data Splits

This dataset only contains two splits. A training split and a validation split with 4983 and 2135 images respectively.

## Dataset Creation

### Curation Rationale

Select images from a large-scale recipe dataset and annotate them with pixel-wise segmentation masks.

### Source Data

The dataset is a curated sample from [Recipe1M](https://github.com/facebookresearch/inversecooking).

#### Initial Data Collection and Normalization

After selecting the source of the data two more steps were added before image selection.

1. Recipe1M contains 1.5k ingredient categoris, but only the top 124 categories were selected + a 'other' category (further became 103).
2. Images should contain between 2 and 16 ingredients.
3. Ingredients should be visible and easy to annotate.

Which then resulted in 7118 images.

### Annotations

#### Annotation process

Third party annotators were hired to annotate the images respecting the following guidelines:

1. Tag ingredients with appropriate categories.
2. Draw pixel-wise masks for each ingredient.
3. Ignore tiny regions (even if contains ingredients) with area covering less than 5% of the image.

#### Refinement process

The refinement process implemented the following steps:

1. Correct mislabelled ingredients.
2. Deleting unpopular categories that are assigned to less than 5 images (resulting in 103 categories in the final dataset).
3. Merging visually similar ingredient categories (e.g. orange and citrus)

#### Who are the annotators?

A third party company that was not mentioned in the paper.

## Additional Information

### Dataset Curators

Authors of the paper [A Large-Scale Benchmark for Food Image Segmentation](https://arxiv.org/pdf/2105.05409.pdf).

### Licensing Information

[Apache 2.0 license.](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1/blob/main/LICENSE)

### Citation Information

```bibtex
@inproceedings{wu2021foodseg,
	title={A Large-Scale Benchmark for Food Image Segmentation},
	author={Wu, Xiongwei and Fu, Xin and Liu, Ying and Lim, Ee-Peng and Hoi, Steven CH and Sun, Qianru},
	booktitle={Proceedings of ACM international conference on Multimedia},
	year={2021}
}
```
