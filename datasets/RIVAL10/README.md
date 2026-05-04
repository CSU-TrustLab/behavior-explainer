# RIVAL10

**RIVAL10** (Robust Image Verification and Attribution Learning) is a benchmark dataset for evaluating image classifiers, particularly in the context of attribution and robustness. It is built on top of ImageNet and provides 10 classes with segmentation masks that distinguish the main object from the background.

## Structure

```
RIVAL10/
├── train/
│   ├── ordinary/       # Standard training images
│   └── entire_object_masks/  # Segmentation masks
├── test/
└── meta/
    ├── label_mappings.json
    ├── wnid_to_class.json
    └── train_test_split_by_url.json
```

## Classes

10 classes inherited from ImageNet (mapped via WordNet IDs).

## Download

> **The dataset images are not tracked in this repository.**
> Please download RIVAL10 from the official repository:
>
> <https://github.com/mmoayeri/RIVAL10>

After downloading, place the contents under `datasets/RIVAL10/` following the structure above.

## Citation

```bibtex
@inproceedings{moayeri2022rival10,
  title     = {A Comprehensive Study of Image Classification Model Sensitivity to Foregrounds, Backgrounds, and Visual Attributes},
  author    = {Moayeri, Mazda and Pope, Phillip and Balaji, Yogesh and Feizi, Soheil},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
