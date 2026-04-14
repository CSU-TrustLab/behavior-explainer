"""
datasets.py — Dataset classes and dataloaders for RIVAL10 and EuroSAT.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_RIVAL10_ROOT = PROJECT_ROOT / "datasets" / "RIVAL10"
_EUROSAT_ROOT = PROJECT_ROOT / "datasets" / "EuroSAT"

_STANDARD_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

random.seed(42)


# ---------------------------------------------------------------------------
# RIVAL10
# ---------------------------------------------------------------------------

class RIVAL10(Dataset):
    """RIVAL10: 10-class ImageNet subset with object segmentation masks."""

    def __init__(self, train=True, return_masks=False, client_transforms=None):
        split = "train" if train else "test"
        self.train = train
        self.return_masks = return_masks
        self.client_transforms = client_transforms
        self.img_root  = str(_RIVAL10_ROOT / split / "ordinary") + "/"
        self.mask_root = str(_RIVAL10_ROOT / split / "entire_object_masks") + "/"
        self.resize = transforms.Resize((224, 224))
        self.instances = self._collect_instances()

    def _collect_instances(self):
        with open(_RIVAL10_ROOT / "meta" / "train_test_split_by_url.json") as f:
            urls = json.load(f)["train" if self.train else "test"]
        with open(_RIVAL10_ROOT / "meta" / "label_mappings.json") as f:
            label_mappings = json.load(f)
        with open(_RIVAL10_ROOT / "meta" / "wnid_to_class.json") as f:
            wnid_to_class = json.load(f)

        wnids      = [u.split("/")[-2] for u in urls]
        class_names = [wnid_to_class[w] for w in wnids]
        labels     = [label_mappings[c][1] for c in class_names]
        paths      = [self.img_root + "_".join(u.split("/")[-2:]) for u in urls]

        return [(p, l) for p, l in zip(paths, labels) if os.path.exists(p)]

    def _transform(self, imgs):
        crop_params = transforms.RandomResizedCrop.get_params(
            imgs[0], scale=(0.8, 1.0), ratio=(0.75, 1.25)
        )
        flip = random.random() < 0.5
        result = []
        for img in imgs:
            if self.client_transforms is not None:
                img = self.client_transforms(img)
            else:
                if self.train:
                    img = TF.crop(img, *crop_params)
                    if flip:
                        img = TF.hflip(img)
                img = TF.to_tensor(self.resize(img))
                if img.shape[0] == 1:
                    img = img.repeat(3, 1, 1)
            result.append(img)
        return result

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        path, label = self.instances[idx]
        img = Image.open(path)
        if img.mode == "L":
            img = Image.fromarray(np.stack([np.array(img)] * 3, axis=-1))
        imgs = [img]
        if self.return_masks:
            wnid, fname = path.split("/")[-2:]
            imgs.append(Image.open(self.mask_root + wnid + "_" + fname))
        imgs = self._transform(imgs)
        return (imgs[0], imgs[1], label) if self.return_masks else (imgs[0], label)


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def get_dataloader_rival10(train=True, batch_size=32, num_workers=2):
    return DataLoader(
        RIVAL10(train=train),
        batch_size=batch_size, shuffle=train, num_workers=num_workers,
    )


def get_dataloader_eurosat(batch_size=32, num_workers=2):
    return DataLoader(
        datasets.ImageFolder(root=str(_EUROSAT_ROOT), transform=_STANDARD_TRANSFORM),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
