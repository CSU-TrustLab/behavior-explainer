"""
compute_means.py — Compute and cache CLIP-space mean vectors.

Three means are provided:

  img_mean_map_{model_name}   : mean of aligned (vision-model → CLIP) image
                                 embeddings over the training split.  Shape (512,).

  img_mean_clip_{dataset_name}: mean of raw CLIP image embeddings over the
                                 training split.  Shape (512,).

  text_mean                   : mean of CLIP text embeddings for all MS-COCO
                                 vocabulary words.  Shape (512, 1) — column vector
                                 suitable for broadcasting with (512, n) matrices.

All three are computed once and cached via Pickler.create_or_read.

Usage
-----
    python src/compute_means.py --model resnet_rival10 --device cuda:0
"""

import sys
from pathlib import Path

import clip
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_LEGACY_SRC = PROJECT_ROOT.parent / "src"
if _LEGACY_SRC.exists():
    sys.path.append(str(_LEGACY_SRC))

from src.datasets import get_dataloader_eurosat, get_dataloader_rival10
from src.train_aligner import get_feature_extractor
from utils.pickler import Pickler

VOCABS_DIR = PROJECT_ROOT / "vocabs"


# ---------------------------------------------------------------------------
# Internal compute functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_img_mean_map(f_enc, align, loader, device):
    """
    Mean of aligned vision-model embeddings over the dataset.

    For each batch: f_enc(images) → align(reps) → accumulate.
    Returns a (512,) CPU tensor.
    """
    all_embs = []
    for images, _ in tqdm(loader, desc="Computing img_mean_map"):
        images = images.to(device)
        reps = f_enc(images).flatten(1)
        mapped = align(reps).to(torch.float64)
        all_embs.append(mapped)
    return torch.cat(all_embs).mean(dim=0).cpu()


@torch.no_grad()
def _compute_img_mean_clip(CLIP_enc, loader, device):
    """
    Mean of raw CLIP image embeddings over the dataset.

    Returns a (512,) CPU tensor.
    """
    all_embs = []
    for images, _ in tqdm(loader, desc="Computing img_mean_clip"):
        images = images.to(device)
        embs = CLIP_enc(images).to(torch.float64)
        all_embs.append(embs)
    return torch.cat(all_embs).mean(dim=0).cpu()


@torch.no_grad()
def _compute_text_mean(clip_model, device):
    """
    Mean of CLIP text embeddings for all MS-COCO words.

    Returns a (512, 1) CPU tensor — a column vector for broadcasting
    with (512, n) concept matrices.
    """
    mscoco_path = VOCABS_DIR / "mscoco.txt"
    words = mscoco_path.read_text().strip().splitlines()

    reps = []
    for word in tqdm(words, desc="Computing text_mean"):
        tokens = clip.tokenize([word]).to(device)
        rep = clip_model.encode_text(tokens)
        reps.append(rep)
    reps = torch.stack(reps)         # (n_words, 1, 512)
    return torch.mean(reps, dim=0).T.cpu()  # (512, 1)


# ---------------------------------------------------------------------------
# Public API — compute once, cache for all future calls
# ---------------------------------------------------------------------------

def get_img_mean_map(model_name, f_enc, align, loader, device):
    """
    Return the cached aligned-image mean for model_name, computing it if needed.

    Cache key: ``img_mean_map_{model_name}``
    Shape: (512,) CPU tensor (float64).
    """
    return Pickler.create_or_read(
        f"img_mean_map_{model_name}",
        lambda: _compute_img_mean_map(f_enc, align, loader, device),
    )


def get_img_mean_clip(dataset_name, CLIP_enc, loader, device):
    """
    Return the cached CLIP-image mean for dataset_name, computing it if needed.

    Cache key: ``img_mean_clip_{dataset_name}``
    Shape: (512,) CPU tensor (float64).
    """
    return Pickler.create_or_read(
        f"img_mean_clip_{dataset_name}",
        lambda: _compute_img_mean_clip(CLIP_enc, loader, device),
    )


def get_text_mean(clip_model, device):
    """
    Return the cached MS-COCO text mean, computing it if needed.

    Cache key: ``text_mean``
    Shape: (512, 1) CPU tensor.
    """
    return Pickler.create_or_read(
        "text_mean",
        lambda: _compute_text_mean(clip_model, device),
    )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Pre-compute and cache image / text CLIP-space means."
    )
    p.add_argument(
        "--model", required=True,
        help="Model name: resnet_rival10 | vgg_rival10 | resnet_eurosat | vgg_eurosat",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--skip-text-mean", action="store_true",
        help="Skip computing the text mean (useful if already cached)",
    )
    args = p.parse_args()

    device     = args.device
    model_name = args.model
    dataset_name = "eurosat" if "eurosat" in model_name else "rival10"
    model_type   = "resnet"  if model_name.startswith("resnet") else "vgg"

    # Load CLIP
    print("Loading CLIP model...")
    t2c = Pickler.read("clip")
    t2c.device = device
    clip_model = t2c.clip_model.to(device)
    clip_model.device = device
    CLIP_enc = clip_model.input_to_representation
    print("+ CLIP loaded.")

    # Load vision model + aligners
    print(f"Loading {model_name}...")
    net = Pickler.read(f"{model_name}_finetuned").to(device).eval()
    f_enc, _, _ = get_feature_extractor(net, model_type)
    align = Pickler.read(f"{model_name}_to_clip").to(device).eval()
    print(f"+ {model_name} and aligner loaded.")

    # Load training dataloader
    loader = (
        get_dataloader_rival10(train=True) if dataset_name == "rival10"
        else get_dataloader_eurosat()
    )

    with torch.no_grad():
        print("\n── img_mean_map ──")
        m = get_img_mean_map(model_name, f_enc, align, loader, device)
        print(f"+ img_mean_map_{model_name}: shape={tuple(m.shape)}, dtype={m.dtype}")

        print("\n── img_mean_clip ──")
        m = get_img_mean_clip(dataset_name, CLIP_enc, loader, device)
        print(f"+ img_mean_clip_{dataset_name}: shape={tuple(m.shape)}, dtype={m.dtype}")

        if not args.skip_text_mean:
            print("\n── text_mean ──")
            m = get_text_mean(clip_model, device)
            print(f"+ text_mean: shape={tuple(m.shape)}, dtype={m.dtype}")

    print("\nDone.")


if __name__ == "__main__":
    main()
