"""
scan_viable_behaviors.py — Diagnostic for selecting a viable behavior.

For each RIVAL10 class and behavior type, reports:
  - total instances
  - instances surviving the LEACE sanity check
  - whether Splice is degenerate (img_mean_map predicts the same class as the
    model's prediction, so erasing all concepts doesn't change anything)

Behaviors scanned
-----------------
  B=2  : correctly classified (pred == label == class_idx)
  B=3  : misclassified       (pred != class_idx, label == class_idx)
  B=5  : specific confusion  (label == class_idx, pred == other_class_idx)

B=5 is scanned for every (class_idx, other_class_idx) pair with ≥10 instances.

Usage
-----
    python scripts/scan_viable_behaviors.py [--min-instances N]

Requires the behavior pkl CM_MCS90_N300_resnet_rival10_e.pkl (built by the
ships integration test).
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT.parent / "src"))

import torch

from src.compute_means import get_img_mean_clip, get_img_mean_map
from src.datasets import get_dataloader_rival10
from src.explain import get_sample_data, sanity_check_leace
from src.run_experiment import load_models, load_vocab, make_pred_fn
from src.train_aligner import VectorToVectorRegression  # noqa: F401
from utils.pickler import Pickler

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
RIVAL10 = ["truck", "car", "plane", "ship", "cat", "dog", "equine", "deer", "frog", "bird"]
CM_NAME = "CM_MCS90_N300_resnet_rival10_e"
MCS, N_CONCEPTS = 90, 300


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--min-instances", type=int, default=10,
                   help="Minimum instances to include a behavior (default: 10)")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def leace_pass_count(filtered, sample_data, C_vectors_N, pred_fn, device):
    """Return number of images that survive the LEACE sanity check (suppress output)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        passed = sanity_check_leace(filtered, sample_data, C_vectors_N, pred_fn, device)
    return len(passed)


def main():
    args = parse_args()
    device = args.device

    # ── Load models and data ──────────────────────────────────────────────────
    CLIP_enc, f_enc, f_head, align, align_inv = load_models("resnet_rival10", device)
    D = get_dataloader_rival10(train=True)

    img_mean_map  = get_img_mean_map("resnet_rival10", f_enc, align, D, device).to(device).to(torch.float64)
    img_mean_clip = get_img_mean_clip("rival10", CLIP_enc, D, device).to(device).to(torch.float64)
    C, C_vectors, class_vectors = load_vocab("RIVAL10", MCS, device)
    pred_fn = make_pred_fn("vision-model", align_inv, f_head,
                           img_mean_map, img_mean_clip, class_vectors, device, False)

    # Splice baseline
    with torch.no_grad():
        _zero = torch.zeros(1, img_mean_map.shape[0], device=device, dtype=torch.float64)
        _logits = f_head(align_inv((_zero + img_mean_map).float()))
        splice_default = _logits.argmax(dim=1).item()
    print(f"\nSplice erase_all baseline → class {splice_default} ({RIVAL10[splice_default]})")
    print(f"Splice is degenerate only for behaviors where pred_class == {splice_default} "
          f"({RIVAL10[splice_default]})\n")

    # ── Load pkl and reorder vocabulary by energy ─────────────────────────────
    print(f"Loading {CM_NAME}.pkl ...")
    all_data = Pickler.read(CM_NAME)

    all_eclips = torch.stack([v[2] for v in all_data.values()], dim=0).to(device).to(torch.float64)
    avg_act = torch.mean(torch.abs(all_eclips @ C_vectors), dim=0)
    _, order = torch.sort(avg_act, descending=True)
    C_vectors  = torch.t(torch.t(C_vectors)[order])
    C_vectors_N = C_vectors[:, :N_CONCEPTS]

    sample_data = get_sample_data(all_data, device, sample_size=500)

    # ── B=2 and B=3 ──────────────────────────────────────────────────────────
    hdr = f"{'B':<3} {'gt':<8} {'pred':<8} {'total':>7} {'leace_ok':>9} {'splice_ok':>10}"
    print(hdr)
    print("-" * len(hdr))

    for i, name in enumerate(RIVAL10):
        # B=2: correctly classified
        sub = {k: v for k, v in all_data.items()
               if int(v[1]) == i and int(v[0]) == i}
        if len(sub) >= args.min_instances:
            lp = leace_pass_count(sub, sample_data, C_vectors_N, pred_fn, device)
            sok = "yes" if splice_default != i else "NO"
            print(f"{'2':<3} {name:<8} {name:<8} {len(sub):>7} {lp:>9} {sok:>10}")

    print()
    for i, name in enumerate(RIVAL10):
        # B=3: any misclassification of class i
        sub = {k: v for k, v in all_data.items()
               if int(v[0]) != i and int(v[1]) == i}
        if len(sub) >= args.min_instances:
            # predicted class for B=3 varies per image — Splice ok if splice_default != pred per image
            # conservatively: at least some images have pred != splice_default
            preds = set(int(v[0]) for v in sub.values())
            sok = "yes" if any(p != splice_default for p in preds) else "NO"
            lp = leace_pass_count(sub, sample_data, C_vectors_N, pred_fn, device)
            print(f"{'3':<3} {name:<8} {'(any)':<8} {len(sub):>7} {lp:>9} {sok:>10}")

    # ── B=5: specific confusion pairs ─────────────────────────────────────────
    print()
    print("B=5 specific confusion pairs (label=gt, pred=other):")
    print(hdr)
    print("-" * len(hdr))

    for i, name_i in enumerate(RIVAL10):
        for j, name_j in enumerate(RIVAL10):
            if i == j:
                continue
            sub = {k: v for k, v in all_data.items()
                   if int(v[1]) == i and int(v[0]) == j}
            if len(sub) < args.min_instances:
                continue
            # predicted class is j; Splice degenerate if img_mean_map → j
            sok = "yes" if splice_default != j else "NO"
            lp = leace_pass_count(sub, sample_data, C_vectors_N, pred_fn, device)
            print(f"{'5':<3} {name_i:<8} {name_j:<8} {len(sub):>7} {lp:>9} {sok:>10}")


if __name__ == "__main__":
    main()