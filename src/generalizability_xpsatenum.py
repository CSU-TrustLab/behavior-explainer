"""
src/generalizability_xpsatenum.py — Generalizability analysis for XpSatEnum.

Splits the behavior image set 50/50 (seed=0), runs XpEnum followed by
XpSatEnum on each half for all three eraser types, and writes per-half binary
CSV files into the same intermediate_results/<cm_name>/ directory used by the
rest of the pipeline.

Output files follow the same naming convention as run_experiment.py but with
an _H1 or _H2 suffix embedded in the behavior ID:

    binary_COXB26-N300_resnet_rival10_H1_A.csv   ← XpEnum AXps, half 1
    binary_COSB26-N300_resnet_rival10_H1_A.csv   ← XpSatEnum AXps, half 1
    binary_COXB26-N300_resnet_rival10_H2_A.csv   ← XpEnum AXps, half 2
    ...

These can then be loaded by analysis/generalizability.py (or a similar script)
to compute IoU(topK_H1, topK_H2) for XpSatEnum.

CLI usage (run from the behavior-explainer repo root):
    python src/generalizability_xpsatenum.py \\
        --model resnet_rival10 --behavior 2 --class-idx 6 --n-concepts 300
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# clip.pkl was pickled with the legacy t2c module.
_LEGACY_SRC = PROJECT_ROOT.parent / "src"
if _LEGACY_SRC.exists():
    sys.path.append(str(_LEGACY_SRC))

from src.compute_means import get_img_mean_clip, get_img_mean_map
from src.train_aligner import VectorToVectorRegression  # noqa: F401 (needed for unpickling)
from src.train_aligner import get_feature_extractor
from src.datasets import get_dataloader_eurosat, get_dataloader_rival10
from src.explain import (
    filter_behavior,
    get_clip_embeddings,
    get_model_acc,
    get_sample_data,
    get_zsclip_acc,
    order_concept_strengths,
    sanity_check_leace,
    sanity_check_ortho,
    select_shared_instances,
    wrapper_XpEnum,
    wrapper_XpSatEnum,
)
from utils.concept_eraser import ClipOrthoEraser, ClipSpliceEraser, LeaceEraserWrapper
from utils.pickler import Pickler

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
VOCABS_DIR       = PROJECT_ROOT / "vocabs"


# ---------------------------------------------------------------------------
# Behavior predicates (mirrors run_experiment.py)
# ---------------------------------------------------------------------------

def make_behavior(behavior_num, class_idx, other_class_idx):
    behaviors = {
        1: lambda pred, lab: (pred == class_idx),
        2: lambda pred, lab: (pred == class_idx) & (lab == class_idx),
        3: lambda pred, lab: (pred != class_idx) & (lab == class_idx),
        4: lambda pred, lab: (pred != lab),
        5: lambda pred, lab: (lab == class_idx) & (pred == other_class_idx),
        6: lambda pred, lab: (pred == class_idx) & (lab != class_idx),
    }
    if behavior_num not in behaviors:
        raise ValueError(f"Unknown behavior number {behavior_num}.")
    return behaviors[behavior_num]


# ---------------------------------------------------------------------------
# Model loading (mirrors run_experiment.py)
# ---------------------------------------------------------------------------

def load_models(model_name, device):
    print("Loading CLIP model...")
    t2c = Pickler.read("clip")
    t2c.device = device
    clip_model = t2c.clip_model.to(device)
    clip_model.device = device
    CLIP_enc = clip_model.input_to_representation
    print("+ CLIP loaded.")

    model_type = "resnet" if model_name.startswith("resnet") else "vgg"

    print(f"Loading {model_name}...")
    net = Pickler.read(f"{model_name}_finetuned").to(device).eval()
    f_enc, f_head, _ = get_feature_extractor(net, model_type)
    print(f"+ {model_name} loaded.")

    print(f"Loading aligners for {model_name}...")
    align     = Pickler.read(f"{model_name}_to_clip").to(device).eval()
    align_inv = Pickler.read(f"clip_to_{model_name}").to(device).eval()
    print("+ Aligners loaded.")

    return CLIP_enc, f_enc, f_head, align, align_inv


def load_vocab(dataset_name, mcs, device):
    vocab_name = f"MCS_{mcs}_NA_{dataset_name}"
    C = (VOCABS_DIR / f"{vocab_name}.txt").read_text().strip().splitlines()
    C_vectors     = Pickler.read(f"{vocab_name}_vecs").to(device).to(torch.float64)
    class_vectors = Pickler.read(f"{vocab_name}_class_vecs").to(device).to(torch.float64)
    print(f"+ Vocab: {len(C)} concepts,  C_vectors {tuple(C_vectors.shape)},  "
          f"class_vectors {tuple(class_vectors.shape)}")
    return C, C_vectors, class_vectors


def make_pred_fn(setup, align_inv, f_head,
                 img_mean_map, img_mean_clip, class_vectors, device, norm_flag):
    if setup == "zero-shot-clip":
        def pred_fn(e_clip_erased, norms, original_pred):
            return get_zsclip_acc(e_clip_erased, norms, original_pred,
                                  img_mean_clip, class_vectors, device, norm_flag)
    else:
        def pred_fn(e_clip_erased, norms, original_pred):
            return get_model_acc(e_clip_erased, norms, align_inv, f_head, original_pred,
                                 img_mean_map, device, norm_flag)
    return pred_fn


def apply_energy_order(all_data, C, C_vectors, mcs, dataset_name, device):
    all_eclips = torch.stack([v[2] for v in all_data.values()], dim=0).to(device).to(torch.float64)
    avg_act = torch.mean(torch.abs(all_eclips @ C_vectors), dim=0)
    _, order = torch.sort(avg_act, descending=True)
    C_vectors  = torch.t(torch.t(C_vectors)[order])
    C = [C[i] for i in order.cpu().tolist()]
    print(f"High-energy concepts (top 10): {C[:10]}")
    Pickler.write(f"MCS_{mcs}_NA_{dataset_name}_e_vecs", C_vectors)
    (INTERMEDIATE_DIR / f"MCS_{mcs}_NA_{dataset_name}_e_vocab.txt").write_text("\n".join(C))
    print("+ Energy-sorted vectors and vocab saved.")
    return C, C_vectors


# ---------------------------------------------------------------------------
# Half-split helpers
# ---------------------------------------------------------------------------

def split_filtered_data(filtered_data, seed=0):
    """Return (half1_dict, half2_dict) by 50/50 random split of filtered_data keys."""
    keys = list(filtered_data.keys())
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    mid  = len(keys) // 2
    h1   = {keys[i]: filtered_data[keys[i]] for i in perm[:mid]}
    h2   = {keys[i]: filtered_data[keys[i]] for i in perm[mid:]}
    return h1, h2


def build_cord_signs(eraser_type, filtered_half, C_vectors_N, device):
    """Compute C_ord_signs for one half of filtered_data."""
    eclips = [v[2] for v in filtered_half.values()]
    if eraser_type == "ortho":
        _, signs = order_concept_strengths(eclips, C_vectors_N)
    elif eraser_type == "splice":
        signs = torch.ones((len(filtered_half), C_vectors_N.shape[1]))
    elif eraser_type == "leace":
        stack = torch.stack(eclips).to(device).to(torch.float64)
        signs = torch.sign(stack @ C_vectors_N)
    return signs


def run_half(
    filtered_half, half_tag, C_N, C_vectors_N, eraser_type, eraser,
    C_ord_signs, pred_fn, prefix, behavior_suffix, results_dir,
    device, xpenum_iters, experiments_per_behavior, timeout,
):
    """Run XpEnum + XpSatEnum on one half and write results."""
    beh_id_xp = f"{prefix}X{behavior_suffix}_{half_tag}"
    xp_results, xp_eclips, xp_preds, xp_idxs = wrapper_XpEnum(
        filtered_half, C_N, C_vectors_N, C_ord_signs, eraser, pred_fn,
        experiments_per_behavior, xpenum_iters,
        beh_id_xp, results_dir, device,
        precomputed=None, timeout=timeout,
    )
    if xp_results:
        if eraser_type == "leace":
            eraser.keep_hashmap = True
        beh_id_sat = f"{prefix}S{behavior_suffix}_{half_tag}"
        wrapper_XpSatEnum(
            filtered_half, C_N, C_vectors_N, C_ord_signs, eraser, pred_fn,
            xp_results, xp_eclips, xp_preds, xp_idxs,
            beh_id_sat, results_dir, device,
            timeout=timeout,
        )
    else:
        print(f"[{half_tag}] No XpEnum results for {prefix} — skipping XpSatEnum.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run XpEnum + XpSatEnum on 50/50 splits for generalizability analysis."
    )
    p.add_argument("--model", required=True,
                   help="Model name: resnet_rival10 | vgg_rival10 | resnet_eurosat | vgg_eurosat")
    p.add_argument("--setup", default="vision-model",
                   choices=["vision-model", "zero-shot-clip"])
    p.add_argument("--mcs", type=int, default=90)
    p.add_argument("--n-concepts", type=int, default=200)
    p.add_argument("--behavior", type=int, default=2, choices=range(1, 7), metavar="1-6")
    p.add_argument("--class-idx", type=int, default=4)
    p.add_argument("--other-class-idx", type=int, default=-1)
    p.add_argument("--experiments-per-behavior", type=int, default=185)
    p.add_argument("--xpenum-iters", type=int, default=250)
    p.add_argument("--norm-flag", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the 50/50 split (default 0)")
    p.add_argument("--no-energy-order", dest="energy_order", action="store_false", default=True)
    p.add_argument("--rebuild-behaviors", action="store_true")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--results-dir", default=None,
                   help="Output directory for CSV files (default: intermediate_results/<cm_name>/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(42)

    device       = args.device
    model_name   = args.model
    dataset_name = "eurosat" if "eurosat" in model_name else "rival10"
    vocab_dataset_name = "EuroSAT" if "eurosat" in model_name else "RIVAL10"

    CLIP_enc, f_enc, f_head, align, align_inv = load_models(model_name, device)

    D = get_dataloader_rival10(train=True) if dataset_name == "rival10" \
        else get_dataloader_eurosat()

    img_mean_map  = get_img_mean_map(model_name, f_enc, align, D, device) \
        .to(device).to(torch.float64)
    img_mean_clip = get_img_mean_clip(dataset_name, CLIP_enc, D, device) \
        .to(device).to(torch.float64)

    C, C_vectors, class_vectors = load_vocab(vocab_dataset_name, args.mcs, device)

    pred_fn = make_pred_fn(
        args.setup, align_inv, f_head,
        img_mean_map, img_mean_clip, class_vectors, device, args.norm_flag,
    )

    cls_str   = str(args.class_idx)       if args.class_idx >= 0       else "-"
    other_str = str(args.other_class_idx) if args.other_class_idx >= 0 else "-"
    model_tag  = f"_{model_name}"  if args.setup == "vision-model" else ""
    energy_tag = "_e"              if args.energy_order             else ""

    cm_name         = f"CM_MCS{args.mcs}_N{args.n_concepts}{model_tag}{energy_tag}"
    behavior_suffix = f"B{args.behavior}{cls_str}{other_str}-N{args.n_concepts}{model_tag}"

    results_dir = Path(args.results_dir) if args.results_dir \
        else INTERMEDIATE_DIR / cm_name
    results_dir.mkdir(parents=True, exist_ok=True)

    cm_pkl = INTERMEDIATE_DIR / f"{cm_name}.pkl"

    with torch.no_grad():
        if not cm_pkl.is_file() or args.rebuild_behaviors:
            print(f"\n── Building behavior data: {cm_name} ──")
            all_data = get_clip_embeddings(
                D, args.setup,
                f_enc, CLIP_enc, align, align_inv, f_head,
                img_mean_map, img_mean_clip, class_vectors,
                device, args.norm_flag,
            )
            if args.energy_order:
                C, C_vectors = apply_energy_order(
                    all_data, C, C_vectors, args.mcs, vocab_dataset_name, device
                )
            C_vectors_N = C_vectors[:, :args.n_concepts]
            C_N = C[:args.n_concepts]
            sample_data = get_sample_data(all_data, device, sample_size=500)
            all_data = sanity_check_ortho(all_data, C_vectors_N, pred_fn, device)
            all_data = sanity_check_leace(all_data, sample_data, C_vectors_N, pred_fn, device)
            Pickler.write(cm_name, all_data)
            print(f"+ Behavior data saved: '{cm_name}.pkl'  ({len(all_data)} images)")
        else:
            print(f"\n── Loading behavior data: {cm_name} ──")
            all_data = Pickler.read(cm_name)
            if args.energy_order:
                C, C_vectors = apply_energy_order(
                    all_data, C, C_vectors, args.mcs, vocab_dataset_name, device
                )
            C_vectors_N = C_vectors[:, :args.n_concepts]
            C_N = C[:args.n_concepts]

        sample_data = get_sample_data(all_data, device, sample_size=500)

        B = make_behavior(args.behavior, args.class_idx, args.other_class_idx)
        filtered_data = filter_behavior(dict(all_data), B)
        print(f"\nBehavior '{behavior_suffix}': {len(filtered_data)} instances")

        if len(filtered_data) == 0:
            print("No instances match this behavior — exiting.")
            return

        # Apply per-behavior LEACE sanity check (mirrors run_experiment.py)
        filtered_data = sanity_check_leace(filtered_data, sample_data, C_vectors_N, pred_fn, device)
        print(f"After LEACE sanity check: {len(filtered_data)} instances")
        if len(filtered_data) == 0:
            print("No instances remain — exiting.")
            return

        # Build erasers (Splice needs at least one instance to initialise)
        filtered_eclips_all = [v[2] for v in filtered_data.values()]
        ortho_eraser  = ClipOrthoEraser(C_vectors_N, device=device, dtype=torch.float64)
        dummy         = torch.stack(filtered_eclips_all[:2]).to(device).to(torch.float64)
        splice_eraser = ClipSpliceEraser(C_vectors_N, dummy, device=device, dtype=torch.float64)
        leace_eraser  = LeaceEraserWrapper(
            sample_data, C_vectors_N, device=device, dtype=torch.float64
        )

        # 50/50 split
        h1, h2 = split_filtered_data(filtered_data, seed=args.seed)
        print(f"\nSplit (seed={args.seed}): H1={len(h1)} images, H2={len(h2)} images")

        eraser_configs = [
            ("CO", "ortho",  ortho_eraser),
            ("CS", "splice", splice_eraser),
            ("CL", "leace",  leace_eraser),
        ]

        for prefix, eraser_type, eraser in eraser_configs:
            print(f"\n{'='*60}\nEraser: {eraser_type.upper()}\n{'='*60}")
            for half_tag, filtered_half in [("H1", h1), ("H2", h2)]:
                print(f"\n── {half_tag} ({len(filtered_half)} images) ──")
                C_ord_signs = build_cord_signs(eraser_type, filtered_half, C_vectors_N, device)
                run_half(
                    filtered_half, half_tag, C_N, C_vectors_N, eraser_type, eraser,
                    C_ord_signs, pred_fn, prefix, behavior_suffix, results_dir,
                    device, args.xpenum_iters, args.experiments_per_behavior, args.timeout,
                )

    print("\nDone!")


if __name__ == "__main__":
    main()