"""
run_experiment.py — Parametrized experiment runner.

Runs all 9 explanation configurations (3 eraser types × 3 algorithms) for a
single behavior, each with a configurable timeout.

Algorithms
----------
    XpEnum     — NINA-based explanation enumeration
    XpSatEnum  — XpEnum followed by cross-instance saturation
    NaiveEnum  — depth-bounded exhaustive search

Usage
-----
    python src/run_experiment.py --model resnet_rival10 --behavior 2 --class-idx 4

Pipeline
--------
1. Load CLIP, vision model, and linear aligners.
2. Load concept vocabulary and class vectors.
3. Build or load sanity-checked behavior data.
4. Filter to the requested behavior (B, class_idx, other_class_idx).
5. For each of the 3 eraser types (Ortho, Splice, LEACE):
       run XpEnum  [→ optionally XpSatEnum]  and  NaiveEnum,
       each capped at --timeout seconds.
6. Write binary CSV results to results/<cm_name>/.

Notes
-----
- img_mean_map and img_mean_clip are computed automatically on first run via
  compute_means.py and cached in intermediate_results/.
- Behavior data is cached as intermediate_results/<cm_name>.pkl.  Pass
  --rebuild-behaviors to force a rebuild.
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# clip.pkl was pickled with the legacy t2c module.
_LEGACY_SRC = PROJECT_ROOT.parent / "src"
if _LEGACY_SRC.exists():
    sys.path.append(str(_LEGACY_SRC))

# VectorToVectorRegression must be importable when unpickling saved aligners.
from src.compute_means import get_img_mean_clip, get_img_mean_map
from src.train_aligner import VectorToVectorRegression  # noqa: F401
from src.train_aligner import get_feature_extractor
from src.datasets import get_dataloader_eurosat, get_dataloader_rival10
from src.explain import (
    filter_behavior,
    get_clip_embeddings,
    get_model_acc,
    get_sample_data,
    get_zsclip_acc,
    order_concept_strengths,
    plot_confusion_matrix,
    sanity_check_leace,
    sanity_check_ortho,
    select_shared_instances,
    wrapper_NaiveEnum,
    wrapper_XpEnum,
    wrapper_XpSatEnum,
)
from utils.concept_eraser import ClipOrthoEraser, ClipSpliceEraser, LeaceEraserWrapper
from utils.pickler import Pickler

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
RESULTS_DIR      = PROJECT_ROOT / "results"
VOCABS_DIR       = PROJECT_ROOT / "vocabs"

# ---------------------------------------------------------------------------
# Behavior predicates
# ---------------------------------------------------------------------------

def make_behavior(behavior_num, class_idx, other_class_idx):
    """Return a behavior predicate B(predicted, label) -> bool."""
    behaviors = {
        1: lambda pred, lab: (pred == class_idx),
        2: lambda pred, lab: (pred == class_idx) & (lab == class_idx),
        3: lambda pred, lab: (pred != class_idx) & (lab == class_idx),
        4: lambda pred, lab: (pred != lab),
        5: lambda pred, lab: (lab == class_idx) & (pred == other_class_idx),
        6: lambda pred, lab: (pred == class_idx) & (lab != class_idx),
    }
    if behavior_num not in behaviors:
        raise ValueError(f"Unknown behavior number {behavior_num}. Expected 1–6.")
    return behaviors[behavior_num]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(model_name, device):
    """
    Load CLIP, fine-tuned vision model, and linear aligners.

    Returns (CLIP_enc, f_enc, f_head, align, align_inv).

    align     : VectorToVectorRegression  (vision-model space → CLIP space)
    align_inv : VectorToVectorRegression  (CLIP space → vision-model space)
    """
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
    """
    Load concept vocabulary (word list + vectors) and class vectors.

    Returns (C, C_vectors, class_vectors).
      C_vectors     : (512, n_concepts) float64 tensor on device  (full vocab)
      class_vectors : (512, n_classes)  float64 tensor on device
    """
    vocab_name = f"MCS_{mcs}_NA_{dataset_name}"
    C = (VOCABS_DIR / f"{vocab_name}.txt").read_text().strip().splitlines()
    C_vectors     = Pickler.read(f"{vocab_name}_vecs").to(device).to(torch.float64)
    class_vectors = Pickler.read(f"{vocab_name}_class_vecs").to(device).to(torch.float64)
    print(f"+ Vocab: {len(C)} concepts,  C_vectors {tuple(C_vectors.shape)},  "
          f"class_vectors {tuple(class_vectors.shape)}")
    return C, C_vectors, class_vectors


# ---------------------------------------------------------------------------
# Prediction function factory
# ---------------------------------------------------------------------------

def make_pred_fn(setup, align_inv, f_head,
                 img_mean_map, img_mean_clip, class_vectors, device, norm_flag):
    """
    Build a unified prediction callable for the current experiment configuration.

    The returned function has signature:
        pred_fn(e_clip_erased, norms, original_pred) -> (acc, per_image)

    This callable is passed to all explanation functions in explain.py,
    keeping model-specific details out of the explanation engine.
    """
    if setup == "zero-shot-clip":
        def pred_fn(e_clip_erased, norms, original_pred):
            return get_zsclip_acc(e_clip_erased, norms, original_pred,
                                  img_mean_clip, class_vectors, device, norm_flag)
    else:  # vision-model
        def pred_fn(e_clip_erased, norms, original_pred):
            return get_model_acc(e_clip_erased, norms, align_inv, f_head, original_pred,
                                 img_mean_map, device, norm_flag)
    return pred_fn


# ---------------------------------------------------------------------------
# Energy-based vocabulary reordering
# ---------------------------------------------------------------------------

def apply_energy_order(all_data, C, C_vectors, mcs, dataset_name, device):
    """
    Reorder C and C_vectors by average absolute concept activation across
    all images in all_data (high-energy concepts first).

    Saves the reordered vectors to intermediate_results/ and returns
    (C_reordered, C_vectors_reordered).
    """
    all_eclips = torch.stack(
        [v[2] for v in all_data.values()], dim=0
    ).to(device).to(torch.float64)
    avg_act = torch.mean(torch.abs(all_eclips @ C_vectors), dim=0)
    _, order = torch.sort(avg_act, descending=True)

    C_vectors  = torch.t(torch.t(C_vectors)[order])
    order_list = order.cpu().tolist()
    C = [C[i] for i in order_list]

    print(f"High-energy concepts (top 10): {C[:10]}")
    Pickler.write(f"MCS_{mcs}_NA_{dataset_name}_e_vecs", C_vectors)
    (INTERMEDIATE_DIR / f"MCS_{mcs}_NA_{dataset_name}_e_vocab.txt").write_text("\n".join(C))
    print("+ Energy-sorted vectors and vocab saved.")
    return C, C_vectors


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run 9-configuration concept explanation experiment."
    )
    p.add_argument(
        "--model", required=True,
        help="Model name: resnet_rival10 | vgg_rival10 | resnet_eurosat | vgg_eurosat",
    )
    p.add_argument(
        "--setup", default="vision-model",
        choices=["vision-model", "zero-shot-clip"],
        help="Embedding and prediction setup (default: vision-model)",
    )
    p.add_argument(
        "--mcs", type=int, default=90,
        help="Max cosine similarity threshold used in build_vocab (default: 90)",
    )
    p.add_argument(
        "--n-concepts", type=int, default=200,
        help="Number of concepts to use after energy ordering (default: 200)",
    )
    p.add_argument(
        "--behavior", type=int, default=2, choices=range(1, 7),
        metavar="1-6",
        help="Behavior predicate number (default: 2 = correct predictions on class-idx)",
    )
    p.add_argument(
        "--class-idx", type=int, default=4,
        help="Primary class index for the behavior predicate (default: 4)",
    )
    p.add_argument(
        "--other-class-idx", type=int, default=-1,
        help="Secondary class index (used only by B5, default: -1)",
    )
    p.add_argument(
        "--experiments-per-behavior", type=int, default=185,
        help="Max instances per behavior per algorithm (default: 185)",
    )
    p.add_argument(
        "--xpenum-iters", type=int, default=250,
        help="NINA loop iterations per instance in XpEnum (default: 250)",
    )
    p.add_argument(
        "--search-depth", type=int, default=2,
        help="Depth for NaiveEnum exhaustive search (default: 2)",
    )
    p.add_argument(
        "--norm-flag", action="store_true",
        help="Unit-normalise CLIP embeddings before erasure (default: off)",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--energy-order", action="store_true", default=True,
        help="Sort vocabulary by energy before experiments (default: on)",
    )
    p.add_argument("--no-energy-order", dest="energy_order", action="store_false")
    p.add_argument("--no-xpenum",    action="store_true", help="Skip XpEnum")
    p.add_argument("--no-xpsatenum", action="store_true", help="Skip XpSatEnum")
    p.add_argument("--no-naiveenum", action="store_true", help="Skip NaiveEnum")
    p.add_argument(
        "--rebuild-behaviors", action="store_true",
        help="Force rebuild of behavior data (ignore existing pkl)",
    )
    p.add_argument(
        "--auto-retry", action="store_true",
        help="If fewer than N shared instances are found, automatically proceed "
             "with however many qualify (instead of prompting). Exits if zero found.",
    )
    p.add_argument(
        "--timeout", type=int, default=3600,
        help="Per-configuration timeout in seconds (default: 3600)",
    )
    p.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help="Directory where CSV result files are written (default: results/)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(42)

    device       = args.device
    model_name   = args.model
    dataset_name      = "eurosat" if "eurosat" in model_name else "rival10"
    # build_vocab.py saves vocab files with title-case dataset names (RIVAL10 / EuroSAT)
    vocab_dataset_name = "EuroSAT" if "eurosat" in model_name else "RIVAL10"

    # ── Load models ──────────────────────────────────────────────────────────
    CLIP_enc, f_enc, f_head, align, align_inv = load_models(model_name, device)

    # ── Load dataset ──────────────────────────────────────────────────────────
    D = get_dataloader_rival10(train=True) if dataset_name == "rival10" \
        else get_dataloader_eurosat()

    # ── Load image means (compute once, cached in intermediate_results/) ──────
    img_mean_map  = get_img_mean_map(model_name, f_enc, align, D, device) \
        .to(device).to(torch.float64)
    img_mean_clip = get_img_mean_clip(dataset_name, CLIP_enc, D, device) \
        .to(device).to(torch.float64)

    # ── Load vocabulary ───────────────────────────────────────────────────────
    C, C_vectors, class_vectors = load_vocab(vocab_dataset_name, args.mcs, device)

    # ── Build prediction function ─────────────────────────────────────────────
    pred_fn = make_pred_fn(
        args.setup, align_inv, f_head,
        img_mean_map, img_mean_clip, class_vectors,
        device, args.norm_flag,
    )

    # ── Diagnostic: Splice erase_all baseline ────────────────────────────────
    # Splice erases all concepts by zeroing weights → zero CLIP vector.
    # pred_fn(zero, ...) adds img_mean_map back, so the "erased" prediction is
    # always f_head(align_inv(img_mean_map)).  If this equals class_idx, every
    # image of that class is structurally degenerate for Splice.
    with torch.no_grad():
        _zero = torch.zeros(1, img_mean_map.shape[0], device=device, dtype=torch.float64)
        _logits = f_head(align_inv((_zero + img_mean_map).float()))
        _default_class = _logits.argmax(dim=1).item()
        print(f"\nSplice erase_all baseline: img_mean_map → class {_default_class}")
        print(f"Target class: {args.class_idx}  "
              f"{'(MATCH — Splice will be degenerate for this class)' if _default_class == args.class_idx else '(no match — Splice should work)'}")

    # ── File-naming components ────────────────────────────────────────────────
    cls_str   = str(args.class_idx)       if args.class_idx >= 0       else "-"
    other_str = str(args.other_class_idx) if args.other_class_idx >= 0 else "-"
    model_tag  = f"_{model_name}"  if args.setup == "vision-model" else ""
    energy_tag = "_e"              if args.energy_order             else ""

    cm_name         = f"CM_MCS{args.mcs}_N{args.n_concepts}{model_tag}{energy_tag}"
    behavior_suffix = f"B{args.behavior}{cls_str}{other_str}N{args.n_concepts}{model_tag}"
    results_dir = Path(args.results_dir) / cm_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Build or load behavior data ───────────────────────────────────────────
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
            plot_confusion_matrix(all_data, "sanity_check_passed", cm_name, results_dir)
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

    # ── Filter behavior ───────────────────────────────────────────────────────
    B = make_behavior(args.behavior, args.class_idx, args.other_class_idx)
    filtered_data = filter_behavior(dict(all_data), B)  # shallow copy; avoids mutating all_data
    print(f"\nBehavior '{behavior_suffix}': {len(filtered_data)} instances before shared selection")

    if len(filtered_data) == 0:
        print("No instances match this behavior — exiting.")
        return

    # ── Build all 3 erasers (needed for shared-instance pre-screening) ────────
    with torch.no_grad():
        filtered_eclips_pre = [v[2] for v in filtered_data.values()]

        ortho_eraser = ClipOrthoEraser(C_vectors_N, device=device, dtype=torch.float64)

        dummy = torch.stack(filtered_eclips_pre[:2]).to(device).to(torch.float64)
        splice_eraser = ClipSpliceEraser(C_vectors_N, dummy, device=device, dtype=torch.float64)

        sample_data = get_sample_data(all_data, device, sample_size=500)
        leace_eraser = LeaceEraserWrapper(
            sample_data, C_vectors_N, device=device, dtype=torch.float64
        )

        # ── Pre-screening LEACE sanity check on filtered behavior ────────────
        # Remove images where erasing all N concepts with this LEACE eraser
        # leaves the prediction unchanged — they cannot produce non-empty AXps.
        # Uses the same sample_data as leace_eraser for consistency.
        # Note: sanity_check_splice is intentionally omitted here because it
        # uses batch ADMM, which is inconsistent with the per-image splice used
        # in select_shared_instances.  Splice is also not a qualifying eraser.
        print("\n── Pre-screening LEACE sanity check ──")
        filtered_data = sanity_check_leace(
            filtered_data, sample_data, C_vectors_N, pred_fn, device
        )
        print(f"Behavior '{behavior_suffix}' (post sanity check): {len(filtered_data)} instances")
        if len(filtered_data) == 0:
            print("No instances remain after pre-screening LEACE sanity check — exiting.")
            return

        # ── Select shared instances: same images for all 9 configurations ─────
        # Only Ortho and LEACE are qualifying: both must produce ≥1 non-empty
        # AXp or CXp.  Splice is cached but does not gate image selection.
        n_target = args.experiments_per_behavior
        shared_data, precomputed = select_shared_instances(
            filtered_data,
            erasers=[("ortho", ortho_eraser), ("splice", splice_eraser), ("leace", leace_eraser)],
            pred_fn=pred_fn,
            n_max=n_target,
            C_vectors=C_vectors_N,
            device=device,
            qualifying_types={"ortho", "leace"},
        )
        n_found = len(shared_data)

        if n_found < n_target:
            print(
                f"\nOnly {n_found} out of {n_target} images are good candidates "
                f"that produce non-empty explanations for all qualifying erasers."
            )
            if n_found == 0:
                print("No candidates found — exiting.")
                return
            if args.auto_retry:
                print(f"[auto-retry] Proceeding with {n_found} images.")
            else:
                answer = input(
                    f"Do you want to retry the experiment with {n_found} images "
                    f"instead of {n_target}? [y/N] "
                ).strip().lower()
                if answer != "y":
                    print("Exiting.")
                    return
                print(f"Retrying with {n_found} images.")

        filtered_data = shared_data
        print(f"Behavior '{behavior_suffix}' (shared): {len(filtered_data)} instances")

        # ── Run 9 configurations ──────────────────────────────────────────────
        eraser_configs = [
            ("CO", "ortho",  ortho_eraser),
            ("CS", "splice", splice_eraser),
            ("CL", "leace",  leace_eraser),
        ]

        for prefix, eraser_type, eraser in eraser_configs:
            print(f"\n{'='*60}\nEraser: {eraser_type.upper()}\n{'='*60}")

            # --- per-instance concept signs (computed from shared filtered_data) ---
            filtered_eclips = [v[2] for v in filtered_data.values()]

            if eraser_type == "ortho":
                _, C_ord_signs = order_concept_strengths(filtered_eclips, C_vectors_N)

            elif eraser_type == "splice":
                C_ord_signs = torch.ones((len(filtered_data), args.n_concepts))

            elif eraser_type == "leace":
                eclips_stack = torch.stack(filtered_eclips).to(device).to(torch.float64)
                C_ord_signs  = torch.sign(eclips_stack @ C_vectors_N)

            # --- XpEnum ---
            xp_results = xp_eclips = xp_preds = xp_idxs = None
            if not args.no_xpenum:
                beh_id = f"{prefix}X{behavior_suffix}"
                xp_results, xp_eclips, xp_preds, xp_idxs = wrapper_XpEnum(
                    filtered_data, C_N, C_vectors_N, C_ord_signs, eraser, pred_fn,
                    args.experiments_per_behavior, args.xpenum_iters,
                    beh_id, results_dir, device,
                    precomputed=precomputed.get(eraser_type),
                    timeout=args.timeout,
                )

                # --- XpSatEnum (requires at least one XpEnum result) ---
                if not args.no_xpsatenum and xp_results:
                    if eraser_type == "leace":
                        eraser.keep_hashmap = True
                    beh_id = f"{prefix}S{behavior_suffix}"
                    wrapper_XpSatEnum(
                        filtered_data, C_N, C_vectors_N, C_ord_signs, eraser, pred_fn,
                        xp_results, xp_eclips, xp_preds, xp_idxs,
                        beh_id, results_dir, device,
                        timeout=args.timeout,
                    )

            # --- NaiveEnum ---
            if not args.no_naiveenum:
                if eraser_type == "leace":
                    eraser.keep_hashmap = True
                beh_id = f"{prefix}N{behavior_suffix}"
                wrapper_NaiveEnum(
                    filtered_data, C_N, C_vectors_N, C_ord_signs, eraser, pred_fn,
                    args.experiments_per_behavior, args.search_depth,
                    beh_id, results_dir, device,
                    timeout=args.timeout,
                )

    print("\nDone!")


if __name__ == "__main__":
    main()
