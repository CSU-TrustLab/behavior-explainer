"""
explain.py — Concept-based abductive and contrastive explanation engine.

Public interface
----------------
Data helpers:
    get_clip_embeddings, order_concept_strengths, filter_behavior,
    repack_tensors, get_sample_data, plot_confusion_matrix

Sanity checks (erase all concepts; verify prediction changes):
    sanity_check_ortho, sanity_check_splice, sanity_check_leace

Prediction helpers (used to build ``pred_fn`` in run_experiment.py):
    get_model_acc, get_zsclip_acc

Explanation core:
    CheckAXpFast, CheckCXp, OneAXp, OneCXp, xp_enum, naive_enum_search

Algorithm wrappers (write CSV results to results_dir):
    wrapper_XpEnum, wrapper_XpSatEnum, wrapper_NaiveEnum

Design note
-----------
All oracle and wrapper functions accept a ``pred_fn`` callable::

    pred_fn(e_clip_erased, norms, original_pred) -> (acc, per_image)

This callable encapsulates model-specific details (setup, alignment maps,
means, class vectors, device, norm_flag) and is constructed once per
experiment configuration in run_experiment.py.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.concept_eraser import ClipOrthoEraser, ClipSpliceEraser, LeaceEraserWrapper
from utils.mhs import minimal_hitting_setA as minimal_hitting_set  # noqa: F401 (re-exported)
from utils.mhs import minimal_hitting_setC as random_hitting_set


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def hash_tensor(tensor):
    """Stable hash of a tensor's values (used as a per-image key)."""
    return hash(tuple(tensor.reshape(-1).tolist()))


def remove_elements(a_list, list_to_remove):
    return list(set(a_list) - set(list_to_remove))


def filter_behavior(data_dict, B):
    """Remove images not matching behavior predicate B(predicted, label)."""
    outside = [k for k, v in data_dict.items() if not B(v[1], v[0])]
    for k in outside:
        del data_dict[k]
    return data_dict


def repack_tensors(data_dict, device):
    """Unpack a data dict into (all_preds, all_eclips, all_norms)."""
    all_preds  = torch.tensor([v[1] for v in data_dict.values()], dtype=torch.int).to(device)
    all_eclips = torch.stack([v[2] for v in data_dict.values()], dim=0).to(device).to(torch.float64)
    all_norms  = [v[3] for v in data_dict.values()]
    return all_preds, all_eclips, all_norms


def get_sample_data(data_dict, device, sample_size=500):
    """Return a random subsample of CLIP embeddings (used for LEACE fitting)."""
    eclips = torch.stack([v[2] for v in data_dict.values()], dim=0).to(device).to(torch.float64)
    print(f"eclips := {tuple(eclips.shape)}  (candidates)")
    idxs   = torch.randperm(eclips.shape[0])[:sample_size].long().to(device)
    eclips = torch.index_select(eclips, 0, idxs)
    print(f"eclips := {tuple(eclips.shape)}  (random sample)")
    return eclips


def order_concept_strengths(e_clip_list, C_vectors):
    """
    Compute per-image concept projection strengths.

    Parameters
    ----------
    e_clip_list : list of 1-D tensors, length n_images
    C_vectors   : (d, n_concepts) tensor

    Returns
    -------
    indices : LongTensor  (n_images, n_concepts) — per-image sorted order (descending |strength|)
    signs   : FloatTensor (n_images, n_concepts) — sign of each concept's raw projection
    """
    n_concepts = C_vectors.shape[1]
    all_strengths = []
    for e in e_clip_list:
        a = e.clone().detach().cpu().numpy()
        strengths = torch.zeros(n_concepts)
        for c in range(n_concepts):
            b = C_vectors[:, c].clone().detach().cpu().numpy()
            strengths[c] = np.dot(a, b) / np.linalg.norm(b)
        all_strengths.append(strengths)
    all_strengths = torch.stack(all_strengths)
    signs = torch.sign(all_strengths.detach().clone())
    _, indices = torch.sort(torch.abs(all_strengths), descending=True)
    return indices, signs


def plot_confusion_matrix(data_dict, title, file_name=None, results_dir=None):
    """Print (and optionally save) a confusion matrix for data_dict."""
    labels = [v[0] for v in data_dict.values()]
    preds  = [v[1] for v in data_dict.values()]
    cm = confusion_matrix(labels, preds)
    print(f"{title}  (n={len(data_dict)})\n{cm}")
    if file_name is not None and results_dir is not None:
        cm_labels = sorted(set(labels + preds))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
        disp.plot()
        plt.title(file_name)
        plt.savefig(Path(results_dir) / f"{file_name}.png", dpi=300)
        plt.close()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def get_model_acc(e_clip_erased, norms, align_inv, f_head, original_pred,
                  img_mean_map, device, norm_flag):
    """
    Predict from an erased CLIP-space embedding via the vision-model pipeline.

    Undoes centering (and optional normalisation), maps back through align_inv,
    and runs the vision-model head.

    Returns (acc, per_image_bool_tensor).
    """
    e = e_clip_erased.detach().clone().to(device)
    if norm_flag and norms is not None:
        if isinstance(norms, (int, float)):
            e = e * norms
        else:
            norms_t = torch.tensor(norms, dtype=e.dtype, device=device).unsqueeze(1)
            e = e * norms_t
    e = e + img_mean_map

    logits = f_head(align_inv(e.float()))
    _, predicted = torch.max(logits, 1)
    per_image = (predicted == original_pred)
    acc = per_image.sum().item() / predicted.shape[0]
    return acc, per_image


def get_zsclip_acc(e_clip_erased, norms, original_pred,
                   img_mean_clip, class_vectors, device, norm_flag):
    """
    Predict from an erased CLIP-space embedding via zero-shot CLIP.

    Returns (acc, per_image_bool_tensor).
    """
    e = e_clip_erased.detach().clone().to(device)
    if norm_flag and norms is not None:
        if isinstance(norms, (int, float)):
            e = e * norms
        else:
            norms_t = torch.tensor(norms, dtype=e.dtype, device=device).unsqueeze(1)
            e = e * norms_t
    e = e + img_mean_clip

    logits = e @ class_vectors
    _, predicted = torch.max(logits, 1)
    per_image = (predicted == original_pred)
    acc = per_image.sum().item() / predicted.shape[0]
    return acc, per_image


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_clip_embeddings(D, setup, f_enc, CLIP_enc, align, align_inv,
                        f_head, img_mean_map, img_mean_clip, class_vectors,
                        device, norm_flag, sample=-1):
    """
    Iterate over dataset D and compute per-image CLIP-space embeddings.

    Parameters
    ----------
    setup : 'vision-model' | 'zero-shot-clip'

    Returns
    -------
    dict  { image_hash: [label, predicted_class, e_clip, norm, image_cpu] }

    ``e_clip`` is the centred (and optionally unit-normalised) embedding used
    downstream for concept erasure.  ``predicted_class`` is derived from the
    full (un-erased) embedding so the data dict records the model's baseline
    predictions.
    """
    data = {}
    for batch_idx, (images, labels) in enumerate(tqdm(D)):
        images = images.to(device)
        labels = labels.to(device)

        # --- compute CLIP-space embedding and centre it ---
        if setup == "zero-shot-clip":
            e_clip = CLIP_enc(images).to(torch.float64) - img_mean_clip
        elif setup == "vision-model":
            e_clip = align(f_enc(images)).to(torch.float64) - img_mean_map
        else:
            raise ValueError(f"Unknown setup: {setup!r}")

        norm = e_clip.norm(dim=-1, keepdim=True)
        if norm_flag:
            e_clip = e_clip / norm
        e_clip_stored = e_clip.detach().clone().cpu()

        # --- reconstruct full embedding for baseline prediction ---
        e_pred = e_clip.clone()
        if norm_flag:
            e_pred = e_pred * norm
        if setup == "zero-shot-clip":
            logits = (e_pred + img_mean_clip) @ class_vectors
        else:  # vision-model
            logits = f_head(align_inv((e_pred + img_mean_map).float()))

        _, predicted = torch.max(logits, 1)

        for label, pred, e, n, img in zip(
            labels, predicted,
            torch.unbind(e_clip_stored, dim=0),
            torch.unbind(norm, dim=0),
            torch.unbind(images, dim=0),
        ):
            h = hash_tensor(img)
            assert h not in data, "Repeated hash value in dataset"
            data[h] = [label.item(), pred.item(), e.cpu(), n.item(), img.cpu()]

        if 0 < sample == batch_idx:
            break

    return data


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_check_ortho(data_dict, C_vectors, pred_fn, device):
    """
    Keep only images where erasing ALL concepts changes the prediction (Ortho).
    Images for which full erasure still predicts the same class are removed.
    """
    all_preds, all_eclips, all_norms = repack_tensors(data_dict, device)
    eraser = ClipOrthoEraser(C_vectors, device=device, dtype=torch.float64)
    erased = eraser.erase_all(all_eclips)
    _, per_image = pred_fn(erased, all_norms, all_preds)
    failed = [k for k, same in zip(data_dict.keys(), per_image.tolist()) if same]
    for k in failed:
        del data_dict[k]
    print(f"sanity_check_ortho:  {len(failed)} removed,  {len(data_dict)} remain")
    return data_dict


def sanity_check_splice(data_dict, C_vectors, pred_fn, device):
    """
    Keep only images where erasing all concepts changes the prediction AND
    keeping all concepts preserves it (Splice / ADMM).
    """
    all_preds, all_eclips, all_norms = repack_tensors(data_dict, device)
    eraser = ClipSpliceEraser(C_vectors, all_eclips, device=device, dtype=torch.float64)
    erased = eraser.erase_all()
    full   = eraser.erase_some([])
    _, per_erased = pred_fn(erased, all_norms, all_preds)
    _, per_full   = pred_fn(full,   all_norms, all_preds)
    failed = [
        k for k, e_same, f_same
        in zip(data_dict.keys(), per_erased.tolist(), per_full.tolist())
        if e_same or not f_same
    ]
    for k in failed:
        del data_dict[k]
    print(f"sanity_check_splice: {len(failed)} removed,  {len(data_dict)} remain")
    return data_dict


def sanity_check_leace(data_dict, train_eclips, C_vectors, pred_fn, device):
    """
    Keep only images where erasing ALL concepts changes the prediction (LEACE).
    """
    all_preds, all_eclips, all_norms = repack_tensors(data_dict, device)
    eraser = LeaceEraserWrapper(train_eclips, C_vectors, device=device, dtype=torch.float64)
    erased = eraser.erase_all(all_eclips)
    _, per_image = pred_fn(erased, all_norms, all_preds)
    failed = [k for k, same in zip(data_dict.keys(), per_image.tolist()) if same]
    for k in failed:
        del data_dict[k]
    print(f"sanity_check_leace:  {len(failed)} removed,  {len(data_dict)} remain")
    return data_dict


# ---------------------------------------------------------------------------
# Explanation core (NINA / AXp / CXp)
# ---------------------------------------------------------------------------

def CheckAXpFast(S, v_full, v_norm, ori_predicted, eraser, pred_fn, Axp=True):
    """
    Oracle: checks whether S is an AXp (Axp=True) or a CXp seed (Axp=False).

    For AXp: erases all concepts outside S; returns True if prediction holds.
    For CXp: erases all concepts inside S; returns True if prediction changes.

    v_full  : 1-D CPU tensor (the centred CLIP embedding)
    v_norm  : float or None (the pre-normalisation norm)
    """
    v = v_full.detach().clone().to(eraser.device).to(torch.float64)
    not_S = remove_elements(list(range(eraser.n)), list(S))

    if isinstance(eraser, ClipSpliceEraser):
        v_partial = eraser.erase_some(not_S)
    else:
        v_partial = eraser.erase_some(v, not_S)

    v_partial = torch.unsqueeze(v_partial, 0)
    acc, _ = pred_fn(v_partial, v_norm, ori_predicted)
    return (acc == 1.0) if Axp else (acc == 0.0)


def CheckCXp(S, F, v_full, v_norm, ori_predicted, eraser, pred_fn):
    """Oracle: checks whether S is a CXp by testing F \\ S."""
    return CheckAXpFast(
        remove_elements(F, list(S)), v_full, v_norm, ori_predicted, eraser, pred_fn, Axp=False
    )


def OneAXp(F, v_full, v_norm, ori_predicted, eraser, pred_fn):
    """Greedily reduce F to a single AXp."""
    S = F.copy()
    for i in F:
        smaller = remove_elements(S, [i])
        if CheckAXpFast(smaller, v_full, v_norm, ori_predicted, eraser, pred_fn):
            S = smaller
    return sorted(S)


def OneCXp(F, v_full, v_norm, realF, ori_predicted, eraser, pred_fn):
    """Greedily reduce F to a single CXp."""
    S = F.copy()
    for i in F:
        smaller = remove_elements(S, [i])
        if CheckCXp(smaller, realF.copy(), v_full, v_norm, ori_predicted, eraser, pred_fn):
            S = smaller
    return sorted(S)


def xp_enum(v, v_norm, ori_predicted, eraser, pred_fn, iters=250):
    """
    NINA-based enumeration of abductive and contrastive explanations (XpEnum).

    Alternates between growing a list of AXps (ninaN) and CXps (ninaP) using
    minimal hitting sets.  Stops early when no new hitting set is found.

    Returns (last_S, AXps, CXps, iteration_summary).
    """
    ninaN, ninaP = [], []
    ninaF   = list(range(eraser.n))
    summary = []
    last_S  = []

    for _ in range(iters):
        last_S = random_hitting_set(ninaP, ninaN)
        if _ > 0 and len(last_S) == 0:
            break

        X = Y = []
        is_AXp = CheckAXpFast(last_S, v, v_norm, ori_predicted, eraser, pred_fn)
        if is_AXp:
            X = OneAXp(last_S, v, v_norm, ori_predicted, eraser, pred_fn)
            if X not in ninaN:
                ninaN.append(X)
        else:
            F_minus_S = remove_elements(ninaF, last_S)
            Y = OneCXp(F_minus_S, v, v_norm, ninaF, ori_predicted, eraser, pred_fn)
            if Y not in ninaP:
                ninaP.append(Y)

        summary.append({
            "lenS": len(last_S), "lenA": len(ninaN), "lenC": len(ninaP),
            "lenX": len(X),      "lenY": len(Y),     "isAXp": int(is_AXp),
        })

    return last_S, ninaN, ninaP, summary


def naive_enum_search(v, ori_predicted, eraser, pred_fn, search_depth=2, findAXps=True):
    """
    Depth-bounded exhaustive search for AXps or CXps (NaiveEnum).

    Iterates over all candidate concept subsets up to ``search_depth`` in size.
    Supersets of already-confirmed explanations are pruned.

    Returns a list of explanation sets (Python sets of concept indices).

    Note: complexity grows exponentially with search_depth — keep it small (≤ 3).
    """
    ninaF  = list(range(eraser.n))
    pruned = []       # confirmed Xps (sets of concept indices)
    tried  = [set()]  # frontier: expand each element by one concept per depth

    for _ in range(search_depth):
        old_tried = tried
        tried = []
        for e_tried in old_tried:
            for c in range(eraser.n):
                if c in e_tried:
                    continue
                tentative = e_tried | {c}
                if any(p.issubset(tentative) for p in pruned):
                    continue
                if tentative in tried:
                    continue
                if findAXps:
                    is_Xp = CheckAXpFast(tentative, v, None, ori_predicted, eraser, pred_fn)
                else:
                    is_Xp = CheckCXp(tentative, ninaF.copy(), v, None, ori_predicted, eraser, pred_fn)
                if is_Xp:
                    pruned.append(tentative)
                else:
                    tried.append(tentative)

    return pruned


# ---------------------------------------------------------------------------
# Result writing
# ---------------------------------------------------------------------------

def _write_binary_csv(path, results_list, instance_idxs, C_len, C_ord_signs):
    """
    Write explanation results to a binary CSV file.

    One blank-line-separated block per instance.  Each row within a block:
        <explanation_size>,<positive_bits>,<negative_bits>
    where positive_bits[i] = 1 if concept i appears with a positive projection sign.
    """
    with open(path, "w") as f:
        for an_experiment, inst_idx in zip(results_list, instance_idxs):
            for explanation in an_experiment:
                if not explanation:
                    continue
                pos = [0] * C_len
                neg = [0] * C_len
                for idx in explanation:
                    if C_ord_signs[inst_idx, idx] > 0:
                        pos[idx] = 1
                    else:
                        neg[idx] = 1
                f.write(
                    f"{len(explanation)},"
                    f"{''.join(str(b) for b in pos)},"
                    f"{''.join(str(b) for b in neg)}\n"
                )
            f.write("\n")


# ---------------------------------------------------------------------------
# Algorithm wrappers
# ---------------------------------------------------------------------------

def wrapper_XpEnum(filtered_data, C, C_vectors, C_ord_signs, eraser, pred_fn,
                   experiments_per_behavior, xpenum_iters, behavior_id,
                   results_dir, device):
    """
    Run XpEnum on every instance in filtered_data (up to experiments_per_behavior).

    For each instance, runs the NINA loop to enumerate AXps and CXps.
    Writes binary CSV results to results_dir.

    Returns (all_results, all_eclips, all_predicted, all_instance_idxs)
    for use by wrapper_XpSatEnum.
    """
    all_results   = []
    all_eclips    = []
    all_predicted = []
    all_idxs      = []

    print(f"\n{behavior_id} ── XpEnum ──────────────────────────")
    t0 = time.time()

    for inst_idx, a_data in enumerate(tqdm(filtered_data.values())):
        ins_predicted = a_data[1]
        instance_v    = a_data[2]
        ins_v_norm    = a_data[3]

        if isinstance(eraser, ClipSpliceEraser):
            eraser = ClipSpliceEraser(C_vectors, instance_v, device=device, dtype=torch.float64)

        _, lastA, lastC, _ = xp_enum(
            instance_v, ins_v_norm, ins_predicted, eraser, pred_fn, iters=xpenum_iters
        )

        if lastA or lastC:
            all_results.append({"A": lastA, "C": lastC})
            all_eclips.append(instance_v)
            all_predicted.append(ins_predicted)
            all_idxs.append(inst_idx)
            if len(all_results) >= experiments_per_behavior:
                break

    C_len = len(C)
    for xp_type in ("A", "C"):
        _write_binary_csv(
            Path(results_dir) / f"binary_{behavior_id}_{xp_type}.csv",
            [r[xp_type] for r in all_results],
            all_idxs, C_len, C_ord_signs,
        )

    elapsed = time.time() - t0
    with open(Path(results_dir) / f"time_{behavior_id}.csv", "w") as f:
        f.write(f"XpEnum:\n{elapsed:.2f}\n")
    print(f"{behavior_id} ── done ({elapsed:.1f}s) ────────────────────")
    return all_results, all_eclips, all_predicted, all_idxs


def wrapper_XpSatEnum(filtered_data, C, C_vectors, C_ord_signs, eraser, pred_fn,
                      all_results, all_eclips, all_predicted, all_idxs,
                      behavior_id, results_dir, device):
    """
    XpSatEnum: saturate explanations found by XpEnum across instances.

    For every instance, tries explanations found on all other instances and
    checks whether they transfer (minimised if necessary).
    Overwrites the XpEnum CSV files with the enriched results.

    Note: uses the source instance's predicted class for the oracle check,
    which is equivalent to using the target's class when all instances in the
    behavior share the same prediction (e.g. B2, B6).
    """
    print(f"\n{behavior_id} ── XpSatEnum ───────────────────────")
    t0 = time.time()

    pre_lens = [{"A": len(r["A"]), "C": len(r["C"])} for r in all_results]
    new_axps = new_cxps = 0

    for out_idx, instance_v in enumerate(tqdm(all_eclips)):
        if isinstance(eraser, ClipSpliceEraser):
            eraser = ClipSpliceEraser(C_vectors, instance_v, device=device, dtype=torch.float64)

        ninaF = list(range(eraser.n))

        for in_idx, (r, pre, ins_predicted_in) in enumerate(
            zip(all_results, pre_lens, all_predicted)
        ):
            if out_idx == in_idx:
                continue

            for explanation in r["A"][:pre["A"]]:
                if explanation in all_results[out_idx]["A"]:
                    continue
                if CheckAXpFast(explanation, instance_v, None, ins_predicted_in, eraser, pred_fn):
                    axp = OneAXp(explanation, instance_v, None, ins_predicted_in, eraser, pred_fn)
                    if axp not in all_results[out_idx]["A"]:
                        new_axps += 1
                        all_results[out_idx]["A"].append(axp)

            for explanation in r["C"][:pre["C"]]:
                if explanation in all_results[out_idx]["C"]:
                    continue
                if CheckCXp(explanation, ninaF.copy(), instance_v, None,
                            ins_predicted_in, eraser, pred_fn):
                    cxp = OneCXp(explanation, instance_v, None, ninaF,
                                 ins_predicted_in, eraser, pred_fn)
                    if cxp not in all_results[out_idx]["C"]:
                        new_cxps += 1
                        all_results[out_idx]["C"].append(cxp)

    print(f"New explanations: AXps={new_axps}  CXps={new_cxps}")

    C_len = len(C)
    for xp_type in ("A", "C"):
        _write_binary_csv(
            Path(results_dir) / f"binary_{behavior_id}_{xp_type}.csv",
            [r[xp_type] for r in all_results],
            all_idxs, C_len, C_ord_signs,
        )

    elapsed = time.time() - t0
    with open(Path(results_dir) / f"time_{behavior_id}.csv", "w") as f:
        f.write(f"XpSatEnum:\n{elapsed:.2f}\n")
    print(f"{behavior_id} ── done ({elapsed:.1f}s) ────────────────────")


def wrapper_NaiveEnum(filtered_data, C, C_vectors, C_ord_signs, eraser, pred_fn,
                      experiments_per_behavior, search_depth, behavior_id,
                      results_dir, device):
    """
    NaiveEnum: depth-bounded exhaustive search for AXps and CXps on every instance.
    Writes binary CSV results to results_dir.
    """
    all_results   = []
    all_eclips    = []
    all_predicted = []
    all_idxs      = []

    print(f"\n{behavior_id} ── NaiveEnum (depth={search_depth}) ─────────")
    t0 = time.time()

    for inst_idx, a_data in enumerate(tqdm(filtered_data.values())):
        ins_predicted = a_data[1]
        instance_v    = a_data[2]

        if isinstance(eraser, ClipSpliceEraser):
            eraser = ClipSpliceEraser(C_vectors, instance_v, device=device, dtype=torch.float64)

        lastA = naive_enum_search(instance_v, ins_predicted, eraser, pred_fn,
                                  search_depth=search_depth, findAXps=True)
        lastC = naive_enum_search(instance_v, ins_predicted, eraser, pred_fn,
                                  search_depth=search_depth, findAXps=False)

        if lastA or lastC:
            all_results.append({"A": lastA, "C": lastC})
            all_eclips.append(instance_v)
            all_predicted.append(ins_predicted)
            all_idxs.append(inst_idx)
            if len(all_results) >= experiments_per_behavior:
                break

    C_len = len(C)
    for xp_type in ("A", "C"):
        _write_binary_csv(
            Path(results_dir) / f"binary_{behavior_id}_{xp_type}.csv",
            [r[xp_type] for r in all_results],
            all_idxs, C_len, C_ord_signs,
        )

    elapsed = time.time() - t0
    with open(Path(results_dir) / f"time_{behavior_id}.csv", "w") as f:
        f.write(f"NaiveEnum:\n{elapsed:.2f}\n")
    print(f"{behavior_id} ── done ({elapsed:.1f}s) ────────────────────")
