#!/usr/bin/env python3
"""
Metric 7: Validity Ratio — plausibility of explanations w.r.t. LLM-graded relevance.

For each signed explanation (pos_bits, neg_bits):

    validity_ratio = (# positive concepts that are RELEVANT
                    + # negative concepts that are IRRELEVANT) / explanation_size

Fractions are weighted by explanation *appearances* (one count per image the
explanation appears in), so explanations with higher individual coverage matter more.

Three bins:
    fully_plausible   (green)  : validity_ratio >= 0.99
    partially_plausible (teal) : 0.01 < validity_ratio < 0.99
    fully_implausible  (red)   : validity_ratio <= 0.01

Prerequisite: run src/classify_concepts.py first to generate the relevance JSON,
then run src/run_experiment.py (or any experiment script) to generate the
energy-ordered vocab text file.

CLI usage (run from repo root):
    python analysis/validity_ratio.py \\
        --model resnet --dataset rival10 --vocab-size 300 --behavior B26
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


ERASERS  = [("CO", "Ortho"), ("CS", "SpLiCE"), ("CL", "LEACE")]
ALGOS    = [("X", "XpEnum"), ("S", "XpSatEnum"), ("N", "NaiveEnum")]
XP_TYPES = [("A", "AXps"), ("C", "CXps")]

ERASER_COLOR = {"Ortho": "orange", "SpLiCE": "purple", "LEACE": "green"}

PLAUS_ORDER   = ["fp", "pp", "fi"]
PLAUS_COLOR   = {"fp": "#2ca02c", "pp": "#80b1d3", "fi": "#d62728"}
PLAUS_LABEL   = {
    "fp": "Fully Plausible (≥0.99)",
    "pp": "Partially Plausible",
    "fi": "Fully Implausible (≤0.01)",
}
XP_HATCH = {"A": "", "C": "///"}
XP_NAME  = {"A": "AXps", "C": "CXps"}

FP_THRESHOLD = 0.99
FI_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _read_time_status(path: Path) -> tuple[bool, bool]:
    if not path.exists():
        return False, False
    lines = path.read_text().strip().splitlines()
    partial = bool(lines) and "partial" in lines[0].lower()
    try:
        degenerate = float(lines[1]) == 0.0
    except (IndexError, ValueError):
        degenerate = False
    return partial, degenerate


def _load_relevance(
    intermediate_results_dir: Path,
    dataset: str,
    behavior: str,
) -> dict[str, str]:
    """Load relevance JSON; raise FileNotFoundError if missing."""
    path = intermediate_results_dir / f"relevance_{dataset.lower()}_{behavior}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Relevance file not found: {path}\n"
            "Run  python src/classify_concepts.py "
            f"--dataset {dataset} --behavior {behavior}  first."
        )
    return json.loads(path.read_text())


def _load_energy_vocab(
    intermediate_results_dir: Path,
    dataset: str,
    mcs: int = 90,
) -> list[str]:
    """Load energy-ordered vocabulary; raise FileNotFoundError if missing."""
    vocab_dataset = "RIVAL10" if "rival10" in dataset.lower() else "EuroSAT"
    path = intermediate_results_dir / f"MCS_{mcs}_NA_{vocab_dataset}_e_vocab.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Energy vocab not found: {path}\n"
            "Run  python src/run_experiment.py ...  to generate it."
        )
    return path.read_text().strip().splitlines()


# ---------------------------------------------------------------------------
# Validity ratio computation
# ---------------------------------------------------------------------------

def _validity_ratio(
    key: str,
    concept_names: list[str],
    relevance: dict[str, str],
) -> float:
    """
    Compute validity ratio for one explanation key 'pos_bits,neg_bits'.
    Empty explanations → 1.0 (vacuously valid).
    """
    mid = key.index(",")
    pos_bits = key[:mid]
    neg_bits = key[mid + 1:]

    total = good = 0
    for i, (p, n) in enumerate(zip(pos_bits, neg_bits)):
        word = concept_names[i] if i < len(concept_names) else None
        if p == "1":
            total += 1
            if word and relevance.get(word, "IRRELEVANT") == "RELEVANT":
                good += 1
        elif n == "1":
            total += 1
            if word and relevance.get(word, "IRRELEVANT") == "IRRELEVANT":
                good += 1

    return good / total if total > 0 else 1.0


def _compute_fractions(
    path: Path,
    concept_names: list[str],
    relevance: dict[str, str],
) -> Optional[dict[str, float]]:
    """
    Parse binary CSV and compute weighted plausibility fractions.

    Each explanation is counted once per image block it appears in
    (weighted by individual coverage).

    Returns {"fp": float, "pp": float, "fi": float} or None if no data.
    """
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None

    xp_vr: dict[str, float] = {}          # key → validity_ratio (computed once)
    xp_appearances: dict[str, int] = {}   # key → appearance count

    for block in text.split("\n\n"):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        for ln in lines:
            try:
                comma = ln.index(",")
                key = ln[comma + 1:]
            except ValueError:
                continue
            if key not in xp_vr:
                xp_vr[key] = _validity_ratio(key, concept_names, relevance)
            xp_appearances[key] = xp_appearances.get(key, 0) + 1

    total = sum(xp_appearances.values())
    if total == 0:
        return None

    fp = sum(cnt for key, cnt in xp_appearances.items()
             if xp_vr[key] >= FP_THRESHOLD) / total
    fi = sum(cnt for key, cnt in xp_appearances.items()
             if xp_vr[key] <= FI_THRESHOLD) / total
    pp = 1.0 - fp - fi

    return {"fp": fp, "pp": max(pp, 0.0), "fi": fi}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_validity_ratio(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path = "intermediate_results",
    mcs: int = 90,
) -> plt.Figure:
    """
    Grouped bar chart: fraction of explanation appearances in each plausibility
    category vs (eraser × algorithm) configuration.

    Within each (eraser × algorithm) group, bars are arranged as three
    plausibility pairs (FP, PP, FI), each pair showing AXps (solid) and
    CXps (hatched) side by side.

    Bar color encodes plausibility (green / teal / red).
    Config labels below the bars are colored by eraser.
    """
    base = Path(intermediate_results_dir)
    relevance    = _load_relevance(base, dataset, behavior)
    energy_vocab = _load_energy_vocab(base, dataset, mcs)
    concept_names = energy_vocab[:vocab_size]

    exp_base = base / f"CM_MCS90_N{vocab_size}_{model}_{dataset}_e"

    # ── Collect data ─────────────────────────────────────────────────────────
    # configs: list of (ename, aname, {xptype: fractions_dict | None})
    configs: list[tuple[str, str, dict]] = []
    for ecode, ename in ERASERS:
        for acode, aname in ALGOS:
            fid = f"{ecode}{acode}{behavior}-N{vocab_size}_{model}_{dataset}"
            ft  = exp_base / f"time_{fid}.csv"
            _, degenerate = _read_time_status(ft)
            if degenerate:
                continue
            xp_data: dict[str, Optional[dict]] = {}
            for fcode, xptype in XP_TYPES:
                fpath = exp_base / f"binary_{fid}_{fcode}.csv"
                xp_data[fcode] = _compute_fractions(fpath, concept_names, relevance)
            if any(v is not None for v in xp_data.values()):
                configs.append((ename, aname, xp_data))

    if not configs:
        raise ValueError("No data found for the given parameters.")

    # ── Layout ───────────────────────────────────────────────────────────────
    bar_w        = 0.30
    pair_gap     = 0.04          # space between A and C within one plausibility pair
    subgroup_gap = 0.40          # space between plausibility pairs within a group
    inter_gap    = 1.10          # space between eraser/algo groups

    pair_step     = bar_w + pair_gap   # = 0.34 → pos_C relative to pos_A
    subgrp_stride = 2 * bar_w + pair_gap + subgroup_gap  # = 1.04

    # Positions within a group (relative to group base_x):
    # fp_A=0, fp_C=0.34, pp_A=1.04, pp_C=1.38, fi_A=2.08, fi_C=2.42
    rel_pos = {
        ("fp", "A"): 0,
        ("fp", "C"): pair_step,
        ("pp", "A"): subgrp_stride,
        ("pp", "C"): subgrp_stride + pair_step,
        ("fi", "A"): 2 * subgrp_stride,
        ("fi", "C"): 2 * subgrp_stride + pair_step,
    }
    group_width = 2 * subgrp_stride + 2 * bar_w + pair_gap   # ≈ 2.72
    stride      = group_width + inter_gap                      # ≈ 3.82

    subgrp_centers = {  # relative to group base_x
        "fp": (rel_pos[("fp", "A")] + rel_pos[("fp", "C")] + bar_w) / 2,
        "pp": (rel_pos[("pp", "A")] + rel_pos[("pp", "C")] + bar_w) / 2,
        "fi": (rel_pos[("fi", "A")] + rel_pos[("fi", "C")] + bar_w) / 2,
    }
    group_center_rel = (subgrp_centers["fp"] + subgrp_centers["fi"]) / 2

    n_groups = len(configs)
    fig_w    = max(12, n_groups * 3.5)
    fig, ax  = plt.subplots(figsize=(fig_w, 5.5))

    tick_positions: list[float] = []
    tick_labels:    list[str]   = []
    group_centers:  list[float] = []

    for g_idx, (ename, aname, xp_data) in enumerate(configs):
        base_x = g_idx * stride
        group_centers.append(base_x + group_center_rel)

        for plaus in PLAUS_ORDER:
            for xptype_code, xptype_name in XP_TYPES:
                frac_dict = xp_data.get(xptype_code)
                height    = frac_dict[plaus] if frac_dict else 0.0
                xpos      = base_x + rel_pos[(plaus, xptype_code)]
                ax.bar(
                    xpos, height, width=bar_w,
                    color=PLAUS_COLOR[plaus],
                    hatch=XP_HATCH[xptype_code],
                    edgecolor="black", linewidth=0.5,
                    alpha=0.85,
                )

        # Bottom tick at each sub-group center
        for plaus in PLAUS_ORDER:
            tick_positions.append(base_x + subgrp_centers[plaus])
            tick_labels.append({"fp": "FP", "pp": "PP", "fi": "FI"}[plaus])

    # ── Separator lines ───────────────────────────────────────────────────────
    for g_idx in range(n_groups - 1):
        sep_x = g_idx * stride + group_width + inter_gap / 2
        ax.axvline(sep_x, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # ── Bottom x-axis ticks (FP / PP / FI) ───────────────────────────────────
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)

    # ── Config labels (eraser + algo, colored by eraser) ─────────────────────
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for g_idx, (ename, aname, _) in enumerate(configs):
        ax.text(
            group_centers[g_idx], -0.22,
            f"{ename}\n{aname}",
            transform=blended, ha="center", va="top",
            fontsize=8, color=ERASER_COLOR[ename], linespacing=1.3,
            clip_on=False,
        )

    # ── Axes limits & decoration ──────────────────────────────────────────────
    x_lo = -0.6
    x_hi = (n_groups - 1) * stride + group_width + 0.6
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of Explanations", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        f"Validity Ratio — behavior {behavior}  ({model}, {dataset}, N={vocab_size})",
        fontsize=11,
    )

    fig.text(0.5, 0.01, "Validity Ratio Category  (FP = Fully Plausible, PP = Partial, FI = Fully Implausible)",
             ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(bottom=0.30)

    # ── Legend ────────────────────────────────────────────────────────────────
    plaus_handles = [
        mpatches.Patch(color=PLAUS_COLOR[p], label=PLAUS_LABEL[p])
        for p in PLAUS_ORDER
    ]
    xp_handles = [
        mpatches.Patch(
            facecolor="white", edgecolor="black", linewidth=0.8,
            hatch=XP_HATCH[code], label=name,
        )
        for code, name in XP_TYPES
    ]
    ax.legend(
        handles=plaus_handles + xp_handles,
        loc="upper right", fontsize=8, framealpha=0.9,
    )

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric 7: validity ratio of explanations."
    )
    parser.add_argument("--model",       required=True)
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--vocab-size",  required=True, type=int)
    parser.add_argument("--behavior",    required=True)
    parser.add_argument("--intermediate-results-dir", default="intermediate_results", metavar="DIR")
    parser.add_argument("--mcs", type=int, default=90)
    parser.add_argument("--output-dir", default=None, metavar="DIR")
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.intermediate_results_dir).parent / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_validity_ratio(
        args.model, args.dataset, args.vocab_size, args.behavior,
        args.intermediate_results_dir, args.mcs,
    )
    plot_path = out_dir / f"metric7_validity_ratio_{args.behavior}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()