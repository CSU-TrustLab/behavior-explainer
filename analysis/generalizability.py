#!/usr/bin/env python3
"""
Metric 6: Generalizability at K.

For each available (eraser × algorithm × xp_type) configuration — XpEnum and
NaiveEnum only (XpSatEnum excluded) — split the image blocks 50/50 (random,
seed=0) and compute IoU(topK_half1, topK_half2) for K = 1 … 5.

IoU of two topK explanation sets S1, S2:
    IoU = |S1 ∩ S2| / |S1 ∪ S2|

Bar plot:
  X-axis: two levels — outer: (eraser / algorithm / xp_type) config label;
          inner: K = 1, 2, 3, 4, 5
  Y-axis: IoU (0 – 1)
  Bar color = eraser (orange / purple / green)

CLI usage (run from repo root):
    python analysis/generalizability.py \
        --model resnet --dataset rival10 --vocab-size 300 --behavior B26
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np


ERASERS  = [("CO", "Ortho"), ("CS", "SpLiCE"), ("CL", "LEACE")]
ALGOS    = [("X", "XpEnum"), ("N", "NaiveEnum")]   # XpSatEnum excluded
XP_TYPES = [("A", "AXps"), ("C", "CXps")]

ERASER_COLOR = {"Ortho": "orange", "SpLiCE": "purple", "LEACE": "green"}
K_VALUES     = [1, 2, 3, 4, 5]
MIN_IMAGES   = 20   # skip config if fewer image blocks


def _read_time_status(path: Path) -> tuple[bool, bool]:
    """Return (is_partial, is_degenerate). Degenerate means elapsed time == 0.0."""
    if not path.exists():
        return False, False
    lines = path.read_text().strip().splitlines()
    partial = bool(lines) and "partial" in lines[0].lower()
    try:
        degenerate = float(lines[1]) == 0.0
    except (IndexError, ValueError):
        degenerate = False
    return partial, degenerate


def _parse_blocks(path: Path) -> Optional[list[list[str]]]:
    """
    Parse a binary CSV into a list of per-image explanation-key lists.
    Each explanation key is "pos_bits,neg_bits".
    Returns None if file is missing or empty.
    """
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None

    blocks: list[list[str]] = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        keys: list[str] = []
        for ln in lines:
            try:
                sep = ln.index(",")
                keys.append(ln[sep + 1:])   # "pos_bits,neg_bits"
            except ValueError:
                continue
        blocks.append(keys)
    return blocks if blocks else None


def _top_k_explanations(blocks: list[list[str]], k: int) -> frozenset:
    """Return the k most-frequent explanation keys across the given image blocks."""
    counts: dict[str, int] = {}
    for keys in blocks:
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
    top = sorted(counts, key=lambda x: counts[x], reverse=True)[:k]
    return frozenset(top)


def _iou(s1: frozenset, s2: frozenset) -> float:
    union = s1 | s2
    if not union:
        return 1.0   # both empty → identical
    return len(s1 & s2) / len(union)


def _generalizability_curve(
    blocks: list[list[str]],
    k_values: list[int],
    rng: np.random.Generator,
) -> list[float]:
    """
    Split blocks 50/50, compute IoU(topK_h1, topK_h2) for each K.
    Returns list aligned with k_values.
    """
    n = len(blocks)
    idx = rng.permutation(n)
    half = n // 2
    h1 = [blocks[i] for i in idx[:half]]
    h2 = [blocks[i] for i in idx[half:]]

    return [_iou(_top_k_explanations(h1, k), _top_k_explanations(h2, k))
            for k in k_values]


def _collect_configs(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path,
    seed: int,
) -> list[tuple[str, str, str, list[float]]]:
    """
    Collect (ename, aname, xptype, iou_per_k) for every available config.
    Skips degenerate runs and configs with < MIN_IMAGES image blocks.
    """
    base = Path(intermediate_results_dir) / f"CM_MCS90_N{vocab_size}_{model}_{dataset}_e"
    configs: list[tuple[str, str, str, list[float]]] = []

    for ecode, ename in ERASERS:
        for acode, aname in ALGOS:
            fid = f"{ecode}{acode}{behavior}-N{vocab_size}_{model}_{dataset}"
            ft  = base / f"time_{fid}.csv"
            _, degenerate = _read_time_status(ft)
            if degenerate:
                continue
            for fcode, xptype in XP_TYPES:
                fpath  = base / f"binary_{fid}_{fcode}.csv"
                blocks = _parse_blocks(fpath)
                if blocks is None or len(blocks) < MIN_IMAGES:
                    continue
                rng    = np.random.default_rng(seed)
                ious   = _generalizability_curve(blocks, K_VALUES, rng)
                configs.append((ename, aname, xptype, ious))

    return configs


def plot_generalizability(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path = "intermediate_results",
    seed: int = 0,
) -> plt.Figure:
    """
    Grouped bar chart: IoU (y, 0–1) vs K (1–5) per configuration.
    X-axis has two levels:
      bottom  — K values (1 … 5, one bar per K per config group)
      overall — (eraser / algorithm / xp_type) config label
    Bar color = eraser.
    """
    configs = _collect_configs(
        model, dataset, vocab_size, behavior, intermediate_results_dir, seed
    )

    if not configs:
        raise ValueError("No data found for the given parameters.")

    n_k     = len(K_VALUES)
    bar_w   = 0.6
    # Within each config group the K bars are spaced 1 unit apart
    # Groups are separated by a larger gap
    inner_step = 1.0
    inter_gap  = 1.5
    stride     = n_k * inner_step + inter_gap   # total width per config

    group_centers: list[float] = []
    k_positions:   list[list[float]] = []   # k_positions[config_i][k_idx]

    for i in range(len(configs)):
        base_x = i * stride
        positions = [base_x + k * inner_step for k in range(n_k)]
        k_positions.append(positions)
        group_centers.append(base_x + (n_k - 1) * inner_step / 2)

    fig_w = max(8, len(configs) * 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0))

    for i, (ename, aname, xptype, ious) in enumerate(configs):
        color = ERASER_COLOR[ename]
        for k_idx, iou in enumerate(ious):
            ax.bar(k_positions[i][k_idx], iou, width=bar_w,
                   color=color, edgecolor="black", linewidth=0.6, alpha=0.8)

    # ── Separator lines between config groups ─────────────────────────────────
    for i in range(len(configs) - 1):
        sep_x = (k_positions[i][-1] + k_positions[i + 1][0]) / 2
        ax.axvline(sep_x, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # ── Bottom tick labels (K values) ─────────────────────────────────────────
    all_pos    = [k_positions[i][k] for i in range(len(configs)) for k in range(n_k)]
    all_labels = [str(k) for k in K_VALUES] * len(configs)
    ax.set_xticks(all_pos)
    ax.set_xticklabels(all_labels, fontsize=8)

    # ── Config labels (outer x-axis level) ────────────────────────────────────
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (ename, aname, xptype, _) in enumerate(configs):
        ax.text(
            group_centers[i], -0.22,
            f"{ename}\n{aname}\n{xptype}",
            transform=blended, ha="center", va="top",
            fontsize=8, color=ERASER_COLOR[ename], linespacing=1.3,
            clip_on=False,
        )

    # ── Axes decoration ───────────────────────────────────────────────────────
    x_lo = k_positions[0][0]   - 0.7
    x_hi = k_positions[-1][-1] + 0.7
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Generalizability (IoU)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        f"Generalizability at K — behavior {behavior}  ({model}, {dataset}, N={vocab_size})",
        fontsize=11,
    )

    # K label via fig.text so it clears the config labels
    fig.text(0.5, 0.01, "K: number of top explanations",
             ha="center", va="bottom", fontsize=11)
    fig.subplots_adjust(bottom=0.38)

    # Legend
    handles = [
        mpatches.Patch(color=ERASER_COLOR[ename], label=ename)
        for _, ename in ERASERS
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.8)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric 6: generalizability at K."
    )
    parser.add_argument("--model",       required=True, help="e.g. resnet")
    parser.add_argument("--dataset",     required=True, help="e.g. rival10")
    parser.add_argument("--vocab-size",  required=True, type=int, help="e.g. 300")
    parser.add_argument("--behavior",    required=True, help="e.g. B26 or B523")
    parser.add_argument("--intermediate-results-dir", default="intermediate_results", metavar="DIR")
    parser.add_argument("--seed", default=0, type=int, help="RNG seed for 50/50 split (default 0)")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="output directory (default: results/)")
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.intermediate_results_dir).parent / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_generalizability(
        args.model, args.dataset, args.vocab_size, args.behavior,
        args.intermediate_results_dir, args.seed,
    )
    plot_path = out_dir / f"metric6_generalizability_{args.behavior}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()