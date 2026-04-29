#!/usr/bin/env python3
"""
Metric 5: Explanation size vs. individual coverage — violin plot.

Compares the individual coverage distribution of explanations grouped by
size L=1, L=2, L=3+ (number of concepts regardless of sign).
Each available (eraser × algorithm × xp_type) triple is one group on the x-axis.
Violins share the eraser color; strip plots are used when fewer than 5 data points.

CLI usage (run from repo root):
    python analysis/size_vs_coverage.py \
        --model resnet --dataset rival10 --vocab-size 300 --behavior B26
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np


ERASERS  = [("CO", "Ortho"), ("CS", "SpLiCE"), ("CL", "LEACE")]
ALGOS    = [("X", "XpEnum"), ("S", "XpSatEnum"), ("N", "NaiveEnum")]
XP_TYPES = [("A", "AXps"), ("C", "CXps")]

ERASER_COLOR = {"Ortho": "orange", "SpLiCE": "purple", "LEACE": "green"}
SIZE_GROUPS  = ["L=1", "L=2", "L=3+"]
MIN_VIOLIN   = 5  # fewer data points → strip plot


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


def _size_coverage_data(path: Path) -> Optional[list[tuple[int, float]]]:
    """
    Parse a binary CSV and return (size, individual_coverage) per unique explanation.
    Size = first column of each row (number of concepts, sign-independent).
    Coverage = # images containing this explanation / # total images.
    Returns None if the file is missing or empty.
    """
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None

    xp_info: dict[str, tuple[int, set]] = {}  # key → (size, {img_idx, …})
    img_idx = 0
    for block in text.split("\n\n"):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        for ln in lines:
            try:
                comma = ln.index(",")
                size  = int(ln[:comma])
                key   = ln[comma + 1:]  # "pos_bits,neg_bits"
            except (ValueError, IndexError):
                continue
            if key not in xp_info:
                xp_info[key] = (size, set())
            xp_info[key][1].add(img_idx)
        img_idx += 1

    n = img_idx
    if n == 0 or not xp_info:
        return None

    return [(size, len(imgs) / n) for size, imgs in xp_info.values()]


def _group_by_size(
    data: list[tuple[int, float]],
) -> dict[str, list[float]]:
    """Partition individual coverage values into size categories."""
    groups: dict[str, list[float]] = {"L=1": [], "L=2": [], "L=3+": []}
    for size, cov in data:
        key = "L=1" if size == 1 else "L=2" if size == 2 else "L=3+"
        groups[key].append(cov)
    return groups


def _draw_violin_or_strip(
    ax: plt.Axes,
    position: float,
    data: list[float],
    color: str,
    rng: np.random.Generator,
    violin_width: float = 0.8,
) -> None:
    """Draw a violin (≥ MIN_VIOLIN points) or a jittered strip plot."""
    if not data:
        return
    if len(data) >= MIN_VIOLIN:
        parts = ax.violinplot(
            [data], [position], widths=violin_width,
            showmedians=True, showextrema=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)
        for key in ("cmedians", "cbars", "cmaxes", "cmins"):
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)
    else:
        jitter = rng.uniform(-0.15, 0.15, len(data))
        ax.scatter(
            np.array([position] * len(data)) + jitter,
            data,
            color=color, alpha=0.8, s=30, zorder=3,
            edgecolors="black", linewidths=0.5,
        )


def plot_size_vs_coverage(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path = "intermediate_results",
    seed: int = 0,
) -> plt.Figure:
    """
    Violin / strip plot: individual coverage (y, 0–1) vs explanation size.
    X-axis has three levels:
      bottom  — L: 1, 2, 3+ (tick labels)
      middle  — (eraser / algorithm / xp_type) configuration label
      overall — 'L: number of concepts in explanations' (figure x-label)
    Violin color = eraser (orange / purple / green).
    """
    base = Path(intermediate_results_dir) / f"CM_MCS90_N{vocab_size}_{model}_{dataset}_e"
    rng  = np.random.default_rng(seed)

    # ── Collect available configurations ──────────────────────────────────────
    configs: list[tuple[str, str, str, list[tuple[int, float]]]] = []
    for ecode, ename in ERASERS:
        for acode, aname in ALGOS:
            ft = base / f"time_{ecode}{acode}{behavior}-N{vocab_size}_{model}_{dataset}.csv"
            _, degenerate = _read_time_status(ft)
            if degenerate:
                continue
            for fcode, xptype in XP_TYPES:
                fpath = base / f"binary_{ecode}{acode}{behavior}-N{vocab_size}_{model}_{dataset}_{fcode}.csv"
                data  = _size_coverage_data(fpath)
                if data:
                    configs.append((ename, aname, xptype, data))

    if not configs:
        raise ValueError("No data found for the given parameters.")

    # ── X-axis layout ─────────────────────────────────────────────────────────
    inner_step = 1.0   # spacing between the 3 size groups within one config
    inter_gap  = 1.5   # gap between config groups
    stride     = 2 * inner_step + inter_gap  # = 3.5 total per config

    group_centers: list[float] = []
    positions_map: list[dict[str, float]] = []
    for i in range(len(configs)):
        base_x = i * stride
        positions_map.append({
            "L=1":  base_x,
            "L=2":  base_x + inner_step,
            "L=3+": base_x + 2 * inner_step,
        })
        group_centers.append(base_x + inner_step)  # centre of three positions

    # ── Draw violins / strips ─────────────────────────────────────────────────
    fig_w = max(8, len(configs) * 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))

    for i, (ename, aname, xptype, data) in enumerate(configs):
        color  = ERASER_COLOR[ename]
        groups = _group_by_size(data)
        for sg in SIZE_GROUPS:
            _draw_violin_or_strip(ax, positions_map[i][sg], groups[sg], color, rng)

    # ── Separator lines between config groups ─────────────────────────────────
    for i in range(len(configs) - 1):
        sep_x = (positions_map[i]["L=3+"] + positions_map[i + 1]["L=1"]) / 2
        ax.axvline(sep_x, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # ── Bottom tick labels (L values) ─────────────────────────────────────────
    all_pos    = [positions_map[i][sg] for i in range(len(configs)) for sg in SIZE_GROUPS]
    all_labels = ["1", "2", "3+"] * len(configs)
    ax.set_xticks(all_pos)
    ax.set_xticklabels(all_labels, fontsize=8)

    # ── Middle x-axis labels (config names, one per group) ────────────────────
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (ename, aname, xptype, _) in enumerate(configs):
        ax.text(
            group_centers[i], -0.30,
            f"{ename}\n{aname}\n{xptype}",
            transform=blended, ha="center", va="top",
            fontsize=8, color=ERASER_COLOR[ename], linespacing=1.3,
            clip_on=False,
        )

    # ── Axes decoration ───────────────────────────────────────────────────────
    x_lo = positions_map[0]["L=1"]  - 0.7
    x_hi = positions_map[-1]["L=3+"] + 0.7
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Individual coverage", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        f"Explanation Size vs. Individual Coverage — "
        f"behavior {behavior}  ({model}, {dataset}, N={vocab_size})",
        fontsize=11,
    )

    # Overall x-axis label placed via fig.text so it clears the config labels
    fig.text(0.5, 0.01, "L: number of concepts in explanations",
             ha="center", va="bottom", fontsize=11)
    fig.subplots_adjust(bottom=0.40)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric 5: explanation size vs. individual coverage."
    )
    parser.add_argument("--model",       required=True, help="e.g. resnet")
    parser.add_argument("--dataset",     required=True, help="e.g. rival10")
    parser.add_argument("--vocab-size",  required=True, type=int, help="e.g. 300")
    parser.add_argument("--behavior",    required=True, help="e.g. B26 or B523")
    parser.add_argument("--intermediate-results-dir", default="intermediate_results", metavar="DIR")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="output directory (default: results/)")
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.intermediate_results_dir).parent / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_size_vs_coverage(
        args.model, args.dataset, args.vocab_size, args.behavior,
        args.intermediate_results_dir,
    )
    plot_path = out_dir / f"metric5_size_vs_coverage_{args.behavior}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()