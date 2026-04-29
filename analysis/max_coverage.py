#!/usr/bin/env python3
"""
Metric 4: Maximum Coverage at K.

Greedy maximum set cover: iteratively select the explanation with the highest
marginal gain in image coverage until K explanations are chosen.
Normalized by total images (1.0 = all images covered).

CLI usage (run from repo root):
    python analysis/max_coverage.py \
        --model resnet --dataset rival10 --vocab-size 300 --behavior B26
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


ERASERS = [("CO", "Ortho"), ("CS", "SpLiCE"), ("CL", "LEACE")]
ALGOS   = [("X", "XpEnum"), ("S", "XpSatEnum"), ("N", "NaiveEnum")]

ERASER_COLOR = {"Ortho": "orange", "SpLiCE": "purple", "LEACE": "green"}
ALGO_STYLE   = {"XpEnum": "-", "XpSatEnum": "--", "NaiveEnum": ":"}


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


def _explanation_image_sets(
    path: Path,
) -> tuple[Optional[list[frozenset]], Optional[int]]:
    """
    Parse a binary CSV and return:
      - one frozenset per unique explanation, containing the image indices it covers
      - total number of image blocks (images in the behavior)
    Returns (None, None) if the file is missing or empty.
    """
    if not path.exists():
        return None, None
    text = path.read_text().strip()
    if not text:
        return None, None

    image_xps: list[set] = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        xps: set = set()
        for ln in lines:
            sep = ln.index(",")
            xps.add(ln[sep + 1:])  # "pos_bits,neg_bits" — unique explanation key
        image_xps.append(xps)

    n = len(image_xps)
    if n == 0:
        return None, None

    xp_to_images: dict[str, set] = {}
    for img_idx, xps in enumerate(image_xps):
        for xp in xps:
            xp_to_images.setdefault(xp, set()).add(img_idx)

    return [frozenset(s) for s in xp_to_images.values()], n


def _greedy_max_coverage(
    subsets: list[frozenset], n_images: int, K: int
) -> list[float]:
    """
    Greedy maximum set cover for K steps.
    Each step picks the explanation with the highest marginal gain (new images covered).
    Returns normalized coverage for k = 1 … K; plateaus at 1.0 once all images covered.
    """
    covered: set = set()
    remaining = list(subsets)
    coverages: list[float] = []

    for _ in range(K):
        if remaining:
            best_idx = max(
                range(len(remaining)),
                key=lambda i: len(remaining[i] - covered),
            )
            covered |= remaining.pop(best_idx)
        coverages.append(len(covered) / n_images)

    return coverages


def _all_max_coverage(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path,
    K: int,
) -> dict[tuple[str, str], dict]:
    """Compute greedy max-coverage curves for all 9 configurations."""
    base = Path(intermediate_results_dir) / f"CM_MCS90_N{vocab_size}_{model}_{dataset}_e"
    result: dict = {}
    for ecode, ename in ERASERS:
        for acode, aname in ALGOS:
            fid = f"{ecode}{acode}{behavior}-N{vocab_size}_{model}_{dataset}"
            fa  = base / f"binary_{fid}_A.csv"
            fc  = base / f"binary_{fid}_C.csv"
            ft  = base / f"time_{fid}.csv"
            _, degenerate = _read_time_status(ft)
            if degenerate:
                result[(ename, aname)] = {"axp": None, "cxp": None}
                continue
            subsets_a, n_a = _explanation_image_sets(fa)
            subsets_c, n_c = _explanation_image_sets(fc)
            result[(ename, aname)] = {
                "axp": _greedy_max_coverage(subsets_a, n_a, K) if subsets_a else None,
                "cxp": _greedy_max_coverage(subsets_c, n_c, K) if subsets_c else None,
            }
    return result


def plot_max_coverage(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path = "intermediate_results",
    K: int = 15,
) -> plt.Figure:
    """
    Line plot: normalized max coverage (y, 0–1) vs K (x, 1–K).
    Color = erasure algorithm; line style = enumeration algorithm.
    Returns a Figure with AXps (left) and CXps (right) subplots.
    """
    data = _all_max_coverage(
        model, dataset, vocab_size, behavior, intermediate_results_dir, K
    )
    ks = list(range(1, K + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, xp_key, title in [
        (axes[0], "axp", "AXps"),
        (axes[1], "cxp", "CXps"),
    ]:
        for _, ename in ERASERS:
            for _, aname in ALGOS:
                covs = data[(ename, aname)][xp_key]
                if covs is None:
                    continue
                ax.plot(
                    ks,
                    covs,
                    color=ERASER_COLOR[ename],
                    linestyle=ALGO_STYLE[aname],
                    linewidth=1.8,
                )
        ax.set_xlim(1, K)
        ax.set_ylim(0, 1)
        ax.set_xticks(ks)
        ax.set_xlabel("K", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Normalized maximum coverage", fontsize=11)

    legend_handles = [
        mpatches.Patch(color=ERASER_COLOR[ename], label=ename)
        for _, ename in ERASERS
    ] + [
        mlines.Line2D(
            [], [], color="black",
            linestyle=ALGO_STYLE[aname], linewidth=1.8, label=aname,
        )
        for _, aname in ALGOS
    ]
    axes[1].legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.8)

    fig.suptitle(
        f"Maximum Coverage at K — behavior {behavior}  ({model}, {dataset}, N={vocab_size})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric 4: Maximum Coverage at K."
    )
    parser.add_argument("--model",       required=True, help="e.g. resnet")
    parser.add_argument("--dataset",     required=True, help="e.g. rival10")
    parser.add_argument("--vocab-size",  required=True, type=int, help="e.g. 300")
    parser.add_argument("--behavior",    required=True, help="e.g. B26 or B523")
    parser.add_argument("--intermediate-results-dir", default="intermediate_results", metavar="DIR")
    parser.add_argument("--K", default=15, type=int, help="max explanations to select (default 15)")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="output directory (default: results/)")
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.intermediate_results_dir).parent / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_max_coverage(
        args.model, args.dataset, args.vocab_size, args.behavior,
        args.intermediate_results_dir, args.K,
    )
    plot_path = out_dir / f"metric4_max_coverage_{args.behavior}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()