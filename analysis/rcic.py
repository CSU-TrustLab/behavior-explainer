#!/usr/bin/env python3
"""
Metric 8: Relative Cumulative Individual Coverage at Length K (RCIC@K).

For a threshold K:

    RCIC(K) = sum_{E : L(E) ≤ K} appearances(E)
              ─────────────────────────────────────
              sum_{all E} appearances(E)

where appearances(E) = number of image blocks where explanation E appears,
and L(E) = number of concepts in E (its size, regardless of sign).

A high RCIC at a small K means most explanation occurrences come from short,
easy-to-interpret explanations.

Same line style as individual_coverage.py:
  color  = erasure algorithm (orange / purple / green)
  style  = enumeration algorithm (solid / dashed / dotted)

CLI usage (run from repo root):
    python analysis/rcic.py \\
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
    if not path.exists():
        return False, False
    lines = path.read_text().strip().splitlines()
    partial = bool(lines) and "partial" in lines[0].lower()
    try:
        degenerate = float(lines[1]) == 0.0
    except (IndexError, ValueError):
        degenerate = False
    return partial, degenerate


def _rcic_curve(path: Path, max_k: int) -> Optional[list[float]]:
    """
    Parse a binary CSV and return RCIC(K) for K = 1 … max_k.

    Each (image, explanation) pair contributes one appearance count.
    Returns None if the file is missing or contains no data.
    """
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None

    xp_size:        dict[str, int] = {}   # explanation key → size
    xp_appearances: dict[str, int] = {}   # explanation key → appearance count

    for block in text.split("\n\n"):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        for ln in lines:
            try:
                comma = ln.index(",")
                size  = int(ln[:comma])
                key   = ln[comma + 1:]
            except (ValueError, IndexError):
                continue
            if key not in xp_size:
                xp_size[key] = size
            xp_appearances[key] = xp_appearances.get(key, 0) + 1

    total = sum(xp_appearances.values())
    if total == 0:
        return None

    return [
        sum(cnt for key, cnt in xp_appearances.items() if xp_size[key] <= k) / total
        for k in range(1, max_k + 1)
    ]


def _all_rcic(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path,
    max_k: int,
) -> dict[tuple[str, str], dict]:
    """Compute RCIC curves for all 9 configurations."""
    base = Path(intermediate_results_dir) / f"CM_MCS90_N{vocab_size}_{model}_{dataset}_e"
    result: dict = {}
    for ecode, ename in ERASERS:
        for acode, aname in ALGOS:
            fid = f"{ecode}{acode}{behavior}-N{vocab_size}_{model}_{dataset}"
            ft  = base / f"time_{fid}.csv"
            _, degenerate = _read_time_status(ft)
            if degenerate:
                result[(ename, aname)] = {"axp": None, "cxp": None}
                continue
            result[(ename, aname)] = {
                "axp": _rcic_curve(base / f"binary_{fid}_A.csv", max_k),
                "cxp": _rcic_curve(base / f"binary_{fid}_C.csv", max_k),
            }
    return result


def plot_rcic(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path = "intermediate_results",
    max_k: int = 8,
) -> plt.Figure:
    """
    Line plot: RCIC(K) (y, 0–1) vs explanation length K (x, 1–max_k).
    Returns a Figure with AXps (left) and CXps (right) subplots.
    Color = erasure algorithm; line style = enumeration algorithm.
    """
    data = _all_rcic(model, dataset, vocab_size, behavior,
                     intermediate_results_dir, max_k)
    ks = list(range(1, max_k + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, xp_key, title in [
        (axes[0], "axp", "AXps"),
        (axes[1], "cxp", "CXps"),
    ]:
        for _, ename in ERASERS:
            for _, aname in ALGOS:
                curve = data[(ename, aname)][xp_key]
                if curve is None:
                    continue
                ax.plot(
                    ks, curve,
                    color=ERASER_COLOR[ename],
                    linestyle=ALGO_STYLE[aname],
                    linewidth=1.8,
                )
        ax.set_xlim(1, max_k)
        ax.set_ylim(0, 1)
        ax.set_xticks(ks)
        ax.set_xlabel("Length K", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Relative Cumulative Individual Coverage", fontsize=11)

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
        f"Relative Cumulative Individual Coverage at K — "
        f"behavior {behavior}  ({model}, {dataset}, N={vocab_size})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric 8: relative cumulative individual coverage at length K."
    )
    parser.add_argument("--model",       required=True)
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--vocab-size",  required=True, type=int)
    parser.add_argument("--behavior",    required=True)
    parser.add_argument("--intermediate-results-dir", default="intermediate_results", metavar="DIR")
    parser.add_argument("--max-k", default=8, type=int,
                        help="maximum explanation length to show (default 8)")
    parser.add_argument("--output-dir", default=None, metavar="DIR")
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.intermediate_results_dir).parent / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_rcic(
        args.model, args.dataset, args.vocab_size, args.behavior,
        args.intermediate_results_dir, args.max_k,
    )
    plot_path = out_dir / f"metric8_rcic_{args.behavior}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()
