#!/usr/bin/env python3
"""
Metric 3: Individual Coverage of explanations for a given behavior.

Individual coverage of explanation E = (# images where E appears) / (# images processed).
A value of 1 means E accounts for every image in the behavior.

CLI usage (run from repo root):
    python analysis/individual_coverage.py \
        --model resnet --dataset rival10 --vocab-size 300 --behavior B26
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


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


def _coverage_ranked(path: Path) -> Optional[list[float]]:
    """
    Parse a binary CSV and return individual coverages sorted descending.
    Returns None if the file is missing or contains no valid explanation blocks.
    """
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None

    image_sets: list[set] = []
    for block in text.split("\n\n"):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        xps: set = set()
        for ln in lines:
            sep = ln.index(",")
            xps.add(ln[sep + 1:])  # "pos_bits,neg_bits" — unique explanation key
        image_sets.append(xps)

    n = len(image_sets)
    if n == 0:
        return None

    counts: dict[str, int] = {}
    for xps in image_sets:
        for xp in xps:
            counts[xp] = counts.get(xp, 0) + 1

    return sorted([c / n for c in counts.values()], reverse=True)


def _all_coverage(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path,
) -> dict[tuple[str, str], dict]:
    """
    Compute coverage data for all 9 configurations.
    Returns {(eraser_name, algo_name): {"axp": list|None, "cxp": list|None, "partial": bool}}.
    """
    base = Path(intermediate_results_dir) / f"CM_MCS90_N{vocab_size}_{model}_{dataset}_e"
    result: dict = {}
    for ecode, ename in ERASERS:
        for acode, aname in ALGOS:
            fid = f"{ecode}{acode}{behavior}-N{vocab_size}_{model}_{dataset}"
            fa  = base / f"binary_{fid}_A.csv"
            fc  = base / f"binary_{fid}_C.csv"
            ft  = base / f"time_{fid}.csv"
            partial, degenerate = _read_time_status(ft)
            if degenerate:
                result[(ename, aname)] = {"axp": None, "cxp": None, "partial": False}
            else:
                result[(ename, aname)] = {
                    "axp": _coverage_ranked(fa),
                    "cxp": _coverage_ranked(fc),
                    "partial": partial,
                }
    return result


def compute_individual_coverage_table(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path = "intermediate_results",
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build top_k × 9 DataFrames of individual coverage values.

    Returns (df_axp, df_cxp).
    Rows: ranks 1..top_k. Columns: MultiIndex (Eraser × Algorithm).
    Cells: "0.XXXX" strings; "-" = missing/degenerate; "*" suffix = partial run.
    """
    data = _all_coverage(model, dataset, vocab_size, behavior, intermediate_results_dir)

    columns = pd.MultiIndex.from_tuples(
        [(ename, aname) for _, ename in ERASERS for _, aname in ALGOS],
        names=["Eraser", "Algorithm"],
    )

    def _build_df(xp_key: str) -> pd.DataFrame:
        rows = []
        for rank in range(1, top_k + 1):
            row: dict = {}
            for _, ename in ERASERS:
                for _, aname in ALGOS:
                    col = (ename, aname)
                    info = data[col]
                    covs = info[xp_key]
                    sfx  = "*" if info["partial"] else ""
                    row[col] = (
                        f"{covs[rank - 1]:.4f}{sfx}"
                        if covs and rank <= len(covs)
                        else "-"
                    )
            rows.append(row)
        df = pd.DataFrame(rows, index=range(1, top_k + 1))
        df.index.name = "Rank"
        df.columns = pd.MultiIndex.from_tuples(
            df.columns.tolist(), names=["Eraser", "Algorithm"]
        )
        return df

    return _build_df("axp"), _build_df("cxp")


def plot_coverage_curves(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path = "intermediate_results",
    max_rank: int = 50,
) -> plt.Figure:
    """
    Line plot: individual coverage (y, 0–1) vs rank of explanation (x, 1–max_rank).
    One line per available configuration. Color = eraser; line style = algorithm.
    Returns a Figure with AXps (left) and CXps (right) subplots.
    """
    data = _all_coverage(model, dataset, vocab_size, behavior, intermediate_results_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, xp_key, title in [
        (axes[0], "axp", "AXps"),
        (axes[1], "cxp", "CXps"),
    ]:
        for _, ename in ERASERS:
            for _, aname in ALGOS:
                info = data[(ename, aname)]
                covs = info[xp_key]
                if not covs:
                    continue
                n = min(len(covs), max_rank)
                ax.plot(
                    range(1, n + 1),
                    covs[:n],
                    color=ERASER_COLOR[ename],
                    linestyle=ALGO_STYLE[aname],
                    linewidth=1.8,
                )
        ax.set_xlim(1, max_rank)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Rank of explanation", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Individual coverage", fontsize=11)

    # Compact shared legend: color patches (erasers) + line styles (algorithms)
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
    axes[1].legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.8)

    fig.suptitle(
        f"Individual Coverage — behavior {behavior}  ({model}, {dataset}, N={vocab_size})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric 3: individual coverage of explanations."
    )
    parser.add_argument("--model",       required=True, help="e.g. resnet")
    parser.add_argument("--dataset",     required=True, help="e.g. rival10")
    parser.add_argument("--vocab-size",  required=True, type=int, help="e.g. 300")
    parser.add_argument("--behavior",    required=True, help="e.g. B26 or B523")
    parser.add_argument("--intermediate-results-dir", default="intermediate_results", metavar="DIR")
    parser.add_argument("--top-k",    default=10, type=int, help="rows in top-k table (default 10)")
    parser.add_argument("--max-rank", default=50, type=int, help="x-axis range in plot (default 50)")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="output directory (default: results/)")
    args = parser.parse_args()

    out_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.intermediate_results_dir).parent / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df_axp, df_cxp = compute_individual_coverage_table(
        args.model, args.dataset, args.vocab_size, args.behavior,
        args.intermediate_results_dir, args.top_k,
    )

    print(f"=== Top-{args.top_k} AXps Individual Coverage ===")
    print(df_axp.to_string())
    print(f"\n=== Top-{args.top_k} CXps Individual Coverage ===")
    print(df_cxp.to_string())

    df_axp.to_csv(out_dir / f"metric3_axp_{args.behavior}.csv")
    df_cxp.to_csv(out_dir / f"metric3_cxp_{args.behavior}.csv")
    print(f"\nTables saved → {out_dir}/metric3_{{axp,cxp}}_{args.behavior}.csv")

    fig = plot_coverage_curves(
        args.model, args.dataset, args.vocab_size, args.behavior,
        args.intermediate_results_dir, args.max_rank,
    )
    plot_path = out_dir / f"metric3_coverage_curves_{args.behavior}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()