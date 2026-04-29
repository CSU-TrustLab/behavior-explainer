#!/usr/bin/env python3
"""
Metric 1: average (± std) number of AXps and CXps per image for a given behavior.

CLI usage (run from repo root):
    python analysis/avg_xp_count.py \
        --model resnet --dataset rival10 --vocab-size 300 --behavior B26
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ERASERS = [("CO", "Ortho"), ("CS", "SpLiCE"), ("CL", "LEACE")]
ALGOS   = [("X", "XpEnum"), ("S", "XpSatEnum"), ("N", "NaiveEnum")]


def _parse_binary_csv(path: Path) -> list[int]:
    """Return number of explanations per image block."""
    blocks = path.read_text().strip().split("\n\n")
    return [
        len([ln for ln in block.strip().split("\n") if ln.strip()])
        for block in blocks
        if block.strip()
    ]


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


def _fmt(counts: list[int], suffix: str) -> str:
    return f"{np.mean(counts):.2f} ± {np.std(counts):.2f}{suffix}"


def compute_avg_xp_count(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path = "intermediate_results",
) -> pd.DataFrame:
    """
    Build a 2 × 9 DataFrame with mean ± std of explanation counts per image.

    Rows    : ["AXps", "CXps"]
    Columns : MultiIndex (Eraser × Algorithm)
    Cells   : "mean ± std" strings; "-" = missing/degenerate; "*" suffix = partial run.
    """
    base = Path(intermediate_results_dir) / f"CM_MCS90_N{vocab_size}_{model}_{dataset}_e"

    columns = pd.MultiIndex.from_tuples(
        [(ename, aname) for _, ename in ERASERS for _, aname in ALGOS],
        names=["Eraser", "Algorithm"],
    )

    axp_row: dict = {}
    cxp_row: dict = {}

    for ecode, ename in ERASERS:
        for acode, aname in ALGOS:
            col = (ename, aname)
            fid = f"{ecode}{acode}{behavior}-N{vocab_size}_{model}_{dataset}"
            fa  = base / f"binary_{fid}_A.csv"
            fc  = base / f"binary_{fid}_C.csv"
            ft  = base / f"time_{fid}.csv"

            partial, degenerate = _read_time_status(ft)

            if degenerate or not fa.exists() or not fc.exists():
                axp_row[col] = cxp_row[col] = "-"
                continue

            sfx = "*" if partial else ""
            counts_a = _parse_binary_csv(fa)
            counts_c = _parse_binary_csv(fc)

            if not counts_a or not counts_c:
                axp_row[col] = cxp_row[col] = "-"
                continue

            axp_row[col] = _fmt(counts_a, sfx)
            cxp_row[col] = _fmt(counts_c, sfx)

    return pd.DataFrame(
        [axp_row, cxp_row],
        index=["AXps", "CXps"],
        columns=columns,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric 1: average number of explanations per image."
    )
    parser.add_argument("--model",       required=True, help="e.g. resnet")
    parser.add_argument("--dataset",     required=True, help="e.g. rival10")
    parser.add_argument("--vocab-size",  required=True, type=int, help="e.g. 300")
    parser.add_argument("--behavior",    required=True,
                        help="e.g. B26 (correct clf on class 6) or B523 (class 2 → class 3)")
    parser.add_argument("--intermediate-results-dir", default="intermediate_results",
                        metavar="DIR")
    parser.add_argument("--output", default=None, metavar="CSV",
                        help="output CSV path (default: results/metric1_<behavior>.csv)")
    args = parser.parse_args()

    df = compute_avg_xp_count(
        model=args.model,
        dataset=args.dataset,
        vocab_size=args.vocab_size,
        behavior=args.behavior,
        intermediate_results_dir=args.intermediate_results_dir,
    )

    print(df.to_string())

    out = (
        Path(args.output) if args.output
        else Path(args.intermediate_results_dir).parent / "results" / f"metric1_{args.behavior}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()