#!/usr/bin/env python3
"""
Metric 2: compute time per image (seconds) for a given behavior.

Time is configuration-wise (one value per eraser × algorithm pair, not split by AXp/CXp).
For partial runs the reported time covers only the images actually processed.

CLI usage (run from repo root):
    python analysis/compute_time.py \
        --model resnet --dataset rival10 --vocab-size 300 --behavior B26
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ERASERS = [("CO", "Ortho"), ("CS", "SpLiCE"), ("CL", "LEACE")]
ALGOS   = [("X", "XpEnum"), ("S", "XpSatEnum"), ("N", "NaiveEnum")]


def _count_image_blocks(path: Path) -> int:
    """Count non-empty image blocks in a binary CSV."""
    blocks = path.read_text().strip().split("\n\n")
    return sum(1 for b in blocks if b.strip())


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


def compute_time_per_image(
    model: str,
    dataset: str,
    vocab_size: int,
    behavior: str,
    intermediate_results_dir: str | Path = "intermediate_results",
) -> pd.DataFrame:
    """
    Build a 1 × 9 DataFrame with compute time per image (seconds).

    Row     : ["Time (s/img)"]
    Columns : MultiIndex (Eraser × Algorithm)
    Cells   : elapsed time / n_images processed; "-" = missing/degenerate; "*" = partial run.
    """
    base = Path(intermediate_results_dir) / f"CM_MCS90_N{vocab_size}_{model}_{dataset}_e"

    columns = pd.MultiIndex.from_tuples(
        [(ename, aname) for _, ename in ERASERS for _, aname in ALGOS],
        names=["Eraser", "Algorithm"],
    )

    row: dict = {}

    for ecode, ename in ERASERS:
        for acode, aname in ALGOS:
            col = (ename, aname)
            fid = f"{ecode}{acode}{behavior}-N{vocab_size}_{model}_{dataset}"
            fa  = base / f"binary_{fid}_A.csv"
            ft  = base / f"time_{fid}.csv"

            partial, degenerate = _read_time_status(ft)

            if degenerate or not ft.exists() or not fa.exists():
                row[col] = "-"
                continue

            lines = ft.read_text().strip().splitlines()
            try:
                total_time = float(lines[1])
            except (IndexError, ValueError):
                row[col] = "-"
                continue

            n_images = _count_image_blocks(fa)
            if n_images == 0:
                row[col] = "-"
                continue

            sfx = "*" if partial else ""
            row[col] = f"{total_time / n_images:.2f}{sfx}"

    return pd.DataFrame([row], index=["Time (s/img)"], columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metric 2: compute time per image (seconds)."
    )
    parser.add_argument("--model",       required=True, help="e.g. resnet")
    parser.add_argument("--dataset",     required=True, help="e.g. rival10")
    parser.add_argument("--vocab-size",  required=True, type=int, help="e.g. 300")
    parser.add_argument("--behavior",    required=True,
                        help="e.g. B26 (correct clf on class 6) or B523 (class 2 → class 3)")
    parser.add_argument("--intermediate-results-dir", default="intermediate_results",
                        metavar="DIR")
    parser.add_argument("--output", default=None, metavar="CSV",
                        help="output CSV path (default: results/metric2_<behavior>.csv)")
    args = parser.parse_args()

    df = compute_time_per_image(
        model=args.model,
        dataset=args.dataset,
        vocab_size=args.vocab_size,
        behavior=args.behavior,
        intermediate_results_dir=args.intermediate_results_dir,
    )

    print(df.to_string())

    out = (
        Path(args.output) if args.output
        else Path(args.intermediate_results_dir).parent / "results" / f"metric2_{args.behavior}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()