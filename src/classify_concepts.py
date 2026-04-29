#!/usr/bin/env python3
"""
src/classify_concepts.py — LLM-as-a-judge concept relevance classifier.

Uses the Anthropic API (Claude) to classify each concept in a base vocabulary
as RELEVANT or IRRELEVANT for a given model behavior.  The output JSON is read
by analysis/validity_ratio.py to compute the Validity Ratio metric.

Requires: ANTHROPIC_API_KEY environment variable.

CLI usage (run from the behavior-explainer repo root):
    python src/classify_concepts.py --dataset rival10 --behavior B26
    python src/classify_concepts.py --dataset rival10 --behavior B523

Output:
    intermediate_results/relevance_<dataset>_<behavior>.json
    {
        "horse":  "RELEVANT",
        "wheel":  "IRRELEVANT",
        ...
    }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import anthropic

PROJECT_ROOT     = Path(__file__).resolve().parent.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
VOCABS_DIR       = PROJECT_ROOT / "vocabs"

# Claude model used for classification (change to your preferred model)
CLAUDE_MODEL = "claude-sonnet-4-6"
BATCH_SIZE   = 200   # concepts per API call

# ---------------------------------------------------------------------------
# Class name maps
# ---------------------------------------------------------------------------

RIVAL10_CLASSES = {
    0: "truck", 1: "car", 2: "plane", 3: "ship",
    4: "cat",   5: "dog", 6: "equine", 7: "deer",
    8: "frog",  9: "bird",
}
EUROSAT_CLASSES = {
    0: "AnnualCrop",          1: "Forest",
    2: "HerbaceousVegetation", 3: "Highway",
    4: "Industrial",           5: "Pasture",
    6: "PermanentCrop",        7: "Residential",
    8: "River",                9: "SeaLake",
}

# ---------------------------------------------------------------------------
# Prompt templates (verbatim from the paper)
# ---------------------------------------------------------------------------

CORRECT_TEMPLATE = """\
Role: You are acting as an expert in Computer Vision and Explainable Artificial \
Intelligence (XAI). Your task is to evaluate the semantic relevance of concepts \
within a predefined vocabulary relative to a vision model's classification behavior.
Context: A vision model has correctly classified an image. \
Ground Truth Label: {class_a}. Model Prediction: {class_a}.
Task: You will be provided with a vocabulary of {n} concepts. Classify each concept \
as either RELEVANT or IRRELEVANT for the prediction of {class_a}.
Evaluation Criteria:
RELEVANT: A concept is relevant if its presence provides useful information about \
the class. Examples: `wheels' is relevant for `car'; `horns' is relevant for `deer'; \
`wet' is relevant for `ship'.
IRRELEVANT: Concepts that are semantically unrelated; these concepts provide no \
information about the class. Examples: `rectangular' is irrelevant for `cat'; \
`text' is irrelevant for `bird'.
Vocabulary: {vocab}
Output: RELEVANT: [comma-separated list], IRRELEVANT: [comma-separated list]."""

MISCLASS_TEMPLATE = """\
Role: You are acting as an expert in Computer Vision and Explainable Artificial \
Intelligence (XAI). Your task is to evaluate the semantic relevance of concepts \
within a predefined vocabulary relative to a vision model's incorrect classification \
behavior.
Context: A vision model has misclassified an image. \
Ground Truth Label: {class_a}. Model Prediction: {class_b}.
Task: You will be provided with a vocabulary of {n} concepts. Classify each concept \
as either RELEVANT or IRRELEVANT in explaining the confusion between {class_a} and \
{class_b}.
Evaluation Criteria:
RELEVANT: Concepts semantically associated with either {class_a} or {class_b}. \
Example: `wings' is relevant when mistaking a `frog' (ground truth) for a `bird' \
(prediction).
IRRELEVANT: Concepts semantically unrelated to both {class_a} and {class_b}. \
Example: `metallic' is irrelevant when mistaking a `cat' (ground truth) for a \
`dog' (prediction).
Vocabulary: {vocab}
Output: RELEVANT: [comma-separated list], IRRELEVANT: [comma-separated list]."""


# ---------------------------------------------------------------------------
# Behavior parsing
# ---------------------------------------------------------------------------

def parse_behavior(behavior: str, dataset: str) -> tuple[str, str, Optional[str]]:
    """
    Parse a behavior ID string like 'B26' or 'B523'.

    Returns (behavior_type, class_a_name, class_b_name_or_None).
    behavior_type is 'correct' (B2) or 'misclassification' (B5).
    """
    class_map = EUROSAT_CLASSES if "eurosat" in dataset.lower() else RIVAL10_CLASSES

    # behavior[0] == 'B', behavior[1] == behavior number
    btype = int(behavior[1])
    if btype == 2:
        class_a = class_map[int(behavior[2])]
        return "correct", class_a, None
    elif btype == 5:
        class_a = class_map[int(behavior[2])]
        class_b = class_map[int(behavior[3])]
        return "misclassification", class_a, class_b
    else:
        raise ValueError(
            f"Unsupported behavior type '{btype}' in '{behavior}'. "
            "Only B2 (correct) and B5 (misclassification) are supported."
        )


# ---------------------------------------------------------------------------
# API call and response parsing
# ---------------------------------------------------------------------------

def _build_prompt(
    behavior_type: str,
    class_a: str,
    class_b: Optional[str],
    batch: list[str],
) -> str:
    vocab_str = ", ".join(batch)
    n = len(batch)
    if behavior_type == "correct":
        return CORRECT_TEMPLATE.format(class_a=class_a, n=n, vocab=vocab_str)
    else:
        return MISCLASS_TEMPLATE.format(
            class_a=class_a, class_b=class_b, n=n, vocab=vocab_str
        )


def _parse_response(response_text: str, concepts: list[str]) -> dict[str, str]:
    """
    Parse LLM output into {word: 'RELEVANT'|'IRRELEVANT'}.
    Unclassified concepts default to 'IRRELEVANT'.
    """
    concepts_lower = {c.lower(): c for c in concepts}
    result = {c: "IRRELEVANT" for c in concepts}   # default

    for line in response_text.strip().splitlines():
        line = line.strip()
        tag = line[:line.index(":")] .strip().upper() if ":" in line else ""
        if tag not in ("RELEVANT", "IRRELEVANT"):
            continue
        label = tag
        words_str = line[line.index(":") + 1:]
        for w in words_str.split(","):
            w_clean = w.strip().lower()
            if w_clean in concepts_lower:
                result[concepts_lower[w_clean]] = label

    return result


def classify_batch(
    client: anthropic.Anthropic,
    batch: list[str],
    behavior_type: str,
    class_a: str,
    class_b: Optional[str],
    retries: int = 3,
    delay: float = 2.0,
) -> dict[str, str]:
    """Call Claude on one batch; retry on transient errors."""
    prompt = _build_prompt(behavior_type, class_a, class_b, batch)
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
            return _parse_response(text, batch)
        except Exception as exc:
            if attempt < retries - 1:
                print(f"  [retry {attempt + 1}/{retries}] {exc}")
                time.sleep(delay * (attempt + 1))
            else:
                print(f"  [failed] {exc} — defaulting batch to IRRELEVANT")
                return {c: "IRRELEVANT" for c in batch}


# ---------------------------------------------------------------------------
# Main classification pipeline
# ---------------------------------------------------------------------------

def classify_all(
    dataset: str,
    behavior: str,
    batch_size: int = BATCH_SIZE,
    output_path: Optional[Path] = None,
    force: bool = False,
) -> dict[str, str]:
    """
    Classify all concepts in the base vocabulary for the given behavior.
    Saves result to JSON and returns the dict.
    """
    vocab_dataset = "EuroSAT" if "eurosat" in dataset.lower() else "RIVAL10"
    vocab_path    = VOCABS_DIR / f"MCS_90_NA_{vocab_dataset}.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

    if output_path is None:
        output_path = INTERMEDIATE_DIR / f"relevance_{dataset.lower()}_{behavior}.json"

    if output_path.exists() and not force:
        print(f"Relevance file already exists: {output_path}\nUse --force to re-classify.")
        return json.loads(output_path.read_text())

    behavior_type, class_a, class_b = parse_behavior(behavior, dataset)
    print(f"Behavior: {behavior_type}  class_a={class_a}" +
          (f"  class_b={class_b}" if class_b else ""))

    concepts = vocab_path.read_text().strip().splitlines()
    print(f"Vocabulary: {len(concepts)} concepts  |  batch size: {batch_size}")

    client = anthropic.Anthropic()
    relevance: dict[str, str] = {}

    batches = [concepts[i: i + batch_size] for i in range(0, len(concepts), batch_size)]
    for b_idx, batch in enumerate(batches):
        print(f"  Batch {b_idx + 1}/{len(batches)}  ({len(batch)} concepts) …", end=" ", flush=True)
        result = classify_batch(client, batch, behavior_type, class_a, class_b)
        relevance.update(result)
        n_rel = sum(1 for v in result.values() if v == "RELEVANT")
        print(f"RELEVANT={n_rel}  IRRELEVANT={len(result) - n_rel}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(relevance, indent=2, sort_keys=True))
    n_total_rel = sum(1 for v in relevance.values() if v == "RELEVANT")
    print(f"\nSaved → {output_path}")
    print(f"Total: RELEVANT={n_total_rel}, IRRELEVANT={len(relevance) - n_total_rel}")
    return relevance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify vocabulary concepts as RELEVANT / IRRELEVANT for a behavior."
    )
    parser.add_argument("--dataset",  required=True, help="rival10 | eurosat")
    parser.add_argument("--behavior", required=True, help="e.g. B26 or B523")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Concepts per API call (default {BATCH_SIZE})")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="Output JSON path (default: intermediate_results/relevance_<dataset>_<behavior>.json)")
    parser.add_argument("--force", action="store_true",
                        help="Re-classify even if output file already exists")
    args = parser.parse_args()

    classify_all(
        dataset=args.dataset,
        behavior=args.behavior,
        batch_size=args.batch_size,
        output_path=Path(args.output) if args.output else None,
        force=args.force,
    )


if __name__ == "__main__":
    main()