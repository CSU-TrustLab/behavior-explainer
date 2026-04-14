"""
test_build_vocab.py — Tests for src/build_vocab.py.

Structure:
  - Unit tests (always run): validate input file loading and class list definitions.
  - Integration test (skipped if intermediate_results/clip.pkl is absent): runs the
    full pipeline end-to-end with a small vocabulary size and verifies that output
    .txt files appear in vocabs/ and pickled vectors appear in intermediate_results/.

    To avoid overwriting any real vocabulary files, the integration test uses a
    "test_" prefix for all output names and cleans up afterwards.
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# clip.pkl was pickled with the t2c module from the original codebase.
# Add that source tree so it can be unpickled correctly.
_LEGACY_SRC = PROJECT_ROOT.parent / "src"
if _LEGACY_SRC.exists():
    sys.path.append(str(_LEGACY_SRC))  # append so our utils/ package takes priority

from src.build_vocab import (
    EUROSAT_CLASSES,
    RIVAL10_CLASSES,
    get_concept_vocab,
    load_core_wordnet,
    load_mscoco,
    load_prompts,
    save_vocab,
)
from utils.pickler import Pickler

VOCABS_DIR       = PROJECT_ROOT / "vocabs"
INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
CLIP_PKL         = INTERMEDIATE_DIR / "clip.pkl"

# ---------------------------------------------------------------------------
# Unit tests — no CLIP model required
# ---------------------------------------------------------------------------

def test_load_prompts():
    prompts = load_prompts(VOCABS_DIR / "prompts.txt")
    assert len(prompts) > 0, "prompts.txt should not be empty"
    assert all("{}" in p for p in prompts), \
        "Every prompt template should contain a '{}' placeholder"


def test_load_mscoco():
    words = load_mscoco(VOCABS_DIR / "mscoco.txt")
    assert len(words) > 0, "mscoco.txt should not be empty"
    assert all(isinstance(w, str) and len(w) > 0 for w in words), \
        "Every MS-COCO entry should be a non-empty string"


def test_load_core_wordnet():
    cwn = load_core_wordnet(VOCABS_DIR / "core-wordnet.txt")
    assert set(cwn.keys()) == {"a", "n", "v"}, \
        "Core WordNet dict should have keys 'a', 'n', 'v'"
    assert len(cwn["n"]) > 0, "Core WordNet should contain nouns"
    assert len(cwn["a"]) > 0, "Core WordNet should contain adjectives"


def test_class_lists():
    assert len(RIVAL10_CLASSES) == 10, "RIVAL10 should have exactly 10 classes"
    assert len(EUROSAT_CLASSES) == 10, "EuroSAT should have exactly 10 classes"
    assert len(set(RIVAL10_CLASSES)) == 10, "RIVAL10 class names should be unique"
    assert len(set(EUROSAT_CLASSES)) == 10, "EuroSAT class names should be unique"


# ---------------------------------------------------------------------------
# Integration test — requires intermediate_results/clip.pkl
# ---------------------------------------------------------------------------

@pytest.fixture
def test_vocab_names():
    """Yield test output names and clean up generated files afterwards."""
    names = {
        "RIVAL10": "test_MCS_90_NA_RIVAL10",
        "EuroSAT": "test_MCS_90_NA_EuroSAT",
    }
    yield names
    # Cleanup: remove all test artefacts so they don't accumulate
    for name in names.values():
        for path in [
            VOCABS_DIR / f"{name}.txt",
            INTERMEDIATE_DIR / f"{name}_vecs.pkl",
            INTERMEDIATE_DIR / f"{name}_class_vecs.pkl",
        ]:
            if path.exists():
                path.unlink()


@pytest.mark.skipif(not CLIP_PKL.is_file(), reason="intermediate_results/clip.pkl not found")
def test_build_vocab_end_to_end(test_vocab_names):
    """
    Runs the vocabulary generation pipeline with N=10 for speed.
    Checks that for each dataset:
      - a .txt file is created in vocabs/ with the right number of words
      - each word in the file is a non-empty string
      - pickled concept vectors are created in intermediate_results/
      - pickled class vectors are created in intermediate_results/
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("\nLoading CLIP model...")
    text_to_concept_obj = Pickler.read("clip")
    text_to_concept_obj.device = device
    clip_model = text_to_concept_obj.clip_model.to(device)
    clip_model.device = device

    prompts    = load_prompts(VOCABS_DIR / "prompts.txt")
    mscoco     = load_mscoco(VOCABS_DIR / "mscoco.txt")
    cwn        = load_core_wordnet(VOCABS_DIR / "core-wordnet.txt")
    candidates = cwn["n"] | cwn["a"]

    N           = 10   # small N for fast testing
    max_cos_sim = 90

    datasets = [
        ("RIVAL10", RIVAL10_CLASSES),
        ("EuroSAT", EUROSAT_CLASSES),
    ]

    with torch.no_grad():
        for dataset_name, classes in datasets:
            out_name = test_vocab_names[dataset_name]

            class_vecs = clip_model.get_concept_vectors(
                classes, prompts, mean_centered=False, BMP=0
            )
            vocab, vocab_vecs = get_concept_vocab(
                mscoco, N, candidates, classes, class_vecs,
                clip_model, prompts, max_cos_sim,
            )
            save_vocab(out_name, vocab, vocab_vecs.cpu(), class_vecs.cpu())

            # --- Check .txt file ---
            txt_path = VOCABS_DIR / f"{out_name}.txt"
            assert txt_path.exists(), \
                f"Vocab .txt not created for {dataset_name}: {txt_path}"

            words = txt_path.read_text().strip().splitlines()
            assert 0 < len(words) <= N, \
                f"Expected 1–{N} words in vocab, got {len(words)}"
            assert all(w.strip() for w in words), \
                "All vocab entries should be non-empty strings"

            # No class name should end up in the vocabulary
            for cls in classes:
                assert cls not in words, \
                    f"Class name '{cls}' should not appear in the concept vocab"

            # --- Check pickled concept vectors ---
            vecs_pkl = INTERMEDIATE_DIR / f"{out_name}_vecs.pkl"
            assert vecs_pkl.exists(), \
                f"Concept vector pkl not created for {dataset_name}: {vecs_pkl}"

            vecs = Pickler.read(out_name + "_vecs")
            assert vecs.shape[1] == len(words), \
                f"Vector count ({vecs.shape[1]}) should match word count ({len(words)})"
            assert vecs.shape[0] == 512, \
                f"CLIP embedding dimension should be 512, got {vecs.shape[0]}"

            # --- Check pickled class vectors ---
            class_vecs_pkl = INTERMEDIATE_DIR / f"{out_name}_class_vecs.pkl"
            assert class_vecs_pkl.exists(), \
                f"Class vector pkl not created for {dataset_name}: {class_vecs_pkl}"

            cls_vecs = Pickler.read(out_name + "_class_vecs")
            assert cls_vecs.shape[1] == len(classes), \
                f"Class vector count should match number of classes ({len(classes)})"
            assert cls_vecs.shape[0] == 512, \
                f"CLIP embedding dimension should be 512, got {cls_vecs.shape[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
