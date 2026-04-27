"""
test_run_experiment.py — Integration test for src/run_experiment.py (step #5).

Tests the XpEnum algorithm on RIVAL10, behavior 2 (correctly classified cars,
class_idx=1), with energy vocabulary ordering, at most 50 images, and a 5-minute
per-configuration timeout.

Structure
---------
- Unit tests (always run): CSV format parsing helpers and _write_binary_csv invariants.
- Integration test (skipped when required intermediate files are absent): calls
  run_experiment.main() and asserts that the output binary CSV files are created
  with the correct names and satisfy the binary encoding invariants.

Required intermediate files (produced by steps 1–4):
    intermediate_results/clip.pkl
    intermediate_results/resnet_rival10_finetuned.pkl
    intermediate_results/resnet_rival10_to_clip.pkl
    intermediate_results/clip_to_resnet_rival10.pkl
    intermediate_results/MCS_90_NA_rival10_vecs.pkl
    intermediate_results/MCS_90_NA_rival10_class_vecs.pkl
    vocabs/MCS_90_NA_rival10.txt
    datasets/RIVAL10/meta/train_test_split_by_url.json

Binary CSV format recap
-----------------------
Each line within a block:
    <size>,<positive_bits>,<negative_bits>
where positive_bits[i] = 1 iff concept i appears in the explanation with a positive
sign (positive cosine similarity / projection), and negative_bits[i] = 1 iff concept
i appears with a negative sign.  The two bit-strings are mutually exclusive per line:
no concept can be simultaneously positive and negative in the same explanation.
Blank lines separate per-image blocks; consecutive lines within a block are different
explanations for the same image.  A concept *may* carry a different sign across blocks
(i.e. positive for one image, negative for another).
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# clip.pkl was pickled with the legacy t2c module from the original codebase.
_LEGACY_SRC = PROJECT_ROOT.parent / "src"
if _LEGACY_SRC.exists():
    sys.path.append(str(_LEGACY_SRC))

INTERMEDIATE_DIR = PROJECT_ROOT / "intermediate_results"
VOCABS_DIR       = PROJECT_ROOT / "vocabs"
DATASETS_DIR     = PROJECT_ROOT / "datasets"

# ---------------------------------------------------------------------------
# Test parameters (must mirror run_experiment.py naming logic exactly)
# ---------------------------------------------------------------------------

_MODEL      = "resnet_rival10"
_BEHAVIOR   = 2          # correctly classified instances of class_idx
_CLASS_IDX  = 1          # class 1 = "car" in RIVAL10
_MCS        = 90
_N_CONCEPTS = 200
_TIMEOUT    = 300        # 5 minutes per configuration
_MAX_IMAGES = 50

# Derived naming components (see run_experiment.py: parse_args / main)
_MODEL_TAG      = f"_{_MODEL}"             # setup == "vision-model"
_ENERGY_TAG     = "_e"                     # energy_order == True
_OTHER_STR      = "-"                      # other_class_idx == -1
_CM_NAME        = f"CM_MCS{_MCS}_N{_N_CONCEPTS}{_MODEL_TAG}{_ENERGY_TAG}"
_BEH_SUFFIX     = f"B{_BEHAVIOR}{_CLASS_IDX}{_OTHER_STR}N{_N_CONCEPTS}{_MODEL_TAG}"
_ERASER_PREFIXES = [("CO", "ortho"), ("CS", "splice"), ("CL", "leace")]

# ---------------------------------------------------------------------------
# Skip condition for the integration test
# ---------------------------------------------------------------------------

_REQUIRED_FILES = [
    INTERMEDIATE_DIR / "clip.pkl",
    INTERMEDIATE_DIR / f"{_MODEL}_finetuned.pkl",
    INTERMEDIATE_DIR / f"{_MODEL}_to_clip.pkl",
    INTERMEDIATE_DIR / f"clip_to_{_MODEL}.pkl",
    INTERMEDIATE_DIR / f"MCS_{_MCS}_NA_rival10_vecs.pkl",
    INTERMEDIATE_DIR / f"MCS_{_MCS}_NA_rival10_class_vecs.pkl",
    VOCABS_DIR      / f"MCS_{_MCS}_NA_rival10.txt",
    DATASETS_DIR    / "RIVAL10" / "meta" / "train_test_split_by_url.json",
]

_missing    = [str(p) for p in _REQUIRED_FILES if not p.is_file()]
_SKIP_COND  = bool(_missing)
_SKIP_MSG   = (
    "Required intermediate files missing (run steps 1–4 first): "
    + ", ".join(_missing)
) if _missing else ""


# ---------------------------------------------------------------------------
# CSV parsing and format-checking helpers (shared by unit and integration tests)
# ---------------------------------------------------------------------------

def parse_binary_csv(path: Path) -> list:
    """
    Parse a binary XpEnum CSV into a list of per-image blocks.

    Returns list[list[tuple[int, list[int], list[int]]]] where each inner list
    is one image block (possibly empty) and each tuple is (size, pos_bits, neg_bits).

    Blank lines mark block boundaries; consecutive non-blank lines belong to the
    same block.
    """
    blocks: list = []
    current: list = []

    for raw in path.read_text().splitlines():
        if not raw.strip():                     # blank line → end of block
            blocks.append(current)
            current = []
        else:
            size_str, pos_str, neg_str = raw.split(",", 2)
            current.append((
                int(size_str),
                [int(c) for c in pos_str],
                [int(c) for c in neg_str],
            ))

    if current:                                 # file doesn't end with a blank line
        blocks.append(current)

    return blocks


def assert_csv_invariants(path: Path, n_concepts: int) -> list:
    """
    Assert all format invariants on a binary CSV file.

    Invariants checked:
      1. File exists.
      2. Each data line parses as <size>,<pos_bits>,<neg_bits>.
      3. pos_bits and neg_bits are binary strings of length n_concepts.
      4. size == popcount(pos_bits) + popcount(neg_bits).
      5. No concept appears as both positive and negative in the same explanation
         (pos[i] and neg[i] are never both 1 for the same index i).

    Returns the parsed blocks for further inspection.
    """
    assert path.is_file(), f"Expected CSV output file not found: {path}"

    blocks = parse_binary_csv(path)

    for block_idx, block in enumerate(blocks):
        for exp_idx, (size, pos, neg) in enumerate(block):
            loc = f"file={path.name!r}, block={block_idx}, explanation={exp_idx}"

            assert len(pos) == n_concepts, (
                f"{loc}: pos_bits length {len(pos)} != n_concepts {n_concepts}"
            )
            assert len(neg) == n_concepts, (
                f"{loc}: neg_bits length {len(neg)} != n_concepts {n_concepts}"
            )
            assert all(b in (0, 1) for b in pos), (
                f"{loc}: pos_bits contains a non-binary value"
            )
            assert all(b in (0, 1) for b in neg), (
                f"{loc}: neg_bits contains a non-binary value"
            )
            assert size == sum(pos) + sum(neg), (
                f"{loc}: size={size} but sum(pos)={sum(pos)}, sum(neg)={sum(neg)}"
            )
            for i in range(n_concepts):
                assert not (pos[i] == 1 and neg[i] == 1), (
                    f"{loc}: concept {i} has pos[i]=1 AND neg[i]=1 "
                    f"(mutual exclusion violated)"
                )

    return blocks


# ---------------------------------------------------------------------------
# Unit tests — no intermediate files required
# ---------------------------------------------------------------------------

def test_parse_binary_csv_basic(tmp_path):
    """parse_binary_csv correctly splits a two-block file."""
    n = 5
    csv = tmp_path / "test.csv"
    # Block 0: two explanations; block 1: one explanation
    csv.write_text(
        "2,11000,00000\n"
        "1,00100,00000\n"
        "\n"
        "1,00001,00000\n"
        "\n"
    )
    blocks = parse_binary_csv(csv)
    assert len(blocks) == 2
    assert len(blocks[0]) == 2
    assert len(blocks[1]) == 1
    assert blocks[0][0] == (2, [1, 1, 0, 0, 0], [0, 0, 0, 0, 0])
    assert blocks[1][0] == (1, [0, 0, 0, 0, 1], [0, 0, 0, 0, 0])


def test_parse_binary_csv_empty_block(tmp_path):
    """An image with no explanations contributes an empty block (just a blank line)."""
    csv = tmp_path / "test.csv"
    csv.write_text(
        "1,10000,00000\n"
        "\n"
        "\n"            # empty block for the second image
        "1,01000,00000\n"
        "\n"
    )
    blocks = parse_binary_csv(csv)
    assert len(blocks) == 3
    assert len(blocks[0]) == 1   # one explanation
    assert len(blocks[1]) == 0   # empty block
    assert len(blocks[2]) == 1   # one explanation


def test_assert_csv_invariants_valid(tmp_path):
    """assert_csv_invariants passes on a correctly formatted file."""
    n = 4
    csv = tmp_path / "valid.csv"
    csv.write_text(
        "2,1100,0000\n"   # size 2, two positive concepts, no negative
        "1,0000,0001\n"   # size 1, one negative concept
        "\n"
        "1,0010,0000\n"
        "\n"
    )
    blocks = assert_csv_invariants(csv, n_concepts=n)
    assert len(blocks) == 2


def test_assert_csv_invariants_size_mismatch(tmp_path):
    """assert_csv_invariants raises if size != popcount(pos) + popcount(neg)."""
    csv = tmp_path / "bad_size.csv"
    csv.write_text("3,1100,0000\n\n")  # size=3 but only 2 bits set
    with pytest.raises(AssertionError, match="size=3"):
        assert_csv_invariants(csv, n_concepts=4)


def test_assert_csv_invariants_mutual_exclusion(tmp_path):
    """assert_csv_invariants raises if a concept appears as both positive and negative."""
    csv = tmp_path / "bad_sign.csv"
    # size=3 matches sum(pos)=2 + sum(neg)=1, so the size check passes;
    # concept 0 has pos[0]=1 AND neg[0]=1, which must be caught next.
    csv.write_text("3,1100,1000\n\n")
    with pytest.raises(AssertionError, match="concept 0"):
        assert_csv_invariants(csv, n_concepts=4)


def test_assert_csv_invariants_wrong_length(tmp_path):
    """assert_csv_invariants raises if bit-string length doesn't match n_concepts."""
    csv = tmp_path / "bad_len.csv"
    csv.write_text("1,100,000\n\n")   # length 3 but n_concepts=5 declared
    with pytest.raises(AssertionError, match="length 3 != n_concepts 5"):
        assert_csv_invariants(csv, n_concepts=5)


def test_write_binary_csv_round_trip(tmp_path):
    """_write_binary_csv output passes assert_csv_invariants and can be round-tripped."""
    from src.explain import _write_binary_csv

    n = 6
    # Simulate two images: image 0 has two AXps, image 1 has one AXp.
    # C_ord_signs: image 0 has all-positive; image 1 has all-negative.
    C_ord_signs = torch.tensor([
        [+1.0] * n,   # image 0: all concepts positive
        [-1.0] * n,   # image 1: all concepts negative
    ])
    results_list  = [
        [{0, 2}, {1}],  # image 0: AXp={0,2}, AXp={1}
        [{3}],          # image 1: AXp={3}
    ]
    instance_idxs = [0, 1]

    out = tmp_path / "round_trip.csv"
    _write_binary_csv(out, results_list, instance_idxs, n, C_ord_signs)

    blocks = assert_csv_invariants(out, n_concepts=n)

    # Block 0: two explanations
    assert len(blocks[0]) == 2
    size0, pos0, neg0 = blocks[0][0]   # {0, 2} with positive sign → pos[0]=1, pos[2]=1
    assert size0 == 2
    assert pos0[0] == 1 and pos0[2] == 1
    assert neg0 == [0] * n

    size1, pos1, neg1 = blocks[0][1]   # {1} with positive sign
    assert size1 == 1
    assert pos1[1] == 1
    assert neg1 == [0] * n

    # Block 1: one explanation, concept 3 with negative sign
    assert len(blocks[1]) == 1
    size2, pos2, neg2 = blocks[1][0]
    assert size2 == 1
    assert pos2 == [0] * n
    assert neg2[3] == 1


def test_write_binary_csv_cross_image_signs(tmp_path):
    """
    A concept can appear positive for one image and negative for another.
    This cross-image variability must NOT raise any invariant violations.
    """
    from src.explain import _write_binary_csv

    n = 3
    # Concept 0 is positive for image 0, negative for image 1.
    C_ord_signs = torch.tensor([
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, +1.0],
    ])
    results_list  = [[{0}], [{0}]]
    instance_idxs = [0, 1]

    out = tmp_path / "cross_sign.csv"
    _write_binary_csv(out, results_list, instance_idxs, n, C_ord_signs)

    blocks = assert_csv_invariants(out, n_concepts=n)

    # Block 0: concept 0 positive
    _, pos0, neg0 = blocks[0][0]
    assert pos0[0] == 1 and neg0[0] == 0

    # Block 1: same concept 0, but now negative (different image, different sign)
    _, pos1, neg1 = blocks[1][0]
    assert pos1[0] == 0 and neg1[0] == 1


# ---------------------------------------------------------------------------
# Integration test — requires all intermediate files from steps 1–4
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_SKIP_COND, reason=_SKIP_MSG)
def test_run_experiment_xpenum_rival10_cars(tmp_path):
    """
    Full integration test for step #5 (run_experiment.py).

    Runs XpEnum (only, skipping XpSatEnum and NaiveEnum) for all three
    erasers (Ortho, Splice, LEACE) on RIVAL10 behavior 2 (correctly classified
    cars, class_idx=1), with energy vocabulary ordering, at most 50 images, and a
    5-minute per-configuration timeout.

    Assertions
    ----------
    1. All 6 binary CSV files are created (3 erasers × {AXp, CXp}) in the
       expected directory with the correct names.
    2. Each time-tracking CSV is created (one per eraser).
    3. Every data line in every CSV passes assert_csv_invariants:
       - correct field count and bit-string lengths
       - size == popcount(pos_bits) + popcount(neg_bits)
       - mutual exclusion: no concept is simultaneously positive and negative
    4. At least one non-empty AXp explanation exists across all erasers
       (confirms the experiment ran successfully and found something).
    5. Informational: if multiple images produced AXp explanations, the number
       of concepts appearing with both signs across images is logged.
    """
    from src.run_experiment import main as run_experiment_main

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    original_argv = sys.argv[:]
    sys.argv = [
        "run_experiment.py",
        "--model",                    _MODEL,
        "--behavior",                 str(_BEHAVIOR),
        "--class-idx",                str(_CLASS_IDX),
        "--experiments-per-behavior", str(_MAX_IMAGES),
        "--mcs",                      str(_MCS),
        "--n-concepts",               str(_N_CONCEPTS),
        "--no-xpsatenum",
        "--no-naiveenum",
        "--timeout",                  str(_TIMEOUT),
        "--results-dir",              str(tmp_path),
        "--device",                   device,
    ]
    try:
        run_experiment_main()
    finally:
        sys.argv = original_argv

    # ── Assert file existence and naming ─────────────────────────────────────
    results_dir = tmp_path / _CM_NAME

    all_axp_blocks: list = []   # collected for cross-image sign check

    for prefix, eraser_name in _ERASER_PREFIXES:
        beh_id = f"{prefix}X{_BEH_SUFFIX}"

        for xp_type in ("A", "C"):
            csv_path = results_dir / f"binary_{beh_id}_{xp_type}.csv"
            blocks = assert_csv_invariants(csv_path, n_concepts=_N_CONCEPTS)
            if xp_type == "A":
                all_axp_blocks.extend(blocks)

        time_csv = results_dir / f"time_{beh_id}.csv"
        assert time_csv.is_file(), (
            f"Time CSV not created for eraser '{eraser_name}': {time_csv}"
        )

    # ── Assert the experiment produced at least one explanation ───────────────
    total_axp_explanations = sum(len(block) for block in all_axp_blocks)
    assert total_axp_explanations > 0, (
        "No AXp explanations were found across all three erasers. "
        "Check that the behavior data contains images that pass the sanity checks."
    )

    # ── Informational: cross-image concept-sign variability ───────────────────
    non_empty_blocks = [b for b in all_axp_blocks if b]
    if len(non_empty_blocks) >= 2:
        pos_count = [0] * _N_CONCEPTS
        neg_count = [0] * _N_CONCEPTS
        for block in non_empty_blocks:
            for _, pos, neg in block:
                for i in range(_N_CONCEPTS):
                    pos_count[i] += pos[i]
                    neg_count[i] += neg[i]
        both_signs = [i for i in range(_N_CONCEPTS)
                      if pos_count[i] > 0 and neg_count[i] > 0]
        print(
            f"\nConcepts appearing with both signs across images (AXp, all erasers): "
            f"{len(both_signs)} / {_N_CONCEPTS}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])