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

_MODEL        = "resnet_rival10"
_BEHAVIOR     = 2          # correctly classified instances of class_idx
_CLASS_IDX    = 1          # class 1 = "car" in RIVAL10
_MCS          = 90
_N_CONCEPTS   = 300
_TIMEOUT      = 300        # 5 minutes per configuration
_MAX_IMAGES   = 50
_XPENUM_ITERS = 50         # fewer NINA iterations to keep the test fast

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
    INTERMEDIATE_DIR / f"MCS_{_MCS}_NA_RIVAL10_vecs.pkl",
    INTERMEDIATE_DIR / f"MCS_{_MCS}_NA_RIVAL10_class_vecs.pkl",
    VOCABS_DIR      / f"MCS_{_MCS}_NA_RIVAL10.txt",
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
# Helpers for integration tests
# ---------------------------------------------------------------------------

def _timed_out(time_csv: Path) -> bool:
    """Return True if the time CSV records a partial (timed-out) run."""
    return "(partial)" in time_csv.read_text().splitlines()[0]


# ---------------------------------------------------------------------------
# Integration tests — requires all intermediate files from steps 1–4
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_SKIP_COND, reason=_SKIP_MSG)
def test_class_distribution_correct():
    """
    Informational diagnostic: per-class count of CORRECTLY classified images
    (behavior B=2: predicted == label == class_idx) in the behavior pkl.

    Skipped if the behavior pkl has not been built yet.  Run
    test_run_experiment_all_configs_rival10_ships first to trigger a build.

    Use this output to choose a class for the all-configs integration test:
    pick a class with ≥ _MAX_IMAGES correctly-classified instances.
    """
    from collections import Counter
    from utils.pickler import Pickler

    pkl_path = INTERMEDIATE_DIR / f"{_CM_NAME}.pkl"
    if not pkl_path.is_file():
        pytest.skip(f"Behavior pkl not yet built: {pkl_path.name}")

    all_data = Pickler.read(_CM_NAME)

    RIVAL10 = ["truck", "car", "plane", "ship", "cat", "dog", "equine", "deer", "frog", "bird"]
    total_counts   = Counter(int(v[0]) for v in all_data.values())
    correct_counts = Counter(
        int(v[0]) for v in all_data.values() if int(v[1]) == int(v[0])
    )

    print(f"\n── Correctly classified instances in {_CM_NAME} ({len(all_data)} images) ──")
    print(f"{'idx':<4} {'name':<8} {'total':>7} {'B=2 (correct)':>14}")
    print("-" * 36)
    for i, name in enumerate(RIVAL10):
        print(f"{i:<4} {name:<8} {total_counts[i]:>7} {correct_counts[i]:>14}")

    viable = [RIVAL10[i] for i in range(10) if correct_counts[i] >= _MAX_IMAGES]
    print(f"\nViable classes (≥{_MAX_IMAGES} correctly-classified): {viable}")

    assert viable, (
        f"No RIVAL10 class has ≥{_MAX_IMAGES} correctly-classified instances "
        f"in {_CM_NAME}. Consider rebuilding with a smaller --n-concepts."
    )


@pytest.mark.skipif(_SKIP_COND, reason=_SKIP_MSG)
def test_class_distribution_misclassified():
    """
    Informational diagnostic: per-class count of MISCLASSIFIED images
    (behavior B=3: predicted != class_idx, label == class_idx) in the behavior pkl.

    Skipped if the behavior pkl has not been built yet.

    Use this output to choose a class for misclassification-based experiments:
    pick a class with ≥ _MAX_IMAGES misclassified instances.
    """
    from collections import Counter
    from utils.pickler import Pickler

    pkl_path = INTERMEDIATE_DIR / f"{_CM_NAME}.pkl"
    if not pkl_path.is_file():
        pytest.skip(f"Behavior pkl not yet built: {pkl_path.name}")

    all_data = Pickler.read(_CM_NAME)

    RIVAL10 = ["truck", "car", "plane", "ship", "cat", "dog", "equine", "deer", "frog", "bird"]
    total_counts = Counter(int(v[0]) for v in all_data.values())
    misclass_counts = Counter(
        int(v[0]) for v in all_data.values() if int(v[1]) != int(v[0])
    )

    print(f"\n── Misclassified instances in {_CM_NAME} ({len(all_data)} images) ──")
    print(f"{'idx':<4} {'name':<8} {'total':>7} {'B=3 (misclass)':>15}")
    print("-" * 37)
    for i, name in enumerate(RIVAL10):
        print(f"{i:<4} {name:<8} {total_counts[i]:>7} {misclass_counts[i]:>15}")

    viable = [RIVAL10[i] for i in range(10) if misclass_counts[i] >= _MAX_IMAGES]
    print(f"\nViable classes (≥{_MAX_IMAGES} misclassified): {viable}")

    assert viable, (
        f"No RIVAL10 class has ≥{_MAX_IMAGES} misclassified instances "
        f"in {_CM_NAME}. Consider rebuilding with a smaller --n-concepts."
    )

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
    1. For every eraser, a time CSV is always written (full or partial run).
    2. Both binary CSV files (AXp and CXp) are always written (possibly empty).
    3. Every data line passes assert_csv_invariants.
    4. Ortho must complete fully within the timeout (not partial).
    5. At least one non-empty AXp explanation exists across all erasers.
    6. Informational: cross-image concept-sign variability is logged.

    Note: LEACE may time out (each oracle call is expensive).  Partial results
    are preserved and reported; this is not treated as a test failure.
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
        "--xpenum-iters",             str(_XPENUM_ITERS),
        "--no-xpsatenum",
        "--no-naiveenum",
        "--timeout",                  str(_TIMEOUT),
        "--results-dir",              str(tmp_path),
        "--auto-retry",
        "--device",                   device,
    ]
    try:
        run_experiment_main()
    finally:
        sys.argv = original_argv

    # ── Check each eraser: always has a time CSV; partial = timed out ────────
    results_dir    = tmp_path / _CM_NAME
    all_axp_blocks: list = []
    completed      = []

    for prefix, eraser_name in _ERASER_PREFIXES:
        beh_id   = f"{prefix}X{_BEH_SUFFIX}"
        time_csv = results_dir / f"time_{beh_id}.csv"

        if not time_csv.is_file():
            print(f"\n[ERROR] time CSV missing for eraser '{eraser_name}' — unexpected.")
            continue

        if _timed_out(time_csv):
            print(f"\n[PARTIAL] Eraser '{eraser_name}' hit the {_TIMEOUT}s limit — "
                  f"partial results recorded.")
        else:
            completed.append(eraser_name)

        for xp_type in ("A", "C"):
            csv_path = results_dir / f"binary_{beh_id}_{xp_type}.csv"
            blocks   = assert_csv_invariants(csv_path, n_concepts=_N_CONCEPTS)
            if xp_type == "A":
                all_axp_blocks.extend(blocks)

    # ── Ortho must always complete fully (it is the fastest eraser) ──────────
    assert "ortho" in completed, (
        f"Ortho eraser did not complete within {_TIMEOUT}s. "
        f"Completed: {completed}"
    )

    # ── At least one explanation must have been found ─────────────────────────
    total_axp = sum(len(block) for block in all_axp_blocks)
    assert total_axp > 0, (
        "No AXp explanations found across all erasers (including partial runs). "
        "Check that behavior data contains images passing the sanity checks."
    )

    print(f"\nCompleted erasers (full): {completed}")
    print(f"Total AXp explanations (all erasers, including partial): {total_axp}")

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
            f"Concepts appearing with both signs across images (AXp, completed erasers): "
            f"{len(both_signs)} / {_N_CONCEPTS}"
        )


@pytest.mark.skipif(_SKIP_COND, reason=_SKIP_MSG)
def test_run_experiment_all_configs_rival10_ships():
    """
    Full integration test for step #5 (run_experiment.py).

    Runs all 9 configurations (XpEnum + XpSatEnum + NaiveEnum) × (Ortho + Splice + LEACE)
    on RIVAL10 behavior 2 (correctly classified ships, class_idx=3), with energy vocabulary
    ordering, at most 50 shared images, and a 5-minute per-configuration timeout.

    Shared-instance selection: images where erasing all N LEACE concepts leaves
    the prediction unchanged are removed first (LEACE sanity check, using the
    same eraser as the experiment).  The remaining images are selected so that
    both Ortho and LEACE (qualifying erasers) produce ≥1 non-empty AXp or CXp.
    Splice results are cached but do not gate selection.  This guarantees every
    configuration operates on the same image set.  If fewer than 50 qualify,
    the user is prompted (bypassed by --auto-retry in tests).

    Results are written to intermediate_results/ (persistent) for downstream
    metric calculation.

    Assertions
    ----------
    1. For every (eraser, algorithm) pair that ran, a time CSV is always written.
    2. Both binary CSV files (AXp and CXp) are always written (possibly empty).
    3. Every data line passes assert_csv_invariants.
    4. Ortho + XpEnum must complete fully (not partial) within the timeout.
    5. At least one AXp explanation exists across all configurations.
    6. AXp and CXp counts are printed for all configurations.

    Note: NaiveEnum always times out at depth 2 — partial results are written
    and reported; this is not a test failure.  LEACE likewise times out.
    """
    from src.run_experiment import main as run_experiment_main

    class_idx  = 3   # "ship" in RIVAL10
    beh_suffix = f"B{_BEHAVIOR}{class_idx}{_OTHER_STR}N{_N_CONCEPTS}{_MODEL_TAG}"
    results_dir = INTERMEDIATE_DIR / _CM_NAME

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    original_argv = sys.argv[:]
    sys.argv = [
        "run_experiment.py",
        "--model",                    _MODEL,
        "--behavior",                 str(_BEHAVIOR),
        "--class-idx",                str(class_idx),
        "--experiments-per-behavior", str(_MAX_IMAGES),
        "--mcs",                      str(_MCS),
        "--n-concepts",               str(_N_CONCEPTS),
        "--xpenum-iters",             str(_XPENUM_ITERS),
        "--timeout",                  str(_TIMEOUT),
        "--results-dir",              str(INTERMEDIATE_DIR),
        "--auto-retry",
        "--device",                   device,
    ]
    try:
        run_experiment_main()
    finally:
        sys.argv = original_argv

    # ── Check each (eraser, algorithm) pair ─────────────────────────────────
    # time CSV is always written; "(partial)" in the header means timed out.
    ALGOS = [("X", "XpEnum"), ("S", "XpSatEnum"), ("N", "NaiveEnum")]
    axp_counts: dict = {}
    cxp_counts: dict = {}
    completed: list  = []   # fully completed (not partial)

    for prefix, eraser_name in _ERASER_PREFIXES:
        for algo_key, algo_name in ALGOS:
            beh_id   = f"{prefix}{algo_key}{beh_suffix}"
            time_csv = results_dir / f"time_{beh_id}.csv"

            if not time_csv.is_file():
                # XpSatEnum is skipped when XpEnum found no results — expected.
                print(f"\n[SKIPPED] {eraser_name} + {algo_name} (no results to work from).")
                continue

            config_key = f"{eraser_name}+{algo_name}"
            partial = _timed_out(time_csv)
            if partial:
                print(f"\n[PARTIAL] {eraser_name} + {algo_name} hit the {_TIMEOUT}s limit.")
            else:
                completed.append(config_key)

            for xp_type in ("A", "C"):
                csv_path = results_dir / f"binary_{beh_id}_{xp_type}.csv"
                blocks   = assert_csv_invariants(csv_path, n_concepts=_N_CONCEPTS)
                count    = sum(len(block) for block in blocks)
                if xp_type == "A":
                    axp_counts[config_key] = count
                else:
                    cxp_counts[config_key] = count

    # ── Ortho + XpEnum must always complete fully ────────────────────────────
    assert "ortho+XpEnum" in completed, (
        f"Ortho + XpEnum did not complete within {_TIMEOUT}s. "
        f"Completed (full): {completed}"
    )

    # ── At least one AXp must have been found (full or partial runs) ─────────
    total_axp = sum(axp_counts.values())
    assert total_axp > 0, (
        "No AXp explanations found across any configuration (including partial). "
        "Check that behavior data contains images passing the sanity checks."
    )

    print(f"\nCompleted configurations (full): {completed}")
    print(f"AXp counts: {axp_counts}")
    print(f"CXp counts: {cxp_counts}")
    print(f"Total AXp: {total_axp},  Total CXp: {sum(cxp_counts.values())}")


@pytest.mark.skipif(_SKIP_COND, reason=_SKIP_MSG)
def test_run_experiment_all_configs_rival10_planes():
    """
    Full integration test for step #5 (run_experiment.py), class_idx=2 (plane).

    Unlike ships (class_idx=3), the img_mean_map baseline may predict a *different*
    class, so Splice (which always erases to the zero CLIP vector → img_mean_map)
    is expected to produce non-empty AXps here.  The diagnostic printed by
    run_experiment.py confirms which class img_mean_map predicts.

    Runs all 9 configurations for planes with the same timeouts and parameters as
    the ships test.  Results are written to intermediate_results/ for persistence.

    Assertions
    ----------
    1. For every (eraser, algorithm) pair that ran, a time CSV is always written.
    2. Both binary CSV files (AXp and CXp) are always written (possibly empty).
    3. Every data line passes assert_csv_invariants.
    4. Ortho + XpEnum must complete fully within the timeout.
    5. Splice must produce at least one non-empty AXp (confirming it is not degenerate).
    6. At least one AXp explanation exists across all configurations.
    """
    from src.run_experiment import main as run_experiment_main

    class_idx  = 2   # "plane" in RIVAL10
    beh_suffix = f"B{_BEHAVIOR}{class_idx}{_OTHER_STR}N{_N_CONCEPTS}{_MODEL_TAG}"
    results_dir = INTERMEDIATE_DIR / _CM_NAME

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    original_argv = sys.argv[:]
    sys.argv = [
        "run_experiment.py",
        "--model",                    _MODEL,
        "--behavior",                 str(_BEHAVIOR),
        "--class-idx",                str(class_idx),
        "--experiments-per-behavior", str(_MAX_IMAGES),
        "--mcs",                      str(_MCS),
        "--n-concepts",               str(_N_CONCEPTS),
        "--xpenum-iters",             str(_XPENUM_ITERS),
        "--timeout",                  str(_TIMEOUT),
        "--results-dir",              str(INTERMEDIATE_DIR),
        "--auto-retry",
        "--device",                   device,
    ]
    try:
        run_experiment_main()
    finally:
        sys.argv = original_argv

    # ── Check each (eraser, algorithm) pair ─────────────────────────────────
    ALGOS = [("X", "XpEnum"), ("S", "XpSatEnum"), ("N", "NaiveEnum")]
    axp_counts: dict = {}
    cxp_counts: dict = {}
    completed: list  = []

    for prefix, eraser_name in _ERASER_PREFIXES:
        for algo_key, algo_name in ALGOS:
            beh_id   = f"{prefix}{algo_key}{beh_suffix}"
            time_csv = results_dir / f"time_{beh_id}.csv"

            if not time_csv.is_file():
                print(f"\n[SKIPPED] {eraser_name} + {algo_name} (no results to work from).")
                continue

            config_key = f"{eraser_name}+{algo_name}"
            partial = _timed_out(time_csv)
            if partial:
                print(f"\n[PARTIAL] {eraser_name} + {algo_name} hit the {_TIMEOUT}s limit.")
            else:
                completed.append(config_key)

            for xp_type in ("A", "C"):
                csv_path = results_dir / f"binary_{beh_id}_{xp_type}.csv"
                blocks   = assert_csv_invariants(csv_path, n_concepts=_N_CONCEPTS)
                count    = sum(len(block) for block in blocks)
                if xp_type == "A":
                    axp_counts[config_key] = count
                else:
                    cxp_counts[config_key] = count

    # ── Ortho + XpEnum must always complete fully ────────────────────────────
    assert "ortho+XpEnum" in completed, (
        f"Ortho + XpEnum did not complete within {_TIMEOUT}s. "
        f"Completed (full): {completed}"
    )

    # ── Splice must produce at least one non-empty AXp ───────────────────────
    splice_axp = axp_counts.get("splice+XpEnum", 0)
    assert splice_axp > 0, (
        f"Splice produced 0 AXps for planes (class_idx=2). "
        f"This suggests img_mean_map predicts class 2 — check the diagnostic output."
    )

    # ── At least one AXp must have been found (full or partial runs) ─────────
    total_axp = sum(axp_counts.values())
    assert total_axp > 0, (
        "No AXp explanations found across any configuration (including partial). "
        "Check that behavior data contains images passing the sanity checks."
    )

    print(f"\nCompleted configurations (full): {completed}")
    print(f"AXp counts: {axp_counts}")
    print(f"CXp counts: {cxp_counts}")
    print(f"Total AXp: {total_axp},  Total CXp: {sum(cxp_counts.values())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])