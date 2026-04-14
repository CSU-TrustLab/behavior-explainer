"""
test_mhs.py — Basic tests for the minimal hitting set utilities.

Hitman reference: https://pysathq.github.io/docs/html/api/examples/hitman.html
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.mhs import minimal_hitting_setA, minimal_hitting_setB, minimal_hitting_setC


def test_hitman_basic():
    """Approach A: Hitman should find a minimal hitting set."""
    # Every hitting set must contain at least one of {1,2,3}, one of {1,4},
    # and one of {5,6,7}.  The smallest valid answer has 3 elements.
    sets = [[1, 2, 3], [1, 4], [5, 6, 7]]
    result = minimal_hitting_setA(sets, [])
    print("Approach A result:", result)
    assert isinstance(result, list)
    assert len(result) > 0
    for s in sets:
        assert any(elem in result for elem in s), f"Set {s} not hit by {result}"


def test_hitman_empty_input():
    """Approach A: empty input should return an empty list."""
    result = minimal_hitting_setA([], [])
    assert result == []


def test_random_basic():
    """Approach B: random picker should hit every non-empty set."""
    sets = [[1, 2, 3], [1, 4], [5, 6, 7]]
    result = minimal_hitting_setB(sets)
    print("Approach B result:", result)
    assert isinstance(result, list)
    for s in sets:
        assert any(elem in result for elem in s), f"Set {s} not hit by {result}"


def test_hybrid_no_blocks():
    """Approach C with no blocks should behave like approach B."""
    sets = [[1, 2, 3], [1, 4], [5, 6, 7]]
    result = minimal_hitting_setC(sets, [])
    print("Approach C (no blocks) result:", result)
    assert isinstance(result, list)
    for s in sets:
        assert any(elem in result for elem in s)


def test_hybrid_with_blocks():
    """Approach C should return a result not in the blocked list."""
    sets = [[1, 2], [3, 4]]
    # Block one possible result so C must find another
    blocks = [[1, 3]]
    result = minimal_hitting_setC(sets, blocks, attempts=100)
    print("Approach C (with blocks) result:", result)
    assert isinstance(result, list)
    assert result not in blocks


if __name__ == "__main__":
    test_hitman_basic()
    test_hitman_empty_input()
    test_random_basic()
    test_hybrid_no_blocks()
    test_hybrid_with_blocks()
    print("All tests passed!")
