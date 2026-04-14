"""
mhs.py — Minimal Hitting Set (MHS) utilities.

A hitting set of a collection A is a set that intersects every set in A.
A minimal hitting set is a hitting set with no redundant elements.

Three approaches are provided:
  A) Exact solver  — uses the Hitman SAT-based enumerator (optimal, but can time out)
  B) Random        — picks one element at random from each set (fast, not guaranteed minimal)
  C) Hybrid        — tries B up to `attempts` times to find a result not blocked by C;
                     falls back to A if all attempts fail

Hitman reference: https://pysathq.github.io/docs/html/api/examples/hitman.html
"""

import random

from pysat.examples.hitman import Hitman  # https://pysathq.github.io/docs/html/api/examples/hitman.html


# ---------------------------------------------------------------------------
# Approach A: exact SAT-based solver
# ---------------------------------------------------------------------------

def minimal_hitting_setA(someA, someC, last=False, longest=False):
    """
    Compute a minimal hitting set of `someA` that is not blocked by `someC`.

    Args:
        someA   : list of lists — the sets to be hit
        someC   : list of sets  — previously found hitting sets to block
        last    : if True, return the last enumerated hitting set
        longest : if True, return the longest enumerated hitting set
    Returns:
        A minimal hitting set as a list, or [] if none found.
    """
    if len(someA) == 0:
        return []

    h = Hitman()
    for s in someA:
        h.hit(s)
    for blocked in someC:
        h.block(blocked)

    try:
        if last:
            *_, result = h.enumerate()
        elif longest:
            result = max(h.enumerate(), key=len)
        else:
            result = h.get()
    except Exception:
        result = None
    finally:
        if result is None:
            result = []
        h.delete()

    return result


# ---------------------------------------------------------------------------
# Approach B: random picker
# ---------------------------------------------------------------------------

def minimal_hitting_setB(list_of_lists):
    """
    Build a hitting set by randomly picking one element from each non-empty set.
    Fast but not guaranteed to be minimal or to avoid blocks.
    """
    result = set()
    for a_list in list_of_lists:
        if len(a_list) > 0:
            result.add(random.choice(a_list))
    return list(result)


# ---------------------------------------------------------------------------
# Approach C: hybrid (random with fallback to exact)
# ---------------------------------------------------------------------------

def minimal_hitting_setC(list_of_lists, list_blocks, attempts=100):
    """
    Try to find a hitting set not in `list_blocks` using random picks (B).
    If a valid result is not found within `attempts` tries, fall back to the
    exact Hitman solver (A).

    Args:
        list_of_lists : list of lists — the sets to be hit
        list_blocks   : list of sets  — hitting sets to avoid
        attempts      : max number of random tries before falling back to A
    Returns:
        A hitting set as a list.
    """
    if len(list_blocks) == 0:
        return minimal_hitting_setB(list_of_lists)

    result = list_blocks[0]  # initialise inside the blocked set to enter the loop
    for _ in range(attempts):
        if result not in list_blocks:
            break
        result = minimal_hitting_setB(list_of_lists)
    else:
        # All random attempts were blocked — fall back to exact solver
        result = minimal_hitting_setA(list_of_lists, list_blocks)

    return result
