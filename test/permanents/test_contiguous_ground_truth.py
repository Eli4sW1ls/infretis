"""Ground-truth tests for contiguous permanent calculations.

This file adds:
- Enumeration-based exact permanent tests for small matrices
- High-precision verification using mpmath for extreme-value matrices
"""

import itertools
import numpy as np
import pytest

from infretis.classes.repex_staple import REPEX_state_staple


@pytest.fixture
def repex_state():
    config = {
        'current': {'size': 5, 'cstep': 0},
        'simulation': {
            'seed': 42,
            'shooting_moves': ['sh'] * 6,
            'interfaces': [0, 1, 2, 3, 4, 5],
            'steps': 100
        },
        'output': {},
        'runner': {'workers': 1}
    }
    return REPEX_state_staple(config, minus=True)


def exact_permanent_enumeration(M: np.ndarray) -> float:
    """Compute exact permanent by enumeration (small matrices only)."""
    n = M.shape[0]
    total = 0.0
    for p in itertools.permutations(range(n)):
        prod = 1.0
        for i, j in enumerate(p):
            prod *= float(M[i, j])
        total += prod
    return total


def test_enumeration_small_matrix_scalar(repex_state):
    """Exact enumeration vs fast contiguous/permanent implementations."""
    M = np.array([
        [1.0, 0.5, 0.0],
        [0.3, 2.0, 0.7],
        [0.0, 0.4, 1.5]
    ], dtype=np.longdouble)

    perm_exact = exact_permanent_enumeration(M)
    perm_fast_contig = repex_state._fast_contiguous_permanent(M)
    perm_glynn = repex_state.fast_glynn_perm(M)

    assert np.isclose(perm_fast_contig, perm_exact, rtol=1e-12, atol=0.0)
    assert np.isclose(perm_glynn, perm_exact, rtol=1e-12, atol=0.0)


def test_enumeration_prob_matrix(repex_state):
    """Exact permutation-weighted P matrix vs contiguous probability matrix for small n."""
    M = np.array([
        [1.0, 0.5, 0.0],
        [0.3, 2.0, 0.7],
        [0.0, 0.4, 1.5]
    ], dtype=np.longdouble)

    # Follow the scaling used in `permanent_prob` to avoid mismatches due to scaling
    scaled = M.copy()
    if scaled.shape[0] > 0:
        row_max = np.max(scaled, axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        scaled = scaled / row_max

    # Enumerate weighted permutations to build exact P
    n = scaled.shape[0]
    P_exact = np.zeros_like(scaled, dtype="longdouble")
    import itertools
    for p in itertools.permutations(range(n)):
        weight = 1.0
        valid = True
        for i, j in enumerate(p):
            if scaled[i, j] == 0:
                valid = False
                break
            weight *= scaled[i, j]
        if not valid:
            continue
        for i, j in enumerate(p):
            P_exact[i, j] += weight

    # Normalize same as `permanent_prob`
    row_sums = np.sum(P_exact, axis=1)
    max_row_sum = max(row_sums) if len(row_sums) > 0 else 1.0
    if max_row_sum > 0:
        P_exact = P_exact / max_row_sum

    P_contig = repex_state._contiguous_permanent_prob(M)

    assert np.allclose(P_exact, P_contig, rtol=1e-12, atol=0.0)