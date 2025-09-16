#!/usr/bin/env python3
"""Test script for the matrix computation case."""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/APPTIS/infretis')

from infretis.classes.repex import REPEX_state

def test_specific_matrix():
    """Test the specific matrix case."""
    
    # Create a minimal config for testing
    config = {
        'current': {'size': 5, 'cstep': 0},
        'simulation': {'seed': 42, 'shooting_moves': ['sh'] * 6, 'interfaces': [0, 1, 2, 3, 4]},
        'output': {},
        'runner': {'workers': 1}
    }
    
    # Create REPEX state instance
    state = REPEX_state(config)
    
    # Your specific matrix
    test_matrix = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], 
        [0, 1, 1, 0.5, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0.5]
    ], dtype=np.longdouble)
    
    # No locks (offset = 0 in your case)
    test_locks = np.zeros(5)

    print("Testing matrix:")
    print(test_matrix)
    print(f"Locks: {test_locks}")
    print(f"Offset: {state._offset}")
    print()
    
    print("=" * 50)
    print("Testing original inf_retis method:")
    try:
        result_original = state._inf_retis_original(test_matrix, test_locks)
        print("SUCCESS: Original method worked!")
        print("Result matrix:")
        print(result_original)
        print(f"Row sums: {result_original.sum(axis=1)}")
        print(f"Col sums: {result_original.sum(axis=0)}")
    except Exception as e:
        print(f"FAILED: Original method failed with error: {e}")
    
    print("=" * 50)
    print("Testing permanent fallback method:")
    try:
        result_fallback = state.inf_retis_permanent_fallback(test_matrix, test_locks)
        print("SUCCESS: Permanent fallback method worked!")
        print("Result matrix:")
        print(result_fallback)
        print(f"Row sums: {result_fallback.sum(axis=1)}")
        print(f"Col sums: {result_fallback.sum(axis=0)}")
    except Exception as e:
        print(f"FAILED: Permanent fallback method failed with error: {e}")
    
    print("=" * 50)
    print("Testing enhanced inf_retis with automatic fallback:")
    try:
        result_enhanced = state.inf_retis(test_matrix, test_locks)
        print("SUCCESS: Enhanced method worked!")
        print("Result matrix:")
        print(result_enhanced)
        print(f"Row sums: {result_enhanced.sum(axis=1)}")
        print(f"Col sums: {result_enhanced.sum(axis=0)}")
    except Exception as e:
        print(f"FAILED: Enhanced method failed with error: {e}")

if __name__ == "__main__":
    test_specific_matrix()
