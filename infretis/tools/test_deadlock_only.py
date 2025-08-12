#!/usr/bin/env python3
"""Simple test to verify deadlock resolution without probability calculation."""

import numpy as np
import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/APPTIS/infretis')

from infretis.classes.repex_staple import REPEX_state_staple

# Configure logger to see the output from the method
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

class DummyTraj:
    """A simple dummy trajectory class for testing."""
    def __init__(self, path_number):
        self.path_number = path_number

    def __repr__(self):
        return f"Traj(pn={self.path_number})"

def test_deadlock_resolution_only():
    """Tests just the deadlock resolution without probability calculation."""
    print("="*60)
    print("Testing deadlock resolution only (no probability calculation)")
    print("="*60)

    # Setup the REPEX_state_staple instance
    config = {
        'current': {'size': 5, 'cstep': 0},
        'simulation': {
            'seed': 42, 
            'shooting_moves': ['sh'] * 6, 
            'interfaces': [0, 1, 2, 3, 4, 5]
        },
        'output': {},
        'runner': {'workers': 1}
    }
    
    state = REPEX_state_staple(config, minus=True)
    state.toinitiate = -1

    # Setup the deadlock state
    deadlock_state = np.array([
       [1. , 0. , 0. , 0. , 0. , 0. ],
       [0. , 0. , 0. , 0.5, 0.5, 0. ],
       [0. , 1. , 1. , 0. , 0. , 0. ],
       [0. , 1. , 1. , 1. , 1. , 0. ],
       [0. , 1. , 1. , 1. , 1. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. ]
    ])
    state.state = deadlock_state
    
    # Create dummy trajectories
    state._trajs = [DummyTraj(i) for i in range(state.n)]
    state._locks = np.zeros(state.n)

    print("Initial state matrix (deadlocked):")
    print(state.state)
    initial_diagonal = np.diag(state.state)
    print(f"Initial diagonal: {initial_diagonal}")
    
    # Verify deadlock exists
    deadlock_exists = initial_diagonal[1] == 0
    print(f"Deadlock exists: {deadlock_exists}")

    # Test the sorting WITHOUT calling prob (which triggers inf_retis)
    print("\nCalling sort_trajstate (deadlock resolution only)...")
    
    # Temporarily override prob to avoid inf_retis call
    def dummy_prob():
        print("Skipping probability calculation for test")
        state._last_prob = np.eye(state.n)  # Dummy matrix
        return state._last_prob
    
    # Save original method and replace
    original_prob = state.__class__.prob
    state.__class__.prob = property(lambda self: dummy_prob())
    
    try:
        state.sort_trajstate()
        print("-" * 30)

        print("Final state matrix:")
        print(state.state)
        final_diagonal = np.diag(state.state[:-1, :-1])
        print(f"Final diagonal (of n-1 x n-1 submatrix): {final_diagonal}")

        # Check if deadlock is resolved
        deadlock_resolved = np.all(final_diagonal != 0)
        
        if deadlock_resolved:
            print("\n✅ SUCCESS: Deadlock resolved using swap() functionality!")
            print("All diagonal elements are non-zero.")
        else:
            print(f"\n❌ FAILURE: Deadlock not resolved. Zero found at diagonal positions: {np.where(final_diagonal == 0)[0]}")

        # Verify trajectories were swapped properly
        print(f"\nTrajectory arrangement after sorting:")
        for i, traj in enumerate(state._trajs[:-1]):
            print(f"  Ensemble {i}: {traj}")
            
    finally:
        # Restore original prob method
        state.__class__.prob = original_prob

    print("="*60)
    return deadlock_resolved

if __name__ == "__main__":
    success = test_deadlock_resolution_only()
    if success:
        print("🎉 Test PASSED: The enhanced sort_trajstate successfully resolves deadlocks!")
    else:
        print("❌ Test FAILED: Deadlock resolution did not work.")
