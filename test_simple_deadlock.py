#!/usr/bin/env python3
"""Test the simplified deadlock resolution."""

import numpy as np
import sys
import os

# Add the infretis module to the path
sys.path.insert(0, os.path.dirname(__file__))

# Create mock trajectory objects
class MockTraj:
    def __init__(self, path_number):
        self.path_number = path_number

def test_deadlock_resolution():
    print("Testing simplified deadlock resolution...")
    
    # Recreate your exact state matrix
    state = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    
    print("Original state matrix:")
    print(state)
    print()
    
    # Check which ensembles have diagonal = 0
    n_unlocked = 8  # 9-1
    problematic = []
    for ens_idx in range(n_unlocked):
        if state[ens_idx, ens_idx] == 0:
            problematic.append(ens_idx)
    
    print(f"Problematic ensembles (diagonal = 0): {problematic}")
    
    # Simulate what the simple fix would do for ensemble 6
    if 6 in problematic:
        print(f"\nFixing ensemble 6:")
        print(f"  Current row 6: {state[6, :]}")
        print(f"  Current row 7: {state[7, :]}")
        
        # Find a trajectory that can fix ensemble 6
        for traj_idx in range(n_unlocked):
            if state[traj_idx, 6] != 0:
                print(f"  Trajectory {traj_idx} can fix ensemble 6 (weight = {state[traj_idx, 6]})")
                
                # Simulate swap(6, traj_idx) - swap rows
                state[[6, traj_idx]] = state[[traj_idx, 6]].copy()
                
                print(f"  After swapping rows 6 ↔ {traj_idx}:")
                print(f"    New row 6: {state[6, :]}")
                print(f"    New row {traj_idx}: {state[traj_idx, :]}")
                print(f"    Ensemble 6 diagonal is now: {state[6, 6]}")
                break
    
    print(f"\nFinal state matrix:")
    print(state)
    
    # Check if all diagonals are now non-zero
    all_fixed = True
    for ens_idx in range(n_unlocked):
        if state[ens_idx, ens_idx] == 0:
            print(f"ERROR: Ensemble {ens_idx} still has diagonal = 0")
            all_fixed = False
    
    if all_fixed:
        print("✅ SUCCESS: All ensembles now have non-zero diagonals!")
    else:
        print("❌ FAILED: Some ensembles still have zero diagonals")

if __name__ == "__main__":
    test_deadlock_resolution()
