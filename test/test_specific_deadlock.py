#!/usr/bin/env python3
"""
Investigate the specific deadlock case that fails.
"""

import sys
import os
import numpy as np
import logging

# Set up logging to see the progress
logging.basicConfig(level=logging.INFO, 
                   format='%(levelname)s: %(message)s')

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock Path class just needs path_number attribute
class MockPath:
    """Mock path object for testing."""
    def __init__(self, path_number):
        self.path_number = path_number

def analyze_deadlock():
    """Analyze the specific failing deadlock case."""
    
    problematic_state = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Ensemble 0 - OK
        [0.0, 1.0, 0.5, 0.0, 0.0, 0.0],  # Ensemble 1 - OK
        [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],  # Ensemble 2 - OK
        [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],  # Ensemble 3 - OK
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Ensemble 4 - PROBLEM (diagonal = 0)
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Ghost ensemble
    ])
    
    print("="*80)
    print("ANALYZING SPECIFIC DEADLOCK CASE")
    print("="*80)
    print("\nInitial state matrix:")
    print(problematic_state)
    print("\nDiagonal values:", [problematic_state[i,i] for i in range(5)])
    print("\nProblematic ensemble: 4 (diagonal = 0)")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Analyze what trajectories can go where
    print("\nFor each ensemble, which trajectories have non-zero weights?")
    for ens in range(5):
        available = [i for i in range(5) if problematic_state[i, ens] != 0]
        print(f"  Ensemble {ens}: trajectories {available}")
    
    print("\nFor each trajectory, which ensembles can it go to?")
    for traj in range(5):
        available = [i for i in range(5) if problematic_state[traj, i] != 0]
        print(f"  Trajectory {traj}: ensembles {available}")
    
    # The key insight: let's trace through what the algorithm should do
    print("\n" + "="*80)
    print("STEP-BY-STEP RESOLUTION")
    print("="*80)
    
    print("\nStep 1: Identify problem - Ensemble 4 has diagonal = 0")
    print("  Current assignment: trajectory i is in ensemble i")
    print("  Problem: trajectory 4 in ensemble 4, but state[4,4] = 0")
    
    print("\nStep 2: Simple swap attempt")
    print("  Looking for trajectory with non-zero weight for ensemble 4...")
    print("  Candidates: trajectories where state[traj, 4] != 0")
    for traj in range(5):
        if problematic_state[traj, 4] != 0:
            print(f"    - Trajectory {traj}: state[{traj}, 4] = {problematic_state[traj, 4]}")
    
    print("\n  No trajectories have non-zero weight for ensemble 4!")
    print("  Simple swap will fail.")
    
    print("\nStep 3: Backtracking search")
    print("  Need to find a valid permutation where all diagonals are non-zero")
    print("  Let's manually find valid assignments:")
    
    # Try to find a valid assignment manually
    print("\n  Current: [0, 1, 2, 3, 4] (traj i in ens i)")
    print("  Diagonal: [1.0, 1.0, 0.5, 0.5, 0.0] ❌")
    
    print("\n  Try swapping 4 with 1:")
    print("  Assignment: [0, 4, 2, 3, 1]")
    test1 = [problematic_state[i, [0, 4, 2, 3, 1][i]] for i in range(5)]
    print(f"  Diagonal: {test1}", "✅" if all(x > 0 for x in test1) else "❌")
    
    print("\n  Try swapping 4 with 3:")
    print("  Assignment: [0, 1, 2, 4, 3]")
    test2 = [problematic_state[i, [0, 1, 2, 4, 3][i]] for i in range(5)]
    print(f"  Diagonal: {test2}", "✅" if all(x > 0 for x in test2) else "❌")
    
    # Let's systematically check all possible permutations
    print("\n" + "="*80)
    print("SYSTEMATIC SEARCH FOR VALID PERMUTATION")
    print("="*80)
    
    from itertools import permutations
    
    count = 0
    valid_perms = []
    
    for perm in permutations(range(5)):
        count += 1
        diagonal = [problematic_state[i, perm[i]] for i in range(5)]
        if all(d > 0 for d in diagonal):
            valid_perms.append((perm, diagonal))
            if len(valid_perms) <= 5:  # Show first 5
                print(f"  Valid permutation {len(valid_perms)}: {perm}")
                print(f"    Diagonal: {diagonal}")
    
    print(f"\nFound {len(valid_perms)} valid permutations out of {count} total")
    
    if valid_perms:
        print("\n✅ Valid permutations exist!")
        print(f"First valid permutation: {valid_perms[0][0]}")
        print(f"Diagonal values: {valid_perms[0][1]}")
        
        # Show how to apply this permutation
        first_perm = valid_perms[0][0]
        print(f"\nTo apply permutation {first_perm}:")
        print("  Need to move:")
        for i in range(5):
            if first_perm[i] != i:
                print(f"    Trajectory {first_perm[i]} → Ensemble {i}")
    else:
        print("\n❌ No valid permutations exist!")
        print("This deadlock is mathematically unsolvable.")
    
    # Now let's trace through what the backtracking algorithm would do
    print("\n" + "="*80)
    print("BACKTRACKING ALGORITHM TRACE")
    print("="*80)
    
    def find_valid_permutation(pos, assignment, used, depth=0):
        """Find first valid permutation where state[i, assignment[i]] != 0 for all i."""
        indent = "  " * depth
        
        if pos == 5:
            # Check if all ensembles have non-zero diagonal
            for i in range(5):
                if problematic_state[i, assignment[i]] == 0:
                    return None
            return assignment[:]
        
        print(f"{indent}Position {pos}: trying trajectories...")
        
        # Try all trajectories for this ensemble
        for traj_idx in range(5):
            if not used[traj_idx] and problematic_state[traj_idx, pos] != 0:
                print(f"{indent}  Try trajectory {traj_idx} (state[{traj_idx},{pos}]={problematic_state[traj_idx, pos]})")
                assignment[pos] = traj_idx
                used[traj_idx] = True
                result = find_valid_permutation(pos + 1, assignment, used, depth + 1)
                if result is not None:
                    print(f"{indent}  ✅ Success!")
                    return result
                print(f"{indent}  ❌ Backtrack")
                used[traj_idx] = False
        
        print(f"{indent}No valid trajectory for position {pos}")
        return None
    
    target_assignment = [-1] * 5
    used_trajs = [False] * 5
    
    result = find_valid_permutation(0, target_assignment, used_trajs)
    
    if result:
        print(f"\n✅ Backtracking found: {result}")
        print(f"Diagonal: {[problematic_state[i, result[i]] for i in range(5)]}")
    else:
        print("\n❌ Backtracking failed to find solution")

if __name__ == "__main__":
    analyze_deadlock()
