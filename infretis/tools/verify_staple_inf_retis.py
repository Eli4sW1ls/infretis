#!/usr/bin/env python3
"""Verification test for the inf_retis method in repex_staple.py."""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/APPTIS/infretis')

from infretis.classes.repex_staple import REPEX_state_staple

def test_probability_matrix_properties():
    """Test that the inf_retis method creates mathematically correct probability matrices."""
    
    # Create a minimal config for testing
    config = {
        'current': {'size': 5, 'cstep': 0},
        'simulation': {'seed': 42, 'shooting_moves': ['sh'] * 6, 'interfaces': [0, 1, 2, 3, 4]},
        'output': {},
        'runner': {'workers': 1}
    }
    
    # Create REPEX state instance
    state = REPEX_state_staple(config)
    
    test_cases = [
        {
            'name': 'User\'s specific matrix',
            'matrix': np.array([    [1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0], 
                                    [0, 1, 1, 0.5, 0],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 0, 0, 1]
                                ], dtype=np.float64),
            
            # np.array([
            #     [1, 0, 0, 0, 0],
            #     [0, 1, 0, 0, 0], 
            #     [0, 1, 1, 0, 0],
            #     [0, 1, 1, 1, 0],
            #     [0, 0, 0, 0, 1]
            # ], dtype=np.float64),
            'locks': np.zeros(5)
        },
        {
            'name': 'Identity matrix',
            'matrix': np.eye(4, dtype=np.float64),
            'locks': np.zeros(4)
        },
        {
            'name': 'Simple chain',
            'matrix': np.array([
                [1, 1, 0],
                [0, 1, 1], 
                [0, 0, 1]
            ], dtype=np.float64),
            'locks': np.zeros(3)
        },
        {
            'name': 'Full connectivity',
            'matrix': np.ones((3, 3), dtype=np.float64),
            'locks': np.zeros(3)
        },
        {
            'name': 'With locks',
            'matrix': np.array([
                [1, 1, 0, 0],
                [0, 1, 1, 0], 
                [0, 0, 1, 1],
                [0, 0, 0, 1]
            ], dtype=np.float64),
            'locks': np.array([0, 1, 0, 0])  # Lock second ensemble
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"Test Case {i+1}: {test_case['name']}")
        print(f"{'='*60}")
        
        matrix = test_case['matrix']
        locks = test_case['locks']
        
        # Ensure matrix has correct dtype
        if matrix.dtype != np.float64:
            matrix = matrix.astype(np.float64)
        
        print("Input matrix:")
        print(matrix)
        print(f"Matrix dtype: {matrix.dtype}")
        print(f"Locks: {locks}")
        
        try:
            # Use the staple version's inf_retis method
            result = state.inf_retis(matrix, locks)
            
            print("\nResult matrix:")
            print(result)
            
            # Verify mathematical properties
            row_sums = result.sum(axis=1)
            col_sums = result.sum(axis=0)
            
            print(f"Row sums: {row_sums}")
            print(f"Col sums: {col_sums}")
            
            # Check if it's doubly stochastic (ignoring locked rows/cols)
            unlocked_indices = np.where(locks == 0)[0]
            unlocked_result = result[np.ix_(unlocked_indices, unlocked_indices)]
            
            if len(unlocked_result) > 0:
                unlocked_row_sums = unlocked_result.sum(axis=1)
                unlocked_col_sums = unlocked_result.sum(axis=0)
                
                row_check = np.allclose(unlocked_row_sums, 1.0, rtol=1e-10)
                col_check = np.allclose(unlocked_col_sums, 1.0, rtol=1e-10)
                non_negative = np.all(result >= -1e-10)  # Allow tiny numerical errors
                
                print(f"✓ Unlocked row sums: {unlocked_row_sums}")
                print(f"✓ Unlocked col sums: {unlocked_col_sums}")
                print(f"✓ Row normalization: {'PASS' if row_check else 'FAIL'}")
                print(f"✓ Column normalization: {'PASS' if col_check else 'FAIL'}")
                print(f"✓ Non-negative entries: {'PASS' if non_negative else 'FAIL'}")
                
                # Check locked entries are zero
                locked_check = True
                for idx in range(len(locks)):
                    if locks[idx] == 1:
                        if not (np.allclose(result[idx, :], 0) and np.allclose(result[:, idx], 0)):
                            locked_check = False
                            break
                
                print(f"✓ Locked entries are zero: {'PASS' if locked_check else 'FAIL'}")
                
                # Verify swapping interpretation for binary matrices
                if np.all((matrix == 0) | (matrix == 1)):
                    print("\n🔍 Binary Matrix Analysis:")
                    print("This matrix represents possible swaps between ensembles.")
                    print("Each entry P[i,j] = probability of trajectory i going to ensemble j")
                    
                    # Check conservation: sum of probabilities should conserve trajectories
                    total_in = unlocked_col_sums.sum()
                    total_out = unlocked_row_sums.sum()
                    conservation = np.isclose(total_in, total_out, rtol=1e-10)
                    print(f"✓ Conservation (total in = total out): {'PASS' if conservation else 'FAIL'}")
                
                test_passed = row_check and col_check and non_negative and locked_check
                if np.all((matrix == 0) | (matrix == 1)):
                    test_passed = test_passed and conservation
                
                if test_passed:
                    print(f"\n🎉 {test_case['name']}: ALL CHECKS PASSED!")
                else:
                    print(f"\n❌ {test_case['name']}: SOME CHECKS FAILED!")
                    all_passed = False
            else:
                print("⚠️  All ensembles are locked - nothing to check")
            
        except Exception as e:
            print(f"\n❌ ERROR in {test_case['name']}: {e}")
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your inf_retis method creates correct probability matrices.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    print(f"{'='*60}")
    
    return all_passed

def test_swapping_interpretation():
    """Test specific swapping scenarios to verify the physical interpretation."""
    
    config = {
        'current': {'size': 3, 'cstep': 0},
        'simulation': {'seed': 42, 'shooting_moves': ['sh'] * 4, 'interfaces': [0, 1, 2]},
        'output': {},
        'runner': {'workers': 1}
    }
    
    state = REPEX_state_staple(config)
    
    print(f"\n{'='*60}")
    print("SWAPPING INTERPRETATION TEST")
    print(f"{'='*60}")
    
    # Test case: trajectory 0 can swap with ensemble 1, trajectory 1 can swap with ensemble 2
    swap_matrix = np.array([
        [1, 1, 0],  # Trajectory 0 can stay in ens 0 or move to ens 1
        [0, 1, 1],  # Trajectory 1 can stay in ens 1 or move to ens 2  
        [0, 0, 1]   # Trajectory 2 can only stay in ens 2
    ], dtype=np.float64)
    
    locks = np.zeros(3)
    
    print("Swap matrix (1 = swap possible, 0 = swap impossible):")
    print(swap_matrix)
    print(f"Matrix dtype: {swap_matrix.dtype}")
    print("\nPhysical interpretation:")
    print("- Trajectory 0: can stay in ensemble 0 OR move to ensemble 1")
    print("- Trajectory 1: can stay in ensemble 1 OR move to ensemble 2")  
    print("- Trajectory 2: can only stay in ensemble 2")
    
    try:
        result = state.inf_retis(swap_matrix, locks)
        
        print(f"\nComputed probability matrix:")
        print(result)
        
        print(f"\nProbability interpretation:")
        for i in range(len(result)):
            for j in range(len(result[0])):
                if result[i,j] > 1e-10:  # Non-zero probability
                    print(f"- P(traj {i} → ens {j}) = {result[i,j]:.4f}")
        
        # Check if probabilities make physical sense
        print(f"\nPhysical consistency checks:")
        
        # Each trajectory should have exactly one destination (row sum = 1)
        for i in range(len(result)):
            row_sum = result[i,:].sum()
            print(f"- Trajectory {i} total probability: {row_sum:.6f} {'✓' if np.isclose(row_sum, 1.0) else '❌'}")
        
        # Each ensemble should receive exactly one trajectory (col sum = 1)  
        for j in range(len(result[0])):
            col_sum = result[:,j].sum()
            print(f"- Ensemble {j} receives probability: {col_sum:.6f} {'✓' if np.isclose(col_sum, 1.0) else '❌'}")
            
        # Impossible swaps should have zero probability
        impossible_swaps = []
        for i in range(len(swap_matrix)):
            for j in range(len(swap_matrix[0])):
                if swap_matrix[i,j] == 0 and result[i,j] > 1e-10:
                    impossible_swaps.append((i,j))
        
        if impossible_swaps:
            print(f"❌ Found impossible swaps with non-zero probability: {impossible_swaps}")
        else:
            print(f"✓ All impossible swaps have zero probability")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    test_probability_matrix_properties()
    test_swapping_interpretation()
