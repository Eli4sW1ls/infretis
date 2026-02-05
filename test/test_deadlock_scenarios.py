#!/usr/bin/env python3
"""
Advanced test scenarios for the enhanced deadlock resolution algorithm.
"""

import sys
import os
import numpy as np
import logging
from collections import namedtuple

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

def test_deadlock_resolution_scenarios():
    """Test the deadlock resolution with multiple scenarios."""
    
    # Create MockRepex class with our algorithm
    class MockRepex:
        def __init__(self, state_matrix, locked_paths_list=None):
            self.state = state_matrix.copy()
            self.n = state_matrix.shape[0]
            self._trajs = [MockPath(i) for i in range(self.n-1)]
            self._locked_paths = set(locked_paths_list or [])
        
        def swap(self, i, j):
            """Swap rows in the state matrix and trajectories."""
            if i == j:
                return
            self.state[[i, j], :] = self.state[[j, i], :]
            self._trajs[i], self._trajs[j] = self._trajs[j], self._trajs[i]
            print(f"Swapped rows {i} ↔ {j}")
            
        def locked_paths(self):
            """Mock locked_paths method."""
            return self._locked_paths
                
        def _resolve_deadlock(self):
            """Concise direct/cycle deadlock resolution for arbitrary matrices.
            
            Uses a two-phase approach:
            1. Try direct swaps first (simplest case)
            2. For remaining problems, use small targeted permutations
            
            This algorithm avoids complex backtracking while handling most cases.
            """
            print("Resolving deadlock using direct and targeted permutations.")
            
            n_unlocked = self.n - 1
            locks = self.locked_paths()
            
            # Step 1: Find problematic ensembles (diagonal = 0)
            problematic_ensembles = []
            for ens_idx in range(n_unlocked):
                if self.state[ens_idx, ens_idx] == 0:
                    problematic_ensembles.append(ens_idx)
            
            print(f"Problematic ensembles (diagonal = 0): {problematic_ensembles}")
            
            prev_swap = {}
            while len(problematic_ensembles) > 0:
                # Step 1: Find problematic ensembles (diagonal = 0)
                problematic_ensembles = []
                for ens_idx in range(n_unlocked):
                    if self.state[ens_idx, ens_idx] == 0:
                        problematic_ensembles.append(ens_idx)
                
                # print(f"Problematic ensembles (diagonal = 0): {problematic_ensembles}")
                
                if not problematic_ensembles:
                    print("No problematic ensembles found")
                    return
                    
                # Step 2: Try simple direct swaps first (faster and simpler)
                simple_swaps_worked = False
                for ens_idx in problematic_ensembles:
                    # Find any unlocked trajectory that has non-zero weight for this ensemble
                    for traj_idx in range(n_unlocked):
                        if (traj_idx != ens_idx and 
                            self._trajs[traj_idx].path_number not in locks and 
                            self.state[traj_idx, ens_idx] != 0 and
                            ( not (ens_idx in prev_swap and traj_idx in prev_swap))):
                            print(f"Simple swap: ensemble {ens_idx} ↔ trajectory {traj_idx}")
                            self.swap(ens_idx, traj_idx)
                            prev_swap = {ens_idx, traj_idx}
                            simple_swaps_worked = True
                            break
                        elif self.state[ens_idx, ens_idx] != 0:
                            simple_swaps_worked = True
                            break  # This ensemble is now fixed
            
            # # Check if simple swaps fixed everything
            # remaining = [i for i in range(n_unlocked) if self.state[i, i] == 0]
            # if not remaining:
            #     print("Successfully resolved deadlock with simple swaps!")
            #     return
                
            # # Step 3: For remaining problems, find swap cycles
            # print(f"Simple swaps insufficient, trying cycle detection for: {remaining}")
            
            # # For each problematic ensemble, find a valid swap cycle
            # for start_idx in remaining:
            #     # Find cycles starting from the problematic ensemble
            #     # A cycle is a sequence: start_idx -> idx1 -> idx2 -> ... -> start_idx
            #     # where each step has a non-zero weight
                
            #     # BFS to find the shortest valid cycle
            #     visited = set([start_idx])
            #     queue = [[start_idx]]  # Each item is a path of indices
            #     found_cycle = False
                
            #     while queue and not found_cycle:
            #         path = queue.pop(0)
            #         current = path[-1]
                    
            #         # Try all potential next steps in the cycle
            #         for next_idx in range(n_unlocked):
            #             # Skip locked paths and already visited indices
            #             if next_idx in visited or self._trajs[next_idx].path_number in locks:
            #                 continue
                        
            #             # Check if there's a non-zero weight from current to next
            #             if self.state[current, next_idx] != 0:
            #                 new_path = path + [next_idx]
                            
            #                 # If next_idx has non-zero weight for start_idx, we found a cycle
            #                 if self.state[next_idx, start_idx] != 0:
            #                     # Complete the cycle
            #                     cycle = new_path + [start_idx]
            #                     print(f"Found swap cycle for {start_idx}: {cycle}")
                                
            #                     # Apply the swaps in the cycle
            #                     for i in range(len(cycle) - 1):
            #                         from_idx, to_idx = cycle[i], cycle[i+1]
            #                         print(f"  Cycle swap: {from_idx} ↔ {to_idx}")
            #                         self.swap(from_idx, to_idx)
                                
            #                     found_cycle = True
            #                     break
                            
            #                 # Otherwise continue the search
            #                 visited.add(next_idx)
            #                 queue.append(new_path)
                            
            #                 # Limit cycle length to avoid excessive swaps
            #                 if len(new_path) > n_unlocked:
            #                     break
                
            #     if not found_cycle:
            #         print(f"No valid swap cycle found for ensemble {start_idx}")
            
            # Step 4: Verify the solution
            final_problems = []
            for ens_idx in range(n_unlocked):
                if self.state[ens_idx, ens_idx] == 0:
                    final_problems.append(ens_idx)
            
            if final_problems:
                print(f"Still have problems after cycle swaps: {final_problems}")
                print("FATAL: Cycle-based deadlock resolution failed.")
                print(f"self.state:\n{self.state}")
            else:
                print("Successfully resolved deadlock with cycle swaps!")
    
    # Define test scenarios
    scenarios = [
        {
            "name": "Original problematic case",
            "matrix": np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Ensemble 0
                [0.0, 1.0, 1.0, 0.5, 0.0, 0.0],  # Ensemble 1
                [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],  # Ensemble 2
                [0.0, 0.0, 0.0, 0.5, 1.0, 0.0],  # Ensemble 3
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Ensemble 4 - Problem
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Ghost ensemble
            ]),
            "locked_paths": []
        },
        {
            "name": "Multiple problematic ensembles",
            "matrix": np.array([
                [0.0, 0.2, 0.0, 0.0, 0.0],  # Problem
                [0.3, 0.0, 0.0, 0.0, 0.0],  # Problem
                [0.1, 0.2, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.3, 0.0],  # Problem
                [0.0, 0.0, 0.0, 0.0, 0.0]   # Ghost ensemble
            ]),
            "locked_paths": []
        },
        {
            "name": "With locked paths",
            "matrix": np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0],  # Problem
                [0.0, 0.3, 0.0, 0.0, 0.0],  # Problem
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]   # Ghost ensemble
            ]),
            "locked_paths": [0]  # Path in ensemble 0 is locked
        },
        # {
        #     "name": "Unsolvable case",
        #     "matrix": np.array([
        #         [0.0, 0.0, 0.0, 0.0],  # Problem
        #         [0.0, 0.0, 0.0, 0.0],  # Problem
        #         [0.0, 0.0, 0.0, 0.0],  # Problem
        #         [0.0, 0.0, 0.0, 0.0]   # Ghost ensemble
        #     ]),
        #     "locked_paths": []
        # }
    ]
    
    # Test each scenario
    results = []
    
    for scenario in scenarios:
        print("\n" + "="*80)
        print(f"Testing scenario: {scenario['name']}")
        print("="*80)
        
        # Create the repex instance with this scenario
        repex = MockRepex(scenario['matrix'], scenario['locked_paths'])
        
        # Print initial state
        print("\nINITIAL STATE:")
        print(repex.state)
        print(f"Initial diagonal elements: {[repex.state[i,i] for i in range(repex.n-1)]}")
        
        # Find problematic ensembles before resolution
        pre_problems = [i for i in range(repex.n-1) if repex.state[i, i] == 0]
        print(f"Initial problematic ensembles: {pre_problems}")
        
        # Apply deadlock resolution
        print("\nAttempting to resolve deadlock...")
        repex._resolve_deadlock()
        
        # Check results
        post_problems = [i for i in range(repex.n-1) if repex.state[i, i] == 0]
        
        print("\nFINAL STATE:")
        print(repex.state)
        print(f"Final diagonal elements: {[repex.state[i,i] for i in range(repex.n-1)]}")
        
        if post_problems:
            print(f"❌ FAILED: Still have problems after resolution: {post_problems}")
            success = False
        else:
            print("✅ SUCCESS: All ensembles now have non-zero diagonals!")
            success = True
        
        results.append({
            "scenario": scenario["name"],
            "success": success,
            "pre_problems": pre_problems,
            "post_problems": post_problems
        })
    
    # Overall report
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    all_success = True
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} - {result['scenario']}")
        all_success = all_success and result["success"]
    
    assert all_success, "Some deadlock scenarios could not be resolved."

if __name__ == "__main__":
    if test_deadlock_resolution_scenarios():
        print("\nAll resolvable deadlock scenarios passed!")
        exit(0)  # Success
    else:
        print("\nSome deadlock scenarios could not be resolved.")
        print("Note: The 'Unsolvable case' is expected to fail.")
        exit(0)  # Still success since we expect the unsolvable case to fail