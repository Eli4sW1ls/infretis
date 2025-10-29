#!/usr/bin/env python3
"""
Comprehensive test of the fixed deadlock resolution algorithm.
Tests multiple scenarios including the previously failing case.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MockPath:
    """Mock path object for testing."""
    def __init__(self, path_number):
        self.path_number = path_number

class MockRepex:
    """Mock REPEX with fixed deadlock resolution."""
    
    def __init__(self, state_matrix, locks=None):
        self.state = state_matrix.copy()
        self.n = state_matrix.shape[0]
        self._trajs = [MockPath(i) for i in range(self.n-1)]
        self._locks = np.zeros(self.n, dtype=int) if locks is None else locks
    
    def swap(self, i, j):
        """Swap rows in the state matrix and trajectories."""
        if i == j:
            return
        self.state[[i, j], :] = self.state[[j, i], :]
        self._trajs[i], self._trajs[j] = self._trajs[j], self._trajs[i]
    
    def _resolve_deadlock(self):
        """Fixed deadlock resolution."""
        n = self.n - 1
        unlocked_trajs = [i for i in range(n) if self._locks[i] == 0]
        
        # Stage 1: Simple swaps
        problems = [i for i in range(n) if self._locks[i] == 0 and self.state[i, i] == 0]
        
        for ens_idx in problems:
            for traj_idx in unlocked_trajs:
                if self.state[traj_idx, ens_idx] != 0 and self.state[ens_idx, traj_idx] != 0:
                    self.swap(ens_idx, traj_idx)
                    break
        
        problems_after = [i for i in range(n) if self._locks[i] == 0 and self.state[i, i] == 0]
        if not problems_after:
            return
        
        # Stage 2: Backtracking
        def find_valid_permutation(traj_pos, target_ens, used_ens):
            """Find first valid permutation where state[traj, target_ens[traj]] != 0."""
            if traj_pos == n:
                for traj in range(n):
                    if self._locks[traj] == 0 and self.state[traj, target_ens[traj]] == 0:
                        return None
                return target_ens[:]
            
            if self._locks[traj_pos] > 0:
                target_ens[traj_pos] = traj_pos
                return find_valid_permutation(traj_pos + 1, target_ens, used_ens)
            
            for ens_idx in unlocked_trajs:
                if not used_ens[ens_idx] and self.state[traj_pos, ens_idx] != 0:
                    target_ens[traj_pos] = ens_idx
                    used_ens[ens_idx] = True
                    result = find_valid_permutation(traj_pos + 1, target_ens, used_ens)
                    if result is not None:
                        return result
                    used_ens[ens_idx] = False
            
            return None
        
        target_ens = [-1] * n
        used_ensembles = [False] * n
        
        for locked_idx in range(n):
            if self._locks[locked_idx] > 0:
                used_ensembles[locked_idx] = True
                target_ens[locked_idx] = locked_idx
        
        target_ens = find_valid_permutation(0, target_ens, used_ensembles)
        
        if target_ens is None:
            raise RuntimeError("Deadlock cannot be resolved")
        
        # Stage 3: Apply permutation
        self._apply_permutation_via_swaps(target_ens)
    
    def _apply_permutation_via_swaps(self, target_ens):
        """Apply permutation via swaps."""
        n = len(target_ens)
        
        inv_perm = [-1] * n
        for traj in range(n):
            ens = target_ens[traj]
            inv_perm[ens] = traj
        
        visited = [False] * n
        
        for start in range(n):
            if visited[start] or self._locks[start] > 0:
                continue
            
            cycle = []
            current = start
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = inv_perm[current]
            
            if len(cycle) > 1:
                for i in range(len(cycle) - 1):
                    self.swap(cycle[i], cycle[i + 1])

def test_cases():
    """Test multiple deadlock scenarios."""
    
    test_scenarios = [
        {
            "name": "Previously failing case",
            "state": np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ])
        },
        {
            "name": "Simple swap case",
            "state": np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ])
        },
        {
            "name": "Multiple problems",
            "state": np.array([
                [0.0, 0.5, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0]
            ])
        },
        {
            "name": "Complex permutation",
            "state": np.array([
                [0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ])
        }
    ]
    
    print("="*80)
    print("COMPREHENSIVE DEADLOCK RESOLUTION TESTS")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for scenario in test_scenarios:
        print(f"\n{'='*80}")
        print(f"Test: {scenario['name']}")
        print(f"{'='*80}")
        
        state = scenario['state']
        n = state.shape[0] - 1
        
        print(f"Initial diagonal: {[state[i,i] for i in range(n)]}")
        problems_before = [i for i in range(n) if state[i, i] == 0]
        print(f"Problems: {problems_before}")
        
        try:
            repex = MockRepex(state)
            repex._resolve_deadlock()
            
            final_diag = [repex.state[i,i] for i in range(n)]
            print(f"Final diagonal: {final_diag}")
            
            if all(d > 0 for d in final_diag):
                print("✅ PASS")
                passed += 1
            else:
                print("❌ FAIL - Some diagonals still zero")
                failed += 1
        except Exception as e:
            print(f"❌ FAIL - Exception: {e}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_scenarios)} tests")
    print(f"{'='*80}")
    
    return failed == 0

if __name__ == "__main__":
    success = test_cases()
    sys.exit(0 if success else 1)
