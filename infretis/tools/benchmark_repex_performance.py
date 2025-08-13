#!/usr/bin/env python3
"""
Performance benchmark script to compare old vs new REPEX_state_staple implementations.
Tests the performance gain from the blocking algorithm integration.
"""

import numpy as np
import time
import sys
import os
from collections import defaultdict

# Add the infretis module to the path
sys.path.insert(0, os.path.abspath('.'))

from infretis.classes.repex_staple import REPEX_state_staple


class OldRepexStaple(REPEX_state_staple):
    """Version without blocking algorithm for comparison."""
    
    def inf_retis(self, input_mat, locks):
        """Original simple permanent calculator without blocking optimization."""
        # Drop locked rows and columns
        bool_locks = locks == 1
        offset = self._offset - sum(bool_locks[: self._offset])
        
        # make insert list
        i = 0
        insert_list = []
        for lock in bool_locks:
            if lock:
                insert_list.append(i)
            else:
                i += 1

        # Drop locked rows and columns
        non_locked = input_mat[~bool_locks, :][:, ~bool_locks]

        # Check if this is a simple permutation matrix (identity-like)
        is_permutation = (
            np.all(non_locked.sum(axis=1) == 1)
            and np.all(non_locked.sum(axis=0) == 1)
            and np.all((non_locked == 0) | (non_locked == 1))
        )

        if is_permutation:
            out = non_locked.astype("longdouble")
        else:
            # Always use the fallback methods (no blocking)
            if len(non_locked) <= 12:
                out = self.permanent_prob(non_locked)
            else:
                out = self.random_prob(non_locked)

        # Validate the result
        row_sums = out.sum(axis=1)
        col_sums = out.sum(axis=0)
        
        # Basic repair if needed
        if not np.allclose(row_sums, 1) or not np.allclose(col_sums, 1):
            # Normalize rows first
            for i in range(len(out)):
                if row_sums[i] > 0:
                    out[i, :] /= row_sums[i]
                else:
                    support = np.where(non_locked[i, :] > 0)[0]
                    if len(support) > 0:
                        out[i, support] = 1.0 / len(support)

        # reinsert zeroes for the locked ensembles
        final_out_rows = np.insert(out, insert_list, 0, axis=0)
        final_out = np.insert(final_out_rows, insert_list, 0, axis=1)

        return final_out


def create_test_matrices():
    """Create various test matrices to benchmark performance."""
    test_cases = {}
    
    # 1. Small equal-weight matrix (should benefit from fast algorithm)
    size = 6
    equal_weight = np.random.choice([0, 1], size=(size, size), p=[0.7, 0.3]).astype(float)
    # Make sure each row has at least one non-zero
    for i in range(size):
        if equal_weight[i].sum() == 0:
            equal_weight[i, np.random.randint(0, size)] = 1
    test_cases["equal_weight_6x6"] = equal_weight
    
    # 2. Medium complex matrix (should benefit from blocking)
    size = 10
    complex_matrix = np.random.random((size, size))
    # Add some structure - make upper triangular with some zeros
    for i in range(size):
        for j in range(i):
            if np.random.random() < 0.6:
                complex_matrix[i, j] = 0
    test_cases["complex_10x10"] = complex_matrix
    
    # 3. Large sparse matrix (should really benefit from blocking)
    size = 15
    sparse_matrix = np.random.choice([0, 1], size=(size, size), p=[0.8, 0.2]).astype(float)
    # Add some structure
    for i in range(size):
        if sparse_matrix[i].sum() == 0:
            sparse_matrix[i, np.random.randint(0, size)] = 1
        # Add some randomness to weights
        sparse_matrix[i] *= np.random.random(size)
    test_cases["sparse_15x15"] = sparse_matrix
    
    # 4. Block-structured matrix (should greatly benefit from blocking)
    size = 12
    block_matrix = np.zeros((size, size))
    # Create 3 blocks of size 4x4
    for block in range(3):
        start = block * 4
        end = start + 4
        block_matrix[start:end, start:end] = np.random.random((4, 4))
    test_cases["block_12x12"] = block_matrix
    
    # 5. Identity-like permutation matrix (should be fast for both)
    size = 8
    perm_matrix = np.eye(size)
    # Shuffle some rows
    perm_indices = np.random.permutation(size)
    perm_matrix = perm_matrix[perm_indices]
    test_cases["permutation_8x8"] = perm_matrix
    
    return test_cases


def benchmark_implementation(repex_class, test_matrices, runs=5):
    """Benchmark a REPEX implementation."""
    # Create minimal config for REPEX initialization
    config = {
        'current': {
            'size': 8,
            'active': list(range(8)),
            'locked': []
        },
        'simulation': {
            'seed': 42,
            'ens_eff': [0] * 8
        },
        'runner': {
            'workers': 1
        },
        'output': {}
    }
    
    repex = repex_class(config, minus=True)
    repex._offset = 2  # Set offset for minus ensembles
    
    results = defaultdict(list)
    
    for name, matrix in test_matrices.items():
        print(f"\nTesting {name} ({matrix.shape[0]}x{matrix.shape[1]})...")
        
        # Create locks array (no locks for simplicity)
        locks = np.zeros(len(matrix))
        
        # Warm-up run
        try:
            _ = repex.inf_retis(matrix, locks)
        except Exception as e:
            print(f"  Warm-up failed: {e}")
            continue
        
        # Benchmark runs
        times = []
        for run in range(runs):
            start_time = time.perf_counter()
            try:
                result = repex.inf_retis(matrix, locks)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
                # Validate result
                if not (np.allclose(result.sum(axis=1), 1) and np.allclose(result.sum(axis=0), 1)):
                    print(f"  WARNING: Invalid result in run {run+1}")
                    
            except Exception as e:
                print(f"  Run {run+1} failed: {e}")
                times.append(float('inf'))
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        
        results[name] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'all_times': times
        }
        
        print(f"  Average time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"  Best time: {min_time:.4f}s")
    
    return results


def main():
    """Run the performance benchmark."""
    print("INFRETIS REPEX Performance Benchmark")
    print("="*50)
    
    # Create test matrices
    print("Creating test matrices...")
    test_matrices = create_test_matrices()
    
    print(f"Created {len(test_matrices)} test cases:")
    for name, matrix in test_matrices.items():
        non_zero_percent = (np.count_nonzero(matrix) / matrix.size) * 100
        print(f"  {name}: {matrix.shape[0]}x{matrix.shape[1]}, {non_zero_percent:.1f}% non-zero")
    
    # Benchmark old implementation
    print("\n" + "="*50)
    print("Benchmarking OLD implementation (no blocking)...")
    print("="*50)
    old_results = benchmark_implementation(OldRepexStaple, test_matrices)
    
    # Benchmark new implementation
    print("\n" + "="*50)
    print("Benchmarking NEW implementation (with blocking)...")
    print("="*50)
    new_results = benchmark_implementation(REPEX_state_staple, test_matrices)
    
    # Compare results
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    total_old_time = 0
    total_new_time = 0
    
    print(f"{'Test Case':<20} {'Old (s)':<12} {'New (s)':<12} {'Speedup':<10} {'Status'}")
    print("-" * 70)
    
    for name in test_matrices.keys():
        if name in old_results and name in new_results:
            old_time = old_results[name]['avg_time']
            new_time = new_results[name]['avg_time']
            
            if old_time != float('inf') and new_time != float('inf'):
                speedup = old_time / new_time
                total_old_time += old_time
                total_new_time += new_time
                
                if speedup > 1.1:
                    status = "FASTER ✓"
                elif speedup < 0.9:
                    status = "SLOWER ✗"
                else:
                    status = "SIMILAR ="
                    
                print(f"{name:<20} {old_time:<12.4f} {new_time:<12.4f} {speedup:<10.2f}x {status}")
            else:
                print(f"{name:<20} {'FAILED':<12} {'FAILED':<12} {'N/A':<10} {'ERROR'}")
    
    print("-" * 70)
    if total_old_time > 0 and total_new_time > 0:
        overall_speedup = total_old_time / total_new_time
        print(f"{'OVERALL':<20} {total_old_time:<12.4f} {total_new_time:<12.4f} {overall_speedup:<10.2f}x")
        
        if overall_speedup > 1.1:
            print(f"\n🎉 NEW implementation is {overall_speedup:.2f}x FASTER overall!")
        elif overall_speedup < 0.9:
            print(f"\n⚠️  NEW implementation is {1/overall_speedup:.2f}x SLOWER overall")
        else:
            print(f"\n📊 Performance is similar (within 10%)")
            
        print(f"\nTime saved per computation: {(total_old_time - total_new_time)/len(test_matrices):.4f}s average")


if __name__ == "__main__":
    main()
