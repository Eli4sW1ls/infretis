"""
Detailed benchmark for contiguous permanent optimization.

This script provides comprehensive performance analysis of the
STAPLE-optimized contiguous permanent calculation.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from infretis.classes.repex_staple import REPEX_state_staple


def create_matrix_by_structure(n, structure='contiguous', seed=None):
    """Create test matrices with different structures."""
    if seed is not None:
        np.random.seed(seed)
    
    if structure == 'contiguous':
        # STAPLE-like contiguous structure with random bandwidth
        matrix = np.zeros((n, n), dtype=np.longdouble)
        for i in range(n):
            # Random bandwidth between 2 and 5 (or n if smaller)
            bandwidth = np.random.randint(2, min(6, n + 1))
            
            # Random starting position (sometimes centered, sometimes off-diagonal)
            if np.random.random() < 0.5:
                # Centered around diagonal
                start = max(0, i - bandwidth // 2)
            else:
                # Random position
                max_start = max(0, n - bandwidth)
                start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            
            end = min(n, start + bandwidth)
            for j in range(start, end):
                matrix[i, j] = np.random.uniform(0.3, 1.0)
    
    elif structure == 'contiguous_narrow':
        # Narrow band (2-3 elements per row)
        matrix = np.zeros((n, n), dtype=np.longdouble)
        for i in range(n):
            bandwidth = np.random.randint(2, min(4, n + 1))
            start = max(0, i - 1)
            end = min(n, start + bandwidth)
            for j in range(start, end):
                matrix[i, j] = np.random.uniform(0.3, 1.0)
    
    elif structure == 'contiguous_wide':
        # Wide band (4-6 elements per row)
        matrix = np.zeros((n, n), dtype=np.longdouble)
        for i in range(n):
            max_bandwidth = min(7, n + 1)
            if max_bandwidth <= 4:
                bandwidth = n  # For small matrices, use full bandwidth
            else:
                bandwidth = np.random.randint(4, max_bandwidth)
            start = max(0, i - 2)
            end = min(n, start + bandwidth)
            for j in range(start, end):
                matrix[i, j] = np.random.uniform(0.3, 1.0)
    
    elif structure == 'upper_tri':
        # Upper triangular
        matrix = np.triu(np.random.uniform(0.3, 1.0, (n, n))).astype(np.longdouble)
    
    elif structure == 'block_diag':
        # Block diagonal with 2 equal blocks
        matrix = np.zeros((n, n), dtype=np.longdouble)
        block_size = n // 2
        matrix[0:block_size, 0:block_size] = np.random.uniform(0.3, 1.0, (block_size, block_size))
        matrix[block_size:n, block_size:n] = np.random.uniform(0.3, 1.0, (n-block_size, n-block_size))
    
    elif structure == 'dense':
        # Dense matrix
        matrix = np.random.uniform(0.3, 1.0, (n, n)).astype(np.longdouble)
    
    elif structure == 'sparse':
        # Sparse matrix (30% non-zero)
        matrix = np.random.uniform(0.3, 1.0, (n, n)).astype(np.longdouble)
        mask = np.random.random((n, n)) > 0.3
        matrix[mask] = 0.0
    
    return matrix


def benchmark_permanent_methods(repex_state, n, structure, iterations=10):
    """Benchmark contiguous vs standard permanent for a given matrix size and structure."""
    matrix = create_matrix_by_structure(n, structure)
    
    # Warm up
    _ = repex_state._contiguous_permanent_prob(matrix)
    _ = repex_state.permanent_prob(matrix)
    
    # Benchmark contiguous method
    start = time.perf_counter()
    for _ in range(iterations):
        result_contiguous = repex_state._contiguous_permanent_prob(matrix)
    time_contiguous = time.perf_counter() - start
    
    # Benchmark standard method
    start = time.perf_counter()
    for _ in range(iterations):
        result_standard = repex_state.permanent_prob(matrix)
    time_standard = time.perf_counter() - start
    
    # Calculate speedup
    speedup = time_standard / time_contiguous if time_contiguous > 0 else float('inf')
    
    # Verify correctness
    max_diff = np.max(np.abs(result_contiguous - result_standard))
    
    return {
        'time_contiguous': time_contiguous / iterations,
        'time_standard': time_standard / iterations,
        'speedup': speedup,
        'max_diff': max_diff,
        'iterations': iterations
    }


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across different sizes and structures."""
    print("="*80)
    print("COMPREHENSIVE PERMANENT CALCULATION BENCHMARK")
    print("="*80)
    
    # Initialize REPEX state
    config = {
        'current': {'size': 10, 'cstep': 0},
        'simulation': {
            'seed': 42, 
            'shooting_moves': ['sh'] * 11, 
            'interfaces': list(range(11))
        },
        'output': {},
        'runner': {'workers': 1}
    }
    repex_state = REPEX_state_staple(config, minus=True)
    
    # Test configurations
    sizes = [3, 4, 5, 6, 7, 8, 9, 10]
    structures = ['contiguous', 'contiguous_narrow', 'contiguous_wide', 
                  'upper_tri', 'block_diag', 'dense', 'sparse']
    
    results = {}
    
    for structure in structures:
        print(f"\n{structure.upper()} MATRICES")
        print("-" * 80)
        print(f"{'Size':>6} {'Iterations':>10} {'Contiguous':>12} {'Standard':>12} {'Speedup':>10} {'Max Diff':>12}")
        print("-" * 80)
        
        results[structure] = {}
        
        for n in sizes:
            # Adjust iterations based on size
            if n <= 5:
                iterations = 100
            elif n <= 7:
                iterations = 20
            else:
                iterations = 5
            
            result = benchmark_permanent_methods(repex_state, n, structure, iterations)
            results[structure][n] = result
            
            print(f"{n:6d} {iterations:10d} {result['time_contiguous']:12.6f}s "
                  f"{result['time_standard']:12.6f}s {result['speedup']:10.2f}x "
                  f"{result['max_diff']:12.2e}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for structure in structures:
        speedups = [results[structure][n]['speedup'] for n in sizes]
        avg_speedup = np.mean(speedups)
        max_speedup = np.max(speedups)
        min_speedup = np.min(speedups)
        
        print(f"\n{structure.upper()}:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Maximum speedup: {max_speedup:.2f}x (size {sizes[np.argmax(speedups)]})")
        print(f"  Minimum speedup: {min_speedup:.2f}x (size {sizes[np.argmin(speedups)]})")
    
    # Best cases
    print("\n" + "="*80)
    print("BEST PERFORMANCE CASES")
    print("="*80)
    
    all_speedups = []
    for structure in structures:
        for n in sizes:
            all_speedups.append((structure, n, results[structure][n]['speedup']))
    
    all_speedups.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 speedups:")
    for i, (structure, n, speedup) in enumerate(all_speedups[:5], 1):
        print(f"  {i}. {structure:12s} {n}x{n}: {speedup:6.2f}x")
    
    return results


def analyze_scaling_behavior():
    """Analyze how performance scales with matrix size."""
    print("\n" + "="*80)
    print("SCALING ANALYSIS")
    print("="*80)
    
    config = {
        'current': {'size': 15, 'cstep': 0},
        'simulation': {
            'seed': 42, 
            'shooting_moves': ['sh'] * 16, 
            'interfaces': list(range(16))
        },
        'output': {},
        'runner': {'workers': 1}
    }
    repex_state = REPEX_state_staple(config, minus=True)
    
    sizes = range(3, 12)
    
    print("\nContiguous structure scaling:")
    print(f"{'Size':>6} {'Time (contiguous)':>18} {'Time (standard)':>18} {'Ratio':>10}")
    print("-" * 60)
    
    for n in sizes:
        iterations = max(1, 100 // (n**2))  # Adjust iterations for fairness
        
        matrix = create_matrix_by_structure(n, 'contiguous')
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = repex_state._contiguous_permanent_prob(matrix)
        time_cont = (time.perf_counter() - start) / iterations
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = repex_state.permanent_prob(matrix)
        time_std = (time.perf_counter() - start) / iterations
        
        ratio = time_std / time_cont
        
        print(f"{n:6d} {time_cont:18.6f}s {time_std:18.6f}s {ratio:10.2f}x")


def test_real_staple_matrices():
    """Test with realistic STAPLE matrices from simulations."""
    print("\n" + "="*80)
    print("REALISTIC STAPLE MATRICES")
    print("="*80)
    
    config = {
        'current': {'size': 6, 'cstep': 0},
        'simulation': {
            'seed': 42, 
            'shooting_moves': ['sh'] * 7, 
            'interfaces': list(range(7))
        },
        'output': {},
        'runner': {'workers': 1}
    }
    repex_state = REPEX_state_staple(config, minus=True)
    
    # Create realistic STAPLE-like matrices with specific patterns
    test_cases = [
        {
            'name': 'Typical STAPLE 5x5',
            'matrix': np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.5]
            ], dtype=np.longdouble)
        },
        {
            'name': 'STAPLE with overlap 6x6',
            'matrix': np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.5, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.2],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
            ], dtype=np.longdouble)
        }
    ]
    
    print(f"\n{'Matrix':30s} {'Contiguous':>12} {'Standard':>12} {'Speedup':>10}")
    print("-" * 70)
    
    for test in test_cases:
        matrix = test['matrix']
        iterations = 100
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = repex_state._contiguous_permanent_prob(matrix)
        time_cont = time.perf_counter() - start
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = repex_state.permanent_prob(matrix)
        time_std = time.perf_counter() - start
        
        speedup = time_std / time_cont
        
        print(f"{test['name']:30s} {time_cont:12.6f}s {time_std:12.6f}s {speedup:10.2f}x")


if __name__ == "__main__":
    # Run all benchmarks
    results = run_comprehensive_benchmark()
    analyze_scaling_behavior()
    test_real_staple_matrices()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
