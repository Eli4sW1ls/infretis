"""
Tests for STAPLE-specific contiguous permanent optimization.

This module tests the optimized permanent calculation that leverages
the contiguous row structure characteristic of STAPLE matrices.
"""

import numpy as np
import pytest
import time
from infretis.classes.repex_staple import REPEX_state_staple


@pytest.fixture
def repex_state():
    """Create a REPEX_state_staple instance for testing."""
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
    return REPEX_state_staple(config, minus=True)


class TestContiguousPermanent:
    """Tests for contiguous permanent optimization."""
    
    def test_upper_triangular_detection(self, repex_state):
        """Test detection of upper triangular matrices."""
        # Upper triangular matrix
        upper_tri = np.array([
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1]
        ])
        assert repex_state._is_upper_triangular(upper_tri)
        
        # Not upper triangular
        not_upper = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 1]
        ])
        assert not repex_state._is_upper_triangular(not_upper)
        
        # Diagonal matrix (also upper triangular)
        diagonal = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        assert repex_state._is_upper_triangular(diagonal)
    
    def test_diagonal_block_detection(self, repex_state):
        """Test detection of diagonal blocks in matrices."""
        # Single block
        single_block = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        blocks = repex_state._find_diagonal_blocks(single_block)
        assert len(blocks) == 1
        assert len(blocks[0]) == 3
        
        # Two diagonal blocks
        two_blocks = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ])
        blocks = repex_state._find_diagonal_blocks(two_blocks)
        assert len(blocks) == 2
        assert len(blocks[0]) == 2
        assert len(blocks[1]) == 2
        
        # Three diagonal blocks
        three_blocks = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1]
        ])
        blocks = repex_state._find_diagonal_blocks(three_blocks)
        assert len(blocks) == 3
    
    def test_upper_triangular_permanent(self, repex_state):
        """Test permanent calculation for upper triangular matrices."""
        # For upper triangular matrix, permanent = product of diagonal
        matrix = np.array([
            [2.0, 0.5, 0.3],
            [0.0, 3.0, 0.7],
            [0.0, 0.0, 4.0]
        ])
        
        perm = repex_state._fast_contiguous_permanent(matrix)
        expected = 2.0 * 3.0 * 4.0  # Product of diagonal
        
        assert np.isclose(perm, expected), f"Expected {expected}, got {perm}"
    
    def test_block_diagonal_permanent(self, repex_state):
        """Test permanent calculation for block diagonal matrices."""
        # Block diagonal: 2x2 block and 1x1 block
        matrix = np.array([
            [1.0, 0.5, 0.0],
            [0.3, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        
        # Permanent of [[1.0, 0.5], [0.3, 2.0]] = 1.0*2.0 + 0.5*0.3 = 2.15
        # Permanent of [[3.0]] = 3.0
        # Total permanent = 2.15 * 3.0 = 6.45
        
        perm = repex_state._fast_contiguous_permanent(matrix)
        expected = 6.45
        
        assert np.isclose(perm, expected, rtol=1e-5), f"Expected {expected}, got {perm}"
    
    def test_contiguous_permanent_prob(self, repex_state):
        """Test the full contiguous permanent probability calculation."""
        # STAPLE-like contiguous matrix
        matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.5]
        ])
        
        result = repex_state._contiguous_permanent_prob(matrix)
        
        # Check that result is a probability matrix
        assert result.shape == matrix.shape
        assert np.all(result >= 0), "All probabilities should be non-negative"
        
        # Check row sums are normalized
        row_sums = np.sum(result, axis=1)
        # The normalization divides by max row sum, so max should be 1
        assert np.isclose(np.max(row_sums), 1.0), f"Max row sum should be 1.0, got {np.max(row_sums)}"
    
    def test_random_contiguous_matrices(self, repex_state):
        """Test with random contiguous matrices of varying bandwidth and position."""
        np.random.seed(42)
        
        for trial in range(10):
            n = np.random.randint(4, 8)  # Random size 4-7
            matrix = np.zeros((n, n), dtype=np.longdouble)
            
            for i in range(n):
                # Random bandwidth (2-5 elements per row)
                bandwidth = np.random.randint(2, min(6, n + 1))
                
                # Random starting position (not always centered on diagonal)
                max_start = max(0, n - bandwidth)
                start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                end = min(n, start + bandwidth)
                
                # Fill contiguous region with random values
                for j in range(start, end):
                    matrix[i, j] = np.random.uniform(0.1, 1.0)
            
            # Verify results match standard method
            result_contiguous = repex_state._contiguous_permanent_prob(matrix)
            result_standard = repex_state.permanent_prob(matrix)
            
            assert np.allclose(result_contiguous, result_standard, rtol=1e-8), \
                f"Trial {trial}: Random contiguous matrix failed correctness check"
    
    def test_off_diagonal_contiguous(self, repex_state):
        """Test contiguous regions that don't include the diagonal."""
        # Matrix where each row has contiguous nonzeros but shifted off diagonal
        matrix = np.array([
            [0.0, 0.5, 0.7, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.8, 0.6],
            [0.4, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.7, 0.9],
            [0.3, 0.0, 0.0, 0.0, 0.0]
        ], dtype=np.longdouble)
        
        result_contiguous = repex_state._contiguous_permanent_prob(matrix)
        result_standard = repex_state.permanent_prob(matrix)
        
        assert np.allclose(result_contiguous, result_standard, rtol=1e-8), \
            "Off-diagonal contiguous matrix failed correctness check"
    
    def test_wide_bandwidth_contiguous(self, repex_state):
        """Test contiguous matrices with wide bandwidth (many nonzeros per row)."""
        n = 6
        matrix = np.zeros((n, n), dtype=np.longdouble)
        
        # Each row has 4-5 contiguous nonzero elements
        for i in range(n):
            bandwidth = 5
            start = max(0, i - 1)
            end = min(n, start + bandwidth)
            for j in range(start, end):
                matrix[i, j] = np.random.uniform(0.2, 1.0)
        
        result_contiguous = repex_state._contiguous_permanent_prob(matrix)
        result_standard = repex_state.permanent_prob(matrix)
        
        assert np.allclose(result_contiguous, result_standard, rtol=1e-8), \
            "Wide bandwidth contiguous matrix failed correctness check"
    
    def test_asymmetric_contiguous(self, repex_state):
        """Test asymmetric contiguous patterns (different for each row)."""
        # Realistic STAPLE pattern with varying overlap regions
        matrix = np.array([
            [1.0, 0.3, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.6, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.7, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.6, 0.4, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.9, 0.7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.8]
        ], dtype=np.longdouble)
        
        result_contiguous = repex_state._contiguous_permanent_prob(matrix)
        result_standard = repex_state.permanent_prob(matrix)
        
        assert np.allclose(result_contiguous, result_standard, rtol=1e-8), \
            "Asymmetric contiguous matrix failed correctness check"
    
    def test_small_matrix_correctness(self, repex_state):
        """Test correctness for small matrices by comparing with standard method."""
        # Small 3x3 matrix
        matrix = np.array([
            [1.0, 0.5, 0.0],
            [0.3, 2.0, 0.7],
            [0.0, 0.4, 1.5]
        ])
        
        # Calculate using contiguous method
        result_contiguous = repex_state._contiguous_permanent_prob(matrix)
        
        # Calculate using standard permanent_prob
        result_standard = repex_state.permanent_prob(matrix)
        
        # Results should be very close
        assert np.allclose(result_contiguous, result_standard, rtol=1e-10), \
            "Contiguous and standard methods should produce same results for small matrices"


class TestContiguousPerformance:
    """Performance tests comparing contiguous vs standard permanent calculation."""
    
    def _create_staple_like_matrix(self, n):
        """Create a matrix with STAPLE-like contiguous structure."""
        matrix = np.zeros((n, n), dtype=np.longdouble)
        
        # Fill with contiguous nonzero structure (mostly upper-diagonal band)
        for i in range(n):
            # Each row has nonzero values in a contiguous range
            start = max(0, i)
            end = min(n, i + 3)  # Width of 3 columns
            for j in range(start, end):
                matrix[i, j] = np.random.uniform(0.3, 1.0)
        
        return matrix
    
    def _create_dense_matrix(self, n):
        """Create a dense matrix for comparison."""
        return np.random.uniform(0.3, 1.0, (n, n)).astype(np.longdouble)
    
    def test_performance_small_matrix(self, repex_state, benchmark=None):
        """Test performance on small matrices (n=5)."""
        matrix = self._create_staple_like_matrix(5)
        
        # Time contiguous method
        start = time.perf_counter()
        for _ in range(100):
            result_contiguous = repex_state._contiguous_permanent_prob(matrix)
        time_contiguous = time.perf_counter() - start
        
        # Time standard method
        start = time.perf_counter()
        for _ in range(100):
            result_standard = repex_state.permanent_prob(matrix)
        time_standard = time.perf_counter() - start
        
        print(f"\n5x5 Matrix Performance:")
        print(f"  Contiguous method: {time_contiguous:.4f}s")
        print(f"  Standard method:   {time_standard:.4f}s")
        print(f"  Speedup:           {time_standard/time_contiguous:.2f}x")
        
        # Verify correctness
        assert np.allclose(result_contiguous, result_standard, rtol=1e-8)
    
    def test_performance_medium_matrix(self, repex_state):
        """Test performance on medium matrices (n=8)."""
        matrix = self._create_staple_like_matrix(8)
        
        # Time contiguous method
        start = time.perf_counter()
        for _ in range(10):
            result_contiguous = repex_state._contiguous_permanent_prob(matrix)
        time_contiguous = time.perf_counter() - start
        
        # Time standard method
        start = time.perf_counter()
        for _ in range(10):
            result_standard = repex_state.permanent_prob(matrix)
        time_standard = time.perf_counter() - start
        
        print(f"\n8x8 Matrix Performance:")
        print(f"  Contiguous method: {time_contiguous:.4f}s")
        print(f"  Standard method:   {time_standard:.4f}s")
        print(f"  Speedup:           {time_standard/time_contiguous:.2f}x")
        
        # Verify correctness
        assert np.allclose(result_contiguous, result_standard, rtol=1e-6)
    
    def test_upper_triangular_speed(self, repex_state):
        """Test that upper triangular matrices are computed very fast."""
        n = 10
        # Create upper triangular matrix
        matrix = np.triu(np.random.uniform(0.3, 1.0, (n, n))).astype(np.longdouble)
        
        # Time the fast method
        start = time.perf_counter()
        for _ in range(1000):
            perm = repex_state._fast_contiguous_permanent(matrix)
        time_fast = time.perf_counter() - start
        
        # Time the standard Glynn method
        start = time.perf_counter()
        for _ in range(1000):
            perm_standard = repex_state.fast_glynn_perm(matrix)
        time_standard = time.perf_counter() - start
        
        print(f"\n10x10 Upper Triangular Performance:")
        print(f"  Fast method (diagonal product): {time_fast:.4f}s")
        print(f"  Glynn's algorithm:              {time_standard:.4f}s")
        print(f"  Speedup:                        {time_standard/time_fast:.2f}x")
        
        # For upper triangular, permanent = product of diagonal
        expected = np.prod(np.diag(matrix))
        assert np.isclose(perm, expected)
    
    def test_block_diagonal_speed(self, repex_state):
        """Test that block diagonal matrices are computed efficiently."""
        # Create a block diagonal matrix: two 3x3 blocks
        block1 = np.random.uniform(0.3, 1.0, (3, 3))
        block2 = np.random.uniform(0.3, 1.0, (3, 3))
        
        matrix = np.zeros((6, 6), dtype=np.longdouble)
        matrix[0:3, 0:3] = block1
        matrix[3:6, 3:6] = block2
        
        # Time the fast method
        start = time.perf_counter()
        for _ in range(1000):
            perm = repex_state._fast_contiguous_permanent(matrix)
        time_fast = time.perf_counter() - start
        
        # Time the standard method
        start = time.perf_counter()
        for _ in range(1000):
            perm_standard = repex_state.fast_glynn_perm(matrix)
        time_standard = time.perf_counter() - start
        
        print(f"\n6x6 Block Diagonal Performance:")
        print(f"  Fast method (block decomposition): {time_fast:.4f}s")
        print(f"  Glynn's algorithm:                 {time_standard:.4f}s")
        print(f"  Speedup:                           {time_standard/time_fast:.2f}x")
        
        # Verify correctness
        assert np.isclose(perm, perm_standard, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases for contiguous permanent."""
    
    def test_empty_matrix(self, repex_state):
        """Test empty matrix handling."""
        matrix = np.array([], dtype=np.longdouble).reshape(0, 0)
        result = repex_state._contiguous_permanent_prob(matrix)
        assert result.shape == (0, 0)
    
    def test_1x1_matrix(self, repex_state):
        """Test 1x1 matrix."""
        matrix = np.array([[0.5]], dtype=np.longdouble)
        result = repex_state._contiguous_permanent_prob(matrix)
        # For 1x1, the permanent is the element itself, normalized
        assert result.shape == (1, 1)
        assert np.isclose(result[0, 0], 1.0)  # Normalized to 1
    
    def test_diagonal_matrix(self, repex_state):
        """Test diagonal matrix (special case of upper triangular)."""
        matrix = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.longdouble)
        
        perm = repex_state._fast_contiguous_permanent(matrix)
        expected = 1.0 * 2.0 * 3.0 * 4.0  # Product of diagonal
        
        assert np.isclose(perm, expected)
    
    def test_matrix_with_zeros(self, repex_state):
        """Test matrix with many zero entries."""
        matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.longdouble)
        
        result = repex_state._contiguous_permanent_prob(matrix)
        
        # Should handle sparse matrices correctly
        assert result.shape == matrix.shape
        assert np.all(result >= 0)
    
    def test_numerical_stability(self, repex_state):
        """Test numerical stability with very small and large values."""
        # Matrix with wide range of values
        matrix = np.array([
            [1e-10, 1e-9, 0.0],
            [0.0, 1e10, 1e9],
            [0.0, 0.0, 1.0]
        ], dtype=np.longdouble)
        
        # Should not overflow or underflow
        result = repex_state._contiguous_permanent_prob(matrix)
        
        assert np.all(np.isfinite(result)), "Result should not contain inf or nan"
        assert np.all(result >= 0), "All probabilities should be non-negative"


def test_timing_comparison_summary(repex_state):
    """Summary test comparing different matrix structures and sizes."""
    print("\n" + "="*70)
    print("CONTIGUOUS PERMANENT PERFORMANCE SUMMARY")
    print("="*70)
    
    sizes = [4, 6, 8]
    structures = ['contiguous', 'upper_tri', 'block_diag', 'dense']
    
    for n in sizes:
        print(f"\nMatrix size: {n}x{n}")
        print("-" * 70)
        
        for structure in structures:
            if structure == 'contiguous':
                # STAPLE-like contiguous structure
                matrix = np.zeros((n, n), dtype=np.longdouble)
                for i in range(n):
                    start = max(0, i)
                    end = min(n, i + 3)
                    for j in range(start, end):
                        matrix[i, j] = np.random.uniform(0.3, 1.0)
            
            elif structure == 'upper_tri':
                matrix = np.triu(np.random.uniform(0.3, 1.0, (n, n))).astype(np.longdouble)
            
            elif structure == 'block_diag':
                matrix = np.zeros((n, n), dtype=np.longdouble)
                block_size = n // 2
                matrix[0:block_size, 0:block_size] = np.random.uniform(0.3, 1.0, (block_size, block_size))
                matrix[block_size:n, block_size:n] = np.random.uniform(0.3, 1.0, (n-block_size, n-block_size))
            
            else:  # dense
                matrix = np.random.uniform(0.3, 1.0, (n, n)).astype(np.longdouble)
            
            # Time the calculation
            iterations = 100 if n <= 6 else 10
            start = time.perf_counter()
            for _ in range(iterations):
                result = repex_state._contiguous_permanent_prob(matrix)
            elapsed = time.perf_counter() - start
            
            print(f"  {structure:12s}: {elapsed:.4f}s ({iterations} iterations)")
    
    print("="*70)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
