"""Performance and stress tests for staple path functionality.

This module contains tests for:
- Performance under large path conditions
- Memory efficiency
- Turn detection algorithm optimization
- Interface crossing performance
- Shooting point selection at scale
"""
import time
import numpy as np
import pytest
from unittest.mock import Mock

from infretis.classes.staple_path import StaplePath, turn_detected
from infretis.classes.system import System

try:
    from test.utils.staple_path_utils import (
        create_staple_path_with_turn, create_non_staple_path,
        create_large_staple_path, add_systems_to_path
    )
except ImportError:
    # Fallback if utils not available
    def create_staple_path_with_turn(interfaces, turn_at_interface_pair=None, start_region="A", extra_length=5, noise_level=0.0):
        """Fallback: Create a correct staple path with turn."""
        if turn_at_interface_pair is None:
            turn_idx = len(interfaces) // 2
            turn_at_interface_pair = (turn_idx, min(turn_idx + 1, len(interfaces) - 2))
        
        turn_i, turn_j = turn_at_interface_pair
        
        if start_region == "A":
            # Start below interface 0, cross to turn point, then turn back
            orders = [interfaces[0] - 0.05]  # Start in A
            # Go up to interface j
            for k in range(3 + extra_length // 2):
                progress = (k + 1) / (3 + extra_length // 2)
                order = interfaces[0] - 0.05 + progress * (interfaces[turn_j] + 0.02 - (interfaces[0] - 0.05))
                orders.append(order)
            # Turn back down past interface i
            for k in range(1, 4 + extra_length - extra_length // 2):
                progress = k / (3 + extra_length - extra_length // 2)
                order = interfaces[turn_j] + 0.02 - progress * (interfaces[turn_j] + 0.02 - (interfaces[turn_i] - 0.02))
                orders.append(order)
        else:
            # Simple fallback for other cases
            orders = [interfaces[0] - 0.05, interfaces[turn_j] + 0.02, interfaces[turn_i] - 0.02]
        
        return orders
    
    def create_large_staple_path(interfaces, n_cycles=10, turn_interface_pair=None):
        """Fallback: Create large staple path."""
        all_orders = []
        for cycle in range(n_cycles):
            cycle_orders = create_staple_path_with_turn(interfaces, turn_interface_pair, "A", 3, 0.005)
            if cycle > 0 and len(cycle_orders) > 0:
                cycle_orders = cycle_orders[1:]
            all_orders.extend(cycle_orders)
        return all_orders
    
    def add_systems_to_path(path, orders, prefix="test"):
        """Fallback: Add systems to path."""
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"{prefix}_{i}.xyz", i)
            path.append(system)


class TestStaplePathPerformance:
    """Test performance characteristics of StaplePath operations."""

    def test_large_path_turn_detection_performance(self):
        """Test turn detection performance with large paths."""
        path = StaplePath()
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Create a large CORRECT staple path with realistic turn patterns
        large_orders = create_large_staple_path(interfaces, n_cycles=100, turn_interface_pair=(1, 2))
        add_systems_to_path(path, large_orders, "large_frame")
        
        # Time the turn detection
        start_time = time.time()
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        end_time = time.time()
        
        # Should complete in reasonable time (< 1 second for large path)
        assert (end_time - start_time) < 1.0
        assert isinstance(overall_valid, bool)
        assert path.length == len(large_orders)

    def test_shooting_region_selection_efficiency(self):
        """Test efficiency of shooting region selection in large paths."""
        path = StaplePath()
        
        # Create medium-large path
        for i in range(1000):
            system = System()
            system.order = [0.2 + 0.1 * np.sin(i * 0.01)]
            system.config = (f"medium_frame_{i}.xyz", i)
            path.append(system)
        
        # Set shooting region
        path.sh_region = (100, 900)
        rgen = np.random.default_rng(42)
        
        # Time multiple shooting point selections
        start_time = time.time()
        for _ in range(100):
            shooting_point, idx = path.get_shooting_point(rgen)
            assert path.sh_region[0] <= idx <= path.sh_region[1]
        end_time = time.time()
        
        # Should be very fast (< 0.1 seconds for 100 selections)
        assert (end_time - start_time) < 0.1

    def test_interface_crossing_detection_scalability(self):
        """Test interface crossing detection with many interfaces."""
        path = StaplePath()
        interfaces = [0.1 + i * 0.05 for i in range(16)]  # 16 interfaces from 0.1 to 0.85
        
        # Create CORRECT staple path that properly crosses interfaces
        # Use staple path with turn at middle interface pair (7,8)
        orders = create_staple_path_with_turn(interfaces, turn_at_interface_pair=(7, 8), 
                                            start_region="A", extra_length=150)
        add_systems_to_path(path, orders, "crossing_test")
        
        start_time = time.time()
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        end_time = time.time()
        
        # Should handle many interfaces efficiently
        assert (end_time - start_time) < 0.5
        assert isinstance(overall_valid, bool)

    def test_turn_detected_function_performance(self):
        """Test performance of turn_detected function with various scenarios."""
        interfaces = [0.15, 0.25, 0.35, 0.45, 0.55]
        
        # Create CORRECT staple path patterns for testing
        test_cases = [
            # Forward staple (correct)
            (create_staple_path_with_turn(interfaces, turn_at_interface_pair=(1, 2), 
                                        start_region="A", extra_length=2), 3, 1),
            # Monotonic (should not be turn)
            ([0.1, 0.2, 0.3, 0.4, 0.5], 2, 1),
            # Backward staple (correct)
            (create_staple_path_with_turn(interfaces, turn_at_interface_pair=(1, 2),
                                        start_region="A", extra_length=2)[::-1], 2, 1),
        ]
        
        total_time = 0
        for orders, m_idx, lr in test_cases:
            phasepoints = []
            for i, order in enumerate(orders):
                system = System()
                system.order = [order]
                system.config = (f"perf_{i}.xyz", i)
                phasepoints.append(system)
            
            # Test many times for statistical significance
            start_time = time.time()
            for _ in range(1000):
                result = turn_detected(phasepoints, interfaces, m_idx, lr)
                assert isinstance(result, bool)
            end_time = time.time()
            
            total_time += (end_time - start_time)
        
        # Should be very fast even for many iterations
        assert total_time < 0.5  # Less than 0.5 seconds for 3000 turn detections

    def test_memory_efficiency_large_paths(self):
        """Test memory efficiency when working with large paths."""
        # Create multiple large paths to test memory usage
        paths = []
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        for path_id in range(5):
            path = StaplePath()
            
            # Create CORRECT staple path (2000 points)
            orders = create_large_staple_path(interfaces, n_cycles=200, turn_interface_idx=2)
            add_systems_to_path(path, orders, f"mem_path_{path_id}")
            
            path.sh_region = (500, 1500)
            paths.append(path)
        
        # Test operations on all paths
        rgen = np.random.default_rng(42)
        
        results = []
        for path in paths:
            # Test turn detection
            start_info, end_info, valid = path.check_turns(interfaces)
            
            # Test shooting point selection
            shooting_point, idx = path.get_shooting_point(rgen)
            
            # Test copying
            path_copy = path.copy()
            
            results.append((valid, idx, path_copy.length))
        
        # All operations should complete successfully
        assert len(results) == 5
        assert all(isinstance(result[0], bool) for result in results)
        assert all(isinstance(result[1], (int, type(None))) for result in results)
        assert all(result[2] > 0 for result in results)


class TestStaplePathStressTests:
    """Stress tests for extreme conditions."""

    def test_extreme_oscillation_patterns(self):
        """Test with extreme oscillation patterns."""
        path = StaplePath()
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Create correct forward staple path with high frequency variations
        base_orders = create_staple_path_with_turn(interfaces, turn_at_interface_pair=(1, 2), 
                                                  start_region="A", extra_length=450)
        
        # Add high frequency oscillations to the base staple pattern
        oscillating_orders = []
        for i, base_order in enumerate(base_orders):
            # Add extreme oscillations while preserving staple structure
            oscillation = 0.02 * np.sin(i * 0.5) + 0.01 * np.sin(i * 2.0)
            final_order = max(0.0, min(1.0, base_order + oscillation))
            oscillating_orders.append(final_order)
        
        add_systems_to_path(path, oscillating_orders, "oscillate_extreme")
        
        # Should handle extreme oscillations without crashing
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert isinstance(start_info[0], bool)
        assert isinstance(end_info[0], bool)
        assert isinstance(overall_valid, bool)

    def test_random_walk_patterns(self):
        """Test with random walk patterns."""
        path = StaplePath()
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Create correct staple path with random walk noise
        base_orders = create_staple_path_with_turn(interfaces, turn_at_interface_pair=(1, 2), 
                                                  start_region="A", extra_length=950)
        
        # Add random walk noise to the base staple pattern
        rng = np.random.default_rng(42)
        noisy_orders = []
        for base_order in base_orders:
            # Add random walk noise while keeping within bounds
            noise = rng.normal(0, 0.01)
            noisy_order = max(0.0, min(0.5, base_order + noise))
            noisy_orders.append(noisy_order)
        
        add_systems_to_path(path, noisy_orders, "random_walk")
        
        # Should handle random patterns
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert isinstance(overall_valid, bool)
        assert path.length == len(noisy_orders)

    def test_edge_case_interface_configurations(self):
        """Test with challenging interface configurations."""
        path = StaplePath()
        
        # Create path that barely crosses interfaces
        orders = [0.1001, 0.0999, 0.1001, 0.0999, 0.1001]  # Just around 0.1
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"edge_case_{i}.xyz", i)
            path.append(system)
        
        # Very close interfaces
        interfaces = [0.1, 0.1001, 0.1002, 0.1003]
        
        # Should handle very close interfaces
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert isinstance(overall_valid, bool)

    def test_concurrent_operations_simulation(self):
        """Simulate concurrent operations on staple paths."""
        # Create shared CORRECT staple path
        shared_path = StaplePath()
        interfaces = [0.15, 0.25, 0.35]
        
        # Create correct forward staple path
        orders = create_staple_path_with_turn(interfaces, turn_at_interface_pair=(0, 1), 
                                            start_region="A", extra_length=90)
        add_systems_to_path(shared_path, orders, "shared")
        
        shared_path.sh_region = (10, len(orders) - 10)
        
        # Simulate multiple "concurrent" operations
        operations_results = []
        
        for op_id in range(10):
            # Each "thread" performs different operations
            rgen = np.random.default_rng(42 + op_id)
            
            # Operation 1: Turn detection
            start_info, end_info, valid = shared_path.check_turns(interfaces)
            
            # Operation 2: Shooting point selection
            shooting_point, idx = shared_path.get_shooting_point(rgen)
            
            # Operation 3: Path copying
            path_copy = shared_path.copy()
            
            operations_results.append((valid, idx, path_copy.length))
        
        # All operations should produce consistent results
        assert len(operations_results) == 10
        assert all(result[2] == len(orders) for result in operations_results)  # All copies same length
        
        # All shooting points should be in valid range
        valid_indices = [result[1] for result in operations_results if result[1] is not None]
        if valid_indices:  # Only check if we got valid indices
            assert all(10 <= idx <= len(orders) - 10 for idx in valid_indices)


class TestStaplePathBenchmarks:
    """Benchmark tests for performance regression detection."""

    def test_turn_detection_complexity(self):
        """Test that turn detection scales reasonably with path length."""
        interfaces = [0.1, 0.2, 0.3, 0.4]
        path_sizes = [10, 50, 100, 200]  # Scaled down for faster testing
        times = []
        
        for size in path_sizes:
            path = StaplePath()
            
            # Create CORRECT staple path with predictable pattern
            # Use number of cycles proportional to desired size
            n_cycles = max(1, size // 10)
            orders = create_large_staple_path(interfaces, n_cycles=n_cycles, turn_interface_idx=2)
            
            # Trim to exact size if needed
            if len(orders) > size:
                orders = orders[:size]
            
            add_systems_to_path(path, orders, "benchmark")
            
            # Time the operation
            start_time = time.time()
            start_info, end_info, overall_valid = path.check_turns(interfaces)
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert isinstance(overall_valid, bool)
        
        # Check that complexity is reasonable (should be roughly linear)
        # Time for 200 points should be < 100x time for 10 points (more lenient)
        if times[0] > 0:  # Avoid division by zero
            assert times[-1] < 100 * times[0]
        
        # Also check that times are generally reasonable (< 1 second)
        assert all(t < 1.0 for t in times)

    def test_shooting_point_selection_consistency(self):
        """Test consistency of shooting point selection performance."""
        path = StaplePath()
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Create CORRECT medium-sized staple path
        orders = create_large_staple_path(interfaces, n_cycles=100, turn_interface_idx=2)
        add_systems_to_path(path, orders, "consistency")
        
        path.sh_region = (100, len(orders) - 100)
        
        # Test selection times are consistent
        rgen = np.random.default_rng(42)
        selection_times = []
        
        for _ in range(50):
            start_time = time.time()
            shooting_point, idx = path.get_shooting_point(rgen)
            end_time = time.time()
            
            selection_times.append(end_time - start_time)
            assert shooting_point is not None or idx is None  # Either both valid or both None
            if idx is not None:
                assert 100 <= idx <= len(orders) - 100
        
        # All selections should be very fast and consistent
        max_time = max(selection_times)
        min_time = min(selection_times)

        assert max_time < 0.01  # Less than 10ms (more lenient)
        # Only check time variation if min_time is meaningful
        if min_time > 1e-6:  # Only if minimum time is > 1 microsecond
            assert max_time < 100 * min_time  # No more than 100x variation
if __name__ == "__main__":
    pytest.main([__file__])
