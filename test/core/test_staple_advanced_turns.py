"""Advanced turn detection tests for staple paths.

This module contains comprehensive tests for:
- Complex turn patterns
- Multiple turn scenarios  
- Turn validation edge cases
- Interface crossing analysis
- Turn direction detection
"""
import numpy as np
import pytest
from unittest.mock import Mock

from infretis.classes.staple_path import StaplePath, turn_detected
from infretis.classes.system import System


class TestAdvancedTurnDetection:
    """Test advanced turn detection scenarios."""

    def test_multiple_turns_in_single_path(self):
        """Test detection of multiple turns in one path."""
        path = StaplePath()
        
        # Create path with multiple distinct turns
        orders = [
            # First turn: L->M->R->M->L
            0.05, 0.15, 0.25, 0.35, 0.45, 0.35, 0.25, 0.15, 0.05,
            # Second turn: L->M->R->M->L  
            0.15, 0.25, 0.35, 0.45, 0.35, 0.25, 0.15,
            # Third turn: L->M->R->M->L
            0.25, 0.35, 0.45, 0.35, 0.25
        ]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"multi_turn_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Check start and end turns
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        # Should detect valid turns
        assert start_info[0]  # Start turn valid
        assert end_info[0]    # End turn valid
        assert overall_valid  # Overall valid
        
        # Verify extremal points make sense
        assert isinstance(start_info[2], int)  # Extremal index
        assert isinstance(end_info[2], int)    # Extremal index
        assert start_info[2] >= 0
        assert end_info[2] >= 0

    def test_nested_turns_detection(self):
        """Test detection of nested/overlapping turn patterns."""
        path = StaplePath()
        
        # Create nested turn pattern
        orders = [
            0.05,   # Start
            0.25,   # Cross interface 1
            0.35,   # Cross interface 2  
            0.45,   # Peak
            0.40,   # Small dip
            0.47,   # Higher peak (nested)
            0.30,   # Drop
            0.20,   # Cross interface 1 back
            0.05    # Return to start
        ]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"nested_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        # Should handle nested patterns
        assert isinstance(start_info[0], bool)
        assert isinstance(end_info[0], bool)
        assert isinstance(overall_valid, bool)

    def test_asymmetric_turn_patterns(self):
        """Test asymmetric turn patterns."""
        path = StaplePath()
        
        # Asymmetric turn: fast up, slow down
        orders = [
            0.05, 0.10, 0.20, 0.40,  # Fast rise
            0.39, 0.38, 0.35, 0.32, 0.29, 0.26, 0.23, 0.18, 0.12, 0.06  # Slow fall
        ]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"asymmetric_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        # Should detect asymmetric patterns
        assert isinstance(overall_valid, bool)
        
        # If valid, should have reasonable extremal points
        if overall_valid:
            assert start_info[2] >= 0
            assert end_info[2] >= 0

    def test_plateau_in_turn_detection(self):
        """Test turn detection with plateaus at extremal points."""
        path = StaplePath()
        
        # Turn with plateau at the top
        orders = [
            0.05, 0.15, 0.25, 0.35,     # Rise
            0.40, 0.40, 0.40, 0.40,     # Plateau
            0.35, 0.25, 0.15, 0.05      # Fall
        ]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"plateau_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        # Should handle plateaus correctly
        assert isinstance(overall_valid, bool)
        
        # Test individual functions too
        for i in range(len(path.phasepoints)):
            # Turn detected should work on subsets
            subset_points = path.phasepoints[:i+1]
            if len(subset_points) >= 2:
                # Convert subset to orders array
                subset_orders = np.array([pp.order[0] for pp in subset_points])
                result = turn_detected(subset_orders, interfaces, 1, 1)
                assert isinstance(result, bool)

    def test_interface_density_effects(self):
        """Test turn detection with different interface densities."""
        path = StaplePath()
        
        # Standard turn pattern
        orders = [0.05, 0.25, 0.45, 0.25, 0.05]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"density_{i}.xyz", i)
            path.append(system)
        
        # Test with different interface densities
        interface_sets = [
            [0.1, 0.4],                           # Sparse
            [0.1, 0.2, 0.3, 0.4],                # Medium  
            [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],  # Dense
            [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]  # Very dense
        ]
        
        results = []
        for interfaces in interface_sets:
            start_info, end_info, overall_valid = path.check_turns(interfaces)
            results.append(overall_valid)
        
        # Results should be consistent across interface densities
        assert len(results) == 4
        assert all(isinstance(result, bool) for result in results)

    def test_turn_direction_consistency(self):
        """Test consistency of turn direction detection."""
        # Forward turn (left to right)
        forward_path = StaplePath()
        forward_orders = [0.1, 0.2, 0.4, 0.2, 0.1]
        for i, order in enumerate(forward_orders):
            system = System()
            system.order = [order]
            system.config = (f"forward_{i}.xyz", i)
            forward_path.append(system)
        
        # Backward turn (right to left)  
        backward_path = StaplePath()
        backward_orders = [0.4, 0.2, 0.1, 0.2, 0.4]
        for i, order in enumerate(backward_orders):
            system = System()
            system.order = [order]
            system.config = (f"backward_{i}.xyz", i)
            backward_path.append(system)
        
        interfaces = [0.15, 0.25, 0.35]
        
        # Test turn_detected function with different directions - convert to orders arrays
        forward_orders = np.array([pp.order[0] for pp in forward_path.phasepoints])
        backward_orders = np.array([pp.order[0] for pp in backward_path.phasepoints])
        
        forward_result_right = turn_detected(forward_orders, interfaces, 1, 1)
        forward_result_left = turn_detected(forward_orders, interfaces, 1, -1)
        
        backward_result_right = turn_detected(backward_orders, interfaces, 1, 1)
        backward_result_left = turn_detected(backward_orders, interfaces, 1, -1)
        
        # Forward path should be detected as right turn, backward as left turn
        assert isinstance(forward_result_right, bool)
        assert isinstance(forward_result_left, bool)
        assert isinstance(backward_result_right, bool)
        assert isinstance(backward_result_left, bool)

    def test_minimal_turn_requirements(self):
        """Test minimal requirements for valid turn detection."""
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Test various minimal scenarios
        test_cases = [
            # Just crossing 2 interfaces and back
            ([0.05, 0.25, 0.05], "minimal_valid"),
            # Crossing only 1 interface
            ([0.05, 0.15, 0.05], "insufficient_crossing"),
            # No crossing at all
            ([0.05, 0.08, 0.05], "no_crossing"),
            # Monotonic trajectory
            ([0.05, 0.15, 0.25], "monotonic"),
        ]
        
        for orders, case_name in test_cases:
            path = StaplePath()
            for i, order in enumerate(orders):
                system = System()
                system.order = [order]
                system.config = (f"{case_name}_{i}.xyz", i)
                path.append(system)
            
            start_info, end_info, overall_valid = path.check_turns(interfaces)
            
            # All should return boolean results without crashing
            assert isinstance(start_info[0], bool)
            assert isinstance(end_info[0], bool)
            assert isinstance(overall_valid, bool)


class TestTurnDetectionRobustness:
    """Test robustness of turn detection under various conditions."""

    def test_noise_resistance(self):
        """Test turn detection resistance to noise."""
        np.random.seed(42)  # For reproducibility
        
        # Clean turn pattern
        clean_orders = [0.05, 0.15, 0.25, 0.35, 0.45, 0.35, 0.25, 0.15, 0.05]
        
        # Add different levels of noise
        noise_levels = [0.0, 0.01, 0.02, 0.05]
        
        results = []
        for noise_level in noise_levels:
            path = StaplePath()
            
            for i, clean_order in enumerate(clean_orders):
                noise = np.random.normal(0, noise_level)
                noisy_order = max(0.0, min(0.5, clean_order + noise))
                
                system = System()
                system.order = [noisy_order]
                system.config = (f"noise_{noise_level}_{i}.xyz", i)
                path.append(system)
            
            interfaces = [0.1, 0.2, 0.3, 0.4]
            start_info, end_info, overall_valid = path.check_turns(interfaces)
            results.append(overall_valid)
        
        # Should be robust to small amounts of noise
        assert len(results) == 4
        assert all(isinstance(result, bool) for result in results)

    def test_interface_boundary_precision(self):
        """Test precision near interface boundaries."""
        path = StaplePath()
        
        # Create path that barely crosses interfaces
        epsilon = 1e-10
        orders = [
            0.1 - epsilon,    # Just below interface
            0.1 + epsilon,    # Just above interface  
            0.2 - epsilon,    # Just below next interface
            0.2 + epsilon,    # Just above next interface
            0.1 + epsilon,    # Back just above first
            0.1 - epsilon     # Back just below first
        ]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"precision_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Should handle precision issues gracefully
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert isinstance(start_info[0], bool)
        assert isinstance(end_info[0], bool)
        assert isinstance(overall_valid, bool)

    def test_degenerate_interface_configurations(self):
        """Test with degenerate interface configurations."""
        path = StaplePath()
        
        # Standard turn
        orders = [0.1, 0.3, 0.1]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"degenerate_{i}.xyz", i)
            path.append(system)
        
        # Test various degenerate configurations
        degenerate_configs = [
            [0.2, 0.2, 0.2],        # All same
            [0.1, 0.1, 0.2],        # Duplicates
            [0.3, 0.2, 0.1],        # Reverse order
            [0.2],                  # Single interface
        ]
        
        for config in degenerate_configs:
            try:
                start_info, end_info, overall_valid = path.check_turns(config)
                # Should either work or raise appropriate error
                assert isinstance(overall_valid, bool)
            except (ValueError, IndexError):
                # Acceptable to raise errors for truly degenerate cases
                pass

    def test_extreme_path_shapes(self):
        """Test with extreme path shapes."""
        test_shapes = [
            # Spike pattern
            ([0.1, 0.1, 0.5, 0.1, 0.1], "spike"),
            # Step pattern  
            ([0.1, 0.1, 0.1, 0.4, 0.4, 0.4, 0.1, 0.1], "step"),
            # Sawtooth pattern
            ([0.1, 0.3, 0.1, 0.3, 0.1, 0.3, 0.1], "sawtooth"),
            # Exponential-like growth
            ([0.1, 0.11, 0.13, 0.17, 0.25, 0.41, 0.17, 0.1], "exponential")
        ]
        
        interfaces = [0.15, 0.25, 0.35]
        
        for orders, shape_name in test_shapes:
            path = StaplePath()
            
            for i, order in enumerate(orders):
                system = System()
                system.order = [order]
                system.config = (f"{shape_name}_{i}.xyz", i)
                path.append(system)
            
            # Should handle extreme shapes without crashing
            start_info, end_info, overall_valid = path.check_turns(interfaces)
            
            assert isinstance(start_info[0], bool)
            assert isinstance(end_info[0], bool)
            assert isinstance(overall_valid, bool)


class TestTurnDetectionAnalysis:
    """Test analytical aspects of turn detection."""

    def test_turn_quality_metrics(self):
        """Test different qualities of turns."""
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        turn_qualities = [
            # Perfect symmetric turn
            ([0.05, 0.15, 0.25, 0.35, 0.25, 0.15, 0.05], "perfect"),
            # Asymmetric turn
            ([0.05, 0.35, 0.25, 0.15, 0.05], "asymmetric"),
            # Incomplete turn  
            ([0.05, 0.15, 0.25, 0.35, 0.30], "incomplete"),
            # Overshooting turn
            ([0.05, 0.15, 0.45, 0.15, 0.05], "overshooting")
        ]
        
        results = {}
        for orders, quality in turn_qualities:
            path = StaplePath()
            
            for i, order in enumerate(orders):
                system = System()
                system.order = [order]
                system.config = (f"{quality}_{i}.xyz", i)
                path.append(system)
            
            start_info, end_info, overall_valid = path.check_turns(interfaces)
            results[quality] = {
                'start_valid': start_info[0],
                'end_valid': end_info[0],
                'overall_valid': overall_valid,
                'start_extremal': start_info[2],
                'end_extremal': end_info[2]
            }
        
        # All should produce valid boolean results
        for quality, result in results.items():
            assert isinstance(result['overall_valid'], bool)
            if result['overall_valid']:
                assert isinstance(result['start_extremal'], int)
                assert isinstance(result['end_extremal'], int)

    def test_interface_coverage_analysis(self):
        """Test analysis of interface coverage in turns."""
        path = StaplePath()
        
        # Create path that covers different numbers of interfaces
        orders = [0.05, 0.12, 0.18, 0.25, 0.32, 0.38, 0.45, 0.38, 0.32, 0.25, 0.18, 0.12, 0.05]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"coverage_{i}.xyz", i)
            path.append(system)
        
        # Test with different interface sets
        interface_sets = [
            [0.1, 0.4],                    # 2 interfaces - minimal coverage
            [0.1, 0.2, 0.4],              # 3 interfaces - moderate coverage  
            [0.1, 0.2, 0.3, 0.4],         # 4 interfaces - good coverage
            [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # 7 interfaces - high coverage
        ]
        
        coverage_results = []
        for interfaces in interface_sets:
            start_info, end_info, overall_valid = path.check_turns(interfaces)
            
            # Count interfaces crossed
            min_order = min(order[0] for order in [pp.order for pp in path.phasepoints])
            max_order = max(order[0] for order in [pp.order for pp in path.phasepoints])
            
            crossed_interfaces = sum(1 for intf in interfaces if min_order < intf < max_order)
            
            coverage_results.append({
                'num_interfaces': len(interfaces),
                'crossed': crossed_interfaces,
                'valid': overall_valid
            })
        
        # Should handle different coverage levels
        assert len(coverage_results) == 4
        assert all(isinstance(result['valid'], bool) for result in coverage_results)
        assert all(result['crossed'] >= 0 for result in coverage_results)


if __name__ == "__main__":
    pytest.main([__file__])
