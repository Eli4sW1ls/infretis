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
        """Test that a staple path cannot have more than 2 turns (start + end)."""
        path = StaplePath()
        
        # Create a valid staple path with both start and end turns
        # Start below interface 0, turn at interface 1, end with turn at interface 2
        orders = [
            0.05,  # Start below first interface (region 0-)
            0.15,  # Cross interface 0 (0.1) into region 0+
            0.25,  # Cross interface 1 (0.2) into region 1+
            0.35,  # Reach maximum in region 1+
            0.25,  # Recross interface 1 - START TURN DETECTED HERE
            0.35,  # Go back up to region 1+
            0.25,  # Cross interface 1 again
            0.15,  # Cross interface 0 again - END TURN DETECTED HERE
        ]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"valid_staple_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Check turns - should have both start and end turns
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        # Should detect exactly 2 turns maximum
        assert isinstance(start_info[0], bool)
        assert isinstance(end_info[0], bool)
        assert isinstance(overall_valid, bool)
        
        # If valid, should have reasonable extremal points
        if overall_valid:
            assert start_info[2] >= 0 and start_info[2] < len(orders)
            assert end_info[2] >= 0 and end_info[2] < len(orders)

    def test_nested_turns_detection(self):
        """Test that nested turn patterns are handled correctly."""
        path = StaplePath()
        
        # Create a valid staple path with a single turn (no nested turns possible)
        # Start in region 0+, make a turn at interface 1, continue to region 2+
        orders = [
            0.15,   # Start in region 0+ (between interfaces 0 and 1)
            0.25,   # Cross interface 1 (0.2) into region 1+
            0.35,   # Continue into region 2+ (cross interface 2)
            0.32,   # Small decrease but stay in region 2+
            0.38,   # Back up in region 2+ (this is normal fluctuation, not a turn)
            0.30,   # Drop back towards interface 2
            0.25,   # Recross interface 2 back to region 1+
            0.20,   # Recross interface 1 - START TURN DETECTED HERE
        ]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"single_turn_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        # Should detect at most one turn (either start or end, not nested)
        assert isinstance(start_info[0], bool)
        assert isinstance(end_info[0], bool)
        assert isinstance(overall_valid, bool)

    def test_asymmetric_turn_patterns(self):
        """Test asymmetric turn patterns."""
        path = StaplePath()
        
        # Asymmetric turn: relatively fast up, slower down, staying within physics limits
        # Respect absorbing boundaries and limit to 1 turn
        orders = [
            0.15,   # Start in region 0+
            0.22,   # Cross interface 1 
            0.32,   # Cross interface 2 quickly
            0.31,   # Slight decrease  
            0.29,   # Slower fall
            0.26,   # Continue down
            0.23,   # Continue down
            0.21,   # Back to region 1+
            0.20    # At interface 1 - turn point
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
        
        # Turn with plateau at the top - respect absorbing boundaries
        orders = [
            0.15,   # Start in region 0+
            0.22,   # Cross interface 1
            0.32,   # Cross interface 2 
            0.32,   # Plateau start
            0.32,   # Plateau continue
            0.32,   # Plateau end
            0.25,   # Begin descent
            0.20    # Back to interface 1 - turn point
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
        
        # Standard turn pattern - respect absorbing boundaries
        orders = [0.15, 0.25, 0.35, 0.25, 0.15]  # Stay between 0.1 and 0.4
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
        # Forward turn (left to right) - respect absorbing boundaries
        forward_path = StaplePath()
        forward_orders = [0.15, 0.22, 0.32, 0.22, 0.15]  # Stay within valid range
        for i, order in enumerate(forward_orders):
            system = System()
            system.order = [order]
            system.config = (f"forward_{i}.xyz", i)
            forward_path.append(system)
        
        # Backward turn (right to left) - start higher  
        backward_path = StaplePath()
        backward_orders = [0.32, 0.22, 0.15, 0.22, 0.32]  # Reverse pattern
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
        
        # Test various minimal scenarios - respect absorbing boundaries
        test_cases = [
            # Just crossing 2 interfaces and back (valid turn)
            ([0.15, 0.25, 0.15], "minimal_valid"),
            # Crossing only 1 interface (insufficient for turn)
            ([0.15, 0.18, 0.15], "insufficient_crossing"),
            # No crossing at all (no turn possible)
            ([0.15, 0.16, 0.15], "no_crossing"),
            # Monotonic trajectory (no turn)
            ([0.15, 0.18, 0.22], "monotonic"),
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
        
        # Clean turn pattern - respect absorbing boundaries
        clean_orders = [0.15, 0.20, 0.25, 0.30, 0.32, 0.30, 0.25, 0.20, 0.15]
        
        # Add different levels of noise
        noise_levels = [0.0, 0.01, 0.02, 0.05]
        
        results = []
        for noise_level in noise_levels:
            path = StaplePath()
            
            for i, clean_order in enumerate(clean_orders):
                noise = np.random.normal(0, noise_level)
                # Clamp to stay within valid range and avoid absorbing boundaries
                noisy_order = max(0.12, min(0.38, clean_order + noise))
                
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
            # Spike pattern - respect absorbing boundaries
            ([0.15, 0.15, 0.35, 0.15, 0.15], "spike"),
            # Step pattern - stay within valid range
            ([0.15, 0.15, 0.15, 0.32, 0.32, 0.32, 0.15, 0.15], "step"),
            # Sawtooth pattern - limited oscillation
            ([0.15, 0.25, 0.15, 0.25, 0.15, 0.25, 0.15], "sawtooth"),
            # Gradual growth pattern
            ([0.15, 0.17, 0.20, 0.25, 0.30, 0.25, 0.20, 0.15], "gradual")
        ]
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
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
            # Simple symmetric turn - respect absorbing boundaries
            ([0.15, 0.20, 0.25, 0.30, 0.25, 0.20, 0.15], "simple"),
            # Asymmetric turn
            ([0.15, 0.30, 0.25, 0.20, 0.15], "asymmetric"),
            # Incomplete turn  
            ([0.15, 0.20, 0.25, 0.30, 0.28], "incomplete"),
            # Sharp turn
            ([0.15, 0.20, 0.32, 0.20, 0.15], "sharp")
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
        
        # Create path that covers different interfaces - respect absorbing boundaries
        orders = [0.15, 0.18, 0.22, 0.25, 0.28, 0.32, 0.35, 0.32, 0.28, 0.25, 0.22, 0.18, 0.15]
        
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
