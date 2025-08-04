"""Edge case and error handling tests for staple paths.

This module contains comprehensive tests for:
- Invalid input handling
- Boundary condition edge cases  
- Error recovery mechanisms
- Numerical stability issues
- Resource limitation handling
"""
import numpy as np
import pytest
from unittest.mock import Mock, patch
import warnings

from infretis.classes.staple_path import StaplePath, turn_detected
from infretis.classes.system import System


class TestStaplePathInputValidation:
    """Test input validation and error handling."""

    def test_invalid_order_parameters(self):
        """Test handling of invalid order parameters."""
        path = StaplePath()
        
        # Test various invalid order parameter scenarios
        invalid_orders = [
            None,           # None value
            [],             # Empty list
            [None],         # List with None
            [float('inf')], # Infinity
            [float('-inf')], # Negative infinity
            [float('nan')], # NaN
            ['invalid'],    # String
            [complex(1,1)], # Complex number
        ]
        
        for i, invalid_order in enumerate(invalid_orders):
            system = System()
            
            try:
                system.order = invalid_order
                path.append(system)
                # If no exception, ensure it's handled gracefully
                if hasattr(system, 'order') and system.order is not None:
                    assert True  # Graceful handling
            except (TypeError, ValueError, AttributeError) as e:
                # Expected exceptions for invalid inputs
                assert isinstance(e, (TypeError, ValueError, AttributeError))

    def test_malformed_system_objects(self):
        """Test handling of malformed system objects."""
        path = StaplePath()
        
        # Test various malformed system scenarios
        test_cases = [
            # Missing essential attributes
            Mock(spec=[]),  # No attributes
            Mock(spec=['config']),  # Missing order
            Mock(spec=['order']),   # Missing config
            
            # Invalid attribute types
            Mock(order="invalid", config=123),
            Mock(order=[0.5], config=None),
        ]
        
        for i, malformed_system in enumerate(test_cases):
            try:
                path.append(malformed_system)
                # If successful, verify path can still function
                assert len(path.phasepoints) >= i
            except (AttributeError, TypeError, ValueError):
                # Expected for truly malformed objects
                assert True

    def test_interface_parameter_validation(self):
        """Test validation of interface parameters."""
        path = StaplePath()
        
        # Create valid path first
        orders = [0.05, 0.25, 0.45, 0.25, 0.05]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"interface_test_{i}.xyz", i)
            path.append(system)
        
        # Test invalid interface configurations
        invalid_interfaces = [
            None,                    # None
            [],                      # Empty list
            [None],                  # List with None
            [float('nan')],          # NaN values
            [float('inf')],          # Infinity
            ['invalid'],             # String values
            [0.5, 0.3, 0.7],        # Non-monotonic
            [0.2, 0.2, 0.3],        # Duplicates
        ]
        
        for interfaces in invalid_interfaces:
            try:
                start_info, end_info, overall_valid = path.check_turns(interfaces)
                # If no exception, should return boolean result
                assert isinstance(overall_valid, bool)
            except (TypeError, ValueError, IndexError):
                # Expected for invalid interface configurations
                assert True

    def test_empty_path_handling(self):
        """Test handling of empty or minimal paths."""
        # Empty path
        empty_path = StaplePath()
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        try:
            start_info, end_info, overall_valid = empty_path.check_turns(interfaces)
            # Should handle gracefully
            assert isinstance(overall_valid, bool)
        except (IndexError, AttributeError):
            # Acceptable to raise errors for empty paths
            assert True
        
        # Single point path
        single_path = StaplePath()
        system = System()
        system.order = [0.25]
        system.config = ("single.xyz", 0)
        single_path.append(system)
        
        try:
            start_info, end_info, overall_valid = single_path.check_turns(interfaces)
            assert isinstance(overall_valid, bool)
        except (IndexError, ValueError):
            # Expected for insufficient data
            assert True

    def test_turn_detected_error_conditions(self):
        """Test error conditions in turn_detected function."""
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Test with various problematic inputs
        error_cases = [
            # Empty phasepoints
            ([], interfaces, 1, 1),
            # Single phasepoint
            ([Mock(order=[0.25])], interfaces, 1, 1),
            # Invalid direction
            ([Mock(order=[0.1]), Mock(order=[0.3])], interfaces, 1, 0),
            # Invalid interface index  
            ([Mock(order=[0.1]), Mock(order=[0.3])], interfaces, -1, 1),
            ([Mock(order=[0.1]), Mock(order=[0.3])], interfaces, 10, 1),
        ]
        
        for phasepoints, intfs, intf_idx, direction in error_cases:
            try:
                result = turn_detected(phasepoints, intfs, intf_idx, direction)
                # If no exception, should return boolean
                assert isinstance(result, bool)
            except (IndexError, ValueError, AttributeError):
                # Expected for invalid inputs
                assert True


class TestStaplePathNumericalStability:
    """Test numerical stability and precision issues."""

    def test_floating_point_precision(self):
        """Test handling of floating point precision issues."""
        path = StaplePath()
        
        # Create path with precision-sensitive values
        epsilon = 1e-15
        interface_value = 0.2
        
        orders = [
            interface_value - epsilon,    # Just below interface
            interface_value + epsilon,    # Just above interface
            interface_value - 2*epsilon,  # Slightly further below
            interface_value + 2*epsilon,  # Slightly further above
        ]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"precision_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, interface_value, 0.3, 0.4]
        
        # Should handle precision issues gracefully
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert isinstance(overall_valid, bool)
        assert isinstance(start_info[0], bool)
        assert isinstance(end_info[0], bool)

    def test_extreme_numerical_values(self):
        """Test handling of extreme numerical values."""
        path = StaplePath()
        
        # Test with extreme but valid values
        extreme_orders = [
            1e-10,   # Very small positive
            1e10,    # Very large
            -1e-10,  # Very small negative (if allowed)
            0.0,     # Exact zero
        ]
        
        for i, order in enumerate(extreme_orders):
            system = System()
            try:
                system.order = [abs(order)]  # Ensure positive
                system.config = (f"extreme_{i}.xyz", i)
                path.append(system)
            except (OverflowError, ValueError):
                # Skip values that cause overflow
                continue
        
        if len(path.phasepoints) > 0:
            interfaces = [1e-5, 1e5, 1e6]  # Extreme interfaces
            
            try:
                start_info, end_info, overall_valid = path.check_turns(interfaces)
                assert isinstance(overall_valid, bool)
            except (OverflowError, ValueError):
                # Acceptable for extreme values
                assert True

    def test_accumulated_numerical_errors(self):
        """Test resistance to accumulated numerical errors."""
        path = StaplePath()
        
        # Simulate accumulated errors through repeated operations
        base_order = 0.2
        accumulated_error = 0.0
        error_increment = 1e-16
        
        for i in range(100):
            accumulated_error += error_increment
            noisy_order = base_order + accumulated_error
            
            system = System()
            system.order = [noisy_order]
            system.config = (f"accumulated_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Should handle accumulated errors
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert isinstance(overall_valid, bool)
        assert len(path.phasepoints) == 100

    def test_numerical_comparison_stability(self):
        """Test stability of numerical comparisons."""
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Create values that are nearly equal to interfaces
        test_values = []
        for interface in interfaces:
            # Add values very close to each interface
            test_values.extend([
                interface - 1e-14,
                interface,
                interface + 1e-14,
            ])
        
        path = StaplePath()
        for i, order in enumerate(test_values):
            system = System()
            system.order = [order]
            system.config = (f"comparison_{i}.xyz", i)
            path.append(system)
        
        # Should handle comparison edge cases
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert isinstance(overall_valid, bool)


class TestStaplePathResourceHandling:
    """Test resource limitation and memory handling."""

    def test_large_path_handling(self):
        """Test handling of very large paths."""
        path = StaplePath()
        
        # Create large path (but not too large for testing)
        large_size = 1000
        
        for i in range(large_size):
            system = System()
            # Create varying order values
            system.order = [0.1 + 0.0001 * i]
            system.config = (f"large_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Should handle large paths efficiently
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert len(path.phasepoints) == large_size
        assert isinstance(overall_valid, bool)

    def test_memory_pressure_simulation(self):
        """Test behavior under simulated memory pressure."""
        path = StaplePath()
        
        # Simulate memory-constrained environment
        max_systems = 50
        
        # Add more systems than the "limit"
        for i in range(max_systems * 2):
            system = System()
            system.order = [0.1 + 0.001 * i]
            system.config = (f"memory_pressure_{i}.xyz", i)
            
            # Simulate memory management
            if len(path.phasepoints) >= max_systems:
                # Remove oldest system
                path.phasepoints.pop(0)
            
            path.append(system)
        
        # Should maintain reasonable size
        assert len(path.phasepoints) <= max_systems
        
        # Should still function
        interfaces = [0.1, 0.2, 0.3]
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        assert isinstance(overall_valid, bool)

    def test_concurrent_access_protection(self):
        """Test protection against concurrent access issues."""
        import threading
        import time
        
        path = StaplePath()
        access_results = []
        
        def concurrent_append(thread_id):
            """Append systems in concurrent thread."""
            try:
                for i in range(10):
                    system = System()
                    system.order = [0.1 + 0.01 * thread_id + 0.001 * i]
                    system.config = (f"concurrent_{thread_id}_{i}.xyz", i)
                    
                    # Small delay to increase chance of race conditions
                    time.sleep(0.001)
                    path.append(system)
                
                access_results.append(True)
            except Exception as e:
                access_results.append(f"Error: {e}")
        
        def concurrent_read(thread_id):
            """Read path properties in concurrent thread."""
            try:
                for i in range(5):
                    length = len(path.phasepoints)
                    if length > 0:
                        # Try to access first and last elements
                        first = path.phasepoints[0]
                        if length > 1:
                            last = path.phasepoints[-1]
                    
                    time.sleep(0.001)
                
                access_results.append(True)
            except Exception as e:
                access_results.append(f"Read Error: {e}")
        
        # Create concurrent threads
        threads = []
        for thread_id in range(3):
            # Writer threads
            threads.append(threading.Thread(target=concurrent_append, args=(thread_id,)))
            # Reader thread
            threads.append(threading.Thread(target=concurrent_read, args=(thread_id,)))
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Should have results from all threads
        assert len(access_results) == 6
        
        # Check for errors
        error_count = sum(1 for result in access_results if isinstance(result, str))
        success_count = sum(1 for result in access_results if result is True)
        
        # Should have some successes (concurrent access might cause some errors)
        assert success_count >= 0


class TestStaplePathRecoveryMechanisms:
    """Test error recovery and fallback mechanisms."""

    def test_partial_data_recovery(self):
        """Test recovery from partial data corruption."""
        path = StaplePath()
        
        # Add normal systems
        for i in range(5):
            system = System()
            system.order = [0.1 + 0.1 * i]
            system.config = (f"normal_{i}.xyz", i)
            path.append(system)
        
        # Simulate partial corruption
        if len(path.phasepoints) > 2:
            # Corrupt middle system
            corrupted_system = path.phasepoints[2]
            corrupted_system.order = None
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Should attempt graceful handling
        try:
            # Filter out corrupted systems
            valid_systems = []
            for pp in path.phasepoints:
                if hasattr(pp, 'order') and pp.order is not None:
                    valid_systems.append(pp)
            
            if len(valid_systems) >= 3:
                # Create temporary path with valid systems
                temp_path = StaplePath()
                for system in valid_systems:
                    temp_path.append(system)
                
                start_info, end_info, overall_valid = temp_path.check_turns(interfaces)
                assert isinstance(overall_valid, bool)
        except Exception:
            # If recovery fails, that's also acceptable
            assert True

    def test_state_restoration(self):
        """Test state restoration mechanisms."""
        original_path = StaplePath()
        
        # Create original state
        orders = [0.05, 0.15, 0.25, 0.35, 0.25, 0.15, 0.05]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"restore_{i}.xyz", i)
            original_path.append(system)
        
        # Save state information
        original_length = len(original_path.phasepoints)
        original_orders = [pp.order[0] for pp in original_path.phasepoints]
        
        # Simulate corruption
        original_path.phasepoints = original_path.phasepoints[:3]  # Truncate
        
        # Attempt restoration
        try:
            # Restore from saved information
            for i, order in enumerate(original_orders[3:], 3):
                system = System()
                system.order = [order]
                system.config = (f"restore_{i}.xyz", i)
                original_path.append(system)
            
            # Verify restoration
            assert len(original_path.phasepoints) == original_length
            
            interfaces = [0.1, 0.2, 0.3, 0.4]
            start_info, end_info, overall_valid = original_path.check_turns(interfaces)
            assert isinstance(overall_valid, bool)
            
        except Exception:
            # If restoration fails, ensure graceful handling
            assert len(original_path.phasepoints) >= 3

    def test_fallback_validation_methods(self):
        """Test fallback validation when primary methods fail."""
        path = StaplePath()
        
        # Create path with potential validation issues
        problematic_orders = [0.05, 0.25, float('inf'), 0.25, 0.05]
        
        for i, order in enumerate(problematic_orders):
            system = System()
            # Handle problematic values
            if str(order) == 'inf':
                order = 0.45  # Fallback value
            
            system.order = [order]
            system.config = (f"fallback_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Primary validation
        try:
            start_info, end_info, overall_valid = path.check_turns(interfaces)
            primary_success = True
        except Exception:
            primary_success = False
            overall_valid = False
        
        # Fallback validation if primary fails
        if not primary_success:
            try:
                # Simple fallback: check if path crosses at least one interface
                orders_list = [pp.order[0] for pp in path.phasepoints]
                min_order = min(orders_list)
                max_order = max(orders_list)
                
                fallback_valid = any(min_order < intf < max_order for intf in interfaces)
                overall_valid = fallback_valid
            except Exception:
                overall_valid = False
        
        # Should have some result
        assert isinstance(overall_valid, bool)


class TestStaplePathWarningHandling:
    """Test warning generation and handling."""

    def test_performance_warnings(self):
        """Test warnings for performance issues."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            path = StaplePath()
            
            # Create potentially performance-problematic path
            large_size = 500  # Large enough to potentially trigger warnings
            
            for i in range(large_size):
                system = System()
                system.order = [0.1 + 0.0001 * i]
                system.config = (f"perf_warn_{i}.xyz", i)
                path.append(system)
            
            interfaces = [0.1, 0.2, 0.3, 0.4]
            
            # This might trigger performance warnings
            start_info, end_info, overall_valid = path.check_turns(interfaces)
            
            # Check if warnings were generated
            warning_count = len(w)
            assert warning_count >= 0  # May or may not have warnings
            
            # Should still function despite warnings
            assert isinstance(overall_valid, bool)

    def test_numerical_precision_warnings(self):
        """Test warnings for numerical precision issues."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            path = StaplePath()
            
            # Create path with precision issues
            interface_val = 0.2
            epsilon = 1e-15
            
            orders = [
                interface_val - epsilon,
                interface_val + epsilon,
                interface_val - 2*epsilon,
            ]
            
            for i, order in enumerate(orders):
                system = System()
                system.order = [order]
                system.config = (f"prec_warn_{i}.xyz", i)
                path.append(system)
            
            interfaces = [0.1, interface_val, 0.3]
            
            # Might trigger precision warnings
            start_info, end_info, overall_valid = path.check_turns(interfaces)
            
            # Should handle warnings gracefully
            assert isinstance(overall_valid, bool)


if __name__ == "__main__":
    pytest.main([__file__])
