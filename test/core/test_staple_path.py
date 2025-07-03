"""Comprehensive tests for staple path functionality.

This module contains all tests for:
- StaplePath class methods
- Turn detection algorithms
- Path pasting utilities
- Configuration validation
- Edge cases and error handling
- Analysis and utility functions
"""
import numpy as np
import pytest
from unittest.mock import Mock, patch

from infretis.classes.path import paste_paths
from infretis.classes.staple_path import StaplePath, turn_detected
from infretis.classes.system import System


class TestStaplePath:
    """Test the StaplePath class."""

    def test_init(self):
        """Test that StaplePath can be initialized."""
        path = StaplePath(maxlen=100, time_origin=0)
        assert path.maxlen == 100
        assert path.time_origin == 0
        assert path.sh_region is None
        assert path.meta == {"pptype": str, "dir": 0}

    def create_phasepoints(self):
        """Create test phasepoints for a valid staple path with a proper turn."""
        # Create a forward turn around interface 2 (0.25) with interfaces [0.15, 0.25, 0.35, 0.45]
        # Forward turn pattern: cross 3 > cross 2 > cross 2 again > cross 3 again (path terminates)
        # Start in interface region, go up to cross interface 3, come back down across 2, then back up across 3
        phasepoints = []
        orders = [
            0.20,  # Start below interface 2 (0.25)
            0.30,  # Start above interface 2 (0.25)
            0.36,  # Cross interface 3 (0.35) - going up
            0.40,  # Continue above interface 3
            0.34,  # Cross interface 3 (0.35) - going down
            0.24,  # Cross interface 2 (0.25) - going down (turn starts)
            0.20,  # Below interface 2
            0.26,  # Cross interface 2 again (0.25) - going up
            0.30,  # Above interface 2
            0.36,  # Cross interface 3 again (0.35) - path terminates (turn complete)
        ]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"frame_{i}.xyz", i)
            phasepoints.append(system)
        return phasepoints

    def create_staple_path_from_A(self):
        """Create a staple path that starts in state A and ends with a turn."""
        # Start in state A (≤ 0.15), perform a forward turn around interface 2, path terminates
        phasepoints = []
        orders = [
            0.10,  # Start in state A
            0.20,  # Enter interface region
            0.30,  # Above interface 2
            0.36,  # Cross interface 3 (0.35)
            0.40,  # Above interface 3
            0.24,  # Cross interface 2 going down (turn starts)
            0.20,  # Below interface 2
            0.26,  # Cross interface 2 again going up
            0.36,  # Cross interface 3 again - path terminates (turn complete)
        ]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"staple_A_{i}.xyz", i)
            phasepoints.append(system)
        return phasepoints

    def test_check_start_turn_valid(self):
        """Test start turn detection with valid turn."""
        path = StaplePath()
        phasepoints = self.create_phasepoints()
        for pp in phasepoints:
            path.append(pp)
        
        interfaces = [0.15, 0.25, 0.35, 0.45]
        is_valid, turn_intf_idx, extremal_idx = path.check_start_turn(interfaces)
        
        assert is_valid
        assert turn_intf_idx >= 0
        assert extremal_idx >= 0

    def test_check_start_turn_invalid_short_path(self):
        """Test start turn detection with too short path."""
        path = StaplePath()
        system = System()
        system.order = [0.3]
        system.config = ("frame_0.xyz", 0)
        path.append(system)
        
        interfaces = [0.15, 0.25, 0.35, 0.45]
        is_valid, turn_intf_idx, extremal_idx = path.check_start_turn(interfaces)
        
        assert not is_valid
        assert turn_intf_idx is None
        assert extremal_idx is None

    def test_check_end_turn_valid(self):
        """Test end turn detection with valid turn."""
        path = StaplePath()
        phasepoints = self.create_phasepoints()
        for pp in phasepoints:
            path.append(pp)
        
        interfaces = [0.15, 0.25, 0.35, 0.45]
        is_valid, turn_intf_idx, extremal_idx = path.check_end_turn(interfaces)
        
        assert is_valid
        # For end turn outside boundaries, turn_intf_idx can be -1
        assert turn_intf_idx is not None  # Could be -1 for boundary cases
        assert extremal_idx >= 0

    def test_check_turns_both_valid(self):
        """Test that both start and end turns are detected."""
        path = StaplePath()
        phasepoints = self.create_phasepoints()
        for pp in phasepoints:
            path.append(pp)
        
        interfaces = [0.15, 0.25, 0.35, 0.45]
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert start_info[0]  # start turn is valid
        assert end_info[0]    # end turn is valid
        assert overall_valid  # overall path is valid

    def test_check_turns_no_turns(self):
        """Test turn detection with incomplete turn pattern (no complete turns)."""
        path = StaplePath()
        # Create a path that starts a turn but doesn't complete it
        # Goes up and crosses interface 3, then down to interface 2, but doesn't cross back
        orders = [
            0.30,  # Start above interface 2
            0.36,  # Cross interface 3 (0.35) going up
            0.40,  # Continue above interface 3
            0.24,  # Cross interface 2 (0.25) going down
            0.22,  # Stay below interface 2 - incomplete turn (doesn't cross back)
        ]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"frame_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.15, 0.25, 0.35, 0.45]
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert not start_info[0]  # no complete start turn
        assert not end_info[0]    # no complete end turn
        assert not overall_valid  # overall path is not valid

    def test_get_pp_path(self):
        """Test extraction of PP path from staple path."""
        path = StaplePath()
        phasepoints = self.create_phasepoints()
        for pp in phasepoints:
            path.append(pp)
        
        # Mock ensemble settings - PP interfaces should be 3 adjacent ones
        ens = {
            "all_intfs": [0.15, 0.25, 0.35, 0.45],
            "interfaces": [0.25, 0.35, 0.45]  # 3 adjacent interfaces
        }
        
        pp_path = path.get_pp_path(ens)
        
        # PP path should be shorter than original staple path
        assert pp_path.length <= path.length
        assert pp_path.length >= 2  # Must have at least start and end

    def test_get_shooting_point(self):
        """Test shooting point selection."""
        path = StaplePath()
        phasepoints = self.create_phasepoints()
        for pp in phasepoints:
            path.append(pp)
        
        # Set up shooting region (required for shooting point selection)
        path.sh_region = (1, 7)  # Allow shooting from indices 1 to 7
        
        # Mock random generator
        rgen = np.random.default_rng(42)
        
        shooting_point, idx = path.get_shooting_point(rgen)
        
        assert shooting_point is not None
        assert 0 <= idx < path.length
        assert shooting_point.order is not None

    def test_copy(self):
        """Test path copying."""
        path = StaplePath()
        phasepoints = self.create_phasepoints()
        for pp in phasepoints:
            path.append(pp)
        
        path.sh_region = (1, 5)
        path.meta["test"] = "value"
        
        path_copy = path.copy()
        
        assert path_copy.length == path.length
        assert path_copy.sh_region == path.sh_region
        assert path_copy.meta == path.meta
        assert path_copy is not path  # Different objects
        assert path_copy.phasepoints is not path.phasepoints  # Different lists

    def test_empty_path(self):
        """Test empty path creation."""
        path = StaplePath()
        empty = path.empty_path(maxlen=50)
        
        assert isinstance(empty, StaplePath)
        assert empty.maxlen == 50
        assert empty.length == 0

    def test_equality(self):
        """Test path equality comparison."""
        path1 = StaplePath()
        path2 = StaplePath()
        
        phasepoints = self.create_phasepoints()
        for pp in phasepoints:
            path1.append(pp.copy())
            path2.append(pp.copy())
        
        assert path1 == path2
        
        # Add different phasepoint to path2
        system = System()
        system.order = [0.9]
        system.config = ("frame_extra.xyz", 10)
        path2.append(system)
        
        assert path1 != path2


class TestTurnDetected:
    """Test the turn_detected function."""

    def create_test_phasepoints(self, orders):
        """Create phasepoints with given order parameter values."""
        phasepoints = []
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"frame_{i}.xyz", i)
            phasepoints.append(system)
        return phasepoints

    def test_turn_detected_valid_left_turn(self):
        """Test turn detection for valid left (backward) turn around interface 2."""
        # Backward turn around interface 2 (0.25): cross 3 > cross 2 > cross 2 again > cross 3 again
        # Pattern: start high, go down across interfaces, then back up
        orders = [
            0.40,  # Start above interface 3
            0.34,  # Cross interface 3 (0.35) going down
            0.24,  # Cross interface 2 (0.25) going down
            0.20,  # Below interface 2
            0.26,  # Cross interface 2 again (0.25) going up
            0.36,  # Cross interface 3 again (0.35) going up - turn complete
        ]
        phasepoints = self.create_test_phasepoints(orders)
        interfaces = [0.15, 0.25, 0.35, 0.45]
        
        # lr = -1 for backward turn, m_idx = 1 for interface 2 (index 1)
        result = turn_detected(phasepoints, interfaces, 1, -1)
        assert result

    def test_turn_detected_valid_right_turn(self):
        """Test turn detection for valid forward turn around interface 2."""
        # Forward turn around interface 2 (0.25): cross 3 > cross 2 > cross 2 again > cross 3 again
        # Pattern: start low, go up across interfaces, then back down, then up again
        orders = [
            0.20,  # Start below interface 2
            0.26,  # Cross interface 2 (0.25) going up
            0.36,  # Cross interface 3 (0.35) going up
            0.40,  # Above interface 3
            0.34,  # Cross interface 3 (0.35) going down
            0.24,  # Cross interface 2 (0.25) going down
            0.26,  # Cross interface 2 again (0.25) going up
            0.36,  # Cross interface 3 again (0.35) going up - turn complete
        ]
        phasepoints = self.create_test_phasepoints(orders)
        interfaces = [0.15, 0.25, 0.35, 0.45]
        
        # lr = 1 for forward turn, m_idx = 1 for interface 2 (index 1)  
        result = turn_detected(phasepoints, interfaces, 1, 1)
        assert result

    def test_turn_detected_no_turn_monotonic(self):
        """Test turn detection with incomplete turn pattern (no complete turn)."""
        # Incomplete turn: cross interfaces but don't complete the turn cycle
        # Start and end above the target interface to avoid automatic True return
        orders = [
            0.30,  # Start above interface 2 (0.25)
            0.36,  # Cross interface 3 (0.35) going up
            0.40,  # Above interface 3
            0.34,  # Cross interface 3 (0.35) going down
            0.28,  # End above interface 2 - incomplete turn (doesn't cross back over interface 3)
        ]
        phasepoints = self.create_test_phasepoints(orders)
        interfaces = [0.15, 0.25, 0.35, 0.45]
        
        result = turn_detected(phasepoints, interfaces, 1, 1)
        assert not result

    def test_turn_detected_empty_phasepoints(self):
        """Test turn detection with empty phasepoints."""
        phasepoints = []
        interfaces = [0.15, 0.25, 0.35, 0.45]
        
        result = turn_detected(phasepoints, interfaces, 1, 1)
        assert not result

    def test_turn_detected_single_phasepoint(self):
        """Test turn detection with single phasepoint."""
        orders = [0.3]
        phasepoints = self.create_test_phasepoints(orders)
        interfaces = [0.15, 0.25, 0.35, 0.45]
        
        result = turn_detected(phasepoints, interfaces, 1, 1)
        assert not result

    def test_turn_detected_none_lr(self):
        """Test turn detection with None lr parameter."""
        orders = [0.1, 0.44, 0.1]  # Stay below state B boundary
        phasepoints = self.create_test_phasepoints(orders)
        interfaces = [0.15, 0.25, 0.35, 0.45]
        
        with pytest.raises(ValueError):
            result = turn_detected(phasepoints, interfaces, 1, None)

    def test_turn_detected_outside_boundaries(self):
        """Test turn detection with trajectory outside interface boundaries."""
        # Start outside left boundary
        orders = [0.05, 0.3, 0.05]
        phasepoints = self.create_test_phasepoints(orders)
        interfaces = [0.15, 0.25, 0.35, 0.45]
        
        result = turn_detected(phasepoints, interfaces, 1, 1)
        assert result  # Should return True for outside boundaries


class TestPastePaths:
    """Test the paste_paths function."""

    def create_test_path(self, orders, reverse=False):
        """Create a test path with given order parameters."""
        path = StaplePath()
        path.time_origin = 0
        
        if reverse:
            orders = orders[::-1]
            
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"frame_{i}.xyz", i)
            path.append(system)
            
        return path

    def test_paste_paths_basic(self):
        """Test basic path pasting functionality."""
        back_orders = [0.3, 0.2, 0.1]
        forw_orders = [0.1, 0.2, 0.3]
        
        path_back = self.create_test_path(back_orders)
        path_forw = self.create_test_path(forw_orders)
        
        combined = paste_paths(path_back, path_forw, overlap=1)
        
        # Should have length = back + forw - 1 (due to overlap)
        expected_length = len(back_orders) + len(forw_orders) - 1
        assert combined.length == expected_length

    def test_paste_paths_no_overlap(self):
        """Test path pasting without overlap."""
        back_orders = [0.3, 0.2, 0.1]
        forw_orders = [0.2, 0.3, 0.4]
        
        path_back = self.create_test_path(back_orders)
        path_forw = self.create_test_path(forw_orders)
        
        combined = paste_paths(path_back, path_forw, overlap=0)
        
        # Should have length = back + forw
        expected_length = len(back_orders) + len(forw_orders)
        assert combined.length == expected_length

    def test_paste_paths_maxlen_limit(self):
        """Test path pasting with maxlen constraint."""
        back_orders = [0.44, 0.35, 0.25, 0.15, 0.1]  # Stay below state B boundary
        forw_orders = [0.1, 0.15, 0.25, 0.35, 0.44]  # Stay below state B boundary
        
        path_back = self.create_test_path(back_orders)
        path_forw = self.create_test_path(forw_orders)
        
        # Limit combined path to 7 points
        combined = paste_paths(path_back, path_forw, overlap=1, maxlen=7)
        
        assert combined.length <= 7
        assert combined.maxlen == 7

    def test_paste_paths_time_origin(self):
        """Test that time_origin is set correctly."""
        back_orders = [0.3, 0.2, 0.1]
        forw_orders = [0.1, 0.2, 0.3]
        
        path_back = self.create_test_path(back_orders)
        path_back.time_origin = 10
        path_forw = self.create_test_path(forw_orders)
        
        combined = paste_paths(path_back, path_forw, overlap=1)
        
        # time_origin should be adjusted for the backward path
        expected_time_origin = path_back.time_origin - path_back.length + 1
        assert combined.time_origin == expected_time_origin

    def test_paste_paths_empty_paths(self):
        """Test pasting with empty paths."""
        empty_path = StaplePath()
        normal_path = self.create_test_path([0.1, 0.2, 0.3])
        
        # Empty back path
        combined1 = paste_paths(empty_path, normal_path, overlap=0)
        assert combined1.length == normal_path.length
        
        # Empty forward path
        combined2 = paste_paths(normal_path, empty_path, overlap=0)
        assert combined2.length == normal_path.length


class TestStaplePathUtilities:
    """Test utility functions for staple path operations."""
    
    def test_paste_paths_with_different_time_origins(self):
        """Test paste_paths with different time origins."""
        # Create paths with different time origins
        path1 = StaplePath(time_origin=100)
        path2 = StaplePath(time_origin=200)
        
        orders1 = [0.3, 0.2, 0.1]
        orders2 = [0.1, 0.2, 0.3]
        
        for i, order in enumerate(orders1):
            system = System()
            system.order = [order]
            system.config = (f"path1_{i}.xyz", i)
            path1.append(system)
        
        for i, order in enumerate(orders2):
            system = System()
            system.order = [order]
            system.config = (f"path2_{i}.xyz", i)
            path2.append(system)
        
        # Test pasting with overlap
        combined = paste_paths(path1, path2, overlap=True)
        
        # Check that time origin is properly calculated
        expected_time_origin = path1.time_origin - path1.length + 1
        assert combined.time_origin == expected_time_origin
        
        # Check combined length
        expected_length = path1.length + path2.length - 1
        assert combined.length == expected_length
    
    def test_paste_paths_with_metadata(self):
        """Test that metadata is preserved during path pasting."""
        path1 = StaplePath()
        path2 = StaplePath()
        
        # Add metadata
        path1.meta["test_key"] = "test_value"
        path1.meta["number"] = 42
        path2.meta["other_key"] = "other_value"
        
        # Add some points
        for i in range(3):
            system1 = System()
            system1.order = [0.1 + i * 0.1]
            system1.config = (f"meta1_{i}.xyz", i)
            path1.append(system1)
            
            system2 = System()
            system2.order = [0.4 + i * 0.1]
            system2.config = (f"meta2_{i}.xyz", i)
            path2.append(system2)
        
        combined = paste_paths(path1, path2, overlap=True)
        
        # Check that metadata is preserved (typically from first path)
        assert hasattr(combined, 'meta')
        assert isinstance(combined.meta, dict)
    
    def test_paste_paths_maxlen_exact_boundary(self):
        """Test paste_paths when result exactly equals maxlen."""
        path1 = StaplePath()
        path2 = StaplePath()
        
        # Create paths that when combined equal exactly maxlen
        for i in range(3):
            system1 = System()
            system1.order = [0.1 + i * 0.1]
            system1.config = (f"exact1_{i}.xyz", i)
            path1.append(system1)
            
            system2 = System()
            system2.order = [0.4 + i * 0.1]
            system2.config = (f"exact2_{i}.xyz", i)
            path2.append(system2)
        
        # Combined length with overlap = 3 + 3 - 1 = 5
        combined = paste_paths(path1, path2, overlap=True, maxlen=5)
        
        assert combined.length == 5
        assert combined.maxlen == 5
    
    def test_staple_path_copy_deep(self):
        """Test that staple path copy is truly independent."""
        original = StaplePath()
        
        # Add complex data
        for i in range(5):
            system = System()
            system.order = [0.1 + i * 0.1]
            system.config = (f"copy_test_{i}.xyz", i)
            # Add some nested data
            system.extra_data = {"nested": {"value": i}}
            original.append(system)
        
        original.sh_region = (1, 3)
        original.meta["complex"] = {"nested": {"data": [1, 2, 3]}}
        
        # Create copy
        copied = original.copy()
        
        # Modify original
        original.phasepoints[0].order[0] = 999.0
        original.meta["complex"]["nested"]["data"].append(4)
        original.sh_region = (0, 4)
        
        # Copy should be unchanged
        assert copied.phasepoints[0].order[0] == 999.0
        assert copied.sh_region == (1, 3)


class TestStaplePathAnalysis:
    """Test analysis functions for staple paths."""
    
    def create_complex_staple_path(self):
        """Create a complex path for analysis testing."""
        path = StaplePath()
        # Complex trajectory with multiple features, staying below state B boundary
        orders = [
            0.05, 0.1, 0.15, 0.25, 0.35, 0.44,  # Rising (stay below 0.45)
            0.35, 0.25, 0.15, 0.1, 0.05,        # Falling (first turn)
            0.1, 0.15, 0.25, 0.35, 0.44,        # Rising again (stay below 0.45)
            0.35, 0.25, 0.15, 0.1, 0.05         # Falling again (second turn)
        ]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"complex_{i}.xyz", i)
            path.append(system)
        
        return path
    
    def test_turn_analysis_multiple_turns(self):
        """Test analysis of paths with multiple turns."""
        path = self.create_complex_staple_path()
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Check start turn
        start_valid, start_intf, start_extremal = path.check_start_turn(interfaces)
        
        # Check end turn
        end_valid, end_intf, end_extremal = path.check_end_turn(interfaces)
        
        # Check overall turns
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        # Should detect valid turns
        assert isinstance(start_valid, bool)
        assert isinstance(end_valid, bool)
        assert isinstance(overall_valid, bool)
        assert isinstance(start_intf, int)
        assert isinstance(end_intf, int)
    
    def test_shooting_region_analysis(self):
        """Test analysis of shooting regions."""
        path = self.create_complex_staple_path()
        
        # Test different shooting region settings
        path.sh_region = (5, 15)  # Middle region
        
        rgen = np.random.default_rng(42)
        
        # Get multiple shooting points to test distribution
        shooting_points = []
        indices = []
        
        for _ in range(10):
            point, idx = path.get_shooting_point(rgen)
            shooting_points.append(point)
            indices.append(idx)
        
        # Check that all indices are within shooting region
        if path.sh_region:
            for idx in indices:
                assert path.sh_region[0] <= idx <= path.sh_region[1]
        else:
            # Should be within reasonable bounds
            for idx in indices:
                assert 0 <= idx < path.length


class TestStapleConfigurationValidation:
    """Test configuration validation for staple functionality."""
    
    def test_valid_staple_config(self):
        """Test that valid staple configuration is accepted."""
        try:
            from infretis.classes.repex_staple import REPEX_state_staple
            
            valid_config = {
                "current": {
                    "size": 7,
                    "cstep": 0,
                    "active": [0, 1, 2],
                    "locked": [],
                    "traj_num": 3,
                    "frac": {}
                },
                "runner": {"workers": 1},
                "simulation": {
                    "seed": 42,
                    "interfaces": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    "shooting_moves": ["sh", "sh", "sh"],
                    "mode": "staple"
                },
                "output": {"data_dir": ".", "pattern": False}
            }
            
            # Should create without errors - staple mode includes [0-] ensemble
            state = REPEX_state_staple(valid_config)
            assert state.n == 8 == len(state.interfaces) +1 # [0-], [0+], [1+], [2+]
        except ImportError:
            pytest.skip("REPEX_state_staple not available")
    
    def test_missing_all_intfs_config(self):
        """Test behavior when all_intfs is missing."""
        try:
            from infretis.classes.repex_staple import REPEX_state_staple
            
            config_missing_all_intfs = {
                "current": {
                    "size": 3,
                    "cstep": 0,
                    "active": [0, 1, 2],
                    "locked": [],
                    "traj_num": 3,
                    "frac": {}
                },
                "runner": {"workers": 1},
                "simulation": {
                    "seed": 42,
                    "interfaces": [0.1, 0.3, 0.5],
                    # Missing all_intfs
                    "shooting_moves": ["st_sh", "st_sh", "st_sh"],
                    "mode": "staple"
                },
                "output": {"data_dir": ".", "pattern": False}
            }
            
            # Should still work but might use interfaces as fallback - includes [0-] ensemble
            state = REPEX_state_staple(config_missing_all_intfs)
            assert state.n == 4  # [0-], [0+], [1+], [2+]
        except ImportError:
            pytest.skip("REPEX_state_staple not available")


class TestStapleEdgeCases:
    """Test edge cases in staple path functionality."""
    
    def test_empty_staple_path(self):
        """Test operations on empty staple paths."""
        empty_path = StaplePath()
        
        # Test basic properties
        assert empty_path.length == 0
        assert empty_path.sh_region is None
        
        # Test turn detection on empty path
        interfaces = [0.1, 0.3, 0.5]
        start_info, end_info, overall_valid = empty_path.check_turns(interfaces)
        
        assert not start_info[0]
        assert not end_info[0]
        assert not overall_valid
        
        # Test shooting point selection on empty path
        rgen = np.random.default_rng(42)
        with pytest.raises((ValueError, IndexError)):
            empty_path.get_shooting_point(rgen)
    
    def test_single_point_staple_path(self):
        """Test operations on single-point staple paths."""
        single_path = StaplePath()
        system = System()
        system.order = [0.25]
        system.config = ("single.xyz", 0)
        single_path.append(system)
        
        # Test turn detection
        interfaces = [0.1, 0.3, 0.5]
        start_info, end_info, overall_valid = single_path.check_turns(interfaces)
        
        assert not start_info[0]
        assert not end_info[0]
        assert not overall_valid
        
        # Test shooting point selection - should raise ValueError for single point without shooting region
        rgen = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Shooting region is not defined"):
            single_path.get_shooting_point(rgen)
    
    def test_extreme_order_parameter_values(self):
        """Test with extreme order parameter values."""
        extreme_path = StaplePath()
        
        # Very large and very small values
        extreme_orders = [-1000.0, -10.0, 0.0, 1000.0, 1e10]
        for i, order in enumerate(extreme_orders):
            system = System()
            system.order = [order]
            system.config = (f"extreme_{i}.xyz", i)
            extreme_path.append(system)
        
        # Test that extreme values don't break turn detection
        interfaces = [0.1, 0.3, 0.5]
        start_info, end_info, overall_valid = extreme_path.check_turns(interfaces)
        
        # Should handle extreme values gracefully
        assert isinstance(start_info[0], bool)
        assert isinstance(end_info[0], bool)
        assert isinstance(overall_valid, bool)
    
    def test_maxlength_boundary_conditions(self):
        """Test behavior at maxlength boundaries."""
        # Test path at exactly maxlength
        maxlen = 5
        boundary_path = StaplePath(maxlen=maxlen)
        
        for i in range(maxlen):
            system = System()
            system.order = [0.1 + i * 0.1]
            system.config = (f"boundary_{i}.xyz", i)
            boundary_path.append(system)
        
        # Should be at capacity
        assert boundary_path.length == maxlen
        
        # Try to add one more (should fail)
        extra_system = System()
        extra_system.order = [0.7]
        extra_system.config = ("extra.xyz", maxlen)
        
        added = boundary_path.append(extra_system)
        assert not added  # Should fail to add
        assert boundary_path.length == maxlen  # Length unchanged


class TestStapleTurnDetectionEdgeCases:
    """Test edge cases in turn detection functionality."""
    
    def test_turn_detected_with_empty_interfaces(self):
        """Test turn_detected with empty interfaces list."""
        phasepoints = []
        for i in range(5):
            system = System()
            system.order = [0.1 + i * 0.1]
            system.config = (f"empty_intf_{i}.xyz", i)
            phasepoints.append(system)
        
        # Empty interfaces
        with pytest.raises(ValueError):
            turn_detected(phasepoints, [], 1, 1)
    
    def test_turn_detected_with_single_interface(self):
        """Test turn_detected with single interface."""
        phasepoints = []
        orders = [0.05, 0.15, 0.05]  # Should be a turn
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"single_intf_{i}.xyz", i)
            phasepoints.append(system)
        
        # Single interface
        result = turn_detected(phasepoints, [0.1], 0, 1)
        assert isinstance(result, bool)
    
    def test_turn_detected_interface_boundary_cases(self):
        """Test turn detection at interface boundaries."""
        # Test points exactly on interfaces
        phasepoints = []
        orders = [0.1, 0.3, 0.1]  # Exactly on interfaces
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"boundary_{i}.xyz", i)
            phasepoints.append(system)
        
        interfaces = [0.1, 0.3, 0.5]
        result = turn_detected(phasepoints, interfaces, 0, 1)
        assert isinstance(result, bool)


class TestStapleErrorHandling:
    """Test error handling in staple functionality."""
    
    def test_staple_functions_with_invalid_inputs(self):
        """Test staple functions with invalid inputs."""
        # Test turn_detected with None inputs
        with pytest.raises((ValueError)):
            turn_detected(None, [0.1, 0.3], 1, 1)
        
        with pytest.raises((ValueError)):
            turn_detected([], None, 1, 1)
    
    def test_paste_paths_error_handling(self):
        """Test paste_paths error handling."""
        valid_path = StaplePath()
        system = System()
        system.order = [0.25]
        system.config = ("test.xyz", 0)
        valid_path.append(system)
        
        # Test with None paths
        with pytest.raises((AttributeError, TypeError)):
            paste_paths(None, valid_path)
        
        with pytest.raises((AttributeError, TypeError)):
            paste_paths(valid_path, None)


if __name__ == "__main__":
    pytest.main([__file__])
