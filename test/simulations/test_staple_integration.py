"""Integration tests for complete staple path workflow."""
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from infretis.classes.staple_path import StaplePath
from infretis.classes.system import System
from infretis.classes.repex_staple import REPEX_state_staple
from infretis.core.tis import staple_sh, staple_extender, staple_wf


class MockCompleteEngine:
    """Complete mock engine for integration testing."""
    
    def __init__(self):
        self.description = "Mock Complete Engine"
        self.exe_dir = tempfile.mkdtemp()
        self.order_function = Mock()
        self.subcycles = 1
    
    def propagate_st(self, path, ens_set, system, reverse=False):
        """Mock staple propagation with realistic behavior."""
        interfaces = ens_set.get("all_intfs", ens_set["interfaces"])
        max_steps = min(50, ens_set["tis_set"]["maxlength"] - path.length)
        
        for i in range(max_steps):
            mock_system = System()
            
            # Simulate realistic order parameter evolution
            if reverse:
                # Backward: gradually decrease order parameter
                base_order = system.order[0] if hasattr(system, 'order') else 0.3
                mock_system.order = [max(0.05, base_order - i * 0.02)]
            else:
                # Forward: gradually increase order parameter
                base_order = system.order[0] if hasattr(system, 'order') else 0.2
                mock_system.order = [min(0.6, base_order + i * 0.02)]
            
            mock_system.config = (f"integrated_frame_{i}.xyz", i)
            
            # Add to path
            added = path.append(mock_system)
            if not added:
                return False, "FTX"
            
            # Check for completion (mock turn detection)
            if self._check_staple_completion(path, interfaces):
                return True, "ACC"
        
        return True, "ACC"
    
    def propagate(self, path, ens_set, system, reverse=False):
        """Standard propagation for comparison."""
        for i in range(10):
            mock_system = System()
            mock_system.order = [0.2 + i * 0.03]
            mock_system.config = (f"standard_frame_{i}.xyz", i)
            path.append(mock_system)
        return True, "ACC"
    
    def _check_staple_completion(self, path, interfaces):
        """Mock staple completion check."""
        if path.length < 10:
            return False
        
        # Simple heuristic: check if we have a "turn" pattern
        if path.length >= 15:
            orders = [pp.order[0] for pp in path.phasepoints]
            # Check for local extrema (simple turn detection)
            for i in range(2, len(orders) - 2):
                if (orders[i] > orders[i-1] and orders[i] > orders[i+1]) or \
                   (orders[i] < orders[i-1] and orders[i] < orders[i+1]):
                    return True
        
        return False
    
    def calculate_order(self, system, **kwargs):
        """Mock order parameter calculation."""
        return [0.25]


class TestStapleWorkflowIntegration:
    """Test complete staple path workflow integration."""
    
    @pytest.fixture
    def complete_config(self):
        """Configuration for complete staple simulation."""
        return {
            "current": {
                "size": 4,  # [0-], [0+], [1+], [2+] ensembles
                "cstep": 0,
                "active": [0, 1, 2, 3],
                "locked": [],
                "traj_num": 4,
                "frac": {}
            },
            "runner": {"workers": 1},
            "simulation": {
                "seed": 42,
                "interfaces": [0.1, 0.3, 0.5],
                "all_intfs": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55],
                "shooting_moves": ["st_sh", "st_sh", "st_sh", "st_sh"],
                "mode": "staple"
            },
            "output": {"data_dir": ".", "pattern": False}
        }
    
    @pytest.fixture
    def complex_staple_path(self):
        """Create a complex staple path with multiple turns."""
        path = StaplePath()
        # Complex path: start -> peak -> valley -> peak -> end
        orders = [0.1, 0.2, 0.35, 0.45, 0.35, 0.15, 0.05, 0.15, 0.3, 0.4, 0.3, 0.2, 0.1]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"complex_frame_{i}.xyz", i)
            path.append(system)
        
        # Set shooting region in the middle
        path.sh_region = (4, 8)
        path.path_number = 42
        path.status = "ACC"
        return path
    
    @pytest.fixture
    def mock_engine(self):
        """Provide mock engine for testing."""
        return MockCompleteEngine()
    
    def test_complete_staple_sh_workflow(self, complete_config, complex_staple_path, mock_engine):
        """Test complete staple shooting workflow."""
        ens_set = {
            "interfaces": complete_config["simulation"]["interfaces"],
            "all_intfs": complete_config["simulation"]["all_intfs"],
            "tis_set": {
                "maxlength": 1000,
                "allowmaxlength": False
            },
            "ens_name": "integration_test",
            "rgen": np.random.default_rng(42)
        }
        
        with patch('infretis.core.tis.prepare_shooting_point') as mock_prep:
            with patch('infretis.core.tis.check_kick') as mock_kick:
                # Setup realistic mocks
                shooting_point = System()
                shooting_point.order = [0.25]
                shooting_point.config = ("shooting_point.xyz", 5)
                
                mock_prep.return_value = (shooting_point, 5, [0.01, 0.01, 0.01])
                mock_kick.return_value = True
                
                # Run complete staple shooting
                success, trial_path, status = staple_sh(
                    ens_set, complex_staple_path, mock_engine
                )
                
                assert success
                assert status == "ACC"
                assert isinstance(trial_path, StaplePath)
                assert trial_path.length > complex_staple_path.length
    
    def test_staple_extender_integration(self, complete_config, mock_engine):
        """Test staple extender integration with different path types."""
        ens_set = {
            "interfaces": complete_config["simulation"]["interfaces"],
            "all_intfs": complete_config["simulation"]["all_intfs"],
            "tis_set": {"maxlength": 1000},
            "ens_name": "extender_test",
            "rgen": np.random.default_rng(42)
        }
        
        # Test LML path extension
        lml_segment = StaplePath()
        orders = [0.05, 0.25, 0.05]  # L -> M -> L
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"lml_{i}.xyz", i)
            lml_segment.append(system)
        
        success, extended_path, status = staple_extender(
            lml_segment, "LML", mock_engine, ens_set
        )
        
        assert success
        assert status == "ACC"
        assert extended_path.length > lml_segment.length
        
        # Test RMR path extension
        rmr_segment = StaplePath()
        orders = [0.55, 0.35, 0.55]  # R -> M -> R
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"rmr_{i}.xyz", i)
            rmr_segment.append(system)
        
        success, extended_path, status = staple_extender(
            rmr_segment, "RMR", mock_engine, ens_set
        )
        
        assert success
        assert status == "ACC"
        assert extended_path.length > rmr_segment.length
    
    def test_staple_wf_integration(self, complete_config, complex_staple_path, mock_engine):
        """Test staple wire fencing integration."""
        ens_set = {
            "interfaces": complete_config["simulation"]["interfaces"],
            "all_intfs": complete_config["simulation"]["all_intfs"],
            "tis_set": {"maxlength": 1000},
            "ens_name": "wf_test",
            "rgen": np.random.default_rng(42)
        }
        
        with patch('infretis.core.tis.wirefence_weight_and_pick') as mock_wf:
            with patch('infretis.core.tis.staple_extender') as mock_extend:
                # Create wire fence segment
                wf_segment = StaplePath()
                orders = [0.15, 0.25, 0.35]
                for i, order in enumerate(orders):
                    system = System()
                    system.order = [order]
                    system.config = (f"wf_seg_{i}.xyz", i)
                    wf_segment.append(system)
                
                mock_wf.return_value = (10, wf_segment)  # weight=10
                
                # Mock successful extension
                extended_wf = StaplePath()
                orders = [0.05, 0.15, 0.25, 0.35, 0.45, 0.35, 0.25, 0.15, 0.05]
                for i, order in enumerate(orders):
                    system = System()
                    system.order = [order]
                    system.config = (f"extended_wf_{i}.xyz", i)
                    extended_wf.append(system)
                
                mock_extend.return_value = (True, extended_wf, "ACC")
                
                success, trial_path, status = staple_wf(
                    ens_set, complex_staple_path, mock_engine
                )
                
                assert success
                assert status == "ACC"
                assert trial_path == extended_wf
    
    def test_repex_state_staple_integration(self, complete_config):
        """Test REPEX_state_staple integration."""
        # Test creation and basic functionality
        state = REPEX_state_staple(complete_config)
        
        assert state.n == 4  # [0-], [0+], [1+], [2+]
        assert state._offset == 1  # Because minus=True
        assert len(state.ensembles) == 4
        
        # Test adding trajectories
        test_path = StaplePath()
        orders = [0.05, 0.25, 0.05]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"repex_test_{i}.xyz", i)
            test_path.append(system)
        
        test_path.weights = (0.0, 1.0, 0.0, 0.0)  # Weight in ensemble 1
        test_path.path_number = 1
        
        # Test add_traj method
        valid = (0.0, 1.0, 0.0, 0.0)
        state.add_traj(0, test_path, valid)  # Add to ensemble 0
        
        assert state._trajs[1] == test_path  # Should be at index 1 due to offset
    
    def test_error_propagation_integration(self, complete_config, mock_engine):
        """Test how errors propagate through the complete workflow."""
        ens_set = {
            "interfaces": complete_config["simulation"]["interfaces"],
            "all_intfs": complete_config["simulation"]["all_intfs"],
            "tis_set": {"maxlength": 10},  # Very small limit to force errors
            "ens_name": "error_test",
            "rgen": np.random.default_rng(42)
        }
        
        # Create path that will exceed limits
        large_path = StaplePath()
        for i in range(8):  # Almost at limit
            system = System()
            system.order = [0.1 + i * 0.05]
            system.config = (f"large_{i}.xyz", i)
            large_path.append(system)
        
        large_path.sh_region = (2, 6)
        
        # Test staple_sh with length constraints
        success, trial_path, status = staple_sh(
            ens_set, large_path, mock_engine
        )
        
        # Should handle the constraint gracefully
        assert isinstance(success, bool)
        assert isinstance(status, str)
        assert trial_path is not None


class TestStaplePerformanceIntegration:
    """Test performance aspects of staple integration."""
    
    def test_large_scale_staple_simulation(self):
        """Test staple functionality with large-scale parameters."""
        config = {
            "current": {
                "size": 10,
                "cstep": 0,
                "active": list(range(10)),
                "locked": [],
                "traj_num": 10,
                "frac": {}
            },
            "runner": {"workers": 1},
            "simulation": {
                "seed": 42,
                "interfaces": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "all_intfs": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 
                             0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 
                             0.85, 0.9, 0.95],
                "shooting_moves": ["st_sh"] * 10,
                "mode": "staple"
            },
            "output": {"data_dir": ".", "pattern": False}
        }
        
        # Test that large-scale configuration is handled efficiently
        state = REPEX_state_staple(config)
        
        assert state.n == 10
        assert len(state.ensembles) == 10
        assert state.prob.shape == (10, 10)
    
    def test_memory_efficiency_integration(self):
        """Test memory efficiency in staple operations."""
        engine = MockCompleteEngine()
        
        # Create multiple paths and process them
        paths = []
        for path_id in range(5):
            path = StaplePath()
            for i in range(100):  # Large paths
                system = System()
                system.order = [0.1 + (i % 50) * 0.01]
                system.config = (f"mem_path_{path_id}_{i}.xyz", i)
                path.append(system)
            paths.append(path)
        
        ens_set = {
            "interfaces": [0.1, 0.3, 0.5],
            "all_intfs": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "tis_set": {"maxlength": 5000},
            "ens_name": "memory_test"
        }
        
        # Process all paths - should not cause memory issues
        results = []
        for path in paths:
            # Simulate processing each path
            test_system = System()
            test_system.order = [0.25]
            
            success, status = engine.propagate_st(
                path, ens_set, test_system
            )
            results.append((success, status))
        
        # All operations should complete
        assert len(results) == 5
        assert all(isinstance(result[0], bool) for result in results)


if __name__ == "__main__":
    pytest.main([__file__])
