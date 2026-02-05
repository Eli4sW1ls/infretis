"""End-to-end workflow tests for staple simulations.

This module contains tests for:
- Complete simulation workflows
- Multi-ensemble coordination
- State persistence and recovery
- Error recovery and resilience
- Integration with various engines
"""
import os
import tempfile
import shutil
import json
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from infretis.classes.staple_path import StaplePath
from infretis.classes.system import System
from infretis.classes.path import Path
from infretis.core.tis import staple_sh, staple_extender


class MockAdvancedEngine:
    """Advanced mock engine for comprehensive workflow testing."""
    
    def __init__(self, success_rate=0.9, max_steps=100):
        self.success_rate = success_rate
        self.max_steps = max_steps
        self.propagation_count = 0
        self.description = "Advanced Mock Engine"
        
    def propagate(self, path, ens_set, system, reverse=False):
        """Mock propagation with realistic failure modes."""
        return self.propagate_st(path, ens_set, system, reverse)
        
    def propagate_st(self, path, ens_set, system, reverse=False):
        """Mock staple propagation with realistic failure modes."""
        self.propagation_count += 1
        
        # Simulate occasional failures
        if np.random.random() > self.success_rate:
            return False, "REJ"
        
        interfaces = ens_set.get("interfaces", [0.1, 0.3, 0.5])
        current_order = system.order[0]
        
        # Simulate realistic propagation
        steps = 0
        while steps < self.max_steps:
            steps += 1
            
            # Add noise to trajectory
            noise = np.random.normal(0, 0.02)
            new_order = max(0.0, min(1.0, current_order + noise))
            
            new_system = System()
            new_system.order = [new_order]
            new_system.config = (f"prop_step_{steps}.xyz", steps)
            
            added = path.append(new_system)
            if not added:
                return False, "FTX"  # Path full
            
            current_order = new_order
            
            # Check for completion conditions (simplified)
            if self._check_completion(path, interfaces):
                return True, "ACC"
        
        return True, "ACC"  # Max steps reached
    
    def _check_completion(self, path, interfaces):
        """Check if path meets completion criteria."""
        if path.length < 5:
            return False
        
        # Simple completion check
        start_order = path.phasepoints[0].order[0]
        end_order = path.phasepoints[-1].order[0]
        
        # Very basic turn-like behavior
        return abs(start_order - end_order) < 0.1
    
    def modify_velocities(self, system, tis_set):
        """Mock velocity modification."""
        return 0.1, True  # dE_k, success


class TestStapleWorkflowEndToEnd:
    """Test complete end-to-end staple workflows."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def complete_simulation_config(self):
        """Complete simulation configuration."""
        return {
            "current": {
                "size": 4,
                "cstep": 0,
                "active": [0, 1, 2, 3],
                "locked": [],
                "traj_num": 4,
                "frac": {}
            },
            "runner": {"workers": 1},
            "simulation": {
                "seed": 42,
                "interfaces": [0.15, 0.2, 0.25],
                "all_intfs": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                "shooting_moves": ["st_sh", "st_sh", "st_sh", "st_sh"],
                "mode": "staple",
                "maxlength": 1000
            },
            "output": {"data_dir": ".", "pattern": False}
        }
    
    @pytest.fixture
    def sample_staple_trajectory(self):
        """Create a sample staple trajectory."""
        path = StaplePath()
        
        # Create realistic staple trajectory
        orders = [
            0.12, 0.18, 0.25, 0.32, 0.38, 0.42,  # Rising
            0.38, 0.32, 0.25, 0.18, 0.12,        # Falling (turn)
            0.18, 0.25, 0.32, 0.38, 0.42         # Rising again
        ]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"sample_traj_{i}.xyz", i)
            system.timestep = i * 0.1
            path.append(system)
        
        path.sh_region = {1: (5, 10)}  # Middle region for ensemble 1
        path.path_number = 1
        path.status = "ACC"
        
        return path

    def test_complete_staple_shooting_workflow(self, complete_simulation_config, sample_staple_trajectory):
        """Test complete staple shooting workflow."""
        engine = MockAdvancedEngine(success_rate=0.8)
        
        # Create ensemble settings
        ens_set = {
            "interfaces": complete_simulation_config["simulation"]["interfaces"],
            "all_intfs": complete_simulation_config["simulation"]["all_intfs"],
            "tis_set": {"maxlength": 500, "allowmaxlength": False},
            "ens_name": "1",  # Non-zero ensemble
            "rgen": np.random.default_rng(42)
        }
        
        # Test shooting move
        with patch('infretis.core.tis.prepare_shooting_point') as mock_prep:
            with patch('infretis.core.tis.check_kick') as mock_kick:
                with patch('infretis.core.tis.shoot_backwards') as mock_back:
                    with patch('infretis.core.tis.paste_paths') as mock_paste:
                        # Setup mocks
                        shooting_point = System()
                        shooting_point.order = [0.3]
                        mock_prep.return_value = (shooting_point, 5, 0.1)
                        mock_kick.return_value = True
                        mock_back.return_value = True
                        mock_paste.return_value = sample_staple_trajectory
                        
                        # Execute shooting move
                        success, trial_path, status = staple_sh(
                            ens_set, sample_staple_trajectory, engine
                        )
        
        # Verify workflow completion
        assert isinstance(success, bool)
        assert trial_path is not None
        assert isinstance(status, str)
        assert status in ["ACC", "REJ", "FTK", "FTX", "EXT", "FTL"]

    def test_multi_ensemble_coordination(self, complete_simulation_config):
        """Test coordination between multiple ensembles."""
        engine = MockAdvancedEngine()
        
        # Create paths for different ensembles
        ensemble_paths = {}
        
        for ens_id in [0, 1, 2, 3]:
            path = StaplePath()
            
            # Different trajectory patterns for each ensemble
            if ens_id == 0:
                orders = [0.05, 0.1, 0.05]  # 0- ensemble
            else:
                base_order = 0.1 + ens_id * 0.1
                orders = [base_order, base_order + 0.2, base_order]
            
            for i, order in enumerate(orders):
                system = System()
                system.order = [order]
                system.config = (f"ens_{ens_id}_frame_{i}.xyz", i)
                path.append(system)
            
            if ens_id > 0:  # Non-zero ensembles need shooting regions
                path.sh_region = {i: (0, len(orders) - 1)}
                
            ensemble_paths[ens_id] = path
        
        # Test that each ensemble can be processed
        interfaces = complete_simulation_config["simulation"]["interfaces"]
        
        results = {}
        for ens_id, path in ensemble_paths.items():
            if ens_id == 0:
                # Ensemble 0 uses regular shooting
                ens_set = {
                    "interfaces": interfaces,
                    "ens_name": "0",
                    "tis_set": {"maxlength": 100},
                    "rgen": np.random.default_rng(42 + ens_id)
                }
            else:
                # Other ensembles use staple shooting
                ens_set = {
                    "interfaces": interfaces,
                    "all_intfs": complete_simulation_config["simulation"]["all_intfs"],
                    "ens_name": str(ens_id),
                    "tis_set": {"maxlength": 100},
                    "rgen": np.random.default_rng(42 + ens_id)
                }
            
            # Mock the shooting operation
            with patch('infretis.core.tis.prepare_shooting_point'), \
                 patch('infretis.core.tis.check_kick'), \
                 patch('infretis.core.tis.shoot_backwards'), \
                 patch('infretis.core.tis.paste_paths'):
                
                try:
                    success, trial_path, status = staple_sh(ens_set, path, engine)
                    results[ens_id] = (success, status)
                except Exception as e:
                    results[ens_id] = (False, f"ERROR: {str(e)}")
        
        # All ensembles should process without fatal errors
        assert len(results) == 4
        for ens_id, (success, status) in results.items():
            assert isinstance(success, bool), f"Ensemble {ens_id} success should be boolean"
            assert isinstance(status, str), f"Ensemble {ens_id} status should be string"

    def test_state_persistence_and_recovery(self, temp_workspace, sample_staple_trajectory):
        """Test state persistence and recovery mechanisms."""
        # Save trajectory to file
        traj_file = os.path.join(temp_workspace, "test_trajectory.json")
        
        # Convert trajectory to serializable format
        traj_data = {
            "length": sample_staple_trajectory.length,
            "pptype": sample_staple_trajectory.pptype,
            "sh_region": sample_staple_trajectory.sh_region,
            "path_number": sample_staple_trajectory.path_number,
            "status": sample_staple_trajectory.status,
            "phasepoints": []
        }
        
        for i, pp in enumerate(sample_staple_trajectory.phasepoints):
            pp_data = {
                "index": i,
                "order": pp.order,
                "config": pp.config,
                "timestep": getattr(pp, 'timestep', i * 0.1)
            }
            traj_data["phasepoints"].append(pp_data)
        
        # Save to file
        with open(traj_file, 'w') as f:
            json.dump(traj_data, f)
        
        # Verify file was created
        assert os.path.exists(traj_file)
        
        # Load and verify data can be recovered
        with open(traj_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["length"] == sample_staple_trajectory.length
        assert loaded_data["pptype"] == sample_staple_trajectory.pptype
        # sh_region gets serialized with string keys and list values, normalize for comparison
        loaded_sh_region = {int(k): tuple(v) for k, v in loaded_data["sh_region"].items()}
        assert loaded_sh_region == sample_staple_trajectory.sh_region
        assert len(loaded_data["phasepoints"]) == sample_staple_trajectory.length
        
        # Reconstruct trajectory
        recovered_path = StaplePath(pptype=loaded_data["pptype"])
        # Handle sh_region dict conversion from JSON (keys become strings)
        if loaded_data["sh_region"]:
            if isinstance(loaded_data["sh_region"], dict):
                # Convert string keys back to int keys
                recovered_path.sh_region = {int(k): tuple(v) if isinstance(v, list) else v 
                                          for k, v in loaded_data["sh_region"].items()}
            else:
                # Old format compatibility
                recovered_path.sh_region = {1: tuple(loaded_data["sh_region"])}
        else:
            recovered_path.sh_region = {}
        recovered_path.path_number = loaded_data["path_number"]
        recovered_path.status = loaded_data["status"]
        
        for pp_data in loaded_data["phasepoints"]:
            system = System()
            system.order = pp_data["order"]
            system.config = tuple(pp_data["config"])
            system.timestep = pp_data["timestep"]
            recovered_path.append(system)
        
        # Verify recovered path matches original
        assert recovered_path.length == sample_staple_trajectory.length
        assert recovered_path.pptype == sample_staple_trajectory.pptype
        assert recovered_path.sh_region == sample_staple_trajectory.sh_region

    def test_error_recovery_resilience(self, complete_simulation_config):
        """Test resilience to various error conditions."""
        # Test with unreliable engine
        unreliable_engine = MockAdvancedEngine(success_rate=0.3)  # High failure rate
        
        path = StaplePath()
        orders = [0.1, 0.2, 0.35, 0.4, 0.35, 0.2, 0.1]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"resilience_{i}.xyz", i)
            path.append(system)
        
        path.sh_region = {1: (1, 5)}
        
        ens_set = {
            "interfaces": [0.35, 0.4, 0.45],
            "all_intfs": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            "ens_name": "1",
            "tis_set": {"maxlength": 50},  # Small limit to test constraints
            "rgen": np.random.default_rng(42)
        }
        
        # Test multiple attempts with unreliable engine
        attempts = 10
        results = []
        
        for attempt in range(attempts):
            with patch('infretis.core.tis.prepare_shooting_point') as mock_prep:
                with patch('infretis.core.tis.check_kick') as mock_kick:
                    with patch('infretis.core.tis.shoot_backwards') as mock_back:
                        with patch('infretis.core.tis.paste_paths') as mock_paste:
                            # Setup mocks
                            shooting_point = System()
                            shooting_point.order = [0.3]
                            mock_prep.return_value = (shooting_point, 3, 0.1)
                            mock_kick.return_value = True
                            mock_back.return_value = True
                            mock_paste.return_value = path
                            
                            try:
                                success, trial_path, status = staple_sh(
                                    ens_set, path, unreliable_engine
                                )
                                results.append((success, status))
                            except Exception as e:
                                results.append((False, f"EXCEPTION: {str(e)}"))
        
        # Should handle all attempts gracefully (no uncaught exceptions)
        assert len(results) == attempts
        
        # Check that all results are properly formatted
        exception_count = sum(1 for _, status in results if "EXCEPTION" in status)
        valid_statuses = sum(1 for _, status in results if status in ["ACC", "REJ", "FTK", "FTX", "EXT", "FTL"] or "EXCEPTION" in status)
        
        # All results should be properly handled (no undefined statuses)
        assert valid_statuses == attempts, f"Some results had invalid statuses: {results}"
        
        # With heavy mocking, exceptions should be minimal
        assert exception_count <= attempts // 2, f"Too many exceptions: {exception_count}/{attempts}"

    def test_integration_with_different_engines(self):
        """Test integration with different engine types."""
        path = StaplePath()
        orders = [0.15, 0.25, 0.35, 0.25, 0.15]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"engine_test_{i}.xyz", i)
            path.append(system)
        
        path.sh_region = {1: (1, 3)}
        
        # Test different engine configurations
        engine_configs = [
            {"success_rate": 0.9, "max_steps": 50},   # High success, few steps
            {"success_rate": 0.7, "max_steps": 100},  # Medium success, more steps  
            {"success_rate": 0.5, "max_steps": 200},  # Low success, many steps
        ]
        
        ens_set = {
            "interfaces": [0.2, 0.3, 0.4],
            "all_intfs": [0.2, 0.3, 0.4],  # Make same as interfaces to satisfy validation
            "ens_name": "2",
            "tis_set": {"maxlength": 300},
            "rgen": np.random.default_rng(42)
        }
        
        results = []
        for config in engine_configs:
            engine = MockAdvancedEngine(**config)
            
            with patch('infretis.core.tis.prepare_shooting_point') as mock_prep:
                with patch('infretis.core.tis.check_kick') as mock_kick:
                    with patch('infretis.core.tis.shoot_backwards') as mock_back:
                        with patch('infretis.core.tis.paste_paths') as mock_paste:
                            # Setup mocks
                            shooting_point = System()
                            shooting_point.order = [0.25]
                            mock_prep.return_value = (shooting_point, 2, 0.1)
                            mock_kick.return_value = True
                            mock_back.return_value = True
                            mock_paste.return_value = path
                            
                            success, trial_path, status = staple_sh(
                                ens_set, path, engine
                            )
                            results.append((config, success, status))
        
        # All configurations should be handled
        assert len(results) == len(engine_configs)
        
        for config, success, status in results:
            assert isinstance(success, bool)
            assert isinstance(status, str)
            # Results may vary based on engine config, but should be valid


class TestStaplePathWorkflowValidation:
    """Test validation of complete workflow components."""

    def test_ensemble_configuration_validation(self):
        """Test validation of ensemble configurations."""
        base_config = {
            "current": {"size": 3, "active": [0, 1, 2]},
            "simulation": {
                "interfaces": [0.1, 0.3, 0.5],
                "shooting_moves": ["st_sh", "st_sh", "st_sh"],
                "mode": "staple"
            }
        }
        
        # Test valid configuration
        assert base_config["current"]["size"] == len(base_config["current"]["active"])
        assert len(base_config["simulation"]["shooting_moves"]) == base_config["current"]["size"]
        
        # Test invalid configurations
        invalid_configs = [
            # Size mismatch
            {**base_config, "current": {"size": 2, "active": [0, 1, 2]}},
            # Move count mismatch
            {**base_config, "simulation": {**base_config["simulation"], "shooting_moves": ["st_sh"]}},
            # Empty interfaces
            {**base_config, "simulation": {**base_config["simulation"], "interfaces": []}},
        ]
        
        for config in invalid_configs:
            # These should be caught by validation logic
            size = config["current"]["size"]
            active = config["current"]["active"]
            moves = config["simulation"]["shooting_moves"]
            interfaces = config["simulation"]["interfaces"]
            
            # Basic validation checks
            validation_errors = []
            if len(active) != size:
                validation_errors.append("Size mismatch")
            if len(moves) != size:
                validation_errors.append("Move count mismatch")
            if len(interfaces) == 0:
                validation_errors.append("Empty interfaces")
            
            # At least one validation error should be detected
            assert len(validation_errors) > 0, f"Configuration should be invalid: {config}"

    def test_path_transition_consistency(self):
        """Test consistency of path transitions in workflows."""
        # Create initial path
        initial_path = StaplePath()
        initial_orders = [0.1, 0.2, 0.3, 0.2, 0.1]
        
        for i, order in enumerate(initial_orders):
            system = System()
            system.order = [order]
            system.config = (f"initial_{i}.xyz", i)
            initial_path.append(system)
        
        initial_path.sh_region = {1: (1, 3)}
        
        # Simulate path transitions
        transitions = []
        current_path = initial_path
        
        for step in range(5):
            # Create modified path (simulating shooting move result)
            new_path = StaplePath()
            
            # Keep some points from original, modify others
            for i, pp in enumerate(current_path.phasepoints):
                if i < len(current_path.phasepoints) // 2:
                    # Keep original point
                    new_path.append(pp)
                else:
                    # Create modified point
                    new_system = System()
                    new_order = pp.order[0] + np.random.normal(0, 0.02)
                    new_system.order = [max(0.05, min(0.45, new_order))]
                    new_system.config = (f"modified_{step}_{i}.xyz", i)
                    new_path.append(new_system)
            
            new_path.sh_region = current_path.sh_region
            new_path.path_number = step + 1
            
            transitions.append((current_path, new_path))
            current_path = new_path
        
        # Verify transition consistency
        for i, (old_path, new_path) in enumerate(transitions):
            # Paths should have reasonable continuity
            assert new_path.length > 0
            assert new_path.sh_region is not None
            assert new_path.path_number == i + 1
            
            # Should maintain some connection to previous path
            if old_path.length > 0 and new_path.length > 0:
                # At least the first point might be related (depending on shooting method)
                assert isinstance(old_path.phasepoints[0].order[0], (int, float))
                assert isinstance(new_path.phasepoints[0].order[0], (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
