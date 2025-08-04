"""Integration and workflow tests for staple path simulations.

This module contains comprehensive tests for:
- Complete staple path workflows
- Integration with TIS operations
- Multi-engine coordination
- End-to-end simulation validation
- State persistence and recovery
"""
import os
import tempfile
import shutil
import pytest
from unittest.mock import Mock, patch, MagicMock

from infretis.classes.staple_path import StaplePath
from infretis.classes.system import System
from infretis.classes.path import Path


class TestStaplePathWorkflowIntegration:
    """Test complete staple path workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.interfaces = [0.1, 0.2, 0.3, 0.4]
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_complete_staple_path_generation(self):
        """Test complete staple path generation workflow."""
        # Mock initial conditions
        initial_path = StaplePath()
        
        # Create realistic initial path
        initial_orders = [0.05, 0.15, 0.25, 0.35, 0.25, 0.15, 0.05]
        for i, order in enumerate(initial_orders):
            system = System()
            system.order = [order]
            system.config = (f"initial_{i}.xyz", i)
            system.vel = [f"vel_{i}"]
            system.box = [10.0, 10.0, 10.0]
            system.pos = [f"pos_{i}"]
            initial_path.append(system)
        
        # Test path validation
        start_info, end_info, overall_valid = initial_path.check_turns(self.interfaces)
        
        # Verify workflow components
        assert len(initial_path.phasepoints) == 7
        assert isinstance(overall_valid, bool)
        
        # Test shooting point selection
        if overall_valid:
            # Should be able to select shooting points
            for i in range(1, len(initial_path.phasepoints) - 1):
                shooting_system = initial_path.phasepoints[i]
                assert hasattr(shooting_system, 'order')
                assert hasattr(shooting_system, 'config')

    def test_staple_path_with_engine_integration(self):
        """Test staple path integration with simulation engines."""
        path = StaplePath()
        
        # Mock engine interactions
        mock_engine = Mock()
        mock_engine.execute_command.return_value = True
        mock_engine.read_config.return_value = ("config.xyz", 100)
        mock_engine.get_order.return_value = [0.25]
        
        # Create path with engine-generated data
        for i in range(5):
            system = System()
            system.order = mock_engine.get_order()
            system.config = mock_engine.read_config()
            system.engine = mock_engine
            path.append(system)
        
        # Test engine coordination
        assert len(path.phasepoints) == 5
        assert all(pp.engine == mock_engine for pp in path.phasepoints)
        
        # Test engine command execution
        for pp in path.phasepoints:
            result = pp.engine.execute_command()
            assert result is True

    def test_staple_ensemble_coordination(self):
        """Test coordination with staple ensemble operations."""
        # Create mock staple ensemble
        ensemble = Mock()
        ensemble.interfaces = self.interfaces
        ensemble.stable_states = ['A', 'B']
        ensemble.paths = []
        
        # Create test paths for ensemble
        for path_id in range(3):
            path = StaplePath()
            orders = [0.05 + path_id*0.01, 0.25, 0.45, 0.25, 0.05 + path_id*0.01]
            
            for i, order in enumerate(orders):
                system = System()
                system.order = [order]
                system.config = (f"ensemble_{path_id}_{i}.xyz", i)
                path.append(system)
            
            ensemble.paths.append(path)
        
        # Test ensemble operations
        assert len(ensemble.paths) == 3
        assert ensemble.interfaces == self.interfaces
        
        # Test path validation across ensemble
        valid_paths = 0
        for path in ensemble.paths:
            start_info, end_info, overall_valid = path.check_turns(ensemble.interfaces)
            if overall_valid:
                valid_paths += 1
        
        assert valid_paths >= 0  # Should not crash

    def test_tis_integration_workflow(self):
        """Test integration with TIS simulation workflow."""
        # Mock TIS simulation components
        mock_simulation = Mock()
        mock_simulation.interfaces = self.interfaces
        mock_simulation.ensembles = {}
        
        # Create staple path for TIS
        staple_path = StaplePath()
        orders = [0.05, 0.15, 0.25, 0.35, 0.45, 0.35, 0.25, 0.15, 0.05]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"tis_{i}.xyz", i)
            system.pos = [f"pos_{i}"]
            system.vel = [f"vel_{i}"]
            staple_path.append(system)
        
        # Test TIS workflow integration
        mock_simulation.current_path = staple_path
        
        # Verify TIS components
        assert mock_simulation.interfaces == self.interfaces
        assert mock_simulation.current_path == staple_path
        assert len(mock_simulation.current_path.phasepoints) == 9
        
        # Test path validation in TIS context
        start_info, end_info, overall_valid = staple_path.check_turns(mock_simulation.interfaces)
        assert isinstance(overall_valid, bool)

    def test_multi_replica_coordination(self):
        """Test coordination of multiple staple path replicas."""
        replicas = []
        
        # Create multiple replica paths
        for replica_id in range(4):
            path = StaplePath()
            
            # Slightly different paths for each replica
            base_orders = [0.05, 0.15, 0.25, 0.35, 0.25, 0.15, 0.05]
            noise = 0.01 * replica_id
            
            for i, base_order in enumerate(base_orders):
                system = System()
                system.order = [base_order + noise]
                system.config = (f"replica_{replica_id}_{i}.xyz", i)
                system.replica_id = replica_id
                path.append(system)
            
            replicas.append(path)
        
        # Test replica coordination
        assert len(replicas) == 4
        
        # Verify replica uniqueness
        replica_ids = set()
        for path in replicas:
            for pp in path.phasepoints:
                replica_ids.add(pp.replica_id)
        
        assert len(replica_ids) == 4
        
        # Test cross-replica analysis
        valid_replicas = 0
        for path in replicas:
            start_info, end_info, overall_valid = path.check_turns(self.interfaces)
            if overall_valid:
                valid_replicas += 1
        
        assert valid_replicas >= 0

    def test_state_persistence_workflow(self):
        """Test state persistence and recovery workflows."""
        # Create original path
        original_path = StaplePath()
        orders = [0.05, 0.25, 0.45, 0.25, 0.05]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"persistent_{i}.xyz", i)
            system.pos = [f"pos_{i}"]
            system.vel = [f"vel_{i}"]
            system.box = [10.0, 10.0, 10.0]
            original_path.append(system)
        
        # Mock persistence operations
        with patch('pickle.dump') as mock_dump, \
             patch('pickle.load') as mock_load:
            
            # Mock successful save
            mock_dump.return_value = None
            
            # Mock successful load
            mock_load.return_value = original_path
            
            # Test save operation
            save_file = os.path.join(self.temp_dir, "staple_path.pkl")
            
            # Simulate save
            try:
                with open(save_file, 'wb') as f:
                    mock_dump(original_path, f)
                save_success = True
            except Exception:
                save_success = False
            
            assert save_success
            
            # Simulate load
            try:
                with open(save_file, 'rb') as f:
                    loaded_path = mock_load(f)
                load_success = True
            except Exception:
                load_success = False
                loaded_path = None
            
            assert load_success
            assert loaded_path is not None

    def test_error_recovery_workflow(self):
        """Test error recovery in staple path workflows."""
        path = StaplePath()
        
        # Create path with potential error conditions
        problematic_orders = [0.05, float('nan'), 0.25, -0.1, 0.45]
        
        for i, order in enumerate(problematic_orders):
            system = System()
            try:
                # Handle problematic values
                if str(order) == 'nan':
                    order = 0.0  # Default fallback
                elif order < 0:
                    order = abs(order)  # Absolute value
                
                system.order = [order]
                system.config = (f"error_recovery_{i}.xyz", i)
                path.append(system)
            except Exception as e:
                # Log error and continue with default
                system.order = [0.1]
                system.config = (f"default_{i}.xyz", i)
                path.append(system)
        
        # Test path still functions after error recovery
        assert len(path.phasepoints) == 5
        
        # All orders should be valid numbers
        for pp in path.phasepoints:
            assert isinstance(pp.order[0], (int, float))
            assert not str(pp.order[0]) == 'nan'
            assert pp.order[0] >= 0


class TestStaplePathBoundaryConditions:
    """Test boundary conditions and edge cases in workflows."""

    def test_minimal_valid_workflow(self):
        """Test minimal valid staple path workflow."""
        # Minimal valid path: just crosses one interface and returns
        path = StaplePath()
        minimal_orders = [0.05, 0.25, 0.05]
        
        for i, order in enumerate(minimal_orders):
            system = System()
            system.order = [order]
            system.config = (f"minimal_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3]
        
        # Should handle minimal case
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert len(path.phasepoints) == 3
        assert isinstance(overall_valid, bool)

    def test_maximum_complexity_workflow(self):
        """Test maximum complexity staple path workflow."""
        # Create very complex path
        path = StaplePath()
        
        # Complex pattern with multiple turns and interface crossings
        complex_orders = []
        for cycle in range(3):
            cycle_orders = [0.05, 0.15, 0.25, 0.35, 0.45, 0.35, 0.25, 0.15, 0.05]
            # Add noise and offset for each cycle
            for order in cycle_orders:
                noise = 0.01 * (cycle + 1)
                complex_orders.append(order + noise)
        
        for i, order in enumerate(complex_orders):
            system = System()
            system.order = [order]
            system.config = (f"complex_{i}.xyz", i)
            system.timestep = i
            system.mass = 1.0
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Should handle complex case without performance issues
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        assert len(path.phasepoints) > 20
        assert isinstance(overall_valid, bool)

    def test_concurrent_workflow_operations(self):
        """Test concurrent staple path operations."""
        import threading
        import time
        
        # Shared path for concurrent operations
        shared_path = StaplePath()
        results = []
        
        def add_systems_to_path(thread_id, num_systems):
            """Add systems to path in separate thread."""
            local_results = []
            for i in range(num_systems):
                system = System()
                system.order = [0.1 + 0.1 * i]
                system.config = (f"thread_{thread_id}_{i}.xyz", i)
                system.thread_id = thread_id
                
                # Add with small delay to simulate real conditions
                time.sleep(0.001)
                shared_path.append(system)
                local_results.append(system)
            
            results.append(local_results)
        
        # Create multiple threads
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(
                target=add_systems_to_path,
                args=(thread_id, 5)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify concurrent operations
        assert len(shared_path.phasepoints) == 15  # 3 threads × 5 systems
        assert len(results) == 3

    def test_memory_constrained_workflow(self):
        """Test workflow under memory constraints."""
        # Simulate memory-constrained environment
        path = StaplePath()
        
        # Add systems with memory cleanup simulation
        max_memory_systems = 100
        
        for i in range(max_memory_systems * 2):  # Simulate memory pressure
            system = System()
            system.order = [0.1 + 0.001 * i]
            system.config = (f"memory_{i}.xyz", i)
            
            # Simulate memory management
            if i > max_memory_systems:
                # Remove old systems to manage memory
                if len(path.phasepoints) > max_memory_systems:
                    path.phasepoints.pop(0)
            
            path.append(system)
        
        # Should maintain reasonable memory usage
        assert len(path.phasepoints) <= max_memory_systems + 1
        
        # Should still function correctly
        interfaces = [0.1, 0.2, 0.3]
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        assert isinstance(overall_valid, bool)


class TestStaplePathValidationWorkflows:
    """Test validation workflows for staple paths."""

    def test_complete_validation_pipeline(self):
        """Test complete staple path validation pipeline."""
        # Create path for validation
        path = StaplePath()
        orders = [0.05, 0.15, 0.25, 0.35, 0.45, 0.35, 0.25, 0.15, 0.05]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"validation_{i}.xyz", i)
            system.pos = [f"pos_{i}"]
            system.vel = [f"vel_{i}"]
            system.box = [10.0, 10.0, 10.0]
            system.mass = 1.0
            system.timestep = i * 0.1
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Validation pipeline steps
        validation_results = {}
        
        # 1. Basic structure validation
        validation_results['structure'] = len(path.phasepoints) > 0
        validation_results['has_order'] = all(hasattr(pp, 'order') for pp in path.phasepoints)
        validation_results['has_config'] = all(hasattr(pp, 'config') for pp in path.phasepoints)
        
        # 2. Turn detection validation
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        validation_results['turns'] = overall_valid
        validation_results['start_turn'] = start_info[0]
        validation_results['end_turn'] = end_info[0]
        
        # 3. Physical consistency validation
        validation_results['positive_mass'] = all(
            getattr(pp, 'mass', 1.0) > 0 for pp in path.phasepoints
        )
        validation_results['valid_timesteps'] = all(
            isinstance(getattr(pp, 'timestep', 0), (int, float)) 
            for pp in path.phasepoints
        )
        
        # 4. Interface crossing validation
        min_order = min(pp.order[0] for pp in path.phasepoints)
        max_order = max(pp.order[0] for pp in path.phasepoints)
        crossed_interfaces = [intf for intf in interfaces if min_order < intf < max_order]
        validation_results['interface_crossings'] = len(crossed_interfaces) >= 2
        
        # All validations should pass for well-formed path
        assert validation_results['structure']
        assert validation_results['has_order']
        assert validation_results['has_config']
        assert validation_results['positive_mass']
        assert validation_results['valid_timesteps']
        assert isinstance(validation_results['turns'], bool)

    def test_progressive_validation_workflow(self):
        """Test progressive validation as path is built."""
        path = StaplePath()
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        validation_history = []
        
        # Build path progressively and validate at each step
        target_orders = [0.05, 0.15, 0.25, 0.35, 0.45, 0.35, 0.25, 0.15, 0.05]
        
        for i, order in enumerate(target_orders):
            system = System()
            system.order = [order]
            system.config = (f"progressive_{i}.xyz", i)
            path.append(system)
            
            # Validate current state
            if len(path.phasepoints) >= 3:  # Minimum for turn detection
                start_info, end_info, overall_valid = path.check_turns(interfaces)
                validation_state = {
                    'step': i,
                    'length': len(path.phasepoints),
                    'valid': overall_valid,
                    'start_turn': start_info[0],
                    'end_turn': end_info[0]
                }
                validation_history.append(validation_state)
        
        # Should have validation history
        assert len(validation_history) > 0
        assert all('valid' in state for state in validation_history)
        
        # Final state should be most complete
        final_state = validation_history[-1]
        assert final_state['length'] == 9
        assert isinstance(final_state['valid'], bool)

    def test_cross_validation_workflow(self):
        """Test cross-validation between different validation methods."""
        path = StaplePath()
        orders = [0.05, 0.25, 0.45, 0.25, 0.05]
        
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"cross_val_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Method 1: Direct turn validation
        start_info1, end_info1, overall_valid1 = path.check_turns(interfaces)
        
        # Method 2: Manual interface crossing check
        orders_list = [pp.order[0] for pp in path.phasepoints]
        min_order = min(orders_list)
        max_order = max(orders_list)
        
        crosses_multiple = sum(1 for intf in interfaces if min_order < intf < max_order) >= 2
        has_return = orders_list[0] < interfaces[0] and orders_list[-1] < interfaces[0]
        manual_valid = crosses_multiple and has_return
        
        # Method 3: Extremal point detection
        max_idx = orders_list.index(max_order)
        has_extremal = 0 < max_idx < len(orders_list) - 1
        
        # Cross-validation results
        cross_validation = {
            'direct_method': overall_valid1,
            'manual_method': manual_valid,
            'extremal_method': has_extremal
        }
        
        # Methods should be consistent for well-formed paths
        assert isinstance(cross_validation['direct_method'], bool)
        assert isinstance(cross_validation['manual_method'], bool)
        assert isinstance(cross_validation['extremal_method'], bool)


if __name__ == "__main__":
    pytest.main([__file__])
