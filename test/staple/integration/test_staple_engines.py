"""Test staple-related engine functionality"""
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from infretis.classes.staple_path import StaplePath
from infretis.classes.system import System
from infretis.classes.engines.enginebase import EngineBase


class MockStapleEngine(EngineBase):
    """Mock engine implementation for testing staple functionality."""
    
    def __init__(self):
        super().__init__("Mock Staple Engine", timestep=0.001, subcycles=1)
        self.order_function = Mock()
        self.order_function.calculate = Mock(return_value=[0.25])
    
    def modify_velocities(self, ensemble, vel_settings):
        return 0.0, 1.0
    
    def set_mdrun(self, md_items):
        pass
    
    def _extract_frame(self, traj_file, idx, out_file):
        # Mock frame extraction
        with open(out_file, 'w') as f:
            f.write(f"Mock frame {idx}\n")
    
    def _read_configuration(self, config_file):
        # Mock configuration reading
        return np.array([[0.0, 0.0, 0.0]]), np.array([[0.1, 0.1, 0.1]]), np.array([1.0, 1.0, 1.0]), None
    
    def _propagate_from(self, name, path, system, ens_set, msg_file, reverse=False):
        """Mock _propagate_from method."""
        # Add some mock propagation behavior
        for i in range(3):
            mock_system = System()
            mock_system.order = [0.2 + i * 0.1]
            mock_system.config = (f"prop_frame_{i}.xyz", i)
            path.append(mock_system)
        return True, "ACC"
    
    def _reverse_velocities(self, filename, outfile):
        """Mock velocity reversal."""
        # Just copy the file to simulate reversal
        import shutil
        shutil.copy2(filename, outfile)
    
    def propagate_st(self, path, ens_set, system, reverse=False):
        """Mock staple propagation method."""
        # Add mock phasepoints simulating staple propagation
        for i in range(5):
            mock_system = System()
            if reverse:
                mock_system.order = [0.5 - i * 0.1]  # Decreasing order
            else:
                mock_system.order = [0.1 + i * 0.1]  # Increasing order
            mock_system.config = (f"staple_frame_{i}.xyz", i)
            path.append(mock_system)
        return True, "ACC"


class TestEngineStapleFunctionality:
    """Test staple-specific engine functionality."""
    
    @pytest.fixture
    def mock_engine(self):
        """Provide a mock engine."""
        return MockStapleEngine()
    
    @pytest.fixture
    def sample_staple_path(self):
        """Create a sample staple path."""
        path = StaplePath()
        orders = [0.2, 0.3, 0.4]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"initial_{i}.xyz", i)
            path.append(system)
        return path
    
    @pytest.fixture
    def mock_ens_set(self):
        """Provide mock ensemble settings."""
        return {
            "interfaces": [0.1, 0.3, 0.5],
            "all_intfs": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "tis_set": {"maxlength": 1000},
            "ens_name": "test_staple_ensemble",
            "rgen": np.random.default_rng(42)
        }

    def test_add_to_path_with_staple_path(self, mock_engine, mock_ens_set):
        """Test add_to_path method with StaplePath objects."""
        interfaces = mock_ens_set["all_intfs"]
        left, right = 0.1, 0.5
        
        # Create a staple path
        staple_path = StaplePath()
        
        # Add initial point
        system1 = System()
        system1.order = [0.25]
        system1.config = ("frame_0.xyz", 0)
        staple_path.append(system1)
        
        # Add second point
        system2 = System()
        system2.order = [0.35]
        system2.config = ("frame_1.xyz", 1)
        
        # Test add_to_path with StaplePath
        status, success, stop, add = mock_engine.add_to_path(
            staple_path, system2, left, right, interfaces
        )
        
        assert add  # Point should be added
        assert staple_path.length == 2
        
        # Check that the function handles StaplePath correctly
        assert status in ["Running propagate...", "Crossed left interface!", 
                         "Crossed right interface!", "Order parameter is not monotonic!"]

    def test_add_to_path_interface_crossing(self, mock_engine):
        """Test add_to_path with interface crossing."""
        path = StaplePath()
        left, right = 0.2, 0.8
        
        # Add point that crosses left interface
        system = System()
        system.order = [0.1]  # Below left interface
        system.config = ("frame_cross.xyz", 0)
        
        status, success, stop, add = mock_engine.add_to_path(
            path, system, left, right
        )
        
        assert success
        assert stop
        assert "Crossed left interface!" in status

    def test_add_to_path_max_length_exceeded(self, mock_engine):
        """Test add_to_path when maximum length is exceeded."""
        path = StaplePath(maxlen=2)  # Very small maxlen
        left, right = 0.2, 0.8
        
        # Fill path to maximum
        for i in range(2):
            system = System()
            system.order = [0.5]
            system.config = (f"frame_{i}.xyz", i)
            path.append(system)
        
        # Try to add one more point
        extra_system = System()
        extra_system.order = [0.6]
        extra_system.config = ("frame_extra.xyz", 3)
        
        status, success, stop, add = mock_engine.add_to_path(
            path, extra_system, left, right
        )
        
        assert not success
        assert stop
        assert not add
        assert "Max. path length exceeded" in status

    def test_propagate_st_forward(self, mock_engine, mock_ens_set, sample_staple_path):
        """Test staple propagation in forward direction."""
        system = System()
        system.order = [0.3]
        system.config = ("start.xyz", 0)
        
        success, status = mock_engine.propagate_st(
            sample_staple_path, mock_ens_set, system, reverse=False
        )
        
        assert success
        assert status == "ACC"
        assert sample_staple_path.length > 3  # Should have added points

    def test_propagate_st_reverse(self, mock_engine, mock_ens_set, sample_staple_path):
        """Test staple propagation in reverse direction."""
        system = System()
        system.order = [0.3]
        system.config = ("start.xyz", 0)
        
        success, status = mock_engine.propagate_st(
            sample_staple_path, mock_ens_set, system, reverse=True
        )
        
        assert success
        assert status == "ACC"
        assert sample_staple_path.length > 3  # Should have added points

    def test_calculate_order_with_staple_path(self, mock_engine):
        """Test order parameter calculation for staple paths."""
        system = System()
        system.pos = np.array([[1.0, 2.0, 3.0]])
        system.vel = np.array([[0.1, 0.2, 0.3]])
        system.box = np.array([10.0, 10.0, 10.0])
        system.config = ("test.xyz", 0)
        
        order = mock_engine.calculate_order(system)
        
        assert order is not None
        assert len(order) == 1
        assert order[0] == 0.25  # From mock


class TestStaplePathTurnDetection:
    """Test turn detection functionality in engine context."""
    
    def test_turn_detected_with_mock_engine(self):
        """Test turn detection functionality."""
        from infretis.classes.staple_path import turn_detected
        
        # Create phasepoints with a turn
        phasepoints = []
        orders = [0.1, 0.3, 0.5, 0.3, 0.1]  # Turn pattern
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"turn_frame_{i}.xyz", i)
            phasepoints.append(system)
        
        interfaces = [0.2, 0.4]
        m_idx = 0  # Middle interface index
        lr = 1  # Right turn
        
        # Convert phasepoints to orders array
        orders = np.array([pp.order[0] for pp in phasepoints])
        result = turn_detected(orders, interfaces, m_idx, lr)
        assert result  # Should detect the turn

    def test_turn_detected_no_turn(self):
        """Test turn detection with no turn present."""
        from infretis.classes.staple_path import turn_detected
        
        # Create monotonic phasepoints
        phasepoints = []
        orders = [0.25, 0.26, 0.27, 0.28, 0.29]  # Monotonic within interface region
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"mono_frame_{i}.xyz", i)
            phasepoints.append(system)
        
        interfaces = [0.2, 0.4]
        m_idx = 0
        lr = 1
        
        # Convert phasepoints to orders array
        orders = np.array([pp.order[0] for pp in phasepoints])
        result = turn_detected(orders, interfaces, m_idx, lr)
        assert not result  # Should not detect turn

    def test_staple_path_integration(self):
        """Test integration between StaplePath and turn detection."""
        path = StaplePath()
        
        # Create path with turns
        orders = [0.05, 0.25, 0.45, 0.25, 0.05]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"integration_{i}.xyz", i)
            path.append(system)
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        start_info, end_info, overall_valid = path.check_turns(interfaces)
        
        # Both turns should be valid
        assert start_info[0]
        assert end_info[0]
        assert overall_valid


class TestEngineWithRealStaplePaths:
    """Test engine methods with actual StaplePath objects."""
    
    def test_engine_dump_config_with_staple_path(self):
        """Test that engine can handle StaplePath configurations."""
        engine = MockStapleEngine()
        
        # Create a staple path system
        system = System()
        system.config = ("test_config.xyz", 0)
        
        # Mock the file operations
        with patch.object(engine, '_extract_frame') as mock_extract:
            with patch.object(engine, '_copyfile') as mock_copy:
                result = engine.dump_config(system.config, deffnm="test_dump")
                
                # Should attempt to extract frame
                mock_extract.assert_called_once()
                assert "test_dump" in result

    def test_engine_clean_up_staple_files(self):
        """Test engine cleanup with staple-related files."""
        engine = MockStapleEngine()
        engine.exe_dir = "/tmp/staple_test"
        
        with patch('os.scandir') as mock_scandir:
            with patch.object(engine, '_remove_files') as mock_remove:
                # Mock file listing
                mock_file = Mock()
                mock_file.name = "staple_trajectory.traj"
                mock_file.is_file.return_value = True
                mock_scandir.return_value = [mock_file]
                
                engine.clean_up()
                
                mock_remove.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
