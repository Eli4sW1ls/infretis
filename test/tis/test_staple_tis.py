"""Test staple-related methods from infretis.core.tis"""
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from infretis.classes.staple_path import StaplePath
from infretis.classes.system import System
from infretis.core.tis import staple_sh, staple_extender, staple_wf


class MockEngine:
    """Mock engine for testing."""
    
    def __init__(self):
        self.description = "Mock Engine"
        self.exe_dir = "/tmp"
        self.order_function = Mock()
    
    def propagate_st(self, path, ens_set, system, reverse=False):
        """Mock propagate_st method."""
        # Add some mock phasepoints to the path
        for i in range(5):
            mock_system = System()
            mock_system.order = [0.1 + i * 0.1]
            mock_system.config = (f"mock_frame_{i}.xyz", i)
            path.append(mock_system)
        return True, "ACC"
    
    def propagate(self, path, ens_set, system, reverse=False):
        """Mock propagate method."""
        for i in range(3):
            mock_system = System()
            mock_system.order = [0.2 + i * 0.1]
            mock_system.config = (f"mock_prop_{i}.xyz", i)
            path.append(mock_system)
        return True, "ACC"


class TestStapleSh:
    """Test the staple_sh function."""
    
    @pytest.fixture
    def mock_ens_set(self):
        """Provide mock ensemble settings."""
        return {
            "interfaces": [0.1, 0.3, 0.5],
            "all_intfs": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "tis_set": {
                "maxlength": 1000,
                "allowmaxlength": False
            },
            "ens_name": "test_ensemble",
            "rgen": np.random.default_rng(42)
        }
    
    @pytest.fixture
    def sample_path(self):
        """Create a sample staple path."""
        path = StaplePath()
        # Create path with shooting region
        orders = [0.05, 0.15, 0.25, 0.35, 0.25, 0.15, 0.05]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"frame_{i}.xyz", i)
            path.append(system)
        
        path.sh_region = (1, 5)  # Shooting region
        path.path_number = 1
        return path
    
    def test_staple_sh_basic(self, mock_ens_set, sample_path):
        """Test basic staple_sh functionality."""
        engine = MockEngine()
        
        with patch('infretis.core.tis.prepare_shooting_point') as mock_prep:
            with patch('infretis.core.tis.check_kick') as mock_kick:
                with patch('infretis.core.tis.shoot_backwards') as mock_back:
                    with patch('infretis.core.tis.paste_paths') as mock_paste:
                        # Setup mocks
                        mock_shooting_point = System()
                        mock_shooting_point.order = [0.25]
                        mock_prep.return_value = (mock_shooting_point, 3, [0.01, 0.01, 0.01])
                        mock_kick.return_value = True
                        mock_back.return_value = True
                        
                        # Mock paste_paths to return a proper path
                        result_path = StaplePath()
                        for i in range(10):
                            sys = System()
                            sys.order = [0.1 + i * 0.05]
                            sys.config = (f"result_{i}.xyz", i)
                            result_path.append(sys)
                        mock_paste.return_value = result_path
                        
                        success, trial_path, status = staple_sh(
                            mock_ens_set, sample_path, engine
                        )
                        
                        assert success
                        assert status == "ACC"
                        assert trial_path is not None

    def test_staple_sh_kick_failure(self, mock_ens_set, sample_path):
        """Test staple_sh with kick failure."""
        engine = MockEngine()
        
        with patch('infretis.core.tis.prepare_shooting_point') as mock_prep:
            with patch('infretis.core.tis.check_kick') as mock_kick:
                # Setup mocks
                mock_shooting_point = System()
                mock_shooting_point.order = [0.25]
                mock_prep.return_value = (mock_shooting_point, 3, [0.01, 0.01, 0.01])
                mock_kick.return_value = False  # Kick fails
                
                success, trial_path, status = staple_sh(
                    mock_ens_set, sample_path, engine
                )
                
                assert not success
                assert trial_path.status in ["REJ", "FTK"]

    def test_staple_sh_with_shooting_point(self, mock_ens_set, sample_path):
        """Test staple_sh with provided shooting point."""
        engine = MockEngine()
        shooting_point = System()
        shooting_point.order = [0.3]
        shooting_point.config = ("provided_frame.xyz", 0)
        
        with patch('infretis.core.tis.check_kick') as mock_kick:
            with patch('infretis.core.tis.shoot_backwards') as mock_back:
                with patch('infretis.core.tis.paste_paths') as mock_paste:
                    # Setup mocks
                    mock_kick.return_value = True
                    mock_back.return_value = True
                    
                    result_path = StaplePath()
                    for i in range(5):
                        sys = System()
                        sys.order = [0.2 + i * 0.05]
                        sys.config = (f"result_{i}.xyz", i)
                        result_path.append(sys)
                    mock_paste.return_value = result_path
                    
                    success, trial_path, status = staple_sh(
                        mock_ens_set, sample_path, engine, shooting_point
                    )
                    
                    assert success
                    assert trial_path is not None


class TestStapleExtender:
    """Test the staple_extender function."""
    
    @pytest.fixture
    def mock_ens_set(self):
        """Provide mock ensemble settings."""
        return {
            "interfaces": [0.1, 0.3, 0.5],
            "all_intfs": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "tis_set": {
                "maxlength": 1000
            },
            "ens_name": "test_ensemble",
            "rgen": np.random.default_rng(42)
        }
    
    @pytest.fixture
    def source_segment(self):
        """Create a source segment for testing."""
        path = StaplePath()
        # Create a short segment
        orders = [0.2, 0.25, 0.3]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"source_{i}.xyz", i)
            path.append(system)
        return path
    
    def test_staple_extender_lml(self, mock_ens_set, source_segment):
        """Test staple_extender with LML path type."""
        engine = MockEngine()
        partial_path_type = "LML"
        
        success, extended_path, status = staple_extender(
            source_segment, partial_path_type, engine, mock_ens_set
        )
        
        # Should extend the path
        assert success
        assert status == "ACC"
        assert extended_path.length > source_segment.length

    def test_staple_extender_rmr(self, mock_ens_set, source_segment):
        """Test staple_extender with RMR path type."""
        engine = MockEngine()
        partial_path_type = "RMR"
        
        success, extended_path, status = staple_extender(
            source_segment, partial_path_type, engine, mock_ens_set
        )
        
        # Should extend the path
        assert success
        assert status == "ACC"
        assert extended_path.length > source_segment.length

    def test_staple_extender_other_path_type(self, mock_ens_set, source_segment):
        """Test staple_extender with other path types (should return original)."""
        engine = MockEngine()
        partial_path_type = "LMR"  # Not LML or RMR
        
        success, extended_path, status = staple_extender(
            source_segment, partial_path_type, engine, mock_ens_set
        )
        
        # Should return original path without extension
        assert success
        assert status == "ACC"
        assert extended_path == source_segment

    def test_staple_extender_propagation_failure(self, mock_ens_set, source_segment):
        """Test staple_extender when propagation fails."""
        # Create engine that fails propagation
        engine = MockEngine()
        engine.propagate_st = Mock(return_value=(False, "FTL"))
        
        partial_path_type = "LML"
        
        success, extended_path, status = staple_extender(
            source_segment, partial_path_type, engine, mock_ens_set
        )
        
        assert not success
        assert status == "FTL"


class TestStapleWf:
    """Test the staple_wf function."""
    
    @pytest.fixture
    def mock_ens_set(self):
        """Provide mock ensemble settings."""
        return {
            "interfaces": [0.1, 0.3, 0.5],
            "all_intfs": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "tis_set": {
                "maxlength": 1000
            },
            "ens_name": "test_ensemble",
            "rgen": np.random.default_rng(42)
        }
    
    @pytest.fixture
    def sample_path(self):
        """Create a sample path for wire fencing."""
        path = StaplePath()
        # Create path that crosses interfaces
        orders = [0.05, 0.15, 0.25, 0.35, 0.45, 0.35, 0.25, 0.15, 0.05]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"wf_frame_{i}.xyz", i)
            path.append(system)
        return path
    
    def test_staple_wf_basic(self, mock_ens_set, sample_path):
        """Test basic staple_wf functionality."""
        engine = MockEngine()
        
        with patch('infretis.core.tis.wirefence_weight_and_pick') as mock_wf:
            with patch('infretis.core.tis.staple_extender') as mock_extend:
                # Setup mocks
                mock_segment = StaplePath()
                for i in range(3):
                    sys = System()
                    sys.order = [0.2 + i * 0.1]
                    sys.config = (f"wf_seg_{i}.xyz", i)
                    mock_segment.append(sys)
                
                mock_wf.return_value = (5, mock_segment)  # weight=5, segment
                
                extended_path = StaplePath()
                for i in range(8):
                    sys = System()
                    sys.order = [0.1 + i * 0.05]
                    sys.config = (f"extended_{i}.xyz", i)
                    extended_path.append(sys)
                mock_extend.return_value = (True, extended_path, "ACC")
                
                success, trial_path, status = staple_wf(
                    mock_ens_set, sample_path, engine
                )
                
                assert success
                assert status == "ACC"
                assert trial_path == extended_path

    def test_staple_wf_zero_weight(self, mock_ens_set, sample_path):
        """Test staple_wf with zero weight (should fail)."""
        engine = MockEngine()
        
        with patch('infretis.core.tis.wirefence_weight_and_pick') as mock_wf:
            # Return zero weight
            empty_segment = StaplePath()
            mock_wf.return_value = (0, empty_segment)
            
            success, trial_path, status = staple_wf(
                mock_ens_set, sample_path, engine
            )
            
            assert not success
            assert status == "NWF"  # No valid wire fence segment

    def test_staple_wf_extension_failure(self, mock_ens_set, sample_path):
        """Test staple_wf when extension fails."""
        engine = MockEngine()
        
        with patch('infretis.core.tis.wirefence_weight_and_pick') as mock_wf:
            with patch('infretis.core.tis.staple_extender') as mock_extend:
                # Setup mocks
                mock_segment = StaplePath()
                for i in range(3):
                    sys = System()
                    sys.order = [0.2 + i * 0.1]
                    sys.config = (f"wf_seg_{i}.xyz", i)
                    mock_segment.append(sys)
                
                mock_wf.return_value = (5, mock_segment)
                mock_extend.return_value = (False, mock_segment, "FTL")  # Extension fails
                
                success, trial_path, status = staple_wf(
                    mock_ens_set, sample_path, engine
                )
                
                assert not success
                assert status == "FTL"


class TestStapleUtilities:
    """Test utility functions related to staple paths."""
    
    def test_get_ptype_import(self):
        """Test that get_ptype function can be imported."""
        try:
            from infretis.classes.repex_staple import get_ptype
            assert callable(get_ptype)
        except ImportError:
            # Function might be defined elsewhere or not yet implemented
            pytest.skip("get_ptype function not found")
    
    def test_path_type_classification(self):
        """Test path type classification for staple paths."""
        # This would test the get_ptype function once it's available
        path = StaplePath()
        orders = [0.05, 0.3, 0.05]  # LML type path
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"frame_{i}.xyz", i)
            path.append(system)
        
        # Test that path has expected properties
        assert path.length == 3
        assert path.phasepoints[0].order[0] == 0.05
        assert path.phasepoints[1].order[0] == 0.3
        assert path.phasepoints[2].order[0] == 0.05


if __name__ == "__main__":
    pytest.main([__file__])
