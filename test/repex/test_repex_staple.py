"""Test methods from infretis.classes.repex_staple"""
import numpy as np
import pytest

from infretis.classes.repex_staple import REPEX_state_staple
from infretis.classes.staple_path import StaplePath
from infretis.classes.system import System


class TestREPEXStateStaple:
    """Test the REPEX_state_staple class."""

    @pytest.fixture
    def basic_config(self):
        """Provide a basic configuration for testing."""
        return {
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
                "shooting_moves": ["st_sh", "st_sh", "st_sh"],
                "mode": "staple"
            },
            "output": {"data_dir": ".", "pattern": False}
        }

    @pytest.fixture
    def sample_staple_path(self):
        """Create a sample staple path for testing."""
        path = StaplePath()
        # Create a path with a turn: 0.05 -> 0.35 -> 0.05
        orders = [0.05, 0.15, 0.25, 0.35, 0.25, 0.15, 0.05]
        for i, order in enumerate(orders):
            system = System()
            system.order = [order]
            system.config = (f"test_frame_{i}.xyz", i)
            path.append(system)
        
        path.path_number = 1
        path.status = "ACC"
        path.weights = (0.0, 1.0, 0.0, 0.0)  # Weight in ensemble 1
        return path

    def test_init(self, basic_config):
        """Test REPEX_state_staple initialization."""
        state = REPEX_state_staple(basic_config, minus=False)
        
        assert state.n == basic_config["current"]["size"] + 1  # +1 for ghost ensemble
        assert state._offset == 1  # Always has offset for minus ensemble
        assert hasattr(state, '_last_prob')

    def test_prob_property(self, basic_config):
        """Test the probability matrix property."""
        state = REPEX_state_staple(basic_config, minus=False)
        
        prob_matrix = state.prob
        
        # Should be identity matrix with ghost ensemble having 0 probability
        assert prob_matrix.shape == (state.n, state.n)
        assert prob_matrix[-1, -1] == 0.0  # Ghost ensemble
        
        # Check diagonal elements (except ghost)
        for i in range(state.n - 1):
            assert prob_matrix[i, i] == 1.0

    def test_add_traj(self, basic_config, sample_staple_path):
        """Test adding trajectory to the state."""
        state = REPEX_state_staple(basic_config, minus=False)
        
        # Add trajectory to ensemble 0 (which becomes index 1 due to offset)
        ens = 0
        traj = sample_staple_path
        valid = traj.weights
        
        state.add_traj(ens, traj, valid)
        
        # Check that trajectory was added
        assert state._trajs[ens + state._offset] == traj
        
        # Check that valid array was modified (should be binary for staple)
        expected_valid = np.zeros(state.n)
        expected_valid[ens + 1] = 1.0
        np.testing.assert_array_equal(state.state[ens + state._offset, :], expected_valid)

    def test_add_traj_ignores_input_valid(self, basic_config, sample_staple_path):
        """Test that add_traj ignores input valid array and creates binary."""
        state = REPEX_state_staple(basic_config, minus=False)
        
        # Provide complex valid array
        complex_valid = (0.2, 0.5, 0.3, 0.0)
        ens = 1
        traj = sample_staple_path
        
        state.add_traj(ens, traj, complex_valid)
        
        # Should ignore complex_valid and create binary
        expected_valid = np.zeros(state.n)
        expected_valid[ens + 1] = 1.0
        np.testing.assert_array_equal(state.state[ens + state._offset, :], expected_valid)

    def test_print_state(self, basic_config, sample_staple_path, caplog):
        """Test the print_state method."""
        state = REPEX_state_staple(basic_config, minus=False)
        
        # Add some trajectories
        for i in range(2):
            traj = sample_staple_path.copy()
            traj.path_number = i
            state.add_traj(i, traj, traj.weights)
            state.traj_data[i] = {
                "max_op": [0.35],
                "min_op": [0.05],
                "length": 7,
                "frac": np.zeros(state.n)
            }
        
        # Test print_state
        with caplog.at_level("INFO"):
            state.print_state()
        
        # Check that logging occurred
        assert "Ensemble numbers" in caplog.text or "max_op" in caplog.text

    def test_print_shooted(self, basic_config, sample_staple_path, caplog):
        """Test the print_shooted method."""
        state = REPEX_state_staple(basic_config, minus=False)
        
        # Mock md_items
        md_items = {
            "moves": ["st_sh"],
            "ens_nums": [0],
            "pnum_old": [1],
            "trial_len": [7],
            "trial_op": [[0.05, 0.35]],
            "status": "ACC",
            "md_start": 0.0,
            "md_end": 1.0
        }
        state.cworker = 0
        
        pn_news = [2]
        
        with caplog.at_level("INFO"):
            state.print_shooted(md_items, pn_news)
        
        # Check that shooting information was logged
        assert "shooted" in caplog.text

    def test_print_start(self, basic_config, sample_staple_path, caplog):
        """Test the print_start method."""
        state = REPEX_state_staple(basic_config, minus=False)
        
        # Add some trajectories
        for i in range(2):
            traj = sample_staple_path.copy()
            traj.path_number = i
            state.add_traj(i, traj, traj.weights)
            state.traj_data[i] = {
                "max_op": [0.35],
                "min_op": [0.05],
                "length": 7,
                "frac": np.zeros(state.n)
            }
        
        with caplog.at_level("INFO"):
            state.print_start()
        
        # Check that start information was logged
        assert "stored ensemble paths" in caplog.text

    def test_print_end(self, basic_config, sample_staple_path, caplog):
        """Test the print_end method."""
        state = REPEX_state_staple(basic_config, minus=False)
        state.cstep = 10
        
        # Add trajectory data
        state.traj_data[0] = {
            "frac": np.array([0.0, 0.5, 0.3, 0.2, 0.0])
        }
        state.traj_data[1] = {
            "frac": np.array([0.0, 0.2, 0.6, 0.2, 0.0])
        }
        
        with caplog.at_level("INFO"):
            state.print_end()
        
        # Check that end information was logged
        assert "live trajs" in caplog.text or "after" in caplog.text

    def test_load_paths(self, basic_config):
        """Test loading paths into the state."""
        state = REPEX_state_staple(basic_config, minus=False)
        state.config = basic_config
        state.mc_moves = ["st_sh", "st_sh", "st_sh"]
        state.cap = None
        
        # Create test paths
        paths = {}
        for i in range(3):
            path = StaplePath()
            path.path_number = i
            path.weights = (0.0, 1.0 if i == 1 else 0.0, 0.0, 0.0)
            # Add some phasepoints
            for j in range(5):
                system = System()
                system.order = [0.1 + j * 0.1]
                system.config = (f"frame_{j}.xyz", j)
                path.append(system)
            paths[i] = path
        
        # Mock ensembles
        state.ensembles = {
            0: {"interfaces": (float("-inf"), 0.1, 0.1)},
            1: {"interfaces": (0.1, 0.1, 0.3)},
            2: {"interfaces": (0.1, 0.3, 0.5)}
        }
        
        state.load_paths(paths)
        
        # Check that paths were loaded
        assert len(state.traj_data) == 3
        for i in range(3):
            assert i in state.traj_data
            assert state.traj_data[i]["length"] == 5

    def test_infinity_swap_functionality_disabled(self, basic_config):
        """Test that infinite swap functionality is disabled (identity matrix)."""
        state = REPEX_state_staple(basic_config, minus=False)
        
        prob_matrix = state.prob
        
        # Should be identity matrix (no swapping)
        expected = np.identity(state.n)
        expected[-1, -1] = 0.0  # Ghost ensemble
        
        np.testing.assert_array_equal(prob_matrix, expected)

    def test_locked_paths_handling(self, basic_config):
        """Test handling of locked paths."""
        state = REPEX_state_staple(basic_config, minus=False)
        
        # Test with no locks
        locks = np.array([0, 0, 0, 0])
        state._locks = locks
        
        prob_matrix = state.prob
        
        # All diagonal elements should be 1 (except ghost)
        for i in range(state.n - 1):
            assert prob_matrix[i, i] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
