#!/usr/bin/env python3
"""
Test script for the visual validation functionality.
"""

import numpy as np
import sys
import os

# Add the infretis module to the path
sys.path.insert(0, os.path.abspath('.'))

def test_plotting_functionality():
    """Test the plotting functionality without full simulation."""
    
    print("Testing INFRETIS Visual Validation Functionality")
    print("=" * 60)
    
    try:
        from infretis.classes.repex_staple import REPEX_state_staple
        print("✓ Successfully imported REPEX_state_staple")
        
        # Create minimal config
        config = {
            'current': {
                'size': 5,
                'active': list(range(5)),
                'locked': []
            },
            'simulation': {
                'seed': 42,
                'ens_eff': [0] * 5,
                'interfaces': [-1.0, -0.5, 0.0, 0.5, 1.0]
            },
            'runner': {
                'workers': 1
            },
            'output': {}
        }
        
        # Create REPEX instance
        repex = REPEX_state_staple(config, minus=True)
        repex._offset = 1
        # interfaces is read from config, so no need to set it manually
        print("✓ Successfully created REPEX instance")
        
        # Create mock trajectory data for testing with ensemble-specific shooting region
        mock_traj_data = {
            'frac': np.zeros(5),
            'max_op': 1.2,
            'min_op': -0.8,
            'length': 100,
            'weights': np.random.random(100),
            'adress': ['test_file1.xyz', 'test_file2.xyz'],
            'ens_save_idx': 0,
            'ptype': 'ABA',
            'sh_region': {
                1: [15, 75],  # Shooting region for ensemble 1
                2: [20, 80],  # Shooting region for ensemble 2 (this will be displayed)
                3: [25, 85]   # Shooting region for ensemble 3
            }
        }
        
        # Create mock trajectory object
        class MockTrajectory:
            def __init__(self):
                self.path_number = 12345
                self.ordermax = 1.2
                self.ordermin = -0.8
                self.length = 100
                
            def get_orders_array(self):
                # Create a mock trajectory that crosses interfaces
                t = np.linspace(0, 2*np.pi, 100)
                return np.sin(t) + 0.1 * np.random.random(100)
        
        mock_traj = MockTrajectory()
        
        print("✓ Created mock trajectory data")
        
        # Test the detailed printing function
        print("\n📋 Testing detailed trajectory data printing...")
        repex.print_traj_data_detailed(mock_traj_data, 12345)
        
        # Test the plotting function
        print("\n📊 Testing path plotting (will open plot window)...")
        print("   Close the plot window to continue...")
        
        repex.plot_path_validation(mock_traj, mock_traj_data, ens_num=2, move_type="sh")
        
        print("\n✅ All tests completed successfully!")
        print("The visual validation functionality is ready to use.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_plotting_functionality()
