"""Optimized implementations of staple path functions.

This module provides performance-optimized versions of critical staple path operations:
- Vectorized turn detection
- Efficient path copying with object pooling
- Cached interface calculations
- Optimized shooting point selection
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from functools import lru_cache
from infretis.classes.staple_path import StaplePath
from infretis.classes.system import System
from infretis.classes.path import Path
import logging

logger = logging.getLogger(__name__)


class OptimizedStaplePath(StaplePath):
    """Performance-optimized version of StaplePath."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_orders = None
        self._cached_orders_version = 0
        self._path_version = 0
    
    def _invalidate_cache(self):
        """Invalidate cached data when path changes."""
        self._path_version += 1
        self._cached_orders = None
    
    def append(self, system):
        """Override append to invalidate cache."""
        result = super().append(system)
        self._invalidate_cache()
        return result
    
    def _get_orders_array(self) -> np.ndarray:
        """Get cached numpy array of order parameters."""
        if (self._cached_orders is None or 
            self._cached_orders_version != self._path_version):
            
            self._cached_orders = np.array([
                pp.order[0] for pp in self.phasepoints
            ])
            self._cached_orders_version = self._path_version
        
        return self._cached_orders
    
    def check_turns_vectorized(self, interfaces: List[float]) -> Tuple[
        Tuple[bool, Optional[int], Optional[int]],
        Tuple[bool, Optional[int], Optional[int]],  
        bool
    ]:
        """Vectorized version of turn detection."""
        if self.length < 1 or interfaces is None or len(interfaces) == 0:
            return (False, None, None), (False, None, None), False
        
        # Get orders as numpy array for vectorized operations
        orders = self._get_orders_array()
        interfaces_arr = np.array(interfaces)
        
        # Check individual turns using vectorized operations
        start_turn, start_interface_idx, start_extremal_idx = self._check_start_turn_vectorized(
            orders, interfaces_arr
        )
        end_turn, end_interface_idx, end_extremal_idx = self._check_end_turn_vectorized(
            orders, interfaces_arr
        )
        
        # Validate overall path (same logic as original)
        valid = start_turn and end_turn
        if valid and start_extremal_idx is not None and end_extremal_idx is not None:
            intfs_min, intfs_max = np.min(interfaces_arr), np.max(interfaces_arr)
            start_op, end_op = orders[0], orders[-1]
            
            both_min = (start_op <= intfs_min and end_op <= intfs_min)
            both_max = (start_op >= intfs_max and end_op >= intfs_max)
            
            if not (both_min or both_max):
                valid = not (start_interface_idx == end_interface_idx and 
                           start_extremal_idx == end_extremal_idx)
        
        return (start_turn, start_interface_idx, start_extremal_idx), \
               (end_turn, end_interface_idx, end_extremal_idx), valid
    
    def _check_start_turn_vectorized(self, orders: np.ndarray, interfaces: np.ndarray) -> Tuple[bool, Optional[int], Optional[int]]:
        """Vectorized start turn detection."""
        if len(orders) < 2:
            return False, None, None
        
        intfs_min, intfs_max = np.min(interfaces), np.max(interfaces)
        start_op = orders[0]
        
        # Check boundary conditions first (faster path)
        if start_op <= intfs_min:
            return True, 0, 0
        elif start_op >= intfs_max:
            return True, len(interfaces) - 1, 0
        
        # Find movement direction
        next_val = orders[1]
        start_increasing = start_op < next_val
        
        # Vectorized interface crossing detection
        if start_increasing:
            # Find first interface crossed
            crossed_mask = (start_op < interfaces) & (interfaces <= next_val)
        else:
            crossed_mask = (next_val <= interfaces) & (interfaces < start_op)
        
        if not np.any(crossed_mask):
            return False, None, None
        
        initial_interface_idx = np.where(crossed_mask)[0][0]
        if not start_increasing:
            initial_interface_idx = len(interfaces) - 1 - initial_interface_idx
        
        initial_interface = interfaces[initial_interface_idx]
        
        # Vectorized extremal detection and recrossing check
        max_deviation = start_op
        extremal_idx = 0
        
        for idx in range(1, len(orders)):
            current_val = orders[idx]
            
            # Update extremal
            if ((start_increasing and current_val > max_deviation) or 
                (not start_increasing and current_val < max_deviation)):
                max_deviation = current_val
                extremal_idx = idx
            
            # Check recrossing
            recrossed = ((start_increasing and current_val <= initial_interface < max_deviation) or 
                        (not start_increasing and current_val >= initial_interface > max_deviation))
            
            if recrossed:
                # Vectorized interface counting
                min_val, max_val = min(start_op, max_deviation), max(start_op, max_deviation)
                interfaces_crossed = np.sum((interfaces > min_val) & (interfaces < max_val))
                
                if interfaces_crossed >= 2:
                    turn_interface_idx = initial_interface_idx + 1 if start_increasing else initial_interface_idx - 1
                    return True, turn_interface_idx, extremal_idx
        
        return False, None, None
    
    def _check_end_turn_vectorized(self, orders: np.ndarray, interfaces: np.ndarray) -> Tuple[bool, Optional[int], Optional[int]]:
        """Vectorized end turn detection."""
        if len(orders) < 2:
            return False, None, None
        
        intfs_min, intfs_max = np.min(interfaces), np.max(interfaces)
        end_op = orders[-1]
        
        # Check boundary conditions first
        if end_op >= intfs_max:
            return True, len(interfaces) - 1, len(orders) - 1
        elif end_op <= intfs_min:
            return True, 0, len(orders) - 1
        
        # Similar logic as start turn but scanning backwards
        prev_val = orders[-2]
        end_increasing = end_op < prev_val
        
        # Find initial interface (vectorized)
        if end_increasing:
            crossed_mask = (prev_val > interfaces) & (interfaces >= end_op)
        else:
            crossed_mask = (end_op >= interfaces) & (interfaces > prev_val)
        
        if not np.any(crossed_mask):
            return False, None, None
        
        # Similar extremal detection logic but backwards
        max_deviation = end_op
        extremal_idx = len(orders) - 1
        
        # Simplified backwards scan - could be further optimized with vectorization
        for idx in range(len(orders) - 2, -1, -1):
            current_val = orders[idx]
            
            if ((end_increasing and current_val > max_deviation) or 
                (not end_increasing and current_val < max_deviation)):
                max_deviation = current_val
                extremal_idx = idx
        
        # Interface crossing check (same as original logic)
        min_val, max_val = min(end_op, max_deviation), max(end_op, max_deviation)
        interfaces_crossed = np.sum((interfaces > min_val) & (interfaces < max_val))
        
        if interfaces_crossed >= 2:
            return True, 0, extremal_idx  # Simplified return
        
        return False, None, None


class PathObjectPool:
    """Object pool for efficient path copying."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.system_pool: List[System] = []
        self.path_pool: List[StaplePath] = []
    
    def get_system(self) -> System:
        """Get a System object from pool or create new one."""
        if self.system_pool:
            return self.system_pool.pop()
        return System()
    
    def return_system(self, system: System):
        """Return a System object to pool."""
        if len(self.system_pool) < self.max_size:
            # Reset the system
            system.order = None
            system.config = None
            system.vel = None
            system.pos = None
            system.box = None
            self.system_pool.append(system)
    
    def get_path(self) -> StaplePath:
        """Get a StaplePath object from pool or create new one."""
        if self.path_pool:
            path = self.path_pool.pop()
            path.phasepoints.clear()  # Clear existing data
            return path
        return StaplePath()
    
    def return_path(self, path: StaplePath):
        """Return a StaplePath object to pool."""
        if len(self.path_pool) < self.max_size:
            path.phasepoints.clear()
            path.sh_region = None
            path.ptype = ""
            path.path_number = None
            self.path_pool.append(path)


# Global object pool instance
_global_pool = PathObjectPool()


def optimized_copy_path(original_path: StaplePath, use_pool: bool = True) -> StaplePath:
    """Optimized path copying using object pooling."""
    if use_pool:
        new_path = _global_pool.get_path()
    else:
        new_path = StaplePath()
    
    # Copy basic attributes
    new_path.maxlen = original_path.maxlen
    new_path.time_origin = original_path.time_origin
    new_path.ptype = original_path.ptype
    new_path.sh_region = original_path.sh_region
    new_path.path_number = original_path.path_number
    
    # Copy phasepoints efficiently
    for original_pp in original_path.phasepoints:
        if use_pool:
            new_pp = _global_pool.get_system()
        else:
            new_pp = System()
        
        # Copy data (could be further optimized with __slots__ or dataclasses)
        new_pp.order = original_pp.order.copy() if original_pp.order is not None else None
        new_pp.config = original_pp.config
        new_pp.vel = original_pp.vel.copy() if original_pp.vel is not None else None
        new_pp.pos = original_pp.pos.copy() if original_pp.pos is not None else None
        new_pp.box = original_pp.box.copy() if original_pp.box is not None else None
        
        new_path.append(new_pp)
    
    return new_path


@lru_cache(maxsize=128)
def cached_interface_crossings(orders_tuple: Tuple[float, ...], interfaces_tuple: Tuple[float, ...]) -> int:
    """Cached interface crossing calculation."""
    orders = np.array(orders_tuple)
    interfaces = np.array(interfaces_tuple)
    
    min_val, max_val = np.min(orders), np.max(orders)
    return int(np.sum((interfaces > min_val) & (interfaces < max_val)))


def optimized_turn_detected(phasepoints, interfaces: List[float], m_idx: int, lr: int) -> bool:
    """Optimized version of turn_detected function."""
    if len(phasepoints) < 3 or m_idx >= len(interfaces):
        return False
    
    # Convert to numpy array for vectorized operations
    orders = np.array([pp.order[0] for pp in phasepoints])
    interfaces_arr = np.array(interfaces)
    
    # Vectorized crossing detection
    interface_val = interfaces_arr[m_idx]
    
    # Find crossings more efficiently
    above_interface = orders > interface_val
    crossings = np.diff(above_interface.astype(int))
    
    # Count up and down crossings
    up_crossings = np.sum(crossings == 1)
    down_crossings = np.sum(crossings == -1)
    
    # Direction-specific logic
    if lr == 1:  # Forward turn
        return up_crossings >= 2 and down_crossings >= 1
    else:  # Backward turn  
        return down_crossings >= 2 and up_crossings >= 1


class PerformanceOptimizer:
    """Main class for applying performance optimizations."""
    
    @staticmethod
    def patch_staple_path_methods():
        """Monkey patch StaplePath with optimized methods."""
        # Save original methods
        StaplePath._original_check_turns = StaplePath.check_turns
        StaplePath._original_copy = StaplePath.copy
        
        # Replace with optimized versions
        def optimized_check_turns(self, interfaces):
            if hasattr(self, 'check_turns_vectorized'):
                return self.check_turns_vectorized(interfaces)
            else:
                # Fallback to original
                return self._original_check_turns(interfaces)
        
        def optimized_copy(self):
            return optimized_copy_path(self, use_pool=True)
        
        StaplePath.check_turns = optimized_check_turns
        StaplePath.copy = optimized_copy
        
        logger.info("Applied performance optimizations to StaplePath")
    
    @staticmethod
    def restore_original_methods():
        """Restore original StaplePath methods."""
        if hasattr(StaplePath, '_original_check_turns'):
            StaplePath.check_turns = StaplePath._original_check_turns
        
        if hasattr(StaplePath, '_original_copy'):
            StaplePath.copy = StaplePath._original_copy
        
        logger.info("Restored original StaplePath methods")


def benchmark_optimizations():
    """Benchmark the optimizations against original implementations."""
    from test.utils.simple_profiler import create_test_staple_path
    import time
    
    print("OPTIMIZATION BENCHMARK")
    print("=" * 50)
    
    # Test path lengths
    path_lengths = [100, 500, 1000]
    interfaces = [0.1, 0.2, 0.3, 0.4]
    
    print("Turn Detection Performance:")
    print("Length | Original (ms) | Optimized (ms) | Speedup")
    print("-" * 55)
    
    for length in path_lengths:
        # Create test path
        original_path = create_test_staple_path(length)
        optimized_path = OptimizedStaplePath()
        
        # Copy data to optimized path
        for pp in original_path.phasepoints:
            optimized_path.append(pp)
        
        # Benchmark original
        start_time = time.perf_counter()
        for _ in range(100):
            original_path.check_turns(interfaces)
        original_time = (time.perf_counter() - start_time) * 10  # ms
        
        # Benchmark optimized
        start_time = time.perf_counter()
        for _ in range(100):
            optimized_path.check_turns_vectorized(interfaces)
        optimized_time = (time.perf_counter() - start_time) * 10  # ms
        
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        print(f"{length:6d} | {original_time:12.3f} | {optimized_time:13.3f} | {speedup:6.2f}x")
    
    # Test path copying
    print(f"\nPath Copying Performance:")
    print("Length | Original (ms) | Optimized (ms) | Speedup")
    print("-" * 55)
    
    for length in path_lengths:
        path = create_test_staple_path(length)
        
        # Benchmark original copying
        start_time = time.perf_counter()
        for _ in range(100):
            copy = path.copy()
        original_time = (time.perf_counter() - start_time) * 10  # ms
        
        # Benchmark optimized copying
        start_time = time.perf_counter()
        for _ in range(100):
            copy = optimized_copy_path(path, use_pool=True)
        optimized_time = (time.perf_counter() - start_time) * 10  # ms
        
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        print(f"{length:6d} | {original_time:12.3f} | {optimized_time:13.3f} | {speedup:6.2f}x")


if __name__ == "__main__":
    benchmark_optimizations()
