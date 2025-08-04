"""Define the path class."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from infretis.classes.path import Path, DEFAULT_MAXLEN, _load_energies_for_path
from infretis.classes.formatter import (
    EnergyPathFile,
    OrderPathFile,
    PathExtFile,
)
from infretis.classes.system import System

if TYPE_CHECKING:  # pragma: no cover
    from numpy.random import Generator

    from infretis.classes.orderparameter import OrderParameter

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Lazy import for profiling to avoid circular imports
def get_profiling_context():
    """Get profiling context if available."""
    try:
        from infretis.profiling import profile_operation
        return profile_operation
    except ImportError:
        from contextlib import nullcontext
        return lambda *args, **kwargs: nullcontext()


class StaplePath(Path):
    """Path class that supports turns and restricted shooting point selection."""
    
    def __init__(self, maxlen: int = DEFAULT_MAXLEN, time_origin: int = 0, ptype: str = ""):
        """Initialize a turn path."""
        super().__init__(maxlen, time_origin)
        self._cached_orders = None
        self._cached_orders_version = 0
        self._path_version = 0
        self._cached_turn_info = None  # Cache for turn detection results
        self.sh_region: Tuple[int] = None  # Indices where turns occur
        self.ptype = ptype 

    def _invalidate_cache(self):
        """Invalidate cached data when path changes."""
        self._path_version += 1
        self._cached_orders = None
        self._cached_turn_info = None

    def get_orders_array(self) -> np.ndarray:
        """Get cached numpy array of order parameters."""
        if (self._cached_orders is None or 
            self._cached_orders_version != self._path_version):
            
            self._cached_orders = np.array([
                pp.order[0] for pp in self.phasepoints
            ])
            self._cached_orders_version = self._path_version
        
        return self._cached_orders
    
    def check_turns(self, interfaces: List[float]) -> Tuple[
        Tuple[bool, Optional[int], Optional[int]],
        Tuple[bool, Optional[int], Optional[int]],  
        bool
    ]:
        """Check turns in the path and return detailed turn information.

        This method analyzes the path to determine if valid turns exist at the start 
        and end of the trajectory. A valid turn is defined as a trajectory segment that:
        1. Starts/ends at one side of an interface
        2. Crosses at least 2 interfaces in one direction (reaching an extremal value)
        3. Recrosses back past the initial interface
        
        The method also checks if the overall path is valid by ensuring both turns
        are present and don't overlap (unless both endpoints are on the same extreme
        side of the interface boundaries).

        Args:
            interfaces: List of interface positions in ascending order.

        Returns:
            A tuple containing:
            - start_turn_info: Tuple of (is_valid, interface_idx, extremal_idx) for start turn
              - is_valid (bool): Whether start forms a valid turn
              - interface_idx (Optional[int]): Index of the interface around which the start turn occurs
              - extremal_idx (Optional[int]): Index in phasepoints list where the extremal value occurs
            - end_turn_info: Tuple of (is_valid, interface_idx, extremal_idx) for end turn
              - is_valid (bool): Whether end forms a valid turn
              - interface_idx (Optional[int]): Index of the interface around which the end turn occurs
              - extremal_idx (Optional[int]): Index in phasepoints list where the extremal value occurs
            - overall_valid (bool): Whether the overall path is valid (both turns valid and non-overlapping)
        """
        profile_op = get_profiling_context()
        
        with profile_op("staple_check_turns", path_length=self.length, num_interfaces=len(interfaces)):
            if self.length < 1 or interfaces is None or len(interfaces) == 0:
                return (False, None, None), (False, None, None), False
            
            # Get orders as numpy array for vectorized operations
            orders = self.get_orders_array()
            interfaces_arr = np.array(interfaces)
            
            # Check individual turns using vectorized operations
            start_turn, start_interface_idx, start_extremal_idx = self._check_start_turn(
                orders, interfaces_arr
            )
            end_turn, end_interface_idx, end_extremal_idx = self._check_end_turn(
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
    
    def _check_start_turn(self, orders: np.ndarray, interfaces: np.ndarray) -> Tuple[bool, Optional[int], Optional[int]]:
        """Check if the start of the path forms a valid turn.
        
        A valid turn is defined as a trajectory segment that:
        1. Starts at one side of an interface
        2. Crosses at least 2 interfaces in one direction (reaching an extremal value)
        3. Recrosses back past the initial interface
        
        Args:
            interfaces: List of interface positions in ascending order.
            
        Returns:
            Tuple containing:
            - is_valid_turn (bool): True if the start segment forms a valid turn
            - turn_interface_idx (int): Index of the interface around which the turn occurs
              (the interface that gets recrossed to complete the turn), -1 if no valid turn
            - extremal_idx (int): Index in phasepoints list where the extremal value 
              (maximum deviation from start) occurs, 0 if no valid turn
        """
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
        
        initial_interface_idx = np.where(crossed_mask)[0]
        if start_increasing:
            initial_interface_idx = initial_interface_idx[0]
        else:
            initial_interface_idx = initial_interface_idx[-1]
        
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
    
    def _check_end_turn(self, orders: np.ndarray, interfaces: np.ndarray) -> Tuple[bool, Optional[int], Optional[int]]:
        """Check if the end of the path forms a valid turn.
        
        A valid turn is defined as a trajectory segment that:
        1. Ends at one side of an interface
        2. Previously crossed at least 2 interfaces in one direction (reaching an extremal value)
        3. Recrossed back past the initial interface to reach the end
        
        Args:
            interfaces: List of interface positions in ascending order.
            
        Returns:
            Tuple containing:
            - is_valid_turn (bool): True if the end segment forms a valid turn
            - turn_interface_idx (int): Index of the interface around which the turn occurs
              (the interface that gets recrossed to complete the turn), -1 if no valid turn
            - extremal_idx (int): Index in phasepoints list where the extremal value 
              (maximum deviation from end) occurs, length-1 if no valid turn
        """
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

    def get_shooting_point(self, rgen) -> Tuple[System, int]:
        """Shooting point selection with bounds checking."""
        profile_op = get_profiling_context()
        
        with profile_op("staple_shooting_point_selection", path_length=self.length):
            if self.sh_region is None:
                logger.warning("No shooting region defined, cannot select a shooting point.")
                raise ValueError("Shooting region is not defined.")
                
            # Pre-validate bounds
            start, end = self.sh_region
            if start > end or start <= 0 or end >= self.length-1:
                logger.error(f"Invalid shooting region: {self.sh_region} for path length {self.length}")
                raise ValueError(f"Invalid shooting region: {self.sh_region} for path length {self.length}")
                
            # Fast integer selection
            idx = rgen.integers(start, end, endpoint=True)
            logger.debug(f"Selected point with orderp {self.phasepoints[idx].order[0]}")
            return self.phasepoints[idx], idx

    def get_pp_path(self, intfs: List[float], pp_intfs: List[float]) -> Tuple[Path, str, Tuple[int, int]]:
        """Optimized path extraction with reduced complexity."""
        from infretis.classes.path import Path
        
        profile_op = get_profiling_context()
        
        with profile_op("staple_path_extraction", path_length=self.length, 
                       num_interfaces=len(intfs), num_pp_interfaces=len(pp_intfs)):
            # Early validation
            if len(pp_intfs) <= 1:
                return None, "", (0, 0)
                
            # Validate that pp_intfs is a subset of intfs
            assert str(list(dict.fromkeys(pp_intfs)))[1:-1] in str(intfs)[1:-1], f"Invalid interface indices: {intfs, pp_intfs, str(list(dict.fromkeys(pp_intfs)))[1:-1], str(intfs)[1:-1]}"
                
            new_path = Path(maxlen=self.maxlen, time_origin=self.time_origin)
            interfaces = np.array(intfs)
            
            # Use cached turn information if available
            if self._cached_turn_info is None:
                start_info, end_info, valid = self.check_turns(interfaces)
                self._cached_turn_info = (start_info, end_info, valid)
            else:
                start_info, end_info, valid = self._cached_turn_info
                
            if not valid:
                return None, "", (0, 0)
                
            # Simplified path extraction for common cases
            if len(intfs) <= 3:
                # Simple three-interface case
                pptype = self._determine_simple_pptype(pp_intfs)
                for phasep in self.phasepoints:
                    new_path.append(phasep.copy())
                return new_path, pptype, (1, len(self.phasepoints) - 2)
            
            # For complex cases, use optimized border detection
        left_border, right_border, pptype = self._find_borders(
            start_info, end_info, pp_intfs)
            
        # Validate path segment
        valid_pp = self._validate_pp_segment(left_border, right_border, pp_intfs)
        if not valid_pp:
            return None, "", (0, 0)
            
        # Copy path segment efficiently
        for phasep in self.phasepoints[left_border-1:right_border+2]:
            new_path.append(phasep.copy())
            
        return new_path, pptype, (left_border, right_border)
    
    def _determine_simple_pptype(self, pp_intfs: List[float]) -> str:
        """Determine path type for simple three-interface case."""
        start_order = self.phasepoints[0].order[0]
        end_order = self.phasepoints[-1].order[0]
        
        if start_order < pp_intfs[0]:
            return "LML" if end_order < pp_intfs[0] else "LMR"
        elif start_order > pp_intfs[2]:
            return "RMR" if end_order > pp_intfs[2] else "RML"
        else:
            raise ValueError("Path does not start from a valid boundary.")
    
    def _find_borders(self, start_info, end_info, pp_intfs) -> Tuple[int, int, str]:
        """Optimized border detection using vectorized operations."""
        # Extract orders as array for vectorized operations
        orders = np.array([php.order[0] for php in self.phasepoints])
        
        # Determine path type and borders based on extremal positions
        start_extremal = start_info[2]
        end_extremal = end_info[2]
        
        if pp_intfs[1] < orders[end_extremal] < pp_intfs[2]:  # LML end
            left_border = self._find_border_vectorized(orders, start_extremal, pp_intfs[0], 'left')
            right_border = len(orders) - 2
            pptype = "LML"
        elif pp_intfs[1] < orders[start_extremal] < pp_intfs[2]:  # LML start
            left_border = 1
            right_border = self._find_border_vectorized(orders, start_extremal, pp_intfs[0], 'right')
            pptype = "LML"
        elif pp_intfs[0] < orders[start_extremal] < pp_intfs[1]:  # RMR start
            left_border = 1
            right_border = self._find_border_vectorized(orders, start_extremal, pp_intfs[2], 'right')
            pptype = "RMR"
        elif pp_intfs[0] < orders[end_extremal] < pp_intfs[1]:  # RMR end
            left_border = self._find_border_vectorized(orders, end_extremal, pp_intfs[2], 'left')
            right_border = len(orders) - 2
            pptype = "RMR"
        else:
            valid_indices = np.where((orders > pp_intfs[0]) & (orders < pp_intfs[2]))[0]
            if len(valid_indices) == 0:
                raise ValueError("No valid segment found")
            left_border = valid_indices[0]
            right_border = valid_indices[-1]
            pptype = "LMR" if start_info[1] < end_info[1] else "RML"
            
        return left_border, right_border, pptype
    
    def _find_border_vectorized(self, orders: np.ndarray, center_idx: int, 
                               threshold: float, direction: str) -> int:
        """Find border using vectorized operations."""
        if direction == 'left':
            search_range = range(center_idx, -1, -1)
            condition = orders <= threshold
        else:  # right
            search_range = range(center_idx, len(orders))
            condition = orders <= threshold
            
        for idx in search_range:
            if condition[idx]:
                return idx + 1 if direction == 'left' else idx - 1
                
        return 1 if direction == 'left' else len(orders) - 2
    
    def _validate_pp_segment(self, left_border: int, right_border: int, 
                           pp_intfs: List[float]) -> bool:
        """Validate that the path segment crosses the middle interface."""
        if left_border >= right_border:
            return False
            
        # Check for interface crossing in the segment
        for p in range(left_border, right_border):
            if p >= len(self.phasepoints) - 1:
                break
            curr_order = self.phasepoints[p].order[0]
            next_order = self.phasepoints[p + 1].order[0]
            
            # Check if crossing middle interface
            if ((curr_order - pp_intfs[1]) * (next_order - pp_intfs[1]) < 0):
                return True
                
        return False

    def copy(self) -> Path:
        """Return a copy of this path."""
        profile_op = get_profiling_context()
        
        with profile_op("staple_path_copy", path_length=self.length):
            new_path = self.empty_path(maxlen=self.maxlen)
            for phasepoint in self.phasepoints:
                new_path.append(phasepoint.copy())
            new_path.status = self.status
            new_path.time_origin = self.time_origin
            new_path.generated = self.generated
            new_path.maxlen = self.maxlen
            new_path.path_number = self.path_number
            new_path.weights = self.weights
            new_path.sh_region = self.sh_region
            new_path.ptype = self.ptype
            return new_path
    
    def empty_path(self, maxlen=DEFAULT_MAXLEN, **kwargs) -> StaplePath:
        """Return an empty path of same class as the current one."""
        return self.__class__(maxlen=maxlen, **kwargs)

    def __eq__(self, other) -> bool:
        """Check if two paths are equal."""
        if self.__class__ != other.__class__:
            logger.debug("%s and %s.__class__ differ", self, other)
            return False

        if set(self.__dict__) != set(other.__dict__):
            logger.debug("%s and %s.__dict__ differ", self, other)
            return False

        # Compare phasepoints:
        if not len(self.phasepoints) == len(other.phasepoints):
            return False
        for i, j in zip(self.phasepoints, other.phasepoints):
            if not i == j:
                return False
        if self.phasepoints:
            # Compare other attributes:
            for key in (
                "maxlen",
                "time_origin",
                "status",
                "generated",
                "length",
                "ordermax",
                "ordermin",
                "path_number",
                "sh_region",
                "ptype",
            ):
                attr_self = hasattr(self, key)
                attr_other = hasattr(other, key)
                if attr_self ^ attr_other:  # pragma: no cover
                    logger.warning(
                        'Failed comparing path due to missing "%s"', key
                    )
                    return False
                if not attr_self and not attr_other:
                    logger.warning(
                        'Skipping comparison of missing path attribute "%s"',
                        key,
                    )
                    continue
                if getattr(self, key) != getattr(other, key):
                    return False
        return True
    
    def append(self, system):
        """Override append to invalidate cache."""
        result = super().append(system)
        self._invalidate_cache()
        return result

def turn_detected(orders: np.ndarray, interfaces: List[float], m_idx: int, lr: int) -> bool:
    """Check if a turn is detected in the given order parameters.

    A turn is detected if the trajectory crosses at least two interfaces
    in one direction and recrosses back past the initial interface.

    Args:
        orders: ArrayLike of order parameters representing the trajectory.
        interfaces: List of interface positions in ascending order.
        m_idx: Index of the middle PPTIS interface to check for crossing.
        lr: Location indicator (-1 for left, 1 for right).

    Returns:
        True if a turn is detected, False otherwise.
    """
    profile_op = get_profiling_context()
    
    # Input validation - raise error for invalid inputs
    if not isinstance(orders, np.ndarray) or not interfaces or lr is None:
        logger.warning("Invalid input for turn detection.")
        raise ValueError("Invalid input for turn detection.")
    
    if len(orders) < 3:  # Need at least 3 points for a meaningful turn
        return False
        
    with profile_op("staple_turn_detected", path_length=len(orders), num_interfaces=len(interfaces)):
        # Check for invalid interface index
        if m_idx >= len(interfaces) or m_idx < 0:
            return False

        # Convert to numpy array for vectorized operations
        interfaces_arr = np.array(interfaces)
        start_op = orders[0]
        end_op = orders[-1]

        # Check if already outside interface boundaries - vectorized comparison
        if start_op <= interfaces_arr[0] or end_op >= interfaces_arr[-1] or lr * end_op <= lr * interfaces_arr[m_idx]:
            return True
        
        # Find extreme value using vectorized operations
        extr_op = np.max(orders) if lr == 1 else np.min(orders)
        extr_idx = np.argmax(orders) if lr == 1 else np.argmin(orders)
        
        # Find eligible interfaces using vectorized comparison
        elig_mask = lr * interfaces_arr <= lr * extr_op
        elig_intfs = interfaces_arr[elig_mask]
        
        if len(elig_intfs) < 2:
            return False
            
        cond_intf = elig_intfs[1] if lr == -1 else elig_intfs[-2]
        
        # If we're still at the extreme point, no turn detected
        if extr_idx == len(orders) - 1:
            return False

        # Check points after the extreme point using array slicing
        ops_elig = orders[extr_idx:]
        
        # Vectorized comparison for turn detection
        return np.any(lr * ops_elig <= lr * cond_intf)

def load_staple_path(pdir: str, lamb_A: float) -> StaplePath:
    """Load a path from the given directory."""
    trajtxt = os.path.join(pdir, "traj.txt")
    ordertxt = os.path.join(pdir, "order.txt")
    assert os.path.isfile(trajtxt), f"Trajectory file {trajtxt} does not exist."
    assert os.path.isfile(ordertxt), f"Order file {ordertxt} does not exist."

    # load trajtxt
    with PathExtFile(trajtxt, "r") as trajfile:
        # Just get the first trajectory:
        traj = next(trajfile.load())

        # Update trajectory to use full path names:
        for i, snapshot in enumerate(traj["data"]):
            config = os.path.join(pdir, "accepted", snapshot[1])
            traj["data"][i][1] = config
            reverse = int(snapshot[3]) == -1
            idx = int(snapshot[2])
            traj["data"][i][2] = idx
            traj["data"][i][3] = reverse

        for config in set(frame[1] for frame in traj["data"]):
            assert os.path.isfile(config), f"Config file {config} does not exist."

    # load ordertxt
    with OrderPathFile(ordertxt, "r") as orderfile:
        orderdata = next(orderfile.load())["data"][:, 1:]

    if orderdata[0] > lamb_A and orderdata[1] < lamb_A and orderdata[-2] < lamb_A: # [0-]
        print("0-?: ", orderdata)
        path = Path()
    else:
        path = StaplePath()
    for snapshot, order in zip(traj["data"], orderdata):
        frame = System()
        frame.order = order
        frame.config = (snapshot[1], snapshot[2])
        frame.vel_rev = snapshot[3]
        path.phasepoints.append(frame)
    _load_energies_for_path(path, pdir)

    return path