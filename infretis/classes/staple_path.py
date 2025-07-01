"""Define the path class."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from infretis.classes.path import Path, DEFAULT_MAXLEN
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


class StaplePath(Path):
    """Path class that supports turns and restricted shooting point selection."""
    
    def __init__(self, maxlen: int = DEFAULT_MAXLEN, time_origin: int = 0):
        """Initialize a turn path."""
        super().__init__(maxlen, time_origin)
        self.sh_region: Tuple[int] = None  # Indices where turns occur
        self.meta: Dict[str, Any] = {"pptype": str, "dir": 0, }

    def check_start_turn(self, interfaces: List[float]) -> Tuple[bool, int, int]:
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
        if self.length < 2:
            return False, None, None
            
        intfs_min, intfs_max = min(interfaces), max(interfaces)
        start_op = self.phasepoints[0].order[0]
        
        # Check if already outside interface boundaries - automatically valid turn
        if start_op <= intfs_min:
            return True, 0, 0
        elif start_op >= intfs_max:
            return True, len(interfaces)-1, 0
            
        next_val = self.phasepoints[1].order[0]
        start_increasing = start_op < next_val
        max_deviation = start_op
        extremal_idx = 0
        
        # Pre-identify the first interface crossed by the initial segment
        initial_interface_idx = -1
        for i in range(len(interfaces)):
            if start_increasing:
                # Moving right, check interfaces from right to left
                if start_op < interfaces[-1-i] <= next_val:
                    initial_interface_idx = len(interfaces) - 1 - i
                    break
            else:
                # Moving left, check interfaces from left to right
                if next_val <= interfaces[i] < start_op:
                    initial_interface_idx = i
                    break
        
        # Scan forward from start to find valid turn
        for idx in range(1, self.length):
            current_val = self.phasepoints[idx].order[0]
            
            # Update maximum deviation from start point and track extremal position
            if (start_increasing and current_val > max_deviation) or \
               (not start_increasing and current_val < max_deviation):
                max_deviation = current_val
                extremal_idx = idx
            
            # Check if we've recrossed the initial interface (turn completed)
            if initial_interface_idx != -1:
                initial_interface = interfaces[initial_interface_idx]
                recrossed = (
                    (start_increasing and current_val <= initial_interface < max_deviation) or 
                    (not start_increasing and current_val >= initial_interface > max_deviation)
                )
                
                if recrossed:
                    # Check if at least 2 interfaces are crossed
                    min_val, max_val = min(start_op, max_deviation), max(start_op, max_deviation)
                    interfaces_crossed = np.sum((interfaces > min_val) & (interfaces < max_val))
                    
                    if interfaces_crossed >= 2:
                        return True, initial_interface_idx, extremal_idx
        
        return False, None, None

    def check_end_turn(self, interfaces: List[float]) -> Tuple[bool, int, int]:
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
        if self.length < 2:
            return False, None, None
            
        intfs_min, intfs_max = min(interfaces), max(interfaces)
        end_op = self.phasepoints[-1].order[0]
        
        # Check if already outside interface boundaries - automatically valid turn
        if end_op <= intfs_min or end_op >= intfs_max:
            return True, -1, self.length - 1
            
        prev_val = self.phasepoints[-2].order[0]
        end_increasing = end_op < prev_val  # Note: direction is relative to backward scan
        max_deviation = end_op
        extremal_idx = self.length - 1
        
        # Pre-identify the first interface crossed by the final segment (scanning backward)
        initial_interface_idx = -1
        for i in range(len(interfaces)):
            if end_increasing:
                # End is lower than previous, so we crossed from high to low
                if prev_val > interfaces[-1-i] >= end_op:
                    initial_interface_idx = len(interfaces) - 1 - i
                    break
            else:
                # End is higher than previous, so we crossed from low to high
                if end_op >= interfaces[i] > prev_val:
                    initial_interface_idx = i
                    break
        
        # Scan backward from end to find valid turn
        for idx in range(self.length-2, -1, -1):
            current_val = self.phasepoints[idx].order[0]
            
            # Update maximum deviation from end point and track extremal position
            if (end_increasing and current_val > max_deviation) or \
               (not end_increasing and current_val < max_deviation):
                max_deviation = current_val
                extremal_idx = idx
            
            # Check if we've recrossed the initial interface (turn completed)
            if initial_interface_idx != -1:
                initial_interface = interfaces[initial_interface_idx]
                recrossed = (
                    (end_increasing and current_val <= initial_interface < max_deviation) or 
                    (not end_increasing and current_val >= initial_interface > max_deviation)
                )
                
                if recrossed:
                    # Check if at least 2 interfaces are crossed
                    min_val, max_val = min(end_op, max_deviation), max(end_op, max_deviation)
                    interfaces_crossed = np.sum((interfaces > min_val) & (interfaces < max_val))
                    
                    if interfaces_crossed >= 2:
                        return True, initial_interface_idx, extremal_idx
        
        return False, None, None

    def check_turns(
        self, interfaces: List[float]
    ) -> Tuple[
        Tuple[bool, Optional[int], Optional[int]],  # start_turn_info
        Tuple[bool, Optional[int], Optional[int]],  # end_turn_info
        bool  # overall_valid
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
        if self.length < 1 or interfaces is None or len(interfaces) == 0:
            logger.warning("Path is too short or interfaces are not well defined.")
            return (False, None, None), (False, None, None), False
            
        # Check individual turns
        start_turn, start_interface_idx, start_extremal_idx = self.check_start_turn(interfaces)
        end_turn, end_interface_idx, end_extremal_idx = self.check_end_turn(interfaces)
        
        # Path is valid only if both turns meet criteria
        valid = start_turn and end_turn
        
        # Check for turn overlap - turns can't be the same turn unless they're both at extremes
        if valid and self.length > 2:
            intfs_min, intfs_max = min(interfaces), max(interfaces)
            start_op = self.phasepoints[0].order[0]
            end_op = self.phasepoints[-1].order[0]
            
            # If both endpoints are on the same extreme side, overlapping is allowed
            both_min = (start_op <= intfs_min and end_op <= intfs_min)
            both_max = (start_op >= intfs_max and end_op >= intfs_max)
            
            if not (both_min or both_max):
                # Otherwise ensure they're not the same turn
                valid = not (start_interface_idx == end_interface_idx and 
                           start_extremal_idx == end_extremal_idx and 
                           start_extremal_idx is not None and end_extremal_idx is not None)
        
        start_turn_info = (start_turn, start_interface_idx, start_extremal_idx)
        end_turn_info = (end_turn, end_interface_idx, end_extremal_idx)
        
        return start_turn_info, end_turn_info, valid
    
    def get_shooting_point(self, rgen: Generator) -> Tuple[System, int]:
        """Pick a random shooting point from the path."""
        ### TODO: probably need an unittest for this to check if correct.
        ### idx = rgen.random_integers(1, self.length - 2)
        if self.sh_region is None:
            logger.warning("No shooting region defined, cannot select a shooting point.")
            raise ValueError("Shooting region is not defined.")
        idx = rgen.integers(self.sh_region[0], self.sh_region[1], endpoint=True)
        order = self.phasepoints[idx].order[0]
        logger.debug(f"Selected point with orderp {order}")
        return self.phasepoints[idx], idx

    def get_pp_path(self, ens: Dict) -> Path:
        """Return the (RE)PPTIS part of the staple path."""
        new_path = self.empty_path(maxlen=self.maxlen)
        interfaces = ens["all_intfs"]
        pp_intfs = ens["interfaces"]
        if len(pp_intfs) != 3:
            logger.warning(
                "Path does not have 3 interfaces, cannot extract (RE)PPTIS part."
            )
            return None

        #TODO: add [0-] case
        if self.sh_region is None or len(self.sh_region) != 2:
            # TODO: add helper function
            logger.debug("No shooting region defined, inducing sh_region from path.")
            start_info, end_info, valid = self.check_turns(interfaces)
            if not valid:
                logger.warning("Path does not have valid turns, cannot extract (RE)PPTIS part.")
                return None
            new_path.status = self.status
            new_path.generated = self.generated
            new_path.maxlen = self.maxlen
            new_path.path_number = self.path_number
            new_path.weights = self.weights
            
            valid_pp = False
            while not valid_pp:
                left_border = next(self.phasepoints[start_info[2] + p].order[0] for p in range(end_info[2] - start_info[2] + 1) if self.phasepoints[start_info[2] + p].order[0] >= pp_intfs[0])
                right_border = next(self.phasepoints[left_border + p].order[0] for p in range(end_info[2] - left_border + 1) if pp_intfs[2] <= self.phasepoints[left_border + p].order[0] <= pp_intfs[0])
                valid_pp = next((True for p in range(1, right_border - left_border + 1) if np.sign(self.phasepoints[left_border + p] - pp_intfs[1]) != np.sign(self.phasepoints[left_border + p - 1] - pp_intfs[1])), False)
            
        else:
            left_border, right_border = self.sh_region

        for phasep in self.phasepoints[left_border-1, right_border+2]:
            new_path.append(phasep.copy())
        return new_path

    # def get_shooting_point(self, ens: Dict, rgen: Generator,
    #                        ) -> Tuple[System, int]:
    #     """Pick a random shooting point from the path."""
    #     ### TODO: probably need an unittest for this to check if correct.
    #     ### idx = rgen.random_integers(1, self.length - 2)
    #     ppath = self.get_pp_path(ens)

    #     idx = rgen.integers(1, ppath.length - 1)
    #     order = ppath.phasepoints[idx].order[0]
    #     logger.debug(f"Selected point with orderp {order}")
    #     return ppath.phasepoints[idx], self.sh_region[0] + idx - 1

    def copy(self) -> Path:
        """Return a copy of this path."""
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
        new_path.meta = self.meta
        return new_path
    
    def empty_path(self, maxlen=DEFAULT_MAXLEN, **kwargs) -> StaplePath:
        """Return an empty path of same class as the current one."""
        time_origin = kwargs.get("time_origin", 0)
        return self.__class__(maxlen=maxlen, time_origin=time_origin)

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
                "meta",
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

def turn_detected(phasepoints: List[System], interfaces: List[float], m_idx: int, lr: int) -> bool:
    """Check if a turn is detected in the given phasepoints.

    A turn is detected if the trajectory crosses at least two interfaces
    in one direction and recrosses back past the initial interface.

    Args:
        phasepoints: List of phasepoints representing the trajectory.
        interfaces: List of interface positions in ascending order.
        lr: Location indicator (-1 for left, 1 for right).

    Returns:
        True if a turn is detected, False otherwise.
    """
    if not phasepoints or not interfaces or lr is None:
        logger.warning("Invalid input for turn detection.")
        raise ValueError("Invalid input for turn detection.")
    
    if len(phasepoints) < 2:
        return False

    start_op = phasepoints[0].order[0]
    end_op = phasepoints[-1].order[0]
    ops = [phasepoints[i].order for i in range(len(phasepoints))]

    # Check if already outside interface boundaries
    if start_op <= interfaces[0] or end_op >= interfaces[-1] or lr*end_op <= lr*interfaces[m_idx]:
        return True
    
    # Find extreme value in the appropriate direction
    extr_op = max(ops) if lr==1 else min(ops)
    
    # Find eligible interfaces for detecting turns
    elig_intfs = [int for int in interfaces if lr*int <= lr*extr_op[0]]
    # extr_idx = intfs.index(elig_intfs[np.abs(elig_intfs - extr_op).argmin()])
    # cond_intf = elig_intfs[np.abs(elig_intfs - extr_op).argmin()-lr]
    cond_intf = elig_intfs[1 if lr==-1 else -2]
    
    # If we're still at the extreme point, no turn detected
    if ops[-1] == extr_op or extr_op is None:
        return False

    # extr_intf = intfs[extr_idx]
    ops_elig = np.array([op[0] for op in ops[ops.index(extr_op):]])
    
    # Check if any of these points cross back over the condition interface
    return np.any(lr*ops_elig <= lr*cond_intf)

    
def paste_paths(
    path_back: Path,
    path_forw: Path,
    overlap: bool = True,
    maxlen: Optional[int] = None,
) -> Path:
    """Merge a backward with a forward path into a new path.

    The resulting path is equal to the two paths stacked, in correct
    time. Note that the ordering is important here so that:
    ``paste_paths(path1, path2) != paste_paths(path2, path1)``.

    There are two things we need to take care of here:

    - `path_back` must be iterated in reverse (it is assumed to be a
      backward trajectory).
    - we may have to remove one point in `path_forw` (if the paths overlap).

    Args:
        path_back: The backward trajectory.
        path_forw: The forward trajectory.
        overlap: If True, `path_back` and `path_forw` have a common
            starting-point; the first point in `path_forw` is
            identical to the first point in `path_back`. In time-space, this
            means that the *first* point in `path_forw` is identical to the
            *last* point in `path_back` (the backward and forward path
            started at the same location in space).
        maxlen: This is the maximum length for the new path.
            If it's not given, it will just be set to the largest of
            the `maxlen` of the two given paths.

    Returns:
        The resulting path from the merge.

    Note:
        Some information about the path will not be set here. This must be
        set elsewhere. This includes how the path was generated
        (`path.generated`) and the status of the path (`path.status`).
    """
    if maxlen is None:
        if path_back.maxlen == path_forw.maxlen:
            maxlen = path_back.maxlen
        else:
            # They are unequal and both is not None, just pick the largest.
            # In case one is None, the other will be picked.
            # Note that now there is a chance of truncating the path while
            # pasting!
            maxlen = max(path_back.maxlen, path_forw.maxlen)
            msg = f"Unequal length: Using {maxlen} for the new path!"
            logger.warning(msg)
    time_origin = path_back.time_origin - path_back.length + 1
    new_path = path_back.empty_path(maxlen=maxlen, time_origin=time_origin)
    for phasepoint in reversed(path_back.phasepoints):
        app = new_path.append(phasepoint)
        if not app:
            msg = "Truncated while pasting backwards at: {}"
            msg = msg.format(new_path.length)
            logger.warning(msg)
            return new_path
    first = True
    for phasepoint in path_forw.phasepoints:
        if first and overlap:
            first = False
            continue
        app = new_path.append(phasepoint)
        if not app:
            msg = f"Truncated path at: {new_path.length}"
            logger.warning(msg)
            return new_path
    return new_path


def load_path(pdir: str) -> Path:
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

    path = Path()
    for snapshot, order in zip(traj["data"], orderdata):
        frame = System()
        frame.order = order
        frame.config = (snapshot[1], snapshot[2])
        frame.vel_rev = snapshot[3]
        path.phasepoints.append(frame)
    _load_energies_for_path(path, pdir)
    # TODO: CHECK PATH SOMEWHERE .acc, sta = _check_path(path, path_ensemble)
    return path


def _load_energies_for_path(path: Path, dirname: str) -> None:
    """Load energy data for a path.

    Args:
        path: The path we are to set up/fill.
        dirname: The path to the directory with the input files.
    """
    energy_file_name = os.path.join(dirname, "energy.txt")
    try:
        with EnergyPathFile(energy_file_name, "r") as energyfile:
            energy = next(energyfile.load())
            path.update_energies(
                energy["data"]["ekin"], energy["data"]["vpot"]
            )
    except FileNotFoundError:
        pass


def load_paths_from_disk(config: Dict[str, Any]) -> List[Path]:
    """Load paths from disk."""
    load_dir = config["simulation"]["load_dir"]
    paths = []
    for pnumber in config["current"]["active"]:
        new_path = load_path(os.path.join(load_dir, str(pnumber)))
        status = "re" if "restarted_from" in config["current"] else "ld"
        ### TODO: important for shooting move if 'ld' is set. need a smart way
        ### to remember if status is 'sh' or 'wf' etc. maybe in the toml file.
        new_path.generated = (status, float("nan"), 0, 0)
        new_path.maxlen = config["simulation"]["tis_set"]["maxlength"]
        paths.append(new_path)
        # assign pnumber
        paths[-1].path_number = pnumber
    return paths
