"""Utility functions for creating correct staple paths in tests.

This module provides utilities to create physically correct staple paths
that follow the proper staple path definition:

- A staple path can start either in region A, region B, or with a turn
- A turn is defined as the recrossing of the two subsequent interfaces 
  that were most recently crossed
- Once a turn is detected, the path is stopped
"""
import numpy as np
from infretis.classes.system import System


def create_staple_path_with_turn(interfaces, turn_at_interface_pair=None, start_region="A", extra_length=5, noise_level=0.0):
    """Create a correct staple path that includes a turn.
    
    Args:
        interfaces: List of interface values (should be sorted)
        turn_at_interface_pair: Tuple (i, i+1) indicating which interface pair to turn at
        start_region: "A" (below interface 0), "B" (above highest interface), or "turn" (start with turn)
        extra_length: Extra points to add variation
        noise_level: Amount of noise to add to trajectory
    
    Returns:
        List of order parameter values forming a correct staple path with turn
    """
    if turn_at_interface_pair is None:
        # Default to middle interface pair
        turn_idx = len(interfaces) // 2
        turn_at_interface_pair = (turn_idx, turn_idx + 1)
    
    turn_i, turn_j = turn_at_interface_pair
    
    if start_region == "A":
        # Start below interface 0, cross interfaces up to turn point, then turn back
        start_order = interfaces[0] - 0.05
        
        # Go up to the turn point (cross interfaces i and j)
        ascending_orders = []
        target_order = interfaces[turn_j] + 0.02  # Just above interface j
        
        for k in range(5 + extra_length // 2):
            progress = k / (4 + extra_length // 2)
            order = start_order + progress * (target_order - start_order)
            if noise_level > 0:
                order += np.random.normal(0, noise_level)
            ascending_orders.append(order)
        
        # Turn: recross interfaces j and i in reverse order
        turn_orders = []
        target_turn_order = interfaces[turn_i] - 0.02  # Just below interface i
        
        for k in range(1, 5 + extra_length - extra_length // 2):
            progress = k / (4 + extra_length - extra_length // 2) 
            order = target_order - progress * (target_order - target_turn_order)
            if noise_level > 0:
                order += np.random.normal(0, noise_level)
            turn_orders.append(order)
        
        # Ensure only one phase point after final interface crossing
        # Find the point that crosses the final interface and keep only one point after it
        final_orders = []
        for idx, order in enumerate(turn_orders):
            final_orders.append(order)
            # Check if we crossed the target interface (interface i)
            if len(final_orders) > 1 and ((turn_orders[idx-1] >= interfaces[turn_i]) != (order >= interfaces[turn_i])):
                # We crossed interface i, add exactly one more point and stop
                if idx + 1 < len(turn_orders):
                    final_orders.append(turn_orders[idx + 1])
                break
        
        return ascending_orders + final_orders
        
    elif start_region == "B":
        # Start above highest interface, cross down to turn point, then turn back
        start_order = interfaces[-1] + 0.05
        
        # Go down to the turn point (cross interfaces j and i)
        descending_orders = []
        target_order = interfaces[turn_i] - 0.02  # Just below interface i
        
        for k in range(5 + extra_length // 2):
            progress = k / (4 + extra_length // 2)
            order = start_order - progress * (start_order - target_order)
            if noise_level > 0:
                order += np.random.normal(0, noise_level)
            descending_orders.append(order)
        
        # Turn: recross interfaces i and j in reverse order
        turn_orders = []
        target_turn_order = interfaces[turn_j] + 0.02  # Just above interface j
        
        for k in range(1, 5 + extra_length - extra_length // 2):
            progress = k / (4 + extra_length - extra_length // 2)
            order = target_order + progress * (target_turn_order - target_order)
            if noise_level > 0:
                order += np.random.normal(0, noise_level)
            turn_orders.append(order)
        
        # Ensure only one phase point after final interface crossing
        # Find the point that crosses the final interface and keep only one point after it  
        final_orders = []
        for idx, order in enumerate(turn_orders):
            final_orders.append(order)
            # Check if we crossed the target interface (interface j)
            if len(final_orders) > 1 and ((turn_orders[idx-1] <= interfaces[turn_j]) != (order <= interfaces[turn_j])):
                # We crossed interface j, add exactly one more point and stop
                if idx + 1 < len(turn_orders):
                    final_orders.append(turn_orders[idx + 1])
                break
        
        return descending_orders + final_orders
        
    elif start_region == "turn":
        # Start with a turn (already between two interfaces)
        # This creates a complete staple path that starts with a turn and ends properly
        start_order = interfaces[turn_j] + 0.01
        
        # Initial turn: go down past interface i, then recross back past interface j
        down_orders = [start_order]
        target_down = interfaces[turn_i] - 0.02
        
        # Go down to create the initial turn 
        for k in range(1, 4 + extra_length // 4):
            progress = k / (3 + extra_length // 4)
            order = start_order - progress * (start_order - target_down)
            if noise_level > 0:
                order += np.random.normal(0, noise_level)
            down_orders.append(order)
        
        # Turn back: recross interfaces i and j
        turn_back_orders = []
        target_turn_back = interfaces[turn_j] + 0.02
        
        for k in range(1, 4 + extra_length // 4):
            progress = k / (3 + extra_length // 4)
            order = target_down + progress * (target_turn_back - target_down)
            if noise_level > 0:
                order += np.random.normal(0, noise_level)
            turn_back_orders.append(order)
        
        # Continue to final destination (region A, B, or end with another turn)
        if turn_j < len(interfaces) - 2:
            # Continue up to create ending in region B or ending turn
            final_orders = []
            target_final = interfaces[-1] + 0.03  # End in region B
            
            for k in range(1, 4 + extra_length // 2):
                progress = k / (3 + extra_length // 2)
                order = target_turn_back + progress * (target_final - target_turn_back)
                if noise_level > 0:
                    order += np.random.normal(0, noise_level)
                final_orders.append(order)
            
            # Add exactly one point after final interface crossing
            final_orders = final_orders[:1]  # Keep only first point after crossing
            
            return down_orders + turn_back_orders + final_orders
        else:
            # End with turn - go back down for final turn
            final_turn_orders = []
            final_target = interfaces[turn_i] - 0.01  # Just below interface i
            
            for k in range(1, 3 + extra_length // 4):
                progress = k / (2 + extra_length // 4)
                order = target_turn_back - progress * (target_turn_back - final_target)
                if noise_level > 0:
                    order += np.random.normal(0, noise_level)
                final_turn_orders.append(order)
            
            # Add exactly one point after final interface crossing
            final_turn_orders = final_turn_orders[:1]  # Keep only first point after crossing
            
            return down_orders + turn_back_orders + final_turn_orders
        
    else:
        raise ValueError(f"Unknown start_region: {start_region}")


def create_non_staple_path(interfaces, path_type="monotonic_up"):
    """Create a non-staple path for negative testing.
    
    Args:
        interfaces: List of interface values
        path_type: Type of non-staple path ("monotonic_up", "monotonic_down", "oscillating", "incomplete_turn")
    
    Returns:
        List of order parameter values that should NOT be classified as a staple path
    """
    start_order = interfaces[0] - 0.05
    end_order = interfaces[-1] + 0.05
    
    if path_type == "monotonic_up":
        # Always increasing, no turn
        return [start_order + i * (end_order - start_order) / 9 for i in range(10)]
    
    elif path_type == "monotonic_down":
        # Always decreasing, no turn
        return [end_order - i * (end_order - start_order) / 9 for i in range(10)]
    
    elif path_type == "oscillating":
        # Oscillating without proper staple structure (no proper turn)
        orders = []
        for i in range(15):
            base_order = (start_order + end_order) / 2
            oscillation = 0.1 * np.sin(i * 0.8)
            orders.append(base_order + oscillation)
        return orders
    
    elif path_type == "incomplete_turn":
        # Crosses interfaces but doesn't complete the turn (recross the same two interfaces)
        middle_idx = len(interfaces) // 2
        orders = [
            interfaces[0] - 0.05,  # Start in A
            interfaces[middle_idx] + 0.05,  # Cross up to middle
            interfaces[min(middle_idx + 1, len(interfaces) - 1)] + 0.05,  # Cross next interface
            interfaces[-1] + 0.05,  # Keep going (no turn)
        ]
        return orders
    
    else:
        raise ValueError(f"Unknown path_type: {path_type}")


def create_large_staple_path(interfaces, n_cycles=10, turn_interface_pair=None):
    """Create a large staple path by repeating staple patterns.
    
    Args:
        interfaces: List of interface values
        n_cycles: Number of staple cycles to create
        turn_interface_pair: Interface pair for turn (default: middle pair)
    
    Returns:
        List of order parameter values forming a large correct staple path
    """
    if turn_interface_pair is None:
        turn_idx = len(interfaces) // 2
        turn_interface_pair = (turn_idx, min(turn_idx + 1, len(interfaces) - 2))
    
    all_orders = []
    
    for cycle in range(n_cycles):
        # Alternate between different start regions
        start_regions = ["A", "B", "turn"]
        start_region = start_regions[cycle % len(start_regions)]
        
        cycle_orders = create_staple_path_with_turn(
            interfaces, turn_interface_pair, start_region,
            extra_length=3, noise_level=0.005
        )
        
        # Skip first point of subsequent cycles to avoid duplication
        if cycle > 0 and len(cycle_orders) > 0:
            cycle_orders = cycle_orders[1:]
        
        all_orders.extend(cycle_orders)
    
    return all_orders


def add_systems_to_path(path, orders, prefix="test"):
    """Add systems with given orders to a StaplePath.
    
    Args:
        path: StaplePath object to add systems to
        orders: List of order parameter values
        prefix: Prefix for config names
    """
    for i, order in enumerate(orders):
        system = System()
        system.order = [order]
        system.config = (f"{prefix}_{i}.xyz", i)
        system.timestep = i * 0.1
        path.append(system)
