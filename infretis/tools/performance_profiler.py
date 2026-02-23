"""Performance profiling utilities for staple path operations.

This module provides tools to:
- Profile individual staple path operations
- Identify performance bottlenecks
- Measure memory usage
- Generate performance reports
- Compare optimization results
"""
import time
import cProfile
import pstats
import io
import tracemalloc
import psutil
import os
from typing import Dict, List, Tuple, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import threading

from infretis.classes.staple_path import *
from infretis.classes.system import System

# Fallback utility functions if staple_path_utils not available
def create_staple_path_with_turn(interfaces, turn_at_interface_pair=None, start_region="A", extra_length=5):
    """Create a correct staple path with turn."""
    if turn_at_interface_pair is None:
        turn_idx = len(interfaces) // 2
        turn_at_interface_pair = (turn_idx, min(turn_idx + 1, len(interfaces) - 2))
    
    turn_i, turn_j = turn_at_interface_pair
    
    if start_region == "A":
        # Start below interface 0, cross to turn point, then turn back
        orders = [interfaces[0] - 0.05]  # Start in A
        # Go up to interface j
        for k in range(3 + extra_length // 2):
            progress = (k + 1) / (3 + extra_length // 2)
            order = interfaces[0] - 0.05 + progress * (interfaces[turn_j] + 0.02 - (interfaces[0] - 0.05))
            orders.append(order)
        # Turn back down past interface i
        for k in range(1, 4 + extra_length - extra_length // 2):
            progress = k / (3 + extra_length - extra_length // 2)
            order = interfaces[turn_j] + 0.02 - progress * (interfaces[turn_j] + 0.02 - (interfaces[turn_i] - 0.02))
            orders.append(order)
    else:
        # Simple fallback for other cases
        orders = [interfaces[0] - 0.05, interfaces[turn_j] + 0.02, interfaces[turn_i] - 0.02]
    
    return orders

def create_large_staple_path(interfaces, n_cycles=10):
    """Create large staple path."""
    all_orders = []
    for cycle in range(n_cycles):
        cycle_orders = create_staple_path_with_turn(interfaces, (1, 2), "A", 3)
        if cycle > 0 and len(cycle_orders) > 0:
            cycle_orders = cycle_orders[1:]
        all_orders.extend(cycle_orders)
    return all_orders

def add_systems_to_path(path, orders, prefix="test"):
    """Add systems to path."""
    for i, order in enumerate(orders):
        system = System()
        system.order = [order]
        system.config = (f"{prefix}_{i}.xyz", i)
        path.append(system)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_name: str
    execution_time: float
    memory_peak: float
    memory_current: float
    cpu_percent: float
    path_length: int
    additional_data: Dict[str, Any] = None


class StaplePerformanceProfiler:
    """Comprehensive performance profiler for staple path operations."""
    
    def __init__(self):
        self.results: List[PerformanceMetrics] = []
        self.process = psutil.Process(os.getpid())
    
    @contextmanager
    def profile_operation(self, operation_name: str, path_length: int = 0, **kwargs):
        """Context manager to profile a staple operation."""
        # Start memory tracking
        tracemalloc.start()
        
        # Record initial state
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = self.process.cpu_percent()
        
        # Start timing
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # End timing
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Get memory stats
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Get CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # Store results
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_peak=peak / 1024 / 1024,  # MB
                memory_current=current_memory,
                cpu_percent=cpu_percent,
                path_length=path_length,
                additional_data=kwargs
            )
            
            self.results.append(metrics)
    
    def profile_turn_detection_scaling(self, max_path_length: int = 2000, step: int = 200):
        """Profile turn detection performance across different path lengths."""
        interfaces = [0.1, 0.2, 0.3, 0.4]
        path_lengths = list(range(step, max_path_length + 1, step))
        
        for length in path_lengths:
            path = StaplePath()
            
            # Create appropriately sized staple path
            n_cycles = max(1, length // 20)
            orders = create_large_staple_path(interfaces, n_cycles=n_cycles)[:length]
            
            with self.profile_operation(f"turn_detection", path_length=length):
                add_systems_to_path(path, orders, "profile")
                start_info, end_info, overall_valid = path.check_turns(interfaces)
    
    def profile_shooting_point_selection(self, path_length: int = 1000, num_selections: int = 1000):
        """Profile shooting point selection performance."""
        path = StaplePath()
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        # Create large path
        orders = create_large_staple_path(interfaces, n_cycles=path_length // 20)[:path_length]
        add_systems_to_path(path, orders, "shooting_profile")
        path.sh_region = {100: path_length - 100}
        
        rgen = np.random.default_rng(42)
        
        with self.profile_operation(f"shooting_point_selection", 
                                  path_length=path_length, 
                                  num_selections=num_selections):
            for _ in range(num_selections):
                shooting_point, idx = path.get_shooting_point(rgen)
    
    def profile_path_copying(self, path_lengths: List[int] = None):
        """Profile path copying performance."""
        if path_lengths is None:
            path_lengths = [100, 500, 1000, 2000]
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        
        for length in path_lengths:
            path = StaplePath()
            orders = create_large_staple_path(interfaces, n_cycles=length // 20)[:length]
            add_systems_to_path(path, orders, "copy_profile")
            path.sh_region = (10, length - 10)
            path.pptype = "LML"
            
            with self.profile_operation(f"path_copying", path_length=length):
                for _ in range(10):  # Multiple copies to get meaningful timing
                    path_copy = path.copy()
    
    def profile_turn_detected_function(self, num_iterations: int = 10000):
        """Profile the turn_detected function specifically."""
        interfaces = [0.15, 0.25, 0.35, 0.45]
        
        # Create different test cases
        test_cases = [
            (create_staple_path_with_turn(interfaces, (1, 2), "A", 2), "valid_turn"),
            ([0.1, 0.2, 0.3, 0.4, 0.5], "monotonic"),
            (create_staple_path_with_turn(interfaces, (1, 2), "A", 2)[::-1], "reverse_turn"),
        ]
        
        for orders, case_name in test_cases:
            phasepoints = []
            for i, order in enumerate(orders):
                system = System()
                system.order = [order]
                system.config = (f"turn_test_{i}.xyz", i)
                phasepoints.append(system)
            
            with self.profile_operation(f"turn_detected_{case_name}", 
                                      path_length=len(orders),
                                      num_iterations=num_iterations):
                for _ in range(num_iterations):
                    result = turn_detected(phasepoints, interfaces, 2, 1)
    
    def profile_get_pp_path(self, path_lengths: List[int] = None):
        """Profile get_pp_path performance."""
        if path_lengths is None:
            path_lengths = [50, 100, 200, 500]
        
        interfaces = [0.1, 0.2, 0.3, 0.4]
        pp_interfaces = [0.2, 0.3, 0.4]
        
        for length in path_lengths:
            path = StaplePath()
            orders = create_staple_path_with_turn(interfaces, (1, 2), "A", length // 5)[:length]
            add_systems_to_path(path, orders, "pp_profile")
            path.pptype = "LML"
            path.sh_region = (5, length - 5)
            
            with self.profile_operation(f"get_pp_path", path_length=length):
                try:
                    for _ in range(100):  # Multiple calls
                        pp_path, ptype, sh_region = path.get_pp_path(interfaces, pp_interfaces)
                except Exception as e:
                    # Some paths might not be valid for PP extraction
                    pass
    
    def detailed_profile_with_cprofile(self, operation_func: Callable, *args, **kwargs):
        """Run detailed profiling using cProfile."""
        pr = cProfile.Profile()
        
        pr.enable()
        result = operation_func(*args, **kwargs)
        pr.disable()
        
        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        return result, s.getvalue()
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        if not self.results:
            return "No profiling results available."
        
        report = ["=" * 80]
        report.append("STAPLE PATH PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Group results by operation
        operations = {}
        for result in self.results:
            op_name = result.operation_name
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(result)
        
        for op_name, results in operations.items():
            report.append(f"Operation: {op_name}")
            report.append("-" * 40)
            
            # Calculate statistics
            times = [r.execution_time for r in results]
            memories = [r.memory_peak for r in results]
            path_lengths = [r.path_length for r in results if r.path_length > 0]
            
            if times:
                report.append(f"  Execution Time:")
                report.append(f"    Min:    {min(times):.6f} s")
                report.append(f"    Max:    {max(times):.6f} s")
                report.append(f"    Mean:   {np.mean(times):.6f} s")
                report.append(f"    Median: {np.median(times):.6f} s")
                report.append(f"    Std:    {np.std(times):.6f} s")
            
            if memories:
                report.append(f"  Memory Usage:")
                report.append(f"    Peak:   {max(memories):.2f} MB")
                report.append(f"    Mean:   {np.mean(memories):.2f} MB")
            
            if path_lengths:
                report.append(f"  Path Length Range: {min(path_lengths)} - {max(path_lengths)}")
            
            # Performance per path length (if applicable)
            if len(path_lengths) > 1 and len(times) == len(path_lengths):
                time_per_point = [t/l for t, l in zip(times, path_lengths)]
                report.append(f"  Time per Path Point:")
                report.append(f"    Mean: {np.mean(time_per_point)*1e6:.2f} μs/point")
            
            report.append("")
        
        # Overall summary
        report.append("SUMMARY")
        report.append("-" * 40)
        total_time = sum(r.execution_time for r in self.results)
        report.append(f"Total profiling time: {total_time:.3f} s")
        report.append(f"Number of operations: {len(self.results)}")
        
        # Identify bottlenecks
        slowest = max(self.results, key=lambda r: r.execution_time)
        most_memory = max(self.results, key=lambda r: r.memory_peak)
        
        report.append("")
        report.append("BOTTLENECKS")
        report.append("-" * 40)
        report.append(f"Slowest operation: {slowest.operation_name} ({slowest.execution_time:.6f} s)")
        report.append(f"Most memory usage: {most_memory.operation_name} ({most_memory.memory_peak:.2f} MB)")
        
        return "\n".join(report)

    def save_results_csv(self, filename: str):
        """Save results to CSV for further analysis."""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Operation', 'Execution_Time', 'Memory_Peak', 
                           'Memory_Current', 'CPU_Percent', 'Path_Length'])
            
            for result in self.results:
                writer.writerow([
                    result.operation_name,
                    result.execution_time,
                    result.memory_peak,
                    result.memory_current,
                    result.cpu_percent,
                    result.path_length
                ])


# ------------------------------------------------------------------
# Global profiler and periodic reporting
# ------------------------------------------------------------------

global_profiler: StaplePerformanceProfiler = StaplePerformanceProfiler()


def start_periodic_reports(interval: float = 300.0):
    """
    Print a summary of profiling results every `interval` seconds.

    This function schedules itself on a background timer and uses the
    `global_profiler` singleton. Call once at startup.
    """
    def _report():
        print("\n" + "=" * 80)
        print(f"[{datetime.now():%Y.%m.%d %H:%M:%S}] stapler performance report")
        print(global_profiler.generate_report())
        print("=" * 80 + "\n")
        threading.Timer(interval, _report).start()

    _report()


def run_comprehensive_profiling():
    """Run a comprehensive profiling session."""
    profiler = StaplePerformanceProfiler()
    
    print("Starting comprehensive staple path performance profiling...")
    print("This may take several minutes...")
    
    # Profile different operations
    print("1. Profiling turn detection scaling...")
    profiler.profile_turn_detection_scaling(max_path_length=1000, step=100)
    
    print("2. Profiling shooting point selection...")
    profiler.profile_shooting_point_selection(path_length=500, num_selections=1000)
    
    print("3. Profiling path copying...")
    profiler.profile_path_copying([100, 200, 500, 1000])
    
    print("4. Profiling turn_detected function...")
    profiler.profile_turn_detected_function(num_iterations=5000)
    
    print("5. Profiling get_pp_path...")
    profiler.profile_get_pp_path([50, 100, 200])
    
    # Generate and save report
    report = profiler.generate_report()
    print("\n" + report)
    
    # Save detailed results
    profiler.save_results_csv("staple_performance_results.csv")
    print("\nDetailed results saved to: staple_performance_results.csv")
    
    return profiler


if __name__ == "__main__":
    profiler = run_comprehensive_profiling()
