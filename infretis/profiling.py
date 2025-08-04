"""Simulation profiling utilities for infretis.

This module provides tools to profile entire infretis simulations, including:
- Staple path operations during simulation
- Overall simulation performance
- Memory usage tracking
- Signal handling for graceful termination with profiling output
- Real-time performance monitoring
"""
import signal
import time
import cProfile
import pstats
import io
import tracemalloc
import psutil
import os
import threading
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimulationProfileMetrics:
    """Container for simulation profiling metrics."""
    operation_name: str
    execution_time: float
    memory_peak: float
    memory_current: float
    cpu_percent: float
    timestamp: float
    thread_id: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)


class SimulationProfiler:
    """Comprehensive profiler for infretis simulations with signal handling."""
    
    def __init__(self, output_dir: str = ".", enable_memory_tracking: bool = True):
        self.results: List[SimulationProfileMetrics] = []
        self.process = psutil.Process(os.getpid())
        self.output_dir = output_dir
        self.enable_memory_tracking = enable_memory_tracking
        
        # Profiling state
        self.active = False
        self.start_time = None
        self.cprofile = None
        self.memory_tracker = None
        
        # Signal handling
        self.original_sigint_handler = None
        self.original_sigterm_handler = None
        self.termination_requested = False
        
        # Performance monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        self.monitoring_interval = 1.0  # seconds
        
        # Staple-specific counters
        self.staple_operations = {
            'turn_detections': 0,
            'path_extractions': 0,
            'shooting_point_selections': 0,
            'path_copies': 0
        }
    
    def start_profiling(self, enable_cprofile: bool = True, enable_signal_handling: bool = True):
        """Start comprehensive profiling of the simulation."""
        if self.active:
            return
        
        print("🔍 Starting simulation profiling...")
        self.active = True
        self.start_time = time.perf_counter()
        
        # Setup signal handlers for graceful termination
        if enable_signal_handling:
            self._setup_signal_handlers()
        
        # Start cProfile if requested
        if enable_cprofile:
            self.cprofile = cProfile.Profile()
            self.cprofile.enable()
        
        # Start memory tracking
        if self.enable_memory_tracking:
            tracemalloc.start()
        
        # Start performance monitoring thread
        self._start_monitoring()
        
        print("✅ Simulation profiling active. Press Ctrl+C for graceful termination with profiling report.")
    
    def stop_profiling(self, generate_report: bool = True):
        """Stop profiling and optionally generate report."""
        if not self.active:
            return
        
        print("🛑 Stopping simulation profiling...")
        self.active = False
        
        # Stop monitoring thread
        self._stop_monitoring()
        
        # Stop cProfile
        if self.cprofile:
            self.cprofile.disable()
        
        # Stop memory tracking
        if self.enable_memory_tracking and tracemalloc.is_tracing():
            tracemalloc.stop()
        
        # Restore signal handlers
        self._restore_signal_handlers()
        
        if generate_report:
            self.generate_and_save_report()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful termination."""
        self.original_sigint_handler = signal.signal(signal.SIGINT, self._signal_handler)
        self.original_sigterm_handler = signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        if self.original_sigint_handler:
            signal.signal(signal.SIGINT, self.original_sigint_handler)
        if self.original_sigterm_handler:
            signal.signal(signal.SIGTERM, self.original_sigterm_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        print(f"\\n🛑 Received {signal_name}. Generating profiling report and terminating gracefully...")
        
        self.termination_requested = True
        self.stop_profiling(generate_report=True)
        
        # Exit gracefully
        exit(0)
    
    def _start_monitoring(self):
        """Start background monitoring thread."""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitoring_thread.start()
    
    def _stop_monitoring(self):
        """Stop background monitoring thread."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
    
    def _monitor_performance(self):
        """Background monitoring of system performance."""
        while self.monitoring_active and self.active:
            try:
                current_time = time.perf_counter()
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                # Get memory stats if tracking is enabled
                memory_peak = 0
                if self.enable_memory_tracking and tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    memory_peak = peak / 1024 / 1024  # MB
                
                metrics = SimulationProfileMetrics(
                    operation_name="system_monitoring",
                    execution_time=current_time - self.start_time if self.start_time else 0,
                    memory_peak=memory_peak,
                    memory_current=memory_info.rss / 1024 / 1024,  # MB
                    cpu_percent=cpu_percent,
                    timestamp=current_time,
                    thread_id=threading.current_thread().name
                )
                
                self.results.append(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Warning: Error in performance monitoring: {e}")
                break
    
    @contextmanager
    def profile_operation(self, operation_name: str, **kwargs):
        """Context manager to profile a specific operation."""
        if not self.active:
            yield
            return
        
        # Track staple operations
        if 'turn' in operation_name.lower():
            self.staple_operations['turn_detections'] += 1
        elif 'path' in operation_name.lower() and 'extract' in operation_name.lower():
            self.staple_operations['path_extractions'] += 1
        elif 'shooting' in operation_name.lower():
            self.staple_operations['shooting_point_selections'] += 1
        elif 'copy' in operation_name.lower():
            self.staple_operations['path_copies'] += 1
        
        # Start timing
        start_time = time.perf_counter()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        memory_peak = 0
        if self.enable_memory_tracking and tracemalloc.is_tracing():
            tracemalloc_start = tracemalloc.get_traced_memory()[0]
        
        try:
            yield
        finally:
            # End timing
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # Get memory stats
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            if self.enable_memory_tracking and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                memory_peak = (peak - tracemalloc_start) / 1024 / 1024 if tracemalloc_start else peak / 1024 / 1024
            
            # Store results
            metrics = SimulationProfileMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_peak=memory_peak,
                memory_current=current_memory,
                cpu_percent=self.process.cpu_percent(),
                timestamp=end_time,
                thread_id=threading.current_thread().name,
                additional_data=kwargs
            )
            
            self.results.append(metrics)
    
    def generate_report(self) -> str:
        """Generate a comprehensive profiling report."""
        if not self.results:
            return "No profiling data available."
        
        total_time = time.perf_counter() - self.start_time if self.start_time else 0
        
        report = ["=" * 80]
        report.append("INFRETIS SIMULATION PROFILING REPORT")
        report.append("=" * 80)
        report.append(f"Total simulation time: {total_time:.2f} seconds")
        report.append(f"Profiling data points: {len(self.results)}")
        report.append("")
        
        # Staple operations summary
        report.append("STAPLE OPERATIONS SUMMARY")
        report.append("-" * 40)
        for op_name, count in self.staple_operations.items():
            if count > 0:
                rate = count / total_time if total_time > 0 else 0
                report.append(f"  {op_name.replace('_', ' ').title()}: {count} ({rate:.2f}/sec)")
        report.append("")
        
        # Group results by operation
        operations = {}
        monitoring_results = []
        
        for result in self.results:
            if result.operation_name == "system_monitoring":
                monitoring_results.append(result)
            else:
                op_name = result.operation_name
                if op_name not in operations:
                    operations[op_name] = []
                operations[op_name].append(result)
        
        # Analyze specific operations
        if operations:
            report.append("OPERATION PERFORMANCE")
            report.append("-" * 40)
            
            for op_name, results in operations.items():
                times = [r.execution_time for r in results]
                memories = [r.memory_peak for r in results if r.memory_peak > 0]
                
                report.append(f"  {op_name}:")
                report.append(f"    Count: {len(results)}")
                if times:
                    report.append(f"    Time - Min: {min(times)*1000:.3f}ms, Max: {max(times)*1000:.3f}ms")
                    report.append(f"           Mean: {np.mean(times)*1000:.3f}ms, Total: {sum(times):.3f}s")
                if memories:
                    report.append(f"    Memory - Peak: {max(memories):.2f}MB, Mean: {np.mean(memories):.2f}MB")
                report.append("")
        
        # System monitoring analysis
        if monitoring_results:
            report.append("SYSTEM PERFORMANCE MONITORING")
            report.append("-" * 40)
            
            cpu_usage = [r.cpu_percent for r in monitoring_results]
            memory_usage = [r.memory_current for r in monitoring_results]
            
            if cpu_usage:
                report.append(f"  CPU Usage:")
                report.append(f"    Mean: {np.mean(cpu_usage):.1f}%")
                report.append(f"    Peak: {max(cpu_usage):.1f}%")
            
            if memory_usage:
                report.append(f"  Memory Usage:")
                report.append(f"    Mean: {np.mean(memory_usage):.1f}MB")
                report.append(f"    Peak: {max(memory_usage):.1f}MB")
            
            report.append("")
        
        # Performance recommendations
        report.append("PERFORMANCE ANALYSIS")
        report.append("-" * 40)
        
        if operations:
            # Find bottlenecks
            all_op_results = [r for results in operations.values() for r in results]
            if all_op_results:
                slowest = max(all_op_results, key=lambda r: r.execution_time)
                most_memory = max(all_op_results, key=lambda r: r.memory_peak)
                
                report.append(f"  Slowest single operation: {slowest.operation_name} ({slowest.execution_time*1000:.3f}ms)")
                report.append(f"  Highest memory operation: {most_memory.operation_name} ({most_memory.memory_peak:.2f}MB)")
        
        # Calculate operation frequency and total time contribution
        if operations:
            total_op_time = sum(sum(r.execution_time for r in results) for results in operations.values())
            if total_op_time > 0:
                report.append(f"  Total operation time: {total_op_time:.3f}s ({total_op_time/total_time*100:.1f}% of simulation)")
        
        report.append("")
        return "\\n".join(report)
    
    def generate_and_save_report(self):
        """Generate report and save to files."""
        report = self.generate_report()
        
        # Print report to console
        print("\\n" + report)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save text report
        report_file = os.path.join(self.output_dir, "infretis_profiling_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Profiling report saved to: {report_file}")
        
        # Save CSV data
        self._save_csv_data()
        
        # Save cProfile data if available
        if self.cprofile:
            self._save_cprofile_data()
    
    def _save_csv_data(self):
        """Save detailed profiling data to CSV."""
        import csv
        
        csv_file = os.path.join(self.output_dir, "infretis_profiling_data.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Operation', 'Execution_Time_ms', 'Memory_Peak_MB', 
                           'Memory_Current_MB', 'CPU_Percent', 'Timestamp', 'Thread_ID'])
            
            for result in self.results:
                writer.writerow([
                    result.operation_name,
                    result.execution_time * 1000,  # Convert to milliseconds
                    result.memory_peak,
                    result.memory_current,
                    result.cpu_percent,
                    result.timestamp,
                    result.thread_id
                ])
        
        print(f"📊 Profiling data saved to: {csv_file}")
    
    def _save_cprofile_data(self):
        """Save cProfile data."""
        if not self.cprofile:
            return
        
        # Save binary profile data
        profile_file = os.path.join(self.output_dir, "infretis_cprofile.prof")
        self.cprofile.dump_stats(profile_file)
        print(f"🐍 cProfile data saved to: {profile_file}")
        
        # Generate and save text report
        s = io.StringIO()
        stats = pstats.Stats(self.cprofile, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(50)  # Top 50 functions
        
        profile_text_file = os.path.join(self.output_dir, "infretis_cprofile_report.txt")
        with open(profile_text_file, 'w') as f:
            f.write("INFRETIS CPROFILE REPORT\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(s.getvalue())
        
        print(f"📈 cProfile report saved to: {profile_text_file}")


# Global profiler instance for easy access
_global_profiler: Optional[SimulationProfiler] = None


def get_simulation_profiler() -> Optional[SimulationProfiler]:
    """Get the global simulation profiler instance."""
    return _global_profiler


def start_simulation_profiling(output_dir: str = ".", enable_memory_tracking: bool = True, 
                             enable_cprofile: bool = True, enable_signal_handling: bool = True) -> SimulationProfiler:
    """Start simulation profiling with the given options."""
    global _global_profiler
    
    if _global_profiler and _global_profiler.active:
        print("Warning: Profiling is already active")
        return _global_profiler
    
    _global_profiler = SimulationProfiler(output_dir=output_dir, 
                                        enable_memory_tracking=enable_memory_tracking)
    _global_profiler.start_profiling(enable_cprofile=enable_cprofile, 
                                   enable_signal_handling=enable_signal_handling)
    
    return _global_profiler


def stop_simulation_profiling():
    """Stop simulation profiling and generate report."""
    global _global_profiler
    
    if _global_profiler:
        _global_profiler.stop_profiling(generate_report=True)
        _global_profiler = None


def profile_operation(operation_name: str, **kwargs):
    """Context manager to profile an operation using the global profiler."""
    global _global_profiler
    
    if _global_profiler and _global_profiler.active:
        return _global_profiler.profile_operation(operation_name, **kwargs)
    else:
        # Return a no-op context manager if profiling is not active
        from contextlib import nullcontext
        return nullcontext()
