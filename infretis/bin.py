"""The functions to be used to run infretis via the terminal."""

import argparse
import debugpy
import os

from infretis.scheduler import scheduler
from infretis.setup import setup_config

def enable_debugging(port=56784):
    """Enable remote debugging for external tools like infinit."""
    try:
        debugpy.listen(port)
        print(f"🐛 Debugger listening on port {port}")
        print("Attach VS Code debugger now...")
        debugpy.wait_for_client()
        print("✅ Debugger attached!")
    except Exception as e:
        print(f"Could not start debugger: {e}")


def infretisrun():
    """Read input and runs infretis."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Location of infretis input file", required=True
    )
    parser.add_argument(
        "--profile", action="store_true", 
        help="Enable profiling mode with performance monitoring and graceful termination"
    )
    parser.add_argument(
        "--profile-output", default=".", 
        help="Directory to save profiling output (default: current directory)"
    )
    parser.add_argument(
        "--profile-memory", action="store_true", default=True,
        help="Enable memory profiling (default: True)"
    )
    parser.add_argument(
        "--profile-cprofile", action="store_true", default=True,
        help="Enable cProfile function-level profiling (default: True)"
    )

    args_dict = vars(parser.parse_args())
    input_file = args_dict["input"]
    
    # Profiling options
    enable_profiling = args_dict["profile"]
    profile_output = args_dict["profile_output"]
    enable_memory = args_dict["profile_memory"]
    enable_cprofile = args_dict["profile_cprofile"]

    # Run the infretis scheduler
    internalrun(input_file, enable_profiling=enable_profiling, 
               profile_output=profile_output, enable_memory=enable_memory, 
               enable_cprofile=enable_cprofile)


def internalrun(input_file, enable_profiling=False, profile_output=".", 
               enable_memory=True, enable_cprofile=True):
    """Run internal runner.

    infretis can now be called directly without argparse.
    """
    if os.environ.get("INFRETIS_DEBUG", "false").lower() == "true":
        enable_debugging()
    
    # Start profiling if requested
    profiler = None
    if enable_profiling:
        from infretis.profiling import start_simulation_profiling
        profiler = start_simulation_profiling(
            output_dir=profile_output,
            enable_memory_tracking=enable_memory,
            enable_cprofile=enable_cprofile,
            enable_signal_handling=True
        )
    
    try:
        config = setup_config(input_file)
        if config is None:
            return
        scheduler(config)
    finally:
        # Stop profiling if it was started
        if profiler:
            from infretis.profiling import stop_simulation_profiling
            stop_simulation_profiling()
