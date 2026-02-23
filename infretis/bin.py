"""The functions to be used to run infretis via the terminal."""

import argparse
import debugpy
import os

from infretis.scheduler import scheduler
from infretis.setup import setup_config
from infretis.tools.performance_profiler import global_profiler, start_periodic_reports

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

    args_dict = vars(parser.parse_args())
    input_file = args_dict["input"]

    # Run the infretis scheduler
    internalrun(input_file)


def internalrun(input_file, enable_profiling=True):
    """Run internal runner.

    infretis can now be called directly without argparse.
    """
    print(f"🔧 Starting internalrun with input: {input_file}")
    
    if os.environ.get("INFRETIS_DEBUG", "false").lower() == "true":
        enable_debugging()
        print("🐛 Debug mode enabled - breakpoints should work now!")
    
    # Note: Comprehensive profiling available in profiling.py module
    if enable_profiling:
        print("📊 Basic profiling requested - use profiling.py module for detailed analysis")
        # start background reporting every five minutes
        start_periodic_reports(interval=300.0)
    
    print("🚀 About to call setup_config - set breakpoint here!")
    try:
        config = setup_config(input_file)
        if config is None:
            return
        print("📋 Configuration loaded, starting scheduler...")
        scheduler(config)
    finally:
        if enable_profiling:
            print("✅ Simulation completed")
