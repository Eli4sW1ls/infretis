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

    args_dict = vars(parser.parse_args())
    input_file = args_dict["input"]

    # Run the infretis scheduler
    internalrun(input_file)


def internalrun(input_file):
    """Run internal runner.

    infretis can now be called directly without argparse.
    """
    if os.environ.get("INFRETIS_DEBUG", "false").lower() == "true":
        enable_debugging()
    
    config = setup_config(input_file)
    if config is None:
        return
    scheduler(config)
