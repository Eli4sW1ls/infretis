#!/usr/bin/env python3
"""
Organized Staple Test Runner

This script runs staple path functionality tests organized by category:
- Core: Basic StaplePath functionality (test/staple/core/)
- Integration: TIS, REPEX, engines, simulations (test/staple/integration/)  
- Validation: Edge cases, error handling (test/staple/validation/)
- All: Run all staple tests
- Original: Run tests from original locations

Usage:
    python run_staple_tests.py [category] [options]
    
Examples:
    python run_staple_tests.py                    # Run all organized tests
    python run_staple_tests.py core               # Run only core tests
    python run_staple_tests.py integration        # Run only integration tests
    python run_staple_tests.py --coverage         # Run with coverage
    python run_staple_tests.py --original         # Run from original locations
"""

import sys
import os
import pytest
import argparse
from pathlib import Path

# Add the infretis root directory to the Python path
INFRETIS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(INFRETIS_ROOT))

# Test categories in organized structure
TEST_CATEGORIES = {
    "core": {
        "description": "Core StaplePath functionality and algorithms", 
        "path": "test/staple/core",
        "files": [
            "test_staple_path.py",
            "test_staple_advanced_turns.py", 
            "test_staple_integration.py",
            "test_staple_performance.py"
        ]
    },
    "integration": {
        "description": "TIS, REPEX, engines, and simulation integration",
        "path": "test/staple/integration",
        "files": [
            "test_repex_staple.py",
            "test_staple_engines.py", 
            "test_staple_integration.py",
            "test_staple_tis.py",
            "test_staple_endtoend.py",
            "test_staple_simplified.py"
        ]
    },
    "validation": {
        "description": "Edge cases, error handling, and validation",
        "path": "test/staple/validation", 
        "files": [
            "test_staple_edge_cases.py",
            "test_staple_validation.py"
        ]
    }
}


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run staple path functionality tests by category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all organized tests
  %(prog)s core               # Run only core tests
  %(prog)s integration        # Run only integration tests  
  %(prog)s validation         # Run only validation tests
  %(prog)s core integration   # Run core and integration tests
  %(prog)s --list             # List available categories
  %(prog)s --specific test/staple/core/test_staple_path.py  # Run specific file
        """
    )
    
    parser.add_argument("categories", nargs="*", default=[],
                       help="Test categories to run: core, integration, validation, all")
    parser.add_argument("--list", action="store_true",
                       help="List available test categories")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Run tests in verbose mode")
    parser.add_argument("--coverage", action="store_true",
                       help="Run tests with coverage reporting")
    parser.add_argument("--specific", "-s", type=str,
                       help="Run specific test file or test function")
    parser.add_argument("--markers", "-m", type=str,
                       help="Run tests with specific pytest markers")
    parser.add_argument("--failed", action="store_true",
                       help="Re-run only failed tests from last run")
    parser.add_argument("--parallel", "-n", type=int, default=1,
                       help="Number of parallel test processes (requires pytest-xdist)")
    
    # Check for --list before parsing to avoid argument validation
    if "--list" in sys.argv:
        print_categories()
        return 0
    
    args = parser.parse_args()
    
    # If no categories specified and not using --specific, default to all
    if not args.categories and not args.specific:
        args.categories = ["all"]
    
    # Validate categories
    valid_categories = list(TEST_CATEGORIES.keys()) + ["all"]
    for cat in args.categories:
        if cat not in valid_categories:
            print(f"Error: Invalid category '{cat}'. Valid categories: {', '.join(valid_categories)}")
            return 1
    
    # Base pytest arguments
    pytest_args = []
    
    # Verbose mode
    if args.verbose:
        pytest_args.extend(["-v", "-s"])
    
    # Coverage reporting
    if args.coverage:
        pytest_args.extend([
            "--cov=infretis.classes.staple_path",
            "--cov=infretis.classes.repex_staple", 
            "--cov=infretis.core.tis",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing"
        ])
    
    # Determine which tests to run
    if args.specific:
        pytest_args.append(args.specific)
    else:
        # Use organized structure
        categories = args.categories if args.categories else ["all"]
        if "all" in categories:
            categories = list(TEST_CATEGORIES.keys())
        
        test_files = get_category_files(categories)
        pytest_args.extend(test_files)
        print(f"📁 Running organized tests for: {', '.join(categories)} ({len(test_files)} files)")
    
    # Additional options
    if args.markers:
        pytest_args.extend(["-m", args.markers])
    if args.failed:
        pytest_args.append("--lf")
    if args.parallel > 1:
        pytest_args.extend(["-n", str(args.parallel)])
    
    # Useful pytest options
    pytest_args.extend([
        "--tb=short",
        "--disable-warnings",
    ])
    
    print("=" * 60)
    print("🧪 Running Staple Path Tests")
    print("=" * 60)
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ All staple tests PASSED!")
    else:
        print("❌ Some staple tests FAILED!")
    print("=" * 60)
    
    return exit_code


def print_categories():
    """Print available test categories."""
    print("\n📋 Available Test Categories:")
    print("=" * 50)
    
    for category, info in TEST_CATEGORIES.items():
        path = INFRETIS_ROOT / info["path"]
        file_count = len(info["files"])
        existing_count = len([f for f in info["files"] if (path / f).exists()])
        
        status = "✅" if existing_count == file_count else "⚠️"
        print(f"\n{status} {category.upper()}")
        print(f"   Description: {info['description']}")
        print(f"   Location: {info['path']}")
        print(f"   Files: {existing_count}/{file_count} available")


def get_category_files(categories):
    """Get test files for specified categories."""
    all_files = []
    for category in categories:
        if category in TEST_CATEGORIES:
            category_info = TEST_CATEGORIES[category]
            category_path = INFRETIS_ROOT / category_info["path"]
            
            for file_name in category_info["files"]:
                full_path = category_path / file_name
                if full_path.exists():
                    all_files.append(str(full_path))
                else:
                    print(f"Warning: Test file not found: {category_info['path']}/{file_name}")
        else:
            print(f"Warning: Unknown category: {category}")
    return all_files


if __name__ == "__main__":
    print("🧪 Staple Path Test Suite")
    print("=" * 40)
    exit_code = main()
    sys.exit(exit_code)
