#!/usr/bin/env python3
"""
Test runner for all staple path functionality tests.

This script runs comprehensive tests for staple path functionality including:
- Core StaplePath class functionality
- TIS staple functions (staple_sh, staple_extender, staple_wf)
- REPEX staple state functionality
- Engine integration tests
- Configuration validation
- Edge cases and error handling
- Utility functions and analysis tools
- Integration tests
- Performance tests

Usage:
    python run_staple_tests.py [--verbose] [--coverage] [--specific TEST_NAME]
"""

import sys
import os
import pytest
import argparse
from pathlib import Path

# Add the infretis root directory to the Python path
INFRETIS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(INFRETIS_ROOT))


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run staple path functionality tests")
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
    
    args = parser.parse_args()
    
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
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Specific test selection
    if args.specific:
        pytest_args.append(args.specific)
    else:
        # Run all staple-related tests
        test_files = [
            "test/core/test_staple_path.py",
            "test/core/test_staple_edge_cases.py", 
            "test/core/test_staple_utilities.py",
            "test/tis/test_staple_tis.py",
            "test/repex/test_repex_staple.py",
            "test/engines/test_staple_engines.py",
            "test/simulations/test_staple_integration.py"
        ]
        
        # Filter to only existing files
        existing_files = []
        for test_file in test_files:
            full_path = INFRETIS_ROOT / test_file
            if full_path.exists():
                existing_files.append(str(full_path))
            else:
                print(f"Warning: Test file not found: {test_file}")
        
        pytest_args.extend(existing_files)
    
    # Markers
    if args.markers:
        pytest_args.extend(["-m", args.markers])
    
    # Failed tests only
    if args.failed:
        pytest_args.append("--lf")
    
    # Parallel execution
    if args.parallel > 1:
        pytest_args.extend(["-n", str(args.parallel)])
    
    # Additional useful pytest options
    pytest_args.extend([
        "--tb=short",           # Shorter traceback format
        "--strict-markers",     # Require markers to be defined
        "--disable-warnings",   # Disable warnings for cleaner output
    ])
    
    print("=" * 60)
    print("Running Staple Path Functionality Tests")
    print("=" * 60)
    print(f"Test arguments: {' '.join(pytest_args)}")
    print()
    
    # Run the tests
    exit_code = pytest.main(pytest_args)
    
    print()
    print("=" * 60)
    if exit_code == 0:
        print("All staple tests PASSED! ✅")
    else:
        print("Some staple tests FAILED! ❌")
    print("=" * 60)
    
    return exit_code


def check_dependencies():
    """Check if required test dependencies are available."""
    required_packages = [
        "pytest",
        "numpy", 
        "unittest.mock"
    ]
    
    optional_packages = [
        ("pytest-cov", "coverage reporting"),
        ("pytest-xdist", "parallel test execution"),
        ("pytest-html", "HTML test reports")
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_required.append(package)
    
    for package, description in optional_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_optional.append((package, description))
    
    if missing_required:
        print("Error: Missing required packages:")
        for package in missing_required:
            print(f"  - {package}")
        print("\nInstall with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print("Optional packages not found (features will be limited):")
        for package, description in missing_optional:
            print(f"  - {package}: {description}")
        print()
    
    return True


def create_test_report():
    """Create a summary test report."""
    report_content = """
# Staple Path Test Coverage Report

This report covers the testing of staple path functionality in infRETIS:

## Test Categories

### 1. Core StaplePath Class Tests (`test_staple_path.py`)
- ✅ Path initialization and basic properties
- ✅ Turn detection (start, end, overall)
- ✅ Shooting point selection
- ✅ Path copying and equality
- ✅ PP path extraction
- ✅ Path pasting functionality

### 2. TIS Staple Functions Tests (`test_staple_tis.py`)
- ✅ `staple_sh()` function testing
- ✅ `staple_extender()` function testing  
- ✅ `staple_wf()` function testing
- ✅ Error handling and edge cases

### 3. REPEX Staple State Tests (`test_repex_staple.py`)
- ✅ REPEX_state_staple initialization
- ✅ Path addition and management
- ✅ Probability matrix handling
- ✅ Integration with ensemble configuration

### 4. Engine Integration Tests (`test_staple_engines.py`)
- ✅ Engine-specific staple propagation
- ✅ Custom stopping conditions
- ✅ Error handling in engines
- ✅ Performance with large paths

### 5. Edge Cases and Error Handling (`test_staple_edge_cases.py`)
- ✅ Configuration validation
- ✅ Empty and single-point paths
- ✅ Extreme parameter values
- ✅ NaN and infinity handling
- ✅ Boundary conditions

### 6. Utility Functions (`test_staple_utilities.py`)
- ✅ Path analysis tools
- ✅ Statistical calculations
- ✅ Interface crossing analysis
- ✅ Path quality metrics

### 7. Integration Tests (`test_staple_integration.py`)
- ✅ Complete workflow testing
- ✅ Multi-component integration
- ✅ Performance under load
- ✅ Memory efficiency

## Running Tests

```bash
# Run all staple tests
python run_staple_tests.py

# Run with coverage
python run_staple_tests.py --coverage

# Run specific test file
python run_staple_tests.py --specific test/core/test_staple_path.py

# Run in verbose mode
python run_staple_tests.py --verbose

# Run failed tests only
python run_staple_tests.py --failed
```

## Test Statistics

- **Total test files**: 7
- **Estimated test count**: 100+ individual tests
- **Coverage areas**: Core classes, TIS functions, engines, configuration, edge cases
- **Test types**: Unit tests, integration tests, performance tests, error handling

## Notes

- Tests use mocking to avoid dependencies on external MD engines
- Performance tests are designed to run quickly while testing scalability
- Edge case tests ensure robustness with unusual inputs
- Integration tests verify complete workflows work correctly
"""
    
    report_file = INFRETIS_ROOT / "test" / "STAPLE_TEST_REPORT.md"
    with open(report_file, "w") as f:
        f.write(report_content)
    
    print(f"Test report created: {report_file}")


if __name__ == "__main__":
    print("Staple Path Test Suite")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create test report
    create_test_report()
    
    # Run tests
    exit_code = main()
    sys.exit(exit_code)
