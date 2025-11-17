"""
Test runner script for manim unit tests in Chapter 10
"""

import pytest
import sys
import os

def run_manim_tests():
    """Run all manim-related tests."""
    
    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the test directory
    os.chdir(test_dir)
    
    print("Running Manim unit tests for Chapter 10: Distribution Theory...")
    print("=" * 60)
    
    # Run tests with specific markers and options
    pytest_args = [
        "test_distribution_manim.py",  # Specific test file
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-m", "manim",  # Only run tests marked with 'manim'
        "--durations=10",  # Show 10 slowest tests
    ]
    
    # Add optional arguments
    if "--all" in sys.argv:
        pytest_args.remove("-m")
        pytest_args.remove("manim")
        print("Running ALL tests (including non-manim tests)")
    
    if "--skip-manim" in sys.argv:
        pytest_args.extend(["-m", "not manim"])
        print("Skipping manim tests, running only mathematical tests")
    
    print(f"Running command: pytest {' '.join(pytest_args)}")
    print("=" * 60)
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    
    return exit_code

def run_specific_test(test_name):
    """Run a specific test function."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(test_dir)
    
    print(f"Running specific test: {test_name}")
    print("=" * 60)
    
    pytest_args = [
        "test_distribution_manim.py", 
        "-v", 
        "--tb=short",
        "-k", test_name  # Run only tests matching this name
    ]
    
    exit_code = pytest.main(pytest_args)
    return exit_code

def list_available_tests():
    """List all available test functions."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(test_dir, "test_distribution_manim.py")
    
    print("Available test functions:")
    print("=" * 60)
    
    with open(test_file, 'r') as f:
        lines = f.readlines()
        
    test_functions = []
    current_class = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('class Test'):
            current_class = line.split('class ')[1].split(':')[0]
        elif line.startswith('def test_') and 'pytest.mark' not in line:
            func_name = line.split('def ')[1].split('(')[0]
            if current_class:
                test_functions.append(f"{current_class}::{func_name}")
            else:
                test_functions.append(func_name)
    
    for i, test_func in enumerate(test_functions, 1):
        print(f"{i:2d}. {test_func}")
    
    print("\nUsage examples:")
    print("  python run_manim_tests.py                    # Run all manim tests")
    print("  python run_manim_tests.py --all              # Run all tests")
    print("  python run_manim_tests.py --skip-manim       # Skip manim tests")
    print("  python run_manim_tests.py test_scene_initialization  # Run specific test")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_available_tests()
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        # Assume it's a specific test name
        exit_code = run_specific_test(sys.argv[1])
        sys.exit(exit_code)
    else:
        # Run all manim tests
        exit_code = run_manim_tests()
        sys.exit(exit_code)