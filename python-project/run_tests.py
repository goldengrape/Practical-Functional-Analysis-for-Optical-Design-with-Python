#!/usr/bin/env python3
"""
Comprehensive test runner for Practical Functional Analysis for Optical Design with Python.

This script runs tests for all chapters and provides a summary of test results.
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
import time
from typing import Dict, List, Tuple


class TestRunner:
    """Test runner for the optical design course."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests"
        self.results = {}
        
    def get_chapter_test_dirs(self) -> List[Path]:
        """Get all chapter test directories."""
        chapter_dirs = []
        for item in self.test_dir.iterdir():
            if item.is_dir() and item.name.startswith("chapter"):
                chapter_dirs.append(item)
        return sorted(chapter_dirs)
    
    def run_chapter_tests(self, chapter_dir: Path, verbose: bool = False, 
                         markers: str = None) -> Dict:
        """Run tests for a specific chapter."""
        chapter_name = chapter_dir.name
        print(f"\n{'='*60}")
        print(f"Running tests for {chapter_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            str(chapter_dir),
            "--tb=short"
        ]
        
        if verbose:
            cmd.append("-v")
        
        if markers:
            cmd.extend(["-m", markers])
        
        # Add coverage options
        cmd.extend([
            "--cov-report=term-missing",
            "--cov-report=html:coverage_html",
            "--junitxml=test-results.xml"
        ])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per chapter
            )
            
            elapsed_time = time.time() - start_time
            
            success = result.returncode == 0
            
            # Parse results
            output_lines = result.stdout.split('\n')
            failed = 0
            passed = 0
            skipped = 0
            
            for line in output_lines:
                if "FAILED" in line and "=" not in line:
                    failed += 1
                elif "PASSED" in line and "=" not in line:
                    passed += 1
                elif "SKIPPED" in line and "=" not in line:
                    skipped += 1
            
            # Look for summary line
            summary_line = None
            for line in reversed(output_lines):
                if "failed" in line.lower() or "passed" in line.lower():
                    summary_line = line.strip()
                    break
            
            chapter_result = {
                'success': success,
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'elapsed_time': elapsed_time,
                'summary_line': summary_line,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
            self.results[chapter_name] = chapter_result
            
            # Print summary
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"{status} - {summary_line or 'No summary available'}")
            print(f"Time: {elapsed_time:.2f}s")
            
            return chapter_result
            
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT - Chapter {chapter_name} tests timed out")
            self.results[chapter_name] = {
                'success': False,
                'passed': 0,
                'failed': 1,
                'skipped': 0,
                'elapsed_time': 300,
                'summary_line': 'TIMEOUT',
                'stdout': '',
                'stderr': 'Test execution timed out after 5 minutes',
                'returncode': -1
            }
            return self.results[chapter_name]
            
        except Exception as e:
            print(f"✗ ERROR - Exception running tests: {e}")
            self.results[chapter_name] = {
                'success': False,
                'passed': 0,
                'failed': 1,
                'skipped': 0,
                'elapsed_time': 0,
                'summary_line': f'ERROR: {e}',
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }
            return self.results[chapter_name]
    
    def run_all_tests(self, verbose: bool = False, chapters: List[str] = None, 
                      markers: str = None) -> Dict:
        """Run tests for all chapters or specified chapters."""
        print("\n" + "="*80)
        print("PRACTICAL FUNCTIONAL ANALYSIS FOR OPTICAL DESIGN")
        print("COMPREHENSIVE TEST RUNNER")
        print("="*80)
        
        chapter_dirs = self.get_chapter_test_dirs()
        
        if chapters:
            # Filter to only specified chapters
            chapter_dirs = [d for d in chapter_dirs if d.name in chapters]
            if not chapter_dirs:
                print(f"No chapters found matching: {chapters}")
                return {}
        
        print(f"Found {len(chapter_dirs)} chapters to test")
        
        total_start_time = time.time()
        
        for chapter_dir in chapter_dirs:
            self.run_chapter_tests(chapter_dir, verbose, markers)
        
        total_elapsed_time = time.time() - total_start_time
        
        # Print final summary
        self.print_summary(total_elapsed_time)
        
        return self.results
    
    def print_summary(self, total_elapsed_time: float):
        """Print a comprehensive summary of all test results."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        
        total_passed = sum(r['passed'] for r in self.results.values())
        total_failed = sum(r['failed'] for r in self.results.values())
        total_skipped = sum(r['skipped'] for r in self.results.values())
        total_tests = total_passed + total_failed + total_skipped
        
        successful_chapters = sum(1 for r in self.results.values() if r['success'])
        total_chapters = len(self.results)
        
        print(f"\nOverall Results:")
        print(f"  Total Chapters: {total_chapters}")
        print(f"  Successful Chapters: {successful_chapters}")
        print(f"  Failed Chapters: {total_chapters - successful_chapters}")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_passed}")
        print(f"  Failed: {total_failed}")
        print(f"  Skipped: {total_skipped}")
        print(f"  Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        print(f"  Total Time: {total_elapsed_time:.2f}s")
        
        print(f"\nDetailed Chapter Results:")
        print(f"{'Chapter':<30} {'Status':<10} {'Passed':<8} {'Failed':<8} {'Skipped':<8} {'Time':<8}")
        print(f"{'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        for chapter_name, result in sorted(self.results.items()):
            status = "✓" if result['success'] else "✗"
            print(f"{chapter_name:<30} {status:<10} {result['passed']:<8} {result['failed']:<8} {result['skipped']:<8} {result['elapsed_time']:<8.1f}")
        
        # Save summary to file
        summary_file = self.project_root / "test_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE TEST SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total Chapters: {total_chapters}\n")
            f.write(f"Successful Chapters: {successful_chapters}\n")
            f.write(f"Failed Chapters: {total_chapters - successful_chapters}\n")
            f.write(f"Total Tests: {total_tests}\n")
            f.write(f"Passed: {total_passed}\n")
            f.write(f"Failed: {total_failed}\n")
            f.write(f"Skipped: {total_skipped}\n")
            f.write(f"Success Rate: {(total_passed/total_tests*100):.1f}%\n" if total_tests > 0 else "N/A\n")
            f.write(f"Total Time: {total_elapsed_time:.2f}s\n\n")
            
            f.write("Detailed Chapter Results:\n")
            f.write(f"{'Chapter':<30} {'Status':<10} {'Passed':<8} {'Failed':<8} {'Skipped':<8} {'Time':<8}\n")
            f.write(f"{'-'*30} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}\n")
            
            for chapter_name, result in sorted(self.results.items()):
                status = "PASSED" if result['success'] else "FAILED"
                f.write(f"{chapter_name:<30} {status:<10} {result['passed']:<8} {result['failed']:<8} {result['skipped']:<8} {result['elapsed_time']:<8.1f}\n")
        
        print(f"\nSummary saved to: {summary_file}")
        
        # Return appropriate exit code
        return 0 if successful_chapters == total_chapters else 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for Practical Functional Analysis for Optical Design with Python"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--chapters", "-c",
        nargs="+",
        help="Specific chapters to test (e.g., chapter00_bridge_week chapter01_functional_foundations)"
    )
    
    parser.add_argument(
        "--markers", "-m",
        help="Pytest markers to use (e.g., 'not slow', 'visualization')"
    )
    
    parser.add_argument(
        "--list-chapters",
        action="store_true",
        help="List available chapters and exit"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.list_chapters:
        chapter_dirs = runner.get_chapter_test_dirs()
        print("Available chapters:")
        for chapter_dir in chapter_dirs:
            print(f"  {chapter_dir.name}")
        return 0
    
    # Run tests
    results = runner.run_all_tests(
        verbose=args.verbose,
        chapters=args.chapters,
        markers=args.markers
    )
    
    # Return appropriate exit code
    successful_chapters = sum(1 for r in results.values() if r['success'])
    total_chapters = len(results)
    
    return 0 if successful_chapters == total_chapters else 1


if __name__ == "__main__":
    sys.exit(main())