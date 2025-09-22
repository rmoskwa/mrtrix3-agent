#!/usr/bin/env python
"""Generate coverage report for agent module."""

import subprocess
import sys


def run_coverage():
    """Run tests with coverage and display terminal report."""
    print("Running tests with coverage...")
    cmd = [
        "python",
        "-m",
        "pytest",
        "tests/agent",
        "tests/integration",
        "--cov=src/agent",
        "--cov-report=term-missing",
        "--cov-branch",
        "-q",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    return result.returncode == 0


if __name__ == "__main__":
    if not run_coverage():
        print("Tests failed. Fix failing tests before analyzing coverage.")
        sys.exit(1)
