#!/usr/bin/env python3
"""
Error Extractor for MRtrix3 Source Code

Extracts error messages from C++ and Python source files in the MRtrix3 repository,
mapping them to their containing functions for documentation purposes.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ErrorExtractor:
    """Extract error patterns from MRtrix3 source code"""

    def __init__(self, source_path: Path):
        """Initialize with path to MRtrix3 source (from sparse checkout)"""
        self.source_path = source_path

    def extract_from_command(self, command_name: str) -> Optional[Dict[str, str]]:
        """Extract error messages for a specific command, mapped to function context"""
        errors = {}

        # Look for C++ command file (correct path: directly in cmd/)
        cpp_file = self.source_path / "cmd" / f"{command_name}.cpp"
        if cpp_file.exists():
            cpp_errors = self.extract_from_file(cpp_file)
            if cpp_errors:
                errors.update(cpp_errors)

        # Look for Python command file in bin directory
        # These are full implementations, not just wrappers
        py_file = self.source_path / "bin" / command_name
        if py_file.exists():
            # Check if it's actually a Python script
            try:
                with open(py_file, "r") as f:
                    first_line = f.readline()
                    if "python" in first_line:
                        py_errors = self.extract_from_file(py_file)
                        if py_errors:
                            errors.update(py_errors)
            except Exception:
                pass
        return errors if errors else None

    def extract_from_file(self, file_path: Path) -> Dict[str, str]:
        """Extract error patterns from a source file with function context"""
        errors = {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()

            # Determine if it's a Python file (either .py extension or shebang)
            is_python = file_path.suffix == ".py" or (lines and "python" in lines[0])

            # First, extract all multi-line exceptions/errors from the full content
            if file_path.suffix == ".cpp":
                # Extract multi-line exceptions - capture everything between Exception( and );
                exception_pattern = r"throw\s+(?:Exception|InvalidImageException|\w+Exception)\s*\((.*?)\);"
                exception_matches = re.findall(exception_pattern, content, re.DOTALL)

                # Also capture WARN and ERROR macros
                warn_pattern = r"WARN\s*\((.*?)\);"
                warn_matches = re.findall(warn_pattern, content, re.DOTALL)

                error_pattern = r"ERROR\s*\((.*?)\);"
                error_matches = re.findall(error_pattern, content, re.DOTALL)

                all_error_contents = exception_matches + warn_matches + error_matches

                # Clean up the extracted error messages
                cleaned_errors = []
                for error_content in all_error_contents:
                    # Remove string concatenation operators and clean up
                    cleaned = re.sub(r"\s*\+\s*", " ", error_content)
                    # Remove variable references like str(something)
                    cleaned = re.sub(r"str\([^)]+\)", "<value>", cleaned)
                    # Remove quotes and clean whitespace
                    cleaned = re.sub(r'["\']\s*["\']\s*', "", cleaned)
                    cleaned = re.sub(r'^["\']|["\']$', "", cleaned.strip())
                    # Replace escaped quotes
                    cleaned = cleaned.replace("\\'", "'").replace('\\"', '"')
                    # Collapse whitespace
                    cleaned = " ".join(cleaned.split())
                    if cleaned:
                        cleaned_errors.append(cleaned)

            # Now find which function each error belongs to
            current_function = "global"

            for line in lines:
                # Try to identify current function context
                if file_path.suffix == ".cpp":
                    # C++ function detection (simplified)
                    func_match = re.match(
                        r"\s*(?:void|int|bool|auto|class\s+\w+::\w+)\s+(\w+)\s*\(", line
                    )
                    if func_match:
                        current_function = func_match.group(1)
                    # Check for class method definitions (not just usage)
                    # Only update function if it's a method definition, not a method call
                    elif re.match(r"\s*\w+::\w+\s*\([^)]*\)\s*\{", line):
                        method_match = re.search(r"(\w+)::\w+\s*\(", line)
                        if method_match:
                            current_function = method_match.group(1)

                    # Check if this line contains an error/exception
                    if any(keyword in line for keyword in ["throw", "WARN", "ERROR"]):
                        # Find the matching cleaned error
                        for cleaned_error in cleaned_errors:
                            # Extract a key part of the error to match
                            error_key = (
                                cleaned_error[:30]
                                if len(cleaned_error) > 30
                                else cleaned_error
                            )
                            # Store the error with its function context
                            if current_function not in errors and error_key not in str(
                                errors.values()
                            ):
                                errors[current_function] = cleaned_error
                                break

            # Handle Python files
            if is_python:
                # Extract multi-line Python errors from full content
                raise_pattern = r"raise\s+\w+(?:Error|Exception)?\s*\((.*?)\)"
                raise_matches = re.findall(raise_pattern, content, re.DOTALL)

                app_error_pattern = r"app\.(?:error|warn)\s*\((.*?)\)"
                app_matches = re.findall(app_error_pattern, content, re.DOTALL)

                py_error_contents = raise_matches + app_matches

                # Clean up Python errors
                py_cleaned_errors = []
                for error_content in py_error_contents:
                    # Remove string formatting and clean up
                    cleaned = re.sub(r"%\s*\([^)]+\)", "<value>", error_content)
                    cleaned = re.sub(r"\{\}", "<value>", cleaned)
                    cleaned = re.sub(r"\.format\([^)]+\)", "", cleaned)
                    # Remove quotes
                    cleaned = re.sub(r'^["\']|["\']$', "", cleaned.strip())
                    # Collapse whitespace
                    cleaned = " ".join(cleaned.split())
                    if cleaned:
                        py_cleaned_errors.append(cleaned)

                # Map Python errors to functions
                current_function = "global"
                for line in lines:
                    # Python function detection
                    func_match = re.match(r"\s*def\s+(\w+)\s*\(", line)
                    if func_match:
                        current_function = func_match.group(1)

                    # Check if this line contains an error
                    if any(
                        keyword in line
                        for keyword in ["raise", "app.error", "app.warn"]
                    ):
                        for cleaned_error in py_cleaned_errors:
                            error_key = (
                                cleaned_error[:30]
                                if len(cleaned_error) > 30
                                else cleaned_error
                            )
                            if current_function not in errors and error_key not in str(
                                errors.values()
                            ):
                                errors[current_function] = cleaned_error
                                break

        except Exception:
            pass

        return errors
