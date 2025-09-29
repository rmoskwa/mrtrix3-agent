"""
Sharefile module for MRtrix3 Assistant.
Provides functionality to analyze MRtrix3 image files using mrinfo.
"""

from .sharefile import main, run_mrinfo, load_file_contents, build_prompt, cleanup_json

__all__ = ["main", "run_mrinfo", "load_file_contents", "build_prompt", "cleanup_json"]
