#!/usr/bin/env python3
"""
Clean up session log files in monitoring/logs directory.
Usage:
    python clean_session_logs.py           # Delete all session logs
    python clean_session_logs.py --keep 5  # Keep 5 most recent session logs
"""

import argparse
from pathlib import Path
import sys


def clean_session_logs(keep_count=0):
    """
    Remove session log files from monitoring/logs directory.

    Args:
        keep_count: Number of most recent session logs to keep (0 = delete all)
    """
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    logs_dir = project_root / "monitoring" / "logs"

    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return 0

    # Find all session log files
    session_files = sorted(
        logs_dir.glob("session_*.txt"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,  # Most recent first
    )

    if not session_files:
        print("No session log files found.")
        return 0

    # Determine which files to delete
    if keep_count > 0:
        files_to_delete = session_files[keep_count:]
        files_to_keep = session_files[:keep_count]
        print(
            f"Found {len(session_files)} session logs. Keeping {len(files_to_keep)} most recent."
        )
    else:
        files_to_delete = session_files
        print(f"Found {len(session_files)} session logs. Deleting all.")

    # Delete the files
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            print(f"Deleted: {file_path.name}")
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path.name}: {e}", file=sys.stderr)

    print(f"\nSummary: Deleted {deleted_count} session log(s)")
    return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description="Clean up session log files in monitoring/logs directory"
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=0,
        metavar="N",
        help="Number of most recent session logs to keep (default: 0, delete all)",
    )

    args = parser.parse_args()

    if args.keep < 0:
        print("Error: --keep value must be non-negative", file=sys.stderr)
        sys.exit(1)

    clean_session_logs(args.keep)


if __name__ == "__main__":
    main()
