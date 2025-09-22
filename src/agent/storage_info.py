#!/usr/bin/env python3
"""Standalone script to display ChromaDB storage information."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent.dependencies import validate_environment
from src.agent.local_storage_manager import LocalDatabaseManager


def main():
    """Display ChromaDB storage information."""
    try:
        # Load environment
        env_vars = validate_environment()

        # Create local database manager
        manager = LocalDatabaseManager(env_vars["CHROMADB_PATH"])

        # Display storage information
        manager.display_storage_info()

        # Offer cleanup option
        stats = manager.get_storage_stats()
        if stats.temp_files_count > 0:
            print(f"\nFound {stats.temp_files_count} temporary files.")
            response = input("Clean up temporary files older than 24 hours? (y/n): ")
            if response.lower() == "y":
                cleaned = manager.cleanup_temp_files(older_than_hours=24)
                print(f"✅ Cleaned up {cleaned} files")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
