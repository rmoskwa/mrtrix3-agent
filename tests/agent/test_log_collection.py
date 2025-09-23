"""Test log collection functionality."""

import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

from src.agent.cli import TeeWriter


class TestTeeWriter:
    """Test the TeeWriter class for dual output."""

    def test_tee_writer_writes_to_both_streams(self, tmp_path):
        """Test that TeeWriter writes to both original stream and log file."""
        # Create a mock for the original stream
        original = Mock()
        original.write = Mock(return_value=None)
        original.flush = Mock(return_value=None)

        # Create a temporary log file
        log_file_path = tmp_path / "test.log"
        with open(log_file_path, "w") as log_file:
            tee = TeeWriter(original, log_file)

            # Test write
            tee.write("Test message")
            original.write.assert_called_once_with("Test message")

            # Test flush
            tee.flush()
            original.flush.assert_called_once()

        # Check that log file contains the message
        assert log_file_path.read_text() == "Test message"

    def test_tee_writer_delegates_attributes(self):
        """Test that TeeWriter delegates unknown attributes to original stream."""
        original = Mock()
        original.custom_attr = "custom_value"
        log_file = Mock()

        tee = TeeWriter(original, log_file)
        assert tee.custom_attr == "custom_value"

    def test_tee_writer_multiple_writes(self, tmp_path):
        """Test that TeeWriter handles multiple writes correctly."""
        original = Mock()
        original.write = Mock(return_value=None)
        original.flush = Mock(return_value=None)

        log_file_path = tmp_path / "test.log"
        with open(log_file_path, "w") as log_file:
            tee = TeeWriter(original, log_file)

            # Multiple writes
            tee.write("Line 1\n")
            tee.write("Line 2\n")
            tee.write("Line 3\n")
            tee.flush()

        # Check both outputs
        assert original.write.call_count == 3
        assert original.flush.call_count == 1
        assert log_file_path.read_text() == "Line 1\nLine 2\nLine 3\n"


class TestLogCollectionIntegration:
    """Test log collection integration."""

    def test_log_file_creation(self):
        """Test that log files are created in the correct location."""
        # Create logs directory
        logs_dir = Path("monitoring/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create a test log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = logs_dir / f"test_{timestamp}.txt"

        # Test writing to log file with TeeWriter
        original_stdout = sys.stdout
        try:
            with open(log_file_path, "w") as log_file:
                sys.stdout = TeeWriter(original_stdout, log_file)
                print("Test log message")
                print("Another test message")
        finally:
            sys.stdout = original_stdout

        # Verify file exists and contains expected content
        assert log_file_path.exists()
        content = log_file_path.read_text()
        assert "Test log message" in content
        assert "Another test message" in content

        # Cleanup
        log_file_path.unlink()

    def test_log_directory_structure(self):
        """Test that the log directory structure is correct."""
        logs_dir = Path("monitoring/logs")

        # Directory should exist (created during setup)
        assert logs_dir.exists()
        assert logs_dir.is_dir()

        # Should have a .gitkeep file
        gitkeep = logs_dir / ".gitkeep"
        assert gitkeep.exists()
