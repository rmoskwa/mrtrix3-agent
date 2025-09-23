"""Test session logging functionality."""

from pathlib import Path
from unittest.mock import patch
import tempfile

from src.agent.session_logger import (
    SessionLogger,
    initialize_session_logger,
    cleanup_session_logger,
)


class TestSessionLogger:
    """Test the SessionLogger class for file-only logging."""

    def test_session_logger_initialization_disabled(self):
        """Test that SessionLogger doesn't create files when disabled."""
        logger = SessionLogger()
        log_path = logger.initialize(collect_logs=False)

        assert log_path is None
        assert not logger.enabled
        assert logger.log_file is None

    def test_session_logger_initialization_enabled(self):
        """Test that SessionLogger creates log file when enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create the monitoring/logs structure
            logs_dir = Path(tmp_dir) / "monitoring" / "logs"
            logs_dir.mkdir(parents=True)

            # Patch Path to use our temp directory
            with patch("src.agent.session_logger.Path") as mock_path:
                mock_path.return_value = logs_dir

                logger = SessionLogger()
                log_path = logger.initialize(collect_logs=True)

                assert logger.enabled
                assert log_path is not None
                assert logger.log_file is not None

                # Clean up
                logger.cleanup()

    def test_session_logger_respects_enabled_flag(self):
        """Test that logging methods respect the enabled flag."""
        import logging

        # Test with disabled logger
        logger = SessionLogger()
        logger.enabled = False

        with patch.object(logging, "getLogger") as mock_get_logger:
            logger.log_user_query("test query")
            logger.log_gemini_response("test response")
            logger.log_rag_results([])

            # Should not call getLogger when disabled
            mock_get_logger.assert_not_called()

    def test_session_logger_cleanup(self):
        """Test that cleanup properly closes resources."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir = Path(tmp_dir) / "monitoring" / "logs"
            logs_dir.mkdir(parents=True)

            with patch("src.agent.session_logger.Path") as mock_path:
                mock_path.return_value = logs_dir

                logger = SessionLogger()
                logger.initialize(collect_logs=True)

                # Ensure file is open
                assert logger.log_file is not None
                assert not logger.log_file.closed

                # Clean up
                logger.cleanup()

                # File should be closed
                assert logger.log_file.closed


class TestSessionLoggerGlobal:
    """Test global session logger functions."""

    def test_initialize_session_logger_disabled(self):
        """Test initializing with logging disabled."""
        logger = initialize_session_logger(collect_logs=False)
        assert logger is None

    def test_initialize_session_logger_enabled(self):
        """Test initializing with logging enabled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir = Path(tmp_dir) / "monitoring" / "logs"
            logs_dir.mkdir(parents=True)

            with patch("src.agent.session_logger.Path") as mock_path:
                mock_path.return_value = logs_dir

                logger = initialize_session_logger(collect_logs=True)
                assert logger is not None
                assert logger.enabled

                # Clean up
                cleanup_session_logger()

    def test_cleanup_session_logger(self):
        """Test that cleanup_session_logger works correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir = Path(tmp_dir) / "monitoring" / "logs"
            logs_dir.mkdir(parents=True)

            with patch("src.agent.session_logger.Path") as mock_path:
                mock_path.return_value = logs_dir

                # Initialize logger
                logger = initialize_session_logger(collect_logs=True)
                assert logger is not None

                # Clean up
                cleanup_session_logger()

                # Logger should be cleaned up
                import src.agent.session_logger as sl

                assert sl._session_logger is None


class TestLogFileCreation:
    """Test actual log file creation and content."""

    def test_log_file_structure(self):
        """Test that the log directory structure is correct."""
        logs_dir = Path("monitoring/logs")

        # Directory should exist (created during setup)
        assert logs_dir.exists()
        assert logs_dir.is_dir()

        # Should have a .gitkeep file
        gitkeep = logs_dir / ".gitkeep"
        assert gitkeep.exists()

    def test_log_file_contains_session_markers(self):
        """Test that log files contain proper session markers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir = Path(tmp_dir) / "monitoring" / "logs"
            logs_dir.mkdir(parents=True)

            with patch("src.agent.session_logger.Path") as mock_path:
                mock_path.return_value = logs_dir

                logger = SessionLogger()
                log_path = logger.initialize(collect_logs=True)

                # Clean up (writes end marker)
                logger.cleanup()

                # Read the log file
                content = Path(log_path).read_text()

                # Should have session markers
                assert "[Session started:" in content
                assert "[Log file:" in content
                assert "[Session ended:" in content
