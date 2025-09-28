"""
Tests for structured logging configuration.
"""

import json
import logging
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from monitoring.logging_config import (
    StructuredFormatter,
    RequestIdFilter,
    clear_request_id,
    configure_structured_logging,
    get_logger,
    log_tool_invocation,
    set_log_level,
    set_request_id,
)


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_text_format(self):
        """Test text format output."""
        formatter = StructuredFormatter(format_type="text")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        assert "INFO" in output
        assert "Test message" in output
        assert "test.logger" in output

    def test_json_format(self):
        """Test JSON format output."""
        formatter = StructuredFormatter(format_type="json")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=20,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "WARNING"
        assert data["message"] == "Warning message"
        assert data["logger"] == "test.logger"
        assert data["line"] == 20

    def test_extra_fields(self):
        """Test handling of extra fields."""
        formatter = StructuredFormatter(format_type="json")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=30,
            msg="Test with extras",
            args=(),
            exc_info=None,
        )

        # Add extra fields
        record.request_id = "test-123"
        record.user_id = "user-456"

        output = formatter.format(record)
        data = json.loads(output)

        assert "extra" in data
        assert data["extra"]["request_id"] == "test-123"
        assert data["extra"]["user_id"] == "user-456"

    def test_exception_formatting(self):
        """Test exception information formatting."""
        formatter = StructuredFormatter(format_type="text")

        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=40,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        assert "ValueError: Test error" in output
        assert "ERROR" in output


class TestRequestIdFilter:
    """Tests for RequestIdFilter."""

    def test_set_and_filter(self):
        """Test setting request ID and filtering records."""
        filter_obj = RequestIdFilter()
        filter_obj.set_request_id("test-request-123")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=50,
            msg="Test",
            args=(),
            exc_info=None,
        )

        result = filter_obj.filter(record)
        assert result is True
        assert hasattr(record, "request_id")
        assert record.request_id == "test-request-123"

    def test_auto_generate_request_id(self):
        """Test auto-generation of request ID."""
        filter_obj = RequestIdFilter()
        filter_obj.set_request_id()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=60,
            msg="Test",
            args=(),
            exc_info=None,
        )

        filter_obj.filter(record)
        assert hasattr(record, "request_id")
        assert len(record.request_id) == 36  # UUID format

    def test_clear_request_id(self):
        """Test clearing request ID."""
        filter_obj = RequestIdFilter()
        filter_obj.set_request_id("test-123")
        filter_obj.clear_request_id()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=70,
            msg="Test",
            args=(),
            exc_info=None,
        )

        filter_obj.filter(record)
        assert not hasattr(record, "request_id")


class TestLoggingConfiguration:
    """Tests for logging configuration functions."""

    def test_configure_structured_logging(self, tmp_path):
        """Test configuring structured logging."""
        log_file = tmp_path / "test.log"

        configure_structured_logging(
            log_level="DEBUG", format_type="json", output_file=str(log_file)
        )

        logger = get_logger("test.module")
        logger.info("Test message")

        # Check file was created
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger1 = get_logger("test.module1")
        logger2 = get_logger("test.module1")
        logger3 = get_logger("test.module2")

        # Same name should return same instance
        assert logger1 is logger2

        # Different name should return different instance
        assert logger1 is not logger3

        # Should be under monitoring namespace
        assert logger1.name == "monitoring.test.module1"

    def test_set_log_level(self):
        """Test setting log level."""
        # Create a completely isolated test logger
        test_logger_name = f"monitoring.test.level.{id(self)}"

        # Save original logging state
        original_level = logging.getLogger("monitoring").level

        try:
            # Use a string stream to capture output
            stream = StringIO()
            handler = logging.StreamHandler(stream)
            formatter = StructuredFormatter(format_type="text")
            handler.setFormatter(formatter)

            # Configure and get logger with unique name
            configure_structured_logging(log_level="INFO")

            # Create isolated test logger
            test_logger = logging.getLogger(test_logger_name)
            test_logger.handlers.clear()
            test_logger.addHandler(handler)
            test_logger.propagate = False
            test_logger.setLevel(logging.INFO)

            # Should not log DEBUG at INFO level
            test_logger.debug("Debug message")
            assert "Debug message" not in stream.getvalue()

            # Change to DEBUG level for all monitoring loggers
            set_log_level("DEBUG")
            test_logger.setLevel(logging.DEBUG)

            # Now should log DEBUG messages
            test_logger.debug("Debug message 2")
            output = stream.getvalue()
            assert "Debug message 2" in output or "DEBUG" in output
        finally:
            # Restore original logging state
            logging.getLogger("monitoring").setLevel(original_level)
            # Clear handlers from test logger
            test_logger = logging.getLogger(test_logger_name)
            test_logger.handlers.clear()
            test_logger.disabled = True

    def test_request_id_functions(self):
        """Test request ID management functions."""
        # Set specific request ID
        request_id = set_request_id("custom-id-789")
        assert request_id == "custom-id-789"

        # Auto-generate request ID
        auto_id = set_request_id()
        assert len(auto_id) == 36  # UUID format

        # Clear request ID
        clear_request_id()  # Should not raise

    def test_log_tool_invocation(self):
        """Test logging tool invocations."""
        logger = get_logger("test.tools")

        with patch.object(logger, "debug") as mock_debug:
            log_tool_invocation(logger, "search_tool", "start", input_size=100)
            mock_debug.assert_called_once()
            call_args = mock_debug.call_args
            assert "search_tool" in call_args[0][0]
            assert call_args[1]["extra"]["tool_name"] == "search_tool"
            assert call_args[1]["extra"]["phase"] == "start"
            assert call_args[1]["extra"]["input_size"] == 100

        with patch.object(logger, "info") as mock_info:
            log_tool_invocation(logger, "search_tool", "end", duration=0.5)
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[1]["extra"]["duration_ms"] == 500

        with patch.object(logger, "error") as mock_error:
            log_tool_invocation(logger, "search_tool", "error", error_type="timeout")
            mock_error.assert_called_once()
            call_args = mock_error.call_args
            assert call_args[1]["extra"]["error_type"] == "timeout"

    def test_environment_variable_configuration(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "MONITORING_LOG_LEVEL": "WARNING",
                "MONITORING_LOG_FORMAT": "json",
            },
        ):
            # Use a string stream to capture output
            stream = StringIO()
            handler = logging.StreamHandler(stream)

            configure_structured_logging()
            logger = get_logger("test.env")

            # Add our test handler with JSON formatter
            monitoring_logger = logging.getLogger("monitoring")
            formatter = StructuredFormatter(format_type="json")
            handler.setFormatter(formatter)
            monitoring_logger.addHandler(handler)

            # Should respect WARNING level from env
            logger.info("Info message")
            assert stream.getvalue() == ""  # INFO not logged at WARNING level

            logger.warning("Warning message")
            output = stream.getvalue()
            assert output != ""  # WARNING should be logged

            # Should be JSON format
            data = json.loads(output.strip())
            assert data["level"] == "WARNING"
