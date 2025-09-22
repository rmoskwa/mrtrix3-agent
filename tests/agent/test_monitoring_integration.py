"""
Tests for monitoring integration helper.
"""

import asyncio
import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from src.agent.monitoring_integration import (
    get_dual_logger,
    log_dual,
    track_tool_invocation,
    record_metric,
    set_monitoring_request_id,
    clear_monitoring_request_id,
)


class TestGetDualLogger:
    """Tests for get_dual_logger function."""

    def test_without_monitoring_enabled(self):
        """Test getting loggers when monitoring is disabled."""
        with patch.dict(os.environ, {"ENABLE_MONITORING": "false"}):
            # Need to reload module to pick up env change
            import importlib
            import src.agent.monitoring_integration as mi

            importlib.reload(mi)

            user_logger, monitoring_logger = mi.get_dual_logger("test.module")

            assert user_logger is not None
            assert isinstance(user_logger, logging.Logger)
            assert user_logger.name == "test.module"
            assert monitoring_logger is None

    @patch("src.agent.monitoring_integration.monitoring_available", True)
    @patch("src.agent.monitoring_integration.get_monitoring_logger")
    def test_with_monitoring_enabled(self, mock_get_monitoring_logger):
        """Test getting loggers when monitoring is enabled."""
        mock_monitoring_logger = MagicMock()
        mock_get_monitoring_logger.return_value = mock_monitoring_logger

        user_logger, monitoring_logger = get_dual_logger("test.module")

        assert user_logger is not None
        assert isinstance(user_logger, logging.Logger)
        assert user_logger.name == "test.module"
        assert monitoring_logger == mock_monitoring_logger
        mock_get_monitoring_logger.assert_called_once_with("test.module")


class TestLogDual:
    """Tests for log_dual function."""

    def test_log_to_user_only(self):
        """Test logging to user logger only when monitoring is None."""
        user_logger = MagicMock(spec=logging.Logger)
        user_logger.info = MagicMock()

        log_dual(
            user_logger,
            None,
            "info",
            "User message",
            "Monitoring message",
            extra_field="value",
        )

        user_logger.info.assert_called_once_with("User message")

    def test_log_to_both_loggers(self):
        """Test logging to both user and monitoring loggers."""
        user_logger = MagicMock(spec=logging.Logger)
        user_logger.warning = MagicMock()

        monitoring_logger = MagicMock(spec=logging.Logger)
        monitoring_logger.warning = MagicMock()

        log_dual(
            user_logger,
            monitoring_logger,
            "warning",
            "User warning",
            "Detailed monitoring warning",
            request_id="123",
            tool_name="test_tool",
        )

        user_logger.warning.assert_called_once_with("User warning")
        monitoring_logger.warning.assert_called_once_with(
            "Detailed monitoring warning",
            extra={"request_id": "123", "tool_name": "test_tool"},
        )

    def test_log_with_default_monitoring_message(self):
        """Test logging when monitoring message is not provided."""
        user_logger = MagicMock(spec=logging.Logger)
        user_logger.error = MagicMock()

        monitoring_logger = MagicMock(spec=logging.Logger)
        monitoring_logger.error = MagicMock()

        log_dual(
            user_logger,
            monitoring_logger,
            "error",
            "Error occurred",
            error_type="ValueError",
        )

        user_logger.error.assert_called_once_with("Error occurred")
        monitoring_logger.error.assert_called_once_with(
            "Error occurred",
            extra={"error_type": "ValueError"},
        )


class TestTrackToolInvocation:
    """Tests for track_tool_invocation decorator."""

    @pytest.mark.asyncio
    @patch("src.agent.monitoring_integration.monitoring_available", True)
    @patch("src.agent.monitoring_integration.get_global_collector")
    @patch("src.agent.monitoring_integration.get_monitoring_logger")
    @patch("src.agent.monitoring_integration.log_tool_invocation")
    async def test_async_function_success(
        self,
        mock_log_tool,
        mock_get_logger,
        mock_get_collector,
    ):
        """Test tracking async function invocation success."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_metrics = MagicMock()
        mock_get_collector.return_value = mock_metrics

        @track_tool_invocation("test_tool")
        async def async_test_function(x, y):
            await asyncio.sleep(0.01)
            return x + y

        result = await async_test_function(2, 3)

        assert result == 5
        mock_metrics.record_counter.assert_called_with("request_count")
        mock_metrics.record_latency.assert_called_once()

        # Check log_tool_invocation calls
        assert mock_log_tool.call_count == 2
        start_call = mock_log_tool.call_args_list[0]
        assert start_call[0][1] == "test_tool"
        assert start_call[0][2] == "start"

        end_call = mock_log_tool.call_args_list[1]
        assert end_call[0][1] == "test_tool"
        assert end_call[0][2] == "end"
        assert "duration" in end_call[1]
        assert end_call[1]["success"] is True

    @pytest.mark.asyncio
    @patch("src.agent.monitoring_integration.monitoring_available", True)
    @patch("src.agent.monitoring_integration.get_global_collector")
    @patch("src.agent.monitoring_integration.get_monitoring_logger")
    @patch("src.agent.monitoring_integration.log_tool_invocation")
    async def test_async_function_error(
        self,
        mock_log_tool,
        mock_get_logger,
        mock_get_collector,
    ):
        """Test tracking async function invocation with error."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_metrics = MagicMock()
        mock_get_collector.return_value = mock_metrics

        @track_tool_invocation("error_tool")
        async def async_error_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await async_error_function()

        mock_metrics.record_counter.assert_any_call("request_count")
        mock_metrics.record_counter.assert_any_call("error_count")

        # Check error logging
        error_call = mock_log_tool.call_args_list[-1]
        assert error_call[0][1] == "error_tool"
        assert error_call[0][2] == "error"
        assert error_call[1]["error_type"] == "ValueError"

    @patch("src.agent.monitoring_integration.monitoring_available", True)
    @patch("src.agent.monitoring_integration.get_global_collector")
    @patch("src.agent.monitoring_integration.get_monitoring_logger")
    @patch("src.agent.monitoring_integration.log_tool_invocation")
    def test_sync_function_success(
        self,
        mock_log_tool,
        mock_get_logger,
        mock_get_collector,
    ):
        """Test tracking sync function invocation success."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_metrics = MagicMock()
        mock_get_collector.return_value = mock_metrics

        @track_tool_invocation("sync_tool")
        def sync_test_function(x, y):
            return x * y

        result = sync_test_function(4, 5)

        assert result == 20
        mock_metrics.record_counter.assert_called_with("request_count")
        mock_metrics.record_latency.assert_called_once()

    @patch("src.agent.monitoring_integration.monitoring_available", False)
    def test_no_monitoring(self):
        """Test decorator when monitoring is not available."""

        @track_tool_invocation("no_monitor_tool")
        def test_function(x):
            return x * 2

        result = test_function(3)
        assert result == 6  # Function should work normally


class TestRecordMetric:
    """Tests for record_metric function."""

    @patch("src.agent.monitoring_integration.monitoring_available", True)
    @patch("src.agent.monitoring_integration.get_global_collector")
    def test_record_counter(self, mock_get_collector):
        """Test recording a counter metric."""
        mock_metrics = MagicMock()
        mock_get_collector.return_value = mock_metrics

        record_metric("counter", "test_counter", 5.0, environment="test")

        mock_metrics.record_counter.assert_called_once_with(
            "test_counter", 5.0, environment="test"
        )

    @patch("src.agent.monitoring_integration.monitoring_available", True)
    @patch("src.agent.monitoring_integration.get_global_collector")
    def test_record_gauge(self, mock_get_collector):
        """Test recording a gauge metric."""
        mock_metrics = MagicMock()
        mock_get_collector.return_value = mock_metrics

        record_metric("gauge", "memory_usage", 1024.5, process="agent")

        mock_metrics.record_gauge.assert_called_once_with(
            "memory_usage", 1024.5, process="agent"
        )

    @patch("src.agent.monitoring_integration.monitoring_available", True)
    @patch("src.agent.monitoring_integration.get_global_collector")
    def test_record_latency(self, mock_get_collector):
        """Test recording a latency metric."""
        mock_metrics = MagicMock()
        mock_get_collector.return_value = mock_metrics

        record_metric("latency", "api_call", 0.250, endpoint="/search")

        mock_metrics.record_latency.assert_called_once_with(
            "api_call", 0.250, endpoint="/search"
        )

    @patch("src.agent.monitoring_integration.monitoring_available", False)
    @patch("src.agent.monitoring_integration.get_global_collector")
    def test_no_recording_when_disabled(self, mock_get_collector):
        """Test that metrics are not recorded when monitoring is disabled."""
        record_metric("counter", "test", 1.0)

        mock_get_collector.assert_not_called()


class TestRequestIdFunctions:
    """Tests for request ID functions."""

    @patch("src.agent.monitoring_integration.monitoring_available", True)
    @patch("src.agent.monitoring_integration.set_request_id")
    def test_set_monitoring_request_id(self, mock_set_request_id):
        """Test setting monitoring request ID."""
        mock_set_request_id.return_value = "test-123"

        result = set_monitoring_request_id("test-123")

        assert result == "test-123"
        mock_set_request_id.assert_called_once_with("test-123")

    @patch("src.agent.monitoring_integration.monitoring_available", True)
    @patch("src.agent.monitoring_integration.clear_request_id")
    def test_clear_monitoring_request_id(self, mock_clear_request_id):
        """Test clearing monitoring request ID."""
        clear_monitoring_request_id()

        mock_clear_request_id.assert_called_once()

    @patch("src.agent.monitoring_integration.monitoring_available", False)
    def test_request_id_when_disabled(self):
        """Test request ID functions when monitoring is disabled."""
        result = set_monitoring_request_id("test")
        assert result is None

        clear_monitoring_request_id()  # Should not raise
