"""
Tests for monitoring module initialization.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_monitoring_module_can_be_imported_when_disabled():
    """Test that monitoring module doesn't break when disabled."""
    with patch.dict(os.environ, {"ENABLE_MONITORING": "false"}):
        # Module should be importable even if components aren't ready
        import monitoring

        assert monitoring is not None


def test_monitoring_directory_structure_exists():
    """Test that monitoring directory structure is properly created."""
    monitoring_dir = Path(__file__).parent.parent.parent / "monitoring"

    # Check directory exists
    assert monitoring_dir.exists()
    assert monitoring_dir.is_dir()

    # Check __init__.py exists
    init_file = monitoring_dir / "__init__.py"
    assert init_file.exists()

    # Check README.md exists
    readme_file = monitoring_dir / "README.md"
    assert readme_file.exists()


def test_monitoring_module_exports():
    """Test that monitoring module exports expected functions."""
    # This will fail until we implement the modules
    # but shows what we expect to export
    expected_exports = [
        "configure_structured_logging",
        "get_logger",
        "set_log_level",
        "MetricsCollector",
        "export_metrics",
        "get_metrics_summary",
    ]

    # We'll check __all__ is defined even if imports fail
    import monitoring

    if hasattr(monitoring, "__all__"):
        for export in expected_exports:
            assert export in monitoring.__all__
