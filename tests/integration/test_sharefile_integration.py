"""Integration tests for /sharefile command with actual file operations."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from src.agent.slash_commands import SlashCommandHandler


@pytest.mark.integration
class TestSharefileIntegration:
    """Integration tests for /sharefile command with realistic file scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = SlashCommandHandler()
        self.test_dir = Path(tempfile.mkdtemp(prefix="mrtrix_test_"))
        self.script_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "workflows"
            / "sharefile"
            / "sharefile.py"
        )

    def teardown_method(self):
        """Clean up test files."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_mock_nifti_file(
        self, filename: str, content: str = "fake nifti data"
    ) -> Path:
        """Create a mock NIfTI file for testing."""
        file_path = self.test_dir / filename
        file_path.write_text(content)
        return file_path

    @patch("src.workflows.sharefile.sharefile.main")
    def test_sharefile_file_not_exists(self, mock_sharefile_main):
        """Test /sharefile with non-existent file."""
        nonexistent_file = "/absolutely/nonexistent/path/file.nii"

        # Mock module to simulate file not found error from sharefile.py
        def mock_main():
            print(f"Error: Path '{nonexistent_file}' does not exist.")
            raise SystemExit(1)

        mock_sharefile_main.side_effect = mock_main

        with patch("src.agent.slash_commands.console") as mock_console:
            result = self.handler.process_command(
                f"/sharefile {nonexistent_file} test query"
            )

            assert result.success is False
            assert result.continue_conversation is False

            # Should show file not found error
            error_call = mock_console.print.call_args[0][0]
            assert "does not exist" in error_call

    @patch("src.workflows.sharefile.sharefile.main")
    def test_sharefile_empty_output(self, mock_sharefile_main):
        """Test /sharefile when sharefile returns empty output."""
        test_file = self.create_mock_nifti_file("test.nii")

        # Mock sharefile returning empty output - this is treated as success with empty content
        def mock_main():
            print("")  # Empty output
            raise SystemExit(0)

        mock_sharefile_main.side_effect = mock_main

        result = self.handler.process_command(f"/sharefile {test_file} analyze")

        # With returncode=0, this is treated as success even with empty output
        assert result.success is True
        assert result.agent_input == ""  # Empty output is passed through

    def test_sharefile_integration_with_real_script_path(self):
        """Test that /sharefile correctly uses module import."""
        # Verify that module import works correctly
        test_file = self.create_mock_nifti_file("test.nii")
        captured_argv = None

        with patch("src.workflows.sharefile.sharefile.main") as mock_sharefile_main:

            def mock_main():
                import sys as sys_module

                nonlocal captured_argv
                captured_argv = sys_module.argv.copy()
                print("test output")
                raise SystemExit(0)

            mock_sharefile_main.side_effect = mock_main

            result = self.handler.process_command(f"/sharefile {test_file} test query")

            # Verify the command succeeded
            assert result.success is True
            assert result.continue_conversation is True
            assert result.agent_input == "test output"

            # Verify module was called
            mock_sharefile_main.assert_called_once()

            # Verify sys.argv was set correctly
            assert captured_argv is not None
            assert "sharefile" in captured_argv[0]  # Program name
            assert str(test_file) in captured_argv[1]  # File path
            assert "test query" == captured_argv[2]  # Query
