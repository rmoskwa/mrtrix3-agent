"""Integration tests for /sharefile command with actual file operations."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
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

    @patch("subprocess.run")
    def test_sharefile_file_not_exists(self, mock_subprocess):
        """Test /sharefile with non-existent file."""
        nonexistent_file = "/absolutely/nonexistent/path/file.nii"

        # Mock subprocess to simulate file not found error from sharefile.py
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = f"Error: Path '{nonexistent_file}' does not exist."
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        with patch("src.agent.slash_commands.console") as mock_console:
            result = self.handler.process_command(
                f"/sharefile {nonexistent_file} test query"
            )

            assert result.success is False
            assert result.continue_conversation is False

            # Should show file not found error
            error_call = mock_console.print.call_args[0][0]
            assert "does not exist" in error_call

    @patch("subprocess.run")
    def test_sharefile_empty_output(self, mock_subprocess):
        """Test /sharefile when sharefile returns empty output."""
        test_file = self.create_mock_nifti_file("test.nii")

        # Mock sharefile returning empty output - this is treated as success with empty content
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.stdout = ""  # Empty output
        mock_subprocess.return_value = mock_result

        result = self.handler.process_command(f"/sharefile {test_file} analyze")

        # With returncode=0, this is treated as success even with empty output
        assert result.success is True
        assert result.agent_input == ""  # Empty output is passed through

    def test_sharefile_integration_with_real_script_path(self):
        """Test that /sharefile correctly locates the sharefile.py script."""
        # Verify the script path calculation matches what's expected
        test_file = self.create_mock_nifti_file("test.nii")

        with patch("subprocess.run") as mock_subprocess:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "test output"
            mock_subprocess.return_value = mock_result

            self.handler.process_command(f"/sharefile {test_file} test query")

            # Verify the script path used
            call_args = mock_subprocess.call_args[0][0]
            script_path_used = call_args[1]

            # Should be the actual sharefile.py script
            assert script_path_used.endswith("sharefile.py")
            assert "workflows" in script_path_used
            assert Path(script_path_used).exists()
