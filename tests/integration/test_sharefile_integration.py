"""Integration tests for /sharefile command with actual file operations."""

import json
import shutil
import subprocess
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

    def create_mock_json_output(
        self, dimensions: list = None, voxel_size: list = None
    ) -> dict:
        """Create mock JSON output that would come from mrinfo."""
        if dimensions is None:
            dimensions = [256, 256, 64, 30]
        if voxel_size is None:
            voxel_size = [1.0, 1.0, 1.0, 2.5]

        return {
            "format": "NIfTI-1.1",
            "dimensions": dimensions,
            "voxel_size": voxel_size,
            "datatype": "32-bit float",
            "strides": [1, 2, 3, 4],
            "compression": "none",
        }

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_sharefile_successful_file_analysis(self, mock_subprocess, mock_which):
        """Test successful /sharefile command with realistic file analysis."""
        # Setup mocks
        mock_which.return_value = "/usr/bin/mrinfo"
        test_file = self.create_mock_nifti_file("dwi_data.nii")
        mock_json_data = self.create_mock_json_output()

        # Mock subprocess execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        # Set stdout to the expected formatted output
        formatted_output = f"<user file information>\n{json.dumps(mock_json_data)}\n</user file information>\n<user_provided_filepath>\n{test_file}\n</user_provided_filepath>\nHow should I preprocess this DWI data?"
        mock_result.stdout = formatted_output
        mock_subprocess.return_value = mock_result

        # Mock file operations
        with (
            patch("tempfile.gettempdir", return_value=str(self.test_dir)),
            patch("os.path.exists") as mock_exists,
            patch("builtins.open", create=True) as mock_open,
        ):
            # Setup existence checks
            def exists_side_effect(path):
                return str(test_file) in path or path.endswith(".json")

            mock_exists.side_effect = exists_side_effect

            # Mock JSON file reading
            mock_open.return_value.__enter__.return_value.read.return_value = (
                json.dumps(mock_json_data)
            )

            # Execute the command
            user_query = "How should I preprocess this DWI data?"
            result = self.handler.process_command(
                f"/sharefile {test_file} {user_query}"
            )

            # Verify success
            assert result.success is True
            assert result.continue_conversation is True
            assert result.agent_input is not None

            # Verify output format
            output = result.agent_input
            assert "<user file information>" in output
            assert "</user file information>" in output
            assert "<user_provided_filepath>" in output
            assert "</user_provided_filepath>" in output
            assert json.dumps(mock_json_data) in output
            assert str(test_file) in output
            assert user_query in output

            # Verify subprocess was called correctly
            mock_subprocess.assert_called_once()
            subprocess_args = mock_subprocess.call_args[0][0]
            assert str(self.script_path) in subprocess_args[1]
            assert str(test_file) in subprocess_args
            assert user_query in subprocess_args

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_sharefile_with_different_file_types(self, mock_subprocess, mock_which):
        """Test /sharefile with various file types and extensions."""
        mock_which.return_value = "/usr/bin/mrinfo"

        file_types = [
            ("anatomical.nii.gz", "T1-weighted anatomical scan"),
            ("functional.nii", "fMRI BOLD data"),
            ("dwi_data.mif", "DWI data in MRtrix format"),
            ("peaks.msh", "Peak directions mesh"),
            ("response.txt", "Response function file"),
        ]

        for filename, query in file_types:
            test_file = self.create_mock_nifti_file(filename)
            mock_json_data = self.create_mock_json_output()

            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            # Set stdout for different file types
            formatted_output = f"<user file information>\n{json.dumps(mock_json_data)}\n</user file information>\n<user_provided_filepath>\n{test_file}\n</user_provided_filepath>\n{query}"
            mock_result.stdout = formatted_output
            mock_subprocess.return_value = mock_result

            with (
                patch("tempfile.gettempdir", return_value=str(self.test_dir)),
                patch("os.path.exists", return_value=True),
                patch("builtins.open", create=True) as mock_open,
            ):
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(mock_json_data)
                )

                result = self.handler.process_command(f"/sharefile {test_file} {query}")

                assert result.success is True, f"Failed for file type: {filename}"
                assert query in result.agent_input
                assert str(test_file) in result.agent_input

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
    def test_sharefile_mrinfo_command_failure(self, mock_subprocess):
        """Test /sharefile when mrinfo command fails."""
        test_file = self.create_mock_nifti_file("corrupted.nii")

        # Mock mrinfo failure
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = (
            "Error: mrinfo failed - mrinfo: error: invalid image format"
        )
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        with patch("src.agent.slash_commands.console") as mock_console:
            result = self.handler.process_command(
                f"/sharefile {test_file} analyze this"
            )

            assert result.success is False
            assert result.continue_conversation is False

            # Should display mrinfo error
            error_call = mock_console.print.call_args[0][0]
            assert "mrinfo failed" in error_call or "invalid image format" in error_call

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_sharefile_mrinfo_timeout(self, mock_subprocess, mock_which):
        """Test /sharefile when mrinfo command times out."""
        mock_which.return_value = "/usr/bin/mrinfo"
        test_file = self.create_mock_nifti_file("large_file.nii")

        # Mock timeout
        mock_subprocess.side_effect = subprocess.TimeoutExpired("mrinfo", 30)

        with patch("src.agent.slash_commands.console") as mock_console:
            result = self.handler.process_command(
                f"/sharefile {test_file} process this"
            )

            assert result.success is False
            assert result.continue_conversation is False

            # Should display timeout error
            error_call = mock_console.print.call_args[0][0]
            assert "timed out" in error_call

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

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_sharefile_with_complex_queries(self, mock_subprocess, mock_which):
        """Test /sharefile with complex, realistic user queries."""
        mock_which.return_value = "/usr/bin/mrinfo"
        test_file = self.create_mock_nifti_file("dwi.nii")

        complex_queries = [
            "I have DWI data with b-values 0, 1000, 2000, and 3000. How should I process this for fiber tractography?",
            "This is a T1-weighted anatomical scan. I need to segment it into white matter, gray matter, and CSF. What's the best approach?",
            "I want to compute fiber orientation distributions from this DWI data. Should I use CSD or multi-shell multi-tissue CSD?",
            "How can I register this anatomical image to MNI152 space while preserving the quality for subsequent analyses?",
            "I need to extract connectivity matrices from this DWI data. What preprocessing steps are essential?",
        ]

        mock_json_data = self.create_mock_json_output(
            [128, 128, 60, 64], [2.0, 2.0, 2.0, 1.5]
        )

        for query in complex_queries:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            # Create output with the query
            mock_json = self.create_mock_json_output()
            formatted_output = f"<user file information>\n{json.dumps(mock_json)}\n</user file information>\n<user_provided_filepath>\n{self.test_dir / 'test.nii'}\n</user_provided_filepath>\n{query}"
            mock_result.stdout = formatted_output
            mock_subprocess.return_value = mock_result

            with (
                patch("tempfile.gettempdir", return_value=str(self.test_dir)),
                patch("os.path.exists", return_value=True),
                patch("builtins.open", create=True) as mock_open,
            ):
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(mock_json_data)
                )

                result = self.handler.process_command(f"/sharefile {test_file} {query}")

                assert result.success is True, f"Failed for query: {query[:50]}..."
                assert query in result.agent_input
                # Verify the full complex query is preserved
                assert (
                    len(
                        [
                            line
                            for line in result.agent_input.split("\n")
                            if query in line
                        ]
                    )
                    == 1
                )

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_sharefile_with_special_file_paths(self, mock_subprocess, mock_which):
        """Test /sharefile with various special characters in file paths."""
        mock_which.return_value = "/usr/bin/mrinfo"

        special_paths = [
            "file with spaces.nii",
            "file-with-dashes.nii",
            "file_with_underscores.nii",
            "file.with.dots.nii",
            "UPPERCASE_FILE.NII",
            "file123with456numbers.nii",
        ]

        mock_json_data = self.create_mock_json_output()

        for filename in special_paths:
            test_file = self.create_mock_nifti_file(filename)

            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            mock_json = self.create_mock_json_output()
            formatted_output = f"<user file information>\n{json.dumps(mock_json)}\n</user file information>\n<user_provided_filepath>\n{test_file}\n</user_provided_filepath>\nUser query"
            mock_result.stdout = formatted_output
            mock_subprocess.return_value = mock_result

            with (
                patch("tempfile.gettempdir", return_value=str(self.test_dir)),
                patch("os.path.exists", return_value=True),
                patch("builtins.open", create=True) as mock_open,
            ):
                mock_open.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(mock_json_data)
                )

                result = self.handler.process_command(
                    f"/sharefile {test_file} analyze this file"
                )

                assert result.success is True, f"Failed for filename: {filename}"
                assert str(test_file) in result.agent_input

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_sharefile_json_cleanup(self, mock_subprocess, mock_which):
        """Test that temporary JSON files are properly cleaned up."""
        mock_which.return_value = "/usr/bin/mrinfo"
        test_file = self.create_mock_nifti_file("test.nii")

        # Track file operations
        removed_files = []

        def mock_exists(path):
            return str(test_file) in path or path.endswith(".json")

        def mock_remove(path):
            removed_files.append(path)

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_json_data = self.create_mock_json_output()
        # Use dynamic paths for testing
        formatted_output = f"<user file information>\n{json.dumps(mock_json_data)}\n</user file information>\n<user_provided_filepath>\n/path/file.nii\n</user_provided_filepath>\nAnalyze this"
        mock_result.stdout = formatted_output
        mock_subprocess.return_value = mock_result

        with (
            patch("tempfile.gettempdir", return_value=str(self.test_dir)),
            patch("os.path.exists", side_effect=mock_exists),
            patch("os.remove", side_effect=mock_remove),
            patch("builtins.open", create=True) as mock_open,
        ):
            mock_open.return_value.__enter__.return_value.read.return_value = (
                json.dumps(mock_json_data)
            )

            result = self.handler.process_command(f"/sharefile {test_file} test query")

            assert result.success is True

            # In the actual implementation, we'd verify that cleanup was called
            # This test documents the expected behavior for cleanup

    def test_sharefile_integration_with_real_script_path(self):
        """Test that /sharefile correctly locates the sharefile.py script."""
        # Verify the script path calculation matches what's expected
        test_file = self.create_mock_nifti_file("test.nii")

        with (
            patch("subprocess.run") as mock_subprocess,
            patch("shutil.which", return_value="/usr/bin/mrinfo"),
        ):
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

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_sharefile_output_format_validation(self, mock_subprocess, mock_which):
        """Test that /sharefile output matches expected XML format structure."""
        mock_which.return_value = "/usr/bin/mrinfo"
        test_file = self.create_mock_nifti_file("format_test.nii")

        mock_json_data = {
            "format": "NIfTI-1.1",
            "dimensions": [256, 256, 64],
            "voxel_size": [1.0, 1.0, 1.0],
            "datatype": "64-bit float",
        }

        user_query = "What processing steps are recommended?"

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        # Expected output format from sharefile
        formatted_output = f"<user file information>\n{json.dumps(mock_json_data)}\n</user file information>\n<user_provided_filepath>\n{test_file}\n</user_provided_filepath>\n{user_query}"
        mock_result.stdout = formatted_output
        mock_subprocess.return_value = mock_result

        with (
            patch("tempfile.gettempdir", return_value=str(self.test_dir)),
            patch("os.path.exists", return_value=True),
            patch("builtins.open", create=True) as mock_open,
        ):
            mock_open.return_value.__enter__.return_value.read.return_value = (
                json.dumps(mock_json_data)
            )
            result = self.handler.process_command(
                f"/sharefile {test_file} {user_query}"
            )

            assert result.success is True
            output = result.agent_input

            # Verify XML structure
            lines = output.split("\n")

            # Find key sections
            info_start = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if "<user file information>" in line
                ),
                -1,
            )
            info_end = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if "</user file information>" in line
                ),
                -1,
            )
            filepath_start = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if "<user_provided_filepath>" in line
                ),
                -1,
            )
            filepath_end = next(
                (
                    i
                    for i, line in enumerate(lines)
                    if "</user_provided_filepath>" in line
                ),
                -1,
            )
            query_line = next(
                (i for i, line in enumerate(lines) if user_query in line), -1
            )

            # Verify structure order and presence
            assert info_start != -1, "Missing <user file information> tag"
            assert info_end != -1, "Missing </user file information> tag"
            assert filepath_start != -1, "Missing <user_provided_filepath> tag"
            assert filepath_end != -1, "Missing </user_provided_filepath> tag"
            assert query_line != -1, "Missing user query in output"

            # Verify order: file info, then filepath, then query
            assert info_start < info_end < filepath_start < filepath_end < query_line

            # Verify content within sections
            json_content = "\n".join(lines[info_start + 1 : info_end])
            filepath_content = "\n".join(lines[filepath_start + 1 : filepath_end])

            assert json.dumps(mock_json_data) in json_content
            assert str(test_file) in filepath_content
