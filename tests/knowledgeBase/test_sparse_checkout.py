"""
Unit tests for git sparse checkout functionality in MRtrix3 Agent.

This module tests the RSTDocumentGatherer class's sparse checkout implementation,
focusing on verifying that all commands from commands_list.rst have corresponding
.rst files in the commands/ directory after sparse checkout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from unittest.mock import patch, MagicMock, call
import subprocess
from typing import List
from knowledge_base.populate_database import RSTDocumentGatherer


class TestSparseCheckout:
    """Test the git sparse checkout functionality."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.gatherer = RSTDocumentGatherer()

    def test_sparse_checkout_setup_calls(
        self, mock_git_subprocess, temp_directory, mock_env
    ):
        """Test that sparse checkout makes the correct git subprocess calls."""

        # Setup mock to succeed
        mock_git_subprocess.return_value.returncode = 0

        # Create fake directories that would exist after checkout
        repo_path = temp_directory / "mrtrix3"
        (repo_path / "docs" / "commands").mkdir(parents=True)
        (repo_path / "cmd").mkdir(parents=True)
        (repo_path / "bin").mkdir(parents=True)

        # Create some dummy files to simulate successful sparse checkout
        (repo_path / "cmd" / "dwiextract.cpp").touch()
        (repo_path / "bin" / "amp2sh").touch()

        # Mock urllib request for version detection
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"tag_name": "v3.0.7"}'
            mock_urlopen.return_value.__enter__.return_value = mock_response

            # Execute the method under test
            documents, version, repo_path_result = (
                self.gatherer.gather_documents_and_source(str(temp_directory))
            )

        # Verify git commands were called in correct order
        expected_calls = [
            call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--sparse",
                    "--depth",
                    "1",
                    "--branch",
                    "master",
                    self.gatherer.repo_url,
                    str(repo_path),
                ],
                check=True,
                capture_output=True,
            ),
            call(
                ["git", "sparse-checkout", "init", "--cone"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            ),
            call(
                ["git", "sparse-checkout", "set", "docs", "cmd", "bin"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            ),
        ]

        mock_git_subprocess.assert_has_calls(expected_calls)
        assert version == "3.0.7"
        assert repo_path_result == repo_path

    def test_commands_list_parsing(
        self, temp_directory, sample_commands_list_rst, mock_env
    ):
        """Test parsing command names from commands_list.rst content."""

        def parse_commands_from_list(commands_list_content: str) -> List[str]:
            """
            Parse command names from commands_list.rst toctree entries.

            Extracts command names from entries like "commands/dirgen" -> "dirgen"
            """
            command_names = []
            lines = commands_list_content.strip().split("\n")

            for line in lines:
                line = line.strip()
                # Look for toctree entries that start with "commands/"
                if line.startswith("commands/"):
                    command_name = line.split("/")[
                        -1
                    ]  # Extract everything after last '/'
                    command_names.append(command_name)

            return sorted(command_names)

        # Test the parsing function
        parsed_commands = parse_commands_from_list(sample_commands_list_rst)

        expected_commands = [
            "5ttgen",
            "amp2sh",
            "connectome2tck",
            "dirgen",
            "dwi2fod",
            "dwiextract",
            "mrcalc",
            "mrconvert",
            "mrstats",
            "tckgen",
            "tcksample",
            "transformcalc",
        ]

        assert parsed_commands == expected_commands

    def test_command_rst_files_verification(
        self, temp_directory, sample_commands_list_rst, sample_command_files, mock_env
    ):
        """Test verification that command .rst files exist for all commands in commands_list.rst."""

        def parse_commands_from_list(commands_list_content: str) -> List[str]:
            """Parse command names from commands_list.rst."""
            command_names = []
            lines = commands_list_content.strip().split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("commands/"):
                    command_name = line.split("/")[-1]
                    command_names.append(command_name)

            return sorted(command_names)

        def get_rst_files_from_commands_dir(commands_dir: Path) -> List[str]:
            """Get list of .rst files (without extension) from commands directory."""
            if not commands_dir.exists():
                return []

            rst_files = []
            for rst_file in commands_dir.glob("*.rst"):
                rst_files.append(rst_file.stem)  # Get filename without .rst extension

            return sorted(rst_files)

        # Create mock commands directory with RST files
        commands_dir = temp_directory / "commands"
        commands_dir.mkdir(parents=True)

        # Create the RST files
        for rst_file in sample_command_files:
            (commands_dir / rst_file).touch()

        # Parse commands from the sample commands_list.rst
        parsed_commands = parse_commands_from_list(sample_commands_list_rst)

        # Get actual RST files from commands directory
        actual_rst_files = get_rst_files_from_commands_dir(commands_dir)

        # Verify that all commands from commands_list.rst have corresponding .rst files
        assert set(
            parsed_commands
        ).issubset(
            set(actual_rst_files)
        ), f"Missing RST files for commands: {set(parsed_commands) - set(actual_rst_files)}"

        # Additional verification: ensure we have exactly the expected commands
        assert parsed_commands == actual_rst_files

    @patch("subprocess.run")
    def test_full_sparse_checkout_workflow(
        self,
        mock_subprocess,
        temp_directory,
        sample_commands_list_rst,
        sample_command_files,
        mock_env,
    ):
        """
        Test the complete sparse checkout workflow including:
        1. Git sparse checkout execution
        2. Reading commands_list.rst
        3. Parsing command names
        4. Verifying corresponding .rst files exist
        """

        # Setup successful git operations
        mock_subprocess.return_value.returncode = 0

        # Create the directory structure that would exist after sparse checkout
        repo_path = temp_directory / "mrtrix3"
        docs_dir = repo_path / "docs"
        commands_dir = docs_dir / "commands"
        commands_dir.mkdir(parents=True)

        # Create cmd and bin directories
        (repo_path / "cmd").mkdir(parents=True)
        (repo_path / "bin").mkdir(parents=True)

        # Add some files to simulate successful sparse checkout
        (repo_path / "cmd" / "dwiextract.cpp").touch()
        (repo_path / "bin" / "amp2sh").touch()

        # Create commands_list.rst file
        commands_list_file = docs_dir / "commands_list.rst"
        commands_list_file.write_text(sample_commands_list_rst)

        # Create all command RST files
        for rst_file in sample_command_files:
            (commands_dir / rst_file).touch()

        # Mock urllib request for version detection
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"tag_name": "v3.0.7"}'
            mock_urlopen.return_value.__enter__.return_value = mock_response

            # Execute sparse checkout
            documents, version, repo_path_result = (
                self.gatherer.gather_documents_and_source(str(temp_directory))
            )

        # Verify the sparse checkout was successful
        assert repo_path_result == repo_path
        assert version == "3.0.7"

        # Now perform the verification workflow
        def parse_commands_from_list(commands_list_content: str) -> List[str]:
            """Parse command names from commands_list.rst."""
            command_names = []
            lines = commands_list_content.strip().split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("commands/"):
                    command_name = line.split("/")[-1]
                    command_names.append(command_name)

            return sorted(command_names)

        def get_rst_files_from_commands_dir(commands_dir: Path) -> List[str]:
            """Get list of .rst files (without extension) from commands directory."""
            rst_files = []
            for rst_file in commands_dir.glob("*.rst"):
                rst_files.append(rst_file.stem)

            return sorted(rst_files)

        # 1. Read commands_list.rst
        commands_list_content = commands_list_file.read_text()

        # 2. Parse command names
        parsed_commands = parse_commands_from_list(commands_list_content)

        # 3. List actual .rst files in commands/ directory
        actual_rst_files = get_rst_files_from_commands_dir(commands_dir)

        # 4. Verify that all commands have corresponding files
        missing_commands = set(parsed_commands) - set(actual_rst_files)
        assert (
            not missing_commands
        ), f"Missing RST files for commands: {missing_commands}"

        # 5. Verify the sorted array is contained within the actual files
        assert set(parsed_commands).issubset(
            set(actual_rst_files)
        ), "Not all commands from commands_list.rst have corresponding .rst files"

        # Additional assertions for completeness
        assert len(parsed_commands) == 12, "Expected 12 commands from sample"
        assert (
            "dwiextract" in parsed_commands
        ), "dwiextract should be in parsed commands"
        assert "dirgen" in parsed_commands, "dirgen should be in parsed commands"

    def test_commands_list_parsing_edge_cases(self, mock_env):
        """Test parsing of commands_list.rst with various edge cases."""

        def parse_commands_from_list(commands_list_content: str) -> List[str]:
            """Parse command names from commands_list.rst."""
            command_names = []
            lines = commands_list_content.strip().split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith("commands/"):
                    command_name = line.split("/")[-1]
                    command_names.append(command_name)

            return sorted(command_names)

        # Test with empty content
        assert parse_commands_from_list("") == []

        # Test with content but no commands
        no_commands_content = """
        Commands list
        =============

        This is just text without any toctree entries.
        """
        assert parse_commands_from_list(no_commands_content) == []

        # Test with mixed content
        mixed_content = """
        Commands list
        =============

        .. toctree::
           :maxdepth: 1

           commands/testcmd1
           not_a_command/something
           commands/testcmd2
           guides/someguide
           commands/testcmd3
        """
        expected = ["testcmd1", "testcmd2", "testcmd3"]
        assert parse_commands_from_list(mixed_content) == expected

    def test_sparse_checkout_error_handling(
        self, mock_git_subprocess, temp_directory, mock_env
    ):
        """Test handling of git subprocess errors during sparse checkout."""

        # Setup git to fail
        mock_git_subprocess.side_effect = subprocess.CalledProcessError(1, "git")

        # Should raise the CalledProcessError
        with pytest.raises(subprocess.CalledProcessError):
            self.gatherer.gather_documents_and_source(str(temp_directory))

    def test_version_fallback_on_api_failure(
        self, mock_git_subprocess, temp_directory, mock_env
    ):
        """Test that version falls back to 'latest' when GitHub API fails."""

        # Setup successful git operations
        mock_git_subprocess.return_value.returncode = 0

        # Create directories
        repo_path = temp_directory / "mrtrix3"
        (repo_path / "docs").mkdir(parents=True)
        (repo_path / "cmd").mkdir(parents=True)
        (repo_path / "bin").mkdir(parents=True)

        # Mock urllib request to fail
        with patch("urllib.request.urlopen", side_effect=Exception("API failure")):
            documents, version, repo_path_result = (
                self.gatherer.gather_documents_and_source(str(temp_directory))
            )

        assert version == "latest"

    def test_directory_structure_validation(
        self, mock_git_subprocess, temp_directory, mock_env
    ):
        """Test that the sparse checkout creates expected directory structure."""

        # Setup successful git operations
        mock_git_subprocess.return_value.returncode = 0

        # Create expected directory structure
        repo_path = temp_directory / "mrtrix3"
        docs_dir = repo_path / "docs"
        cmd_dir = repo_path / "cmd"
        bin_dir = repo_path / "bin"

        docs_dir.mkdir(parents=True)
        cmd_dir.mkdir(parents=True)
        bin_dir.mkdir(parents=True)

        # Add files to verify sparse checkout worked
        (cmd_dir / "test.cpp").touch()
        (bin_dir / "test_script").touch()

        # Mock version API
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"tag_name": "v3.0.7"}'
            mock_urlopen.return_value.__enter__.return_value = mock_response

            documents, version, repo_path_result = (
                self.gatherer.gather_documents_and_source(str(temp_directory))
            )

        # Verify directories exist
        assert docs_dir.exists(), "docs directory should exist after sparse checkout"
        assert cmd_dir.exists(), "cmd directory should exist after sparse checkout"
        assert bin_dir.exists(), "bin directory should exist after sparse checkout"
        assert (cmd_dir / "test.cpp").exists(), "C++ files should be present"
        assert (bin_dir / "test_script").exists(), "Python scripts should be present"
