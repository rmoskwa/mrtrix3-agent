#!/usr/bin/env python3
"""
Command generator workflow for MRtrix3.
Analyzes image files and helps users build appropriate MRtrix3 commands.
"""

import os
import random
import subprocess
import sys
import tempfile
import shutil


def run_mrinfo(file_path: str) -> tuple[bool, str, str]:
    """
    Run mrinfo on the given file/directory and save output to temporary JSON.

    Args:
        file_path: Path to file or directory to analyze

    Returns:
        Tuple of (success, json_path, error_message)
    """
    # Check if mrinfo is available
    if not shutil.which("mrinfo"):
        return (
            False,
            "",
            "Error: 'mrinfo' is not found on the system. Please ensure MRtrix3 is installed.",
        )

    # Check if path exists
    if not os.path.exists(file_path):
        return False, "", f"Error: Path '{file_path}' does not exist."

    # Generate random filename for JSON output
    random_id = random.randint(100000, 999999)
    json_filename = f"{random_id}.json"
    json_path = os.path.join(tempfile.gettempdir(), json_filename)

    try:
        # Run mrinfo with -json_all flag
        result = subprocess.run(
            ["mrinfo", "-json_all", json_path, file_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return False, "", f"Error: mrinfo failed - {error_msg}"

        # Verify JSON was created
        if not os.path.exists(json_path):
            return False, "", "Error: mrinfo did not create expected JSON output file."

        return True, json_path, ""

    except subprocess.TimeoutExpired:
        return False, "", "Error: mrinfo command timed out after 30 seconds."
    except Exception as e:
        return False, "", f"Error running mrinfo: {str(e)}"


def load_file_contents(file_path: str) -> tuple[bool, str]:
    """
    Load contents of a file.

    Args:
        file_path: Path to file to load

    Returns:
        Tuple of (success, content/error_message)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return True, f.read()
    except Exception as e:
        return False, f"Error reading file {file_path}: {str(e)}"


def build_prompt(json_content: str, user_filepath: str, user_query: str) -> str:
    """
    Build the prompt with file metadata and user query.

    Args:
        json_content: Contents of the mrinfo JSON output
        user_filepath: The filepath provided by the user
        user_query: The user's question about the file

    Returns:
        Formatted prompt with file info and query
    """
    # Build a simple prompt with just the file info and user query
    prompt = f"""<user file information>
{json_content}
</user file information>

<user_provided_filepath>
{user_filepath}
</user_provided_filepath>

{user_query}"""

    return prompt


def cleanup_json(json_path: str):
    """Remove temporary JSON file."""
    try:
        if os.path.exists(json_path):
            os.remove(json_path)
    except Exception:
        pass  # Silently ignore cleanup errors


def main():
    """Main entry point for sharefile workflow."""
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: /sharefile <path to file or directory> <query>")
        return 1

    file_path = sys.argv[1]
    user_query = sys.argv[2]

    # Run mrinfo to get file information
    success, json_path, error_msg = run_mrinfo(file_path)
    if not success:
        print(error_msg)
        return 1

    try:
        # Load the JSON content
        success, json_content = load_file_contents(json_path)
        if not success:
            print(json_content)  # This contains the error message
            return 1

        # Build the prompt with file info and user query
        prompt = build_prompt(json_content, file_path, user_query)

        # Output the prompt for the LLM
        # The CLI will capture this and send it to the agent
        print(prompt)
        return 0

    finally:
        # Always cleanup the temporary JSON file
        cleanup_json(json_path)


if __name__ == "__main__":
    sys.exit(main())
