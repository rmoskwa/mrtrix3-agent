"""Runtime validation of agent setup."""

import os
from pathlib import Path
from platformdirs import user_config_dir
import google.generativeai as genai


def check_setup() -> tuple[bool, str]:
    """
    Check if the agent is properly configured.

    Returns:
        Tuple of (success: bool, message: str)
    """
    app_name = "mrtrix3-agent"
    config_dir = Path(user_config_dir(app_name))
    config_file = config_dir / "config"

    # Check for development .env file first
    possible_env_paths = [
        Path.cwd() / ".env",
        Path(__file__).parent.parent.parent / ".env",  # Project root
    ]

    env_file_exists = any(p.exists() for p in possible_env_paths)

    # Check if either config file or .env exists
    if not config_file.exists() and not env_file_exists:
        return False, (
            "No configuration found.\n"
            "Please run 'setup-mrtrixbot' to configure your API key."
        )

    # Check if API key is set
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Try loading from config file or .env
        from dotenv import load_dotenv

        if config_file.exists():
            load_dotenv(config_file)

        # Also try .env files for development
        for env_path in possible_env_paths:
            if env_path.exists():
                load_dotenv(env_path, override=False)
                break

        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        return False, (
            "Google API key not found.\n"
            "Please run 'setup-mrtrixbot' to configure your API key."
        )

    # Quick validation of the API key
    try:
        genai.configure(api_key=api_key)
        # Just configure, don't make API call during startup
        return True, "Configuration validated successfully."
    except Exception as e:
        error_str = str(e).lower()
        if "invalid" in error_str or "api key" in error_str:
            return False, (
                "Invalid Google API key.\n"
                "Please run 'mrtrixBot-setup' to reconfigure."
            )
        else:
            # Other errors might be transient, allow startup
            return True, f"Warning: {e}"
