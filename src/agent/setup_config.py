#!/usr/bin/env python3
"""Setup configuration for MRtrix3 Agent."""

import sys
from pathlib import Path
from platformdirs import user_config_dir
import google.generativeai as genai


def validate_api_key(api_key: str) -> bool:
    """
    Validate the Google API key by attempting to use it.

    Args:
        api_key: The API key to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Configure the API key
        genai.configure(api_key=api_key)

        # Try to list available models (lightweight API call)
        models = genai.list_models()

        # Check if we can access Gemini models
        gemini_models = [m for m in models if "gemini" in m.name.lower()]
        if not gemini_models:
            print("\n⚠️  API key is valid but doesn't have access to Gemini models.")
            return False

        return True
    except Exception as e:
        # Parse common error types
        error_str = str(e).lower()
        if "invalid" in error_str or "api key" in error_str:
            print("\n❌ Invalid API key. Please check and try again.")
        elif "quota" in error_str:
            print("\n⚠️  API key is valid but you've exceeded your quota.")
            return True  # Key is valid, just over quota
        elif "network" in error_str or "connection" in error_str:
            print("\n❌ Network error. Please check your internet connection.")
        else:
            print(f"\n❌ Error validating API key: {e}")
        return False


def setup_config():
    """Interactive setup for MRtrix3 Agent configuration."""
    app_name = "mrtrix3-agent"

    print("\n🤖 MRtrix3 Agent Configuration Setup\n")
    print("This will configure your Google Gemini API key for the agent.")

    # Get platform-specific config directory
    config_dir = Path(user_config_dir(app_name))
    config_file = config_dir / "config"

    print(f"Configuration will be saved to: {config_file}\n")

    # Check if config already exists
    if config_file.exists():
        response = input("Configuration already exists. Overwrite? (y/N): ")
        if response.lower() != "y":
            print("Setup cancelled.")
            return

    # Collect required keys
    config = {}

    # Google Gemini API Key
    print("Google Gemini API Key Required")
    print("Get your free API key from: https://makersuite.google.com/app/apikey")
    print("")

    # Loop until we get a valid key or user cancels
    while True:
        google_key = input("Enter your GOOGLE_API_KEY: ").strip()

        if not google_key:
            print("\n❌ API key cannot be empty.")
            continue

        # Validate the API key
        print("\nValidating API key...")
        if validate_api_key(google_key):
            print("✅ API key validated successfully!")
            config["GOOGLE_API_KEY"] = google_key
            break
        else:
            retry = input("\nWould you like to try again? (y/N): ")
            if retry.lower() != "y":
                print("Setup cancelled.")
                sys.exit(1)

    # Create config directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    # Write configuration
    with open(config_file, "w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

    # Set permissions to be readable only by user (Unix-like systems)
    try:
        config_file.chmod(0o600)
    except (AttributeError, OSError):
        # Windows doesn't support chmod in the same way
        pass

    print(f"\n✅ Configuration saved to {config_file}")
    print("🎉 You can now run 'mrtrixBot' to start the agent!\n")


if __name__ == "__main__":
    try:
        setup_config()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)
