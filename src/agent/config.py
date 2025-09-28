"""Configuration constants for MRtrix3 Agent."""

# Read-only Supabase credentials for accessing public knowledge base
# These are safe to include in the package as they only provide read access
SUPABASE_URL = "https://pklxtyzyxcwfjyuqodce.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBrbHh0eXp5eGN3Zmp5dXFvZGNlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgzNDg2MzcsImV4cCI6MjA3MzkyNDYzN30.4inBjjmSw2RppBXoTk1w905XOTdQyO2oXkWNIMb9CuU"

# Fixed embedding configuration
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768

# App name for platformdirs
APP_NAME = "mrtrix3-agent"
