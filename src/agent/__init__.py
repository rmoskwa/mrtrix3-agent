"""MRtrix3 Agent module"""

import logging
import os
import sys

from .agent import MRtrixAssistant
from .dependencies import setup_search_knowledgebase_dependencies
from .models import (
    AgentConfiguration,
    DocumentResult,
    SearchKnowledgebaseDependencies,
    SearchToolParameters,
)

# Configure agent-specific logger
logger = logging.getLogger("agent")
logger.setLevel(logging.DEBUG)

# Create handlers for different log levels
# Console handler for INFO and above
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(console_formatter)


# Add filter to prevent double logging
class UniqueFilter(logging.Filter):
    def __init__(self):
        self.logged = set()

    def filter(self, record):
        key = (record.name, record.levelno, record.msg)
        if key in self.logged:
            return False
        self.logged.add(key)
        return True


console_handler.addFilter(UniqueFilter())
logger.addHandler(console_handler)

# Optional file handler for DEBUG level (if DEBUG_LOG_FILE env var is set)
debug_log_file = os.getenv("DEBUG_LOG_FILE")
if debug_log_file:
    file_handler = logging.FileHandler(debug_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Debug logging enabled to file: {debug_log_file}")

# Log initialization
logger.info("MRtrix3 Agent module initialized")

__all__ = [
    "MRtrixAssistant",
    "SearchKnowledgebaseDependencies",
    "AgentConfiguration",
    "DocumentResult",
    "SearchToolParameters",
    "setup_search_knowledgebase_dependencies",
]
