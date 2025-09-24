"""MRtrix3 Agent module"""

import logging
import os

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

# Only set up logging if COLLECT_LOGS is enabled
collect_logs = os.getenv("COLLECT_LOGS", "false").lower() == "true"

if collect_logs:
    logger.setLevel(logging.DEBUG)

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
else:
    # Suppress all logging when COLLECT_LOGS is false
    logger.setLevel(logging.CRITICAL)
    logger.addHandler(logging.NullHandler())

__all__ = [
    "MRtrixAssistant",
    "SearchKnowledgebaseDependencies",
    "AgentConfiguration",
    "DocumentResult",
    "SearchToolParameters",
    "setup_search_knowledgebase_dependencies",
]
