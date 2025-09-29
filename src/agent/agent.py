"""MRtrix3 Assistant agent implementation using PydanticAI framework."""

import logging
from pydantic_ai import Agent

from .models import SearchKnowledgebaseDependencies
from .tools import search_knowledgebase

logger = logging.getLogger(__name__)


class MRtrixAssistant:
    """Main PydanticAI agent for MRtrix3 documentation assistance."""

    def __init__(self, dependencies: SearchKnowledgebaseDependencies):
        """
        Initialize MRtrix3 Assistant agent.

        Args:
            dependencies: Dependency injection container with ChromaDB client for local queries,
                         Supabase client for sync operations, embedding model, and rate limiter.
        """
        self.dependencies = dependencies

        self.system_prompt = """You are an MRtrix3 documentation assistant specialized in helping users
with MRtrix3 neuroimaging software. You have access to comprehensive MRtrix3 documentation
including commands, tutorials, guides, and reference materials.

You can help with:
- MRtrix3 commands, tools, and workflows
- General MRI concepts and principles
- Diffusion MRI theory and analysis
- Neuroimaging techniques and methodologies
- Brain imaging and connectomics
- Medical imaging concepts relevant to MRtrix3

If a user asks a question completely unrelated to these topics (like general knowledge, programming
unrelated to neuroimaging, or other non-medical topics), politely inform them that you specialize
in MRtrix3 and neuroimaging, and suggest they ask about those topics instead.

When users ask questions, first analyze whether the question is complex or multi-part.
If it is, break it down into smaller, well-defined sub-questions.
Answer each sub-question individually, then combine the answers into a clear,
cohesive final response for the user.

When needed, use the MRtrix3 knowledge base to search for accurate and relevant information.
Always be helpful, precise, and provide clear explanations.
When discussing MRtrix3 commands, include usage examples and relevant parameters when appropriate.
"""

        self.agent = Agent(
            "google-gla:gemini-2.5-flash",
            system_prompt=self.system_prompt,
            deps_type=SearchKnowledgebaseDependencies,
            model_settings={
                "max_tokens": 65536,  # Gemini 2.5 Flash actual output limit
                "temperature": 0.7,
            },
        )

        self._register_tools()

    def _register_tools(self):
        """Register tools with the agent."""
        self.agent.tool(search_knowledgebase)

    async def run(self, query: str, message_history):
        """
        Run the agent with a user query.

        Args:
            query: Natural language query from the user.
            message_history: Conversation history for context (required, can be empty list for new conversations).

        Returns:
            Agent response with relevant MRtrix3 documentation.
        """
        logger.debug(f"Agent.run called with query: {query[:100]}...")

        try:
            result = await self.agent.run(
                query, deps=self.dependencies, message_history=message_history
            )

            # Log result info for debugging if needed
            if result.output:
                logger.debug(
                    f"Raw agent result.output length: {len(result.output)} chars"
                )

            return result

        except Exception as e:
            # Check for Gemini 500 errors which might indicate off-topic questions
            error_str = str(e)
            if "500" in error_str and "INTERNAL" in error_str:
                logger.warning(
                    f"Gemini API 500 error, likely from off-topic query: {query[:100]}"
                )
                # Re-raise with context about the error
                # The CLI layer will handle this with a user-friendly message
                raise RuntimeError(
                    "Gemini API returned 500 error - likely due to off-topic question conflicting with specialized prompt"
                ) from e
            else:
                # Re-raise other exceptions unchanged
                raise
