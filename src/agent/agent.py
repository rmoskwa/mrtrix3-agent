"""MRtrix3 Assistant agent implementation using PydanticAI framework."""

from pydantic_ai import Agent

from .models import SearchKnowledgebaseDependencies
from .tools import search_knowledgebase


class MRtrixAssistant:
    """Main PydanticAI agent for MRtrix3 documentation assistance."""

    def __init__(self, dependencies: SearchKnowledgebaseDependencies):
        """
        Initialize MRtrix3 Assistant agent.

        Args:
            dependencies: Dependency injection container with Supabase client,
                         embedding model, and rate limiter.
        """
        self.dependencies = dependencies

        self.system_prompt = """You are an MRtrix3 documentation assistant specialized in helping users
        with MRtrix3 neuroimaging software. You have access to comprehensive MRtrix3 documentation
        including commands, tutorials, guides, and reference materials.

        When users ask questions, you can search through the MRtrix3 knowledge base to provide
        accurate, relevant information.

        Be helpful, precise, and provide clear explanations. When discussing MRtrix3 commands,
        include usage examples and relevant parameters when appropriate."""

        self.agent = Agent(
            "google-gla:gemini-2.5-flash",
            system_prompt=self.system_prompt,
            deps_type=SearchKnowledgebaseDependencies,
        )

        self._register_tools()

    def _register_tools(self):
        """Register tools with the agent."""
        self.agent.tool(search_knowledgebase)

    async def run(self, query: str):
        """
        Run the agent with a user query.

        Args:
            query: Natural language query from the user.

        Returns:
            Agent response with relevant MRtrix3 documentation.
        """
        result = await self.agent.run(query, deps=self.dependencies)
        return result
