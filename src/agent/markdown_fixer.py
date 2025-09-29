"""Markdown code block fixer for cleaning up AI-generated markdown."""


class MarkdownCodeBlockFixer:
    """Fixes malformed code blocks in markdown content."""

    def fix_markdown(self, content: str) -> str:
        """Fix malformed code blocks in markdown content.

        The main issues we fix:
        - Code blocks with indented ``` markers (move to column 0)
        - Lines with 4+ spaces being interpreted as code blocks (remove ALL leading spaces)

        Args:
            content: Raw markdown content that may have malformed code blocks

        Returns:
            Fixed markdown content with properly formatted code blocks
        """
        if not content:
            return content

        lines = content.split("\n")
        fixed_lines = []
        in_code_block = False

        for line in lines:
            # Check if this line contains a code fence (at any indentation level)
            stripped = line.strip()

            # Track if we're inside a code block
            if "```" in line and stripped.startswith("```"):
                # This line has a code fence marker - put it at column 0
                fixed_lines.append(stripped)
                in_code_block = not in_code_block
            elif in_code_block:
                # Inside a code block - preserve as-is
                fixed_lines.append(line)
            else:
                # Outside code blocks - remove ALL leading spaces
                # This prevents markdown from treating any indented line as code
                fixed_lines.append(line.lstrip())

        return "\n".join(fixed_lines)
