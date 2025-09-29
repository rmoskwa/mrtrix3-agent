"""Markdown code block fixer for cleaning up AI-generated markdown."""

import re


class MarkdownCodeBlockFixer:
    """Fixes malformed code blocks in markdown content, especially those indented in lists."""

    def fix_markdown(self, content: str) -> str:
        """Fix malformed code blocks in markdown content.

        The main issues we fix:
        1. Code blocks with indented ``` markers (move to column 0)
        2. Unclosed code blocks (add closing ```)
        3. List items with 4-space indentation that get treated as code blocks

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

            if "```" in line and stripped.startswith("```"):
                # This line has a code fence marker
                if not in_code_block:
                    # Opening a code block - put fence at column 0
                    fixed_lines.append(stripped)
                    in_code_block = True
                else:
                    # Closing a code block - put fence at column 0
                    fixed_lines.append("```")
                    in_code_block = False
            else:
                # Regular content
                if in_code_block:
                    # Inside a fenced code block - remove excess indentation
                    # (code inside fenced blocks shouldn't be indented from list context)
                    if line.startswith("        "):
                        # 8+ spaces - likely list-indented code
                        fixed_lines.append(line[8:] if len(line) > 8 else "")
                    elif line.startswith("    "):
                        # 4 spaces - remove to clean up code
                        fixed_lines.append(line[4:] if len(line) > 4 else "")
                    else:
                        fixed_lines.append(line)
                else:
                    # Not in a fenced code block
                    # Check if this line starts with 4+ spaces (would be treated as code block)
                    if line.startswith("    "):
                        # Count leading spaces
                        space_count = len(line) - len(line.lstrip(" "))
                        stripped_content = line.strip()

                        # If it contains text that shouldn't be in a code block
                        # (has markdown formatting, punctuation suggesting prose, etc.)
                        if stripped_content and (
                            "**" in stripped_content  # Bold markdown
                            or "`" in stripped_content  # Inline code
                            or stripped_content.startswith("*")  # List item
                            or stripped_content.startswith("-")  # List item
                            or stripped_content.startswith("+")  # List item
                            or stripped_content.startswith("•")  # Bullet point
                            or ". " in stripped_content  # Sentence
                            or ", " in stripped_content
                        ):  # Prose with commas
                            # Reduce to 3 spaces to break code block interpretation
                            # but maintain some indentation for readability
                            if space_count >= 4:
                                fixed_lines.append("   " + line.lstrip())
                            else:
                                fixed_lines.append(line)
                        else:
                            # Keep as-is - might be intentional code
                            fixed_lines.append(line)
                    else:
                        # Pass through unchanged
                        fixed_lines.append(line)

        # If we ended while still in a code block, close it
        if in_code_block:
            fixed_lines.append("```")

        result = "\n".join(fixed_lines)

        # Clean up any empty code blocks (```\n``` with nothing between)
        result = re.sub(r"```[a-zA-Z]*\n```", "", result)

        return result
