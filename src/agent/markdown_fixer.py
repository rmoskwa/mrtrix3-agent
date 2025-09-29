"""Markdown code block fixer for cleaning up AI-generated markdown."""

import re


class MarkdownCodeBlockFixer:
    """Fixes malformed code blocks in markdown content, especially those indented in lists."""

    def __init__(self):
        self.default_language = "bash"  # Default for MRtrix3 commands

    def fix_markdown(self, content: str) -> str:
        """Fix malformed code blocks in markdown content.

        The main issues we fix:
        1. Code blocks indented as part of list items (move ``` to column 0)
        2. Unclosed code blocks (missing closing ```)
        3. Missing language specifier on opening ```
        4. Code content that's indented within code blocks

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
        code_block_lang = None

        for line in lines:
            # Check if this line contains a code fence (at any indentation level)
            stripped = line.strip()

            if "```" in line:
                # This line has a code fence - extract it
                if stripped.startswith("```"):
                    # Extract language if present
                    lang = stripped[3:].strip()

                    if not in_code_block:
                        # Opening a code block
                        in_code_block = True
                        code_block_lang = lang if lang else self.default_language
                        # Code fences should NOT be indented - put at column 0
                        fixed_lines.append(f"```{code_block_lang}")
                    else:
                        # Closing a code block
                        # Check if this might be trying to open a new block
                        if lang and lang != code_block_lang:
                            # Close current and open new
                            fixed_lines.append("```")
                            fixed_lines.append(f"```{lang}")
                            code_block_lang = lang
                        else:
                            # Normal closing - put at column 0
                            fixed_lines.append("```")
                            in_code_block = False
                            code_block_lang = None
                else:
                    # The ``` is not at the start of the stripped line
                    # This might be inline or malformed - just pass through
                    fixed_lines.append(line)
            else:
                # Regular content
                if in_code_block:
                    # Inside a code block - remove list indentation but preserve code structure
                    if line.startswith("        "):
                        # 8 spaces = list item indentation for code
                        fixed_lines.append(line[8:] if len(line) > 8 else "")
                    elif line.startswith("    "):
                        # 4 spaces = could be list indentation or code indentation
                        # Remove it for cleaner code blocks
                        fixed_lines.append(line[4:] if len(line) > 4 else "")
                    else:
                        # No indentation or other - pass through
                        fixed_lines.append(line)
                else:
                    # Not in code block - pass through as is
                    fixed_lines.append(line)

        # If we ended while still in a code block, close it
        if in_code_block:
            fixed_lines.append("```")

        result = "\n".join(fixed_lines)

        # Clean up any empty code blocks
        result = re.sub(r"```(\w+)\n```", "", result)

        return result
