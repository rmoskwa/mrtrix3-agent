"""Unit tests for message reconstruction when multiple tool calls are made."""

from unittest.mock import Mock
from pydantic_ai.result import StreamedRunResult
from pydantic_ai.messages import (
    ModelResponse,
    ModelRequest,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)


class TestMessageReconstruction:
    """Test message reconstruction from fragmented PydanticAI responses."""

    def test_single_message_single_part(self):
        """Test simple case: single message with single text part."""
        # Create mock result
        result = Mock(spec=StreamedRunResult)

        # Single message with single text part
        text_part = Mock(spec=TextPart)
        text_part.__class__.__name__ = "TextPart"
        text_part.content = "This is a simple response."

        message = Mock(spec=ModelResponse)
        message.parts = [text_part]

        result.all_messages.return_value = [message]
        result.output = "This is a simple response."

        # Simulate the reconstruction logic from cli.py
        all_text_parts = []
        assistant_message_parts = []

        for idx, msg in enumerate(result.all_messages()):
            msg_parts = []
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    part_type = (
                        part.__class__.__name__ if hasattr(part, "__class__") else ""
                    )
                    if part_type == "TextPart" and hasattr(part, "content"):
                        if part.content and len(part.content.strip()) > 10:
                            msg_parts.append(part.content)
                            all_text_parts.append(part.content)

            if msg_parts:
                assistant_message_parts.append((idx, msg_parts))

        # Should use result.output for simple single-part messages
        assert len(assistant_message_parts) == 1
        assert assistant_message_parts[0][1] == ["This is a simple response."]
        full_response = result.output
        assert full_response == "This is a simple response."

    def test_single_message_multiple_parts(self):
        """Test single message with multiple text parts."""
        # Create mock result
        result = Mock(spec=StreamedRunResult)

        # Single message with multiple text parts
        part1 = Mock(spec=TextPart)
        part1.__class__.__name__ = "TextPart"
        part1.content = "First part of the response."

        part2 = Mock(spec=TextPart)
        part2.__class__.__name__ = "TextPart"
        part2.content = "Second part of the response."

        part3 = Mock(spec=TextPart)
        part3.__class__.__name__ = "TextPart"
        part3.content = "Third part of the response."

        message = Mock(spec=ModelResponse)
        message.parts = [part1, part2, part3]

        result.all_messages.return_value = [message]
        result.output = "First part"  # Incomplete output

        # Simulate the reconstruction logic
        all_text_parts = []
        assistant_message_parts = []

        for idx, msg in enumerate(result.all_messages()):
            msg_parts = []
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    part_type = (
                        part.__class__.__name__ if hasattr(part, "__class__") else ""
                    )
                    if part_type == "TextPart" and hasattr(part, "content"):
                        if part.content and len(part.content.strip()) > 10:
                            msg_parts.append(part.content)
                            all_text_parts.append(part.content)

            if msg_parts:
                assistant_message_parts.append((idx, msg_parts))

        # Should reconstruct from multiple parts
        assert len(assistant_message_parts) == 1
        assert len(assistant_message_parts[0][1]) == 3

        # Reconstruct the response
        last_assistant_idx, last_assistant_parts = assistant_message_parts[-1]
        if len(last_assistant_parts) > 1:
            reconstructed = "\n".join(last_assistant_parts)
            full_response = reconstructed
        else:
            full_response = result.output

        expected = "First part of the response.\nSecond part of the response.\nThird part of the response."
        assert full_response == expected

    def test_multiple_messages_with_tool_calls(self):
        """Test multiple messages from multiple tool calls - the main fragmentation case."""
        # Create mock result
        result = Mock(spec=StreamedRunResult)

        # Message 1: User request
        user_msg = Mock(spec=ModelRequest)
        user_msg.parts = []

        # Message 2: First tool call and partial response
        tool_call_part = Mock(spec=ToolCallPart)
        tool_call_part.__class__.__name__ = "ToolCallPart"

        text_part1 = Mock(spec=TextPart)
        text_part1.__class__.__name__ = "TextPart"
        text_part1.content = "Based on my search, I found the following:"

        msg2 = Mock(spec=ModelResponse)
        msg2.parts = [tool_call_part, text_part1]

        # Message 3: Tool return
        tool_return = Mock(spec=ToolReturnPart)
        tool_return.__class__.__name__ = "ToolReturnPart"

        msg3 = Mock(spec=ModelRequest)
        msg3.parts = [tool_return]

        # Message 4: Second tool call with more response
        tool_call_part2 = Mock(spec=ToolCallPart)
        tool_call_part2.__class__.__name__ = "ToolCallPart"

        text_part2 = Mock(spec=TextPart)
        text_part2.__class__.__name__ = "TextPart"
        text_part2.content = "The mrconvert command can be used as follows:"

        text_part3 = Mock(spec=TextPart)
        text_part3.__class__.__name__ = "TextPart"
        text_part3.content = "```bash\nmrconvert input.dcm output.mif\n```"

        msg4 = Mock(spec=ModelResponse)
        msg4.parts = [tool_call_part2, text_part2, text_part3]

        # Message 5: Another tool return
        msg5 = Mock(spec=ModelRequest)
        msg5.parts = [tool_return]

        # Message 6: Final response parts
        text_part4 = Mock(spec=TextPart)
        text_part4.__class__.__name__ = "TextPart"
        text_part4.content = "Additionally, you should verify the gradient scheme:"

        text_part5 = Mock(spec=TextPart)
        text_part5.__class__.__name__ = "TextPart"
        text_part5.content = "```bash\nmrinfo output.mif -dwgrad\n```"

        msg6 = Mock(spec=ModelResponse)
        msg6.parts = [text_part4, text_part5]

        result.all_messages.return_value = [user_msg, msg2, msg3, msg4, msg5, msg6]
        result.output = "Based on my search"  # Incomplete - only first fragment

        # Simulate the reconstruction logic from cli.py
        all_text_parts = []
        assistant_message_parts = []

        for idx, msg in enumerate(result.all_messages()):
            msg_parts = []
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    part_type = (
                        part.__class__.__name__ if hasattr(part, "__class__") else ""
                    )
                    if part_type == "TextPart" and hasattr(part, "content"):
                        if part.content and len(part.content.strip()) > 10:
                            msg_parts.append(part.content)
                            all_text_parts.append(part.content)

            if msg_parts:
                assistant_message_parts.append((idx, msg_parts))

        # Should have found response parts across multiple messages
        assert len(assistant_message_parts) == 3  # Messages 2, 4, and 6

        # Check if we have response parts spread across multiple messages
        if len(assistant_message_parts) > 1:
            # Combine ALL assistant message parts in order
            all_parts_combined = []
            for msg_idx, parts in assistant_message_parts:
                all_parts_combined.extend(parts)

            # Join all parts with proper spacing
            complete_response = "\n".join(all_parts_combined)
            full_response = complete_response
        else:
            full_response = result.output

        # Verify the reconstructed response contains all parts
        assert "Based on my search, I found the following:" in full_response
        assert "The mrconvert command can be used as follows:" in full_response
        assert "mrconvert input.dcm output.mif" in full_response
        assert "Additionally, you should verify the gradient scheme:" in full_response
        assert "mrinfo output.mif -dwgrad" in full_response

    def test_filter_short_content(self):
        """Test that short content (<=10 chars) is filtered out."""
        # Create mock result
        result = Mock(spec=StreamedRunResult)

        # Mix of short and long content
        part1 = Mock(spec=TextPart)
        part1.__class__.__name__ = "TextPart"
        part1.content = "OK"  # Too short, should be filtered

        part2 = Mock(spec=TextPart)
        part2.__class__.__name__ = "TextPart"
        part2.content = "This is a longer response that should be included."

        part3 = Mock(spec=TextPart)
        part3.__class__.__name__ = "TextPart"
        part3.content = "Sure!"  # Too short

        part4 = Mock(spec=TextPart)
        part4.__class__.__name__ = "TextPart"
        part4.content = "Here is another substantial part of the response."

        message = Mock(spec=ModelResponse)
        message.parts = [part1, part2, part3, part4]

        result.all_messages.return_value = [message]
        result.output = "OK This is"  # Incomplete

        # Simulate the reconstruction logic
        all_text_parts = []
        assistant_message_parts = []

        for idx, msg in enumerate(result.all_messages()):
            msg_parts = []
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    part_type = (
                        part.__class__.__name__ if hasattr(part, "__class__") else ""
                    )
                    if part_type == "TextPart" and hasattr(part, "content"):
                        # Filter out short content
                        if part.content and len(part.content.strip()) > 10:
                            msg_parts.append(part.content)
                            all_text_parts.append(part.content)

            if msg_parts:
                assistant_message_parts.append((idx, msg_parts))

        # Should only include the longer parts
        assert len(assistant_message_parts) == 1
        assert len(assistant_message_parts[0][1]) == 2
        assert "OK" not in assistant_message_parts[0][1][0]
        assert "Sure!" not in "\n".join(assistant_message_parts[0][1])

    def test_non_text_parts_ignored(self):
        """Test that non-TextPart objects are properly ignored."""
        # Create mock result
        result = Mock(spec=StreamedRunResult)

        # Mix of different part types
        tool_part = Mock(spec=ToolCallPart)
        tool_part.__class__.__name__ = "ToolCallPart"

        text_part = Mock(spec=TextPart)
        text_part.__class__.__name__ = "TextPart"
        text_part.content = "This is the actual response text."

        return_part = Mock(spec=ToolReturnPart)
        return_part.__class__.__name__ = "ToolReturnPart"

        # Some unknown part type
        unknown_part = Mock()
        unknown_part.__class__.__name__ = "UnknownPart"
        unknown_part.content = "This should be ignored"

        message = Mock(spec=ModelResponse)
        message.parts = [tool_part, text_part, return_part, unknown_part]

        result.all_messages.return_value = [message]
        result.output = "This is the actual response text."

        # Simulate the reconstruction logic
        all_text_parts = []
        assistant_message_parts = []

        for idx, msg in enumerate(result.all_messages()):
            msg_parts = []
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    part_type = (
                        part.__class__.__name__ if hasattr(part, "__class__") else ""
                    )
                    if part_type == "TextPart" and hasattr(part, "content"):
                        if part.content and len(part.content.strip()) > 10:
                            msg_parts.append(part.content)
                            all_text_parts.append(part.content)

            if msg_parts:
                assistant_message_parts.append((idx, msg_parts))

        # Should only extract the TextPart content
        assert len(assistant_message_parts) == 1
        assert len(assistant_message_parts[0][1]) == 1
        assert assistant_message_parts[0][1][0] == "This is the actual response text."
        assert "This should be ignored" not in assistant_message_parts[0][1][0]

    def test_empty_messages_handled(self):
        """Test handling of messages with no parts or empty parts."""
        # Create mock result
        result = Mock(spec=StreamedRunResult)

        # Message with no parts attribute
        msg1 = Mock(spec=ModelRequest)
        if hasattr(msg1, "parts"):
            delattr(msg1, "parts")

        # Message with empty parts list
        msg2 = Mock(spec=ModelResponse)
        msg2.parts = []

        # Message with text part but empty content
        empty_part = Mock(spec=TextPart)
        empty_part.__class__.__name__ = "TextPart"
        empty_part.content = ""

        msg3 = Mock(spec=ModelResponse)
        msg3.parts = [empty_part]

        # Valid message
        valid_part = Mock(spec=TextPart)
        valid_part.__class__.__name__ = "TextPart"
        valid_part.content = "This is the only valid response."

        msg4 = Mock(spec=ModelResponse)
        msg4.parts = [valid_part]

        result.all_messages.return_value = [msg1, msg2, msg3, msg4]
        result.output = "Partial"

        # Simulate the reconstruction logic
        all_text_parts = []
        assistant_message_parts = []

        for idx, msg in enumerate(result.all_messages()):
            msg_parts = []
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    part_type = (
                        part.__class__.__name__ if hasattr(part, "__class__") else ""
                    )
                    if part_type == "TextPart" and hasattr(part, "content"):
                        if part.content and len(part.content.strip()) > 10:
                            msg_parts.append(part.content)
                            all_text_parts.append(part.content)

            if msg_parts:
                assistant_message_parts.append((idx, msg_parts))

        # Should only have one valid message part
        assert len(assistant_message_parts) == 1
        assert assistant_message_parts[0][1] == ["This is the only valid response."]
