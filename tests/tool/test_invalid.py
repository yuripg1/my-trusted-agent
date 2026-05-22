from tool.invalid import (
    InvalidArguments,
    InvalidToolCall,
    get_invalid_message,
    get_invalid_permission,
    invalid,
)


class TestGetInvalidMessage:
    """Tests for the `get_invalid_message` function"""

    def test_format(self) -> None:
        """Format the message correctly"""
        tool_name: str = "write_file"
        tool_call: InvalidToolCall = {
            "tool_name": "invalid",
            "arguments": {"tool_name": tool_name, "error_message": "irrelevant"},
        }
        assert get_invalid_message(tool_call) == f"Skipping invalid tool call **{tool_name}**"


class TestGetInvalidPermission:
    """Tests for the `get_invalid_permission` function"""

    def test_auto_approved(self) -> None:
        """Permission should be automatically granted since this is just an error report"""
        tool_call: InvalidToolCall = {
            "tool_name": "invalid",
            "arguments": {"tool_name": "irrelevant", "error_message": "irrelevant"},
        }
        assert get_invalid_permission(tool_call) is True


class TestExecuteInvalid:
    """Tests for the `execute_invalid` tool"""

    def test_format(self) -> None:
        """Return the error in a structured XML format"""
        tool_name: str = "write_file"
        error_message: str = "Missing required argument: 'path'"
        arguments: InvalidArguments = InvalidArguments(tool_name=tool_name, error_message=error_message)
        expected: str = f'<skipped_invalid_tool_call tool_name="{tool_name}">\n<error>{error_message}</error>\n</skipped_invalid_tool_call>'
        assert invalid(arguments) == expected

    def test_json_parse_failure(self) -> None:
        """Handle the JSON parsing error scenario"""
        tool_name: str = "read_file"
        error_message: str = "There was a problem parsing the arguments JSON"
        arguments: InvalidArguments = InvalidArguments(tool_name=tool_name, error_message=error_message)
        expected: str = f'<skipped_invalid_tool_call tool_name="{tool_name}">\n<error>{error_message}</error>\n</skipped_invalid_tool_call>'
        assert invalid(arguments) == expected

    def test_unknown_tool_name(self) -> None:
        """Handle the unknown tool name scenario"""
        tool_name: str = "send_email"
        error_message: str = "Invalid tool call"
        arguments: InvalidArguments = InvalidArguments(tool_name=tool_name, error_message=error_message)
        expected: str = f'<skipped_invalid_tool_call tool_name="{tool_name}">\n<error>{error_message}</error>\n</skipped_invalid_tool_call>'
        assert invalid(arguments) == expected
