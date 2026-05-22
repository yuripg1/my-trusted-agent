from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class InvalidArguments(TypedDict):
    tool_name: Required[str]
    error_message: Required[str]


class InvalidToolCall(BaseToolCall):
    tool_name: Required[Literal["invalid"]]
    arguments: Required[InvalidArguments]


def get_invalid_message(tool_call: InvalidToolCall) -> str:
    return f"Skipping invalid tool call **{tool_call['arguments']['tool_name']}**"


def get_invalid_permission(tool_call: InvalidToolCall) -> bool:
    return True


def invalid(arguments: InvalidArguments) -> str:
    return f'<skipped_invalid_tool_call tool_name="{arguments["tool_name"]}">\n<error>{arguments["error_message"]}</error>\n</skipped_invalid_tool_call>'
