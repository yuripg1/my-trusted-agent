from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class ReadFileArguments(TypedDict):
    path: Required[str]


class ReadFileToolCall(BaseToolCall):
    tool_name: Required[Literal["read_file"]]
    arguments: Required[ReadFileArguments]


def get_read_file_message(tool_call: ReadFileToolCall) -> str:
    return f"Reading file at **{tool_call['arguments']['path']}**"


def get_read_file_permission(tool_call: ReadFileToolCall) -> bool:
    return False


def read_file(path: str, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>File reading manually denied by the user</error>")
    else:
        try:
            with open(path) as file:
                file_content = file.read()
                output_entries.append(f"<content>\n{file_content}\n</content>")
        except FileNotFoundError:
            output_entries.append("<error>File not found</error>")
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except:
            output_entries.append("<error>Could not read file</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<file_read path="{path}">\n{joined_output_entries}\n</file_read>'
