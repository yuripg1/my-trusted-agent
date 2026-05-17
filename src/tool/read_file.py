from typing import Literal, NotRequired, Required, TypedDict

from tool.common import BaseToolCall


class ReadFileArguments(TypedDict):
    path: Required[str]
    start_line: NotRequired[int]
    end_line: NotRequired[int]


class ReadFileToolCall(BaseToolCall):
    tool_name: Required[Literal["read_file"]]
    arguments: Required[ReadFileArguments]


def get_read_file_message(tool_call: ReadFileToolCall) -> str:
    start_line = tool_call["arguments"].get("start_line")
    end_line = tool_call["arguments"].get("end_line")
    if start_line is not None and end_line is not None:
        return f"Reading file at **{tool_call['arguments']['path']}** (lines **{start_line}** to **{end_line}**)"
    elif start_line is not None:
        return f"Reading file at **{tool_call['arguments']['path']}** (from line **{start_line}**)"
    elif end_line is not None:
        return f"Reading file at **{tool_call['arguments']['path']}** (up to line **{end_line}**)"
    else:
        return f"Reading file at **{tool_call['arguments']['path']}**"


def get_read_file_permission(tool_call: ReadFileToolCall) -> bool:
    return False


def read_file(
    path: str, tool_call_permission: bool = True, *, start_line: int | None = None, end_line: int | None = None
) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>File reading manually denied by the user</error>")
    else:
        try:
            if start_line is not None and start_line < 1:
                output_entries.append('<error>"start_line" must be greater than or equal to 1</error>')
            elif end_line is not None and end_line < 1:
                output_entries.append('<error>"end_line" must be greater than or equal to 1</error>')
            elif start_line is not None and end_line is not None and start_line > end_line:
                output_entries.append('<error>"start_line" must be less than or equal to "end_line"</error>')
            else:
                with open(path) as file:
                    file_content = file.read()
                if start_line is not None or end_line is not None:
                    lines = file_content.splitlines()
                    start_idx = (start_line - 1) if start_line is not None else 0
                    end_idx = end_line if end_line is not None else len(lines)
                    file_content = "\n".join(lines[start_idx:end_idx])
                output_entries.append(f"<content>\n{file_content}\n</content>")
        except FileNotFoundError:
            output_entries.append("<error>File not found</error>")
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except Exception:
            output_entries.append("<error>Could not read file</error>")
    tag_attributes: str = f'path="{path}"'
    if start_line is not None:
        tag_attributes += f' start_line="{start_line}"'
    if end_line is not None:
        tag_attributes += f' end_line="{end_line}"'
    joined_output_entries: str = "\n".join(output_entries)
    return f"<file_read {tag_attributes}>\n{joined_output_entries}\n</file_read>"
