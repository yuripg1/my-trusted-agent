from difflib import unified_diff
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall, make_safe_code_fence


class EditFileArguments(TypedDict):
    path: Required[str]
    search_for: Required[str]
    replace_with: Required[str]
    number_of_substitutions: Required[int]


class EditFileToolCall(BaseToolCall):
    tool_name: Required[Literal["edit_file"]]
    arguments: Required[EditFileArguments]


def get_edit_file_permission(tool_call: EditFileToolCall) -> bool:
    return False


def get_edit_file_message(tool_call: EditFileToolCall) -> str:
    search_for_text: str = tool_call["arguments"]["search_for"]
    replace_with_text: str = tool_call["arguments"]["replace_with"]
    file_path: str = tool_call["arguments"]["path"]
    diff_lines: list[str] = list(
        unified_diff(
            search_for_text.splitlines(),
            replace_with_text.splitlines(),
            fromfile=file_path,
            tofile=file_path,
            lineterm="",
        )
    )
    edit_content: str = make_safe_code_fence("\n".join(diff_lines), "diff")
    return f"Editing file at **{tool_call['arguments']['path']}** (**{tool_call['arguments']['number_of_substitutions']}** substitutions)\n\n{edit_content}"


def edit_file(
    path: str, search_for: str, replace_with: str, number_of_substitutions: int, tool_call_permission: bool = True
) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>File editing manually denied by the user</error>")
    else:
        number_of_occurrences: int | None = None
        try:
            with open(path) as file:
                file_content: str = file.read()
            number_of_occurrences = file_content.count(search_for)
            if number_of_occurrences == 0:
                output_entries.append("<error>No occurrences of the searched text were found</error>")
            elif number_of_occurrences != number_of_substitutions:
                output_entries.append(
                    "<error>The number of occurrences of the searched text does not match the expected number of substitutions</error>"
                )
            else:
                new_content: str = file_content.replace(search_for, replace_with, number_of_substitutions)
                with open(path, "w") as file:
                    file.write(new_content)
                output_entries.append("<result>File edited successfully</result>")
        except FileNotFoundError:
            output_entries.append("<error>File not found</error>")
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except:
            output_entries.append("<error>Could not edit file</error>")
        if number_of_occurrences is not None:
            output_entries.append(f"<number_of_occurrences>{number_of_occurrences}</number_of_occurrences>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<file_edit path="{path}" number_of_substitutions="{number_of_substitutions}">\n{joined_output_entries}\n</file_edit>'
