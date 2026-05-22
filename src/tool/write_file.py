from pathlib import Path
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall, get_language_from_filename, make_safe_code_fence


class WriteFileArguments(TypedDict):
    path: Required[str]
    mode: Required[str]
    content: Required[str]


class WriteFileToolCall(BaseToolCall):
    tool_name: Required[Literal["write_file"]]
    arguments: Required[WriteFileArguments]


def get_write_file_permission(tool_call: WriteFileToolCall) -> bool:
    return False


def get_write_file_message(tool_call: WriteFileToolCall) -> str:
    write_path: str = tool_call["arguments"]["path"]
    write_mode: str = tool_call["arguments"]["mode"]
    write_info_string: str = ""
    if write_mode in ["create_or_overwrite", "create_if_not_exists"]:
        write_info_string = get_language_from_filename(write_path)
    write_formatted_content: str = make_safe_code_fence(tool_call["arguments"]["content"], write_info_string)
    return f"Writing file at **{write_path}** (**{write_mode}** mode)\n\n{write_formatted_content}"


def write_file(arguments: WriteFileArguments, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    content: str = arguments["content"]
    if not tool_call_permission:
        output_entries.append(
            "<error>File writing manually denied by the user. No content was written to the file</error>"
        )
    else:
        wrote_successfully: bool = False
        old_number_of_file_lines: int | None = None
        new_number_of_file_lines: int = 0
        try:
            file_path: Path = Path(arguments["path"])
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if arguments["mode"] == "create_or_overwrite":
                if file_path.exists():
                    with open(file_path) as file:
                        old_number_of_file_lines = len(file.read().splitlines())
                new_number_of_file_lines = len(content.splitlines())
                with open(file_path, "w") as file:
                    file.write(content)
                wrote_successfully = True
            elif arguments["mode"] == "create_if_not_exists":
                if file_path.exists():
                    output_entries.append("<error>File already exists</error>")
                else:
                    new_number_of_file_lines = len(content.splitlines())
                    with open(file_path, "x") as file:
                        file.write(content)
                    wrote_successfully = True
            elif arguments["mode"] == "append":
                if file_path.exists():
                    with open(file_path) as file:
                        old_content: str = file.read()
                    old_number_of_file_lines = len(old_content.splitlines())
                    new_number_of_file_lines = len((old_content + content).splitlines())
                else:
                    new_number_of_file_lines = len(content.splitlines())
                with open(file_path, "a") as file:
                    file.write(content)
                wrote_successfully = True
            else:
                output_entries.append(f'<error>Invalid mode "{arguments["mode"]}"</error>')
            if wrote_successfully:
                output_entries.append("<result>File written successfully</result>")
                if old_number_of_file_lines is not None:
                    output_entries.append(
                        f"<old_number_of_file_lines>{old_number_of_file_lines}</old_number_of_file_lines>"
                    )
                output_entries.append(
                    f"<new_number_of_file_lines>{new_number_of_file_lines}</new_number_of_file_lines>"
                )
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except Exception:
            output_entries.append("<error>Could not write file</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<file_write path="{arguments["path"]}" mode="{arguments["mode"]}">\n{joined_output_entries}\n</file_write>'
