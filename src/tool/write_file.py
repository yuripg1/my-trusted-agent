from pathlib import Path
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class WriteFileArguments(TypedDict):
    path: Required[str]
    mode: Required[str]
    content: Required[str]


class WriteFileToolCall(BaseToolCall):
    tool_name: Required[Literal["write_file"]]
    arguments: Required[WriteFileArguments]


def write_file(path: str, mode: str, content: str, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>File writing manually denied by the user</error>")
    else:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            file_path: Path = Path(path)
            if mode == "create_or_overwrite":
                with open(file_path, "w") as file:
                    file.write(content)
                output_entries.append("<result>File written successfully</result>")
            elif mode == "create_if_not_exists":
                if file_path.exists():
                    output_entries.append("<error>File already exists</error>")
                else:
                    with open(file_path, "x") as file:
                        file.write(content)
                    output_entries.append("<result>File written successfully</result>")
            elif mode == "append":
                with open(file_path, "a") as file:
                    file.write(content)
                output_entries.append("<result>File written successfully</result>")
            else:
                output_entries.append(f'<error>Invalid mode "{mode}"</error>')
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except:
            output_entries.append("<error>Could not write file</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<file_write path="{path}" mode="{mode}">\n{joined_output_entries}\n</file_write>'
