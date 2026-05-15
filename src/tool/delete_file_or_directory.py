from pathlib import Path
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class DeleteFileOrDirectoryArguments(TypedDict):
    type: Required[str]
    path: Required[str]


class DeleteFileOrDirectoryToolCall(BaseToolCall):
    tool_name: Required[Literal["delete_file_or_directory"]]
    arguments: Required[DeleteFileOrDirectoryArguments]


def get_delete_file_or_directory_message(tool_call: DeleteFileOrDirectoryToolCall) -> str:
    return f"Deleting **{tool_call['arguments']['path']}** (**{tool_call['arguments']['type']}**)"


def delete_file_or_directory(type: str, path: str, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>File or directory deletion manually denied by the user</error>")
    else:
        try:
            path_obj: Path = Path(path)
            if not path_obj.exists():
                output_entries.append("<error>File or directory not found</error>")
            elif type == "file":
                if path_obj.is_dir():
                    output_entries.append("<error>Expected a file but found a directory</error>")
                else:
                    path_obj.unlink()
                    output_entries.append("<result>File deleted successfully</result>")
            elif type == "directory":
                if path_obj.is_file():
                    output_entries.append("<error>Expected a directory but found a file</error>")
                else:
                    path_obj.rmdir()
                    output_entries.append("<result>Directory deleted successfully</result>")
            else:
                output_entries.append(f'<error>Invalid type "{type}"</error>')
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except:
            output_entries.append("<error>Could not delete file or directory</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<file_or_directory_deletion type="{type}" path="{path}">\n{joined_output_entries}\n</file_or_directory_deletion>'
