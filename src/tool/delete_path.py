from pathlib import Path
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class DeletePathArguments(TypedDict):
    type: Required[str]
    path: Required[str]


class DeletePathToolCall(BaseToolCall):
    tool_name: Required[Literal["delete_path"]]
    arguments: Required[DeletePathArguments]


def get_delete_path_message(tool_call: DeletePathToolCall) -> str:
    return f"Deleting **{tool_call['arguments']['path']}** (**{tool_call['arguments']['type']}**)"


def get_delete_path_permission(tool_call: DeletePathToolCall) -> bool:
    return False


def delete_path(type: str, path: str, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>Path deletion manually denied by the user. The path was not deleted</error>")
    else:
        try:
            path_obj: Path = Path(path)
            if not path_obj.exists() and not path_obj.is_symlink():
                output_entries.append("<error>Path not found</error>")
            elif type == "file":
                if path_obj.is_file() and not path_obj.is_symlink():
                    path_obj.unlink()
                    output_entries.append("<result>File deleted successfully</result>")
                else:
                    output_entries.append("<error>Expected a file but found a different type</error>")
            elif type == "directory":
                if path_obj.is_dir() and not path_obj.is_symlink():
                    path_obj.rmdir()
                    output_entries.append("<result>Directory deleted successfully</result>")
                else:
                    output_entries.append("<error>Expected a directory but found a different type</error>")
            elif type == "symlink":
                if path_obj.is_symlink():
                    path_obj.unlink()
                    output_entries.append("<result>Symlink deleted successfully</result>")
                else:
                    output_entries.append("<error>Expected a symlink but found a different type</error>")
            else:
                output_entries.append(f'<error>Invalid type "{type}"</error>')
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except Exception:
            output_entries.append("<error>Could not delete path</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<path_deletion type="{type}" path="{path}">\n{joined_output_entries}\n</path_deletion>'
