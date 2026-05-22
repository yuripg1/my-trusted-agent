from pathlib import Path
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class DeletePathArguments(TypedDict):
    type: Required[str]
    path: Required[str]


class DeletePathToolCall(BaseToolCall):
    tool_name: Required[Literal["delete_path"]]
    arguments: Required[DeletePathArguments]


def get_delete_path_message(arguments: DeletePathArguments) -> str:
    return f"Deleting **{arguments['path']}** (**{arguments['type']}**)"


def get_delete_path_permission(arguments: DeletePathArguments) -> bool:
    return False


def delete_path(arguments: DeletePathArguments, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>Path deletion manually denied by the user. The path was not deleted</error>")
    else:
        try:
            path_obj: Path = Path(arguments["path"])
            if not path_obj.exists() and not path_obj.is_symlink():
                output_entries.append("<error>Path not found</error>")
            elif arguments["type"] == "file":
                if path_obj.is_file() and not path_obj.is_symlink():
                    path_obj.unlink()
                    output_entries.append("<result>File deleted successfully</result>")
                else:
                    output_entries.append("<error>Expected a file but found a different type</error>")
            elif arguments["type"] == "directory":
                if path_obj.is_dir() and not path_obj.is_symlink():
                    path_obj.rmdir()
                    output_entries.append("<result>Directory deleted successfully</result>")
                else:
                    output_entries.append("<error>Expected a directory but found a different type</error>")
            elif arguments["type"] == "symlink":
                if path_obj.is_symlink():
                    path_obj.unlink()
                    output_entries.append("<result>Symlink deleted successfully</result>")
                else:
                    output_entries.append("<error>Expected a symlink but found a different type</error>")
            else:
                output_entries.append(f'<error>Invalid type "{arguments["type"]}"</error>')
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except Exception:
            output_entries.append("<error>Could not delete path</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<path_deletion type="{arguments["type"]}" path="{arguments["path"]}">\n{joined_output_entries}\n</path_deletion>'
