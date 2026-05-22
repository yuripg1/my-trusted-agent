from pathlib import Path
from shutil import move
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class MovePathArguments(TypedDict):
    type: Required[str]
    source: Required[str]
    destination: Required[str]


class MovePathToolCall(BaseToolCall):
    tool_name: Required[Literal["move_path"]]
    arguments: Required[MovePathArguments]


def get_move_path_message(tool_call: MovePathToolCall) -> str:
    return f"Moving **{tool_call['arguments']['type']}** at **{tool_call['arguments']['source']}** to **{tool_call['arguments']['destination']}**"


def get_move_path_permission(tool_call: MovePathToolCall) -> bool:
    return False


def move_path(arguments: MovePathArguments, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>Path moving manually denied by the user. The path was not moved</error>")
    else:
        try:
            source_path: Path = Path(arguments["source"])
            dest_path: Path = Path(arguments["destination"])
            if not source_path.exists() and not source_path.is_symlink():
                output_entries.append("<error>Source path not found</error>")
            elif arguments["type"] == "file":
                if source_path.is_file() and not source_path.is_symlink():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    move(str(source_path), str(dest_path))
                    output_entries.append("<result>File moved successfully</result>")
                else:
                    output_entries.append("<error>Expected a file but found a different type</error>")
            elif arguments["type"] == "directory":
                if source_path.is_dir() and not source_path.is_symlink():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    move(str(source_path), str(dest_path))
                    output_entries.append("<result>Directory moved successfully</result>")
                else:
                    output_entries.append("<error>Expected a directory but found a different type</error>")
            elif arguments["type"] == "symlink":
                if source_path.is_symlink():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    move(str(source_path), str(dest_path))
                    output_entries.append("<result>Symlink moved successfully</result>")
                else:
                    output_entries.append("<error>Expected a symlink but found a different type</error>")
            else:
                output_entries.append(f'<error>Invalid type "{arguments["type"]}"</error>')
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except Exception:
            output_entries.append("<error>Could not move path</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<path_move type="{arguments["type"]}" source="{arguments["source"]}" destination="{arguments["destination"]}">\n{joined_output_entries}\n</path_move>'
