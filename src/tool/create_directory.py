from pathlib import Path
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class CreateDirectoryArguments(TypedDict):
    path: Required[str]


class CreateDirectoryToolCall(BaseToolCall):
    tool_name: Required[Literal["create_directory"]]
    arguments: Required[CreateDirectoryArguments]


def get_create_directory_message(arguments: CreateDirectoryArguments) -> str:
    return f"Creating directory at **{arguments['path']}**"


def get_create_directory_permission(arguments: CreateDirectoryArguments) -> bool:
    return True


def create_directory(arguments: CreateDirectoryArguments) -> str:
    output_entries: list[str] = []
    try:
        Path(arguments["path"]).mkdir(parents=True, exist_ok=True)
        output_entries.append("<result>Directory created successfully</result>")
    except PermissionError:
        output_entries.append("<error>Permission denied by the system</error>")
    except Exception:
        output_entries.append("<error>Could not create directory</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<directory_creation path="{arguments["path"]}">\n{joined_output_entries}\n</directory_creation>'
