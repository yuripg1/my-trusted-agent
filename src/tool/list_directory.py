from pathlib import Path
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class ListDirectoryArguments(TypedDict):
    path: Required[str]


class ListDirectoryToolCall(BaseToolCall):
    tool_name: Required[Literal["list_directory"]]
    arguments: Required[ListDirectoryArguments]


def get_list_directory_message(tool_call: ListDirectoryToolCall) -> str:
    return f"Listing directory at **{tool_call['arguments']['path']}**"


def list_directory(path: str) -> str:
    output_entries: list[str] = []
    try:
        directory_path: Path = Path(path)
        for directory_entry in directory_path.iterdir():
            directory_entry_type: str = ""
            try:
                if directory_entry.is_symlink():
                    directory_entry_type = "symlink"
                elif directory_entry.is_dir():
                    directory_entry_type = "directory"
                elif directory_entry.is_file():
                    directory_entry_type = "file"
            except:
                pass
            if len(directory_entry_type) != 0:
                output_entries.append(f'<entry type="{directory_entry_type}">{directory_entry.name}</entry>')
            else:
                output_entries.append(f"<entry>{directory_entry.name}</entry>")
    except FileNotFoundError:
        output_entries.append("<error>Directory not found</error>")
    except NotADirectoryError:
        output_entries.append("<error>Path is not a directory</error>")
    except PermissionError:
        output_entries.append("<error>Permission denied by the system</error>")
    except:
        output_entries.append("<error>Could not list directory</error>")
    if len(output_entries) == 0:
        output_entries.append("<error>No entries found</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<directory_listing path="{path}">\n{joined_output_entries}\n</directory_listing>'
