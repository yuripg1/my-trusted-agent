from pathlib import Path
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class ListDirectoryArguments(TypedDict):
    path: Required[str]


class ListDirectoryToolCall(BaseToolCall):
    tool_name: Required[Literal["list_directory"]]
    arguments: Required[ListDirectoryArguments]


_MAX_COMPRESSION_DEPTH: int = 10


def get_list_directory_message(tool_call: ListDirectoryToolCall) -> str:
    return f"Listing directory at **{tool_call['arguments']['path']}**"


def get_list_directory_permission(tool_call: ListDirectoryToolCall) -> bool:
    return True


def _resolve_compressed_path(path: Path, depth: int = 0) -> Path:
    if depth >= _MAX_COMPRESSION_DEPTH:
        return path
    try:
        entries: list[Path] = list(path.iterdir())
        if len(entries) == 1 and entries[0].is_dir() and not entries[0].is_symlink():
            return _resolve_compressed_path(entries[0], depth + 1)
    except Exception:
        pass
    return path


def list_directory(path: str) -> str:
    output_entries: list[str] = []
    try:
        directory_path: Path = Path(path)
        compressed_path: Path = _resolve_compressed_path(directory_path)
        prefix_relative: str = ""
        if compressed_path != directory_path:
            prefix_relative = str(compressed_path.relative_to(directory_path)) + "/"
        for directory_entry in compressed_path.iterdir():
            directory_entry_type: str = ""
            extra_attributes: str = ""
            entry_name: str = prefix_relative + directory_entry.name
            try:
                if directory_entry.is_symlink():
                    directory_entry_type = "symlink"
                    try:
                        target_path: Path = directory_entry.resolve(strict=False)
                        target: str = str(target_path)
                        extra_attributes += f' target="{target}"'
                        if target_path.is_dir():
                            extra_attributes += ' target_type="directory"'
                        elif target_path.is_file():
                            extra_attributes += ' target_type="file"'
                    except Exception:
                        pass
                elif directory_entry.is_dir():
                    directory_entry_type = "directory"
                elif directory_entry.is_file():
                    directory_entry_type = "file"
            except Exception:
                pass
            if len(directory_entry_type) != 0:
                output_entries.append(f'<entry type="{directory_entry_type}"{extra_attributes}>{entry_name}</entry>')
            else:
                output_entries.append(f"<entry>{entry_name}</entry>")
    except FileNotFoundError:
        output_entries.append("<error>Directory not found</error>")
    except NotADirectoryError:
        output_entries.append("<error>Path is not a directory</error>")
    except PermissionError:
        output_entries.append("<error>Permission denied by the system</error>")
    except Exception:
        output_entries.append("<error>Could not list directory</error>")
    if len(output_entries) == 0:
        output_entries.append("<note>The directory is empty</note>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<directory_listing path="{path}">\n{joined_output_entries}\n</directory_listing>'
