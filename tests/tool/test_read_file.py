from pathlib import Path
from unittest.mock import patch

from tool.read_file import ReadFileToolCall, get_read_file_message, read_file


class TestGetReadFileMessage:
    """Tests for the `get_read_file_message` function"""

    def test_format(self) -> None:
        """Format the message correctly"""
        tool_call: ReadFileToolCall = {"tool_name": "read_file", "arguments": {"path": "/some/file.txt"}}
        assert get_read_file_message(tool_call) == "Reading file at **/some/file.txt**"


class TestReadFile:
    """Tests for the `read_file` tool"""

    def test_read_file_successfully(self, tmp_path: Path) -> None:
        """Read a file that exists"""
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("hello world")
        result: str = read_file(str(target))
        assert result == f'<file_read path="{str(target)}">\n<content>\nhello world\n</content>\n</file_read>'

    def test_read_file_not_found(self, tmp_path: Path) -> None:
        """Do not read a file that does not exist"""
        target: Path = tmp_path.joinpath("nonexistent")
        result: str = read_file(str(target))
        assert result == f'<file_read path="{str(target)}">\n<error>File not found</error>\n</file_read>'

    def test_permission_denied(self, tmp_path: Path) -> None:
        """Do not read a file due to permission error"""
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        with patch("builtins.open", side_effect=PermissionError("Permission error")):
            result: str = read_file(str(target))
            assert (
                result
                == f'<file_read path="{str(target)}">\n<error>Permission denied by the system</error>\n</file_read>'
            )

    def test_unknown_error(self, tmp_path: Path) -> None:
        """Do not read a file due to an exception"""
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        with patch("builtins.open", side_effect=Exception("Exception")):
            result: str = read_file(str(target))
            assert result == f'<file_read path="{str(target)}">\n<error>Could not read file</error>\n</file_read>'

    def test_reading_denied_by_user(self, tmp_path: Path) -> None:
        """Do not read a file due to being denied by the user"""
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = read_file(str(target), tool_call_permission=False)
        assert (
            result
            == f'<file_read path="{str(target)}">\n<error>File reading manually denied by the user</error>\n</file_read>'
        )
