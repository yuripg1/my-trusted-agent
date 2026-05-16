from pathlib import Path
from unittest.mock import patch

from tool.create_directory import (
    CreateDirectoryToolCall,
    create_directory,
    get_create_directory_message,
    get_create_directory_permission,
)


class TestGetCreateDirectoryMessage:
    """Tests for the `get_create_directory_message` function"""

    def test_format(self) -> None:
        """Format the message correctly"""
        tool_call: CreateDirectoryToolCall = {"tool_name": "create_directory", "arguments": {"path": "/some/path"}}
        assert get_create_directory_message(tool_call) == "Creating directory at **/some/path**"


class TestGetCreateDirectoryPermission:
    """Tests for the `get_create_directory_permission` function"""

    def test_auto_approved(self) -> None:
        """Permission should be automatically granted"""
        tool_call: CreateDirectoryToolCall = {"tool_name": "create_directory", "arguments": {"path": "/some/path"}}
        assert get_create_directory_permission(tool_call) is True


class TestCreateDirectory:
    """Tests for the `create_directory` tool"""

    def test_success_new_directory(self, tmp_path: Path) -> None:
        """Create a new directory that does not exist"""
        target: Path = tmp_path.joinpath("new_dir")
        result: str = create_directory(str(target))
        assert (
            result
            == f'<directory_creation path="{str(target)}">\n<result>Directory created successfully</result>\n</directory_creation>'
        )
        assert target.is_dir()

    def test_directory_already_exists(self, tmp_path: Path) -> None:
        """Create a directory that already exists (exist_ok=True)"""
        target: Path = tmp_path.joinpath("existing")
        target.mkdir()
        result: str = create_directory(str(target))
        assert (
            result
            == f'<directory_creation path="{str(target)}">\n<result>Directory created successfully</result>\n</directory_creation>'
        )
        assert target.is_dir()

    def test_nested_directories(self, tmp_path: Path) -> None:
        """Create nested directories (parents=True)"""
        target: Path = tmp_path.joinpath("a", "b", "c")
        result: str = create_directory(str(target))
        assert (
            result
            == f'<directory_creation path="{str(target)}">\n<result>Directory created successfully</result>\n</directory_creation>'
        )
        assert target.is_dir()

    def test_permission_error(self, tmp_path: Path) -> None:
        """Do not create directory due to permission error"""
        target: Path = tmp_path.joinpath("protected")
        with patch.object(Path, "mkdir", side_effect=PermissionError("Permission error")):
            result: str = create_directory(str(target))
            assert (
                result
                == f'<directory_creation path="{str(target)}">\n<error>Permission denied by the system</error>\n</directory_creation>'
            )
            assert not target.exists()

    def test_unknown_error(self, tmp_path: Path) -> None:
        """Do not create directory due to an exception"""
        target: Path = tmp_path.joinpath("protected")
        with patch.object(Path, "mkdir", side_effect=Exception("Exception")):
            result: str = create_directory(str(target))
            assert (
                result
                == f'<directory_creation path="{str(target)}">\n<error>Could not create directory</error>\n</directory_creation>'
            )
            assert not target.exists()
