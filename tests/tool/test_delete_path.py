from pathlib import Path
from unittest.mock import patch

from tool.delete_path import DeletePathToolCall, delete_path, get_delete_path_message, get_delete_path_permission


class TestGetDeletePathMessage:
    """Tests for the `get_delete_path_message` function"""

    def test_format(self) -> None:
        """Format the message correctly"""
        tool_call: DeletePathToolCall = {
            "tool_name": "delete_path",
            "arguments": {"type": "file", "path": "/some/file.txt"},
        }
        assert get_delete_path_message(tool_call) == "Deleting **/some/file.txt** (**file**)"


class TestGetDeletePathPermission:
    """Tests for the `get_delete_path_permission` function"""

    def test_requires_approval(self) -> None:
        """Permission should require user approval"""
        tool_call: DeletePathToolCall = {
            "tool_name": "delete_path",
            "arguments": {"type": "file", "path": "/some/file.txt"},
        }
        assert get_delete_path_permission(tool_call) is False


class TestDeletePath:
    """Tests for the `delete_path` tool"""

    def test_delete_file_successfully(self, tmp_path: Path) -> None:
        """Delete a file that exists"""
        type: str = "file"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = delete_path(type, str(target))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(target)}">\n<result>File deleted successfully</result>\n</path_deletion>'
        )
        assert not target.exists()

    def test_delete_directory_successfully(self, tmp_path: Path) -> None:
        """Delete a directory that exists and is empty"""
        type: str = "directory"
        target: Path = tmp_path.joinpath("dir")
        target.mkdir()
        result: str = delete_path(type, str(target))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(target)}">\n<result>Directory deleted successfully</result>\n</path_deletion>'
        )
        assert not target.exists()

    def test_path_not_found(self, tmp_path: Path) -> None:
        """Do not delete a path that does not exist"""
        type: str = "file"
        target: Path = tmp_path.joinpath("nonexistent")
        result: str = delete_path(type, str(target))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(target)}">\n<error>Path not found</error>\n</path_deletion>'
        )

    def test_directory_as_file(self, tmp_path: Path) -> None:
        """Do not delete a directory with type 'file'"""
        type: str = "file"
        target: Path = tmp_path.joinpath("dir")
        target.mkdir()
        result: str = delete_path(type, str(target))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(target)}">\n<error>Expected a file but found a different type</error>\n</path_deletion>'
        )
        assert target.exists()

    def test_file_as_directory(self, tmp_path: Path) -> None:
        """Do not delete a file with type 'directory'"""
        type: str = "directory"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = delete_path(type, str(target))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(target)}">\n<error>Expected a directory but found a different type</error>\n</path_deletion>'
        )
        assert target.exists()

    def test_invalid_type(self, tmp_path: Path) -> None:
        """Do not delete a path with an invalid type"""
        type: str = "socket"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = delete_path(type, str(target))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(target)}">\n<error>Invalid type "{type}"</error>\n</path_deletion>'
        )
        assert target.exists()

    def test_delete_symlink_to_file(self, tmp_path: Path) -> None:
        """Delete a symlink pointing to a file"""
        type: str = "file"
        target: Path = tmp_path.joinpath("target.txt")
        target.write_text("content")
        symlink: Path = tmp_path.joinpath("link_to_file")
        symlink.symlink_to(target)
        result: str = delete_path(type, str(symlink))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(symlink)}">\n<result>Symlink deleted successfully</result>\n</path_deletion>'
        )
        assert not symlink.exists()
        assert symlink.is_symlink() is False
        assert target.exists()

    def test_delete_symlink_to_directory(self, tmp_path: Path) -> None:
        """Delete a symlink pointing to a directory (which previously failed)"""
        type: str = "directory"
        target: Path = tmp_path.joinpath("target_dir")
        target.mkdir()
        symlink: Path = tmp_path.joinpath("link_to_dir")
        symlink.symlink_to(target)
        result: str = delete_path(type, str(symlink))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(symlink)}">\n<result>Symlink deleted successfully</result>\n</path_deletion>'
        )
        assert not symlink.exists()
        assert symlink.is_symlink() is False
        assert target.exists()

    def test_delete_symlink_with_explicit_type(self, tmp_path: Path) -> None:
        """Delete a symlink using type='symlink' explicitly"""
        type: str = "symlink"
        target: Path = tmp_path.joinpath("target.txt")
        target.write_text("content")
        symlink: Path = tmp_path.joinpath("my_link")
        symlink.symlink_to(target)
        result: str = delete_path(type, str(symlink))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(symlink)}">\n<result>Symlink deleted successfully</result>\n</path_deletion>'
        )
        assert not symlink.exists()
        assert target.exists()

    def test_symlink_type_on_regular_file(self, tmp_path: Path) -> None:
        """Do not delete a regular file with type='symlink'"""
        type: str = "symlink"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = delete_path(type, str(target))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(target)}">\n<error>Expected a symlink but found a different type</error>\n</path_deletion>'
        )
        assert target.exists()

    def test_symlink_type_on_directory(self, tmp_path: Path) -> None:
        """Do not delete a regular directory with type='symlink'"""
        type: str = "symlink"
        target: Path = tmp_path.joinpath("dir")
        target.mkdir()
        result: str = delete_path(type, str(target))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(target)}">\n<error>Expected a symlink but found a different type</error>\n</path_deletion>'
        )
        assert target.exists()

    def test_delete_broken_symlink(self, tmp_path: Path) -> None:
        """Delete a symlink whose target no longer exists"""
        type: str = "file"
        target: Path = tmp_path.joinpath("target.txt")
        target.write_text("content")
        symlink: Path = tmp_path.joinpath("broken_link")
        symlink.symlink_to(target)
        target.unlink()
        assert not target.exists()
        assert symlink.is_symlink()
        result: str = delete_path(type, str(symlink))
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(symlink)}">\n<result>Symlink deleted successfully</result>\n</path_deletion>'
        )
        assert not symlink.exists()

    def test_permission_error(self, tmp_path: Path) -> None:
        """Do not delete a path due to permission error"""
        type: str = "file"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        with patch.object(Path, "unlink", side_effect=PermissionError("Permission error")):
            result: str = delete_path(type, str(target))
            assert (
                result
                == f'<path_deletion type="{type}" path="{str(target)}">\n<error>Permission denied by the system</error>\n</path_deletion>'
            )
            assert target.exists()

    def test_unknown_error(self, tmp_path: Path) -> None:
        """Do not delete a path due to an exception"""
        type: str = "file"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        with patch.object(Path, "unlink", side_effect=Exception("Exception")):
            result: str = delete_path(type, str(target))
            assert (
                result
                == f'<path_deletion type="{type}" path="{str(target)}">\n<error>Could not delete path</error>\n</path_deletion>'
            )
            assert target.exists()

    def test_deletion_denied_by_user(self, tmp_path: Path) -> None:
        """Do not delete a path due to being denied by the user"""
        type: str = "file"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = delete_path(type, str(target), tool_call_permission=False)
        assert (
            result
            == f'<path_deletion type="{type}" path="{str(target)}">\n<error>Path deletion manually denied by the user</error>\n</path_deletion>'
        )
        assert target.exists()
