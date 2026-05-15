from pathlib import Path
from unittest.mock import patch

from tool.delete_file_or_directory import delete_file_or_directory


class TestDeleteFileOrDirectory:
    """Tests for the `delete_file_or_directory` tool"""

    def test_delete_file_successfully(self, tmp_path: Path) -> None:
        """Delete a file that exists"""
        type: str = "file"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = delete_file_or_directory(type, str(target))
        assert (
            result
            == f'<file_or_directory_deletion type="{type}" path="{str(target)}">\n<result>File deleted successfully</result>\n</file_or_directory_deletion>'
        )
        assert not target.exists()

    def test_delete_directory_successfully(self, tmp_path: Path) -> None:
        """Delete a directory that exists and is empty"""
        type: str = "directory"
        target: Path = tmp_path.joinpath("dir")
        target.mkdir()
        result: str = delete_file_or_directory(type, str(target))
        assert (
            result
            == f'<file_or_directory_deletion type="{type}" path="{str(target)}">\n<result>Directory deleted successfully</result>\n</file_or_directory_deletion>'
        )
        assert not target.exists()

    def test_path_not_found(self, tmp_path: Path) -> None:
        """Do not delete a path that does not exist"""
        type: str = "file"
        target: Path = tmp_path.joinpath("nonexistent")
        result: str = delete_file_or_directory(type, str(target))
        assert (
            result
            == f'<file_or_directory_deletion type="{type}" path="{str(target)}">\n<error>File or directory not found</error>\n</file_or_directory_deletion>'
        )

    def test_directory_as_file(self, tmp_path: Path) -> None:
        """Do not delete a directory with type 'file'"""
        type: str = "file"
        target: Path = tmp_path.joinpath("dir")
        target.mkdir()
        result: str = delete_file_or_directory(type, str(target))
        assert (
            result
            == f'<file_or_directory_deletion type="{type}" path="{str(target)}">\n<error>Expected a file but found a directory</error>\n</file_or_directory_deletion>'
        )
        assert target.exists()

    def test_file_as_directory(self, tmp_path: Path) -> None:
        """Do not delete a file with type 'directory'"""
        type: str = "directory"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = delete_file_or_directory(type, str(target))
        assert (
            result
            == f'<file_or_directory_deletion type="{type}" path="{str(target)}">\n<error>Expected a directory but found a file</error>\n</file_or_directory_deletion>'
        )
        assert target.exists()

    def test_invalid_type(self, tmp_path: Path) -> None:
        """Do not delete a path with an invalid type"""
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = delete_file_or_directory("symlink", str(target))
        assert (
            result
            == f'<file_or_directory_deletion type="symlink" path="{str(target)}">\n<error>Invalid type "symlink"</error>\n</file_or_directory_deletion>'
        )
        assert target.exists()

    def test_permission_error(self, tmp_path: Path) -> None:
        """Do not delete a path due to permission error"""
        type: str = "file"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        with patch.object(Path, "unlink", side_effect=PermissionError("Permission error")):
            result: str = delete_file_or_directory(type, str(target))
            assert (
                result
                == f'<file_or_directory_deletion type="{type}" path="{str(target)}">\n<error>Permission denied by the system</error>\n</file_or_directory_deletion>'
            )
            assert target.exists()

    def test_unknown_error(self, tmp_path: Path) -> None:
        """Do not delete a path due to an exception"""
        type: str = "file"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        with patch.object(Path, "unlink", side_effect=Exception("Exception")):
            result: str = delete_file_or_directory(type, str(target))
            assert (
                result
                == f'<file_or_directory_deletion type="{type}" path="{str(target)}">\n<error>Could not delete file or directory</error>\n</file_or_directory_deletion>'
            )
            assert target.exists()

    def test_deletion_denied_by_user(self, tmp_path: Path) -> None:
        """Do not delete a path due to being denied by the user"""
        type: str = "file"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = delete_file_or_directory(type, str(target), tool_call_permission=False)
        assert (
            result
            == f'<file_or_directory_deletion type="{type}" path="{str(target)}">\n<error>File or directory deletion manually denied by the user</error>\n</file_or_directory_deletion>'
        )
        assert target.exists()
