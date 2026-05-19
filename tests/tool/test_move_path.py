from pathlib import Path
from unittest.mock import patch

from tool.move_path import MovePathToolCall, get_move_path_message, get_move_path_permission, move_path


class TestGetMovePathMessage:
    """Tests for the `get_move_path_message` function"""

    def test_format(self) -> None:
        """Format the message correctly"""
        tool_call: MovePathToolCall = {
            "tool_name": "move_path",
            "arguments": {"type": "file", "source": "/old/path.txt", "destination": "/new/path.txt"},
        }
        assert get_move_path_message(tool_call) == "Moving **file** at **/old/path.txt** to **/new/path.txt**"


class TestGetMovePathPermission:
    """Tests for the `get_move_path_permission` function"""

    def test_requires_approval(self) -> None:
        """Permission should require user approval"""
        tool_call: MovePathToolCall = {
            "tool_name": "move_path",
            "arguments": {"type": "file", "source": "/old/path.txt", "destination": "/new/path.txt"},
        }
        assert get_move_path_permission(tool_call) is False


class TestMovePath:
    """Tests for the `move_path` tool"""

    def test_move_file_successfully(self, tmp_path: Path) -> None:
        """Move a file to a new location"""
        type: str = "file"
        source: Path = tmp_path.joinpath("source.txt")
        source.write_text("content")
        destination: Path = tmp_path.joinpath("dest", "moved.txt")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<result>File moved successfully</result>\n</path_move>'
        )
        assert not source.exists()
        assert destination.read_text() == "content"

    def test_rename_file_same_directory(self, tmp_path: Path) -> None:
        """Rename a file within the same directory"""
        type: str = "file"
        source: Path = tmp_path.joinpath("old_name.txt")
        source.write_text("content")
        destination: Path = tmp_path.joinpath("new_name.txt")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<result>File moved successfully</result>\n</path_move>'
        )
        assert not source.exists()
        assert destination.read_text() == "content"

    def test_move_directory_successfully(self, tmp_path: Path) -> None:
        """Move a directory to a new location"""
        type: str = "directory"
        source: Path = tmp_path.joinpath("mydir")
        source.mkdir()
        source.joinpath("file.txt").write_text("hello")
        destination: Path = tmp_path.joinpath("newdir")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<result>Directory moved successfully</result>\n</path_move>'
        )
        assert not source.exists()
        assert destination.is_dir()
        assert destination.joinpath("file.txt").read_text() == "hello"

    def test_move_symlink_to_file(self, tmp_path: Path) -> None:
        """Move a symlink pointing to a file"""
        type: str = "symlink"
        target: Path = tmp_path.joinpath("target.txt")
        target.write_text("content")
        source: Path = tmp_path.joinpath("link")
        source.symlink_to(target)
        destination: Path = tmp_path.joinpath("moved_link")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<result>Symlink moved successfully</result>\n</path_move>'
        )
        assert not source.exists()
        assert destination.is_symlink()
        assert destination.read_text() == "content"

    def test_move_symlink_to_directory(self, tmp_path: Path) -> None:
        """Move a symlink pointing to a directory"""
        type: str = "symlink"
        target: Path = tmp_path.joinpath("target_dir")
        target.mkdir()
        target.joinpath("note.txt").write_text("hello")
        source: Path = tmp_path.joinpath("link_to_dir")
        source.symlink_to(target)
        destination: Path = tmp_path.joinpath("moved_link_to_dir")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<result>Symlink moved successfully</result>\n</path_move>'
        )
        assert not source.exists()
        assert destination.is_symlink()
        assert destination.joinpath("note.txt").read_text() == "hello"

    def test_move_broken_symlink(self, tmp_path: Path) -> None:
        """Move a symlink whose target no longer exists"""
        type: str = "symlink"
        target: Path = tmp_path.joinpath("ghost.txt")
        target.write_text("content")
        source: Path = tmp_path.joinpath("broken_link")
        source.symlink_to(target)
        target.unlink()
        assert not target.exists()
        destination: Path = tmp_path.joinpath("moved_broken_link")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<result>Symlink moved successfully</result>\n</path_move>'
        )
        assert not source.exists()
        assert destination.is_symlink()
        assert not destination.exists()

    def test_source_not_found(self, tmp_path: Path) -> None:
        """Do not move a path that does not exist"""
        type: str = "file"
        source: Path = tmp_path.joinpath("nonexistent")
        destination: Path = tmp_path.joinpath("nowhere")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<error>Source path not found</error>\n</path_move>'
        )

    def test_type_mismatch_file_as_directory(self, tmp_path: Path) -> None:
        """Do not move a file when type is 'directory'"""
        type: str = "directory"
        source: Path = tmp_path.joinpath("file.txt")
        source.write_text("content")
        destination: Path = tmp_path.joinpath("dest")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<error>Expected a directory but found a different type</error>\n</path_move>'
        )
        assert source.exists()

    def test_type_mismatch_directory_as_file(self, tmp_path: Path) -> None:
        """Do not move a directory when type is 'file'"""
        type: str = "file"
        source: Path = tmp_path.joinpath("mydir")
        source.mkdir()
        destination: Path = tmp_path.joinpath("dest")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<error>Expected a file but found a different type</error>\n</path_move>'
        )
        assert source.exists()

    def test_type_mismatch_symlink_as_file(self, tmp_path: Path) -> None:
        """Do not move a symlink when type is 'file'"""
        type: str = "file"
        target: Path = tmp_path.joinpath("target.txt")
        target.write_text("content")
        source: Path = tmp_path.joinpath("link")
        source.symlink_to(target)
        destination: Path = tmp_path.joinpath("dest")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<error>Expected a file but found a different type</error>\n</path_move>'
        )
        assert source.exists()

    def test_type_mismatch_dir_as_symlink(self, tmp_path: Path) -> None:
        """Do not move a directory when type is 'symlink'"""
        type: str = "symlink"
        source: Path = tmp_path.joinpath("mydir")
        source.mkdir()
        destination: Path = tmp_path.joinpath("dest")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<error>Expected a symlink but found a different type</error>\n</path_move>'
        )
        assert source.exists()

    def test_invalid_type(self, tmp_path: Path) -> None:
        """Do not move a path with an invalid type"""
        type: str = "socket"
        source: Path = tmp_path.joinpath("file.txt")
        source.write_text("content")
        destination: Path = tmp_path.joinpath("dest")
        result: str = move_path(type, str(source), str(destination))
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<error>Invalid type "{type}"</error>\n</path_move>'
        )
        assert source.exists()

    def test_permission_error(self, tmp_path: Path) -> None:
        """Do not move a path due to permission error"""
        type: str = "file"
        source: Path = tmp_path.joinpath("file.txt")
        source.write_text("content")
        destination: Path = tmp_path.joinpath("dest.txt")
        with patch("tool.move_path.move", side_effect=PermissionError("Permission error")):
            result: str = move_path(type, str(source), str(destination))
            assert (
                result
                == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<error>Permission denied by the system</error>\n</path_move>'
            )
            assert source.exists()

    def test_unknown_error(self, tmp_path: Path) -> None:
        """Do not move a path due to an exception"""
        type: str = "file"
        source: Path = tmp_path.joinpath("file.txt")
        source.write_text("content")
        destination: Path = tmp_path.joinpath("dest.txt")
        with patch("tool.move_path.move", side_effect=Exception("Exception")):
            result: str = move_path(type, str(source), str(destination))
            assert (
                result
                == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<error>Could not move path</error>\n</path_move>'
            )
            assert source.exists()

    def test_moving_denied_by_user(self, tmp_path: Path) -> None:
        """Do not move a path due to being denied by the user"""
        type: str = "file"
        source: Path = tmp_path.joinpath("file.txt")
        source.write_text("content")
        destination: Path = tmp_path.joinpath("dest.txt")
        result: str = move_path(type, str(source), str(destination), tool_call_permission=False)
        assert (
            result
            == f'<path_move type="{type}" source="{str(source)}" destination="{str(destination)}">\n<error>Path moving manually denied by the user. The path was not moved</error>\n</path_move>'
        )
        assert source.exists()
