from pathlib import Path
from unittest.mock import patch

from tool.write_file import write_file


class TestWriteFile:
    """Tests for the `write_file` tool"""

    def test_create_or_overwrite_new_file(self, tmp_path: Path) -> None:
        """Create a new file with 'create_or_overwrite' mode"""
        target: Path = tmp_path.joinpath("new_file.txt")
        mode: str = "create_or_overwrite"
        content: str = "hello"
        result: str = write_file(str(target), mode, content)
        assert (
            result
            == f'<file_write path="{str(target)}" mode="{mode}">\n<result>File written successfully</result>\n</file_write>'
        )
        assert target.read_text() == content

    def test_create_or_overwrite_existing_file(self, tmp_path: Path) -> None:
        """Overwrite an existing file with 'create_or_overwrite' mode"""
        target: Path = tmp_path.joinpath("existing.txt")
        target.write_text("old content")
        mode: str = "create_or_overwrite"
        content: str = "new content"
        result: str = write_file(str(target), mode, content)
        assert (
            result
            == f'<file_write path="{str(target)}" mode="{mode}">\n<result>File written successfully</result>\n</file_write>'
        )
        assert target.read_text() == content

    def test_create_if_not_exists_new_file(self, tmp_path: Path) -> None:
        """Create a new file with 'create_if_not_exists' mode"""
        target: Path = tmp_path.joinpath("new_file.txt")
        mode: str = "create_if_not_exists"
        content: str = "hello"
        result: str = write_file(str(target), mode, content)
        assert (
            result
            == f'<file_write path="{str(target)}" mode="{mode}">\n<result>File written successfully</result>\n</file_write>'
        )
        assert target.read_text() == content

    def test_create_if_not_exists_existing_file(self, tmp_path: Path) -> None:
        """Do not create a file that already exists with 'create_if_not_exists' mode"""
        original_content: str = "original"
        target: Path = tmp_path.joinpath("existing.txt")
        target.write_text(original_content)
        mode: str = "create_if_not_exists"
        content: str = "new content"
        result: str = write_file(str(target), mode, content)
        assert (
            result
            == f'<file_write path="{str(target)}" mode="{mode}">\n<error>File already exists</error>\n</file_write>'
        )
        assert target.read_text() == original_content

    def test_append_to_existing_file(self, tmp_path: Path) -> None:
        """Append to an existing file with 'append' mode"""
        original_content: str = "line1\n"
        target: Path = tmp_path.joinpath("log.txt")
        target.write_text(original_content)
        mode: str = "append"
        content: str = "line2\n"
        result: str = write_file(str(target), mode, content)
        assert (
            result
            == f'<file_write path="{str(target)}" mode="{mode}">\n<result>File written successfully</result>\n</file_write>'
        )
        assert target.read_text() == f"{original_content}{content}"

    def test_append_to_new_file(self, tmp_path: Path) -> None:
        """Append to a new file with 'append' mode (creates it)"""
        target: Path = tmp_path.joinpath("new_log.txt")
        mode: str = "append"
        content: str = "line1\n"
        result: str = write_file(str(target), mode, content)
        assert (
            result
            == f'<file_write path="{str(target)}" mode="{mode}">\n<result>File written successfully</result>\n</file_write>'
        )
        assert target.read_text() == content

    def test_invalid_mode(self, tmp_path: Path) -> None:
        """Do not write a file with an invalid mode"""
        target: Path = tmp_path.joinpath("file.txt")
        mode: str = "invalid"
        content: str = "content"
        result: str = write_file(str(target), mode, content)
        assert (
            result
            == f'<file_write path="{str(target)}" mode="{mode}">\n<error>Invalid mode "{mode}"</error>\n</file_write>'
        )

    def test_permission_error(self, tmp_path: Path) -> None:
        """Do not write a file due to permission error"""
        target: Path = tmp_path.joinpath("file.txt")
        mode: str = "create_or_overwrite"
        content: str = "content"
        with patch("builtins.open", side_effect=PermissionError("Permission error")):
            result: str = write_file(str(target), mode, content)
            assert (
                result
                == f'<file_write path="{str(target)}" mode="{mode}">\n<error>Permission denied by the system</error>\n</file_write>'
            )
            assert not target.exists()

    def test_unknown_error(self, tmp_path: Path) -> None:
        """Do not write a file due to an exception"""
        target: Path = tmp_path.joinpath("file.txt")
        mode: str = "create_or_overwrite"
        content: str = "content"
        with patch("builtins.open", side_effect=Exception("Exception")):
            result: str = write_file(str(target), mode, content)
            assert (
                result
                == f'<file_write path="{str(target)}" mode="{mode}">\n<error>Could not write file</error>\n</file_write>'
            )
            assert not target.exists()

    def test_writing_denied_by_user(self, tmp_path: Path) -> None:
        """Do not write a file due to being denied by the user"""
        target: Path = tmp_path.joinpath("file.txt")
        mode: str = "create_or_overwrite"
        content: str = "content"
        result: str = write_file(str(target), mode, content, tool_call_permission=False)
        assert not target.exists()
        assert (
            result
            == f'<file_write path="{str(target)}" mode="{mode}">\n<error>File writing manually denied by the user</error>\n</file_write>'
        )
