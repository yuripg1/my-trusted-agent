from pathlib import Path
from unittest.mock import patch

from tool.create_directory import create_directory


class TestCreateDirectory:
    """Tests for the `create_directory` tool"""

    def test_success_new_directory(self, tmp_path: Path) -> None:
        """Create a new directory that does not exist"""
        target: Path = tmp_path.joinpath("new_dir")
        result: str = create_directory(str(target))
        assert target.is_dir()
        assert (
            f'<directory_creation path="{target}">\n<result>Directory created successfully</result>\n</directory_creation>'
            in result
        )

    def test_directory_already_exists(self, tmp_path: Path) -> None:
        """Create a directory that already exists (exist_ok=True)"""
        target: Path = tmp_path.joinpath("existing")
        target.mkdir()
        result: str = create_directory(str(target))
        assert target.is_dir()
        assert (
            f'<directory_creation path="{target}">\n<result>Directory created successfully</result>\n</directory_creation>'
            in result
        )

    def test_nested_directories(self, tmp_path: Path) -> None:
        """Create nested directories (parents=True)"""
        target: Path = tmp_path.joinpath("a", "b", "c")
        result: str = create_directory(str(target))
        assert target.is_dir()
        assert (
            f'<directory_creation path="{target}">\n<result>Directory created successfully</result>\n</directory_creation>'
            in result
        )

    def test_permission_error(self) -> None:
        """Do not create directory due to permission error"""
        with patch.object(Path, "mkdir", side_effect=PermissionError("Permission error")):
            target: str = "/some/protected/path"
            result: str = create_directory(target)
            assert (
                f'<directory_creation path="{target}">\n<error>Permission denied by the system</error>\n</directory_creation>'
                in result
            )
            assert "<result>" not in result

    def test_unknown_error(self) -> None:
        """Do not create directory due to an exception"""
        with patch.object(Path, "mkdir", side_effect=Exception("Exception")):
            target: str = "/some/path"
            result: str = create_directory(target)
            assert (
                f'<directory_creation path="{target}">\n<error>Could not create directory</error>\n</directory_creation>'
                in result
            )
            assert "<result>" not in result
