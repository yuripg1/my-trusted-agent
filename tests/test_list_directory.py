from pathlib import Path
from unittest.mock import patch

from tool.list_directory import list_directory


class TestListDirectory:
    """Tests for the `list_directory` tool"""

    def test_list_directory_with_entries(self, tmp_path: Path) -> None:
        """List a directory containing files and subdirectories"""
        target: Path = tmp_path.joinpath("docs")
        target.mkdir()
        readme_file: Path = target.joinpath("readme.txt")
        readme_file.write_text("hello")
        symlink: Path = target.joinpath("link_to_readme")
        symlink.symlink_to(readme_file)
        src_directory: Path = target.joinpath("src")
        src_directory.mkdir()
        result: str = list_directory(str(target))
        read_file_entry: str = f'<entry type="file">{readme_file.name}</entry>'
        symlink_entry: str = f'<entry type="symlink">{symlink.name}</entry>'
        sec_directory_entry: str = f'<entry type="directory">{src_directory.name}</entry>'
        assert result.startswith(f'<directory_listing path="{str(target)}">\n<entry')
        assert result.endswith("</entry>\n</directory_listing>")
        assert read_file_entry in result
        assert symlink_entry in result
        assert sec_directory_entry in result

    def test_list_empty_directory(self, tmp_path: Path) -> None:
        """List an empty directory"""
        target: Path = tmp_path.joinpath("empty_dir")
        target.mkdir()
        result: str = list_directory(str(target))
        assert (
            result == f'<directory_listing path="{str(target)}">\n<error>No entries found</error>\n</directory_listing>'
        )

    def test_directory_not_found(self) -> None:
        """DO not list a directory that does not exist"""
        result: str = list_directory("/nonexistent/path")
        assert (
            result
            == '<directory_listing path="/nonexistent/path">\n<error>Directory not found</error>\n</directory_listing>'
        )

    def test_path_is_not_a_directory(self, tmp_path: Path) -> None:
        """Do not list a path that is a file, not a directory"""
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text("content")
        result: str = list_directory(str(target))
        assert (
            result
            == f'<directory_listing path="{str(target)}">\n<error>Path is not a directory</error>\n</directory_listing>'
        )

    def test_permission_denied(self, tmp_path: Path) -> None:
        """Do not list a directory due to permission error"""
        target: Path = tmp_path.joinpath("secret")
        target.mkdir()
        with patch.object(Path, "iterdir", side_effect=PermissionError("Permission error")):
            result: str = list_directory(str(target))
            assert (
                result
                == f'<directory_listing path="{str(target)}">\n<error>Permission denied by the system</error>\n</directory_listing>'
            )

    def test_unknown_error(self, tmp_path: Path) -> None:
        """Do not list a directory due to an exception"""
        target: Path = tmp_path.joinpath("broken")
        target.mkdir()
        with patch.object(Path, "iterdir", side_effect=Exception("Exception")):
            result: str = list_directory(str(target))
            assert (
                result
                == f'<directory_listing path="{str(target)}">\n<error>Could not list directory</error>\n</directory_listing>'
            )

    def test_entry_classification_exception(self, tmp_path: Path) -> None:
        """List a directory containing an entry whose type cannot be determined"""
        target: Path = tmp_path.joinpath("mystery")
        target.mkdir()
        mystery_file: Path = target.joinpath("unknown")
        mystery_file.write_text("content")
        with patch.object(Path, "is_file", side_effect=Exception("Exception")):
            result: str = list_directory(str(target))
            mystery_file_entry: str = f"<entry>{mystery_file.name}</entry>"
            assert result == f'<directory_listing path="{str(target)}">\n{mystery_file_entry}\n</directory_listing>'

    def test_unknown_entry_type(self, tmp_path: Path) -> None:
        """List a directory containing an entry that is neither a file, a directory, nor a symlink"""
        target: Path = tmp_path.joinpath("mystery")
        target.mkdir()
        mystery_file: Path = target.joinpath("unknown")
        mystery_file.write_text("content")
        with (
            patch.object(Path, "is_symlink", return_value=False),
            patch.object(Path, "is_dir", return_value=False),
            patch.object(Path, "is_file", return_value=False),
        ):
            result: str = list_directory(str(target))
            mystery_file_entry: str = f"<entry>{mystery_file.name}</entry>"
            assert result == f'<directory_listing path="{str(target)}">\n{mystery_file_entry}\n</directory_listing>'
