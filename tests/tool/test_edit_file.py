from pathlib import Path
from random import randint
from unittest.mock import patch

from tool.edit_file import EditFileToolCall, edit_file, get_edit_file_message


class TestGetEditFileMessage:
    """Tests for the `get_edit_file_message` function"""

    def test_format(self) -> None:
        """Format the message correctly"""
        tool_call: EditFileToolCall = {
            "tool_name": "edit_file",
            "arguments": {
                "path": "/path/file.txt",
                "search_for": "hello",
                "replace_with": "goodbye",
                "number_of_substitutions": 1,
            },
        }
        result: str = get_edit_file_message(tool_call)
        assert result.startswith("Editing file at **/path/file.txt** (**1** substitutions)\n\n```diff")
        assert result.endswith("```")


class TestEditFile:
    """Tests for the `edit_file` tool"""

    def test_successful_substitution(self, tmp_path: Path) -> None:
        """Edit a file with a correct number of substitutions"""
        number_of_occurrences: int = randint(2, 8)
        searched_for_text: str = "hello"
        original_file_content: str = f"pre_content {f' {searched_for_text} ' * number_of_occurrences} post_content"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text(original_file_content)
        replaced_with_text: str = "goodbye"
        number_of_substitutions: int = number_of_occurrences
        result: str = edit_file(str(target), searched_for_text, replaced_with_text, number_of_substitutions)
        assert (
            result
            == f'<file_edit path="{str(target)}" number_of_substitutions="{number_of_substitutions}">\n<result>File edited successfully</result>\n<number_of_occurrences>{number_of_occurrences}</number_of_occurrences>\n</file_edit>'
        )
        modified_file_content: str = f"pre_content {f' {replaced_with_text} ' * number_of_occurrences} post_content"
        assert target.read_text() == modified_file_content

    def test_search_text_not_found(self, tmp_path: Path) -> None:
        """Do not edit a file due to no occurrences of the searched text"""
        original_file_content: str = "pre_content content post_content"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text(original_file_content)
        searched_for_text: str = "hello"
        replaced_with_text: str = "goodbye"
        number_of_substitutions: int = 1
        result: str = edit_file(str(target), searched_for_text, replaced_with_text, number_of_substitutions)
        number_of_occurrences: int = 0
        assert (
            result
            == f'<file_edit path="{str(target)}" number_of_substitutions="{number_of_substitutions}">\n<error>No occurrences of the searched text were found</error>\n<number_of_occurrences>{number_of_occurrences}</number_of_occurrences>\n</file_edit>'
        )
        assert target.read_text() == original_file_content

    def test_occurrence_mismatch(self, tmp_path: Path) -> None:
        """Do not edit a file due to the number of occurrences not matching expected substitutions"""
        number_of_occurrences: int = randint(2, 8)
        searched_for_text: str = "hello"
        original_file_content: str = f"pre_content {f' {searched_for_text} ' * number_of_occurrences} post_content"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text(original_file_content)
        replaced_with_text: str = "goodbye"
        number_of_substitutions: int = number_of_occurrences + 1
        result: str = edit_file(str(target), searched_for_text, replaced_with_text, number_of_substitutions)
        assert (
            result
            == f'<file_edit path="{str(target)}" number_of_substitutions="{number_of_substitutions}">\n<error>The number of occurrences of the searched text does not match the expected number of substitutions</error>\n<number_of_occurrences>{number_of_occurrences}</number_of_occurrences>\n</file_edit>'
        )
        assert target.read_text() == original_file_content

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Do not edit a file that does not exist"""
        target: Path = tmp_path.joinpath("nonexistent")
        searched_for_text: str = "hello"
        replaced_with_text: str = "goodbye"
        number_of_substitutions: int = 1
        result: str = edit_file(str(target), searched_for_text, replaced_with_text, number_of_substitutions)
        assert (
            result
            == f'<file_edit path="{str(target)}" number_of_substitutions="{number_of_substitutions}">\n<error>File not found</error>\n</file_edit>'
        )

    def test_permission_error(self, tmp_path: Path) -> None:
        """Do not edit a file due to permission error"""
        number_of_occurrences: int = randint(2, 8)
        searched_for_text: str = "hello"
        original_file_content: str = f"pre_content {f' {searched_for_text} ' * number_of_occurrences} post_content"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text(original_file_content)
        replaced_with_text: str = "goodbye"
        number_of_substitutions: int = number_of_occurrences
        with patch("builtins.open", side_effect=PermissionError("Permission error")):
            result: str = edit_file(str(target), searched_for_text, replaced_with_text, number_of_substitutions)
            assert (
                result
                == f'<file_edit path="{str(target)}" number_of_substitutions="{number_of_substitutions}">\n<error>Permission denied by the system</error>\n</file_edit>'
            )
            assert target.read_text() == original_file_content

    def test_unknown_error(self, tmp_path: Path) -> None:
        """Do not edit a file due to an exception"""
        number_of_occurrences: int = randint(2, 8)
        searched_for_text: str = "hello"
        original_file_content: str = f"pre_content {f' {searched_for_text} ' * number_of_occurrences} post_content"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text(original_file_content)
        replaced_with_text: str = "goodbye"
        number_of_substitutions: int = number_of_occurrences
        with patch("builtins.open", side_effect=Exception("Exception")):
            result: str = edit_file(str(target), searched_for_text, replaced_with_text, number_of_substitutions)
            assert (
                result
                == f'<file_edit path="{str(target)}" number_of_substitutions="{number_of_substitutions}">\n<error>Could not edit file</error>\n</file_edit>'
            )
            assert target.read_text() == original_file_content

    def test_editing_denied_by_user(self, tmp_path: Path) -> None:
        """Do not edit a file due to being denied by the user"""
        number_of_occurrences: int = randint(2, 8)
        searched_for_text: str = "hello"
        original_file_content: str = f"pre_content {f' {searched_for_text} ' * number_of_occurrences} post_content"
        target: Path = tmp_path.joinpath("file.txt")
        target.write_text(original_file_content)
        replaced_with_text: str = "goodbye"
        number_of_substitutions: int = number_of_occurrences
        result: str = edit_file(
            str(target), searched_for_text, replaced_with_text, number_of_substitutions, tool_call_permission=False
        )
        assert (
            result
            == f'<file_edit path="{str(target)}" number_of_substitutions="{number_of_substitutions}">\n<error>File editing manually denied by the user</error>\n</file_edit>'
        )
        assert target.read_text() == original_file_content
