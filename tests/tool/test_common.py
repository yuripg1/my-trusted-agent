from unittest.mock import MagicMock, patch

from tool.common import get_language_from_filename, make_safe_code_fence


class TestMakeSafeCodeFence:
    """Tests for the `make_safe_code_fence` function"""

    def test_basic_fence(self) -> None:
        """Create a basic code fence with no info string"""
        result: str = make_safe_code_fence("hello")
        assert result == "```\nhello\n```"

    def test_with_info_string(self) -> None:
        """Create a code fence with an info string"""
        result: str = make_safe_code_fence("print('hi')", "python")
        assert result == "```python\nprint('hi')\n```"

    def test_fence_with_backticks(self) -> None:
        """Create a code fence when content contains backticks"""
        result: str = make_safe_code_fence("`code`", "")
        assert result == "```\n`code`\n```"

    def test_fence_with_triple_backticks(self) -> None:
        """Create a code fence when content contains triple backticks"""
        result: str = make_safe_code_fence("```code```", "")
        assert result == "````\n```code```\n````"


class TestGetLanguageFromFilename:
    """Tests for the `get_language_from_filename` function"""

    def test_known_extension(self) -> None:
        """Get language for a known file extension"""
        assert get_language_from_filename("main.py") == "python"
        assert get_language_from_filename("style.css") == "css"
        assert get_language_from_filename("script.js") == "javascript"

    def test_unknown_extension(self) -> None:
        """Get language for an unknown file extension"""
        assert get_language_from_filename("file.xyz") == ""

    def test_path_with_directories(self) -> None:
        """Get language from a full path"""
        assert get_language_from_filename("/some/dir/main.py") == "python"

    def test_lexer_without_aliases(self) -> None:
        """Get language when lexer has no aliases"""
        mock_lexer: MagicMock = MagicMock()
        mock_lexer.aliases = []
        with patch("tool.common.get_lexer_for_filename", return_value=mock_lexer):
            assert get_language_from_filename("main.py") == ""
