from unittest.mock import MagicMock, patch

from tool.search_web import SearchWebToolCall, get_search_web_message, search_web


class TestGetSearchWebMessage:
    """Tests for the `get_search_web_message` function"""

    def test_format(self) -> None:
        """Format the message correctly"""

        tool_call: SearchWebToolCall = {
            "tool_name": "search_web",
            "arguments": {"query": "python", "max_results_per_page": 10, "results_page_number": 2},
        }
        assert get_search_web_message(tool_call) == "Searching the web for **python** (**10** results - page **2**)"


class TestSearchWeb:
    """Tests for the `search_web` tool"""

    def test_search_with_results(self) -> None:
        """Search the web and get results on page 2"""
        query: str = "test query"
        max_results_per_page: int = 10
        results_page_number: int = 2
        mock_results: list[dict[str, str]] = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "First result"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Second result"},
        ]
        search_result_entries: list[str] = []
        for entry_number, search_result in enumerate(mock_results, 1):
            search_result_number: int = ((results_page_number - 1) * max_results_per_page) + entry_number
            search_result_entry: str = f'<search_result result_number="{search_result_number}">\n<title>{search_result["title"]}</title>\n<href>{search_result["href"]}</href>\n<snippet>\n{search_result["body"]}\n</snippet>\n</search_result>'
            search_result_entries.append(search_result_entry)
        mock_ddgs_instance: MagicMock = MagicMock()
        mock_ddgs_instance.text.return_value = mock_results
        with patch("tool.search_web.DDGS", return_value=mock_ddgs_instance):
            result: str = search_web(query, max_results_per_page, results_page_number)
        expected_result: str = f'<web_search max_results_per_page="{max_results_per_page}" results_page_number="{results_page_number}">\n<query>{query}</query>\n{"\n".join(search_result_entries)}\n</web_search>'
        assert result == expected_result

    def test_no_results(self) -> None:
        """Search the web and get no results"""
        query: str = "test query"
        max_results_per_page: int = 10
        results_page_number: int = 1
        mock_ddgs_instance: MagicMock = MagicMock()
        mock_ddgs_instance.text.return_value = []
        with patch("tool.search_web.DDGS", return_value=mock_ddgs_instance):
            result: str = search_web(query, max_results_per_page, results_page_number)
        expected_result: str = f'<web_search max_results_per_page="{max_results_per_page}" results_page_number="{results_page_number}">\n<query>{query}</query>\n<error>No search results found</error>\n</web_search>'
        assert result == expected_result

    def test_search_exception(self) -> None:
        """Search the web and get an exception"""
        query: str = "test query"
        max_results_per_page: int = 10
        results_page_number: int = 1
        mock_ddgs_instance: MagicMock = MagicMock()
        mock_ddgs_instance.text.side_effect = Exception("Exception")
        with patch("tool.search_web.DDGS", return_value=mock_ddgs_instance):
            result: str = search_web(query, max_results_per_page, results_page_number)
        expected_result: str = f'<web_search max_results_per_page="{max_results_per_page}" results_page_number="{results_page_number}">\n<query>{query}</query>\n<error>No search results found</error>\n</web_search>'
        assert result == expected_result
