from typing import Literal, Required, TypedDict

from ddgs import DDGS

from tool.common import BaseToolCall


class SearchWebArguments(TypedDict):
    query: Required[str]
    max_results_per_page: Required[int]
    results_page_number: Required[int]


class SearchWebToolCall(BaseToolCall):
    tool_name: Required[Literal["search_web"]]
    arguments: Required[SearchWebArguments]


_TIMEOUT: int = 300
_SAFESEARCH: str = "off"


def get_search_web_message(tool_call: SearchWebToolCall) -> str:
    return f"Searching the web for **{tool_call['arguments']['query']}** (**{tool_call['arguments']['max_results_per_page']}** results - page **{tool_call['arguments']['results_page_number']}**)"


def search_web(query: str, max_results_per_page: int, results_page_number: int) -> str:
    output_entries: list[str] = []
    output_entries.append(f"<query>{query}</query>")
    raw_search_results = []
    try:
        raw_search_results = list(
            DDGS(timeout=_TIMEOUT).text(
                query=query, safesearch=_SAFESEARCH, max_results=max_results_per_page, page=results_page_number
            )
        )
    except:
        pass
    if len(raw_search_results) == 0:
        output_entries.append("<error>No search results found</error>")
    else:
        for page_result_number, search_result_data in enumerate(raw_search_results, 1):
            search_result_number: int = ((results_page_number - 1) * max_results_per_page) + page_result_number
            output_entries.append(
                f'<search_result result_number="{search_result_number}">\n<title>{str(search_result_data["title"]).strip()}</title>\n<href>{str(search_result_data["href"]).strip()}</href>\n<snippet>\n{str(search_result_data["body"]).strip()}\n</snippet>\n</search_result>'
            )
    joined_output_entries: str = "\n".join(output_entries)
    return f'<web_search max_results_per_page="{max_results_per_page}" results_page_number="{results_page_number}">\n{joined_output_entries}\n</web_search>'
