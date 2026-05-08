from ddgs import DDGS
from ddgs.exceptions import DDGSException
from io import BytesIO
from pypdf import PdfReader
from random import randint
from requests import get, Response
from subprocess import run
from trafilatura import extract, fetch_url
from typing import Any, Literal, Mapping, NotRequired, Required, TypedDict

FunctionNameType = Literal["run_bash_command", "get_random_integer", "search_web", "read_pdf_document", "read_web_page"]

SEARCH_TIMEOUT: int = 60
SEARCH_SAFESEARCH: str = "off"
DOCUMENT_REQUEST_TIMEOUT: int = 60


class ToolCallArguments(TypedDict):
    command: NotRequired[str]
    min: NotRequired[int]
    max: NotRequired[int]
    query: NotRequired[str]
    max_results_per_page: NotRequired[int]
    page_number: NotRequired[int]
    url: NotRequired[str]
    source_type: NotRequired[str]
    source: NotRequired[str]


class ToolCall(TypedDict):
    id: NotRequired[str]
    function_name: NotRequired[FunctionNameType]
    arguments: Required[ToolCallArguments]


def get_tool_call_message(tool_call: ToolCall) -> str:
    if tool_call["function_name"] == "run_bash_command":
        return f"$ {tool_call["arguments"]["command"]}"
    elif tool_call["function_name"] == "get_random_integer":
        return f'Generating a random integer between "{tool_call["arguments"]["min"]}" and "{tool_call["arguments"]["max"]}"'
    elif tool_call["function_name"] == "search_web":
        return f'Searching the web for "{tool_call["arguments"]["query"]}" ({tool_call["arguments"]["max_results_per_page"]} results - page {tool_call["arguments"]["page_number"]})'
    elif tool_call["function_name"] == "read_pdf_document":
        return (
            f'Reading PDF document from "{tool_call["arguments"]["source"]}" ({tool_call["arguments"]["source_type"]})'
        )
    elif tool_call["function_name"] == "read_web_page":
        return f'Fetching content from "{tool_call["arguments"]["url"]}"'
    return ""


def get_default_tool_call_permission(tool_call: ToolCall) -> bool:
    if tool_call["function_name"] in ["get_random_integer", "search_web", "read_web_page"]:
        return True
    elif tool_call["function_name"] == "read_pdf_document":
        return tool_call["arguments"]["source_type"] == "remote"
    else:
        return False


def execute_bash_command(permission_granted: bool, command: str) -> tuple[str, str, int]:
    if not permission_granted:
        return "", "", 0
    result = run(command, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def get_formatted_bash_command_output(
    command: str, permission_granted: bool, stdout: str, stderr: str, returncode: int
) -> str:
    output_entries: list[str] = []
    output_entries.append(f"<command>\n{command.strip()}\n</command>")
    if not permission_granted:
        output_entries.append("<error>Bash command execution manually denied by the user</error>")
    else:
        trimmed_stdout = stdout.strip()
        if len(trimmed_stdout) != 0:
            output_entries.append(f"<stdout>\n{trimmed_stdout}\n</stdout>")
        trimmed_stderr = stderr.strip()
        if len(trimmed_stderr) != 0:
            output_entries.append(f"<stderr>\n{trimmed_stderr}\n</stderr>")
        output_entries.append(f"<returncode>{returncode}</returncode>")
    joined_output_entries: str = "\n".join(output_entries)
    return f"<bash_command_execution>\n{joined_output_entries}\n</bash_command_execution>"


def get_random_integer(min: int, max: int) -> str:
    random_integer: int = randint(min, max)
    return f'<random_integer min="{min}" max="{max}">{random_integer}</random_integer>'


def search_web(query: str, max_results_per_page: int, page_number: int) -> str:
    output_entries: list[str] = []
    raw_search_results = []
    try:
        raw_search_results = list(
            DDGS(timeout=SEARCH_TIMEOUT).text(
                query=query, safesearch=SEARCH_SAFESEARCH, max_results=max_results_per_page, page=page_number
            )
        )
    except DDGSException:
        pass
    if len(raw_search_results) == 0:
        output_entries.append("<error>No search results found</error>")
    else:
        for page_result_number, search_result_data in enumerate(raw_search_results, 1):
            search_result_number: int = ((page_number - 1) * max_results_per_page) + page_result_number
            output_entries.append(
                f'<search_result result_number="{search_result_number}">\n<title>{str(search_result_data["title"]).strip()}</title>\n<href>{str(search_result_data["href"]).strip()}</href>\n<body>\n{str(search_result_data["body"]).strip()}\n</body>\n</search_result>'
            )
    joined_output_entries: str = "\n".join(output_entries)
    return f'<web_search query="{query}" max_results_per_page="{max_results_per_page}" page_number="{page_number}">\n{joined_output_entries}\n</web_search>'


def read_pdf_document(source_type: str, source: str) -> str:
    output_entries: list[str] = []
    errored: bool = False
    raw_pdf_content: Any = None
    content_type: str = ""
    try:
        if source_type == "local":
            with open(source, "rb") as pdf_file:
                raw_pdf_content = pdf_file.read()
        elif source_type == "remote":
            response: Response = get(source, timeout=DOCUMENT_REQUEST_TIMEOUT)
            raw_pdf_content = response.content
            response_content_type: str | None = response.headers.get("Content-Type")
            if response_content_type is not None:
                trimmed_response_content_type = response_content_type.strip()
                if len(trimmed_response_content_type) != 0:
                    content_type = trimmed_response_content_type
    except:
        output_entries.append("<error>Could not fetch the PDF document</error>")
        errored = True
    if not errored:
        try:
            if raw_pdf_content[:4] != b"%PDF":
                if len(content_type) != 0:
                    output_entries.append(f"<content_type>{content_type}</content_type>")
                output_entries.append("<error>The fetched file does not seem to be a valid PDF document</error>")
            else:
                pdf_document_reader = PdfReader(BytesIO(raw_pdf_content))
                output_pages_entries: list[str] = []
                for page_number, raw_pdf_document_page in enumerate(pdf_document_reader.pages, 1):
                    pdf_document_page_text = raw_pdf_document_page.extract_text().strip()
                    if len(pdf_document_page_text) != 0:
                        output_pages_entries.append(f'<page number="{page_number}">\n{pdf_document_page_text}\n</page>')
                if len(output_pages_entries) != 0:
                    joined_output_pages_entries = "\n".join(output_pages_entries)
                    output_entries.append(f"<pages>\n{joined_output_pages_entries}\n</pages>")
        except:
            if len(content_type) != 0:
                output_entries.append(f"<content_type>{content_type}</content_type>")
            output_entries.append("<error>Could not read the PDF document</error>")
            errored = True
    if len(output_entries) == 0:
        output_entries.append("<error>Could not read the PDF document</error>")
    joined_output_entries = "\n".join(output_entries)
    return f'<pdf_document source_type="{source_type}" source="{source}">\n{joined_output_entries}\n</pdf_document>'


def read_web_page(url: str) -> str:
    output_entries: list[str] = []
    errored: bool = False
    try:
        raw_content: str | None = fetch_url(url)
    except:
        output_entries.append("<error>Could not read the web page</error>")
        errored = True
    if not errored:
        try:
            if raw_content is not None:
                extracted_content = extract(raw_content, output_format="markdown", with_metadata=False)
                if extracted_content is not None:
                    trimmed_extracted_content = extracted_content.strip()
                    if len(trimmed_extracted_content) != 0:
                        output_entries.append(f"<content>\n{trimmed_extracted_content}\n</content>")
        except:
            output_entries.append("<error>Could not read the web page</error>")
            errored = True
    joined_output_entries = "\n".join(output_entries)
    return f'<web_page url="{url}">\n{joined_output_entries}\n</web_page>'


def execute_tool_call(tool_call: ToolCall, tool_call_permission: bool) -> str:
    if tool_call["function_name"] == "run_bash_command":
        command: str = tool_call["arguments"]["command"]
        stdout, stderr, returncode = execute_bash_command(tool_call_permission, command)
        return get_formatted_bash_command_output(command, tool_call_permission, stdout, stderr, returncode)
    elif tool_call["function_name"] == "get_random_integer":
        min: int = tool_call["arguments"]["min"]
        max: int = tool_call["arguments"]["max"]
        return get_random_integer(min, max)
    elif tool_call["function_name"] == "search_web":
        query: str = tool_call["arguments"]["query"]
        max_results_per_page: int = tool_call["arguments"]["max_results_per_page"]
        page_number: int = tool_call["arguments"]["page_number"]
        return search_web(query, max_results_per_page, page_number)
    elif tool_call["function_name"] == "read_web_page":
        url: str = tool_call["arguments"]["url"]
        return read_web_page(url)
    elif tool_call["function_name"] == "read_pdf_document":
        source_type: str = tool_call["arguments"]["source_type"]
        source: str = tool_call["arguments"]["source"]
        return read_pdf_document(source_type, source)
    return ""
