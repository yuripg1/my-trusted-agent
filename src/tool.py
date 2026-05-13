from ddgs import DDGS
from io import BytesIO
from pathlib import Path
from primp import Client, Response
from pypdf import PdfReader
from random import randint
from subprocess import CompletedProcess, run
from trafilatura import extract
from typing import Any, Literal, Mapping, NotRequired, Required, TypeAlias, TypedDict


class CreateDirectoryArguments(TypedDict):
    path: Required[str]


class DeleteFileOrDirectoryArguments(TypedDict):
    type: Required[str]
    path: Required[str]


class EditFileArguments(TypedDict):
    path: Required[str]
    search_for: Required[str]
    replace_with: Required[str]
    max_substitutions: Required[int]


class ExecuteShellCommandArguments(TypedDict):
    command: Required[str]


class GetRandomIntegerArguments(TypedDict):
    min: Required[int]
    max: Required[int]


class ListDirectoryArguments(TypedDict):
    path: Required[str]


class ReadFileArguments(TypedDict):
    path: Required[str]


class ReadPdfDocumentArguments(TypedDict):
    location_type: Required[str]
    location: Required[str]


class ReadWebPageArguments(TypedDict):
    url: Required[str]


class SearchWebArguments(TypedDict):
    query: Required[str]
    max_results_per_page: Required[int]
    results_page_number: Required[int]


class WriteFileArguments(TypedDict):
    path: Required[str]
    content: Required[str]


class BaseToolCall(TypedDict):
    id: NotRequired[str]


class CreateDirectoryToolCall(BaseToolCall):
    tool_name: Required[Literal["create_directory"]]
    arguments: Required[CreateDirectoryArguments]


class DeleteFileOrDirectoryToolCall(BaseToolCall):
    tool_name: Required[Literal["delete_file_or_directory"]]
    arguments: Required[DeleteFileOrDirectoryArguments]


class EditFileToolCall(BaseToolCall):
    tool_name: Required[Literal["edit_file"]]
    arguments: Required[EditFileArguments]


class ExecuteShellCommandToolCall(BaseToolCall):
    tool_name: Required[Literal["execute_shell_command"]]
    arguments: Required[ExecuteShellCommandArguments]


class GetRandomIntegerToolCall(BaseToolCall):
    tool_name: Required[Literal["generate_random_integer"]]
    arguments: Required[GetRandomIntegerArguments]


class ListDirectoryToolCall(BaseToolCall):
    tool_name: Required[Literal["list_directory"]]
    arguments: Required[ListDirectoryArguments]


class ReadFileToolCall(BaseToolCall):
    tool_name: Required[Literal["read_file"]]
    arguments: Required[ReadFileArguments]


class ReadPdfDocumentToolCall(BaseToolCall):
    tool_name: Required[Literal["read_pdf_document"]]
    arguments: Required[ReadPdfDocumentArguments]


class ReadWebPageToolCall(BaseToolCall):
    tool_name: Required[Literal["read_web_page"]]
    arguments: Required[ReadWebPageArguments]


class SearchWebToolCall(BaseToolCall):
    tool_name: Required[Literal["search_web"]]
    arguments: Required[SearchWebArguments]


class WriteFileToolCall(BaseToolCall):
    tool_name: Required[Literal["write_file"]]
    arguments: Required[WriteFileArguments]


ToolCall: TypeAlias = (
    CreateDirectoryToolCall
    | DeleteFileOrDirectoryToolCall
    | EditFileToolCall
    | ExecuteShellCommandToolCall
    | GetRandomIntegerToolCall
    | ListDirectoryToolCall
    | ReadFileToolCall
    | ReadPdfDocumentToolCall
    | ReadWebPageToolCall
    | SearchWebToolCall
    | WriteFileToolCall
)

PDF_DOCUMENT_REQUEST_TIMEOUT: int = 300
WEB_PAGE_REQUEST_TIMEOUT: int = 300
WEB_SEARCH_TIMEOUT: int = 300
WEB_SEARCH_SAFESEARCH: str = "off"


def get_individual_tool_call_message(tool_call: ToolCall) -> str:
    tool_name: str = ""
    try:
        tool_name = tool_call["tool_name"]
        if tool_call["tool_name"] == "create_directory":
            return f"Creating directory at **{tool_call["arguments"]["path"]}**"
        elif tool_call["tool_name"] == "delete_file_or_directory":
            return f"Deleting **{tool_call["arguments"]["path"]}** (**{tool_call["arguments"]["type"]}**)"
        elif tool_call["tool_name"] == "edit_file":
            return f"Editing file at **{tool_call["arguments"]["path"]}** (**{tool_call["arguments"]["max_substitutions"]}** substitutions)\n\n**Searching for**:\n```\n{tool_call["arguments"]["search_for"]}\n```\n\n**Replacing with**:\n```\n{tool_call["arguments"]["replace_with"]}\n```"
        elif tool_call["tool_name"] == "execute_shell_command":
            return f"```shell\n$ {tool_call["arguments"]["command"]}\n```"
        elif tool_call["tool_name"] == "generate_random_integer":
            return f"Generating a random integer between **{tool_call["arguments"]["min"]}** and **{tool_call["arguments"]["max"]}**"
        elif tool_call["tool_name"] == "list_directory":
            return f"Listing directory at **{tool_call["arguments"]["path"]}**"
        elif tool_call["tool_name"] == "read_file":
            return f"Reading file at **{tool_call["arguments"]["path"]}**"
        elif tool_call["tool_name"] == "read_pdf_document":
            return f"Reading PDF document at **{tool_call["arguments"]["location"]}** (**{tool_call["arguments"]["location_type"]}**)"
        elif tool_call["tool_name"] == "read_web_page":
            return f"Reading web site at **{tool_call["arguments"]["url"]}**"
        elif tool_call["tool_name"] == "search_web":
            return f"Searching the web for **{tool_call["arguments"]["query"]}** (**{tool_call["arguments"]["max_results_per_page"]}** results - page **{tool_call["arguments"]["results_page_number"]}**)"
        elif tool_call["tool_name"] == "write_file":
            return (
                f"Writing file at **{tool_call["arguments"]["path"]}**\n\n```\n{tool_call["arguments"]["content"]}\n```"
            )
    except:
        pass
    if len(tool_name) != 0:
        return f'Error on "{tool_name}"'
    return "Error"


def get_group_tool_call_messages(tool_calls: list[ToolCall]) -> list[str]:
    messages: list[str] = []
    for tool_call in tool_calls:
        messages.append(get_individual_tool_call_message(tool_call))
    return messages


def get_individual_tool_call_permission(tool_call: ToolCall) -> bool:
    if tool_call["tool_name"] in [
        "create_directory",
        "generate_random_integer",
        "list_directory",
        "read_web_page",
        "search_web",
    ]:
        return True
    elif tool_call["tool_name"] == "read_pdf_document":
        return tool_call["arguments"]["location_type"] == "web"
    else:
        return False


def get_group_tool_call_permission(tool_calls: list[ToolCall]) -> bool:
    for tool_call in tool_calls:
        if not get_individual_tool_call_permission(tool_call):
            return False
    return True


def create_directory(path: str) -> str:
    output_entries: list[str] = []
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        output_entries.append("<result>Directory created successfully</result>")
    except PermissionError:
        output_entries.append("<error>Permission denied by the system</error>")
    except:
        output_entries.append("<error>Could not create directory</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<directory_creation path="{path}">\n{joined_output_entries}\n</directory_creation>'


def delete_file_or_directory(type: str, path: str, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>File or directory deletion manually denied by the user</error>")
    else:
        try:
            path_obj: Path = Path(path)
            if not path_obj.exists():
                output_entries.append("<error>File or directory not found</error>")
            elif type == "file":
                if path_obj.is_dir():
                    output_entries.append("<error>Expected a file but found a directory</error>")
                else:
                    path_obj.unlink()
                    output_entries.append("<result>File deleted successfully</result>")
            elif type == "directory":
                if path_obj.is_file():
                    output_entries.append("<error>Expected a directory but found a file</error>")
                else:
                    path_obj.rmdir()
                    output_entries.append("<result>Directory deleted successfully</result>")
            else:
                output_entries.append(f'<error>Invalid type "{type}"</error>')
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except:
            output_entries.append("<error>Could not delete file or directory</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<file_or_directory_deletion type="{type}" path="{path}">\n{joined_output_entries}\n</file_or_directory_deletion>'


def edit_file(
    path: str, search_for: str, replace_with: str, max_substitutions: int, tool_call_permission: bool = True
) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>File editing manually denied by the user</error>")
    else:
        if max_substitutions < 1:
            output_entries.append("<error>max_substitutions must be greater than or equal to 1</error>")
        elif len(search_for) == 0:
            output_entries.append("<error>search_for cannot be empty</error>")
        else:
            try:
                with open(path, "r") as file:
                    file_content: str = file.read()
                number_of_occurrences: int = file_content.count(search_for)
                output_entries.append(f"<number_of_occurrences>{number_of_occurrences}</number_of_occurrences>")
                if number_of_occurrences == 0:
                    output_entries.append(
                        "<error>No occurrences of the searched text were found. No substitutions were made.</error>"
                    )
                elif number_of_occurrences > max_substitutions:
                    output_entries.append(
                        f"<error>Found {number_of_occurrences} occurrences of the searched text, but max_substitutions was set to {max_substitutions}. Please be more specific and include more context in the search text to narrow down the match.</error>"
                    )
                else:
                    new_content: str = file_content.replace(search_for, replace_with, max_substitutions)
                    with open(path, "w") as file:
                        file.write(new_content)
                    output_entries.append("<result>File edited successfully</result>")
            except FileNotFoundError:
                output_entries.append("<error>File not found</error>")
            except PermissionError:
                output_entries.append("<error>Permission denied by the system</error>")
            except:
                output_entries.append("<error>Could not edit file</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<file_edit path="{path}" max_substitutions="{max_substitutions}">\n{joined_output_entries}\n</file_edit>'


def execute_shell_command(command: str, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    output_entries.append(f"<command>\n{command.strip()}\n</command>")
    if not tool_call_permission:
        output_entries.append("<error>Bash command execution manually denied by the user</error>")
    else:
        command_execution_result: CompletedProcess[str] = run(command, shell=True, capture_output=True, text=True)
        trimmed_stdout = command_execution_result.stdout.strip()
        if len(trimmed_stdout) != 0:
            output_entries.append(f"<stdout>\n{trimmed_stdout}\n</stdout>")
        trimmed_stderr = command_execution_result.stderr.strip()
        if len(trimmed_stderr) != 0:
            output_entries.append(f"<stderr>\n{trimmed_stderr}\n</stderr>")
        output_entries.append(f"<exit_code>{command_execution_result.returncode}</exit_code>")
    joined_output_entries: str = "\n".join(output_entries)
    return f"<shell_command_execution>\n{joined_output_entries}\n</shell_command_execution>"


def generate_random_integer(min: int, max: int) -> str:
    output_entries: list[str] = []
    if max < min:
        output_entries.append('<error>"max" is less than "min"</error>')
    else:
        random_integer: int = randint(min, max)
        output_entries.append(f"<result>{random_integer}</result>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<random_integer_generation min="{min}" max="{max}">\n{joined_output_entries}\n</random_integer_generation>'


def list_directory(path: str) -> str:
    output_entries: list[str] = []
    try:
        directory_path: Path = Path(path)
        for directory_entry in directory_path.iterdir():
            directory_entry_type: str = ""
            try:
                if directory_entry.is_symlink():
                    directory_entry_type = "symlink"
                elif directory_entry.is_dir():
                    directory_entry_type = "directory"
                elif directory_entry.is_file():
                    directory_entry_type = "file"
            except:
                pass
            if len(directory_entry_type) != 0:
                output_entries.append(f'<entry type="{directory_entry_type}">{directory_entry.name}</entry>')
            else:
                output_entries.append(f"<entry>{directory_entry.name}</entry>")
    except FileNotFoundError:
        output_entries.append("<error>Directory not found</error>")
    except NotADirectoryError:
        output_entries.append("<error>Path is not a directory</error>")
    except PermissionError:
        output_entries.append("<error>Permission denied by the system</error>")
    except:
        output_entries.append("<error>Could not list directory</error>")
    if len(output_entries) == 0:
        output_entries.append("<error>No entries found</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<directory_listing path="{path}">\n{joined_output_entries}\n</directory_listing>'


def read_file(path: str, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>File reading manually denied by the user</error>")
    else:
        try:
            with open(path, "r") as file:
                file_content = file.read()
                output_entries.append(f"<content>\n{file_content}\n</content>")
        except FileNotFoundError:
            output_entries.append("<error>File not found</error>")
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except:
            output_entries.append("<error>Could not read file</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<file_read path="{path}">\n{joined_output_entries}\n</file_read>'


def read_pdf_document(location_type: str, location: str, tool_call_permission: bool = True, note: str = "") -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>PDF document reading manually denied by the user</error>")
    else:
        errored: bool = False
        status_code: int | None = None
        content_type: str = ""
        raw_pdf_document_content: Any = None
        try:
            if location_type == "web" or location.startswith(("http", "https")):
                client: Client = Client(
                    impersonate="random", impersonate_os="random", timeout=PDF_DOCUMENT_REQUEST_TIMEOUT
                )
                response: Response = client.get(location)
                status_code = response.status_code
                if status_code < 200 or status_code > 299:
                    output_entries.append("<error>Could not fetch the PDF document</error>")
                    errored = True
                else:
                    raw_pdf_document_content = response.content
                    content_type = response.headers.get("content-type", "").strip()
            elif location_type == "local":
                with open(location, "rb") as pdf_document_file:
                    raw_pdf_document_content = pdf_document_file.read()
            else:
                output_entries.append(f'<error>Invalid location_type "{location_type}"</error>')
                errored = True
        except:
            output_entries.append("<error>Could not fetch the PDF document</error>")
            errored = True
        if not errored:
            try:
                if raw_pdf_document_content[:4] != b"%PDF":
                    output_entries.append("<error>The fetched file does not seem to be a valid PDF document</error>")
                    errored = True
                else:
                    pdf_document_reader = PdfReader(BytesIO(raw_pdf_document_content))
                    output_pages_entries: list[str] = []
                    for page_number, raw_pdf_document_page in enumerate(pdf_document_reader.pages, 1):
                        pdf_document_page_text = raw_pdf_document_page.extract_text().strip()
                        if len(pdf_document_page_text) != 0:
                            output_pages_entries.append(
                                f'<page number="{page_number}">\n{pdf_document_page_text}\n</page>'
                            )
                    if len(output_pages_entries) != 0:
                        joined_output_pages_entries = "\n".join(output_pages_entries)
                        output_entries.append(f"<pages>\n{joined_output_pages_entries}\n</pages>")
            except:
                output_entries.append("<error>Could not read the PDF document</error>")
                errored = True
        if len(output_entries) == 0:
            output_entries.append("<error>Could not read the PDF document</error>")
            errored = True
        if errored:
            if status_code is not None:
                output_entries.append(f"<status_code>{status_code}</status_code>")
            if len(content_type) != 0:
                output_entries.append(f"<content_type>{content_type}</content_type>")
        if len(note) != 0:
            output_entries.insert(0, f"<note>{note}</note>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<pdf_document_read location_type="{location_type}" location="{location}">\n{joined_output_entries}\n</pdf_document_read>'


def read_web_page(url: str) -> str:
    output_entries: list[str] = []
    errored: bool = False
    status_code: int | None = None
    content_type: str = ""
    raw_web_page_content: str = ""
    try:
        client = Client(impersonate="random", impersonate_os="random", timeout=WEB_PAGE_REQUEST_TIMEOUT)
        response: Response = client.get(url)
        status_code = response.status_code
        if status_code < 200 or status_code > 299:
            output_entries.append("<error>Could not fetch the web page</error>")
            errored = True
        else:
            raw_web_page_content = response.text
            content_type = response.headers.get("content-type", "").strip()
    except:
        output_entries.append("<error>Could not fetch the web page</error>")
        errored = True
    if "application/pdf" in content_type.lower():
        return read_pdf_document("web", url, note='Redirected from "read_web_page"')
    if not errored:
        try:
            extracted_content: str | None = extract(raw_web_page_content, output_format="markdown", include_links=True)
            if extracted_content is not None:
                trimmed_extracted_content: str = extracted_content.strip()
                if len(trimmed_extracted_content) != 0:
                    output_entries.append(f"<content>\n{trimmed_extracted_content}\n</content>")
        except:
            output_entries.append("<error>Could not read the web page</error>")
            errored = True
    if len(output_entries) == 0:
        output_entries.append("<error>Could not read the web page</error>")
        errored = True
    if errored:
        if status_code is not None:
            output_entries.append(f"<status_code>{status_code}</status_code>")
        if len(content_type) != 0:
            output_entries.append(f"<content_type>{content_type}</content_type>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<web_page_read url="{url}">\n{joined_output_entries}\n</web_page_read>'


def search_web(query: str, max_results_per_page: int, results_page_number: int) -> str:
    output_entries: list[str] = []
    output_entries.append(f"<query>{query}</query>")
    raw_search_results = []
    try:
        raw_search_results = list(
            DDGS(timeout=WEB_SEARCH_TIMEOUT).text(
                query=query,
                safesearch=WEB_SEARCH_SAFESEARCH,
                max_results=max_results_per_page,
                page=results_page_number,
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


def write_file(path: str, content: str, tool_call_permission: bool = True) -> str:
    output_entries: list[str] = []
    if not tool_call_permission:
        output_entries.append("<error>File writing manually denied by the user</error>")
    else:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as file:
                file.write(content)
            output_entries.append("<result>File written successfully</result>")
        except PermissionError:
            output_entries.append("<error>Permission denied by the system</error>")
        except:
            output_entries.append("<error>Could not write file</error>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<file_write path="{path}">\n{joined_output_entries}\n</file_write>'


def execute_tool_call(tool_call: ToolCall, tool_call_permission: bool) -> str:
    tool_name: str = ""
    try:
        tool_name = tool_call["tool_name"]
        if tool_call["tool_name"] == "create_directory":
            create_directory_path: str = tool_call["arguments"]["path"]
            return create_directory(create_directory_path)
        elif tool_call["tool_name"] == "delete_file_or_directory":
            delete_type: str = tool_call["arguments"]["type"]
            delete_path: str = tool_call["arguments"]["path"]
            return delete_file_or_directory(delete_type, delete_path, tool_call_permission)
        elif tool_call["tool_name"] == "edit_file":
            path: str = tool_call["arguments"]["path"]
            search_for: str = tool_call["arguments"]["search_for"]
            replace_with: str = tool_call["arguments"]["replace_with"]
            max_substitutions: int = tool_call["arguments"]["max_substitutions"]
            return edit_file(path, search_for, replace_with, max_substitutions, tool_call_permission)
        elif tool_call["tool_name"] == "execute_shell_command":
            command: str = tool_call["arguments"]["command"]
            return execute_shell_command(command, tool_call_permission)
        elif tool_call["tool_name"] == "generate_random_integer":
            min: int = tool_call["arguments"]["min"]
            max: int = tool_call["arguments"]["max"]
            return generate_random_integer(min, max)
        elif tool_call["tool_name"] == "list_directory":
            directory_path: str = tool_call["arguments"]["path"]
            return list_directory(directory_path)
        elif tool_call["tool_name"] == "read_file":
            file_path: str = tool_call["arguments"]["path"]
            return read_file(file_path, tool_call_permission)
        elif tool_call["tool_name"] == "read_pdf_document":
            location_type: str = tool_call["arguments"]["location_type"]
            location: str = tool_call["arguments"]["location"]
            return read_pdf_document(location_type, location, tool_call_permission=tool_call_permission)
        elif tool_call["tool_name"] == "read_web_page":
            url: str = tool_call["arguments"]["url"]
            return read_web_page(url)
        elif tool_call["tool_name"] == "search_web":
            query: str = tool_call["arguments"]["query"]
            max_results_per_page: int = tool_call["arguments"]["max_results_per_page"]
            results_page_number: int = tool_call["arguments"]["results_page_number"]
            return search_web(query, max_results_per_page, results_page_number)
        elif tool_call["tool_name"] == "write_file":
            write_file_path: str = tool_call["arguments"]["path"]
            write_file_content: str = tool_call["arguments"]["content"]
            return write_file(write_file_path, write_file_content, tool_call_permission)
    except:
        pass
    if len(tool_name) != 0:
        return f'Error on "{tool_name}"'
    return "Error"
