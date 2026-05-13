from typing import TypeAlias

from tool.create_directory import CreateDirectoryToolCall, create_directory
from tool.delete_file_or_directory import DeleteFileOrDirectoryToolCall, delete_file_or_directory
from tool.edit_file import EditFileToolCall, edit_file
from tool.execute_shell_command import ExecuteShellCommandToolCall, execute_shell_command
from tool.generate_random_integer import GenerateRandomIntegerToolCall, generate_random_integer
from tool.list_directory import ListDirectoryToolCall, list_directory
from tool.read_file import ReadFileToolCall, read_file
from tool.read_pdf_document import ReadPdfDocumentToolCall, read_pdf_document
from tool.read_web_page import ReadWebPageToolCall, read_web_page
from tool.search_web import SearchWebToolCall, search_web
from tool.write_file import WriteFileToolCall, write_file

ToolCall: TypeAlias = (
    CreateDirectoryToolCall
    | DeleteFileOrDirectoryToolCall
    | EditFileToolCall
    | ExecuteShellCommandToolCall
    | GenerateRandomIntegerToolCall
    | ListDirectoryToolCall
    | ReadFileToolCall
    | ReadPdfDocumentToolCall
    | ReadWebPageToolCall
    | SearchWebToolCall
    | WriteFileToolCall
)


def make_safe_code_fence(content: str, info_string: str = "") -> str:
    max_backtick_run: int = 0
    current_backtick_run: int = 0
    for character in content:
        if character == "`":
            current_backtick_run += 1
            if current_backtick_run > max_backtick_run:
                max_backtick_run = current_backtick_run
        else:
            current_backtick_run = 0
    fence_length: int = 3
    if max_backtick_run >= 3:
        fence_length = max_backtick_run + 1
    fence: str = "`" * fence_length
    info: str = info_string.strip()
    if len(info) != 0:
        return f"{fence}{info}\n{content}\n{fence}"
    else:
        return f"{fence}\n{content}\n{fence}"


def get_individual_tool_call_message(tool_call: ToolCall) -> str:
    tool_name: str = ""
    try:
        tool_name = tool_call["tool_name"]
        if tool_call["tool_name"] == "create_directory":
            return f"Creating directory at **{tool_call["arguments"]["path"]}**"
        elif tool_call["tool_name"] == "delete_file_or_directory":
            return f"Deleting **{tool_call["arguments"]["path"]}** (**{tool_call["arguments"]["type"]}**)"
        elif tool_call["tool_name"] == "edit_file":
            search_for_content: str = make_safe_code_fence(tool_call["arguments"]["search_for"])
            replace_with_content: str = make_safe_code_fence(tool_call["arguments"]["replace_with"])
            return f"Editing file at **{tool_call["arguments"]["path"]}** (**{tool_call["arguments"]["number_of_substitutions"]}** substitutions)\n\n**Searching for**:\n{search_for_content}\n\n**Replacing with**:\n{replace_with_content}"
        elif tool_call["tool_name"] == "execute_shell_command":
            command_content: str = make_safe_code_fence(f"$ {tool_call["arguments"]["command"]}", "shell")
            return f"Executing shell command\n\n{command_content}"
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
            write_content: str = make_safe_code_fence(tool_call["arguments"]["content"])
            return f"Writing file at **{tool_call["arguments"]["path"]}** (**{tool_call["arguments"]["mode"]}** mode)\n\n{write_content}"
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
            number_of_substitutions: int = tool_call["arguments"]["number_of_substitutions"]
            return edit_file(path, search_for, replace_with, number_of_substitutions, tool_call_permission)
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
            write_file_mode: str = tool_call["arguments"]["mode"]
            write_file_content: str = tool_call["arguments"]["content"]
            return write_file(write_file_path, write_file_mode, write_file_content, tool_call_permission)
    except:
        pass
    if len(tool_name) != 0:
        return f'Error on "{tool_name}"'
    return "Error"
