from typing import TypeAlias

from tool.create_directory import (
    CreateDirectoryToolCall,
    create_directory,
    get_create_directory_message,
    get_create_directory_permission,
)
from tool.delete_path import DeletePathToolCall, delete_path, get_delete_path_message, get_delete_path_permission
from tool.edit_file import EditFileToolCall, edit_file, get_edit_file_message, get_edit_file_permission
from tool.execute_shell_command import (
    ExecuteShellCommandToolCall,
    execute_shell_command,
    get_execute_shell_command_message,
    get_execute_shell_command_permission,
)
from tool.generate_random_integer import (
    GenerateRandomIntegerToolCall,
    generate_random_integer,
    get_generate_random_integer_message,
    get_generate_random_integer_permission,
)
from tool.list_directory import (
    ListDirectoryToolCall,
    get_list_directory_message,
    get_list_directory_permission,
    list_directory,
)
from tool.move_path import MovePathToolCall, get_move_path_message, get_move_path_permission, move_path
from tool.read_file import ReadFileToolCall, get_read_file_message, get_read_file_permission, read_file
from tool.read_pdf_document import (
    ReadPdfDocumentToolCall,
    get_read_pdf_document_message,
    get_read_pdf_document_permission,
    read_pdf_document,
)
from tool.read_web_page import (
    ReadWebPageToolCall,
    get_read_web_page_message,
    get_read_web_page_permission,
    read_web_page,
)
from tool.search_web import SearchWebToolCall, get_search_web_message, get_search_web_permission, search_web
from tool.write_file import WriteFileToolCall, get_write_file_message, get_write_file_permission, write_file

ToolCall: TypeAlias = (
    CreateDirectoryToolCall
    | DeletePathToolCall
    | EditFileToolCall
    | ExecuteShellCommandToolCall
    | GenerateRandomIntegerToolCall
    | ListDirectoryToolCall
    | MovePathToolCall
    | ReadFileToolCall
    | ReadPdfDocumentToolCall
    | ReadWebPageToolCall
    | SearchWebToolCall
    | WriteFileToolCall
)


def get_individual_tool_call_message(tool_call: ToolCall) -> str:
    tool_name: str = ""
    try:
        tool_name = tool_call["tool_name"]
        if tool_call["tool_name"] == "create_directory":
            return get_create_directory_message(tool_call)
        elif tool_call["tool_name"] == "delete_path":
            return get_delete_path_message(tool_call)
        elif tool_call["tool_name"] == "edit_file":
            return get_edit_file_message(tool_call)
        elif tool_call["tool_name"] == "execute_shell_command":
            return get_execute_shell_command_message(tool_call)
        elif tool_call["tool_name"] == "generate_random_integer":
            return get_generate_random_integer_message(tool_call)
        elif tool_call["tool_name"] == "list_directory":
            return get_list_directory_message(tool_call)
        elif tool_call["tool_name"] == "move_path":
            return get_move_path_message(tool_call)
        elif tool_call["tool_name"] == "read_file":
            return get_read_file_message(tool_call)
        elif tool_call["tool_name"] == "read_pdf_document":
            return get_read_pdf_document_message(tool_call)
        elif tool_call["tool_name"] == "read_web_page":
            return get_read_web_page_message(tool_call)
        elif tool_call["tool_name"] == "search_web":
            return get_search_web_message(tool_call)
        elif tool_call["tool_name"] == "write_file":
            return get_write_file_message(tool_call)
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
    try:
        if tool_call["tool_name"] == "create_directory":
            return get_create_directory_permission(tool_call)
        elif tool_call["tool_name"] == "delete_path":
            return get_delete_path_permission(tool_call)
        elif tool_call["tool_name"] == "edit_file":
            return get_edit_file_permission(tool_call)
        elif tool_call["tool_name"] == "execute_shell_command":
            return get_execute_shell_command_permission(tool_call)
        elif tool_call["tool_name"] == "generate_random_integer":
            return get_generate_random_integer_permission(tool_call)
        elif tool_call["tool_name"] == "list_directory":
            return get_list_directory_permission(tool_call)
        elif tool_call["tool_name"] == "move_path":
            return get_move_path_permission(tool_call)
        elif tool_call["tool_name"] == "read_file":
            return get_read_file_permission(tool_call)
        elif tool_call["tool_name"] == "read_pdf_document":
            return get_read_pdf_document_permission(tool_call)
        elif tool_call["tool_name"] == "read_web_page":
            return get_read_web_page_permission(tool_call)
        elif tool_call["tool_name"] == "search_web":
            return get_search_web_permission(tool_call)
        elif tool_call["tool_name"] == "write_file":
            return get_write_file_permission(tool_call)
    except:
        pass
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
        elif tool_call["tool_name"] == "delete_path":
            type: str = tool_call["arguments"]["type"]
            delete_path_path: str = tool_call["arguments"]["path"]
            return delete_path(type, delete_path_path, tool_call_permission)
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
        elif tool_call["tool_name"] == "move_path":
            move_type: str = tool_call["arguments"]["type"]
            move_source: str = tool_call["arguments"]["source"]
            move_destination: str = tool_call["arguments"]["destination"]
            return move_path(move_type, move_source, move_destination, tool_call_permission)
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
