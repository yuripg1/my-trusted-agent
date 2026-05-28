from contextlib import suppress
from typing import TypeAlias

from tool.create_directory import (
    CreateDirectoryToolCall,
    create_directory,
    get_create_directory_message,
    get_create_directory_permission,
)
from tool.delete_path import DeletePathToolCall, delete_path, get_delete_path_message, get_delete_path_permission
from tool.edit_file import (
    EditFileToolCall,
    edit_file,
    get_edit_file_message,
    get_edit_file_permission,
    get_edit_file_read_path,
)
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
from tool.invalid import InvalidToolCall, get_invalid_message, get_invalid_permission, invalid
from tool.list_directory import (
    ListDirectoryToolCall,
    get_list_directory_message,
    get_list_directory_permission,
    list_directory,
)
from tool.move_path import (
    MovePathToolCall,
    get_move_path_message,
    get_move_path_permission,
    get_move_path_read_path,
    move_path,
)
from tool.read_file import (
    ReadFileToolCall,
    get_read_file_message,
    get_read_file_permission,
    get_read_file_read_path,
    read_file,
)
from tool.read_pdf_document import (
    ReadPdfDocumentToolCall,
    get_read_pdf_document_message,
    get_read_pdf_document_permission,
    get_read_pdf_document_read_path,
    read_pdf_document,
)
from tool.read_web_page import (
    ReadWebPageToolCall,
    get_read_web_page_message,
    get_read_web_page_permission,
    read_web_page,
)
from tool.search_web import SearchWebToolCall, get_search_web_message, get_search_web_permission, search_web
from tool.write_file import (
    WriteFileToolCall,
    get_write_file_message,
    get_write_file_permission,
    get_write_file_read_path,
    write_file,
)

ToolCall: TypeAlias = (
    CreateDirectoryToolCall
    | DeletePathToolCall
    | EditFileToolCall
    | ExecuteShellCommandToolCall
    | GenerateRandomIntegerToolCall
    | InvalidToolCall
    | ListDirectoryToolCall
    | MovePathToolCall
    | ReadFileToolCall
    | ReadPdfDocumentToolCall
    | ReadWebPageToolCall
    | SearchWebToolCall
    | WriteFileToolCall
)


def get_tool_system_instructions() -> list[str]:
    system_instructions: list[str] = [
        "You have access to tools",
        "When you issue many tool calls in one go, they are executed sequentially and in order",
        "Whenever possible, you should strongly prioritize issuing as many tool calls as possible in one go instead of issuing them one by one",
    ]
    return system_instructions


def get_individual_tool_call_message(tool_call: ToolCall) -> str:
    tool_name: str = ""
    with suppress(Exception):
        tool_name = tool_call["tool_name"]
        if tool_call["tool_name"] == "create_directory":
            return get_create_directory_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "delete_path":
            return get_delete_path_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "edit_file":
            return get_edit_file_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "execute_shell_command":
            return get_execute_shell_command_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "generate_random_integer":
            return get_generate_random_integer_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "invalid":
            return get_invalid_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "list_directory":
            return get_list_directory_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "move_path":
            return get_move_path_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "read_file":
            return get_read_file_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "read_pdf_document":
            return get_read_pdf_document_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "read_web_page":
            return get_read_web_page_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "search_web":
            return get_search_web_message(tool_call["arguments"])
        elif tool_call["tool_name"] == "write_file":
            return get_write_file_message(tool_call["arguments"])
    if len(tool_name) != 0:
        return f'Error on "{tool_name}"'
    return "Error"


def get_group_tool_call_messages(tool_calls: list[ToolCall]) -> list[str]:
    messages: list[str] = []
    for tool_call in tool_calls:
        messages.append(get_individual_tool_call_message(tool_call))
    return messages


def get_individual_tool_call_permission(tool_call: ToolCall, session_read_allowlist: list[str]) -> bool:
    with suppress(Exception):
        if tool_call["tool_name"] == "create_directory":
            return get_create_directory_permission(tool_call["arguments"])
        elif tool_call["tool_name"] == "delete_path":
            return get_delete_path_permission(tool_call["arguments"])
        elif tool_call["tool_name"] == "edit_file":
            return get_edit_file_permission(tool_call["arguments"])
        elif tool_call["tool_name"] == "execute_shell_command":
            return get_execute_shell_command_permission(tool_call["arguments"])
        elif tool_call["tool_name"] == "generate_random_integer":
            return get_generate_random_integer_permission(tool_call["arguments"])
        elif tool_call["tool_name"] == "invalid":
            return get_invalid_permission(tool_call["arguments"])
        elif tool_call["tool_name"] == "list_directory":
            return get_list_directory_permission(tool_call["arguments"])
        elif tool_call["tool_name"] == "move_path":
            return get_move_path_permission(tool_call["arguments"])
        elif tool_call["tool_name"] == "read_file":
            return get_read_file_permission(tool_call["arguments"], session_read_allowlist)
        elif tool_call["tool_name"] == "read_pdf_document":
            return get_read_pdf_document_permission(tool_call["arguments"], session_read_allowlist)
        elif tool_call["tool_name"] == "read_web_page":
            return get_read_web_page_permission(tool_call["arguments"])
        elif tool_call["tool_name"] == "search_web":
            return get_search_web_permission(tool_call["arguments"])
        elif tool_call["tool_name"] == "write_file":
            return get_write_file_permission(tool_call["arguments"])
    return False


def get_number_of_required_permissions(tool_calls: list[ToolCall], session_read_allowlist: list[str]) -> int:
    number_of_required_permissions: int = 0
    for tool_call in tool_calls:
        if not get_individual_tool_call_permission(tool_call, session_read_allowlist):
            number_of_required_permissions += 1
    return number_of_required_permissions


def get_tool_read_path(tool_call: ToolCall, tool_call_permission: bool) -> str | None:
    if tool_call["tool_name"] == "edit_file":
        return get_edit_file_read_path(tool_call["arguments"], tool_call_permission)
    elif tool_call["tool_name"] == "move_path":
        return get_move_path_read_path(tool_call["arguments"], tool_call_permission)
    elif tool_call["tool_name"] == "read_file":
        return get_read_file_read_path(tool_call["arguments"], tool_call_permission)
    elif tool_call["tool_name"] == "read_pdf_document":
        return get_read_pdf_document_read_path(tool_call["arguments"], tool_call_permission)
    elif tool_call["tool_name"] == "write_file":
        return get_write_file_read_path(tool_call["arguments"], tool_call_permission)
    else:
        return None


def execute_tool_call(tool_call: ToolCall, tool_call_permission: bool) -> str:
    tool_name: str = ""
    with suppress(Exception):
        tool_name = tool_call["tool_name"]
        if tool_call["tool_name"] == "create_directory":
            return create_directory(tool_call["arguments"])
        elif tool_call["tool_name"] == "delete_path":
            return delete_path(tool_call["arguments"], tool_call_permission)
        elif tool_call["tool_name"] == "edit_file":
            return edit_file(tool_call["arguments"], tool_call_permission)
        elif tool_call["tool_name"] == "execute_shell_command":
            return execute_shell_command(tool_call["arguments"], tool_call_permission)
        elif tool_call["tool_name"] == "generate_random_integer":
            return generate_random_integer(tool_call["arguments"])
        elif tool_call["tool_name"] == "invalid":
            return invalid(tool_call["arguments"])
        elif tool_call["tool_name"] == "list_directory":
            return list_directory(tool_call["arguments"])
        elif tool_call["tool_name"] == "move_path":
            return move_path(tool_call["arguments"], tool_call_permission)
        elif tool_call["tool_name"] == "read_file":
            return read_file(tool_call["arguments"], tool_call_permission)
        elif tool_call["tool_name"] == "read_pdf_document":
            return read_pdf_document(tool_call["arguments"], tool_call_permission)
        elif tool_call["tool_name"] == "read_web_page":
            return read_web_page(tool_call["arguments"])
        elif tool_call["tool_name"] == "search_web":
            return search_web(tool_call["arguments"])
        elif tool_call["tool_name"] == "write_file":
            return write_file(tool_call["arguments"], tool_call_permission)
    if len(tool_name) != 0:
        return f'Error on "{tool_name}"'
    return "Error"
