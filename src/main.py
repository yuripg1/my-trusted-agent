from sqlite3 import Connection

from ai.core import Ai, AiMessage
from database import close_db_connection, init_db, open_db_connection
from entity.session import Session
from environment import Environment
from tool.core import (
    ToolCall,
    execute_tool_call,
    get_group_tool_call_messages,
    get_individual_tool_call_message,
    get_individual_tool_call_permission,
    get_number_of_required_permissions,
    get_tool_system_instruction,
)
from tool.execute_shell_command import execute_shell_command
from ui.core import Ui


def _get_default_system_messages(environment: Environment, ui_system_message: str) -> list[str]:
    system_messages: list[str] = []
    default_instruction_messages: list[str] = [
        f"By default, you must always reply using {environment.language} with proper grammar (unless you see the need to reply in a different language)",
        "By default, you must always reply using strict Markdown syntax with proper formatting (unless you see the need to reply in a different format)",
        "You are a general-purpose AI agent with access to tools",
    ]
    system_commands: list[str] = [
        "python -V",
        "getent passwd ${USER}",
        "cat /etc/os-release",
        "uname -a",
        "hostnamectl",
        "date",
    ]
    system_messages.extend(default_instruction_messages)
    system_messages.append(get_tool_system_instruction())
    system_messages.append(ui_system_message)
    for system_command in system_commands:
        system_messages.append(execute_shell_command(system_command))
    return system_messages


def _get_default_tool_names() -> list[str]:
    tool_names: list[str] = [
        "create_directory",
        "delete_path",
        "edit_file",
        "execute_shell_command",
        "generate_random_integer",
        "list_directory",
        "move_path",
        "read_file",
        "read_pdf_document",
        "read_web_page",
        "search_web",
        "write_file",
    ]
    return tool_names


def _handle_new_command(ai: Ai) -> Session:
    return Session(ai)


def _handle_raw_command(ai: Ai) -> Session:
    return Session(ai, is_raw=True)


def _handle_load_command(ai: Ai, db_connection: Connection, user_input: str, ui: Ui) -> Session:
    load_input_parts: list[str] = user_input.split(" ")
    referenced_session_id = int(load_input_parts[1])
    should_replay: bool = len(load_input_parts) >= 3 and load_input_parts[2] == "replay"
    session = Session(ai).load(ai, referenced_session_id, db_connection)
    if should_replay:
        replay_message_index: int = 0
        replay_message = session.get_nth_message(ai, replay_message_index)
        while replay_message is not None:
            if replay_message["role"] == "user":
                ui.display_user_message(session.id, session.context_length, replay_message["message"])
            elif replay_message["role"] == "assistant":
                ui.display_assistant_message(session.id, session.context_length, replay_message["message"])
            replay_message_index += 1
            replay_message = session.get_nth_message(ai, replay_message_index)
    return session


def _handle_rewind_command(session: Session, ai: Ai, ui: Ui) -> None:
    session.rewind_message(ai)
    last_message_index: int = session.get_messages_count(ai) - 1
    last_message: AiMessage | None = session.get_nth_message(ai, last_message_index)
    if last_message is not None and last_message["role"] == "assistant":
        ui.display_assistant_message(
            session.id, session.context_length, last_message["message"], last_message["reasoning"]
        )


def _process_user_input(
    environment: Environment, db_connection: Connection, ai: Ai, ui: Ui, session: Session, user_input: str
) -> None:
    if not session.is_raw and session.get_messages_count(ai) == 0:
        session.add_system_messages(ai, _get_default_system_messages(environment, ui.get_system_instruction()))
        session.add_tools(ai, _get_default_tool_names())
    has_added_user_message: bool = session.add_user_message(ai, user_input)
    if not has_added_user_message:
        return
    while True:
        session.request_assistant_reply(ai)
        assistant_message_index: int = session.get_messages_count(ai) - 1
        assistant_message: AiMessage | None = session.get_nth_message(ai, assistant_message_index)
        if assistant_message is None:
            break
        ui.display_assistant_message(
            session.id, session.context_length, assistant_message["message"], assistant_message["reasoning"]
        )
        tool_calls: list[ToolCall] = session.get_tool_calls_from_nth_message(ai, assistant_message_index)
        if len(tool_calls) == 0:
            session.auto_save(ai, db_connection)
            break
        number_of_required_permissions: int = get_number_of_required_permissions(tool_calls)
        group_tool_call_permission: bool = number_of_required_permissions == 0
        group_tool_call_messages: list[str] = get_group_tool_call_messages(tool_calls)
        group_tool_call_permission = ui.display_group_tool_call_message(
            session.id, session.context_length, group_tool_call_messages, group_tool_call_permission
        )
        number_of_denied_permissions: int = 0
        for tool_call in tool_calls:
            tool_call_output: str = ""
            if group_tool_call_permission:
                tool_call_output = execute_tool_call(tool_call, group_tool_call_permission)
            else:
                individual_tool_call_permission: bool = get_individual_tool_call_permission(tool_call)
                if number_of_required_permissions > 1:
                    individual_tool_call_message: str = get_individual_tool_call_message(tool_call)
                    individual_tool_call_permission = ui.display_individual_tool_call_message(
                        individual_tool_call_message, individual_tool_call_permission
                    )
                if not individual_tool_call_permission:
                    number_of_denied_permissions += 1
                tool_call_output = execute_tool_call(tool_call, individual_tool_call_permission)
            session.add_tool_call(ai, tool_call, tool_call_output)
        if not group_tool_call_permission and number_of_required_permissions == number_of_denied_permissions:
            break


def _ai_chat_loop(environment: Environment, db_connection: Connection, ai: Ai, ui: Ui) -> None:
    graceful_exit: bool = True
    session: Session = Session(ai)
    ui.startup()
    while True:
        try:
            user_input: str = ui.get_user_input(session.id, session.context_length)
            if user_input == "/new":
                session = _handle_new_command(ai)
            elif user_input == "/raw":
                session = _handle_raw_command(ai)
            elif user_input.startswith("/load "):
                session = _handle_load_command(ai, db_connection, user_input, ui)
            elif user_input == "/rewind":
                _handle_rewind_command(session, ai, ui)
            elif user_input == "/exit":
                break
            else:
                _process_user_input(environment, db_connection, ai, ui, session, user_input)
        except KeyboardInterrupt:
            graceful_exit = False
            break
    ui.teardown(graceful_exit)


def main() -> None:
    environment = Environment()
    db_connection = open_db_connection(environment.db_path)
    init_db(db_connection)
    ai = Ai(environment)
    ui = Ui(environment)
    _ai_chat_loop(environment, db_connection, ai, ui)
    close_db_connection(db_connection)


if __name__ == "__main__":
    main()
