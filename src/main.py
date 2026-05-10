from sqlite3 import Connection

from ai.core import Ai, AiMessage
from database import close_db_connection, init_db, open_db_connection
from environment import Environment
from entity.session import Session
from tool import (
    execute_bash_command,
    execute_tool_call,
    get_default_tool_call_permission,
    get_tool_call_assistant_message,
    ToolCall,
)
from ui.core import Ui


def get_default_system_messages(environment: Environment, ui_system_message: str) -> list[str]:
    system_messages: list[str] = []
    default_instruction_messages: list[str] = [
        f"By default, you must always reply using {environment.language} with proper grammar (unless you see the need to reply in a different language)",
        "By default, you must always reply using strict Markdown syntax with proper formatting (unless you see the need to reply in a different format)",
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
    system_messages.append(ui_system_message)
    for system_command in system_commands:
        system_messages.append(execute_bash_command(system_command))
    return system_messages


def get_default_tool_names() -> list[str]:
    tool_names: list[str] = [
        "execute_bash_command",
        "generate_random_integer",
        "read_pdf_document",
        "read_web_page",
        "search_web",
    ]
    return tool_names


def ai_chat_loop(environment: Environment, db_connection: Connection, ai: Ai, ui: Ui) -> None:
    graceful_exit: bool = True
    session: Session = Session(ai)
    ui.startup()
    while True:
        try:
            user_input: str = ui.get_user_input(session.id, session.context_length)
            if user_input == "/new":
                session = Session(ai)
            elif user_input == "/raw":
                session = Session(ai, is_raw=True)
            elif user_input.startswith("/load "):
                referenced_session_id = int(user_input.split(" ")[1])
                session = Session(ai).load(ai, referenced_session_id, db_connection)
            elif user_input == "/replay":
                replay_message_index: int = 0
                replay_message = session.get_nth_message(ai, replay_message_index)
                while replay_message is not None:
                    if replay_message["role"] == "user":
                        ui.display_user_input(session.id, session.context_length, replay_message["message"])
                    elif replay_message["role"] == "assistant":
                        ui.display_assistant_message(session.id, session.context_length, replay_message["message"], "")
                    replay_message_index += 1
                    replay_message = session.get_nth_message(ai, replay_message_index)
            elif user_input == "/rewind":
                session.rewind_message(ai)
            elif user_input == "/exit":
                break
            else:
                if not session.is_raw and session.get_messages_count(ai) == 0:
                    session.add_system_messages(
                        ai, get_default_system_messages(environment, ui.get_system_instruction())
                    )
                    session.add_tools(ai, get_default_tool_names())
                has_added_user_message: bool = session.add_user_message(ai, user_input)
                if not has_added_user_message:
                    continue
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
                    for tool_call in tool_calls:
                        tool_call_message: str = get_tool_call_assistant_message(tool_call)
                        default_tool_call_permission: bool = get_default_tool_call_permission(tool_call)
                        final_tool_call_permission: bool = ui.display_tool_call_message(
                            session.id, session.context_length, tool_call_message, default_tool_call_permission
                        )
                        tool_call_output: str = execute_tool_call(tool_call, final_tool_call_permission)
                        session.add_tool_call(ai, tool_call, tool_call_output)
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
    ai_chat_loop(environment, db_connection, ai, ui)
    close_db_connection(db_connection)


if __name__ == "__main__":
    main()
