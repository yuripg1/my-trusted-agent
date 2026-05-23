from contextlib import suppress
from sqlite3 import Connection

from agent import get_agent_config, get_agent_name
from ai.core import Ai, AiMessage
from entity.session import Session
from environment import Environment
from tool.core import (
    ToolCall,
    execute_tool_call,
    get_group_tool_call_messages,
    get_individual_tool_call_message,
    get_individual_tool_call_permission,
    get_number_of_required_permissions,
    get_tool_read_path,
)
from ui.core import Ui


def _handle_new_command(environment: Environment, ai: Ai, user_input: str) -> Session:
    new_input_parts: list[str] = user_input.split(" ", 1)
    agent_name_input: str = ""
    if len(new_input_parts) >= 2:
        agent_name_input = new_input_parts[1]
    agent_name: str = get_agent_name(environment, agent_name_input)
    return Session(ai, agent_name)


def _handle_load_command(
    environment: Environment, ai: Ai, db_connection: Connection, user_input: str, ui: Ui
) -> Session:
    load_input_parts: list[str] = user_input.split(" ", 2)
    session_id: int = 0
    if len(load_input_parts) >= 2:
        with suppress(Exception):
            session_id = int(load_input_parts[1])
    replay: bool = False
    if len(load_input_parts) >= 3:
        replay = load_input_parts[2] == "replay"
    default_agent_name: str = get_agent_name(environment)
    session = Session(ai, default_agent_name).load(ai, session_id, db_connection)
    if replay:
        replay_message_index: int = 0
        while True:
            replay_message: AiMessage | None = session.get_nth_message(ai, replay_message_index)
            if replay_message is None:
                break
            if replay_message["role"] == "user":
                ui.display_user_message(session.id, session.context_length, replay_message["message"])
            elif replay_message["role"] == "assistant":
                ui.display_assistant_message(session.id, session.context_length, replay_message["message"])
            replay_message_index += 1
    return session


def _handle_system_command(ai: Ai, session: Session, user_input: str) -> None:
    system_input_parts: list[str] = user_input.split(" ", 1)
    if len(system_input_parts) >= 2:
        system_prompt: str = system_input_parts[1]
        session.add_system_messages(ai, [system_prompt])


def _handle_export_command(environment: Environment, ai: Ai, session: Session) -> None:
    if session.id is None:
        return
    if session.get_messages_count(ai) == 0:
        return
    markdown_content: str = session.render_to_markdown(ai)
    export_file_path: str = f"{environment.export_path}_{session.id}.md"
    with open(export_file_path, "w") as file:
        file.write(markdown_content)


def _handle_rewind_command(session: Session, ai: Ai, ui: Ui) -> None:
    session.rewind_message(ai)
    last_message_index: int = session.get_messages_count(ai) - 1
    last_message: AiMessage | None = session.get_nth_message(ai, last_message_index)
    if last_message is not None and last_message["role"] == "assistant":
        ui.display_assistant_message(
            session.id, session.context_length, last_message["message"], last_message["reasoning"]
        )


def _handle_user_input(
    environment: Environment, db_connection: Connection, ai: Ai, ui: Ui, session: Session, user_input: str
) -> None:
    if session.get_messages_count(ai) == 0:
        agent_config = get_agent_config(session.agent_name, environment, ui)
        session.add_system_messages(ai, agent_config["system_prompts"])
        session.add_tools(ai, agent_config["tool_names"])
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
        number_of_required_permissions: int = get_number_of_required_permissions(tool_calls, session.read_allowlist)
        group_tool_call_permission: bool = number_of_required_permissions == 0
        group_tool_call_messages: list[str] = get_group_tool_call_messages(tool_calls)
        group_tool_call_permission = ui.display_group_tool_call_message(
            session.id, session.context_length, group_tool_call_messages, group_tool_call_permission
        )
        number_of_denied_permissions: int = 0
        for tool_call in tool_calls:
            individual_tool_call_permission: bool = group_tool_call_permission
            if not individual_tool_call_permission:
                individual_tool_call_permission = get_individual_tool_call_permission(tool_call, session.read_allowlist)
                if number_of_required_permissions > 1:
                    individual_tool_call_message: str = get_individual_tool_call_message(tool_call)
                    individual_tool_call_permission = ui.display_individual_tool_call_message(
                        individual_tool_call_message, individual_tool_call_permission
                    )
            if not individual_tool_call_permission:
                number_of_denied_permissions += 1
            read_path: str | None = get_tool_read_path(tool_call, individual_tool_call_permission)
            if read_path is not None:
                session.add_to_read_allowlist(read_path)
            tool_call_output: str = execute_tool_call(tool_call, individual_tool_call_permission)
            session.add_tool_call(ai, tool_call, tool_call_output)
        if not group_tool_call_permission and number_of_required_permissions == number_of_denied_permissions:
            break


def chat_loop(environment: Environment, db_connection: Connection, ai: Ai, ui: Ui) -> None:
    graceful_exit: bool = True
    default_agent_name: str = get_agent_name(environment)
    session: Session = Session(ai, default_agent_name)
    ui.startup()
    while True:
        try:
            user_input: str = ui.get_user_input(session.id, session.context_length)
            if user_input.startswith("/new"):
                session = _handle_new_command(environment, ai, user_input)
            elif user_input.startswith("/load"):
                session = _handle_load_command(environment, ai, db_connection, user_input, ui)
            elif user_input == "/rewind":
                _handle_rewind_command(session, ai, ui)
            elif user_input.startswith("/system"):
                _handle_system_command(ai, session, user_input)
            elif user_input == "/export":
                _handle_export_command(environment, ai, session)
            elif user_input == "/exit":
                break
            else:
                _handle_user_input(environment, db_connection, ai, ui, session, user_input)
        except KeyboardInterrupt:
            graceful_exit = False
            break
    ui.teardown(graceful_exit)
