from sys import exit as sys_exit
from typing import Required, TypedDict

from environment import Environment
from tool.core import get_tool_system_instructions
from tool.execute_shell_command import execute_shell_command
from ui.core import Ui


class AgentConfig(TypedDict):
    name: Required[str]
    system_prompts: Required[list[str]]
    tool_names: Required[list[str]]


def get_agent_name(environment: Environment, agent_name_input: str = "") -> str:
    if agent_name_input in ["default", "raw"]:
        return agent_name_input
    else:
        return environment.default_agent_name


def _get_default_agent_config(environment: Environment, ui: Ui) -> AgentConfig:
    agent_name: str = "default"
    system_messages: list[str] = []
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
    instruction_messages: list[str] = [
        f"By default, you must always reply using {environment.language} with proper grammar (unless you see the need to reply in a different language)",
        "By default, you must always reply using strict Markdown syntax with proper formatting (unless you see the need to reply in a different format)",
        "You are a general-purpose AI agent",
        "Whenever you encounter README files of any kind, you should strongly prioritize reading them",
    ]
    system_messages.extend(instruction_messages)
    system_messages.extend(get_tool_system_instructions())
    system_messages.extend(ui.get_system_instructions())
    system_commands: list[str] = [
        "python --version",
        "git --version",
        "getent passwd ${USER}",
        "cat /etc/os-release",
        "uname -a",
        "hostnamectl",
        "date",
    ]
    for system_command in system_commands:
        system_messages.append(execute_shell_command(system_command))
    return AgentConfig(name=agent_name, system_prompts=system_messages, tool_names=tool_names)


def _get_raw_agent_config(environment: Environment, ui: Ui) -> AgentConfig:
    agent_name: str = "raw"
    system_messages: list[str] = []
    tool_names: list[str] = []
    return AgentConfig(name=agent_name, system_prompts=system_messages, tool_names=tool_names)


def get_agent_config(name: str, environment: Environment, ui: Ui, defaulting: bool = False) -> AgentConfig:
    if name == "default":
        return _get_default_agent_config(environment, ui)
    elif name == "raw":
        return _get_raw_agent_config(environment, ui)
    else:
        if not defaulting:
            return get_agent_config(environment.default_agent_name, environment, ui, defaulting=True)
        else:
            sys_exit(1)
