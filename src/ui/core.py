from typing import Literal, cast

from environment import Environment
from ui.terminal import TerminalUi

UiChannelType = Literal["terminal"]


class Ui:
    channel: UiChannelType
    terminal_ui: TerminalUi | None

    def __init__(self, environment: Environment) -> None:
        self.channel = cast(UiChannelType, environment.ui_channel)
        if self.channel == "terminal":
            self.terminal_ui = TerminalUi(show_reasoning=environment.show_reasoning)

    def get_system_instructions(self) -> list[str]:
        if self.channel == "terminal" and self.terminal_ui is not None:
            return self.terminal_ui.get_system_instructions()
        else:
            system_instructions: list[str] = []
            return system_instructions

    def startup(self) -> None:
        if self.channel == "terminal" and self.terminal_ui is not None:
            self.terminal_ui.startup()

    def teardown(self, graceful_exit: bool = True) -> None:
        if self.channel == "terminal" and self.terminal_ui is not None:
            self.terminal_ui.teardown(graceful_exit)

    def display_user_message(self, session_id: int | None, context_length: int, message: str) -> None:
        if self.channel == "terminal" and self.terminal_ui is not None:
            self.terminal_ui.display_user_message(session_id, context_length, message)

    def get_user_input(self, session_id: int | None, context_length: int) -> str:
        if self.channel == "terminal" and self.terminal_ui is not None:
            return self.terminal_ui.get_user_input(session_id, context_length)
        else:
            return ""

    def display_assistant_message(
        self, session_id: int | None, context_length: int, message: str, reasoning: str = ""
    ) -> None:
        if self.channel == "terminal" and self.terminal_ui is not None:
            self.terminal_ui.display_assistant_message(session_id, context_length, message, reasoning)

    def display_group_tool_call_message(
        self,
        session_id: int | None,
        context_length: int,
        tool_call_messages: list[str],
        tool_call_permission: bool = True,
    ) -> bool:
        if self.channel == "terminal" and self.terminal_ui is not None:
            return self.terminal_ui.display_group_tool_call_message(
                session_id, context_length, tool_call_messages, tool_call_permission
            )
        else:
            return False

    def display_individual_tool_call_message(self, tool_call_message: str, tool_call_permission: bool = True) -> bool:
        if self.channel == "terminal" and self.terminal_ui is not None:
            return self.terminal_ui.display_individual_tool_call_message(tool_call_message, tool_call_permission)
        else:
            return False
