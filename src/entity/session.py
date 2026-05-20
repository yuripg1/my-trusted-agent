from sqlite3 import Connection, Cursor
from typing import Self

from ai.core import Ai, AiMessage, AiMessages, AiProviderType, AiTools
from tool.core import ToolCall


class Session:
    id: int | None
    ai_provider: AiProviderType
    agent_name: str
    context_length: int
    __tools: AiTools
    __messages: AiMessages

    def __init__(self, ai: Ai, agent_name: str) -> None:
        self.id = None
        self.ai_provider = ai.provider
        self.agent_name = agent_name
        self.context_length = 0
        self.__tools = ai.create_tools()
        self.__messages = ai.create_messages()

    def load(self, ai: Ai, id: int, db_connection: Connection) -> Self:
        cursor: Cursor = db_connection.execute(
            "SELECT agent_name, context_length, tools, messages FROM sessions WHERE id = ? and ai_provider = ?",
            (id, str(ai.provider)),
        )
        fetched_data = cursor.fetchone()
        if fetched_data is not None:
            self.id = id
            self.ai_provider = ai.provider
            self.agent_name = str(fetched_data["agent_name"])
            self.context_length = int(fetched_data["context_length"])
            self.__tools = ai.decode_tools_json(str(fetched_data["tools"]))
            self.__messages = ai.decode_messages_json(str(fetched_data["messages"]))
        return self

    def auto_save(self, ai: Ai, db_connection: Connection) -> None:
        tools: str = ai.encode_tools_json(self.__tools)
        messages_json: str = ai.encode_messages_json(self.__messages)
        if self.id is None:
            cursor: Cursor = db_connection.execute(
                "INSERT INTO sessions (ai_provider, agent_name, context_length, tools, messages) VALUES (?, ?, ?, ?, ?)",
                (str(self.ai_provider), self.agent_name, self.context_length, tools, messages_json),
            )
            self.id = cursor.lastrowid
        else:
            db_connection.execute(
                'UPDATE sessions SET context_length = ?, messages = ?, updated_at = datetime("now") WHERE id = ?',
                (self.context_length, messages_json, self.id),
            )
        db_connection.commit()

    def rewind_message(self, ai: Ai) -> None:
        self.id = None
        self.context_length = 0
        ai.rewind_message(self.__messages)

    def add_tools(self, ai: Ai, tool_names: list[str]) -> None:
        ai.add_tools(self.__tools, tool_names)

    def add_system_messages(self, ai: Ai, system_messages: list[str]) -> None:
        ai.add_system_messages(self.__messages, system_messages)

    def add_user_message(self, ai: Ai, user_message: str) -> bool:
        return ai.add_user_message(self.__messages, user_message)

    def add_tool_call(self, ai: Ai, tool_call: ToolCall, tool_call_output: str) -> None:
        return ai.add_tool_call(self.__messages, tool_call, tool_call_output)

    def request_assistant_reply(self, ai: Ai) -> None:
        self.context_length = ai.request_assistant_reply(self.__messages, self.__tools)

    def get_messages_count(self, ai: Ai) -> int:
        return ai.get_messages_count(self.__messages)

    def get_nth_message(self, ai: Ai, message_index: int) -> AiMessage | None:
        return ai.get_nth_message(self.__messages, message_index)

    def get_tool_calls_from_nth_message(self, ai: Ai, message_index: int) -> list[ToolCall]:
        return ai.get_tool_calls_from_nth_message(self.__messages, message_index)

    def render_to_markdown(self, ai: Ai) -> str:
        lines: list[str] = []
        session_id: int = self.id if self.id is not None else 0
        lines.append(f"# Session {session_id}")
        lines.append("")
        message_index: int = 0
        while True:
            message: AiMessage | None = self.get_nth_message(ai, message_index)
            if message is None:
                break
            message_content: str = message["message"]
            if len(message_content) == 0:
                message_index += 1
                continue
            if message["role"] == "user":
                lines.append("---")
                lines.append("")
                lines.append("## User")
                lines.append("")
                lines.append(message_content)
                lines.append("")
            elif message["role"] == "assistant":
                lines.append("## Assistant")
                lines.append("")
                lines.append(message_content)
                lines.append("")
            message_index += 1
        return "\n".join(lines).strip() + "\n"
