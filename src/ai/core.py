from json import dumps, loads
from typing import cast, Literal, NotRequired, Required, TypedDict

from ai.deepseek import (
    DeepSeekAi,
    DeepSeekMessage,
    DeepSeekModelType,
    DeepSeekReasoningEffortType,
    DeepSeekThinkingType,
)
from ai.deepseek_api_tools import DeepSeekTool
from environment import Environment
from tool import ToolCall

AiProviderType = Literal["deepseek"]
AiRoleType = Literal["assistant", "system", "tool", "user"]


class AiMessages(TypedDict):
    deepseek_messages: NotRequired[list[DeepSeekMessage]]


class AiTools(TypedDict):
    deepseek_tools: NotRequired[list[DeepSeekTool]]


class AiMessage(TypedDict):
    role: Required[AiRoleType]
    message: Required[str]
    reasoning: Required[str]


class Ai:
    provider: AiProviderType
    deepseek_ai: DeepSeekAi | None

    def __init__(self, environment: Environment) -> None:
        self.provider = cast(AiProviderType, environment.ai_provider)
        if (
            self.provider == "deepseek"
            and environment.deepseek_model is not None
            and environment.deepseek_thinking is not None
            and environment.deepseek_reasoning_effort is not None
        ):
            self.deepseek_ai = DeepSeekAi(
                api_key=environment.deepseek_api_key,
                base_url=environment.deepseek_base_url,
                model=cast(DeepSeekModelType, environment.deepseek_model),
                thinking=cast(DeepSeekThinkingType, environment.deepseek_thinking),
                reasoning_effort=cast(DeepSeekReasoningEffortType, environment.deepseek_reasoning_effort),
                max_tokens=environment.deepseek_max_tokens,
            )

    def create_messages(self) -> AiMessages:
        if self.provider == "deepseek" and self.deepseek_ai is not None:
            return AiMessages(deepseek_messages=self.deepseek_ai.create_messages())
        else:
            return AiMessages()

    def create_tools(self) -> AiTools:
        if self.provider == "deepseek" and self.deepseek_ai is not None:
            return AiTools(deepseek_tools=self.deepseek_ai.create_tools())
        else:
            return AiTools()

    def rewind_message(self, messages: AiMessages) -> None:
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_messages" in messages:
            self.deepseek_ai.rewind_message(messages["deepseek_messages"])

    def add_tools(self, tools: AiTools, tool_names: list[str]) -> None:
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_tools" in tools:
            self.deepseek_ai.add_tools(tools["deepseek_tools"], tool_names)

    def add_system_messages(self, messages: AiMessages, system_messages: list[str]) -> None:
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_messages" in messages:
            self.deepseek_ai.add_system_messages(messages["deepseek_messages"], system_messages)

    def add_user_message(self, messages: AiMessages, user_message: str) -> bool:
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_messages" in messages:
            return self.deepseek_ai.add_user_message(messages["deepseek_messages"], user_message)
        else:
            return False

    def add_tool_call(self, messages: AiMessages, tool_call: ToolCall, tool_call_output: str) -> None:
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_messages" in messages:
            return self.deepseek_ai.add_tool_call(messages["deepseek_messages"], tool_call, tool_call_output)

    def request_assistant_reply(self, messages: AiMessages, tools: AiTools) -> int:
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_messages" in messages:
            context_length: int = self.deepseek_ai.request_assistant_reply(
                messages["deepseek_messages"], tools["deepseek_tools"]
            )
            return context_length
        else:
            return 0

    def get_messages_count(self, messages: AiMessages) -> int:
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_messages" in messages:
            return self.deepseek_ai.get_messages_count(messages["deepseek_messages"])
        else:
            return 0

    def get_nth_message(self, messages: AiMessages, message_index: int) -> AiMessage | None:
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_messages" in messages:
            nth_message: DeepSeekMessage | None = self.deepseek_ai.get_nth_message(
                messages["deepseek_messages"], message_index
            )
            if nth_message is not None:
                role: AiRoleType | None = None
                if nth_message["role"] == "assistant":
                    role = "assistant"
                elif nth_message["role"] == "system":
                    role = "system"
                elif nth_message["role"] == "tool":
                    role = "tool"
                elif nth_message["role"] == "user":
                    role = "user"
                if role is not None:
                    message: str = ""
                    if "content" in nth_message:
                        message = nth_message["content"]
                    reasoning: str = ""
                    if "reasoning_content" in nth_message:
                        reasoning = nth_message["reasoning_content"]
                    return AiMessage(role=role, message=message, reasoning=reasoning)
            return None
        else:
            return None

    def get_tool_calls_from_nth_message(self, messages: AiMessages, message_index: int) -> list[ToolCall]:
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_messages" in messages:
            return self.deepseek_ai.get_tool_calls_from_nth_message(messages["deepseek_messages"], message_index)
        else:
            tool_calls: list[ToolCall] = []
            return tool_calls

    def decode_messages_json(self, messages_json: str) -> AiMessages:
        parsed_messages = loads(messages_json)
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_messages" in parsed_messages:
            return AiMessages(
                deepseek_messages=self.deepseek_ai.decode_messages_json(parsed_messages["deepseek_messages"])
            )
        else:
            return AiMessages()

    def encode_messages_json(self, messages: AiMessages) -> str:
        return dumps(messages)

    def decode_tools_json(self, tools_json: str) -> AiTools:
        parsed_tools = loads(tools_json)
        if self.provider == "deepseek" and self.deepseek_ai is not None and "deepseek_tools" in parsed_tools:
            return AiTools(deepseek_tools=self.deepseek_ai.decode_tools_json(parsed_tools["deepseek_tools"]))
        else:
            return AiTools()

    def encode_tools_json(self, tools: AiTools) -> str:
        return dumps(tools)
