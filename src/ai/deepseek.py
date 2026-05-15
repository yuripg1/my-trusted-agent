from collections.abc import Mapping
from json import dumps, loads
from time import sleep
from typing import Any, Literal, NotRequired, Required, TypedDict

from requests import Response, post

from ai.deepseek_api_tools import DEEPSEEK_TOOLS, DeepSeekTool, DeepSeekToolFunction
from tool.core import ToolCall
from tool.create_directory import CreateDirectoryArguments, CreateDirectoryToolCall
from tool.delete_file_or_directory import DeleteFileOrDirectoryArguments, DeleteFileOrDirectoryToolCall
from tool.edit_file import EditFileArguments, EditFileToolCall
from tool.execute_shell_command import ExecuteShellCommandArguments, ExecuteShellCommandToolCall
from tool.generate_random_integer import GenerateRandomIntegerArguments, GenerateRandomIntegerToolCall
from tool.list_directory import ListDirectoryArguments, ListDirectoryToolCall
from tool.read_file import ReadFileArguments, ReadFileToolCall
from tool.read_pdf_document import ReadPdfDocumentArguments, ReadPdfDocumentToolCall
from tool.read_web_page import ReadWebPageArguments, ReadWebPageToolCall
from tool.search_web import SearchWebArguments, SearchWebToolCall
from tool.write_file import WriteFileArguments, WriteFileToolCall

DeepSeekRoleType = Literal["assistant", "system", "tool", "user"]
DeepSeekToolChoiceType = Literal["none", "auto", "required"]
DeepSeekModelType = Literal["deepseek-v4-flash", "deepseek-v4-pro"]
DeepSeekThinkingType = Literal["enabled", "disabled"]
DeepSeekReasoningEffortType = Literal["high", "max"]
DeepSeekResponseFormat = Literal["text", "json_object"]


class DeepSeekToolCallFunction(TypedDict):
    name: Required[str]
    arguments: Required[str]


class DeepSeekToolCall(TypedDict):
    id: Required[str]
    type: Required[str]
    function: Required[DeepSeekToolCallFunction]


class DeepSeekMessage(TypedDict):
    role: Required[DeepSeekRoleType]
    content: NotRequired[str]
    reasoning_content: NotRequired[str]
    tool_calls: NotRequired[list[DeepSeekToolCall]]
    tool_call_id: NotRequired[str]


class DeepSeekRequestThinking(TypedDict):
    type: Required[DeepSeekThinkingType]


class DeepSeekRequest(TypedDict):
    model: Required[DeepSeekModelType]
    messages: Required[list[DeepSeekMessage]]
    thinking: Required[DeepSeekRequestThinking]
    reasoning_effort: NotRequired[DeepSeekReasoningEffortType]
    max_tokens: Required[int]
    stream: Required[bool]
    tools: Required[list[DeepSeekTool]]
    tool_choice: Required[str]


API_STREAM: bool = False
API_TOOL_CHOICE: DeepSeekToolChoiceType = "auto"
API_WAIT_AFTER_ERROR: int = 2
API_REQUEST_TIMEOUT: int = 600


class DeepSeekAi:
    api_key: str
    base_url: str
    model: DeepSeekModelType
    thinking: DeepSeekThinkingType
    reasoning_effort: DeepSeekReasoningEffortType
    max_tokens: int
    stream: bool
    tools: list[DeepSeekTool]
    tool_choice: DeepSeekToolChoiceType

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: DeepSeekModelType,
        thinking: DeepSeekThinkingType,
        reasoning_effort: DeepSeekReasoningEffortType,
        max_tokens: int,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens

    def __add_to_messages(
        self,
        messages: list[DeepSeekMessage],
        role: DeepSeekRoleType,
        content: str,
        reasoning_content: str = "",
        tool_calls: list[DeepSeekToolCall] = [],
        tool_call_id: str = "",
    ) -> None:
        trimmed_content: str = content.strip()
        trimmed_reasoning_content: str = reasoning_content.strip()
        if role in ["assistant", "system", "user"]:
            new_generic_message: DeepSeekMessage = {"role": role, "content": trimmed_content}
            if len(trimmed_reasoning_content) != 0:
                new_generic_message["reasoning_content"] = trimmed_reasoning_content
            if len(tool_calls) != 0:
                new_generic_message["tool_calls"] = tool_calls
            messages.append(new_generic_message)
        elif role == "tool":
            new_tool_message: DeepSeekMessage = {"role": role, "content": trimmed_content, "tool_call_id": tool_call_id}
            messages.append(new_tool_message)

    def create_messages(self) -> list[DeepSeekMessage]:
        messages: list[DeepSeekMessage] = []
        return messages

    def create_tools(self) -> list[DeepSeekTool]:
        tools: list[DeepSeekTool] = []
        return tools

    def rewind_message(self, messages: list[DeepSeekMessage]) -> None:
        while len(messages) != 0 and messages[-1]["role"] != "user":
            del messages[-1]
        if len(messages) != 0:
            del messages[-1]

    def add_tools(self, tools: list[DeepSeekTool], tool_names: list[str]) -> None:
        for tool_name in tool_names:
            if tool_name in DEEPSEEK_TOOLS:
                tools.append(DEEPSEEK_TOOLS[tool_name])

    def add_system_messages(self, messages: list[DeepSeekMessage], system_messages: list[str]) -> None:
        for system_message in system_messages:
            trimmed_system_message: str = system_message.strip()
            if len(trimmed_system_message) != 0:
                self.__add_to_messages(messages, "system", trimmed_system_message)

    def add_user_message(self, messages: list[DeepSeekMessage], user_message: str) -> bool:
        trimmed_user_message: str = user_message.strip()
        if len(trimmed_user_message) != 0:
            self.__add_to_messages(messages, "user", trimmed_user_message)
            return True
        else:
            return False

    def add_tool_call(self, messages: list[DeepSeekMessage], tool_call: ToolCall, tool_call_output: str) -> None:
        trimmed_tool_call_output: str = tool_call_output.strip()
        if len(trimmed_tool_call_output) != 0:
            self.__add_to_messages(messages, "tool", trimmed_tool_call_output, "", [], tool_call["id"])

    def request_assistant_reply(self, messages: list[DeepSeekMessage], tools: list[DeepSeekTool]) -> int:
        headers: Mapping[str, str] = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload_thinking: DeepSeekRequestThinking = {"type": self.thinking}
        payload: DeepSeekRequest = {
            "model": self.model,
            "messages": messages,
            "thinking": payload_thinking,
            "max_tokens": self.max_tokens,
            "stream": API_STREAM,
            "tool_choice": API_TOOL_CHOICE,
            "tools": tools,
        }
        if payload["thinking"]["type"] == "enabled":
            payload["reasoning_effort"] = self.reasoning_effort
        response: Response | None = None
        try:
            response = post(
                f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=API_REQUEST_TIMEOUT
            )
        except:
            pass
        if response is None or response.status_code != 200:
            if (
                response is None
                or response.status_code == 429
                or (response.status_code >= 500 and response.status_code <= 599)
            ):
                sleep(API_WAIT_AFTER_ERROR)
                return self.request_assistant_reply(messages, tools)
            print(dumps(payload, indent=2))
            print(response.status_code)
            print(dumps(response.json(), indent=2))
        data = response.json()
        total_tokens: int = int(data["usage"]["total_tokens"])
        message = data["choices"][0]["message"]
        content: str = message.get("content", "").strip()
        reasoning_content: str = message.get("reasoning_content", "").strip()
        tool_calls: list[DeepSeekToolCall] = []
        message_tool_calls = message.get("tool_calls", [])
        for message_tool_call in message_tool_calls:
            tool_calls.append(
                DeepSeekToolCall(
                    id=message_tool_call["id"],
                    type=message_tool_call["type"],
                    function=DeepSeekToolCallFunction(
                        name=message_tool_call["function"]["name"], arguments=message_tool_call["function"]["arguments"]
                    ),
                )
            )
        self.__add_to_messages(messages, "assistant", content, reasoning_content, tool_calls)
        return total_tokens

    def get_messages_count(self, messages: list[DeepSeekMessage]) -> int:
        return len(messages)

    def get_nth_message(self, messages: list[DeepSeekMessage], message_index: int) -> DeepSeekMessage | None:
        if len(messages) <= message_index or message_index < 0:
            return None
        else:
            return messages[message_index]

    def get_tool_calls_from_nth_message(self, messages: list[DeepSeekMessage], message_index: int) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        if len(messages) > message_index:
            nth_message: DeepSeekMessage = messages[message_index]
            if "tool_calls" in nth_message:
                for tool_call in nth_message["tool_calls"]:
                    if tool_call["function"]["name"] == "create_directory":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            CreateDirectoryToolCall(
                                id=tool_call["id"],
                                tool_name="create_directory",
                                arguments=CreateDirectoryArguments(path=tool_call_arguments["path"]),
                            )
                        )
                    elif tool_call["function"]["name"] == "delete_file_or_directory":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            DeleteFileOrDirectoryToolCall(
                                id=tool_call["id"],
                                tool_name="delete_file_or_directory",
                                arguments=DeleteFileOrDirectoryArguments(
                                    type=tool_call_arguments["type"], path=tool_call_arguments["path"]
                                ),
                            )
                        )
                    elif tool_call["function"]["name"] == "edit_file":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            EditFileToolCall(
                                id=tool_call["id"],
                                tool_name="edit_file",
                                arguments=EditFileArguments(
                                    path=tool_call_arguments["path"],
                                    search_for=tool_call_arguments["search_for"],
                                    replace_with=tool_call_arguments["replace_with"],
                                    number_of_substitutions=tool_call_arguments["number_of_substitutions"],
                                ),
                            )
                        )
                    elif tool_call["function"]["name"] == "execute_shell_command":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            ExecuteShellCommandToolCall(
                                id=tool_call["id"],
                                tool_name="execute_shell_command",
                                arguments=ExecuteShellCommandArguments(command=tool_call_arguments["command"]),
                            )
                        )
                    elif tool_call["function"]["name"] == "generate_random_integer":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            GenerateRandomIntegerToolCall(
                                id=tool_call["id"],
                                tool_name="generate_random_integer",
                                arguments=GenerateRandomIntegerArguments(
                                    min=tool_call_arguments["min"], max=tool_call_arguments["max"]
                                ),
                            )
                        )
                    elif tool_call["function"]["name"] == "list_directory":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            ListDirectoryToolCall(
                                id=tool_call["id"],
                                tool_name="list_directory",
                                arguments=ListDirectoryArguments(path=tool_call_arguments["path"]),
                            )
                        )
                    elif tool_call["function"]["name"] == "read_file":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            ReadFileToolCall(
                                id=tool_call["id"],
                                tool_name="read_file",
                                arguments=ReadFileArguments(path=tool_call_arguments["path"]),
                            )
                        )
                    elif tool_call["function"]["name"] == "read_pdf_document":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            ReadPdfDocumentToolCall(
                                id=tool_call["id"],
                                tool_name="read_pdf_document",
                                arguments=ReadPdfDocumentArguments(
                                    location_type=tool_call_arguments["location_type"],
                                    location=tool_call_arguments["location"],
                                ),
                            )
                        )
                    elif tool_call["function"]["name"] == "read_web_page":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            ReadWebPageToolCall(
                                id=tool_call["id"],
                                tool_name="read_web_page",
                                arguments=ReadWebPageArguments(url=tool_call_arguments["url"]),
                            )
                        )
                    elif tool_call["function"]["name"] == "search_web":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            SearchWebToolCall(
                                id=tool_call["id"],
                                tool_name="search_web",
                                arguments=SearchWebArguments(
                                    query=tool_call_arguments["query"],
                                    max_results_per_page=tool_call_arguments["max_results_per_page"],
                                    results_page_number=tool_call_arguments["results_page_number"],
                                ),
                            )
                        )
                    elif tool_call["function"]["name"] == "write_file":
                        tool_call_arguments = loads(tool_call["function"]["arguments"])
                        tool_calls.append(
                            WriteFileToolCall(
                                id=tool_call["id"],
                                tool_name="write_file",
                                arguments=WriteFileArguments(
                                    path=tool_call_arguments["path"],
                                    mode=tool_call_arguments["mode"],
                                    content=tool_call_arguments["content"],
                                ),
                            )
                        )
        return tool_calls

    def decode_messages_json(self, parsed_messages: Any) -> list[DeepSeekMessage]:
        messages: list[DeepSeekMessage] = []
        for parsed_message in parsed_messages:
            new_message: DeepSeekMessage = DeepSeekMessage(role=parsed_message["role"])
            if "content" in parsed_message:
                new_message["content"] = str(parsed_message["content"])
            if "reasoning_content" in parsed_message:
                new_message["reasoning_content"] = str(parsed_message["reasoning_content"])
            if "tool_call_id" in parsed_message:
                new_message["tool_call_id"] = str(parsed_message["tool_call_id"])
            if "tool_calls" in parsed_message:
                new_tool_calls: list[DeepSeekToolCall] = []
                for parsed_tool_calls in parsed_message["tool_calls"]:
                    new_tool_calls.append(
                        DeepSeekToolCall(
                            id=str(parsed_tool_calls["id"]),
                            type=str(parsed_tool_calls["type"]),
                            function=DeepSeekToolCallFunction(
                                name=str(parsed_tool_calls["function"]["name"]),
                                arguments=str(parsed_tool_calls["function"]["arguments"]),
                            ),
                        )
                    )
                new_message["tool_calls"] = new_tool_calls
            messages.append(new_message)
        return messages

    def decode_tools_json(self, parsed_tools: Any) -> list[DeepSeekTool]:
        tools: list[DeepSeekTool] = []
        for parsed_tool in parsed_tools:
            tools.append(
                DeepSeekTool(
                    type=str(parsed_tool["type"]),
                    function=DeepSeekToolFunction(
                        name=str(parsed_tool["function"]["name"]),
                        description=str(parsed_tool["function"]["description"]),
                        parameters=parsed_tool["function"]["parameters"],
                    ),
                )
            )
        return tools
