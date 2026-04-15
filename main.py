import requests
import typing

ModelType = typing.Literal["deepseek-chat", "deepseek-reasoner"]
RoleType = typing.Literal["assistant", "user", "system"]


class DeepSeekMessage(typing.TypedDict):
    role: RoleType
    content: str
    reasoning_content: str | None


class DeepSeekRequest(typing.TypedDict):
    model: ModelType
    messages: list[DeepSeekMessage]
    max_tokens: int
    stream: bool
    temperature: float


API_KEY: str = ""
BASE_URL: str = "https://api.deepseek.com"

MODEL: ModelType = "deepseek-chat"
MAX_TOKENS: int = 8192

# MODEL: ModelType = "deepseek-reasoner"
# MAX_TOKENS: int = 65536

# TEMPERATURE: float = 0.0
TEMPERATURE: float = 1.0
# TEMPERATURE: float = 1.3
# TEMPERATURE: float = 1.5

SYSTEM_MESSAGE: str = ""


def get_llm_output(
    llm_base_url: str,
    llm_api_key: str,
    llm_model: ModelType,
    llm_messages: list[DeepSeekMessage],
    llm_max_tokens: int,
    llm_temperature: float,
) -> tuple[str, str]:
    payload: DeepSeekRequest = {
        "model": llm_model,
        "messages": llm_messages,
        "max_tokens": llm_max_tokens,
        "stream": False,
        "temperature": llm_temperature,
    }
    headers = {"Authorization": f"Bearer {llm_api_key}", "Content-Type": "application/json"}
    response = requests.post(f"{llm_base_url}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    content = message.get("content", "").strip()
    reasoning_content = message.get("reasoning_content", "").strip()
    if not content and reasoning_content:
        content = reasoning_content
        reasoning_content = ""
    return content, reasoning_content


def add_to_llm_messages(
    llm_messages: list[DeepSeekMessage], role: RoleType, content: str, reasoning_content: str = ""
) -> None:
    trimmed_content: str = content.strip()
    if len(trimmed_content) == 0:
        return

    trimmed_reasoning_content: str = reasoning_content.strip()
    if role in ["system", "user"]:
        new_message: DeepSeekMessage = {"role": role, "content": trimmed_content, "reasoning_content": None}
        llm_messages.append(new_message)
    elif role == "assistant":
        new_assistant_message: DeepSeekMessage = {
            "role": role,
            "content": trimmed_content,
            "reasoning_content": trimmed_reasoning_content if (len(trimmed_reasoning_content) != 0) else None,
        }
        llm_messages.append(new_assistant_message)


def create_llm_messages(llm_system_message: str) -> list[DeepSeekMessage]:
    llm_messages: list[DeepSeekMessage] = []
    trimmed_llm_system_message: str = llm_system_message.strip()
    if len(trimmed_llm_system_message) != 0:
        add_to_llm_messages(llm_messages, "system", trimmed_llm_system_message)
    return llm_messages


def main() -> None:
    llm_messages = create_llm_messages(SYSTEM_MESSAGE)
    try:
        while True:
            llm_input = input("> ").strip()
            if len(llm_input) == 0 and (len(llm_messages) != 1 or llm_messages[0]["role"] != "system"):
                continue
            add_to_llm_messages(llm_messages, "user", llm_input)
            llm_output, llm_output_reasoning = get_llm_output(
                BASE_URL, API_KEY, MODEL, llm_messages, MAX_TOKENS, TEMPERATURE
            )
            if len(llm_output) != 0:
                add_to_llm_messages(llm_messages, "assistant", llm_output, llm_output_reasoning)
            print(
                f"\n--------------------------------------------------------------------------------\n{llm_output}\n--------------------------------------------------------------------------------\n"
            )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
