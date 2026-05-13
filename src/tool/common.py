from typing import NotRequired, TypedDict


class BaseToolCall(TypedDict):
    id: NotRequired[str]
