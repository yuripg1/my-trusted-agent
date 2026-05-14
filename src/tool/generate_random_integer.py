from random import randint
from typing import Literal, TypedDict, Required

from tool.common import BaseToolCall


class GenerateRandomIntegerArguments(TypedDict):
    min: Required[int]
    max: Required[int]


class GenerateRandomIntegerToolCall(BaseToolCall):
    tool_name: Required[Literal["generate_random_integer"]]
    arguments: Required[GenerateRandomIntegerArguments]


def generate_random_integer(min: int, max: int) -> str:
    output_entries: list[str] = []
    if min > max:
        output_entries.append('<error>"min" is greater than "max"</error>')
    else:
        random_integer: int = randint(min, max)
        output_entries.append(f"<result>{random_integer}</result>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<random_integer_generation min="{min}" max="{max}">\n{joined_output_entries}\n</random_integer_generation>'
