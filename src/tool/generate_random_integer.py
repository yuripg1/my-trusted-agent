from random import randint
from typing import Literal, Required, TypedDict

from tool.common import BaseToolCall


class GenerateRandomIntegerArguments(TypedDict):
    min: Required[int]
    max: Required[int]


class GenerateRandomIntegerToolCall(BaseToolCall):
    tool_name: Required[Literal["generate_random_integer"]]
    arguments: Required[GenerateRandomIntegerArguments]


def get_generate_random_integer_message(arguments: GenerateRandomIntegerArguments) -> str:
    return f"Generating a random integer between **{arguments['min']}** and **{arguments['max']}**"


def get_generate_random_integer_permission(arguments: GenerateRandomIntegerArguments) -> bool:
    return True


def generate_random_integer(arguments: GenerateRandomIntegerArguments) -> str:
    output_entries: list[str] = []
    if arguments["min"] > arguments["max"]:
        output_entries.append('<error>"max" must be greater than or equal to "min"</error>')
    else:
        random_integer: int = randint(arguments["min"], arguments["max"])
        output_entries.append(f"<result>{random_integer}</result>")
    joined_output_entries: str = "\n".join(output_entries)
    return f'<random_integer_generation min="{arguments["min"]}" max="{arguments["max"]}">\n{joined_output_entries}\n</random_integer_generation>'
