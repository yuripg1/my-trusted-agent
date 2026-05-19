from random import randint

from tool.generate_random_integer import (
    GenerateRandomIntegerToolCall,
    generate_random_integer,
    get_generate_random_integer_message,
    get_generate_random_integer_permission,
)


class TestGetGenerateRandomIntegerMessage:
    """Tests for the `get_generate_random_integer_message` function"""

    def test_format(self) -> None:
        """Format the message correctly"""
        tool_call: GenerateRandomIntegerToolCall = {
            "tool_name": "generate_random_integer",
            "arguments": {"min": 1, "max": 10},
        }
        assert get_generate_random_integer_message(tool_call) == "Generating a random integer between **1** and **10**"


class TestGetGenerateRandomIntegerPermission:
    """Tests for the `get_generate_random_integer_permission` function"""

    def test_auto_approved(self) -> None:
        """Permission should be automatically granted"""
        tool_call: GenerateRandomIntegerToolCall = {
            "tool_name": "generate_random_integer",
            "arguments": {"min": 1, "max": 10},
        }
        assert get_generate_random_integer_permission(tool_call) is True


class TestGenerateRandomInteger:
    """Tests for the `generate_random_integer` tool"""

    def test_success(self) -> None:
        """Generate a random integer"""
        min: int = randint(11, 19)
        max: int = randint(31, 39)
        result: str = generate_random_integer(min, max)
        assert result.startswith(f'<random_integer_generation min="{min}" max="{max}">\n<result>')
        assert result.endswith("</result>\n</random_integer_generation>")
        generated_integer: int = int(result.split("<result>")[1].split("</result>")[0])
        assert generated_integer >= min
        assert generated_integer <= max

    def test_is_inclusive(self) -> None:
        """Assert that the range is inclusive with equal values for 'min' and 'max'"""
        min_and_max: int = randint(11, 19)
        result: str = generate_random_integer(min_and_max, min_and_max)
        assert (
            result
            == f'<random_integer_generation min="{min_and_max}" max="{min_and_max}">\n<result>{min_and_max}</result>\n</random_integer_generation>'
        )

    def test_error_min_greater_than_max(self) -> None:
        """Do not generate a random integer when 'min' is greater than 'max'"""
        min: int = randint(31, 39)
        max: int = randint(11, 19)
        result: str = generate_random_integer(min, max)
        assert (
            result
            == f'<random_integer_generation min="{min}" max="{max}">\n<error>"max" must be greater than or equal to "min"</error>\n</random_integer_generation>'
        )
