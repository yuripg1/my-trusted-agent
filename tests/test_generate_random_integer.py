from random import randint

from tool.generate_random_integer import generate_random_integer


class TestGenerateRandomInteger:
    """Tests for the `generate_random_integer` tool"""

    def test_success_result_format(self) -> None:
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
            == f'<random_integer_generation min="{min}" max="{max}">\n<error>"min" is greater than "max"</error>\n</random_integer_generation>'
        )
