import re

from tool.generate_random_integer import generate_random_integer


class TestGenerateRandomInteger:
    """Tests for the `generate_random_integer` tool"""

    def test_success_result_format(self) -> None:
        """Test the success result format"""
        result: str = generate_random_integer(0, 10)
        assert (
            '<random_integer_generation min="0" max="10">\n<result>' in result
            and "</result>\n</random_integer_generation>" in result
        )

    def test_is_inclusive(self) -> None:
        """Test if the range is inclusive"""
        result: str = generate_random_integer(5, 5)
        assert '<random_integer_generation min="5" max="5">\n<result>5</result>\n</random_integer_generation>' in result

    def test_error_min_greater_than_max(self) -> None:
        """Test error when min is greater than max"""
        result: str = generate_random_integer(10, 0)
        assert (
            '<random_integer_generation min="10" max="0">\n<error>"min" is greater than "max"</error>\n</random_integer_generation>'
            in result
        )
