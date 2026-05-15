from tool.execute_shell_command import execute_shell_command


class TestExecuteShellCommand:
    """Tests for the `execute_shell_command` tool"""

    def test_successful_command(self) -> None:
        """Execute a command that succeeds"""
        stdout_text: str = "stdout stdout_text"
        command: str = f'echo "{stdout_text}"'
        result: str = execute_shell_command(command)
        exit_code: int = 0
        assert (
            result
            == f"<shell_command_execution>\n<command>\n{command}\n</command>\n<stdout>\n{stdout_text}\n</stdout>\n<exit_code>{exit_code}</exit_code>\n</shell_command_execution>"
        )

    def test_stderr_output(self) -> None:
        """Execute a command that produces stderr"""
        stderr_text: str = "stderr message"
        command: str = f'echo "{stderr_text}" >&2'
        result: str = execute_shell_command(command)
        exit_code: int = 0
        assert (
            result
            == f"<shell_command_execution>\n<command>\n{command}\n</command>\n<stderr>\n{stderr_text}\n</stderr>\n<exit_code>{exit_code}</exit_code>\n</shell_command_execution>"
        )

    def test_non_zero_exit_code(self) -> None:
        """Execute a command that fails"""
        command: str = "false"
        result: str = execute_shell_command(command)
        exit_code: int = 1
        assert (
            result
            == f"<shell_command_execution>\n<command>\n{command}\n</command>\n<exit_code>{exit_code}</exit_code>\n</shell_command_execution>"
        )

    def test_command_denied_by_user(self) -> None:
        """Do not execute a command due to being denied by the user"""
        command: str = 'echo "test"'
        result: str = execute_shell_command(command, tool_call_permission=False)
        assert (
            result
            == f"<shell_command_execution>\n<command>\n{command}\n</command>\n<error>Bash command execution manually denied by the user</error>\n</shell_command_execution>"
        )
