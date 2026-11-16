"""Contrato das ferramentas falsas: registrar a chamada e não sair do processo."""

from prompt_injection_study.models import ToolCall
from prompt_injection_study.tools import FakeToolEnvironment


def test_stubs_record_calls_without_reading_the_filesystem(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-CONTENT")
    env = FakeToolEnvironment()
    arguments = {"path": str(secret), "to": "person@example.com"}

    read_result = env.execute(ToolCall("read_file", arguments))
    arguments["path"] = "mutated"
    mail_result = env.execute(ToolCall("send_email", {"to": "person@example.com"}))

    assert read_result == "recorded:read_file"
    assert mail_result == "recorded:send_email"
    assert "TOP-SECRET-CONTENT" not in read_result
    assert secret.read_text() == "TOP-SECRET-CONTENT"
    assert env.calls[0].arguments["path"] == str(secret)
    assert {spec.name for spec in env.specs()} == {"send_email", "read_file"}


def test_unknown_tool_is_recorded_and_labeled():
    env = FakeToolEnvironment()
    result = env.execute(ToolCall("shell", {"cmd": "uname"}))

    assert result == "recorded:unknown:shell"
    assert env.calls[0].name == "shell"
    assert env.calls[0].arguments == {"cmd": "uname"}
