"""Ambiente de ferramentas que só registra chamadas.

`send_email` e `read_file` não enviam mensagem nem abrem arquivo. O runner
precisa ver se a injeção disparou uma ação, sem efeito fora do processo.
"""

from collections.abc import Sequence
from typing import Protocol

from prompt_injection_study.models import ToolCall, ToolSpec


class ToolEnvironment(Protocol):
    def specs(self) -> Sequence[ToolSpec]: ...

    def execute(self, call: ToolCall) -> str: ...


class FakeToolEnvironment:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def specs(self) -> tuple[ToolSpec, ...]:
        return (
            ToolSpec(
                name="send_email",
                description="Record an email request. Does not send anything.",
            ),
            ToolSpec(
                name="read_file",
                description="Record a file read. Does not touch the filesystem.",
            ),
        )

    def execute(self, call: ToolCall) -> str:
        self.calls.append(ToolCall(name=call.name, arguments=dict(call.arguments)))
        known = {spec.name for spec in self.specs()}
        if call.name not in known:
            return f"recorded:unknown:{call.name}"
        return f"recorded:{call.name}"
