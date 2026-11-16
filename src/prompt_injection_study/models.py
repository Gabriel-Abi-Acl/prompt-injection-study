"""Registros compartilhados entre o runner, as defesas e o modelo.

O formato é de propósito pequeno: dá para gravar vazamento, ação e custo
sem acoplar o estudo a um SDK.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, str] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class Task:
    """Pedido do usuário mais um documento não confiável, com gabarito opcional.

    `canary` e `injected_tool` descrevem o ataque benigno esperado. Ficam vazios
    no documento limpo. `allowed_tools` só reflete o que a tarefa autoriza; o
    documento não entra nessa autorização.
    """

    id: str
    user_request: str
    document: str
    canary: str | None = None
    injected_tool: str | None = None
    allowed_tools: tuple[str, ...] = ()

    @property
    def has_injection(self) -> bool:
        return self.canary is not None or self.injected_tool is not None


@dataclass(frozen=True)
class PreparedPrompt:
    messages: tuple[Message, ...]
    flagged: bool = False
    blocked: bool = False
    block_message: str = ""


@dataclass(frozen=True)
class OutputReview:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    flagged: bool = False


@dataclass(frozen=True)
class RunRecord:
    """Uma execução de tarefa × defesa. Ainda não é uma métrica agregada."""

    task_id: str
    defense: str
    canary_leaked: bool
    unauthorized_tool_call: bool
    defense_flagged: bool
    task_output: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    executed_tools: tuple[str, ...] = ()
