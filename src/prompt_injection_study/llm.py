"""Interface de modelo independente de provedor, com um fake determinístico.

Os SDKs reais são importados só na primeira chamada ao vivo. Teste e CI
rodam com `ScriptedFakeLLM`, sem rede e sem chave.
"""

import json
import time
from collections.abc import Sequence
from enum import Enum
from typing import Protocol

from prompt_injection_study.config import Settings, load_settings
from prompt_injection_study.heuristics import (
    CANARY_INSTRUCTION,
    FAKE_TOOL_INSTRUCTION,
    injection_lines,
)
from prompt_injection_study.markers import outside_untrusted
from prompt_injection_study.models import LLMResponse, Message, ToolCall, ToolSpec


class LLMConfigError(RuntimeError):
    """Provedor desconhecido ou credencial vazia antes de qualquer chamada de rede."""


class FakePolicy(Enum):
    """Como o fake trata instrução embutida no prompt.

    `OBEY_OUTSIDE_DELIMITERS` obedece só o que ficou fora do bloco delimitado,
    para exercitar spotlight sem afirmar que todo modelo respeita a fronteira.
    """

    OBEY = "obey"
    IGNORE = "ignore"
    OBEY_OUTSIDE_DELIMITERS = "obey_outside_delimiters"


class LLM(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse: ...


class ScriptedFakeLLM:
    """Modelo offline configurável.

    Sem texto scriptado, resume o pedido e só emite canário ou ferramenta se a
    política mandar obedecer a uma instrução explícita. Menção solta a
    `CANARY-n` não conta como obediência.
    """

    def __init__(
        self,
        policy: FakePolicy = FakePolicy.IGNORE,
        *,
        scripted_text: str | None = None,
        scripted_tool_calls: tuple[ToolCall, ...] | None = None,
        delay_s: float = 0.0,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
    ) -> None:
        self.policy = policy
        self.scripted_text = scripted_text
        self.scripted_tool_calls = scripted_tool_calls
        self.delay_s = delay_s
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.calls = 0

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        # O catálogo não filtra o fake; quem autoriza a ferramenta é a defesa.
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.scripted_text is not None or self.scripted_tool_calls is not None:
            return self._response(
                self.scripted_text or "",
                self.scripted_tool_calls or (),
            )

        # Juiz: só o prompt que nós montamos, pelo prefixo. Texto igual dentro
        # do documento não pode sequestrar o modelo da tarefa.
        if _is_judge_prompt(messages):
            document = messages[0].content.split("Document:\n", 1)[1]
            verdict = "FLAGGED" if injection_lines(document) else "CLEAN"
            return self._response(verdict, ())

        blob = "\n".join(message.content for message in messages)

        region = _obedience_region(blob, self.policy)
        canaries = list(dict.fromkeys(match[1] for match in CANARY_INSTRUCTION.findall(region)))
        tool_names = list(dict.fromkeys(FAKE_TOOL_INSTRUCTION.findall(region)))
        text = f"Summary: {_excerpt(messages)}"
        if canaries:
            text = f"{text} {' '.join(canaries)}"
        tool_calls = tuple(
            ToolCall(name=name, arguments={"note": "recorded"}) for name in tool_names
        )
        return self._response(text, tool_calls)

    def _response(self, text: str, tool_calls: tuple[ToolCall, ...]) -> LLMResponse:
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=self.cost_usd,
        )


def _is_judge_prompt(messages: Sequence[Message]) -> bool:
    if len(messages) != 1 or messages[0].role != "user":
        return False
    content = messages[0].content
    return content.startswith("You are a prompt-injection judge") and "Document:\n" in content


def _obedience_region(blob: str, policy: FakePolicy) -> str:
    if policy is FakePolicy.IGNORE:
        return ""
    if policy is FakePolicy.OBEY_OUTSIDE_DELIMITERS:
        return outside_untrusted(blob)
    return blob


def _excerpt(messages: Sequence[Message]) -> str:
    """Trecho estável do pedido, sem linhas de injeção e sem token de canário.

    O resumo não pode vazar o canário por acidente de cópia. Obediência é um
    acréscimo separado, controlado pela política.
    """

    user = next((message.content for message in reversed(messages) if message.role == "user"), "")
    kept: list[str] = []
    for line in user.splitlines():
        if injection_lines(line):
            continue
        if "CANARY-" in line:
            continue
        kept.append(line)
    compact = " ".join(" ".join(kept).split())
    return compact[:160]


class OpenAIChatLLM:
    """Cliente OpenAI. O pacote `openai` entra só dentro de `complete`."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = load_settings()
        self.api_key = settings.openai_api_key if api_key is None else api_key
        self.model = settings.openai_model if model is None else model

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMConfigError("OPENAI_API_KEY is empty")
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        kwargs: dict[str, object] = {}
        if tools:
            kwargs["tools"] = [_openai_tool(tool) for tool in tools]
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": message.role, "content": message.content} for message in messages],
            **kwargs,
        )
        choice = completion.choices[0].message
        parsed_calls: list[ToolCall] = []
        for call in choice.tool_calls or []:
            parsed_calls.append(
                ToolCall(name=call.function.name, arguments=_json_object(call.function.arguments))
            )
        usage = completion.usage
        return LLMResponse(
            text=choice.content or "",
            tool_calls=tuple(parsed_calls),
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            cost_usd=None,
        )


class AnthropicChatLLM:
    """Cliente Anthropic. O pacote `anthropic` entra só dentro de `complete`."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        settings = load_settings()
        self.api_key = settings.anthropic_api_key if api_key is None else api_key
        self.model = settings.anthropic_model if model is None else model

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMConfigError("ANTHROPIC_API_KEY is empty")
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        system = "\n".join(message.content for message in messages if message.role == "system")
        dialogue = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role in {"user", "assistant"}
        ]
        if not dialogue:
            dialogue = [{"role": "user", "content": ""}]
        kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": dialogue,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [_anthropic_tool(tool) for tool in tools]
        response = client.messages.create(**kwargs)
        text_parts: list[str] = []
        parsed_calls: list[ToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                raw = block.input if isinstance(block.input, dict) else {}
                parsed_calls.append(
                    ToolCall(
                        name=block.name,
                        arguments={str(key): str(value) for key, value in raw.items()},
                    )
                )
        usage = response.usage
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tuple(parsed_calls),
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            cost_usd=None,
        )


def build_llm(settings: Settings | None = None) -> LLM:
    """Monta o modelo pedido por `LLM_PROVIDER`. Padrão: fake offline."""

    resolved = settings or load_settings()
    if resolved.llm_provider == "fake":
        return ScriptedFakeLLM()
    if resolved.llm_provider == "openai":
        return OpenAIChatLLM(api_key=resolved.openai_api_key, model=resolved.openai_model)
    if resolved.llm_provider == "anthropic":
        return AnthropicChatLLM(api_key=resolved.anthropic_api_key, model=resolved.anthropic_model)
    raise LLMConfigError(f"Unknown LLM_PROVIDER: {resolved.llm_provider}")


def _openai_tool(tool: ToolSpec) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {"type": "object", "properties": {}, "additionalProperties": True},
        },
    }


def _anthropic_tool(tool: ToolSpec) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": True},
    }


def _json_object(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): str(value) for key, value in parsed.items()}
