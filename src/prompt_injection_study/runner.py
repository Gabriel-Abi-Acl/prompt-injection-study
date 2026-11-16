"""Executa uma tarefa sob uma defesa e grava o que aconteceu.

A latência cobre preparação, modelo e ferramentas. Tokens e custo só entram
quando a resposta do modelo os traz. Bloqueio antes do modelo deixa esses
campos vazios: não há uso a inventar.
"""

import time
from collections.abc import Callable, Sequence

from prompt_injection_study.defenses import Defense
from prompt_injection_study.detection import canary_leaked, unauthorized_tool_names
from prompt_injection_study.llm import LLM
from prompt_injection_study.models import RunRecord, Task
from prompt_injection_study.tools import FakeToolEnvironment, ToolEnvironment


def run_task(task: Task, defense: Defense, llm: LLM, tools: ToolEnvironment) -> RunRecord:
    started = time.perf_counter()
    prepared = defense.prepare(task)
    flagged = prepared.flagged
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    executed: list[str] = []

    if prepared.blocked:
        output = prepared.block_message
    else:
        response = llm.complete(prepared.messages, tools.specs())
        review = defense.review(task, response)
        flagged = flagged or review.flagged
        for call in review.tool_calls:
            tools.execute(call)
            executed.append(call.name)
        output = review.text
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
        cost_usd = response.cost_usd

    elapsed_ms = (time.perf_counter() - started) * 1000
    return RunRecord(
        task_id=task.id,
        defense=defense.name,
        canary_leaked=canary_leaked(output, task.canary),
        unauthorized_tool_call=bool(unauthorized_tool_names(executed, task)),
        defense_flagged=flagged,
        task_output=output,
        latency_ms=elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        executed_tools=tuple(executed),
    )


def run_suite(
    tasks: Sequence[Task],
    defense_factory: Callable[[], Defense],
    llm_factory: Callable[[], LLM],
    tool_factory: Callable[[], ToolEnvironment] | None = None,
) -> tuple[RunRecord, ...]:
    """Uma ferramenta nova por tarefa, para uma chamada não contaminar a outra."""

    make_tools = tool_factory or FakeToolEnvironment
    return tuple(run_task(task, defense_factory(), llm_factory(), make_tools()) for task in tasks)
