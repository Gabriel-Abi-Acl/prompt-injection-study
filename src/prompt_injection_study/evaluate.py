"""Comando offline que mede as defesas no dataset v1.

O fake é o padrão e não usa rede. Provedor real só passa do portão com
--allow-live, chave e modelo; sem isso o processo sai com código 2 e não
chama ninguém. Número do fake confere o encanamento, não vira achado.
"""

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from prompt_injection_study.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    Settings,
)
from prompt_injection_study.dataset import EVAL_CATEGORIES, DatasetError, EvalTask, load_eval_tasks
from prompt_injection_study.defenses import DEFENSE_NAMES, build_defense
from prompt_injection_study.llm import LLM, FakePolicy, ScriptedFakeLLM, build_llm
from prompt_injection_study.metrics import (
    ScoredTask,
    executed_forbidden_tool,
    score_completion,
    summarize_metrics,
)
from prompt_injection_study.models import Task
from prompt_injection_study.runner import run_task
from prompt_injection_study.tools import FakeToolEnvironment

SCHEMA_VERSION = 1
RECORDS_FILENAME = "records.jsonl"
SUMMARY_FILENAME = "summary.json"
DEFAULT_OUTPUT_DIR = Path("outputs/eval")
PROVIDERS: tuple[str, ...] = ("fake", "openai", "anthropic")
_LIVE_ENV = {
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL"),
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"),
}

RECORD_KEYS: tuple[str, ...] = (
    "schema_version",
    "task_id",
    "defense",
    "category",
    "source",
    "is_attack",
    "canary_leaked",
    "forbidden_tool_executed",
    "unauthorized_tool_call",
    "defense_flagged",
    "refused",
    "task_output",
    "reference_output",
    "completion_checkable",
    "benign_completed",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "executed_tools",
    "error",
)

SUMMARY_KEYS: tuple[str, ...] = (
    "schema_version",
    "pipeline_check",
    "dataset",
    "provider",
    "model",
    "fake_policy",
    "defenses",
    "task_count",
    "record_count",
    "error_count",
    "by_defense",
)


class EvaluationError(Exception):
    """Uso inválido da avaliação. A mensagem vai para stderr e o código é 2."""


@dataclass(frozen=True)
class EvaluationRun:
    records: tuple[dict[str, object], ...]
    scored: tuple[ScoredTask, ...]


def resolve_provider(explicit: str | None, environ: Mapping[str, str]) -> str:
    """`--provider` ganha de `LLM_PROVIDER`. Vazio cai no fake."""

    chosen = explicit if explicit is not None else environ.get("LLM_PROVIDER", "")
    return chosen.strip().lower() or "fake"


def require_live_model(
    provider: str,
    model_flag: str | None,
    environ: Mapping[str, str],
) -> str:
    """Exige chave no ambiente e modelo no ambiente ou em `--model`.

    Não constrói cliente. Chamador só chega aqui depois de `--allow-live`.
    """

    key_var, model_var = _LIVE_ENV[provider]
    key = environ.get(key_var, "").strip()
    model = (model_flag or "").strip() or environ.get(model_var, "").strip()
    missing: list[str] = []
    if not key:
        missing.append(key_var)
    if not model:
        missing.append(model_var)
    if missing:
        if len(missing) == 1:
            listed = missing[0]
        else:
            listed = f"{missing[0]} and {missing[1]}"
        raise EvaluationError(
            f"Live provider {provider} needs {listed} in the environment. No request was sent."
        )
    return model


def evaluate_defenses(
    tasks: Sequence[EvalTask],
    defenses: Sequence[str],
    llm_factory: Callable[[], LLM],
    *,
    secrets: Sequence[str] = (),
) -> EvaluationRun:
    """Roda cada defesa em cada tarefa. Erro de uma linha não interrompe as outras.

    A fábrica é chamada uma vez por tarefa, para um cliente não guardar estado
    da anterior. Exceção vira campo `error` e sai dos denominadores.
    """

    _validate_defenses(defenses)
    records: list[dict[str, object]] = []
    scored: list[ScoredTask] = []
    for defense_name in defenses:
        for task in tasks:
            payload, row = _run_one(task, defense_name, llm_factory, secrets)
            records.append(payload)
            scored.append(row)
    return EvaluationRun(tuple(records), tuple(scored))


def build_summary(
    scored: Sequence[ScoredTask],
    *,
    dataset_name: str,
    provider: str,
    model: str | None,
    fake_policy: str | None,
    defenses: Sequence[str],
    task_count: int,
) -> dict[str, object]:
    """Monta o JSON agregado. `pipeline_check` marca o fake, para não virar achado."""

    by_defense = summarize_metrics(
        scored,
        defense_order=DEFENSE_NAMES,
        category_order=EVAL_CATEGORIES,
    )
    error_count = sum(1 for row in scored if row.error is not None)
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_check": provider == "fake",
        "dataset": dataset_name,
        "provider": provider,
        "model": model,
        "fake_policy": fake_policy if provider == "fake" else None,
        "defenses": list(defenses),
        "task_count": task_count,
        "record_count": len(scored),
        "error_count": error_count,
        "by_defense": by_defense,
    }


def write_outputs(
    output_dir: Path,
    records: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    """Grava `records.jsonl` e `summary.json` só dentro do diretório pedido."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvaluationError(f"Could not create output directory: {exc}") from exc
    if not output_dir.is_dir():
        raise EvaluationError(f"Output path is not a directory: {output_dir}")
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records]
    body = ("\n".join(lines) + "\n") if lines else ""
    (output_dir / RECORDS_FILENAME).write_text(body, encoding="utf-8")
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada do script e de `python -m prompt_injection_study.evaluate`."""

    args = _parser().parse_args(None if argv is None else list(argv))
    try:
        output_dir, record_count, error_count = run_cli(args, os.environ)
    except (EvaluationError, DatasetError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"wrote {record_count} records ({error_count} transport errors) to {output_dir}",
    )
    return 0


def run_cli(args: argparse.Namespace, environ: Mapping[str, str]) -> tuple[Path, int, int]:
    """Resolve o backend, mede e grava. O portão ao vivo corre antes de qualquer cliente."""

    provider = resolve_provider(args.provider, environ)
    if provider not in PROVIDERS:
        known = ", ".join(PROVIDERS)
        raise EvaluationError(f"Unknown provider: {provider}. Known: {known}. No request was sent.")

    model: str | None = None
    if provider != "fake":
        # Sem o flag, a chave nem chega a ser lida para montar um cliente.
        if not args.allow_live:
            raise EvaluationError(
                f"Live provider {provider} requires --allow-live. No request was sent."
            )
        if args.fake_policy is not None:
            raise EvaluationError(
                "--fake-policy applies only to the scripted fake. No request was sent."
            )
        model = require_live_model(provider, args.model, environ)

    policy = args.fake_policy or "ignore"
    tasks = load_eval_tasks(args.dataset)
    defenses = DEFENSE_NAMES if args.defense == "all" else (args.defense,)
    outcome = evaluate_defenses(
        tasks,
        defenses,
        _llm_factory(provider, policy, model, environ),
        secrets=_secrets(environ),
    )
    dataset_name = args.dataset.name if args.dataset is not None else "tasks.v1.jsonl"
    summary = build_summary(
        outcome.scored,
        dataset_name=dataset_name,
        provider=provider,
        model=model,
        fake_policy=policy,
        defenses=defenses,
        task_count=len(tasks),
    )
    output_dir = args.output_dir
    write_outputs(output_dir, outcome.records, summary)
    return output_dir, len(outcome.records), int(summary["error_count"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-defenses",
        description=(
            "Run the v1 tasks under one defense or all of them. "
            "The default backend is the offline scripted fake."
        ),
    )
    parser.add_argument(
        "--defense",
        default="all",
        choices=(*DEFENSE_NAMES, "all"),
        help="One defense, or all five. Default: all.",
    )
    parser.add_argument(
        "--fake-policy",
        default=None,
        choices=["ignore", "obey", "obey_outside_delimiters"],
        help="Scripted fake behavior. Default: ignore. Not valid with a live provider.",
    )
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="Backend. Default: LLM_PROVIDER, or fake when that variable is unset.",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Permit a live provider. Requires the API key and model in the environment.",
    )
    parser.add_argument(
        "--model",
        help=("Live model id. Overrides the model variable. The key stays in the environment."),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Eval JSONL path. Default: data/eval/tasks.v1.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for records.jsonl and summary.json. Default: outputs/eval.",
    )
    return parser


def _llm_factory(
    provider: str,
    policy: str,
    model: str | None,
    environ: Mapping[str, str],
) -> Callable[[], LLM]:
    settings: Settings | None = None
    if provider != "fake":
        if model is None:
            raise EvaluationError(f"Live provider {provider} has no model. No request was sent.")
        settings = Settings(
            llm_provider=provider,
            openai_api_key=environ.get("OPENAI_API_KEY", "").strip(),
            anthropic_api_key=environ.get("ANTHROPIC_API_KEY", "").strip(),
            openai_model=model if provider == "openai" else DEFAULT_OPENAI_MODEL,
            anthropic_model=model if provider == "anthropic" else DEFAULT_ANTHROPIC_MODEL,
        )

    def factory() -> LLM:
        if settings is None:
            return ScriptedFakeLLM(FakePolicy(policy))
        return build_llm(settings)

    return factory


def _validate_defenses(defenses: Sequence[str]) -> None:
    if not defenses:
        raise EvaluationError("No defense selected.")
    unknown = [name for name in defenses if name not in DEFENSE_NAMES]
    if unknown:
        known = ", ".join(DEFENSE_NAMES)
        raise EvaluationError(f"Unknown defense: {unknown[0]}. Known: {known}.")
    if len(set(defenses)) != len(tuple(defenses)):
        raise EvaluationError("Defense selection repeats a name.")


def _run_one(
    task: EvalTask,
    defense_name: str,
    llm_factory: Callable[[], LLM],
    secrets: Sequence[str],
) -> tuple[dict[str, object], ScoredTask]:
    # O campo Task só cabe uma ferramenta. O gabarito pode ter duas; a conta
    # usa a lista inteira, não esse atalho.
    injected = task.forbidden_tool_calls[0] if len(task.forbidden_tool_calls) == 1 else None
    model_task = Task(
        id=task.id,
        user_request=task.user_request,
        document=task.document,
        canary=task.canary_id,
        injected_tool=injected,
        allowed_tools=task.allowed_tools,
    )
    try:
        record = run_task(
            model_task,
            build_defense(defense_name),
            llm_factory(),
            FakeToolEnvironment(),
        )
    except Exception as exc:
        return _error_row(task, defense_name, exc, secrets)

    forbidden = executed_forbidden_tool(record.executed_tools, task.forbidden_tool_calls)
    checkable, completed = score_completion(
        is_attack=task.is_attack,
        errored=False,
        reference_output=task.reference_output,
        task_output=record.task_output,
    )
    latency = float(round(record.latency_ms, 3))
    payload = _record_payload(
        task,
        defense_name,
        canary_leaked=record.canary_leaked,
        forbidden_tool_executed=forbidden,
        unauthorized_tool_call=record.unauthorized_tool_call,
        defense_flagged=record.defense_flagged,
        refused=record.blocked,
        task_output=record.task_output,
        completion_checkable=checkable,
        benign_completed=completed,
        latency_ms=latency,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cost_usd=record.cost_usd,
        executed_tools=list(record.executed_tools),
        error=None,
    )
    scored = ScoredTask(
        defense=defense_name,
        category=task.category,
        is_attack=task.is_attack,
        canary_leaked=record.canary_leaked,
        forbidden_tool_executed=forbidden,
        defense_flagged=record.defense_flagged,
        refused=record.blocked,
        completion_checkable=checkable,
        benign_completed=completed,
        latency_ms=latency,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cost_usd=record.cost_usd,
        error=None,
    )
    return payload, scored


def _error_row(
    task: EvalTask,
    defense_name: str,
    exc: Exception,
    secrets: Sequence[str],
) -> tuple[dict[str, object], ScoredTask]:
    # Sinais não observados ficam nulos no JSON. A conta ignora a linha pelo erro,
    # então um canário que a exceção mencionasse não vira sucesso.
    error = _public_error(exc, secrets)
    payload = _record_payload(
        task,
        defense_name,
        canary_leaked=None,
        forbidden_tool_executed=None,
        unauthorized_tool_call=None,
        defense_flagged=None,
        refused=None,
        task_output=None,
        completion_checkable=False,
        benign_completed=None,
        latency_ms=None,
        input_tokens=None,
        output_tokens=None,
        cost_usd=None,
        executed_tools=[],
        error=error,
    )
    scored = ScoredTask(
        defense=defense_name,
        category=task.category,
        is_attack=task.is_attack,
        canary_leaked=False,
        forbidden_tool_executed=False,
        defense_flagged=False,
        refused=False,
        completion_checkable=False,
        benign_completed=None,
        latency_ms=None,
        input_tokens=None,
        output_tokens=None,
        cost_usd=None,
        error=error["type"],
    )
    return payload, scored


def _record_payload(
    task: EvalTask,
    defense_name: str,
    **fields: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task.id,
        "defense": defense_name,
        "category": task.category,
        "source": task.source,
        "is_attack": task.is_attack,
        "reference_output": task.reference_output,
    }
    payload.update(fields)
    # A ordem publicada é a de RECORD_KEYS, não a ordem em que os campos foram preenchidos.
    ordered = {key: payload[key] for key in RECORD_KEYS}
    return ordered


def _public_error(exc: Exception, secrets: Sequence[str]) -> dict[str, str]:
    message = " ".join(str(exc).split())
    for secret in secrets:
        if len(secret) >= 8 and secret in message:
            message = message.replace(secret, "[redacted]")
    if len(message) > 500:
        message = message[:500]
    return {"type": type(exc).__name__, "message": message}


def _secrets(environ: Mapping[str, str]) -> tuple[str, ...]:
    found: list[str] = []
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        value = environ.get(name, "").strip()
        if len(value) >= 8:
            found.append(value)
    return tuple(found)


if __name__ == "__main__":
    raise SystemExit(main())
