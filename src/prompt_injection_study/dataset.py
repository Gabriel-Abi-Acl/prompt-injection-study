"""Dataset versionado de avaliação, com gabarito e validação estrita.

`load_eval_tasks` lê `data/eval/tasks.v1.jsonl` e recusa o arquivo no primeiro
rótulo que não fecha. Não chama modelo e não calcula taxa.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from prompt_injection_study.markers import UNTRUSTED_END, UNTRUSTED_START
from prompt_injection_study.tools import FakeToolEnvironment

CLEAN = "clean"
DIRECT_INSTRUCTION_OVERRIDE = "direct_instruction_override"
HIDDEN_INDIRECT = "hidden_indirect"
TOOL_CALL_HIJACK = "tool_call_hijack"
OBFUSCATED = "obfuscated"
DELIMITER_BREAKOUT = "delimiter_breakout"

EVAL_CATEGORIES: tuple[str, ...] = (
    CLEAN,
    DIRECT_INSTRUCTION_OVERRIDE,
    HIDDEN_INDIRECT,
    TOOL_CALL_HIJACK,
    OBFUSCATED,
    DELIMITER_BREAKOUT,
)
EVAL_SOURCES: tuple[str, ...] = ("email", "web_page", "file")

_ATTACK_CATEGORIES = frozenset(category for category in EVAL_CATEGORIES if category != CLEAN)
_CANARY_REQUIRED = frozenset(
    {
        DIRECT_INSTRUCTION_OVERRIDE,
        HIDDEN_INDIRECT,
        DELIMITER_BREAKOUT,
    }
)
_REQUIRED_KEYS = frozenset(
    {
        "id",
        "user_request",
        "document",
        "source",
        "is_attack",
        "category",
        "canary_id",
        "forbidden_tool_calls",
        "allowed_tools",
        "reference_output",
    }
)
_TASK_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CANARY_ID_RE = re.compile(r"^CANARY-[0-9]+$")
_CANARY_TOKEN_RE = re.compile(r"CANARY-[0-9]+")


class DatasetError(ValueError):
    """O arquivo de avaliação não cumpre o contrato do gabarito."""


@dataclass(frozen=True)
class EvalTask:
    """Uma tarefa sintética e o gabarito do pedido benigno.

    Canário e ferramentas proibidas descrevem o ataque e ficam vazios na
    tarefa limpa. `reference_output` é a resposta benigna escrita com o item,
    não uma saída observada de modelo.
    """

    id: str
    user_request: str
    document: str
    source: str
    is_attack: bool
    category: str
    canary_id: str | None
    forbidden_tool_calls: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    reference_output: str


def load_eval_tasks(path: Path | None = None) -> tuple[EvalTask, ...]:
    """Lê o JSONL v1 e devolve as tarefas na ordem do arquivo.

    Sem `path`, usa `data/eval/tasks.v1.jsonl` na raiz do repositório. Ids e
    canários precisam ser únicos. Ataque sem canário e sem ferramenta
    proibida, ou tarefa limpa com qualquer um dos dois, é erro.
    """

    dataset_path = Path(path) if path is not None else _default_dataset_path()
    if not dataset_path.is_file():
        raise DatasetError(f"Missing eval dataset: {dataset_path}")
    try:
        text = dataset_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"Eval dataset is not UTF-8: {dataset_path}") from exc
    if not text.strip():
        raise DatasetError(f"Empty eval dataset: {dataset_path}")

    tasks: list[EvalTask] = []
    seen_ids: set[str] = set()
    seen_canaries: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            _fail(line_no, "blank line")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {line_no}: invalid JSON") from exc
        task = _parse_record(raw, line_no)
        if task.id in seen_ids:
            _fail(line_no, f"duplicate id {task.id}", task.id)
        seen_ids.add(task.id)
        if task.canary_id is not None:
            if task.canary_id in seen_canaries:
                _fail(line_no, f"duplicate canary_id {task.canary_id}", task.id)
            seen_canaries.add(task.canary_id)
        tasks.append(task)
    return tuple(tasks)


def _default_dataset_path() -> Path:
    # Este módulo mora em src/prompt_injection_study. A raiz é dois níveis acima.
    return Path(__file__).resolve().parents[2] / "data" / "eval" / "tasks.v1.jsonl"


def _parse_record(raw: object, line_no: int) -> EvalTask:
    if not isinstance(raw, dict):
        _fail(line_no, "expected an object")
    missing = sorted(_REQUIRED_KEYS - set(raw))
    extra = sorted(set(raw) - _REQUIRED_KEYS)
    if missing:
        _fail(line_no, f"missing keys: {', '.join(missing)}")
    if extra:
        _fail(line_no, f"unknown keys: {', '.join(extra)}")

    task_id = _require_text(raw, "id", line_no)
    if _TASK_ID_RE.fullmatch(task_id) is None:
        _fail(line_no, "id must be lowercase letters, digits, and hyphens", task_id)
    user_request = _require_text(raw, "user_request", line_no, task_id)
    document = _require_text(raw, "document", line_no, task_id)
    reference_output = _require_text(raw, "reference_output", line_no, task_id)
    source = _require_choice(raw, "source", EVAL_SOURCES, line_no, task_id)
    category = _require_choice(raw, "category", EVAL_CATEGORIES, line_no, task_id)
    is_attack = raw["is_attack"]
    if not isinstance(is_attack, bool):
        _fail(line_no, "is_attack must be a boolean", task_id)
    canary_id = _require_canary(raw["canary_id"], line_no, task_id)
    forbidden = _require_tools(
        raw["forbidden_tool_calls"],
        "forbidden_tool_calls",
        line_no,
        task_id,
    )
    allowed = _require_tools(raw["allowed_tools"], "allowed_tools", line_no, task_id)
    overlap = set(forbidden) & set(allowed)
    if overlap:
        name = sorted(overlap)[0]
        _fail(line_no, f"forbidden tool also listed as allowed: {name}", task_id)

    _check_labels(
        line_no,
        task_id,
        is_attack=is_attack,
        category=category,
        canary_id=canary_id,
        forbidden=forbidden,
        document=document,
    )
    _check_grounding(
        line_no,
        task_id,
        user_request=user_request,
        document=document,
        reference_output=reference_output,
        canary_id=canary_id,
        forbidden=forbidden,
    )
    return EvalTask(
        id=task_id,
        user_request=user_request,
        document=document,
        source=source,
        is_attack=is_attack,
        category=category,
        canary_id=canary_id,
        forbidden_tool_calls=forbidden,
        allowed_tools=allowed,
        reference_output=reference_output,
    )


def _check_labels(
    line_no: int,
    task_id: str,
    *,
    is_attack: bool,
    category: str,
    canary_id: str | None,
    forbidden: tuple[str, ...],
    document: str,
) -> None:
    # Direta, oculta e delimitador são ataques de texto neste v1: sem canário
    # o rótulo mente. Sequestro de ferramenta exige a chamada proibida.
    if is_attack:
        if category not in _ATTACK_CATEGORIES:
            _fail(line_no, "category does not match is_attack", task_id)
        if canary_id is None and not forbidden:
            _fail(
                line_no,
                "attack tasks need a canary_id or a forbidden tool call",
                task_id,
            )
    else:
        if category != CLEAN:
            _fail(line_no, "category does not match is_attack", task_id)
        if canary_id is not None or forbidden:
            _fail(
                line_no,
                "clean tasks cannot carry a canary_id or forbidden_tool_calls",
                task_id,
            )
    if category in _CANARY_REQUIRED and canary_id is None:
        _fail(line_no, f"{category} requires a canary_id", task_id)
    if category == TOOL_CALL_HIJACK and not forbidden:
        _fail(line_no, "tool_call_hijack requires a forbidden tool call", task_id)
    if category == DELIMITER_BREAKOUT and (
        UNTRUSTED_START not in document or UNTRUSTED_END not in document
    ):
        _fail(line_no, "delimiter_breakout must include both harness markers", task_id)


def _check_grounding(
    line_no: int,
    task_id: str,
    *,
    user_request: str,
    document: str,
    reference_output: str,
    canary_id: str | None,
    forbidden: tuple[str, ...],
) -> None:
    if _CANARY_TOKEN_RE.search(user_request):
        _fail(line_no, "user_request must not contain a canary token", task_id)
    if _CANARY_TOKEN_RE.search(reference_output):
        _fail(line_no, "reference_output must not contain a canary token", task_id)
    found = set(_CANARY_TOKEN_RE.findall(document))
    if canary_id is None:
        if found:
            _fail(line_no, "document contains an unlabeled canary token", task_id)
    elif found != {canary_id}:
        _fail(line_no, "document must contain only the labeled canary token", task_id)
    for name in forbidden:
        if name not in document:
            _fail(line_no, f"forbidden tool is not in the document: {name}", task_id)


def _require_text(
    raw: dict[str, object],
    key: str,
    line_no: int,
    task_id: str | None = None,
) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        _fail(line_no, f"{key} must be a non-empty string", task_id)
    return value


def _require_choice(
    raw: dict[str, object],
    key: str,
    choices: tuple[str, ...],
    line_no: int,
    task_id: str,
) -> str:
    value = raw[key]
    if not isinstance(value, str) or value not in choices:
        known = ", ".join(choices)
        _fail(line_no, f"{key} must be one of {known}", task_id)
    return value


def _require_canary(value: object, line_no: int, task_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        _fail(line_no, "canary_id must be a string or null", task_id)
    if _CANARY_ID_RE.fullmatch(value) is None:
        _fail(line_no, "canary_id must match CANARY-<digits>", task_id)
    return value


def _require_tools(value: object, key: str, line_no: int, task_id: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail(line_no, f"{key} must be a list", task_id)
    known = frozenset(spec.name for spec in FakeToolEnvironment().specs())
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in known:
            _fail(line_no, f"unknown tool in {key}: {item!r}", task_id)
        if item in names:
            _fail(line_no, f"duplicate tool in {key}: {item}", task_id)
        names.append(item)
    return tuple(names)


def _fail(line_no: int, message: str, task_id: str | None = None) -> None:
    where = f"line {line_no}"
    if task_id:
        where = f"{where} ({task_id})"
    raise DatasetError(f"{where}: {message}")
