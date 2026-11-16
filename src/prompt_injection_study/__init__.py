"""Estudo defensivo de prompt injection em aplicações que leem conteúdo não confiável.

Os exemplos de ataque são canários benignos. As ferramentas falsas só registram
chamadas: nada é enviado nem lido do disco.
"""

from prompt_injection_study.config import Settings, load_settings
from prompt_injection_study.dataset import (
    EVAL_CATEGORIES,
    DatasetError,
    EvalTask,
    load_eval_tasks,
)
from prompt_injection_study.defenses import (
    DEFENSE_NAMES,
    DetectorDefense,
    HeuristicFilterDefense,
    HeuristicInjectionDetector,
    LLMJudgeDetector,
    NoDefense,
    OutputCheckDefense,
    SpotlightDefense,
    build_defense,
)
from prompt_injection_study.detection import canary_leaked, unauthorized_tool_names
from prompt_injection_study.fixtures import load_placeholder_tasks
from prompt_injection_study.llm import (
    AnthropicChatLLM,
    FakePolicy,
    LLMConfigError,
    OpenAIChatLLM,
    ScriptedFakeLLM,
    build_llm,
)
from prompt_injection_study.metrics import summarize_metrics
from prompt_injection_study.models import (
    LLMResponse,
    Message,
    RunRecord,
    Task,
    ToolCall,
    ToolSpec,
)
from prompt_injection_study.runner import run_suite, run_task
from prompt_injection_study.tools import FakeToolEnvironment

__version__ = "0.1.0"

__all__ = [
    "AnthropicChatLLM",
    "DEFENSE_NAMES",
    "DatasetError",
    "DetectorDefense",
    "EVAL_CATEGORIES",
    "EvalTask",
    "FakePolicy",
    "FakeToolEnvironment",
    "HeuristicFilterDefense",
    "HeuristicInjectionDetector",
    "LLMConfigError",
    "LLMJudgeDetector",
    "LLMResponse",
    "Message",
    "NoDefense",
    "OpenAIChatLLM",
    "OutputCheckDefense",
    "RunRecord",
    "ScriptedFakeLLM",
    "Settings",
    "SpotlightDefense",
    "Task",
    "ToolCall",
    "ToolSpec",
    "__version__",
    "build_defense",
    "build_llm",
    "canary_leaked",
    "load_eval_tasks",
    "load_placeholder_tasks",
    "load_settings",
    "run_suite",
    "run_task",
    "summarize_metrics",
    "unauthorized_tool_names",
]
