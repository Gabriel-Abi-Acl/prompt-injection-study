"""Contrato das fixtures sintéticas, da config vazia e dos SDKs preguiçosos."""

import builtins
import subprocess
import sys
from pathlib import Path

import pytest

from prompt_injection_study.config import load_settings
from prompt_injection_study.fixtures import load_placeholder_tasks
from prompt_injection_study.llm import (
    AnthropicChatLLM,
    LLMConfigError,
    OpenAIChatLLM,
    ScriptedFakeLLM,
    build_llm,
)
from prompt_injection_study.models import Message

ROOT = Path(__file__).resolve().parents[1]


def test_placeholder_fixtures_mix_clean_documents_and_benign_canaries():
    tasks = load_placeholder_tasks()
    ids = [task.id for task in tasks]

    assert len(ids) == len(set(ids))
    assert any(not task.has_injection for task in tasks)
    assert any(task.canary == "CANARY-123" for task in tasks)
    assert any(task.injected_tool == "send_email" for task in tasks)
    assert all(task.allowed_tools == () for task in tasks)
    assert all("CANARY-" not in task.user_request for task in tasks)


def test_env_example_keeps_secrets_empty():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assignments = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        assignments[key] = value

    assert assignments["LLM_PROVIDER"] == "fake"
    assert assignments["OPENAI_API_KEY"] == ""
    assert assignments["ANTHROPIC_API_KEY"] == ""


def test_settings_default_to_the_offline_provider():
    settings = load_settings({})

    assert settings.llm_provider == "fake"
    assert settings.openai_api_key == ""
    assert settings.anthropic_api_key == ""
    assert build_llm(settings).__class__ is ScriptedFakeLLM


def test_settings_normalize_provider_name_and_reject_unknown():
    settings = load_settings({"LLM_PROVIDER": "OpenAI", "OPENAI_API_KEY": ""})

    assert settings.llm_provider == "openai"
    assert isinstance(build_llm(settings), OpenAIChatLLM)
    with pytest.raises(LLMConfigError, match="LLM_PROVIDER"):
        build_llm(load_settings({"LLM_PROVIDER": "local-shell"}))


def test_importing_the_package_does_not_import_optional_sdks():
    script = (
        "import sys; import prompt_injection_study; "
        "assert 'openai' not in sys.modules; "
        "assert 'anthropic' not in sys.modules"
    )
    subprocess.check_call([sys.executable, "-c", script], cwd=ROOT)


def test_live_sdks_are_imported_only_when_a_call_has_a_key(monkeypatch):
    seen: list[str] = []
    real_import = builtins.__import__

    def tracking(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"openai", "anthropic"} or name.startswith(("openai.", "anthropic.")):
            seen.append(name)
            raise ImportError(f"{name} is optional")
        return real_import(name, globals, locals, fromlist, level)

    openai_llm = OpenAIChatLLM(api_key="test-openai-key", model="gpt-4o-mini")
    anthropic_llm = AnthropicChatLLM(api_key="test-anthropic-key", model="claude-test")
    monkeypatch.setattr(builtins, "__import__", tracking)

    with pytest.raises(LLMConfigError):
        OpenAIChatLLM(api_key="").complete((Message("user", "hi"),))
    with pytest.raises(LLMConfigError):
        AnthropicChatLLM(api_key="").complete((Message("user", "hi"),))
    assert seen == []

    with pytest.raises(ImportError, match="openai is optional"):
        openai_llm.complete((Message("user", "hi"),))
    with pytest.raises(ImportError, match="anthropic is optional"):
        anthropic_llm.complete((Message("user", "hi"),))
    assert "openai" in seen
    assert "anthropic" in seen
