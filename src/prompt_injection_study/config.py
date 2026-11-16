"""Configuração por variáveis de ambiente.

Não há leitura automática de arquivo `.env` e não há chave embutida. O
exemplo versionado fica vazio de propósito.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "fake"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openai_model: str = DEFAULT_OPENAI_MODEL
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if environ is None else environ
    provider = env.get("LLM_PROVIDER", "fake").strip().lower() or "fake"
    return Settings(
        llm_provider=provider,
        openai_api_key=env.get("OPENAI_API_KEY", ""),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        openai_model=env.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL,
        anthropic_model=env.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        or DEFAULT_ANTHROPIC_MODEL,
    )
