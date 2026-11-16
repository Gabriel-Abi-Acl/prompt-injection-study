"""Regras estreitas para injeção explícita em texto não confiável.

O filtro e o detector de placeholder compartilham os mesmos moldes. Eles
existem para o harness offline, não como classificador de produção: frase
benigna sem esse formato não deve casar.
"""

import re

INJECTION_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+", re.IGNORECASE),
    re.compile(r"(reply|respond|say|output)\s+with\s+CANARY-\d+", re.IGNORECASE),
    re.compile(r"call\s+the\s+fake\s+[A-Za-z_]+\s+tool", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+\w+", re.IGNORECASE),
)

CANARY_INSTRUCTION = re.compile(
    r"(reply|respond|say|output)\s+with\s+(CANARY-\d+)",
    re.IGNORECASE,
)
FAKE_TOOL_INSTRUCTION = re.compile(
    r"call\s+the\s+fake\s+([A-Za-z_]+)\s+tool",
    re.IGNORECASE,
)


def injection_lines(document: str) -> tuple[str, ...]:
    hits: list[str] = []
    for line in document.splitlines():
        if any(pattern.search(line) for pattern in INJECTION_LINE_PATTERNS):
            hits.append(line)
    return tuple(hits)


def strip_injection_lines(document: str) -> tuple[str, bool]:
    """Remove linhas que casam com injeção explícita.

    A linha inteira sai, não só o trecho: o canário e o pedido de ferramenta
    desta linha não podem sobrar no texto enviado ao modelo.
    """

    kept: list[str] = []
    flagged = False
    for line in document.splitlines():
        if any(pattern.search(line) for pattern in INJECTION_LINE_PATTERNS):
            flagged = True
            continue
        kept.append(line)
    return "\n".join(kept).strip(), flagged
