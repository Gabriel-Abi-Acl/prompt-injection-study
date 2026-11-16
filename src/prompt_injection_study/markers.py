"""Marcadores que separam o documento não confiável do pedido do usuário.

A defesa de spotlight e o modelo falso usam os mesmos tokens. Assim o teste
offline distingue um modelo que respeita a fronteira de um que a ignora.
"""

import re

UNTRUSTED_START = "<<UNTRUSTED_DOCUMENT>>"
UNTRUSTED_END = "<<END_UNTRUSTED_DOCUMENT>>"

_BLOCK = re.compile(
    re.escape(UNTRUSTED_START) + r".*?" + re.escape(UNTRUSTED_END),
    re.DOTALL,
)


def shield_markers(document: str) -> str:
    """Troca marcadores que já venham no documento por um texto neutro.

    Sem isso o documento fecharia o bloco cedo e o restante sairia da zona
    delimitada. O estudo ainda pode medir a versão sem esse cuidado no helper
    `outside_untrusted`.
    """

    return document.replace(UNTRUSTED_START, "[marker]").replace(UNTRUSTED_END, "[marker]")


def outside_untrusted(text: str) -> str:
    """Devolve o texto que ficou fora dos blocos delimitados e fechados."""

    return _BLOCK.sub("", text)
