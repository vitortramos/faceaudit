"""Estratégias de naming pro `DatasetLocalizationService` (PR L1c).

Decidem como o arquivo de destino é nomeado durante COPY_ORIGINALS:

  - PRESERVE (default): mantém `file_name` original; em colisão entre
    duas imagens com mesmo nome de pastas diferentes, sufixa
    `<stem>_1<ext>`, `<stem>_2<ext>`, etc. Legível, intuitivo.
  - HASH: `<sha256[:16]>.<ext>`. Único por construção. Ideal pra
    datasets grandes onde colisões PRESERVE seriam frequentes.
  - BY_IDENTITY: `<identity>/<file_name>` (com `_N` em colisão dentro
    do mesmo ID). Cria subpastas por identidade — agrupa visualmente.

PRESERVE é o default escolhido pelo usuário (legibilidade > unicidade
absoluta no caso típico de datasets pequenos a médios).
"""

from __future__ import annotations

from enum import Enum


class NamingStrategy(str, Enum):
    PRESERVE = "preserve"
    HASH = "hash"
    BY_IDENTITY = "by_identity"

    @property
    def label_pt(self) -> str:
        return {
            "preserve":    "Preservar nome original (sufixar em colisão)",
            "hash":        "Hash SHA-256 (16 chars) como nome",
            "by_identity": "Subpasta por identidade",
        }.get(self.value, self.value)
