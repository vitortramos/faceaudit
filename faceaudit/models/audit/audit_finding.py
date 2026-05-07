"""DTOs de auditoria de dataset.

Objetos retornados pelo `DatasetAuditService`. Estruturados de forma
unificada pra que a UI consuma uma `List[AuditFinding]` sem precisar
saber sobre o tipo específico de auditoria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class AuditSeverity(str, Enum):
    """Severidade do problema detectado.

    CRITICAL — bloqueia uso forense (path órfão, hash conflitante).
    WARNING  — revisar (qualidade baixa, identidade singleton).
    INFO     — boa prática / informativo (cobertura incompleta de
               enricher opcional).
    """
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

    @property
    def emoji(self) -> str:
        return {"critical": "🔴", "warning": "🟠", "info": "⚪"}[self.value]


@dataclass(frozen=True)
class AuditIssue:
    """Uma instância concreta de problema dentro de uma `AuditFinding`.

    `locators` carrega `image_path`(s) — chave estável que sobrevive a
    reordenações ou re-leituras do dataset. `row_indices` é a posição no
    DataFrame *no momento da auditoria* — pode ficar inválido se o df for
    modificado; serve só pra UI fazer scroll/seleção rápida.

    Padrão por tipo de auditoria:
      - **Row-based** (paths órfãos, qualidade baixa): cada AuditIssue
        cobre 1 face → `locators` tem 1 entrada.
      - **Group-based** (hash conflitante, identidade duplicada): 1
        AuditIssue agrega N faces que compartilham um problema →
        `locators` tem N entradas.
    """
    locators: Tuple[str, ...]                  # image_paths das faces afetadas
    description: str                           # texto curto pro usuário
    row_indices: Optional[Tuple[int, ...]] = None  # opcional, pra UI
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditFinding:
    """Resultado de uma auditoria — agrupa issues + metadados de execução.

    `audit_id` é a chave estável (ex: `"orphan_paths"`); `audit_label` é o
    nome legível (ex: `"Paths órfãos"`); `category` agrupa na UI (ex:
    `"Integridade Física"`). `severity` propaga pra cada AuditIssue se não
    for definida individualmente.
    """
    audit_id: str
    audit_label: str
    category: str
    severity: AuditSeverity
    issues: Tuple[AuditIssue, ...]
    summary: str                              # ex: "47 faces com path inexistente"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.issues)

    @property
    def is_clean(self) -> bool:
        return self.count == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialização leve pra logging/persistência (não pretende ser exaustiva)."""
        return {
            "audit_id": self.audit_id,
            "audit_label": self.audit_label,
            "category": self.category,
            "severity": self.severity.value,
            "count": self.count,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }
