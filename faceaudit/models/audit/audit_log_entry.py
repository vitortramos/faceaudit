"""Audit log entry com cadeia de hashes (PR 25, Tema 5.4).

Cada entrada carrega o hash da entrada anterior (`parent_entry_hash`)
e seu próprio hash (`entry_hash`), formando uma chain estilo blockchain
simples. Adulterar uma entrada antiga quebra todos os hashes
subsequentes — detectável via `AuditLogService.verify_chain`.

Formato em disco: array JSON de `to_dict()` no sidecar
`<dataset_stem>.audit.json` ao lado do parquet.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from faceaudit.enums.audit_operation import AuditOperation


@dataclass(frozen=True)
class AuditLogEntry:
    """Entrada imutável no audit log de um dataset.

    Campos canônicos (todos serializáveis em JSON):
      - `entry_id`: UUID4 da entrada (estável entre runs)
      - `timestamp`: datetime UTC ISO-8601 (gravado em string ao serializar)
      - `operator`: `username@hostname` no momento do evento
      - `operation`: `AuditOperation` enum
      - `description`: texto livre — o quê aconteceu, em pt-BR
      - `metadata`: dict arbitrário com detalhes específicos da operação
                    (ex: paths, contagens, parâmetros). DEVE ser
                    JSON-serializável.
      - `dataset_hash`: hash SHA-256 do parquet no momento da gravação
                        (None quando o parquet ainda não foi salvo —
                         caso típico em CREATE)
      - `parent_entry_hash`: `entry_hash` da entrada anterior (None
                              apenas na primeira entrada do log)
      - `entry_hash`: SHA-256 sobre todos os campos acima — calculado
                      por `compute_entry_hash` no momento da criação

    A imutabilidade (`frozen=True`) protege contra mutação acidental
    em memória; a verificação criptográfica via `entry_hash` protege
    contra adulteração no disco.
    """

    entry_id: str
    timestamp: str  # ISO-8601 UTC
    operator: str
    operation: AuditOperation
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dataset_hash: Optional[str] = None
    parent_entry_hash: Optional[str] = None
    entry_hash: str = ""

    # ── Construção ─────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        operation: AuditOperation,
        description: str,
        operator: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_hash: Optional[str] = None,
        parent_entry_hash: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        entry_id: Optional[str] = None,
    ) -> "AuditLogEntry":
        """Constrói uma entrada nova, calculando `entry_hash` do conteúdo.

        Se `timestamp` for None, usa `datetime.now(timezone.utc)`.
        Se `entry_id` for None, gera UUID4.
        """
        ts = timestamp or datetime.now(timezone.utc)
        ts_str = ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        eid = entry_id or str(uuid.uuid4())
        meta = dict(metadata or {})
        digest = _compute_entry_hash(
            entry_id=eid,
            timestamp=ts_str,
            operator=operator,
            operation=operation.value,
            description=description,
            metadata=meta,
            dataset_hash=dataset_hash,
            parent_entry_hash=parent_entry_hash,
        )
        return cls(
            entry_id=eid,
            timestamp=ts_str,
            operator=operator,
            operation=operation,
            description=description,
            metadata=meta,
            dataset_hash=dataset_hash,
            parent_entry_hash=parent_entry_hash,
            entry_hash=digest,
        )

    # ── Verificação ────────────────────────────────────────────────────────

    def verify_self_hash(self) -> bool:
        """Recalcula `entry_hash` a partir dos campos e compara."""
        recalc = _compute_entry_hash(
            entry_id=self.entry_id,
            timestamp=self.timestamp,
            operator=self.operator,
            operation=self.operation.value,
            description=self.description,
            metadata=self.metadata,
            dataset_hash=self.dataset_hash,
            parent_entry_hash=self.parent_entry_hash,
        )
        return recalc == self.entry_hash

    # ── Serialização ───────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "operator": self.operator,
            "operation": self.operation.value,
            "description": self.description,
            "metadata": dict(self.metadata),
            "dataset_hash": self.dataset_hash,
            "parent_entry_hash": self.parent_entry_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLogEntry":
        return cls(
            entry_id=data["entry_id"],
            timestamp=data["timestamp"],
            operator=data["operator"],
            operation=AuditOperation(data["operation"]),
            description=data["description"],
            metadata=dict(data.get("metadata") or {}),
            dataset_hash=data.get("dataset_hash"),
            parent_entry_hash=data.get("parent_entry_hash"),
            entry_hash=data.get("entry_hash", ""),
        )

    @property
    def timestamp_dt(self) -> datetime:
        """Parse `timestamp` ISO de volta pra `datetime`."""
        return datetime.strptime(self.timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc,
        )


# ─── Helpers internos ──────────────────────────────────────────────────────


def _canonical_json(payload: Any) -> str:
    """JSON canônico: keys ordenadas, sem espaços, ASCII-safe.

    Garante que dois dicts com mesmo conteúdo produzam o mesmo hash —
    independente da ordem de inserção das chaves no Python.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )


def _compute_entry_hash(
    *,
    entry_id: str,
    timestamp: str,
    operator: str,
    operation: str,
    description: str,
    metadata: Dict[str, Any],
    dataset_hash: Optional[str],
    parent_entry_hash: Optional[str],
) -> str:
    """SHA-256 sobre representação canônica dos campos."""
    payload = {
        "entry_id": entry_id,
        "timestamp": timestamp,
        "operator": operator,
        "operation": operation,
        "description": description,
        "metadata": metadata,
        "dataset_hash": dataset_hash,
        "parent_entry_hash": parent_entry_hash,
    }
    blob = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
