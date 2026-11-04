"""Audit Log Service — cadeia de custódia forense (PR 25, Tema 5.4).

Service stateless que gerencia o sidecar JSON `<dataset_stem>.audit.json`
ao lado do parquet do dataset:

  meu_caso.parquet
  meu_caso.audit.json   ← gravado/lido pelo AuditLogService

Cada `AuditLogEntry` referencia o `entry_hash` da entrada anterior via
`parent_entry_hash`, formando uma chain SHA-256. Adulteração de
qualquer campo de uma entrada antiga quebra:
  (a) o `entry_hash` daquela entrada (verify_self_hash falha)
  (b) o vínculo com a próxima entrada (parent mismatch)

Limites do modelo:
  - Não previne perda do sidecar — quem recebe o parquet sem o
    `.audit.json` não tem evidência da cadeia.
  - Não previne reescrita coordenada de TODO o log (atacante com
    acesso de escrita pode regenerar a chain inteira). Mitigação real
    exige assinatura externa ou anchor em sistema imutável (fora de
    escopo dessa iteração).
  - O `operator` é `getpass.getuser()@socket.gethostname()` — confia
    na identidade do SO. Em ambiente forense profissional, deve
    eventualmente ser substituído por identidade autenticada (smart
    card / 2FA) — fora de escopo.
"""

from __future__ import annotations

import getpass
import json
import socket
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.models.audit.audit_log_entry import AuditLogEntry
from faceaudit.services.software_provenance_service import software_provenance


class JustificationRequiredError(ValueError):
    """Operação que exige justificativa chegou sem ela (ou só com brancos)."""


# Operações em que a decisão do perito precisa vir acompanhada do PORQUÊ.
# A cadeia só as aceita com `metadata["justification"]` não-branca.
OPERATIONS_REQUIRING_JUSTIFICATION = frozenset({AuditOperation.REMOVE_FACES})


def require_justification(text: Optional[str]) -> str:
    """Devolve a justificativa sem brancos nas pontas, ou levanta.

    Ponto único da regra "justificativa obrigatória": o diálogo desabilita
    o botão até haver texto, mas a regra mora aqui — outra tela, um script
    ou um teste não conseguem gravar exclusão sem motivo.
    """
    limpo = (text or "").strip()
    if not limpo:
        raise JustificationRequiredError(
            "Justificativa obrigatória: descreva por que a operação foi feita."
        )
    return limpo


def validate_entry_request(
    operation: AuditOperation, metadata: Optional[Dict[str, Any]],
) -> None:
    """Recusa, antes de gravar, a entrada que a cadeia não pode aceitar."""
    if operation in OPERATIONS_REQUIRING_JUSTIFICATION:
        require_justification((metadata or {}).get("justification"))


@dataclass(frozen=True)
class AuditChainProblem:
    """Problema detectado em uma entrada durante verify_chain."""

    entry_index: int
    entry_id: str
    code: str  # "self_hash_mismatch" | "parent_mismatch" | "missing_parent"
    description: str


class AuditLogService:
    """Service stateless de I/O e verificação de audit logs."""

    SIDECAR_SUFFIX = ".audit.json"

    # ── I/O em disco ────────────────────────────────────────────────────────

    @classmethod
    def sidecar_path_for(cls, dataset_path: str | Path) -> Path:
        """`/path/foo.parquet` → `/path/foo.audit.json`."""
        p = Path(dataset_path)
        return p.with_suffix(cls.SIDECAR_SUFFIX)

    @classmethod
    def load(cls, dataset_path: str | Path) -> List[AuditLogEntry]:
        """Carrega o log do sidecar. Sidecar ausente ou vazio → []."""
        sidecar = cls.sidecar_path_for(dataset_path)
        if not sidecar.exists():
            return []
        try:
            with sidecar.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            warnings.warn(
                f"AuditLog: falha ao ler sidecar {sidecar.name}: {e}",
                stacklevel=2,
            )
            return []
        if not isinstance(payload, list):
            warnings.warn(
                f"AuditLog: sidecar {sidecar.name} não é uma lista JSON; "
                f"ignorando.",
                stacklevel=2,
            )
            return []
        out: List[AuditLogEntry] = []
        for raw in payload:
            try:
                out.append(AuditLogEntry.from_dict(raw))
            except (KeyError, ValueError) as e:
                warnings.warn(
                    f"AuditLog: entrada inválida no sidecar: {e}",
                    stacklevel=2,
                )
        return out

    @classmethod
    def save(cls, dataset_path: str | Path, entries: List[AuditLogEntry]) -> None:
        """Sobrescreve o sidecar com a lista atual.

        Escrita atômica: grava `<sidecar>.tmp` e move atomicamente
        sobre o sidecar real. Evita corrupção em caso de crash no meio
        da escrita.
        """
        sidecar = cls.sidecar_path_for(dataset_path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
        payload = [e.to_dict() for e in entries]
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        tmp.replace(sidecar)

    # ── Append (operação de mais alto nível) ────────────────────────────────

    @classmethod
    def append(
        cls,
        dataset_path: str | Path,
        operation: AuditOperation,
        description: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        dataset_hash: Optional[str] = None,
        operator: Optional[str] = None,
    ) -> AuditLogEntry:
        """Carrega log existente, calcula `parent_entry_hash` da última
        entrada, cria nova entrada, persiste e retorna a entry criada.

        Levanta `JustificationRequiredError` em operação que exige
        justificativa (REMOVE_FACES) sem `metadata["justification"]`."""
        validate_entry_request(operation, metadata)
        entries = cls.load(dataset_path)
        parent_hash = entries[-1].entry_hash if entries else None
        op_str = operator or cls.current_operator()
        # Toda entrada diz qual FaceAudit a produziu (versão + commit) —
        # dentro do metadata, portanto coberta pelo entry_hash.
        meta = dict(metadata or {})
        meta.setdefault("software", software_provenance())
        new_entry = AuditLogEntry.create(
            operation=operation,
            description=description,
            operator=op_str,
            metadata=meta,
            dataset_hash=dataset_hash,
            parent_entry_hash=parent_hash,
        )
        entries.append(new_entry)
        cls.save(dataset_path, entries)
        return new_entry

    @staticmethod
    def current_operator() -> str:
        """`username@hostname` do SO. Falhas de getpass/socket ficam
        registradas como `unknown@unknown`."""
        try:
            user = getpass.getuser()
        except Exception:
            user = "unknown"
        try:
            host = socket.gethostname()
        except Exception:
            host = "unknown"
        return f"{user}@{host}"

    # ── Verificação de cadeia ───────────────────────────────────────────────

    @classmethod
    def verify_chain(
        cls, entries: List[AuditLogEntry],
    ) -> Tuple[bool, List[AuditChainProblem]]:
        """Verifica integridade da chain. Retorna (ok, problems).

        Checks:
          1. Cada entrada bate com seu próprio `entry_hash` recalculado.
          2. `parent_entry_hash` aponta corretamente para a entrada
             anterior (`None` apenas na primeira).
        """
        problems: List[AuditChainProblem] = []
        for i, entry in enumerate(entries):
            if not entry.verify_self_hash():
                problems.append(AuditChainProblem(
                    entry_index=i,
                    entry_id=entry.entry_id,
                    code="self_hash_mismatch",
                    description=(
                        f"entry_hash recalculado não bate com o registrado "
                        f"— campos foram adulterados após gravação."
                    ),
                ))
            if i == 0:
                if entry.parent_entry_hash is not None:
                    problems.append(AuditChainProblem(
                        entry_index=i,
                        entry_id=entry.entry_id,
                        code="missing_parent",
                        description=(
                            f"primeira entrada não deveria ter "
                            f"parent_entry_hash"
                        ),
                    ))
            else:
                expected = entries[i - 1].entry_hash
                if entry.parent_entry_hash != expected:
                    problems.append(AuditChainProblem(
                        entry_index=i,
                        entry_id=entry.entry_id,
                        code="parent_mismatch",
                        description=(
                            f"parent_entry_hash não corresponde ao "
                            f"entry_hash da entrada anterior — chain "
                            f"quebrada (provável remoção/inserção)."
                        ),
                    ))
        return (len(problems) == 0, problems)

    # ── Migração de log ao mover dataset ────────────────────────────────────

    @classmethod
    def migrate_sidecar(
        cls, old_path: str | Path, new_path: str | Path,
    ) -> bool:
        """Renomeia o sidecar quando o parquet é renomeado/copiado.

        Retorna True se houve sidecar pra migrar, False se não.
        Útil em SAVE_AS — após copiar o parquet, chamar isso pro sidecar
        seguir junto. Não cria o sidecar quando ausente.
        """
        old_side = cls.sidecar_path_for(old_path)
        new_side = cls.sidecar_path_for(new_path)
        if not old_side.exists():
            return False
        new_side.parent.mkdir(parents=True, exist_ok=True)
        # Use replace pra garantir atomicidade quando old e new estão
        # no mesmo filesystem; em filesystems diferentes Path.replace
        # ainda fará a movimentação (com fallback interno).
        try:
            old_side.replace(new_side)
        except OSError:
            # Cross-filesystem: cai pra cópia + remove
            import shutil
            shutil.copy2(old_side, new_side)
            old_side.unlink(missing_ok=True)
        return True

    @classmethod
    def copy_sidecar(
        cls, old_path: str | Path, new_path: str | Path,
    ) -> bool:
        """Copia o sidecar para o parquet novo, mantendo o da origem.

        É o que o Salvar Como precisa: o arquivo novo herda a história até
        aqui, e o de origem continua existindo com a dele. Mover (como faz
        `migrate_sidecar`, que serve a renomear) deixava a origem SEM cadeia
        de custódia — um dataset do qual se derivou um recorte perdia todo
        o registro de criação e exclusões.

        Não sobrescreve sidecar já existente no destino. Retorna True se
        copiou.
        """
        import shutil

        old_side = cls.sidecar_path_for(old_path)
        new_side = cls.sidecar_path_for(new_path)
        if not old_side.exists() or new_side.exists():
            return False
        new_side.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_side, new_side)
        return True
