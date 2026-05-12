# faceaudit/models/storage/artifact_integrity_result.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from faceaudit.enums.artifact_integrity_status import ArtifactIntegrityStatus
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference


@dataclass(frozen=True)
class ArtifactIntegrityResult:
    """
    Resultado da verificação de integridade de um único artefato do projeto.

    Produzido por LocalProjectArtifactStorage.verify_integrity() para cada
    StrategyArtifactReference inspecionada.

    Campos:
        artifact_ref:   Referência lógica ao artefato verificado (path relativo + tipo).
        status:         Resultado da verificação (OK, MODIFIED, MISSING, HASH_UNAVAILABLE).
        expected_hash:  Hash SHA-256 registrado em artifact_ref no momento da gravação.
                        None se a referência não armazenou hash.
        actual_hash:    Hash SHA-256 calculado agora sobre o arquivo em disco.
                        None se o arquivo não existir (MISSING).
        abs_path:       Caminho absoluto resolvido do artefato.
                        None se o arquivo não existir.
    """

    artifact_ref: StrategyArtifactReference
    status: ArtifactIntegrityStatus
    expected_hash: Optional[str]
    actual_hash: Optional[str]
    abs_path: Optional[Path]

    @property
    def is_ok(self) -> bool:
        return self.status.is_ok

    @property
    def is_problem(self) -> bool:
        return self.status.is_problem

    @property
    def relative_path(self) -> str:
        return self.artifact_ref.path

    def describe(self) -> str:
        """
        Retorna uma mensagem legível descrevendo o resultado da verificação.
        """
        path_str = str(self.abs_path) if self.abs_path else self.relative_path

        if self.status == ArtifactIntegrityStatus.OK:
            return f"[OK] {self.relative_path}"

        if self.status == ArtifactIntegrityStatus.MISSING:
            return f"[AUSENTE] {self.relative_path}"

        if self.status == ArtifactIntegrityStatus.MODIFIED:
            return (
                f"[MODIFICADO] {self.relative_path}\n"
                f"  Esperado : {self.expected_hash}\n"
                f"  Atual    : {self.actual_hash}"
            )

        if self.status == ArtifactIntegrityStatus.HASH_UNAVAILABLE:
            return f"[SEM HASH] {self.relative_path} — arquivo presente, hash não registrado"

        if self.status == ArtifactIntegrityStatus.ORPHAN:
            return f"[ÓRFÃO] {self.relative_path} — arquivo presente, sem estratégia que o referencie"

        return f"[{self.status}] {self.relative_path}"
