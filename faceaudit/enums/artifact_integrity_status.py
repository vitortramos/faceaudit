# faceaudit/enums/artifact_integrity_status.py

from __future__ import annotations
from enum import Enum


class ArtifactIntegrityStatus(Enum):
    """
    Status de integridade de um artefato do projeto.

    Retornado por LocalProjectArtifactStorage.verify_integrity() para cada
    StrategyArtifactReference verificada.
    """

    OK = "OK"
    """Arquivo presente e hash confere com o registrado."""

    MODIFIED = "MODIFIED"
    """Arquivo presente, mas hash atual difere do registrado. Possível adulteração ou substituição."""

    MISSING = "MISSING"
    """Arquivo não encontrado no path esperado."""

    HASH_UNAVAILABLE = "HASH_UNAVAILABLE"
    """Arquivo presente, mas a referência não contém hash para comparar (gravado sem hash)."""

    ORPHAN = "ORPHAN"
    """Arquivo presente em subpasta auditada, mas não referenciado por nenhuma estratégia do projeto."""

    def __str__(self) -> str:
        return self.value

    @property
    def is_ok(self) -> bool:
        return self == ArtifactIntegrityStatus.OK

    @property
    def is_problem(self) -> bool:
        return self in (ArtifactIntegrityStatus.MODIFIED, ArtifactIntegrityStatus.MISSING)
