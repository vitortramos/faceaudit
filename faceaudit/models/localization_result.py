"""Resultado da operação de localização (PR L1c).

DTO frozen retornado por `DatasetLocalizationService.localize()` —
contém o novo `FaceDataset` (com paths relativos) e métricas pra
exibir na UI / registrar no audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.localization_naming import NamingStrategy

if TYPE_CHECKING:
    from faceaudit.models.face_dataset import FaceDataset


@dataclass(frozen=True)
class LocalizationResult:
    """Resultado de uma operação de localização.

    Attributes:
        dataset: novo `FaceDataset` com paths relativos. O caller deve
            chamar `dataset.to_parquet(target_dir / "<name>.parquet")`
            pra finalizar a portabilidade.
        target_dir: pasta âncora do dataset relativizado.
        mode: modo executado.
        naming: estratégia de naming usada.
        n_processed: total de faces visitadas (== `len(dataset_input)`).
        n_copied: faces que tiveram arquivo copiado pro target_dir
            (sempre 0 em REWRITE_RELATIVE).
        n_skipped: faces puladas porque o arquivo destino já existia
            com hash equivalente (idempotência).
        n_failed: faces que falharam (arquivo fonte ausente, erro de
            cópia, fora do target_dir em REWRITE_RELATIVE).
        failures: lista de `(image_path_original, mensagem_erro)`.
        duration_seconds: tempo total da operação.

    A soma `n_copied + n_skipped + n_failed` deve bater com
    `n_processed` (em REWRITE_RELATIVE: `n_skipped + n_failed`,
    pois nada é copiado).
    """

    dataset: "FaceDataset"
    target_dir: Path
    mode: LocalizationMode
    naming: NamingStrategy
    n_processed: int
    n_copied: int
    n_skipped: int
    n_failed: int
    failures: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    duration_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        """True quando nenhuma face falhou."""
        return self.n_failed == 0

    def summary(self) -> str:
        """Mensagem curta pra logs/UI."""
        parts = [
            f"{self.mode.value}",
            f"{self.n_processed} faces",
        ]
        if self.n_copied:
            parts.append(f"{self.n_copied} copiadas")
        if self.n_skipped:
            parts.append(f"{self.n_skipped} puladas")
        if self.n_failed:
            parts.append(f"⚠ {self.n_failed} falhas")
        parts.append(f"{self.duration_seconds:.1f}s")
        return " · ".join(parts)
