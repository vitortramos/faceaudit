"""Resultado da operação de "Selar projeto" (PR L2.b).

DTO frozen retornado por `ProjectSealService.seal()`. Agrega os
resultados de localizar todos os DATASETs do projeto numa única
operação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class SealResult:
    """Resultado consolidado de selar um projeto FaceAudit.

    Attributes:
        project_root: pasta raiz do projeto.
        n_datasets_processed: total de DATASETs visitados.
        n_datasets_sealed: de fato modificados (estavam ABSOLUTE,
            agora estão RELATIVE com imagens copiadas).
        n_datasets_skipped: pulados porque já estavam em RELATIVE
            (idempotência).
        n_datasets_failed: falharam (erro de I/O, parquet corrupto, etc.).
        total_images_copied: soma de imagens copiadas em todos os
            datasets selados.
        total_images_skipped: imagens não copiadas porque o destino já
            tinha o mesmo conteúdo (idempotência via skip_existing).
        total_failures: pares (dataset_id, mensagem) — útil pra UI.
        duration_seconds: tempo total da operação.
        dry_run: True quando a chamada foi simulação (nada foi escrito).
    """

    project_root: Path
    n_datasets_processed: int
    n_datasets_sealed: int
    n_datasets_skipped: int
    n_datasets_failed: int
    total_images_copied: int
    total_images_skipped: int
    total_failures: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    duration_seconds: float = 0.0
    dry_run: bool = False

    @property
    def succeeded(self) -> bool:
        """True quando nenhum dataset falhou."""
        return self.n_datasets_failed == 0

    def summary(self) -> str:
        """Mensagem curta pra logs/UI."""
        parts = []
        if self.dry_run:
            parts.append("[simulação]")
        parts.append(f"{self.n_datasets_processed} datasets")
        if self.n_datasets_sealed:
            parts.append(f"{self.n_datasets_sealed} selados")
        if self.n_datasets_skipped:
            parts.append(f"{self.n_datasets_skipped} já localizados")
        if self.n_datasets_failed:
            parts.append(f"⚠ {self.n_datasets_failed} falhas")
        if self.total_images_copied:
            parts.append(f"{self.total_images_copied} imagens copiadas")
        if self.total_images_skipped:
            parts.append(f"{self.total_images_skipped} já existiam")
        parts.append(f"{self.duration_seconds:.1f}s")
        return " · ".join(parts)
