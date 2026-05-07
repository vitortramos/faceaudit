"""DTO retornado por `ImagePipelineBatchService.run` (PR I5).

Resultado consolidado de aplicar um pipeline de imagens em todo um
dataset: contagens + lista de falhas + metadata pra escrever no
manifest.json e audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class BatchProcessingResult:
    """Resultado de processar um dataset inteiro com um pipeline."""

    target_dir: Path
    parquet_path: Path
    pipeline_name: str
    n_processed: int
    n_succeeded: int
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
        parts = [f"{self.n_processed} faces"]
        if self.n_succeeded:
            parts.append(f"{self.n_succeeded} processadas")
        if self.n_skipped:
            parts.append(f"{self.n_skipped} já existiam")
        if self.n_failed:
            parts.append(f"⚠ {self.n_failed} falhas")
        parts.append(f"{self.duration_seconds:.1f}s")
        return " · ".join(parts)
