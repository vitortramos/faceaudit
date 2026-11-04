"""Resultado da operação de localização (PR L1c).

DTO frozen retornado por `DatasetLocalizationService.localize()` —
contém o novo `FaceDataset` (com paths relativos) e métricas pra
exibir na UI / registrar no audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

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

    @property
    def n_faces_remaining(self) -> int:
        """Faces que sobraram no dataset devolvido.

        Falha de cópia REMOVE a face do resultado (não há imagem a que
        apontar), então este número pode ser menor que `n_processed`.
        """
        return len(getattr(self.dataset, "faces", None) or [])

    @property
    def is_empty(self) -> bool:
        """True quando não sobrou face nenhuma — o resultado não pode ser
        aplicado nem gravado. Parquet de zero linhas é aceito pela UI,
        listado como recurso e some dos seletores sem explicação."""
        return self.n_faces_remaining == 0

    def failures_preview(self, limit: int = 5) -> str:
        """Lista curta das falhas, uma por linha, para mostrar ao usuário."""
        linhas = [f"  • {p}: {msg}" for p, msg in list(self.failures)[:limit]]
        if len(self.failures) > limit:
            linhas.append(f"  … (+{len(self.failures) - limit})")
        return "\n".join(linhas)

    def refusal_message(self) -> Optional[str]:
        """Motivo para NÃO aplicar o resultado, ou None quando pode seguir."""
        if not self.is_empty:
            return None
        msg = (
            f"Nenhuma das {self.n_processed} face(s) sobrou após a operação "
            "— o resultado não foi aplicado e o dataset aberto continua "
            "como estava."
        )
        if self.failures:
            msg += (
                f"\n\n{self.n_failed} falha(s):\n" + self.failures_preview()
            )
        return msg

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
