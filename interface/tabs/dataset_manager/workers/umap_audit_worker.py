"""Worker para projeção UMAP de embeddings.

Roda `UmapEmbeddingService.compute(...)` em background e empacota o
`UmapEmbeddingResult` num `AuditFinding`. Cumpre o contrato esperado
pelo `WorkerProgressRunner`:
  - `progress_update(str, int)` para feedback durante o cálculo
  - `finished(object)` com `List[AuditFinding]` (ou None se cancelado)
  - `error(str)` em falha
  - `stop()` para cancelamento cooperativo
"""

from __future__ import annotations

import traceback
import uuid
from typing import List, Optional

from PyQt6 import QtCore

from faceaudit.models.audit import AuditFinding, AuditSeverity
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.umap_config import UmapConfig
from faceaudit.services.comparison_service import ComparisonService
from faceaudit.services.dataset_audit_service import CATEGORY_VISUALIZATION
from faceaudit.services.umap_embedding_service import UmapEmbeddingService


class UmapAuditWorker(QtCore.QObject):
    """Executa UMAP em thread separado. O service emite mensagens
    discretas via `logger_callback`; o worker mapeia cada estágio
    pra uma faixa de progresso (UMAP em si é um black-box que não
    expõe % intermediário)."""

    progress_update = QtCore.pyqtSignal(str, int)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    # Faixa de progresso por estágio. UMAP não emite progresso interno
    # — chutamos os pesos pra dar sensação de avanço gradual.
    _STAGE_PCT = {
        "convert": 5,
        "subsample": 15,
        "fit_start": 25,
        "build_samples": 90,
        "done": 100,
    }

    def __init__(
        self,
        dataset: FaceDataset,
        comparison_service: ComparisonService,
        config: Optional[UmapConfig] = None,
    ):
        super().__init__()
        self._dataset = dataset
        self._comparison_service = comparison_service
        self._config = config or UmapConfig()
        self._is_interrupted = False

    def stop(self) -> None:
        """Cancelamento cooperativo. UMAP não tem hook de cancel
        mid-flight; o flag aborta apenas após o fit terminar (worker
        emite finished(None))."""
        self._is_interrupted = True

    def run(self) -> None:
        try:
            self.progress_update.emit("Iniciando UMAP…", 1)

            service = UmapEmbeddingService(self._comparison_service)
            result = service.compute(
                self._dataset,
                config=self._config,
                logger_callback=self._service_log_to_progress,
            )

            if self._is_interrupted:
                self.finished.emit(None)
                return

            if not result.samples:
                self.error.emit(
                    "UMAP não produziu samples — dataset sem embeddings "
                    "ou conversão falhou."
                )
                self.finished.emit(None)
                return

            self.progress_update.emit("Empacotando finding…", 95)
            finding = self._build_finding(result)

            self.progress_update.emit("Concluído.", 100)
            self.finished.emit([finding])

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            self.finished.emit(None)

    def _service_log_to_progress(self, msg: str) -> None:
        """Mapeia mensagens do service pra estágios de progresso. UMAP
        não tem progresso interno; usamos o conteúdo da mensagem pra
        avançar a barra em jumps discretos."""
        text = str(msg).strip().replace("\n", " ")
        lower = text.lower()
        if "convertendo" in lower:
            pct = self._STAGE_PCT["convert"]
        elif "subsample" in lower:
            pct = self._STAGE_PCT["subsample"]
        elif "rodando umap" in lower:
            pct = self._STAGE_PCT["fit_start"]
        elif "empacotando" in lower:
            pct = self._STAGE_PCT["build_samples"]
        elif "concluído" in lower or "concluido" in lower:
            pct = self._STAGE_PCT["done"]
        else:
            # Mensagem desconhecida — preserva pct atual via -1 sentinel
            pct = -1
        if pct >= 0:
            self.progress_update.emit(text, pct)

    def _build_finding(self, result) -> AuditFinding:
        """Empacota `UmapEmbeddingResult` num `AuditFinding`. O DTO
        completo vai em `metadata.result` — o widget consome direto
        sem refazer parse de listas paralelas."""
        n = result.n_samples_used
        n_total = result.n_total_in_dataset
        avail = ", ".join(result.available_attributes) or "—"
        subsample_note = (
            f" (subsample de {n_total:,})" if n_total > n else ""
        )
        summary = (
            f"{n:,} pontos{subsample_note} | "
            f"n_neighbors={result.config.n_neighbors}, "
            f"min_dist={result.config.min_dist}, "
            f"metric={result.config.metric} | "
            f"atributos disponíveis: {avail}"
        )

        # Audit_id único por execução — cada run vira item próprio na
        # tree, permitindo comparar configurações lado a lado (2D vs
        # 3D, n_neighbors=15 vs 50, faces vs centroides) sem
        # sobrescrever a anterior. O view dispatch usa
        # `startswith("umap_embeddings")`.
        unique_id = f"umap_embeddings_{uuid.uuid4().hex[:8]}"

        # Label embute a config pra distinguir execuções na tree
        cfg = result.config
        mode_str = (
            "centroides" if cfg.mode == "centroids" else "faces"
        )
        cfg_str = (
            f"n={cfg.n_neighbors}, d={cfg.min_dist:.2f}, {cfg.metric}"
        )
        label = (
            f"UMAP {result.n_components}D · {mode_str} · {cfg_str} · "
            f"{n:,} pts"
        )

        return AuditFinding(
            audit_id=unique_id,
            audit_label=label,
            category=CATEGORY_VISUALIZATION,
            severity=AuditSeverity.INFO,
            issues=(),
            summary=summary,
            metadata={
                # O widget acessa via `meta["result"]` e usa atributos
                # diretamente (samples list, available_attributes etc.).
                "result": result,
                "n_samples_used": n,
                "n_total_in_dataset": n_total,
                "available_attributes": list(result.available_attributes),
            },
        )
