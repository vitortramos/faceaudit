"""Worker para análise de hubness O(N²·D).

Roda `HubnessAnalysisService.compute_analysis` em background e empacota
o resultado num `AuditFinding` consumível pela tree de auditorias.
Cumpre o contrato esperado pelo `WorkerProgressRunner`:
  - `progress_update(str, int)` para feedback durante o cálculo
  - `finished(object)` com `List[AuditFinding]` (ou None se cancelado)
  - `error(str)` em falha
  - `stop()` para cancelamento cooperativo
"""

from __future__ import annotations

import re
import traceback
from typing import List, Optional

from PyQt6 import QtCore

from faceaudit.models.audit import AuditFinding, AuditIssue, AuditSeverity
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.services.comparison_service import ComparisonService
from faceaudit.services.dataset_audit_service import CATEGORY_HUBNESS
from faceaudit.services.hubness_analysis_service import HubnessAnalysisService


# Captura o "PCT%" no log emitido pelo service:
#   "  ↳ Hubness analysis: 30% (3/10 lotes)"
_PCT_RE = re.compile(r"(\d+)%")


class HubnessAuditWorker(QtCore.QObject):
    """Executa `compute_analysis` num thread separado. O service emite
    mensagens periódicas via `logger_callback`; este worker faz parsing
    do "%XX" embutido na mensagem pra atualizar a barra de progresso."""

    progress_update = QtCore.pyqtSignal(str, int)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        dataset: FaceDataset,
        comparison_service: ComparisonService,
        k_neighbors: int = 20,
        batch_size: int = 1000,
        chunk_size: int = 10000,
    ):
        super().__init__()
        self._dataset = dataset
        self._comparison_service = comparison_service
        self._k_neighbors = int(k_neighbors)
        self._batch_size = int(batch_size)
        self._chunk_size = int(chunk_size)
        self._is_interrupted = False
        # Memo do último percent emitido — só re-emite quando muda
        # (evita spam pro signal slot da UI).
        self._last_pct = 0

    def stop(self) -> None:
        """Cancelamento cooperativo. O `compute_analysis` não tem
        gancho de cancelamento mid-flight; o flag aborta apenas após
        o cálculo terminar (worker emite finished(None))."""
        self._is_interrupted = True

    def run(self) -> None:
        try:
            self.progress_update.emit(
                "Convertendo para embedding dataset…", 5,
            )
            emb_ds = self._dataset.to_embedding_dataset()
            if not emb_ds.items:
                self.error.emit(
                    "Dataset sem embeddings — análise de hubness inviável."
                )
                self.finished.emit(None)
                return

            if self._is_interrupted:
                self.finished.emit(None)
                return

            self.progress_update.emit(
                f"Iniciando KNN exaustivo (k={self._k_neighbors})…", 10,
            )
            service = HubnessAnalysisService(
                self._comparison_service,
                k_neighbors=self._k_neighbors,
            )

            analysis = service.compute_analysis(
                emb_ds,
                batch_size=self._batch_size,
                chunk_size=self._chunk_size,
                logger_callback=self._service_log_to_progress,
            )

            if self._is_interrupted:
                self.finished.emit(None)
                return

            self.progress_update.emit("Empacotando resultado…", 95)
            histogram_finding = self._build_finding(analysis)
            tables_finding = self._build_tables_finding(analysis)

            self.progress_update.emit("Concluído.", 100)
            # Dois findings: histograma (distribuição) + tabelas
            # (inspeção de extremos com FIQA / norma / Nk por linha).
            self.finished.emit([histogram_finding, tables_finding])

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            self.finished.emit(None)

    def _service_log_to_progress(self, msg: str) -> None:
        """Adapta as mensagens do service (string-only) pra o protocolo
        `(msg, pct)` esperado pelo runner. Faz parse do "NN%" quando
        presente; senão mantém o último pct conhecido."""
        m = _PCT_RE.search(str(msg))
        if m:
            try:
                # Remapeia o percent do service [0,100] pra [10,95] do
                # worker — reservamos as bordas pra setup/empacotamento.
                raw = int(m.group(1))
                self._last_pct = 10 + int(0.85 * raw)
            except ValueError:
                pass
        # Limpa quebras de linha que o service põe em alguns logs
        clean = str(msg).strip().replace("\n", " ")
        self.progress_update.emit(clean, self._last_pct)

    def _build_finding(self, analysis) -> AuditFinding:
        """Empacota a HubnessAnalysisInfo num AuditFinding com o mesmo
        formato dos histogramas — values + image_paths + captions —
        pra reusar o HistogramWidget na view."""
        n_samples = len(analysis.samples)
        nk_values: List[float] = []
        image_paths: List[str] = []
        captions: List[str] = []
        for s in analysis.samples:
            nk_values.append(float(s.nk_score))
            face = (
                self._dataset.faces[s.index]
                if s.index < len(self._dataset.faces) else None
            )
            image_paths.append(
                getattr(face, "image_path", "") or ""
                if face is not None else ""
            )
            ident = (s.identity or "?")
            ident_short = ident if len(ident) <= 12 else ident[:11] + "…"
            captions.append(f"{ident_short}\nNk={s.nk_score}")

        # Métricas escalares + correlações pro summary
        max_nk = analysis.max_nk if analysis.max_nk is not None else 0
        mean_nk = analysis.mean_nk if analysis.mean_nk is not None else 0.0
        std_nk = analysis.std_nk if analysis.std_nk is not None else 0.0
        n_anti_hubs = sum(1 for v in nk_values if v == 0)

        # Pearson Nk vs norma — sinaliza se faces de baixa norma sofrem
        # mais hubness (esperado: corrleação positiva com mean_impostor)
        pearson_norm = analysis.pearson_norm_vs_hubness
        pearson_imp = analysis.pearson_hubness_vs_mean_impostor

        summary_parts = [
            f"k={analysis.k_neighbors}",
            f"n={n_samples:,}",
            f"max Nk={max_nk}",
            f"média Nk={mean_nk:.1f}±{std_nk:.1f}",
            f"anti-hubs (Nk=0): {n_anti_hubs:,}",
        ]
        if pearson_norm is not None:
            summary_parts.append(f"r(Nk,norma)={pearson_norm:+.3f}")
        if pearson_imp is not None:
            summary_parts.append(f"r(Nk,impostor)={pearson_imp:+.3f}")

        return AuditFinding(
            audit_id="histogram_hubness_nk",
            audit_label="Histograma de Nk (hubness)",
            category=CATEGORY_HUBNESS,
            severity=AuditSeverity.INFO,
            issues=(),
            summary=" | ".join(summary_parts),
            metadata={
                "values": nk_values,
                "image_paths": image_paths,
                "captions": captions,
                "metric_name": "HUBNESS_NK",
                "metric_display": (
                    f"Nk score (top-{analysis.k_neighbors} vizinhos errados)"
                ),
                "k_neighbors": analysis.k_neighbors,
                "max_nk": int(max_nk),
                "mean_nk": float(mean_nk),
                "std_nk": float(std_nk),
                "n_anti_hubs": int(n_anti_hubs),
                "pearson_norm_vs_nk": pearson_norm,
                "pearson_impostor_vs_nk": pearson_imp,
            },
        )

    def _build_tables_finding(self, analysis) -> AuditFinding:
        """Empacota top-N hubs (Nk maior) e top-N anti-hubs (Nk menor)
        em listas tabulares com colunas Identidade / Foto / FIQA /
        Norma / Nk / Fotos do ID. Consumido pelo `HubnessTablesWidget`.

        A coluna "Fotos do ID" (samples_per_identity) ajuda a triar
        viés de desbalanceamento: hubs em identidades grandes podem
        ser artefato (cluster denso atrai vizinhança); hubs em
        identidades pequenas tendem a ser problemas topológicos reais."""
        from collections import Counter

        TOP_N = 50

        # Contagem de fotos por identidade no dataset inteiro — não
        # nos samples (em datasets sem yaw/pitch o sample analysis
        # cobre 100% mesmo, mas ser defensivo evita surpresa).
        samples_per_id: Counter = Counter(
            str(getattr(f, "identity", "") or "")
            for f in self._dataset.faces
            if getattr(f, "identity", None)
        )

        def to_row(s) -> dict:
            face = (
                self._dataset.faces[s.index]
                if s.index < len(self._dataset.faces) else None
            )
            ident = s.identity or ""
            return {
                "identity": ident,
                "image_path": (
                    getattr(face, "image_path", "") or ""
                    if face is not None else ""
                ),
                "fiqa": float(s.quality_score) if s.quality_score else None,
                "norm": float(s.norm),
                "nk": int(s.nk_score),
                "n_id_photos": int(samples_per_id.get(ident, 0)),
            }

        # Top hubs: Nk decrescente
        hubs_sorted = sorted(analysis.samples, key=lambda s: -s.nk_score)
        hubs = [to_row(s) for s in hubs_sorted[:TOP_N]]

        # Top anti-hubs: Nk=0 primeiro, depois Nk crescente
        anti_sorted = sorted(analysis.samples, key=lambda s: s.nk_score)
        anti_hubs = [to_row(s) for s in anti_sorted[:TOP_N]]

        return AuditFinding(
            audit_id="hubness_tables",
            audit_label=f"Top {TOP_N} hubs e anti-hubs (tabela)",
            category=CATEGORY_HUBNESS,
            severity=AuditSeverity.INFO,
            issues=(),
            summary=(
                f"Top {len(hubs)} hubs (Nk={hubs[0]['nk'] if hubs else 0} "
                f"a {hubs[-1]['nk'] if hubs else 0}) | "
                f"Top {len(anti_hubs)} anti-hubs (Nk="
                f"{anti_hubs[0]['nk'] if anti_hubs else 0} a "
                f"{anti_hubs[-1]['nk'] if anti_hubs else 0})"
            ),
            metadata={
                "hubs": hubs,
                "anti_hubs": anti_hubs,
            },
        )
