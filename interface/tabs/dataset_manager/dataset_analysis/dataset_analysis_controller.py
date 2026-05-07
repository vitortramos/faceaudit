"""Controlador do diálogo de Auditoria & Sanitização (PR 12).

Ponta ao DatasetAuditService (PR 11). Roda as auditorias O(N) na abertura,
deixa o hard mining O(N²) atrás de botão dedicado, e aplica ações de
sanitização (excluir, marcar) mutando o dataset em memória + emitindo
`dataset_modified` pra o controller pai refrescar tabela e marcar dirty.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QMessageBox

from faceaudit.models.audit import AuditFinding, AuditIssue, AuditSeverity
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.services.comparison_service import ComparisonService
from faceaudit.services.dataset_audit_service import (
    CATEGORY_BIOMETRIC,
    DatasetAuditService,
)

from .dataset_analysis_view import DatasetAnalysisView
from interface.tabs.dataset_manager.workers.dataset_analysis_worker import (
    DatasetAnalysisWorker,
)
from interface.tabs.dataset_manager.workers.worker_progress_runner import (
    WorkerProgressRunner,
)


class DatasetAnalysisController(QtCore.QObject):
    """Orquestra a visualização de auditorias + ações sobre o dataset."""

    # Emitido quando o dataset foi modificado (delete, futuramente rename/merge).
    # O controller pai deve recarregar a tabela e marcar dirty.
    dataset_modified = QtCore.pyqtSignal()

    def __init__(
        self,
        dataset: FaceDataset,
        audit_service: DatasetAuditService,
        comparison_service: ComparisonService,
        parent_view: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent_view)
        self.dataset = dataset
        self._audit_service = audit_service
        self._comparison_service = comparison_service
        self._mining_runner: Optional[WorkerProgressRunner] = None
        self._mining_findings: List[AuditFinding] = []  # cache do último mining

        self.view = DatasetAnalysisView(
            dataset_name=dataset.name or "Dataset",
            parent=parent_view,
        )
        self.view.delete_issues_requested.connect(self._on_delete_issues)
        self.view.rerun_requested.connect(self._run_quick_audits)
        self.view.run_heavy_mining_requested.connect(self._run_heavy_mining)

        # Roda as auditorias O(N) imediatamente ao abrir.
        self._run_quick_audits()

    def exec(self):
        """Exibe o diálogo modalmente."""
        return self.view.exec()

    # ── Auditorias ───────────────────────────────────────────────────────

    def _run_quick_audits(self) -> None:
        """Roda `audit_all_quick` e renderiza. Mantém os findings de hard
        mining (se já rodaram antes) anexados ao final."""
        try:
            findings = self._audit_service.audit_all_quick(self.dataset)
        except Exception as e:
            QMessageBox.critical(
                self.view, "Erro", f"Falha ao rodar auditorias: {e}",
            )
            return

        # Mantém findings de mining anexados (se válidos pro dataset atual).
        if self._mining_findings:
            findings = list(findings) + list(self._mining_findings)
        self.view.set_findings(findings)

    def _run_heavy_mining(self) -> None:
        """Dispara hard mining O(N²) em worker thread. Resultado vira 2
        AuditFindings anexadas à árvore."""
        if self._mining_runner is not None:
            QMessageBox.information(
                self.view, "Aguarde", "Mineração já em execução.",
            )
            return

        worker = DatasetAnalysisWorker(
            dataset=self.dataset,
            comparison_service=self._comparison_service,
            top_n=100,
        )
        # O worker emite finished(df_genuines, df_impostors) — assinatura
        # custom, não casa com o WorkerProgressRunner default. Conectamos
        # manualmente.
        worker.finished.connect(self._on_mining_done)
        worker.error.connect(self._on_mining_error)

        self._mining_runner = WorkerProgressRunner(
            self.view,
            worker,
            label="Mineração biométrica…",
            allow_cancel=True,
            determinate=True,
            on_finished=lambda *_args: self._cleanup_mining_runner(),
            on_error=lambda *_args: self._cleanup_mining_runner(),
        )
        # Desabilita o botão pra evitar duplo clique
        self.view.btn_run_mining.setEnabled(False)
        self._mining_runner.start()

    def _on_mining_done(self, df_genuines, df_impostors) -> None:
        """Converte os DataFrames retornados pelo worker em AuditFindings
        e anexa à árvore."""
        new_findings: List[AuditFinding] = []
        new_findings.append(
            self._df_to_finding(
                df_genuines,
                audit_id="hard_genuines",
                audit_label="Hard genuines (FN potenciais)",
                severity=AuditSeverity.WARNING,
                description_template="similaridade={cosine:.3f}",
                ident_cols=("identity_1",),
            )
        )
        new_findings.append(
            self._df_to_finding(
                df_impostors,
                audit_id="hard_impostors",
                audit_label="Hard impostors (FP potenciais)",
                severity=AuditSeverity.WARNING,
                description_template="similaridade={cosine:.3f}",
                ident_cols=("identity_1", "identity_2"),
            )
        )
        self._mining_findings = new_findings
        self.view.btn_run_mining.setEnabled(True)
        self.view.btn_run_mining.setText("✅ Mineração concluída — reexecutar?")
        self._run_quick_audits()  # re-render com os novos findings

    def _on_mining_error(self, msg: str) -> None:
        self.view.btn_run_mining.setEnabled(True)
        QMessageBox.critical(
            self.view, "Erro na mineração", msg,
        )

    def _cleanup_mining_runner(self) -> None:
        self._mining_runner = None

    def _df_to_finding(
        self,
        df,
        audit_id: str,
        audit_label: str,
        severity: AuditSeverity,
        description_template: str,
        ident_cols: tuple,
    ) -> AuditFinding:
        """Converte o DataFrame de hard mining em `AuditFinding` —
        cada linha do df vira uma `AuditIssue` com 2 paths (par)."""
        if df is None or df.empty:
            return AuditFinding(
                audit_id=audit_id,
                audit_label=audit_label,
                category=CATEGORY_BIOMETRIC,
                severity=severity,
                issues=(),
                summary="Nenhum par anômalo encontrado",
            )

        issues: List[AuditIssue] = []
        for _, row in df.iterrows():
            p1 = row.get("image_path_1", "") or ""
            p2 = row.get("image_path_2", "") or ""
            cosine = float(row.get("cosine_similarity", 0.0))
            f1 = row.get("file_name_1", "")
            f2 = row.get("file_name_2", "")
            ids = " ↔ ".join(str(row.get(c, "?")) for c in ident_cols)
            desc = (
                f"{ids} — {f1} ↔ {f2}  "
                + description_template.format(cosine=cosine)
            )
            issues.append(AuditIssue(
                locators=tuple(p for p in (p1, p2) if p),
                description=desc,
                metadata={"cosine": cosine, "file1": f1, "file2": f2},
            ))
        return AuditFinding(
            audit_id=audit_id,
            audit_label=audit_label,
            category=CATEGORY_BIOMETRIC,
            severity=severity,
            issues=tuple(issues),
            summary=f"{len(issues)} par(es) anômalo(s) (top {len(issues)})",
        )

    # ── Ações de sanitização ─────────────────────────────────────────────

    def _on_delete_issues(self, image_paths: List[str]) -> None:
        """Remove faces do dataset cujo `image_path` está na lista. Mutação
        em memória — só vira persistente quando o usuário salvar o dataset."""
        if not image_paths:
            return
        target = set(image_paths)
        before = len(self.dataset.faces)
        self.dataset.faces = [
            f for f in self.dataset.faces
            if getattr(f, "image_path", None) not in target
        ]
        deleted = before - len(self.dataset.faces)

        # Limpa cache de mining — paths podem ter sido removidos, mas é
        # cara de manter sync por face. Re-rodar mining é decisão do usuário.
        if self._mining_findings:
            self._mining_findings = []

        # Libera o cache de pixmaps do preview — paths excluídos não
        # serão mais necessários. Memória O(N_thumbs * 200x200 RGBA).
        self.view.clear_preview_cache()

        # Avisa o pai pra refrescar tabela + marcar dirty
        self.dataset_modified.emit()

        QMessageBox.information(
            self.view, "Excluído",
            f"{deleted} face(s) removida(s).\n\n"
            "Salve o dataset para tornar permanente.",
        )

        # Re-roda auditorias com o dataset reduzido
        self._run_quick_audits()
