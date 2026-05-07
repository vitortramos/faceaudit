"""Controlador do diálogo de Auditoria & Sanitização (PR 12).

Ponta ao DatasetAuditService (PR 11). Roda as auditorias O(N) na abertura,
deixa o hard mining O(N²) atrás de botão dedicado, e aplica ações de
sanitização (excluir, marcar) mutando o dataset em memória + emitindo
`dataset_modified` pra o controller pai refrescar tabela e marcar dirty.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QInputDialog, QMessageBox

from faceaudit.models.audit import AuditFinding, AuditIssue, AuditSeverity
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.services.comparison_service import ComparisonService
from faceaudit.services.dataset_audit_service import (
    CATEGORY_BIOMETRIC,
    DatasetAuditService,
)

from .audit_action import AuditActionRecord
from .dataset_analysis_view import DatasetAnalysisView
from interface.tabs.dataset_manager.workers.biometric_audit_worker import (
    BiometricAuditWorker,
)
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
        self._biometric_runner: Optional[WorkerProgressRunner] = None
        self._biometric_findings: List[AuditFinding] = []  # cache da última biometric

        # PR 14: pilha de ações reversíveis (LIFO). Cada AuditActionRecord
        # guarda o que precisa fazer pra reverter aquela operação.
        self._action_history: List[AuditActionRecord] = []

        self.view = DatasetAnalysisView(
            dataset_name=dataset.name or "Dataset",
            parent=parent_view,
        )
        self.view.delete_issues_requested.connect(self._on_delete_issues)
        self.view.rerun_requested.connect(self._run_quick_audits)
        self.view.run_heavy_mining_requested.connect(self._run_heavy_mining)
        self.view.run_biometric_audit_requested.connect(self._run_biometric_audit)
        self.view.rename_identity_requested.connect(self._on_rename_identity)
        self.view.merge_identities_requested.connect(self._on_merge_identities)
        self.view.undo_requested.connect(self._on_undo)

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

        # Mantém findings de mining e auditoria biométrica anexados (se já
        # rodaram pra esse dataset). Re-execuções vão limpar esses caches.
        if self._biometric_findings:
            findings = list(findings) + list(self._biometric_findings)
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

    # ── Auditoria biométrica O(N²) — duplicate / mixed identities ────────

    def _run_biometric_audit(self) -> None:
        """Dispara `audit_duplicate_identities` + `audit_mixed_identities`
        em worker thread. Resultado vira 2 AuditFindings na categoria
        biométrica, anexados à árvore."""
        if self._biometric_runner is not None:
            QMessageBox.information(
                self.view, "Aguarde", "Auditoria biométrica já em execução.",
            )
            return

        worker = BiometricAuditWorker(
            dataset=self.dataset,
            audit_service=self._audit_service,
            duplicate_threshold=0.85,
            mixed_outlier_threshold=0.4,
            mixed_min_samples=3,
        )
        worker.finished.connect(self._on_biometric_done)
        worker.error.connect(self._on_biometric_error)

        self._biometric_runner = WorkerProgressRunner(
            self.view,
            worker,
            label="Auditoria biométrica de identidades…",
            allow_cancel=True,
            determinate=True,
            on_finished=lambda *_a: self._cleanup_biometric_runner(),
            on_error=lambda *_a: self._cleanup_biometric_runner(),
        )
        self.view.btn_run_biometric.setEnabled(False)
        self._biometric_runner.start()

    def _on_biometric_done(self, payload) -> None:
        """payload é List[AuditFinding] ou None (cancelado/erro)."""
        if not payload:
            self.view.btn_run_biometric.setEnabled(True)
            return
        self._biometric_findings = list(payload)
        self.view.btn_run_biometric.setEnabled(True)
        self.view.btn_run_biometric.setText("✅ Auditoria de IDs concluída — reexecutar?")
        self._run_quick_audits()  # re-render com os novos findings

    def _on_biometric_error(self, msg: str) -> None:
        self.view.btn_run_biometric.setEnabled(True)
        QMessageBox.critical(
            self.view, "Erro na auditoria biométrica", msg,
        )

    def _cleanup_biometric_runner(self) -> None:
        self._biometric_runner = None

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
        """Remove faces do dataset cujo `image_path` está na lista. Empilha
        action record pra permitir undo. Mutação em memória — só vira
        persistente quando o usuário salvar o dataset."""
        if not image_paths:
            return
        target = set(image_paths)

        # Captura faces afetadas ANTES da remoção pra permitir undo
        deleted_faces = [
            f for f in self.dataset.faces
            if getattr(f, "image_path", None) in target
        ]
        if not deleted_faces:
            return

        self.dataset.faces = [
            f for f in self.dataset.faces
            if getattr(f, "image_path", None) not in target
        ]

        self._push_action(AuditActionRecord(
            op_id="delete",
            description=f"excluir {len(deleted_faces)} face(s)",
            deleted_faces=deleted_faces,
        ))

        self._invalidate_heavy_caches()
        self.view.clear_preview_cache()
        self.dataset_modified.emit()

        QMessageBox.information(
            self.view, "Excluído",
            f"{len(deleted_faces)} face(s) removida(s).\n\n"
            "Salve o dataset para tornar permanente.",
        )

        self._run_quick_audits()

    # ── PR 14: Renomear / Mesclar identidades + Undo ─────────────────────

    def _on_rename_identity(self, image_paths: List[str]) -> None:
        """Renomeia em lote a identidade comum às faces selecionadas.

        UX: assume que todas as faces selecionadas têm a MESMA identidade
        (botão é gated nesse caso). O renomeio é APLICADO A TODAS AS
        FACES com aquela identidade no dataset — não só às selecionadas
        (semântica de bulk rename típica de auditoria forense).
        """
        if not image_paths:
            return

        # Descobre a identidade alvo a partir da primeira face encontrada
        path_set = set(image_paths)
        old_identity: Optional[str] = None
        for f in self.dataset.faces:
            if getattr(f, "image_path", None) in path_set:
                old_identity = getattr(f, "identity", None)
                break
        if not old_identity:
            QMessageBox.warning(
                self.view, "Renomear", "Não foi possível determinar a identidade atual.",
            )
            return

        new_identity, ok = QInputDialog.getText(
            self.view, "Renomear identidade",
            f"Renomear TODAS as faces de '{old_identity}' para:",
            text=old_identity,
        )
        if not ok:
            return
        new_identity = (new_identity or "").strip()
        if not new_identity or new_identity == old_identity:
            return

        # Coleta (path, old_id) das faces afetadas pra undo
        changes: Dict[str, str] = {}
        for f in self.dataset.faces:
            if getattr(f, "identity", None) == old_identity:
                changes[getattr(f, "image_path", "")] = old_identity
                f.identity = new_identity
        if not changes:
            return

        self._push_action(AuditActionRecord(
            op_id="rename",
            description=f"renomear '{old_identity}' → '{new_identity}'",
            identity_changes=changes,
        ))

        self._invalidate_heavy_caches()
        self.dataset_modified.emit()
        QMessageBox.information(
            self.view, "Renomeado",
            f"{len(changes)} face(s) atualizada(s) de '{old_identity}' para '{new_identity}'.",
        )
        self._run_quick_audits()

    def _on_merge_identities(self, image_paths: List[str]) -> None:
        """Funde 2+ identidades em uma única.

        UX: a identidade alvo é escolhida via dialog (default = a com
        mais faces no dataset, semântica forense usual). Todas as faces
        das outras identidades selecionadas viram da identidade alvo.
        """
        if not image_paths:
            return

        # Identifica as identidades distintas entre as faces selecionadas
        path_set = set(image_paths)
        identities_in_selection: Set[str] = set()
        for f in self.dataset.faces:
            if getattr(f, "image_path", None) in path_set:
                ident = getattr(f, "identity", None)
                if ident:
                    identities_in_selection.add(str(ident))
        if len(identities_in_selection) < 2:
            QMessageBox.warning(
                self.view, "Mesclar",
                "É preciso selecionar faces de pelo menos 2 identidades distintas.",
            )
            return

        # Conta faces por identidade no dataset inteiro pra sugerir o alvo
        counts: Dict[str, int] = {}
        for f in self.dataset.faces:
            ident = getattr(f, "identity", None)
            if ident in identities_in_selection:
                counts[ident] = counts.get(ident, 0) + 1
        sorted_idents = sorted(identities_in_selection, key=lambda x: -counts.get(x, 0))

        target, ok = QInputDialog.getItem(
            self.view, "Mesclar identidades",
            f"Escolha a identidade ALVO ({len(sorted_idents)} candidatas, "
            f"a primeira tem mais faces):",
            sorted_idents, 0, editable=False,
        )
        if not ok or not target:
            return

        # Aplica: todas as faces das identidades não-alvo recebem o alvo.
        # Captura (path, old_id) pra undo.
        changes: Dict[str, str] = {}
        sources = identities_in_selection - {target}
        for f in self.dataset.faces:
            ident = getattr(f, "identity", None)
            if ident in sources:
                changes[getattr(f, "image_path", "")] = ident
                f.identity = target
        if not changes:
            return

        self._push_action(AuditActionRecord(
            op_id="merge",
            description=(
                f"mesclar {sorted(sources)} → '{target}'"
                if len(sources) <= 3
                else f"mesclar {len(sources)} identidades → '{target}'"
            ),
            identity_changes=changes,
        ))

        self._invalidate_heavy_caches()
        self.dataset_modified.emit()
        QMessageBox.information(
            self.view, "Mesclado",
            f"{len(changes)} face(s) atribuída(s) à identidade '{target}'.",
        )
        self._run_quick_audits()

    def _on_undo(self) -> None:
        """Desfaz a última ação. Stack LIFO."""
        if not self._action_history:
            return
        action = self._action_history.pop()
        action.apply_undo(self.dataset)

        self._invalidate_heavy_caches()
        self.view.clear_preview_cache()
        self.dataset_modified.emit()

        QMessageBox.information(
            self.view, "Desfeito",
            f"Ação revertida: {action.description}",
        )
        self._sync_undo_button()
        self._run_quick_audits()

    # ── Helpers compartilhados ──────────────────────────────────────────

    def _push_action(self, action: AuditActionRecord) -> None:
        """Registra ação na stack e atualiza o botão de undo."""
        self._action_history.append(action)
        self._sync_undo_button()

    def _sync_undo_button(self) -> None:
        if self._action_history:
            top = self._action_history[-1]
            self.view.set_undo_state(True, description=top.description)
        else:
            self.view.set_undo_state(False)

    def _invalidate_heavy_caches(self) -> None:
        """Drop dos findings de mining e biométrica — paths/identidades
        mudaram, os caches estão stale e re-rodar é decisão do usuário."""
        if self._mining_findings:
            self._mining_findings = []
        if self._biometric_findings:
            self._biometric_findings = []
            self.view.btn_run_biometric.setText("🔬 Auditar identidades (pesada)")
