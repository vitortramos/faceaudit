"""Controlador do diálogo de Auditoria & Sanitização (PR 12).

Ponta ao DatasetAuditService (PR 11). Roda as auditorias O(N) na abertura,
deixa o hard mining O(N²) atrás de botão dedicado, e aplica ações de
sanitização (excluir, marcar) mutando o dataset em memória + emitindo
`dataset_modified` pra o controller pai refrescar tabela e marcar dirty.
"""

from __future__ import annotations

import uuid
from pathlib import Path as _Path
from typing import Dict, List, Optional, Set, Tuple

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
        self._hubness_runner: Optional[WorkerProgressRunner] = None
        self._hubness_findings: List[AuditFinding] = []  # cache do último hubness
        self._umap_runner: Optional[WorkerProgressRunner] = None
        self._umap_findings: List[AuditFinding] = []  # cache do último UMAP
        # Cache de UmapConfig — preserva ajustes do usuário entre
        # aberturas do diálogo. None = usa defaults na primeira vez.
        self._umap_config = None  # Optional[UmapConfig]

        # Thresholds configuráveis pela UI. Defaults conservadores —
        # casam com os defaults dos métodos do DatasetAuditService.
        # Modificados via `audit_settings_changed` do view.
        # `low_quality_thresholds` é um dict `{metric.name: threshold}`
        # — só métricas disponíveis no dataset entram (preenchido logo
        # abaixo após inspecionar via `available_quality_metrics`).
        self._audit_settings: dict = {
            "low_quality_thresholds": {},
            "min_samples_singleton": 2,
            "duplicate_threshold": 0.35,
            "mixed_outlier_threshold": 0.20,
            "mixed_min_samples": 3,
        }

        # Descobre quais métricas de qualidade são auditáveis no dataset
        # — checa schema (column_specs com role=quality) + EMBEDDING_NORM
        # quando há embeddings. Threshold default por métrica embutido
        # nos UI hints do view (FIQA=1.50, OFIQ=25.0, etc.).
        from .dataset_analysis_view import _ui_hints_for
        available = self._audit_service.available_quality_metrics(dataset)
        self._available_quality_metrics: List[str] = [m.name for m in available]
        for metric_name in self._available_quality_metrics:
            self._audit_settings["low_quality_thresholds"][metric_name] = (
                _ui_hints_for(metric_name)["default"]
            )
        # Cache do último resultado de `audit_all_quick` para que mining/
        # biometric possam re-renderizar a tree sem reexecutar O(N) auditorias
        # caras (varredura de paths, hash conflict, missing embeddings, etc.).
        # Em datasets grandes, rerodar tudo a cada conclusão de mining
        # bloqueia a main thread por segundos.
        self._quick_findings: List[AuditFinding] = []

        # PR 14: pilha de ações reversíveis (LIFO). Cada AuditActionRecord
        # guarda o que precisa fazer pra reverter aquela operação.
        self._action_history: List[AuditActionRecord] = []

        self.view = DatasetAnalysisView(
            dataset_name=dataset.name or "Dataset",
            parent=parent_view,
        )
        # Propaga dataset pros widgets que carregam thumbnails
        # — sem isso, datasets em modo RELATIVE (Image Pipeline)
        # mostram thumbnails vazias em todos os painéis de análise.
        if hasattr(self.view, "set_dataset"):
            self.view.set_dataset(dataset)
        self.view.delete_issues_requested.connect(self._on_delete_issues)
        self.view.rerun_requested.connect(self._run_quick_audits)
        self.view.run_heavy_mining_requested.connect(self._run_heavy_mining)
        self.view.run_biometric_audit_requested.connect(self._run_biometric_audit)
        self.view.run_hubness_audit_requested.connect(self._run_hubness_audit)
        self.view.run_umap_audit_requested.connect(self._run_umap_audit)
        self.view.rename_identity_requested.connect(self._on_rename_identity)
        self.view.merge_identities_requested.connect(self._on_merge_identities)
        self.view.undo_requested.connect(self._on_undo)
        self.view.undo_action_at_requested.connect(self._on_undo_action_at)
        self.view.audit_settings_changed.connect(self._on_audit_settings_changed)
        # Propaga métricas disponíveis + defaults iniciais pra view (que
        # os pré-popula no diálogo).
        self.view.set_quality_metric_options(self._available_quality_metrics)
        self.view.set_audit_settings(self._audit_settings)

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
            self._quick_findings = list(
                self._audit_service.audit_all_quick(
                    self.dataset,
                    singleton_min_samples=int(
                        self._audit_settings["min_samples_singleton"]
                    ),
                    low_quality_thresholds=dict(
                        self._audit_settings.get("low_quality_thresholds", {})
                    ),
                )
            )
        except Exception as e:
            QMessageBox.critical(
                self.view, "Erro", f"Falha ao rodar auditorias: {e}",
            )
            return
        self._rebuild_findings_view()

    def _rebuild_findings_view(self) -> None:
        """Combina os 4 caches (quick + biometric + mining + hubness) e
        atualiza a view sem reexecutar nenhuma auditoria. Usado pelos
        slots de conclusão dos workers onde só a tree precisa
        redesenhar — rerodar `audit_all_quick` em dataset grande na
        main thread era a causa do hang após "Finalizando renderização".
        """
        findings = list(self._quick_findings)
        if self._biometric_findings:
            findings.extend(self._biometric_findings)
        if self._mining_findings:
            findings.extend(self._mining_findings)
        if self._hubness_findings:
            findings.extend(self._hubness_findings)
        if self._umap_findings:
            findings.extend(self._umap_findings)
        self.view.set_findings(findings)

    def _run_heavy_mining(self) -> None:
        """Dispara hard mining O(N²) em worker thread. Resultado vira 2
        AuditFindings anexadas à árvore."""
        if self._mining_runner is not None:
            QMessageBox.information(
                self.view, "Aguarde", "Mineração já em execução.",
            )
            return

        if not self._confirm_heavy_op(kind="mining"):
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
        # NÃO rerodar `audit_all_quick` — em dataset grande isso bloqueia
        # a main thread por segundos. Só re-renderizar a tree com os
        # caches já populados.
        self._rebuild_findings_view()

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

        if not self._confirm_heavy_op(kind="biometric"):
            return

        worker = BiometricAuditWorker(
            dataset=self.dataset,
            audit_service=self._audit_service,
            duplicate_threshold=float(self._audit_settings["duplicate_threshold"]),
            mixed_outlier_threshold=float(
                self._audit_settings["mixed_outlier_threshold"]
            ),
            mixed_min_samples=int(self._audit_settings["mixed_min_samples"]),
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
        # NÃO rerodar `audit_all_quick` — só re-renderizar com caches.
        self._rebuild_findings_view()

    def _on_biometric_error(self, msg: str) -> None:
        self.view.btn_run_biometric.setEnabled(True)
        QMessageBox.critical(
            self.view, "Erro na auditoria biométrica", msg,
        )

    def _cleanup_biometric_runner(self) -> None:
        self._biometric_runner = None

    # ── Hubness — KNN exaustivo O(N²·D) por face ─────────────────────────

    def _run_hubness_audit(self) -> None:
        """Dispara `HubnessAnalysisService.compute_analysis` em worker
        thread. Resultado vira 1 AuditFinding na categoria Hubness."""
        if self._hubness_runner is not None:
            QMessageBox.information(
                self.view, "Aguarde", "Análise de hubness já em execução.",
            )
            return

        if not self._confirm_heavy_op(kind="hubness"):
            return

        from interface.tabs.dataset_manager.workers.hubness_audit_worker import (
            HubnessAuditWorker,
        )

        worker = HubnessAuditWorker(
            dataset=self.dataset,
            comparison_service=self._comparison_service,
            k_neighbors=20,
        )
        worker.finished.connect(self._on_hubness_done)
        worker.error.connect(self._on_hubness_error)

        self._hubness_runner = WorkerProgressRunner(
            self.view,
            worker,
            label="Análise de hubness (KNN exaustivo)…",
            allow_cancel=True,
            determinate=True,
            on_finished=lambda *_a: self._cleanup_hubness_runner(),
            on_error=lambda *_a: self._cleanup_hubness_runner(),
        )
        self.view.btn_run_hubness.setEnabled(False)
        self._hubness_runner.start()

    def _on_hubness_done(self, payload) -> None:
        """payload é List[AuditFinding] (com 1 finding) ou None
        (cancelado/erro)."""
        if not payload:
            self.view.btn_run_hubness.setEnabled(True)
            return
        self._hubness_findings = list(payload)
        self.view.btn_run_hubness.setEnabled(True)
        self.view.btn_run_hubness.setText("✅ Hubness concluída — reexecutar?")
        self._rebuild_findings_view()

    def _on_hubness_error(self, msg: str) -> None:
        self.view.btn_run_hubness.setEnabled(True)
        QMessageBox.critical(
            self.view, "Erro na análise de hubness", msg,
        )

    def _cleanup_hubness_runner(self) -> None:
        self._hubness_runner = None

    # ── UMAP — projeção 2D dos embeddings ─────────────────────────────────

    def _run_umap_audit(self) -> None:
        """Abre o diálogo de configuração do UMAP. Se aceito, dispara
        worker em background — mesmo padrão dos outros ops pesados."""
        if self._umap_runner is not None:
            QMessageBox.information(
                self.view, "Aguarde", "UMAP já em execução.",
            )
            return

        # Diálogo pré-execução com hyperparams + estimativa de tempo
        from .umap_config_dialog import UmapConfigDialog
        from interface.tabs.dataset_manager.workers.umap_audit_worker import (
            UmapAuditWorker,
        )

        n_faces = len(self.dataset.faces)
        dlg = UmapConfigDialog(
            self.view,
            config=self._umap_config,
            n_dataset_faces=n_faces,
        )
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        self._umap_config = dlg.get_config()

        # Guard de tamanho — alinhado com o estilo dos outros heavy ops
        if not self._confirm_heavy_op(kind="umap"):
            return

        worker = UmapAuditWorker(
            dataset=self.dataset,
            comparison_service=self._comparison_service,
            config=self._umap_config,
        )
        worker.finished.connect(self._on_umap_done)
        worker.error.connect(self._on_umap_error)

        self._umap_runner = WorkerProgressRunner(
            self.view,
            worker,
            label="UMAP — projeção 2D dos embeddings…",
            allow_cancel=True,
            determinate=True,
            on_finished=lambda *_a: self._cleanup_umap_runner(),
            on_error=lambda *_a: self._cleanup_umap_runner(),
        )
        self.view.btn_run_umap.setEnabled(False)
        self._umap_runner.start()

    def _on_umap_done(self, payload) -> None:
        """payload é List[AuditFinding] (1 finding) ou None
        (cancelado/erro). Cada execução é appendada ao cache — vira
        um item próprio na tree, permitindo comparar configurações
        sem sobrescrever a anterior."""
        if not payload:
            self.view.btn_run_umap.setEnabled(True)
            return
        # Append em vez de replace — cada finding tem audit_id único
        # gerado pelo worker (uuid hex), então convivem na tree.
        self._umap_findings.extend(payload)
        self.view.btn_run_umap.setEnabled(True)
        self.view.btn_run_umap.setText(
            f"🌌 UMAP (Embeddings) — +{len(self._umap_findings)} no histórico"
        )
        self._rebuild_findings_view()

    def _on_umap_error(self, msg: str) -> None:
        self.view.btn_run_umap.setEnabled(True)
        QMessageBox.critical(
            self.view, "Erro no UMAP", msg,
        )

    def _cleanup_umap_runner(self) -> None:
        self._umap_runner = None

    def _on_audit_settings_changed(self, settings: dict) -> None:
        """Recebe novos thresholds da view e dispara as atualizações.

        - Thresholds de qualidade (por métrica) e min_samples_singleton
          afetam apenas auditorias rápidas → reroda `_run_quick_audits()`
          automaticamente quando algum mudou.
        - Os 3 thresholds biométricos (duplicate, mixed outlier, mixed
          min_samples) só passam a valer no próximo clique em "Auditar
          identidades" — o controller não reroda automaticamente porque
          é uma operação O(K²) potencialmente cara.
        """
        prev_quick_keys = (
            dict(self._audit_settings.get("low_quality_thresholds", {})),
            self._audit_settings["min_samples_singleton"],
        )
        # Merge raso, com cuidado pra dict aninhado de thresholds
        for key, value in settings.items():
            if key == "low_quality_thresholds" and isinstance(value, dict):
                self._audit_settings.setdefault("low_quality_thresholds", {})
                self._audit_settings["low_quality_thresholds"].update(value)
            else:
                self._audit_settings[key] = value
        new_quick_keys = (
            dict(self._audit_settings.get("low_quality_thresholds", {})),
            self._audit_settings["min_samples_singleton"],
        )
        if prev_quick_keys != new_quick_keys:
            self._run_quick_audits()
        else:
            # Só thresholds biométricos mudaram — apenas re-renderiza
            # com cache atual. Próximo "Auditar identidades" usa novos.
            self._rebuild_findings_view()

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
            # `identity_1` / `identity_2` são consumidos pela view ao
            # renderizar o layout customizado de hard mining (4 colunas).
            metadata = {
                "cosine": cosine,
                "file1": f1,
                "file2": f2,
                "identity_1": row.get("identity_1", ""),
            }
            if "identity_2" in ident_cols:
                metadata["identity_2"] = row.get("identity_2", "")
            issues.append(AuditIssue(
                locators=tuple(p for p in (p1, p2) if p),
                description=desc,
                metadata=metadata,
            ))

        # Ordena agrupando por par de identidades — todas as comparações
        # entre os mesmos dois IDs ficam adjacentes na tabela. Pares
        # normalizados (sorted tuple) tornam (A,B) e (B,A) o mesmo grupo,
        # útil pra hard impostors caso o algoritmo produza ambas as
        # ordenações. Secundário: cosseno em ordem crescente para
        # genuines (pior primeiro = mais difícil), decrescente para
        # impostors (pior primeiro = mais confundível).
        if "identity_2" in ident_cols:
            issues.sort(key=lambda i: (
                tuple(sorted((
                    str(i.metadata.get("identity_1", "")),
                    str(i.metadata.get("identity_2", "")),
                ))),
                -float(i.metadata.get("cosine", 0.0)),
            ))
        else:
            issues.sort(key=lambda i: (
                str(i.metadata.get("identity_1", "")),
                float(i.metadata.get("cosine", 0.0)),
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
        persistente quando o usuário salvar o dataset.

        Filtra paths já excluídos por ações anteriores na pilha (defesa
        contra double-delete: a UI já marca essas faces em vermelho e
        ignora cliques, mas o atalho de teclado/seleção via tabela ainda
        poderia mandar paths já deletados — e re-deletar criaria um
        AuditActionRecord vazio que confunde o histórico)."""
        if not image_paths:
            return

        already_deleted, _ = self._compute_active_marks()
        fresh_paths = [p for p in image_paths if p not in already_deleted]
        if not fresh_paths:
            QMessageBox.information(
                self.view, "Excluir",
                "Todas as faces selecionadas já foram excluídas em ações "
                "anteriores. Use 'Desfazer' no histórico para revertê-las.",
            )
            return
        target = set(fresh_paths)

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
            op_id=f"delete-{uuid.uuid4()}",
            description=self._describe_delete(deleted_faces),
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
            op_id=f"rename-{uuid.uuid4()}",
            description=self._describe_rename(
                old_identity, new_identity, changes,
            ),
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
            op_id=f"merge-{uuid.uuid4()}",
            description=self._describe_merge(sources, target, changes),
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
        """Desfaz a última ação. Stack LIFO. Disparado pelo Ctrl+Z."""
        if not self._action_history:
            return
        action = self._action_history.pop()
        self._apply_and_finalize_undo(action)

    def _on_undo_action_at(self, op_id: str) -> None:
        """Desfaz uma ação específica do histórico pelo op_id. Disparado
        pelo botão "↶" da lista de histórico no painel esquerdo.

        Diferença vs `_on_undo`: esta operação remove **uma** ação do
        meio da pilha, não necessariamente a última. As outras ações
        continuam aplicadas — mas o undo reverte só a escolhida.
        """
        if not op_id:
            return
        for idx, rec in enumerate(self._action_history):
            if getattr(rec, "op_id", None) == op_id:
                action = self._action_history.pop(idx)
                self._apply_and_finalize_undo(action)
                return
        QMessageBox.information(
            self.view, "Desfazer",
            "Ação não encontrada no histórico — pode já ter sido desfeita.",
        )

    def _apply_and_finalize_undo(self, action: AuditActionRecord) -> None:
        """Reverte a ação no dataset e refresca a UI. Compartilhado pelos
        dois handlers de undo (Ctrl+Z global e botão por linha)."""
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
        """Registra ação na stack e atualiza o histórico na view."""
        self._action_history.append(action)
        self._sync_undo_button()

    def _sync_undo_button(self) -> None:
        """Sincroniza a lista de histórico no painel esquerdo com a
        pilha atual + propaga ao preview as marcas visuais (paths
        excluídos / renomeados) derivadas das ações ativas."""
        if self._action_history:
            top = self._action_history[-1]
            self.view.set_undo_state(True, description=top.description)
        else:
            self.view.set_undo_state(False)
        self.view.set_action_history(self._action_history)
        # Marcações visuais: deriva a cada sync pra refletir undo/redo
        # corretamente (paths que perderam sua ação correspondente
        # voltam ao estilo default).
        deleted, renamed = self._compute_active_marks()
        self.view.set_active_marks(deleted, renamed)

    def _compute_active_marks(self) -> Tuple[Set[str], Dict[str, Tuple[str, str]]]:
        """Deriva, da pilha de ações ativas, dois conjuntos:

        - `deleted_paths`: image_paths que estão em algum
          `AuditActionRecord.deleted_faces` da pilha.
        - `renamed_paths`: dict `path → (old_id, new_id)` para paths
          cuja identidade foi alterada por `AuditActionRecord.identity_changes`.
          `new_id` é a identidade ATUAL no dataset (ou da face deletada)
          — lookup via face index.
        """
        deleted_paths: Set[str] = set()
        renamed_paths: Dict[str, Tuple[str, str]] = {}

        # Indexa identidade corrente por path. Path de face deletada
        # não está no `dataset.faces` — pra esses, usamos a identity
        # registrada no FaceInfo deletado.
        current_id_by_path: Dict[str, str] = {
            getattr(f, "image_path", ""): getattr(f, "identity", "")
            for f in self.dataset.faces
        }

        for rec in self._action_history:
            for face in rec.deleted_faces:
                p = getattr(face, "image_path", "")
                if p:
                    deleted_paths.add(p)
            for path, old_id in rec.identity_changes.items():
                if path in deleted_paths:
                    # Path foi posteriormente deletado — marca delete vence.
                    continue
                new_id = current_id_by_path.get(path, "")
                if new_id and new_id != old_id:
                    renamed_paths[path] = (old_id, new_id)

        return deleted_paths, renamed_paths

    # ── Descrições enriquecidas pro histórico ───────────────────────────

    @staticmethod
    def _short_filename(path: str) -> str:
        if not path:
            return "?"
        try:
            return _Path(path).name
        except Exception:
            return path

    @classmethod
    def _summarize_filenames(cls, paths: List[str], head: int = 3) -> str:
        """Lista os primeiros `head` filenames; sufixa '(+N mais)' se truncar."""
        names = [cls._short_filename(p) for p in paths if p]
        if not names:
            return ""
        if len(names) <= head:
            return ", ".join(names)
        return ", ".join(names[:head]) + f" (+{len(names) - head} mais)"

    @classmethod
    def _describe_delete(cls, deleted_faces: List) -> str:
        """Descrição rica pra histórico: agrupa por identidade e lista
        filenames truncados. Ex: 'excluir 5 face(s) [João: a.jpg, b.jpg
        (+2 mais); Maria: c.jpg]'."""
        n = len(deleted_faces)
        by_ident: Dict[str, List[str]] = {}
        for f in deleted_faces:
            ident = str(getattr(f, "identity", "") or "?")
            path = getattr(f, "image_path", "") or ""
            by_ident.setdefault(ident, []).append(path)
        parts = []
        for ident, paths in list(by_ident.items())[:3]:
            parts.append(f"{ident}: {cls._summarize_filenames(paths)}")
        suffix = ""
        if len(by_ident) > 3:
            suffix = f"; +{len(by_ident) - 3} ID(s)"
        return f"excluir {n} face(s) [{'; '.join(parts)}{suffix}]"

    @classmethod
    def _describe_rename(
        cls, old_identity: str, new_identity: str, changes: Dict[str, str],
    ) -> str:
        names = cls._summarize_filenames(list(changes.keys()))
        return (
            f"renomear '{old_identity}' → '{new_identity}' "
            f"({len(changes)} face(s): {names})"
        )

    @classmethod
    def _describe_merge(
        cls, sources: Set[str], target: str, changes: Dict[str, str],
    ) -> str:
        srcs = sorted(sources)
        if len(srcs) <= 3:
            srcs_text = ", ".join(f"'{s}'" for s in srcs)
        else:
            srcs_text = (
                ", ".join(f"'{s}'" for s in srcs[:3])
                + f" (+{len(srcs) - 3} mais)"
            )
        names = cls._summarize_filenames(list(changes.keys()))
        return (
            f"mesclar {srcs_text} → '{target}' "
            f"({len(changes)} face(s): {names})"
        )

    # ── Guard pre-check pra operações pesadas ────────────────────────────

    # Limites empíricos. Acima do "warn" mostra confirmação; acima do
    # "block" recusa direto (operação inviável na arquitetura atual).
    _BIOMETRIC_WARN_AT_K = 25_000
    _BIOMETRIC_BLOCK_AT_K = 200_000
    _MINING_WARN_AT_N = 20_000
    _MINING_BLOCK_AT_N = 200_000
    # Hubness: O(N²·D). Com BLAS + Zero-Copy Masks roda 10k em ~1s,
    # 100k em ~30s. Acima de 300k vira impraticável (horas).
    _HUBNESS_WARN_AT_N = 50_000
    _HUBNESS_BLOCK_AT_N = 300_000
    # UMAP: subsample resolve datasets grandes — o N efetivo é
    # min(dataset, config.max_samples). Por isso os limites aqui são
    # bem altos: o dialog pré-execução já mostra a estimativa de tempo
    # com base no max_samples configurado.
    _UMAP_WARN_AT_N = 100_000   # avisa só quando o dataset é grande
    _UMAP_BLOCK_AT_N = 5_000_000  # praticamente sem block (subsample protege)

    def _confirm_heavy_op(self, *, kind: str) -> bool:
        """Pre-check de tamanho antes de rodar mineração ou auditoria
        biométrica. Mostra dialog de aviso quando o dataset é grande
        e dialog de bloqueio quando é grande demais. Retorna True se
        o usuário confirmou (ou se o dataset é pequeno e nem mostrou
        nada). False se cancelou ou foi bloqueado.

        `kind`: "biometric" (auditoria de IDs, custo K²) ou "mining"
        (hard mining, custo N²)."""
        n_faces = len(self.dataset.faces)
        # Conta identidades únicas — 1-2s pra 1M faces, aceitável só
        # antes de uma op de minutos.
        n_identities = len({
            getattr(f, "identity", None) for f in self.dataset.faces
            if getattr(f, "identity", None)
        })

        if kind == "biometric":
            # K² em centróides — chunked já não OOM, mas wall-time
            # fica horas se K muito grande.
            if n_identities >= self._BIOMETRIC_BLOCK_AT_K:
                QMessageBox.critical(
                    self.view, "Operação inviável",
                    f"Auditoria biométrica não é viável em datasets com "
                    f"{n_identities:,} identidades (limite atual: "
                    f"{self._BIOMETRIC_BLOCK_AT_K:,}).\n\n"
                    f"O custo é O(K²) = ~{(n_identities ** 2) / 1e9:.1f} "
                    f"bilhões de comparações de centroides — wall-time "
                    f"vai a horas.\n\n"
                    f"Considere subdividir o dataset por origem/source "
                    f"antes de auditar.",
                )
                return False
            if n_identities >= self._BIOMETRIC_WARN_AT_K:
                est_seconds = max(5, int(n_identities * n_identities / 5e8))
                est_str = (
                    f"~{est_seconds}s" if est_seconds < 60
                    else f"~{est_seconds // 60}min"
                )
                res = QMessageBox.warning(
                    self.view, "Operação custosa",
                    f"Esse dataset tem <b>{n_identities:,} identidades</b>. "
                    f"A auditoria biométrica é O(K²): estimativa "
                    f"<b>{est_str}</b> de processamento.\n\n"
                    f"Os pares acima do threshold ({self._audit_settings.get('duplicate_threshold', 0.35):.2f}) "
                    f"são coletados em chunks — sem risco de OOM, "
                    f"mas leva tempo.\n\n"
                    f"Continuar?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                return res == QMessageBox.StandardButton.Yes
            return True

        if kind == "mining":
            # O(N²) em faces. Sem heap streaming, é hard-blocked acima
            # de 200k (1 trilhão de comparações pra 1M).
            if n_faces >= self._MINING_BLOCK_AT_N:
                QMessageBox.critical(
                    self.view, "Operação inviável",
                    f"Mineração biométrica não é viável em datasets com "
                    f"{n_faces:,} faces (limite atual: "
                    f"{self._MINING_BLOCK_AT_N:,}).\n\n"
                    f"O custo é O(N²) = ~{(n_faces ** 2) / 1e9:,.1f} "
                    f"bilhões de comparações face-a-face — wall-time "
                    f"vai a dias e a memória usada é proporcional.\n\n"
                    f"Pra esse tamanho, considere amostrar uma subset "
                    f"≤ 50k faces antes de minerar.",
                )
                return False
            if n_faces >= self._MINING_WARN_AT_N:
                # Estimativa grosseira: ~1M comparações/seg
                est_seconds = max(5, int(n_faces * n_faces / 1e6))
                est_str = (
                    f"~{est_seconds}s" if est_seconds < 60
                    else f"~{est_seconds // 60}min"
                    if est_seconds < 3600
                    else f"~{est_seconds // 3600}h"
                )
                res = QMessageBox.warning(
                    self.view, "Operação custosa",
                    f"Esse dataset tem <b>{n_faces:,} faces</b>. "
                    f"A mineração biométrica é O(N²): estimativa "
                    f"<b>{est_str}</b> de processamento + RAM "
                    f"proporcional.\n\n"
                    f"Continuar?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                return res == QMessageBox.StandardButton.Yes
            return True

        if kind == "hubness":
            # O(N²·D) sobre embeddings — BLAS acelera bastante mas
            # acima de 300k vira impraticável.
            if n_faces >= self._HUBNESS_BLOCK_AT_N:
                QMessageBox.critical(
                    self.view, "Operação inviável",
                    f"Análise de hubness não é viável em datasets com "
                    f"{n_faces:,} faces (limite atual: "
                    f"{self._HUBNESS_BLOCK_AT_N:,}).\n\n"
                    f"O custo é O(N²·D) = ~{(n_faces ** 2) / 1e9:,.1f} "
                    f"bilhões de operações de produto interno. Pra esse "
                    f"tamanho, considere amostrar uma subset ≤ 100k "
                    f"faces antes de analisar.",
                )
                return False
            if n_faces >= self._HUBNESS_WARN_AT_N:
                # Estimativa: ~3M comparações/seg em CPU com BLAS
                est_seconds = max(5, int(n_faces * n_faces / 3e6))
                est_str = (
                    f"~{est_seconds}s" if est_seconds < 60
                    else f"~{est_seconds // 60}min"
                    if est_seconds < 3600
                    else f"~{est_seconds // 3600}h"
                )
                res = QMessageBox.warning(
                    self.view, "Operação custosa",
                    f"Esse dataset tem <b>{n_faces:,} faces</b>. "
                    f"A análise de hubness é O(N²·D): estimativa "
                    f"<b>{est_str}</b> de processamento + RAM "
                    f"proporcional ao batch.\n\n"
                    f"Continuar?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                return res == QMessageBox.StandardButton.Yes
            return True

        if kind == "umap":
            # UMAP usa subsample reproduzível, então o tamanho do
            # dataset bruto não é o gargalo; é o `max_samples` do
            # config. O guard aqui é leve — só avisa em datasets
            # genuinamente grandes.
            if n_faces >= self._UMAP_BLOCK_AT_N:
                QMessageBox.critical(
                    self.view, "Operação inviável",
                    f"UMAP não é viável em datasets com {n_faces:,} "
                    f"faces (limite atual: {self._UMAP_BLOCK_AT_N:,}). "
                    f"Considere subdividir o dataset antes.",
                )
                return False
            if n_faces >= self._UMAP_WARN_AT_N:
                # Lê o max_samples do config corrente — guarda contra
                # acidentes onde o usuário esqueceu de subsample
                # agressivo o suficiente em dataset gigante.
                cfg = getattr(self, "_umap_config", None)
                max_samples = (
                    cfg.max_samples if cfg is not None else 100_000
                )
                effective = min(n_faces, max_samples)
                est_seconds = max(10, int(effective / 30_000)) + 10
                est_str = (
                    f"~{est_seconds}s" if est_seconds < 60
                    else f"~{est_seconds // 60}min"
                )
                res = QMessageBox.warning(
                    self.view, "Operação custosa",
                    f"Dataset com <b>{n_faces:,} faces</b>. "
                    f"UMAP vai operar em <b>{effective:,}</b> amostras "
                    f"(subsample reproduzível) — estimativa "
                    f"<b>{est_str}</b>.\n\nContinuar?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,  # default Yes — dialog já avisou
                )
                return res == QMessageBox.StandardButton.Yes
            return True

        # Tipo desconhecido — passa direto (defesa)
        return True

    def _invalidate_heavy_caches(self) -> None:
        """Mantém os findings de mining/biométrica visíveis após edições
        pequenas (delete/rename/merge). Paths que foram removidos do
        dataset continuam referenciados nas issues — o preview mostrará
        "arquivo não encontrado" pra eles, mas o resto da informação
        permanece útil. Pra refresh real, o usuário clica em "🔬 Auditar
        identidades" / "🔍 Mineração Biométrica" novamente.

        Este método é mantido como hook (intencionalmente vazio agora)
        pra preservar o ponto de extensão — futuro pruning seletivo
        (remover só issues que dependem do path deletado) entra aqui.
        """
        return
