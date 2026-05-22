# interface/tabs/strategy_manager_v2/strategy_manager_v2_controller.py

from __future__ import annotations

import copy
import json
import re
import uuid
from pathlib import Path
from typing import List, Optional, Set

from PyQt6 import QtWidgets, QtCore, QtGui

from faceaudit.enums.artifact_integrity_status import ArtifactIntegrityStatus
from faceaudit.enums.resource_category import ResourceCategory
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.factories.strategy_executor_factory import StrategyExecutorFactory
from faceaudit.models.instructions.strategy_instruction_definition import StrategyInstructionDefinition
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.protocols.strategy_execution_context_protocol import StrategyExecutionContextProtocol
from faceaudit.services.dataset_stats_service import DatasetStatsService
from faceaudit.services.project_service import FaceAuditProject, FaceAuditProjectService
from faceaudit.storage.global_library_storage import GlobalLibraryStorage

from .comparison.comparison_window import ComparisonWindow
from .dialogs.manage_library.manage_library_controller import ManageLibraryController
from .dialogs.new_project_v2.new_project_v2_controller import NewProjectV2Controller
from .dialogs.new_strategy.new_strategy_controller import NewStrategyController
from .left_panel.left_panel_widget import LeftPanelWidget
from .strategy_editor.strategy_editor_widget import StrategyEditorWidget
from .strategy_manager_v2_view import StrategyManagerV2View
from .toolbar.project_toolbar import ProjectToolbar
from .workers.instruction_execution_worker import InstructionExecutionWorker
from .workers.strategy_execution_worker import StrategyExecutionWorker

# Reutilizados da arch antiga
from interface.tabs.strategy_manager.log_progress_dialog.log_progress_dialog import LogProgressDialog
from interface.tabs.strategy_manager.pipeline_creator.pipeline_creator_controller import PipelineCreatorController
from interface.tabs.strategy_manager.new_mining_config_dialog.new_mining_config_dialog_controller import NewMiningConfigDialogController
from interface.tabs.strategy_manager.new_split_config_dialog.new_split_config_dialog_controller import NewSplitConfigDialogController


# UUID v4 no início do stem + `_` → marca de arquivo produzido pelo
# executor via convenção `{strategy_id}_{resource_id}`. Usado em
# `_collect_project_resources` pra distinguir derivados (incluindo
# órfãos de estratégias removidas) de primários no scan do `datasets/`.
_UUID_PREFIX_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_"
)


class StrategyManagerV2Controller(QtCore.QObject):
    """
    Controlador principal da aba Strategy Manager V2 (nova arquitetura declarativa).

    Estado central:
        _project: FaceAuditProject | None — projeto atualmente aberto
        _current_definition: LRStrategyDefinition | None — estratégia em edição
        _definitions: List[LRStrategyDefinition] — todas as definições do projeto
    """

    # Pedido de abrir um dataset (path absoluto) no Dataset Manager.
    # Emitido pelo botão "Abrir no Gerenciador de Datasets" no painel de
    # propriedades. Propagado pro MainGUIController, que troca de aba e
    # carrega o dataset.
    open_dataset_in_manager_requested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        parent=None,
        global_storage: Optional[GlobalLibraryStorage] = None,
        app_config=None,
    ) -> None:
        super().__init__(parent)

        self._project: Optional[FaceAuditProject] = None
        self._current_definition: Optional[LRStrategyDefinition] = None
        self._definitions: List[LRStrategyDefinition] = []
        self._global_storage = global_storage
        self._app_config = app_config

        self._active_thread: Optional[QtCore.QThread] = None

        # Estado de execução persistente entre runs individuais
        self._execution_context: Optional[StrategyExecutionContextProtocol] = None
        self._executed_instruction_ids: set = set()
        self._executor = None

        # Flag de "alterações não salvas". Setada por _on_strategy_modified
        # (edição em andamento via UI), zerada em todo save bem-sucedido e
        # ao abrir/fechar projeto. Consultada por request_close().
        self._is_dirty: bool = False

        self._build_ui(parent)
        self._connect_signals()

    # ── UI Assembly ───────────────────────────────────────────────────────────

    def _build_ui(self, parent) -> None:
        self.view = StrategyManagerV2View(parent=parent)

        # Toolbar unificada (projeto + biblioteca global)
        self._project_toolbar = ProjectToolbar()
        self.view.project_toolbar = self._project_toolbar

        # Substituir placeholder na view
        toolbar_layout = self.view.toolbar_container.layout()
        while toolbar_layout.count():
            item = toolbar_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        toolbar_layout.addWidget(self._project_toolbar)

        # Left panel
        self._left_panel = LeftPanelWidget()
        self.view.set_left_panel(self._left_panel)

        # Strategy editor
        self._strategy_editor = StrategyEditorWidget()
        self.view.set_strategy_editor(self._strategy_editor)

        # Janela de comparação top-level — criada lazy no primeiro clique do
        # botão "Comparar" do left_panel. Mantida como atributo pra preservar
        # referência forte enquanto visível (evita GC do dialog).
        self._comparison_window: Optional[ComparisonWindow] = None

    def _connect_signals(self) -> None:
        # Toolbar unificada
        self._project_toolbar.new_project_requested.connect(self.new_project)
        self._project_toolbar.open_project_requested.connect(self.open_project)
        self._project_toolbar.save_project_requested.connect(self.save_project)
        self._project_toolbar.close_project_requested.connect(self.close_project)
        self._project_toolbar.seal_project_requested.connect(self.seal_project)  # PR L2.b
        self._project_toolbar.add_resources_requested.connect(
            self._on_add_resources_from_library
        )
        self._project_toolbar.verify_integrity_requested.connect(
            self._on_verify_integrity_requested
        )
        self._project_toolbar.pipeline_requested.connect(self._on_pipeline)
        self._project_toolbar.mining_requested.connect(self._on_mining)
        self._project_toolbar.split_requested.connect(self._on_split)
        self._project_toolbar.manage_library_requested.connect(self._on_manage_library)
        self._project_toolbar.toggle_left_panel_requested.connect(self.view.toggle_left_panel)
        # Botão integrado no header do LeftPanelWidget (visível quando
        # o painel está aberto) dispara o mesmo handler.
        if hasattr(self._left_panel, "toggle_collapse_requested"):
            self._left_panel.toggle_collapse_requested.connect(
                self.view.toggle_left_panel
            )
        # Left panel — estratégias
        self._left_panel.strategy_selected.connect(self._on_strategy_opened)
        self._left_panel.strategy_opened.connect(self._on_strategy_opened)
        self._left_panel.new_strategy_requested.connect(self._on_new_strategy)
        self._left_panel.duplicate_requested.connect(self._on_duplicate_strategy)
        self._left_panel.rename_requested.connect(self._on_rename_strategy)
        self._left_panel.remove_requested.connect(self._on_remove_strategy)
        self._left_panel.run_strategy_requested.connect(self.run_strategy)
        self._left_panel.save_as_template_requested.connect(self.save_as_template)
        self._left_panel.comparison_requested.connect(self._on_open_comparison)

        # Left panel — importação/remoção de recursos
        self._left_panel.add_dataset_requested.connect(self._add_dataset)
        self._left_panel.add_split_config_requested.connect(self._add_split_config)
        self._left_panel.add_mining_config_requested.connect(self._add_mining_config)
        self._left_panel.add_pipeline_config_requested.connect(self._add_pipeline_config)
        self._left_panel.remove_resource_requested.connect(self._on_remove_resource)
        self._left_panel.rename_resource_requested.connect(self._on_rename_resource)
        self._left_panel.edit_dataset_in_manager_requested.connect(
            self.open_dataset_in_manager_requested
        )

        # Strategy editor
        self._strategy_editor.strategy_modified.connect(self._on_strategy_modified)
        self._strategy_editor.run_instruction_requested.connect(self._on_run_instruction)
        self._strategy_editor.validate_instruction_requested.connect(self._on_validate_instruction)
        self._strategy_editor.view_trajectory_requested.connect(self._on_view_trajectory)
        self._strategy_editor.view_score_distribution_requested.connect(
            self._on_view_score_distribution
        )
        self._strategy_editor.view_validation_result_requested.connect(
            self._on_view_validation_result
        )
        self._strategy_editor.view_casework_visualization_requested.connect(
            self._on_view_casework_visualization
        )
        self._strategy_editor.view_hubness_report_requested.connect(
            self._on_view_hubness_report
        )
        self._strategy_editor.view_hubness_ablation_requested.connect(
            self._on_view_hubness_ablation
        )
        self._strategy_editor.view_typicality_distribution_requested.connect(
            self._on_view_typicality_distribution
        )
        self._strategy_editor.resource_edit_requested.connect(
            self._on_resource_edit_requested
        )
        self._strategy_editor.instruction_applied.connect(
            self._on_instruction_applied
        )
        self._strategy_editor.instruction_removed.connect(
            self._on_instruction_removed
        )
        self._strategy_editor.hydrate_resource_requested.connect(
            self._on_hydrate_resource_requested
        )
        self._strategy_editor.embedding_column_change_requested.connect(
            self._on_embedding_column_change_requested
        )
        # Phase 3 — diálogos de regra de validação (Add + Save Template).
        self._strategy_editor.add_validation_rule_requested.connect(
            self._on_add_validation_rule_requested
        )
        self._strategy_editor.save_validation_rule_as_template_requested.connect(
            self._on_save_validation_rule_as_template_requested
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def view(self) -> StrategyManagerV2View:
        return self._view

    @view.setter
    def view(self, v: StrategyManagerV2View) -> None:
        self._view = v

    def request_close(self) -> bool:
        """
        Chamado por main_gui_controller.handle_app_close().

        Returns:
            True se pode fechar, False se o usuário cancelou.
        """
        if self._project and self._has_unsaved_changes():
            reply = QtWidgets.QMessageBox.question(
                self._view,
                "Projeto não salvo",
                "Existem alterações não salvas no projeto. Salvar antes de fechar?",
                QtWidgets.QMessageBox.StandardButton.Save |
                QtWidgets.QMessageBox.StandardButton.Discard |
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Save:
                self.save_project()
            elif reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                return False
        return True

    # ── Project operations ────────────────────────────────────────────────────

    def new_project(self) -> None:
        ctrl = NewProjectV2Controller(parent=self._view)
        project = ctrl.exec()
        if project:
            self._set_project(project)

    def open_project(self) -> None:
        fa_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._view,
            "Abrir Projeto FaceAudit",
            str(Path.home()),
            "Projeto FaceAudit (*.fa)",
        )
        if not fa_path:
            return

        try:
            QtWidgets.QApplication.setOverrideCursor(
                QtCore.Qt.CursorShape.WaitCursor
            )
            project = FaceAuditProjectService.open(
                Path(fa_path), verify_integrity=True
            )
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao abrir projeto",
                f"Não foi possível abrir o projeto:\n{exc}",
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self._set_project(project)
        # Oferece migração wipe-and-rerun ANTES de processar integridade —
        # se o usuário aceitar, integrity_results pré-migração ficam stale
        # (arquivos legados foram apagados, definitions invalidadas), e
        # `_migrate_legacy_artifacts_layout_if_needed` re-roda check_integrity
        # ao final pra refletir o novo estado.
        migrated = self._migrate_legacy_artifacts_layout_if_needed()
        if not migrated:
            self._process_integrity_results(project.integrity_results or [])

    def _on_verify_integrity_requested(self) -> None:
        """Re-roda check_integrity on-demand pelo botão "Verificar Integridade".

        Detecta arquivos que foram apagados ou modificados externamente
        depois da abertura do projeto, mesmo quando o usuário não fechou
        e reabriu. Mesmo fluxo de tratamento que `open_project`.
        """
        if not self._project:
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(
                QtCore.Qt.CursorShape.WaitCursor
            )
            results = self._project.check_integrity()
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self._view, "Erro ao verificar integridade",
                f"Falha ao verificar integridade:\n{exc}",
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        # Sem problemas detectados → notifica e sai.
        if not any(r.is_problem for r in results):
            QtWidgets.QMessageBox.information(
                self._view, "Integridade OK",
                "Todos os arquivos referenciados pelas estratégias "
                "estão presentes e com hash íntegro.",
            )
            return

        self._process_integrity_results(results)

    def _process_integrity_results(self, results: list) -> None:
        """Despacha resultados de `check_integrity()` para os handlers de
        MISSING e MODIFIED. Reusado por `open_project` e pelo botão de
        re-check manual.
        """
        missing_results = [
            r for r in results
            if r.status == ArtifactIntegrityStatus.MISSING
        ]
        modified_results = [
            r for r in results
            if r.status == ArtifactIntegrityStatus.MODIFIED
        ]

        if missing_results:
            self._handle_missing_artifact_refs(missing_results)

        if modified_results:
            self._handle_modified_artifact_refs(modified_results)

    def _handle_modified_artifact_refs(self, modified_results: list) -> None:
        """Diálogo Yes/No quando arquivos têm hash divergente do registrado.

        Yes → atualiza `artifact_ref.hash` nas definitions com o hash atual
        do disco e dispara cascade dos recursos derivados produzidos a
        partir desses primários (eles foram gerados com a versão antiga).
        No → preserva o aviso (status quo), próxima abertura mostra de novo.
        """
        n = len(modified_results)
        sample_lines = [r.describe() for r in modified_results[:10]]
        sample = "\n".join(sample_lines)
        more = f"\n... e mais {n - 10}" if n > 10 else ""

        reply = QtWidgets.QMessageBox.question(
            self._view,
            "Arquivos modificados",
            f"{n} arquivo(s) têm hash diferente do registrado — foram "
            "modificados externamente desde o último uso pelo projeto.\n\n"
            f"Exemplos:\n{sample}{more}\n\n"
            "Aceitar a nova versão e invalidar todos os recursos derivados "
            "produzidos a partir desses arquivos? (Os derivados ficarão "
            "stale por construção: foram gerados de uma versão anterior "
            "da entrada.)\n\n"
            "• Sim — atualiza o hash registrado e remove os derivados.\n"
            "• Não — mantém o aviso; próxima abertura repete o alerta.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,  # default conservador
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        updated, invalidated = self._accept_modified_artifact_refs(
            modified_results
        )
        if updated == 0:
            return

        QtWidgets.QMessageBox.information(
            self._view,
            "Versão atualizada",
            f"{updated} hash(es) atualizado(s) nas definitions. "
            f"{invalidated} recurso(s) derivado(s) foram invalidados — "
            "re-execute as instruções afetadas pra reprocessar.",
        )
        self._refresh_left_panel()
        self._refresh_instruction_states()
        self._refresh_produced_resources()

    def on_external_dataset_saved(self, saved_path: str) -> None:
        """Cross-tab hook: Dataset Manager acabou de sobrescrever um
        arquivo via `fluxo_salvar`. Se o path bate com um primary do
        projeto aberto, atualiza `artifact_ref.hash` e dispara cascade
        dos dependentes — sem diálogo de confirmação, pois o usuário
        chegou aqui via o botão "Abrir no Gerenciador" e já consentiu
        a edição ao clicar Save.

        Path mismatch (Save As pra outro lugar, ou nenhum projeto
        aberto) = no-op silencioso.
        """
        if not saved_path or not self._project:
            return
        try:
            saved_abs = Path(saved_path).resolve()
        except Exception:
            return

        try:
            project_root = self._project.storage.project_root.resolve()
        except Exception:
            return

        # Path tem de estar DENTRO da raiz do projeto.
        try:
            saved_abs.relative_to(project_root)
        except ValueError:
            return  # path fora do projeto — Save As pra outro lugar

        # Identifica resources com esse path.
        rel_path = saved_abs.relative_to(project_root).as_posix()
        touched_strategy_ids: Set[str] = set()

        from faceaudit.services.hash_service import HashService
        new_hash: Optional[str] = None
        try:
            new_hash = HashService().calculate_file_hash(saved_abs)
        except Exception:
            new_hash = None

        affected_unique_ids: Set[str] = set()
        for defn in self._definitions:
            for res in defn.resources:
                ref = res.artifact_ref
                if ref is None or ref.path != rel_path:
                    continue
                # Se o hash não mudou de fato, ignora — usuário pode ter
                # apenas salvado sem alterar conteúdo. Sem essa checagem,
                # o cascade dispararia mesmo em casos sem alteração real.
                if new_hash and ref.hash == new_hash:
                    continue
                if new_hash:
                    ref.hash = new_hash
                # Zera payload: forçar re-hidratação do parquet na próxima
                # leitura. Sem isso, `column_specs` em memória continua o
                # antigo — o combo de embedding NÃO mostraria novas
                # colunas após enriquecimento (caso comum do fluxo
                # Dataset Manager → enrich → save).
                res.payload = None
                if (
                    self._execution_context is not None
                    and self._execution_context.has_resource(res.resource_id)
                ):
                    try:
                        ctx_res = self._execution_context.get_resource(res.resource_id)
                        if ctx_res is not None:
                            ctx_res.payload = None
                    except Exception:
                        pass
                if res.id:
                    affected_unique_ids.add(res.id)
                touched_strategy_ids.add(defn.strategy_id)

        if not affected_unique_ids:
            return

        for defn in self._definitions:
            if defn.strategy_id in touched_strategy_ids:
                try:
                    self._project.definition_repository.save_definition(defn)
                except Exception:
                    pass

        # Cascade dos dependentes — reusa o BFS já existente.
        invalidated_total = 0
        for uid in affected_unique_ids:
            before = sum(
                1 for defn in self._definitions for r in defn.resources
                if r.category == ResourceCategory.DERIVED
                and r.artifact_ref is not None
            )
            self._invalidate_dependents_of(uid)
            after = sum(
                1 for defn in self._definitions for r in defn.resources
                if r.category == ResourceCategory.DERIVED
                and r.artifact_ref is not None
            )
            invalidated_total += max(0, before - after)

        self._refresh_left_panel()
        self._refresh_instruction_states()
        self._refresh_produced_resources()

        QtWidgets.QMessageBox.information(
            self._view,
            "Dataset atualizado",
            f"Dataset '{rel_path}' atualizado a partir do Gerenciador "
            f"de Datasets.\n\n{invalidated_total} recurso(s) derivado(s) "
            "foram invalidados — re-execute as instruções afetadas "
            "para reprocessar com a versão nova.",
        )

    def _accept_modified_artifact_refs(
        self, modified_results: list,
    ) -> tuple:
        """Atualiza `artifact_ref.hash` pra cada path em `modified_results`
        com o hash atual do arquivo no disco, em todas as definitions que
        referenciam aquele path. Dispara cascade dos derivados produzidos
        a partir de cada um.

        Retorna `(num_hashes_updated, num_derived_invalidated)`.
        """
        if not self._project:
            return (0, 0)

        from faceaudit.services.hash_service import HashService

        hash_service = HashService()
        project_root = self._project.storage.project_root

        # Mapeia path → novo hash recém-computado.
        new_hashes: dict = {}
        for r in modified_results:
            if not r.artifact_ref or not r.artifact_ref.path:
                continue
            abs_path = project_root / r.artifact_ref.path
            if not abs_path.exists():
                continue
            try:
                new_hashes[r.artifact_ref.path] = (
                    hash_service.calculate_file_hash(abs_path)
                )
            except Exception:
                continue

        if not new_hashes:
            return (0, 0)

        # Atualiza hash + coleta ids únicos afetados (pra cascade).
        affected_unique_ids: Set[str] = set()
        updated_count = 0
        touched_strategy_ids: Set[str] = set()
        for defn in self._definitions:
            for res in defn.resources:
                ref = res.artifact_ref
                if ref and ref.path in new_hashes:
                    ref.hash = new_hashes[ref.path]
                    # Força re-hidratação na próxima leitura. Crítico
                    # pra que `_collect_primary_embedding_columns` veja
                    # `column_specs` novos (ex: dataset re-enriquecido
                    # com nova coluna de embedding).
                    res.payload = None
                    if (
                        self._execution_context is not None
                        and self._execution_context.has_resource(res.resource_id)
                    ):
                        try:
                            ctx_res = self._execution_context.get_resource(
                                res.resource_id
                            )
                            if ctx_res is not None:
                                ctx_res.payload = None
                        except Exception:
                            pass
                    updated_count += 1
                    if res.id:
                        affected_unique_ids.add(res.id)
                    touched_strategy_ids.add(defn.strategy_id)

        # Salva primeiro as definitions com hash atualizado.
        for defn in self._definitions:
            if defn.strategy_id in touched_strategy_ids:
                try:
                    self._project.definition_repository.save_definition(defn)
                except Exception:
                    pass

        # Cascade: invalida todos os dependentes de cada id único afetado.
        # `_invalidate_dependents_of` já cuida do save após.
        total_invalidated = 0
        for uid in affected_unique_ids:
            before = len([
                r for defn in self._definitions for r in defn.resources
                if r.category == ResourceCategory.DERIVED
                and r.artifact_ref is not None
            ])
            self._invalidate_dependents_of(uid)
            after = len([
                r for defn in self._definitions for r in defn.resources
                if r.category == ResourceCategory.DERIVED
                and r.artifact_ref is not None
            ])
            total_invalidated += max(0, before - after)

        return (updated_count, total_invalidated)

    def _handle_missing_artifact_refs(self, missing_results: list) -> None:
        """Pergunta ao usuário se quer limpar refs cujos arquivos sumiram.

        Caso afirmativo, zera `artifact_ref`/`payload` de cada `StrategyResource`
        cujo path consta em `missing_results`, salva as definitions tocadas
        e atualiza a UI (instruções afetadas perdem o ✅).
        """
        n = len(missing_results)
        # Mostra primeiros 10 paths como amostra; lista completa fica longa
        # demais pra um diálogo.
        sample_lines = [r.describe() for r in missing_results[:10]]
        sample = "\n".join(sample_lines)
        more = f"\n... e mais {n - 10}" if n > 10 else ""

        reply = QtWidgets.QMessageBox.question(
            self._view,
            "Referências quebradas",
            f"O projeto tem {n} referência(s) a arquivos que não existem "
            "no disco. Pode ter sido limpeza manual ou um salvamento que "
            "falhou em sessão anterior.\n\n"
            f"Exemplos:\n{sample}{more}\n\n"
            "Deseja limpar essas referências? As instruções correspondentes "
            "perderão o ✅ verde e precisarão ser re-executadas para "
            "produzir os arquivos novamente.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        cleaned = self._clear_missing_artifact_refs(missing_results)
        if cleaned == 0:
            return

        QtWidgets.QMessageBox.information(
            self._view,
            "Referências limpas",
            f"{cleaned} referência(s) zerada(s) nas definitions. "
            "Re-execute as instruções afetadas pra gerar os arquivos.",
        )
        # Refresh UI: árvore esquerda, dropdowns, estados das instruções.
        # Estados dependem do current_definition; recém abertos não têm
        # uma estratégia ativa, então essas atualizações ficam pra quando
        # o usuário selecionar uma estratégia.
        self._refresh_left_panel()

    def _clear_missing_artifact_refs(self, missing_results: list) -> int:
        """Zera `artifact_ref` e `payload` de cada recurso cujo path aparece
        em `missing_results`. Salva as definitions tocadas no repositório.

        Retorna o número de refs zeradas.
        """
        if not self._project:
            return 0

        missing_paths: Set[str] = {
            r.artifact_ref.path
            for r in missing_results
            if r.artifact_ref and r.artifact_ref.path
        }
        if not missing_paths:
            return 0

        cleaned_count = 0
        touched_strategy_ids: Set[str] = set()
        for defn in self._definitions:
            for res in defn.resources:
                if res.artifact_ref and res.artifact_ref.path in missing_paths:
                    res.artifact_ref = None
                    res.payload = None
                    cleaned_count += 1
                    touched_strategy_ids.add(defn.strategy_id)

        for defn in self._definitions:
            if defn.strategy_id in touched_strategy_ids:
                try:
                    self._project.definition_repository.save_definition(defn)
                except Exception:
                    pass

        return cleaned_count

    def save_project(self) -> None:
        if not self._project:
            return
        try:
            for defn in self._definitions:
                self._project.definition_repository.save_definition(defn)
            self._is_dirty = False
            QtWidgets.QMessageBox.information(
                self._view, "Projeto salvo", "Projeto salvo com sucesso."
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro ao salvar", f"Não foi possível salvar o projeto:\n{exc}"
            )

    def close_project(self) -> None:
        if not self.request_close():
            return
        self._project = None
        self._current_definition = None
        self._definitions.clear()
        self._left_panel.clear()
        self._strategy_editor.show_empty()
        self._project_toolbar.set_project_open(False)
        self._left_panel.set_strategy_actions_enabled(False)
        self._view.show_welcome_screen()
        self._is_dirty = False

    # ── PR L2.b — Selar projeto ──────────────────────────────────────────
    def seal_project(self) -> None:
        """Abre o `SealProjectDialog`, coleta opções e dispara o
        `SealProjectWorker` em thread separada com LogProgressDialog.

        Após sucesso: re-carrega definitions (path_modes mudaram em
        memória) e força save do projeto.
        """
        if not self._project:
            return

        from .seal_project_dialog import SealProjectDialog
        dialog = SealProjectDialog(self._project, parent=self._view)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        dry_run = dialog.dry_run
        self._dispatch_seal_worker(dry_run=dry_run)

    def _dispatch_seal_worker(self, *, dry_run: bool) -> None:
        """Cria worker + thread + LogProgressDialog seguindo o padrão
        de `run_strategy`."""
        from .workers.seal_project_worker import SealProjectWorker

        title = (
            "Simulando selamento do projeto…" if dry_run
            else "Selando projeto…"
        )
        progress_dialog = LogProgressDialog(title, parent=self._view)
        progress_dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        progress_dialog.show()

        thread = QtCore.QThread()
        worker = SealProjectWorker(self._project, dry_run=dry_run)
        worker.moveToThread(thread)
        self._active_worker = worker

        thread.started.connect(worker.run)
        worker.progress_update.connect(
            lambda msg, pct: progress_dialog.append_log(f"[{pct}%] {msg}")
        )

        def on_finished(result):
            if result is None:
                progress_dialog.finish(False)
                return
            progress_dialog.finish(True)
            self._show_seal_summary(result)
            # Após selar, definitions referenciam parquets que foram
            # reescritos — refaça o refresh do left panel pra sincronizar
            # os hashes nas refs em memória.
            if not result.dry_run and result.n_datasets_sealed > 0:
                # Marca dirty pra que o usuário saiba que precisa salvar
                # o projeto (definitions têm artifact_refs com hashes
                # antigos — o save vai recalcular).
                self._is_dirty = True
                self._refresh_left_panel()

        def on_error(msg):
            progress_dialog.finish(False)
            QtWidgets.QMessageBox.critical(
                self._view, "Erro ao selar", msg,
            )

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._active_thread = thread
        thread.start()

    def _show_seal_summary(self, result) -> None:
        """Mostra QMessageBox com resumo da operação de selo."""
        if result.n_datasets_failed:
            failures_preview = "\n".join(
                f"  • {ds_id}: {msg}"
                for ds_id, msg in list(result.total_failures)[:5]
            )
            extra = (
                f"\n\n⚠ {result.n_datasets_failed} falha(s):\n{failures_preview}"
                + ("\n  …" if result.n_datasets_failed > 5 else "")
            )
        else:
            extra = ""

        title = "Simulação concluída" if result.dry_run else "Projeto selado"
        suffix = (
            "\n\n[Simulação] Nenhum arquivo foi modificado. "
            "Desmarque a opção 'Apenas simular' pra selar de verdade."
            if result.dry_run else ""
        )
        QtWidgets.QMessageBox.information(
            self._view, title,
            f"{result.summary()}{extra}{suffix}",
        )

    def _add_dataset(self) -> None:
        """Importa um dataset (.parquet) da biblioteca global para o projeto."""
        self._import_resource(
            title="Importar Dataset",
            config_path_attr="datasets",
            file_filter="Dataset (*.parquet);;Todos (*.*)",
            artifact_type=StrategyArtifactType.GLOBAL_DATASET,
            resource_type=StrategyResourceType.DATASET,
            artifact_kwargs_key="dataset_id",
            compute_stats=True,
        )

    def import_dataset_from_path(self, parquet_path: str) -> None:
        """Entry point público chamado pelo MainGUIController quando o
        Dataset Manager pede "Enviar pra Estratégia" (PR 17).

        Diferente do `_add_dataset` (que abre QFileDialog), este recebe
        o path direto. Se nenhum projeto está aberto, oferece ao usuário:
        abrir existente, criar novo, ou cancelar. Após garantir o
        projeto, segue o mesmo caminho de import (resource_id editável,
        copy_artifact_to_project, refresh do left panel, sidecar de
        stats).
        """
        if not parquet_path:
            return
        path_obj = Path(parquet_path)
        if not path_obj.exists():
            QtWidgets.QMessageBox.warning(
                self._view, "Arquivo inexistente",
                f"O arquivo informado não existe:\n{parquet_path}",
            )
            return

        # 1. Garante projeto aberto
        if self._project is None:
            msg = QtWidgets.QMessageBox(self._view)
            msg.setWindowTitle("Selecionar projeto")
            msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
            msg.setText(
                "Para enviar o dataset, é preciso ter um projeto aberto."
            )
            msg.setInformativeText(
                "Deseja abrir um projeto existente ou criar um novo?"
            )
            btn_open = msg.addButton(
                "Abrir existente…", QtWidgets.QMessageBox.ButtonRole.AcceptRole
            )
            btn_new = msg.addButton(
                "Criar novo…", QtWidgets.QMessageBox.ButtonRole.AcceptRole
            )
            msg.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked is btn_open:
                self.open_project()
            elif clicked is btn_new:
                self.new_project()
            else:
                return
            # Se o usuário cancelou o sub-dialog, _project continua None
            if self._project is None:
                return

        # 2. Pede o resource_id (default = stem do parquet)
        suggested_id = path_obj.stem
        resource_id, ok = QtWidgets.QInputDialog.getText(
            self._view,
            "ID do recurso",
            "Confirme ou edite o ID do recurso\n"
            "(deve ser único dentro do projeto):",
            text=suggested_id,
        )
        if not ok or not resource_id.strip():
            return
        resource_id = resource_id.strip()

        # 3. Importa pro projeto (absolutiza paths se RELATIVE) + sidecar
        # de stats + refresh. Ver `LocalProjectArtifactStorage.import_dataset`
        # — RELATIVE com `path_root` ancorado em `(parquet_dir / path_root)`
        # viraria broken assim que o parquet fosse copiado pra cá; a
        # absolutização preserva os paths apontando pras imagens originais.
        try:
            ref = self._project.storage.import_dataset(
                source_path=parquet_path,
                dataset_id=resource_id,
            )
            self._compute_and_save_dataset_stats(
                self._project.storage.project_root / ref.path
            )
            self._refresh_left_panel()
            QtWidgets.QMessageBox.information(
                self._view, "Recurso adicionado",
                f"Dataset '{resource_id}' importado com sucesso ao projeto.\n"
                f"Path no projeto: {ref.path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro ao importar",
                f"Não foi possível importar o dataset:\n{exc}",
            )

    def _add_split_config(self) -> None:
        """Importa uma Split Config (.json) da biblioteca global para o projeto."""
        self._import_resource(
            title="Importar Split Config",
            config_path_attr="split_configs",
            file_filter="Split Config (*.json);;Todos (*.*)",
            artifact_type=StrategyArtifactType.SPLIT_CONFIG,
            resource_type=StrategyResourceType.SPLIT_CONFIG,
            artifact_kwargs_key="config_id",
        )

    def _add_mining_config(self) -> None:
        """Importa uma Mining Config (.json) da biblioteca global para o projeto."""
        self._import_resource(
            title="Importar Mining Config",
            config_path_attr="mining_configs",
            file_filter="Mining Config (*.json);;Todos (*.*)",
            artifact_type=StrategyArtifactType.MINING_CONFIG,
            resource_type=StrategyResourceType.ARTIFACT,
            artifact_kwargs_key="config_id",
        )

    def _add_pipeline_config(self) -> None:
        """Importa uma Pipeline Config (.json) da biblioteca global para o projeto."""
        self._import_resource(
            title="Importar Pipeline Config",
            config_path_attr="pipelines",
            file_filter="Pipeline Config (*.json);;Todos (*.*)",
            artifact_type=StrategyArtifactType.PIPELINE_CONFIG,
            resource_type=StrategyResourceType.PIPELINE_CONFIG,
            artifact_kwargs_key="config_id",
        )

    def _import_resource(
        self,
        title: str,
        config_path_attr: str,
        file_filter: str,
        artifact_type: StrategyArtifactType,
        resource_type: StrategyResourceType,
        artifact_kwargs_key: str,
        compute_stats: bool = False,
    ) -> None:
        """
        Fluxo comum de importação de recurso para o projeto:
        1. Abre FileDialog na pasta correta (app_config ou home)
        2. Sugere resource_id a partir do nome do arquivo
        3. Permite ao usuário editar o ID
        4. Copia o arquivo para o projeto via copy_artifact_to_project()
        5. Atualiza o painel esquerdo
        """
        if not self._project:
            return

        # Pasta inicial: lê do app_config, depois tenta global_storage, por fim home
        initial_dir = str(Path.home())
        if self._app_config is not None:
            cfg_path = getattr(getattr(self._app_config, "paths", None), config_path_attr, None)
            if cfg_path:
                p = Path(cfg_path)
                if p.exists():
                    initial_dir = str(p)
                else:
                    initial_dir = str(p)  # abre mesmo que não exista ainda

        source_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._view, title, initial_dir, file_filter,
        )
        if not source_path:
            return

        # Sugere ID a partir do nome do arquivo (sem extensão)
        suggested_id = Path(source_path).stem

        resource_id, ok = QtWidgets.QInputDialog.getText(
            self._view,
            "ID do recurso",
            "Confirme ou edite o ID do recurso\n(deve ser único dentro da estratégia):",
            text=suggested_id,
        )
        if not ok or not resource_id.strip():
            return

        resource_id = resource_id.strip()

        # PR L2.a — só pra DATASET, oferece a opção de copiar imagens
        # pra dentro do projeto. Default Não — perito pode rodar
        # "Selar projeto" (L2.b) no final pra empacotar o pendrive.
        copy_images_to_project = False
        if artifact_type == StrategyArtifactType.GLOBAL_DATASET:
            copy_images_to_project = self._ask_copy_images(resource_id)

        try:
            if copy_images_to_project:
                # Opção 2 — copia imagens pra `<projeto>/images/<id>/` e
                # reescreve paths como relativos (Selar-no-import).
                ref = self._import_dataset_with_image_copy(
                    source_path, resource_id,
                )
            elif artifact_type == StrategyArtifactType.GLOBAL_DATASET:
                # Opção 1 — absolutiza paths (se RELATIVE) e copia o parquet
                # pro projeto. Imagens continuam onde estavam; projeto não é
                # autocontido até o usuário rodar "Selar projeto".
                ref = self._project.storage.import_dataset(
                    source_path=source_path,
                    dataset_id=resource_id,
                )
            else:
                # Demais artifact types (split/mining/pipeline configs) —
                # cópia direta, sem semântica de path.
                ref = self._project.storage.copy_artifact_to_project(
                    source_path=source_path,
                    artifact_type=artifact_type,
                    **{artifact_kwargs_key: resource_id},
                )

            if compute_stats:
                self._compute_and_save_dataset_stats(
                    self._project.storage.project_root / ref.path
                )

            self._refresh_left_panel()
            extra = (
                "\n\nImagens copiadas pra <projeto>/images/{id}/"
                .format(id=resource_id)
                if copy_images_to_project
                else ""
            )
            QtWidgets.QMessageBox.information(
                self._view,
                "Recurso adicionado",
                f"Recurso '{resource_id}' importado com sucesso.\n"
                f"Path: {ref.path}{extra}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro ao importar", f"Não foi possível importar o recurso:\n{exc}"
            )

    # ── PR L2.a — Helpers de import com cópia opcional de imagens ────────

    def _ask_copy_images(self, resource_id: str) -> bool:
        """Pergunta se deseja copiar imagens referenciadas pro projeto.
        Default: Não. Operação custosa em disco e tempo — perito pode
        deixar pra rodar "Selar projeto" no final do trabalho."""
        msg = QtWidgets.QMessageBox(self._view)
        msg.setWindowTitle("Copiar imagens pro projeto?")
        msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg.setText(
            f"<b>Copiar as imagens referenciadas pelo dataset "
            f"'{resource_id}' pra dentro do projeto?</b>"
        )
        msg.setInformativeText(
            "<p>Recomendado quando o projeto será compartilhado ou levado a "
            "outra máquina (autocontido / pendrive paradigm).</p>"
            "<p>Pode demorar e ocupar bastante espaço em disco.</p>"
            "<p><b>Padrão: Não</b> — você pode rodar isso depois com "
            "<i>Selar projeto</i> na toolbar.</p>"
        )
        msg.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
        )
        msg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
        return msg.exec() == QtWidgets.QMessageBox.StandardButton.Yes

    def _import_dataset_with_image_copy(
        self, source_path: str, resource_id: str,
    ):
        """Carrega o parquet, copia imagens pra `<projeto>/images/<id>/`,
        salva o parquet localizado em `<projeto>/datasets/<id>.parquet`
        e retorna o `StrategyArtifactReference`.

        Reusa `DatasetLocalizationService` com mode=COPY_ORIGINALS e
        naming=PRESERVE (defaults sensatos do plano L1).
        """
        from faceaudit.enums.localization_mode import LocalizationMode
        from faceaudit.enums.localization_naming import NamingStrategy
        from faceaudit.enums.project_folder import ProjectFolder
        from faceaudit.enums.strategy_file_type import StrategyFileType
        from faceaudit.models.face_dataset import FaceDataset
        from faceaudit.models.storage.strategy_artifact_reference import (
            StrategyArtifactReference,
        )
        from faceaudit.services.dataset_localization_service import (
            DatasetLocalizationService,
        )

        project_root = Path(self._project.storage.project_root)
        target_images_dir = project_root / ProjectFolder.IMAGES.value / resource_id

        # 1. Carrega o parquet original (de fora do projeto).
        ds = FaceDataset.from_parquet(source_path)

        # 2. Localiza com COPY_ORIGINALS pro <project>/images/<id>/
        service = DatasetLocalizationService()
        result = service.localize(
            ds,
            target_images_dir,
            mode=LocalizationMode.COPY_ORIGINALS,
            naming=NamingStrategy.PRESERVE,
            skip_existing=True,
        )

        if result.n_failed:
            # Falhas de localização não abortam — perito vê na próxima tela
            # que algumas imagens não foram copiadas. Mas o dataset
            # importado contém só as que tiveram sucesso.
            print(
                f"⚠ Localização: {result.n_failed} de {result.n_processed} "
                f"falharam. Veja `result.failures` no log."
            )

        # 3. Ajusta path_root pro destino dentro do projeto. O service
        # cria a estrutura `<target_images_dir>/images/<file>` com
        # path_root="images" — mas como já estamos dentro de
        # `<project>/images/<id>/`, o path_root "images" aponta pra
        # `<project>/images/<id>/images/`. Pra ficar mais limpo, ajustamos
        # o path_root do dataset salvo pra apontar `images/<id>/images`
        # relativo ao parquet (que vai pra `<project>/datasets/`).
        #
        # Math: parquet em `<project>/datasets/<id>.parquet`, imagens em
        # `<project>/images/<id>/images/`. Path_root relativo ao parquet
        # = `../images/<id>/images`. POSIX, cross-platform.
        result.dataset.path_root = f"../{ProjectFolder.IMAGES.value}/{resource_id}/images"

        # 4. Salva parquet localizado
        parquet_dest = (
            project_root
            / StrategyFileType.SOURCE_DATASET.build_path(dataset_id=resource_id)
        )
        parquet_dest.parent.mkdir(parents=True, exist_ok=True)
        result.dataset.to_parquet(parquet_dest)

        # 5. Calcula hash + monta artifact_ref
        from faceaudit.services.hash_service import HashService
        file_hash = HashService().calculate_file_hash(parquet_dest)
        relative_path = parquet_dest.relative_to(project_root)

        return StrategyArtifactReference(
            artifact_type=StrategyArtifactType.GLOBAL_DATASET,
            path=relative_path.as_posix(),
            hash=file_hash,
        )

    def run_strategy(self) -> None:
        if not self._project or not self._current_definition:
            return

        self._reset_execution_state()

        dialog = LogProgressDialog("Executando estratégia…", parent=self._view)
        # NonModal para liberar entrega de sinais queued no Windows (ver CLAUDE.md).
        dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dialog.show()

        try:
            executor = StrategyExecutorFactory.build(
                artifact_storage=self._project.storage,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro", f"Não foi possível inicializar executor:\n{exc}"
            )
            dialog.finish(success=False)
            return

        # Recolhe recursos do projeto (DATASET / PIPELINE_CONFIG /
        # MINING_CONFIG do disco) — eles não estão em definition.resources
        # mas as instruções os referenciam por id. Sem isso, validators e
        # ensure_payload disparam &quot;Recurso X não encontrado no contexto&quot;.
        project_resources = self._collect_project_resources()

        # Defensive: zera payload dos PRIMARY DATASETs em `defn.resources`
        # antes de cada Run All — força rehidratação com a `embedding_column`
        # atual da estratégia. Sem isso, payload cacheado de uma run
        # anterior (cancelada ou completa) era reusado via early-return em
        # `ensure_payload`. Causa raiz do bug onde usuário começou Run All
        # com embedding_v2, cancelou, trocou pra v6, re-rodou — o V6 silenciosamente
        # usava os vetores v2 cacheados do run anterior. `artifact_ref`
        # preservado — arquivo continua no disco.
        for res in (self._current_definition.resources or []):
            if (
                res.resource_type == StrategyResourceType.DATASET
                and res.category == ResourceCategory.PRIMARY
            ):
                res.payload = None

        # CRÍTICO: PRIMARY DATASETs são compartilhados por referência entre
        # estratégias que os referenciam (deduplicação por resource_id em
        # `_collect_project_resources`). Se a estratégia anterior hidratou
        # o payload com sua `embedding_column`, o objeto compartilhado fica
        # cacheado COM AQUELA COLUNA. A próxima estratégia (com outra
        # embedding_column) chama `ensure_payload`, que faz early-return
        # se `payload is not None` — e silenciosamente usa os vetores da
        # coluna ERRADA. Sintoma: scores byte-iguais entre estratégias com
        # embedding_columns diferentes (ex: V4 e V6 dando outputs
        # idênticos porque ambas reusam o payload V2 que a V2 hidratou
        # primeiro).
        #
        # Fix: shallow-copy de cada PRIMARY DATASET com `payload=None`
        # antes de passar como additional_resources. Cada strategy run
        # recebe uma cópia independente que será rehidratada com a
        # `embedding_column` da estratégia atual. ArtifactReference fica
        # compartilhado (metadata imutável), só o slot payload é zerado.
        isolated_resources: List[StrategyResource] = []
        for res in project_resources:
            if (
                res.resource_type == StrategyResourceType.DATASET
                and res.category == ResourceCategory.PRIMARY
            ):
                clone = copy.copy(res)
                clone.payload = None
                isolated_resources.append(clone)
            else:
                isolated_resources.append(res)
        project_resources = isolated_resources

        thread = QtCore.QThread()
        worker = StrategyExecutionWorker(
            executor,
            self._current_definition,
            additional_resources=project_resources,
        )
        worker.moveToThread(thread)

        # Mantém referência Python ao worker para evitar GC prematuro.
        self._active_worker = worker

        cancel_state = {"requested": False}

        thread.started.connect(worker.run)
        worker.log_message.connect(dialog.append_log)

        def on_cancel_requested():
            # Sinaliza pro worker parar entre instruções (cancelamento
            # cooperativo). Antes só setávamos o flag local e o worker
            # continuava executando até o fim — usuário via "Cancelando..."
            # mas a estratégia rodava inteira.
            cancel_state["requested"] = True
            worker.request_cancel()

        dialog.cancellation_requested.connect(on_cancel_requested)

        def close_dialog_if_cancelled():
            if cancel_state["requested"]:
                dialog.accept()

        def on_finished(model):
            for instr in self._current_definition.instructions:
                self._executed_instruction_ids.add(instr.instruction_id)

            # A persistência (pickle/parquet pesado) já rodou na thread do
            # worker — aqui só sincronizamos os artifact_refs já gravados
            # com a definição em memória (operação rápida, sem I/O).
            self._sync_trained_model_to_definition(model)

            self._refresh_instruction_states()
            self._refresh_produced_resources()
            # Repopula a árvore de recursos e os combos do editor para refletir
            # os recursos produzidos durante o Run All.
            self._refresh_left_panel()
            dialog.finish(True)
            close_dialog_if_cancelled()

        def on_error(_msg):
            dialog.finish(False)
            # Refresh do estado mesmo em erro/cancelamento — instruções
            # que rodaram com sucesso antes do erro/cancel persistiram
            # seus outputs em disco, e o `_refresh_instruction_states`
            # lê do disco pra repopular ✅. Sem isso, a UI mostra
            # nenhuma ✅ apesar do trabalho parcial ter sido salvo.
            self._refresh_instruction_states()
            self._refresh_produced_resources()
            self._refresh_left_panel()
            close_dialog_if_cancelled()

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._active_thread = thread
        thread.start()

    def _on_open_comparison(self) -> None:
        """
        Abre a janela de comparação entre estratégias treinadas. Reutiliza a
        instância existente quando já está aberta — só traz pra frente em vez
        de criar uma duplicata.

        Injeta definitions atuais + project_root + loader pra que a janela
        possa: (1) listar items disponíveis por kind via availability_resolver,
        (2) hidratar SimilarityScoreDataset via sidecar quando precisar render
        a distribuição de calibração, (3) ler VALIDATION_RESULT/CASEWORK
        pickles direto do disco resolvido relativo ao project_root.
        """
        if self._comparison_window is not None and self._comparison_window.isVisible():
            self._comparison_window.raise_()
            self._comparison_window.activateWindow()
            return
        if not self._project:
            return
        # Garante que o resource_loader esteja construído. Útil quando o
        # usuário abre Comparar antes de tocar em qualquer instrução do editor.
        self._ensure_executor_and_context(silent=True)
        loader = (
            getattr(self._executor, "_resource_loader", None)
            if self._executor is not None
            else None
        )
        self._comparison_window = ComparisonWindow(
            definitions=self._definitions,
            project_root=self._project.storage.project_root,
            resource_loader=loader,
            parent=self.view,
            # Re-lê do disco no botão "Atualizar" — sem isso o snapshot
            # passado aqui congela e a janela nunca enxerga execuções
            # feitas depois da abertura.
            definitions_provider=lambda: list(
                self._project.definition_repository.list_definitions()
            ) if self._project else [],
        )
        # Abre maximizada — alinha com o comportamento da janela principal
        # do app. O usuário ainda pode des-maximizar pelo botão da decoração.
        self._comparison_window.showMaximized()
        self._comparison_window.raise_()
        self._comparison_window.activateWindow()

    def save_as_template(self) -> None:
        """
        Salva a estratégia atualmente aberta como template na biblioteca global.

        Comportamento:
        - Sem estratégia selecionada → aviso explícito.
        - Sem biblioteca configurada → aviso explícito apontando o caminho do
          fallback (`~/.faceaudit/library`) e o config.json como overrides.
        - O template é uma cópia LEVE da definição: instruções e metadados são
          preservados, mas `payload` e `artifact_ref` de cada recurso são
          descartados (templates não carregam dados — só a estrutura). Isso
          também evita problemas de `copy.deepcopy` com objetos runtime
          não-copiáveis (ex: tensores torch carregados pelo SpeechBrain PLDA).
        """
        if not self._current_definition:
            QtWidgets.QMessageBox.warning(
                self._view,
                "Nenhuma estratégia selecionada",
                "Selecione uma estratégia na lista à esquerda para salvá-la "
                "como template.",
            )
            return

        if not self._global_storage:
            QtWidgets.QMessageBox.warning(
                self._view,
                "Biblioteca global indisponível",
                "Não foi possível acessar a biblioteca global de templates.\n\n"
                "Possíveis causas:\n"
                "• A pasta padrão (~/.faceaudit/library) não pôde ser criada — "
                "verifique permissões.\n"
                "• Você definiu paths.library no config.json com um caminho "
                "inválido — corrija ou remova essa entrada para usar o padrão.",
            )
            return

        try:
            template = self._build_template_from_current_definition()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao preparar template",
                f"Falha ao gerar a cópia leve da estratégia:\n{exc}",
            )
            return

        try:
            self._global_storage.save_artifact(
                StrategyArtifactType.STRATEGY_TEMPLATE,
                template,
                template_id=template.strategy_id,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao salvar template",
                f"Não foi possível salvar o template em disco:\n{exc}",
            )
            return

        library_root = getattr(self._global_storage, "library_root", None)
        location = (
            f"\n\nLocal: {library_root}/strategy_templates/{template.strategy_id}.json"
            if library_root else ""
        )
        QtWidgets.QMessageBox.information(
            self._view,
            "Template salvo",
            f"Estratégia '{template.name}' salva como template na biblioteca "
            f"global.{location}",
        )

    def _build_template_from_current_definition(self):
        """
        Cria uma cópia leve da definição atual sem `payload` e sem
        `artifact_ref` nos recursos — templates devem ser portáveis entre
        projetos. Usa `to_dict()` + `from_dict()` em vez de `deepcopy`
        para evitar copiar objetos runtime pesados (ex: ndarray, tensores)
        que não fazem sentido em um template.
        """
        from faceaudit.models.strategy.lr_strategy_definition import (
            LRStrategyDefinition,
        )

        snapshot = self._current_definition.to_dict()
        template = LRStrategyDefinition.from_dict(snapshot)

        if template is None:
            raise RuntimeError(
                "from_dict retornou None ao reconstruir a definição."
            )

        template.strategy_id = str(uuid.uuid4())
        for resource in template.resources:
            resource.artifact_ref = None
            resource.payload = None

        return template

    # ── Internal slots ────────────────────────────────────────────────────────

    def _set_project(self, project: FaceAuditProject) -> None:
        self._project = project
        self._definitions = list(project.definition_repository.list_definitions())
        self._current_definition = None
        self._project_toolbar.set_project_open(True)
        self._left_panel.set_strategy_actions_enabled(False)
        self._left_panel.set_project_root(project.root)
        if hasattr(self._strategy_editor, "set_project_root"):
            self._strategy_editor.set_project_root(project.root)
        self._left_panel.set_add_resource_enabled(True)
        self._refresh_left_panel()
        self._strategy_editor.show_empty()
        self._view.show_workspace_screen()
        # Projeto recém-aberto não tem alterações pendentes.
        self._is_dirty = False

    def _refresh_left_panel(self) -> None:
        if not self._project:
            return
        self._definitions = list(self._project.definition_repository.list_definitions())

        # CRÍTICO: precisamos manter três instâncias-Python em sincronia
        # — `controller._current_definition`, `editor._definition` e o
        # elemento correspondente em `controller._definitions`. Senão
        # divergem após cada Run/Apply (que faz reload do disco aqui).
        #
        # Histórico: a versão antiga deste método re-apontava
        # `_current_definition` para a instância recém-carregada do disco
        # (`for defn in self._definitions: if match: _current_definition = defn`).
        # Bug: `editor._definition` (setado UMA ÚNICA vez em
        # `_on_strategy_opened`) NÃO era atualizado, ficando órfão. Próximo
        # `strategy_modified` emitido pelo editor (Add/Apply/edit) emitia
        # essa instância órfã com mutações stale (ex: artifact_refs do
        # último sync NÃO estavam nela porque o sync mutou a instância
        # recém-carregada do disco). `_on_strategy_modified` então fazia
        # `_current_definition = órfã_stale` → `_compute_executed_ids_from_disk`
        # via outputs sem artifact_ref → ✅ desaparecia da instrução
        # executada anteriormente.
        #
        # Solução: preservar a instância in-memory (que tem todas as
        # mutações de hidratação/sync) e SUBSTITUÍ-LA em `_definitions`
        # no lugar da versão recém-carregada do disco. Assim editor,
        # controller e a lista compartilham a MESMA instância Python —
        # impossível divergir.
        if self._current_definition is not None:
            target_id = self._current_definition.strategy_id
            for i, defn in enumerate(self._definitions):
                if defn.strategy_id == target_id:
                    self._definitions[i] = self._current_definition
                    break
        # Árvore esquerda mostra TODOS os recursos do projeto (agrupados por
        # estratégia para tipos produzidos). Editor recebe apenas os recursos
        # acessíveis à estratégia atual: globais (datasets, configs) + os
        # produzidos pela própria estratégia.
        all_resources = self._collect_project_resources()
        resource_to_strategy = self._build_resource_to_strategy_map()
        # Preserva o highlight visual da estratégia aberta — populate
        # limpa+reconstrói a lista, então sem informar selected_strategy_id
        # ela perde a seleção depois de toda execução / edição / refresh.
        selected_id = (
            self._current_definition.strategy_id
            if self._current_definition is not None
            else None
        )
        self._left_panel.populate_strategies(
            self._definitions, selected_strategy_id=selected_id
        )
        self._left_panel.populate_resources(all_resources, resource_to_strategy=resource_to_strategy)

        editor_resources = self._collect_editor_resources(self._current_definition)
        self._strategy_editor.update_available_resources(editor_resources)

        # Combo de embedding precisa refletir adições/remoções de datasets
        # primários (importação, bulk add, remoção de strategy). Sem
        # re-chamar aqui, o combo só atualizava ao trocar de estratégia.
        self._strategy_editor.set_available_embedding_columns(
            self._collect_primary_embedding_columns()
        )

    def _collect_editor_resources(self, current_definition) -> List[StrategyResource]:
        """
        Lista de recursos disponíveis aos dropdowns do editor para a estratégia
        atualmente aberta, filtrada por `ResourceCategory` e (para datasets)
        por compatibilidade de embedding:

        - **PRIMÁRIOS** (category=PRIMARY): visíveis em qualquer estratégia.
          DATASETS primários são adicionalmente filtrados por
          `current_definition.embedding_column` — só aparecem aqueles cujo
          schema contém a coluna escolhida (verificada via
          `column_specs` ou metadata do parquet).
        - **DERIVADOS** (category=DERIVED): visíveis SOMENTE quando declarados
          na própria estratégia atual. Datasets transformados, scores, modelos
          treinados etc. produzidos por OUTRAS estratégias ficam invisíveis.
          Não filtra por embedding — derivados foram produzidos PELA estratégia
          atual, então herdam o embedding por construção.

        Sem `current_definition` (nenhuma estratégia aberta) retorna lista vazia.
        """
        if current_definition is None:
            return []

        embedding_col = getattr(current_definition, "embedding_column", None)
        seen_ids: set = set()
        resources: List[StrategyResource] = []

        # 1. PRIMÁRIOS — qualquer estratégia ou disk-scan. DATASETS primários
        # são filtrados pela coluna de embedding selecionada.
        for r in self._collect_project_resources():
            if r.category != ResourceCategory.PRIMARY:
                continue
            if r.resource_id in seen_ids:
                continue
            if (
                embedding_col
                and r.resource_type == StrategyResourceType.DATASET
            ):
                if embedding_col not in self._resource_embedding_columns(r):
                    continue
            seen_ids.add(r.resource_id)
            resources.append(r)

        # 2. DERIVADOS — somente os da estratégia atual.
        for r in current_definition.resources:
            if (
                r.category == ResourceCategory.DERIVED
                and r.resource_id not in seen_ids
            ):
                seen_ids.add(r.resource_id)
                resources.append(r)

        return resources

    def _build_resource_to_strategy_map(self) -> dict:
        """
        Mapeia `id(resource) → nome da estratégia que o declara`.

        Key é a **identidade do objeto Python** (`id()`), não o
        `resource_id`-string. Necessário porque duplicar estratégia
        (`copy.deepcopy`) preserva os mesmos `resource_id`s do original,
        mas cria instâncias distintas — cada instância pertence à sua
        estratégia owner. Sem isso, derivados da cópia ficavam mapeados
        à estratégia original (primeira ocorrência vencia).

        Usado pela árvore de recursos pra sub-agrupar derivados por
        estratégia. Caller deve passar a MESMA lista de instâncias
        retornada por `_collect_project_resources` (não cópias).
        """
        mapping: dict = {}
        for defn in self._definitions:
            for res in defn.resources:
                mapping[id(res)] = defn.name
        return mapping

    def _infer_embedding_dimensions(self, definition) -> Optional[int]:
        """
        Lê a dimensionalidade da `embedding_column` da estratégia a partir
        do `column_specs` do primeiro DATASET que ofereça aquela coluna.

        Funciona pra qualquer coluna com role=EMBEDDING (sepael, DeepFace,
        ArcFace, etc.).
        """
        col = getattr(definition, "embedding_column", None)
        if not col:
            return None

        for resource in (definition.resources or []):
            if resource.resource_type != StrategyResourceType.DATASET:
                continue
            payload = getattr(resource, "payload", None)
            specs = getattr(payload, "column_specs", None) if payload else None
            if not specs:
                continue
            spec = specs.get(col)
            dim = getattr(spec, "dimensions", None) if spec else None
            if isinstance(dim, int) and dim > 0:
                return dim
        return None

    def _resource_embedding_columns(self, resource) -> Set[str]:
        """Retorna o conjunto de colunas role=EMBEDDING de um resource
        DATASET. Lê de `payload.column_specs` se disponível, senão faz
        leitura colunar do metadata do parquet (sem hidratar). Retorna
        set vazio quando não consegue determinar.
        """
        if resource is None or resource.resource_type != StrategyResourceType.DATASET:
            return set()
        payload = getattr(resource, "payload", None)
        specs = getattr(payload, "column_specs", None) if payload else None
        if not specs:
            specs = self._read_column_specs_from_parquet(resource)
        if not specs:
            return set()
        return {
            name for name, spec in specs.items()
            if str(getattr(spec, "role", "")).lower().endswith("embedding")
        }

    def _collect_primary_embedding_columns(self) -> List[str]:
        """
        Coleta a UNIÃO das colunas role=EMBEDDING dos datasets PRIMÁRIOS
        do projeto. Sem primários, retorna lista vazia.

        Derivados não entram — eles herdam o embedding da estratégia que
        os produziu, então restringir o universo por eles era confuso.
        """
        all_cols: Set[str] = set()
        for resource in self._collect_project_resources():
            if resource.category != ResourceCategory.PRIMARY:
                continue
            all_cols |= self._resource_embedding_columns(resource)
        return sorted(all_cols)


    def _read_column_specs_from_parquet(self, resource) -> Optional[dict]:
        """
        Lê só o `faceaudit:column_specs` da metadata do parquet, sem
        hidratar o dataset. Evita pagar o custo de carregar parquets
        gigantes só pra popular o combo de embedding.
        """
        ref = getattr(resource, "artifact_ref", None)
        if not ref or not self._project:
            return None

        try:
            import json as _json
            import pyarrow.parquet as _pq

            from faceaudit.models.column_catalog import ColumnSpec

            abs_path = self._project.storage.resolve_artifact(ref)
            meta = _pq.read_metadata(abs_path).metadata or {}
            specs_raw = meta.get(b"faceaudit:column_specs")
            if not specs_raw:
                return None

            specs_dict = _json.loads(specs_raw.decode("utf-8"))
            # Reconstrói objetos lightweight só pro consumidor checar
            # `.role` e `.dimensions` (não precisamos de ColumnSpec
            # completo aqui — um SimpleNamespace basta).
            from types import SimpleNamespace
            return {
                name: SimpleNamespace(**spec) if isinstance(spec, dict) else spec
                for name, spec in specs_dict.items()
            }
        except Exception:
            return None

    def _collect_project_resources(self) -> List[StrategyResource]:
        """
        Coleta recursos para exibição na árvore:
        1. Recursos declarados nas definições de estratégia.
        2. Arquivos importados em disco mas ainda não referenciados em nenhuma definição.

        **Dedup**:
        - PRIMÁRIOS: deduplicados por `resource_id` (o mesmo dataset BFW
          referenciado por 2 estratégias é a mesma coisa — não duplica).
        - DERIVADOS: NÃO deduplicados. Após `Duplicar estratégia`, a cópia
          carrega derivados com os MESMOS `resource_id` da original
          (deep copy), mas são **instâncias distintas pertencentes a
          estratégias distintas** — devem aparecer ambas no painel.
        """
        seen_primary_ids: set = set()
        seen_primary_paths: set = set()
        seen_derived_paths: set = set()
        resources: List[StrategyResource] = []

        # 1. Recursos declarados nas definições
        for defn in self._definitions:
            for res in defn.resources:
                if res.category == ResourceCategory.PRIMARY:
                    if res.resource_id in seen_primary_ids:
                        continue
                    seen_primary_ids.add(res.resource_id)
                    if res.artifact_ref and res.artifact_ref.path:
                        seen_primary_paths.add(res.artifact_ref.path)
                else:
                    # Rastreia paths de derivados declarados pra evitar que
                    # o disk-scan os redescubra como "primários soltos".
                    # Sem isso, um dataset produzido por DATASET_SPLIT (que
                    # vive em `datasets/{strategy_id}_{rid}.parquet`)
                    # aparecia duas vezes: uma vez DERIVED (passo 1, da
                    # definition) e uma vez PRIMARY (passo 2, varrendo o
                    # disco).
                    if res.artifact_ref and res.artifact_ref.path:
                        seen_derived_paths.add(res.artifact_ref.path)
                # DERIVADOS sempre são adicionados — pertencem à estratégia
                # que os declara, sem dedup cross-strategy.
                resources.append(res)

        # 2. Arquivos importados no disco (não vinculados a nenhuma definição ainda)
        _import_scan = [
            (StrategyArtifactType.GLOBAL_DATASET,  StrategyResourceType.DATASET),
            (StrategyArtifactType.SPLIT_CONFIG,    StrategyResourceType.SPLIT_CONFIG),
            (StrategyArtifactType.MINING_CONFIG,   StrategyResourceType.MINING_CONFIG),
            (StrategyArtifactType.PIPELINE_CONFIG, StrategyResourceType.PIPELINE_CONFIG),
        ]
        for artifact_type, resource_type in _import_scan:
            for ref in self._project.storage.list_artifacts(artifact_type):
                # Skip se o arquivo já está declarado como primário ou
                # derivado de alguma estratégia (ambos os casos cobertos
                # pelas duas estruturas).
                if ref.path in seen_primary_paths:
                    continue
                if ref.path in seen_derived_paths:
                    continue
                resource_id = Path(ref.path).stem
                if resource_id in seen_primary_ids:
                    continue

                # Filtragem defensiva de derivados em pastas que compartilham
                # extensão com primários. Aplica-se SÓ a GLOBAL_DATASET
                # (datasets/*.parquet acomoda tanto SOURCE quanto DERIVED).
                # Configs (pipeline/mining/split) não compartilham folder
                # com derivados, então a checagem aqui pegaria falso-positivos
                # (ex: criar `pipe_v2` com `pipe` já declarado descartaria
                # silenciosamente o novo).
                if artifact_type == StrategyArtifactType.GLOBAL_DATASET:
                    # (1) Stem começa com UUID de strategy_id seguido de
                    # `_` → arquivo produzido pela convenção do executor
                    # (`{strategy_id}_{resource_id}.parquet`). É derivado
                    # por construção, mesmo que a strategy original tenha
                    # sido removida (arquivo órfão).
                    if _UUID_PREFIX_RE.match(resource_id):
                        continue
                    # (2) Stem começa com `{primary_id}_` — match por
                    # PREFIXO + `_` (não substring solto, pra não pegar
                    # nomes onde o id aparece no meio).
                    if any(
                        rid and resource_id.startswith(f"{rid}_")
                        for rid in seen_primary_ids
                    ):
                        continue

                seen_primary_ids.add(resource_id)
                # category=PRIMARY: tudo que vem de `_import_scan` foi
                # importado pelo usuário via "Adicionar Recurso" e vive
                # nas pastas globais do projeto (datasets/, pipelines/,
                # split_configs/, mining_configs/).
                resources.append(StrategyResource(
                    resource_id=resource_id,
                    name=resource_id,
                    resource_type=resource_type,
                    artifact_ref=ref,
                    category=ResourceCategory.PRIMARY,
                ))

        return resources

    def _on_strategy_opened(self, definition: LRStrategyDefinition) -> None:
        # Cursor de espera durante o carregamento da estratégia — popular
        # 3 abas com instruções, configurar dropdowns de recursos e calcular
        # estados de execução pode travar a UI por segundos em estratégias
        # grandes. Sem isso o usuário clica e parece que nada acontece.
        with self._wait_cursor():
            self._current_definition = definition
            # Garante highlight visual no painel lateral mesmo quando a
            # abertura veio por outra via que não o itemClicked (ex: após
            # criar nova estratégia, depois de duplicar, etc.).
            self._left_panel.select_strategy_by_id(definition.strategy_id)
            self._left_panel.set_strategy_actions_enabled(True)
            # Editor recebe apenas recursos acessíveis à estratégia atual:
            # globais + produzidos pela própria estratégia. Produzidos por
            # outras estratégias não aparecem nos dropdowns.
            editor_resources = self._collect_editor_resources(definition)
            self._strategy_editor.load_strategy(definition, available_resources=editor_resources)
            self._reset_execution_state()
            self._refresh_produced_resources()

            # Informa ao left panel as dimensões do embedding da estratégia
            # selecionada. Lê do `column_specs` do primeiro dataset que
            # ofereça a coluna escolhida — substitui o lookup antigo via
            # enum EmbeddingType.dimensions.
            emb_dims = self._infer_embedding_dimensions(definition)
            self._left_panel.set_embedding_dimensions(emb_dims)

            # Atualiza opções do combo de embedding com a UNIÃO das colunas
            # role=EMBEDDING dos datasets PRIMÁRIOS do projeto.
            self._strategy_editor.set_available_embedding_columns(
                self._collect_primary_embedding_columns()
            )

    def _on_new_strategy(self) -> None:
        ctrl = NewStrategyController(
            parent=self._view,
            global_storage=self._global_storage,
        )
        definition = ctrl.exec()
        if definition and self._project:
            try:
                self._project.definition_repository.save_definition(definition)
                self._is_dirty = False
                self._refresh_left_panel()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self._view, "Erro", f"Não foi possível salvar a estratégia:\n{exc}"
                )

    def _on_duplicate_strategy(self, definition: LRStrategyDefinition) -> None:
        if not self._project:
            return

        # WaitCursor: a duplicação pode demorar perceptivelmente em
        # estratégias grandes (deepcopy de muitos resources + iteração
        # por todos os DERIVED zerando refs + regen de ids + save no
        # disco + refresh do painel esquerdo). Sem o cursor, o usuário
        # fica sem feedback de que algo está acontecendo.
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor
        )
        try:
            cloned = copy.deepcopy(definition)
            cloned.strategy_id = str(uuid.uuid4())
            cloned.name = f"{definition.name} (cópia)"

            # CRÍTICO: zerar artifact_ref e payload de TODOS os recursos DERIVED
            # do clone. Sem isso, o clone herda refs apontando para arquivos
            # com o UUID da estratégia ORIGINAL — se o usuário depois deletar
            # o clone, `_on_remove_strategy` percorreria essas refs e
            # APAGARIA OS ARQUIVOS DA ORIGINAL (porque o cleanup vai pelo
            # `artifact_ref.path`, não pelo UUID do dono). Recursos PRIMARY
            # (datasets/configs importados) mantêm os refs porque vivem na
            # biblioteca compartilhada do projeto.
            #
            # Regenerar `id` dos DERIVED (Fase 5): `copy.deepcopy` literalmente
            # duplica os ids. Se mantivéssemos os mesmos ids, quando ambas as
            # estratégias (original + clone) fossem carregadas num mesmo
            # contexto de execução (caso raro mas possível em Run All
            # cross-strategy), `set_resource` do clone sobrescreveria o do
            # original no `_id_to_resource_id_index`. Regenerar o id garante
            # unicidade global em todo o projeto. PRIMARY mantém id porque
            # representa a mesma identidade física do recurso compartilhado.
            from faceaudit.models.strategy.resource_id_generator import (
                generate_resource_id,
            )
            for res in cloned.resources:
                if res.category == ResourceCategory.DERIVED:
                    res.artifact_ref = None
                    res.payload = None
                    res.id = generate_resource_id(res.resource_type)

            try:
                self._project.definition_repository.save_definition(cloned)
                self._is_dirty = False
                self._refresh_left_panel()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self._view, "Erro", f"Não foi possível duplicar a estratégia:\n{exc}"
                )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_rename_strategy(self, definition: LRStrategyDefinition) -> None:
        if not self._project:
            return
        new_name, ok = QtWidgets.QInputDialog.getText(
            self._view,
            "Renomear estratégia",
            "Novo nome:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            definition.name,
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name == definition.name:
            return
        definition.name = new_name
        try:
            self._project.definition_repository.save_definition(definition)
            self._is_dirty = False
            self._refresh_left_panel()
            if self._current_definition and self._current_definition.strategy_id == definition.strategy_id:
                self._strategy_editor.set_strategy_name(new_name)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro", f"Não foi possível renomear a estratégia:\n{exc}"
            )

    def _on_remove_strategy(self, definition: LRStrategyDefinition) -> None:
        """Remove a estratégia e TODOS os seus artefatos derivados em disco.

        Limpeza em 4 frentes (todas best-effort — falha em uma não aborta
        as outras):
          1. Recursos derivados declarados em `definition.resources` —
             apaga `artifact_ref.path` + sidecars conhecidos
             (`_stats.json`, `_dist.json`, `_metrics.json`).
          2. Pasta `<project>/graphics/<strategy_id>/` inteira — onde
             vivem os PNGs gerados pelos persists (distribution, casework,
             validation_result com 4 PNGs).
          3. Pickle do `LRStrategyTrainedModel` (Run All produz isso).
          4. JSON da definição em `definitions/`.

        Recursos PRIMARY (datasets/configs importados pelo usuário) NÃO
        são tocados — eles vivem na biblioteca do projeto, compartilhados
        entre estratégias. O filtro `category=DERIVED` na varredura
        garante isso.
        """
        if not self._project:
            return
        reply = QtWidgets.QMessageBox.question(
            self._view,
            "Remover estratégia",
            f"Remover a estratégia '{definition.name}'?\n"
            f"Todos os recursos derivados (datasets transformados, scores, "
            f"modelos treinados, resultados, gráficos) serão apagados do "
            f"disco. Recursos importados (datasets, configs) permanecem.\n"
            f"Essa ação não pode ser desfeita.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        import shutil
        from faceaudit.enums.project_folder import ProjectFolder
        from faceaudit.enums.strategy_artifact_type import StrategyArtifactType as SAT
        from faceaudit.enums.strategy_file_type import StrategyFileType
        from faceaudit.models.similarity.distribution_plot import _sanitize_name
        from faceaudit.models.storage.strategy_artifact_reference import (
            StrategyArtifactReference,
        )

        project_root = self._project.storage.project_root
        strategy_id = definition.strategy_id
        safe_strategy = _sanitize_name(strategy_id)
        sidecar_suffixes = ("_stats.json", "_dist.json", "_metrics.json")

        try:
            # 1. Arquivos dos recursos derivados declarados.
            for res in definition.resources:
                if res.category != ResourceCategory.DERIVED:
                    continue
                if res.artifact_ref is None or not res.artifact_ref.path:
                    continue
                # GUARD: só apaga se o path do artefato pertencer a ESTA
                # estratégia (UUID dela aparece no caminho). Defesa contra
                # refs órfãs herdadas de duplicates antigos (anteriores ao
                # fix de _on_duplicate_strategy que zera DERIVED refs):
                # se uma estratégia duplicada manteve refs apontando pro
                # UUID da original, deletar a duplicata NÃO PODE apagar
                # arquivos da original. Arquivos órfãos com este caso
                # ficarão para o próximo open detectar via integrity check.
                rel_path = res.artifact_ref.path.replace("\\", "/")
                if strategy_id not in rel_path:
                    continue
                try:
                    abs_path = project_root / res.artifact_ref.path
                    if abs_path.exists():
                        abs_path.unlink()
                    # Sidecars convencionais ao lado do artefato.
                    stem = abs_path.stem
                    for suffix in sidecar_suffixes:
                        sc = abs_path.parent / (stem + suffix)
                        if sc.exists():
                            sc.unlink()
                except Exception:
                    pass  # individual file failure não aborta a varredura

            # 2. Pasta de gráficos da estratégia (recursiva).
            graphics_dir = (
                project_root / ProjectFolder.GRAPHICS.value / safe_strategy
            )
            if graphics_dir.exists():
                shutil.rmtree(graphics_dir, ignore_errors=True)

            # 3. Pickle do LRStrategyTrainedModel (output de Run All).
            try:
                model_rel = StrategyFileType.TRAINED_STRATEGY_MODEL.build_path(
                    strategy_id=strategy_id,
                )
                model_abs = project_root / model_rel
                if model_abs.exists():
                    model_abs.unlink()
            except Exception:
                pass

            # Sweep defensivo: pega órfãos da estratégia em pastas conhecidas
            # de outputs. Cobre arquivos que ficaram fora de
            # `definition.resources` por inconsistência (ex: output cuja
            # referência foi limpa por Apply Alterações mas o file ficou
            # no disco).
            #
            # MODELS/SCORES/RESULTS são pastas hierárquicas (`<folder>/<sid>/...`)
            # — apaga a subpasta inteira da estratégia.
            for folder in (
                ProjectFolder.MODELS,
                ProjectFolder.SCORES,
                ProjectFolder.RESULTS,
            ):
                target_dir = project_root / folder.value / safe_strategy
                if target_dir.exists():
                    shutil.rmtree(target_dir, ignore_errors=True)
            # DATASETS/CHECKPOINTS/ARTIFACTS continuam flat (datasets derivados
            # de PIPELINE_APPLICATION/SPLIT, checkpoints, e arquivos legados
            # de projetos pré-migração). Iterdir + prefixo `<sid>_`.
            for folder in (
                ProjectFolder.DATASETS,
                ProjectFolder.CHECKPOINTS,
                ProjectFolder.ARTIFACTS,
            ):
                fp = project_root / folder.value
                if not fp.exists():
                    continue
                for p in fp.iterdir():
                    if not p.is_file():
                        continue
                    if p.name.startswith(safe_strategy + "_"):
                        try:
                            p.unlink()
                        except Exception:
                            pass

            # 4. JSON da definição.
            rel = StrategyFileType.STRATEGY_DEFINITION.build_path(
                strategy_id=strategy_id,
            )
            ref = StrategyArtifactReference(
                artifact_type=SAT.STRATEGY_DEFINITION, path=str(rel), hash=None,
            )
            abs_path = self._project.storage.resolve_artifact(ref)
            if abs_path and abs_path.exists():
                abs_path.unlink()

            if (
                self._current_definition
                and self._current_definition.strategy_id == strategy_id
            ):
                self._current_definition = None
                self._strategy_editor.show_empty()
                self._left_panel.set_strategy_actions_enabled(False)
            self._refresh_left_panel()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro", f"Não foi possível remover a estratégia:\n{exc}"
            )

    def _on_strategy_modified(self, definition: LRStrategyDefinition) -> None:
        self._current_definition = definition
        self._sync_all_output_placeholders()
        self._reset_execution_state()
        self._refresh_produced_resources()
        # Edição em andamento: marca o projeto como sujo até que algo persista.
        self._is_dirty = True

    # ── Execution state management ─────────────────────────────────────────────

    def _reset_execution_state(self) -> None:
        """Reseta contexto e estados visuais quando a estratégia muda.

        OBS: o conjunto `_executed_instruction_ids` é zerado aqui, mas
        `_refresh_instruction_states` o repovoa imediatamente com base no
        estado do disco — instruções cujos outputs já estão persistidos
        continuam marcadas como executadas. Isso garante que reabrir o
        projeto preserve os ✅.
        """
        self._execution_context = None
        self._executed_instruction_ids.clear()
        self._executor = None
        self._refresh_instruction_states()

    def _delete_persisted_resource_files(
        self,
        resource: StrategyResource,
        strategy_id: Optional[str] = None,
    ) -> None:
        """Apaga do disco artefato + sidecars + PNGs de UM recurso persistido.

        Não mexe em `definition.resources` nem em contexto vivo — caller decide.
        Falhas individuais são engolidas: uma sidecar/PNG faltante ou bloqueada
        não pode abortar a varredura maior.

        `strategy_id` opcional — caller passa quando a estratégia dona não é
        a atual (cascade cross-strategy). Default: estratégia atual.
        """
        if not self._project or resource is None:
            return
        ref = resource.artifact_ref
        if ref is None or not ref.path:
            return

        project_root = self._project.storage.project_root
        if strategy_id is None and self._current_definition is not None:
            strategy_id = self._current_definition.strategy_id

        try:
            abs_path = project_root / ref.path
            if abs_path.exists():
                abs_path.unlink()
            stem = abs_path.stem
            parent = abs_path.parent
            for suffix in ("_stats.json", "_dist.json", "_metrics.json"):
                sc = parent / (stem + suffix)
                if sc.exists():
                    sc.unlink()

            if not strategy_id:
                return

            # PNGs em `<project>/graphics/<strategy_id>/...` — gating
            # por resource_type, mesmo padrão que o executor usa pra gravar.
            if resource.resource_type in (
                StrategyResourceType.SIMILARITY_SCORES,
                StrategyResourceType.CALIBRATED_SCORES,
            ):
                from faceaudit.models.similarity.distribution_plot import (
                    score_distribution_png_path,
                )
                png_path = score_distribution_png_path(
                    project_root, strategy_id, resource.resource_id,
                    calibrated=(
                        resource.resource_type
                        == StrategyResourceType.CALIBRATED_SCORES
                    ),
                )
                if png_path.exists():
                    png_path.unlink()
            elif resource.resource_type == StrategyResourceType.CASEWORK_VISUALIZATION:
                from faceaudit.models.casework_visualization_plot import (
                    casework_visualization_png_path,
                )
                png_path = casework_visualization_png_path(
                    project_root, strategy_id, resource.resource_id,
                )
                if png_path.exists():
                    png_path.unlink()
            elif resource.resource_type == StrategyResourceType.VALIDATION_RESULT:
                from faceaudit.models.validation_result_plot import (
                    validation_result_png_paths,
                )
                for png in validation_result_png_paths(
                    project_root, strategy_id, resource.resource_id,
                ).values():
                    if png.exists():
                        png.unlink()
        except Exception:
            pass  # falha individual não aborta a varredura

    def _delete_instruction_output_files(
        self,
        instruction: StrategyInstructionDefinition,
        definition: Optional[LRStrategyDefinition] = None,
    ) -> List[str]:
        """Apaga arquivos físicos de TODOS os outputs declarados pela
        instrução: artefato principal + `_stats.json` / `_dist.json` /
        `_metrics.json` sidecars + PNG(s) em `<project>/graphics/`.

        NÃO mexe em `definition.resources` nem no contexto — só toca disco.
        Caller decide se mantém placeholders (Apply Alterações) ou se
        remove tudo (remoção da instrução).

        `definition` opcional — quando None usa `self._current_definition`.
        Necessário pra invalidação em cascata cross-strategy (edição de
        config compartilhada via ✎ pode atingir instruções de OUTRAS
        estratégias que não são a atualmente selecionada).

        Retorna a lista de `output_resource_id` encontrados na instrução
        (independente de terem arquivo no disco) — útil pra o caller saber
        em quais placeholders mexer.
        """
        if definition is None:
            definition = self._current_definition
        if not self._project or definition is None or instruction is None:
            return []

        try:
            specs = instruction.get_resource_field_specs()
        except Exception:
            return []

        strategy_id = definition.strategy_id
        touched: List[str] = []

        for spec in specs:
            if not spec.is_output:
                continue
            output_id = getattr(instruction, spec.field_name, None)
            if not output_id:
                continue
            touched.append(output_id)

            existing = definition.get_resource_by_id(output_id)
            if existing is None:
                continue
            self._delete_persisted_resource_files(existing, strategy_id=strategy_id)

        return touched

    def _invalidate_dependents_of(self, root_unique_id: str) -> None:
        """Invalida em cascata todas as instruções que dependem direta ou
        transitivamente do recurso editado.

        Algoritmo BFS sobre o grafo de dependência usando `StrategyResource.id`
        como chave (não `resource_id`, que é apenas o nome legível e PODE
        repetir entre estratégias quando duplicate preservou labels).

        Escopo natural:
        - **DERIVED**: cada DERIVED tem `id` único globalmente (regenerado
          pelo `_on_duplicate_strategy` via `generate_resource_id`). O BFS
          só atinge instruções DA MESMA estratégia que dona o derivado —
          sem precisar de filtro de strategy_id.
        - **PRIMARY**: configs/datasets importados compartilham `id` entre
          todas as estratégias que os referenciam (representam mesma
          identidade física). O BFS naturalmente atinge todas — exatamente
          o que se espera de edição de config compartilhada via ✎ ou
          re-save de dataset primário externo.

        O mapeamento `resource_id → id` é local por estratégia, porque
        cada `instr.input_resource_ids` usa `resource_id` (label).
        """
        if not self._project or not root_unique_id:
            return

        dirty_ids: Set[str] = {root_unique_id}
        processed: Set[str] = set()
        touched_strategy_ids: Set[str] = set()

        while True:
            to_process = dirty_ids - processed
            if not to_process:
                break
            processed |= to_process

            for defn in self._definitions:
                # Map local desta defn: resource_id (label) -> id (chave única).
                # Recalculado a cada iteração porque outputs invalidados
                # podem ter sido removidos/zerados em iterações anteriores
                # (raro mas defensivo).
                rid_to_uid = {
                    r.resource_id: r.id
                    for r in defn.resources
                    if r.id
                }
                for instr in defn.instructions:
                    input_rids = getattr(instr, "input_resource_ids", []) or []
                    # Resolve cada input para id único via mapeamento local.
                    input_uids = {
                        rid_to_uid[rid]
                        for rid in input_rids
                        if rid in rid_to_uid
                    }
                    if not (input_uids & to_process):
                        continue

                    # Apaga arquivos físicos + descobre os output rids da
                    # instrução. `_delete_instruction_output_files` retorna
                    # `resource_id`s (labels locais à defn) — convertemos
                    # pra ids únicos via rid_to_uid pra propagar no BFS.
                    output_rids = self._delete_instruction_output_files(
                        instr, definition=defn,
                    )
                    if not output_rids:
                        continue

                    for output_rid in output_rids:
                        existing = defn.get_resource_by_id(output_rid)
                        if existing is None:
                            continue
                        existing.artifact_ref = None
                        existing.payload = None

                        if existing.id:
                            dirty_ids.add(existing.id)

                        # Remove do contexto vivo apenas quando a definition
                        # invalidada é a atual (contexto é por estratégia).
                        if (
                            defn is self._current_definition
                            and self._execution_context is not None
                            and self._execution_context.has_resource(output_rid)
                        ):
                            try:
                                self._execution_context.remove_resource(output_rid)
                            except Exception:
                                pass

                    self._executed_instruction_ids.discard(instr.instruction_id)
                    touched_strategy_ids.add(defn.strategy_id)

        # GATE pela presença de instruções atingidas pelo BFS, NÃO por
        # `any_invalidated` (que só vira True quando algum output existia
        # em `defn.resources` e teve `artifact_ref` zerado). Se uma
        # instrução foi alcançada pelo cascade mas os placeholders dos
        # outputs estão fora de sync com `defn.resources` por alguma razão,
        # `_executed_instruction_ids.discard` E `_delete_instruction_output_files`
        # já rodaram — precisamos persistir os discards + atualizar UI
        # mesmo se nenhum `existing.artifact_ref = None` aconteceu, senão
        # o disco fica inconsistente (arquivos apagados, mas JSON ainda
        # marca como executada) e a UI segue mostrando ✅ falso.
        if not touched_strategy_ids:
            return

        # Persiste todas as definitions afetadas. Erros são surfaceados —
        # se a invalidação fica só em memória mas não chega ao disco, o
        # próximo open mostra estratégias com ✅ falso em outputs que já
        # foram deletados do disco.
        save_errors: List[str] = []
        for defn in self._definitions:
            if defn.strategy_id in touched_strategy_ids:
                try:
                    self._project.definition_repository.save_definition(defn)
                except Exception as exc:
                    save_errors.append(f"  • {defn.name} ({defn.strategy_id}): {exc}")
        if save_errors:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao persistir invalidação",
                "Falha ao salvar a definition após invalidação em cascata:\n\n"
                + "\n".join(save_errors),
            )

        # Refresh visual: gates 🔒/✅ + painel de produced resources + árvore.
        self._refresh_instruction_states()
        self._refresh_produced_resources()
        self._refresh_left_panel()

    def _on_instruction_applied(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Disparado quando o usuário clica "Aplicar alterações" numa instrução.
        Invalida os outputs persistidos: limpa artifact_ref/payload do recurso
        e apaga o arquivo do disco. Resultado: a instrução perde o ✅ até ser
        reexecutada.
        """
        if not self._project or not self._current_definition or instruction is None:
            return

        output_ids = self._delete_instruction_output_files(instruction)
        if not output_ids:
            return

        any_invalidated = False
        for output_id in output_ids:
            existing = self._current_definition.get_resource_by_id(output_id)
            if existing is None:
                continue
            # Limpa placeholder em memória — instrução PODE ser reexecutada
            # e produzir o output de novo; mantém o resource pra que o
            # `_sync_all_output_placeholders` não tente recriar.
            existing.artifact_ref = None
            existing.payload = None
            any_invalidated = True

            # Remove do contexto vivo — força nova execução pra repopular.
            if (
                self._execution_context is not None
                and self._execution_context.has_resource(output_id)
            ):
                try:
                    self._execution_context.remove_resource(output_id)
                except Exception:
                    pass

        if any_invalidated:
            self._executed_instruction_ids.discard(instruction.instruction_id)
            self._save_current_definition()
            # Cascade downstream: instruções que consumiam os outputs
            # invalidados precisam perder o ✅ E ter seus próprios outputs
            # apagados. Sem isso, Y permanece marcada como executada apesar
            # dos inputs estarem zerados — UI mente sobre o estado válido
            # da estratégia. _invalidate_dependents_of salva as definitions
            # afetadas e dispara os refreshes visuais ao final.
            # Cascade usa o `id` único do recurso DERIVED. Como o
            # _on_duplicate_strategy regenera ids dos DERIVED, esse id
            # é exclusivo da estratégia atual — o BFS não atinge outras.
            for output_rid in output_ids:
                res = self._current_definition.get_resource_by_id(output_rid)
                if res is None or not res.id:
                    continue
                self._invalidate_dependents_of(res.id)
            # Recursos foram invalidados — instruções DOWNSTREAM que dependem
            # delas precisam ser revalidadas pelo gate de dependência.
            self._refresh_instruction_states()
            self._refresh_produced_resources()
            self._refresh_left_panel()

    def _on_instruction_removed(self, instruction: StrategyInstructionDefinition) -> None:
        """Disparado pelo editor ANTES da instrução sumir do phase_list.

        Diferente de Apply Alterações (que invalida outputs mas mantém os
        placeholders na expectativa de reexecução), aqui a instrução
        deixa de existir — então também removemos os placeholders dos
        recursos produzidos de `definition.resources`. Senão eles ficariam
        órfãos na árvore, sem instrução pai pra reproduzí-los.
        """
        if not self._project or not self._current_definition or instruction is None:
            return

        output_ids = self._delete_instruction_output_files(instruction)
        if not output_ids:
            return

        # Captura os `id` únicos dos outputs ANTES de qualquer mutação em
        # `defn.resources`. _invalidate_dependents_of recebe esses ids como
        # raiz do BFS e os usa pra encontrar instruções dependentes via
        # comparação com `rid_to_uid` (mapeamento local da defn).
        output_unique_ids: List[str] = []
        for output_rid in output_ids:
            res = self._current_definition.get_resource_by_id(output_rid)
            if res is not None and res.id:
                output_unique_ids.append(res.id)

        # PRIMEIRO o cascade downstream: invalida instruções que consumiam
        # os outputs como input. Crítico rodar AQUI (antes de remover os
        # placeholders) porque o BFS constrói `rid_to_uid` a partir de
        # `defn.resources` — se removermos antes, downstream input_rids
        # não mapeiam para ids únicos e o cascade não propaga.
        # Sem isso, Y permanece com ✅ apesar dos inputs terem sumido —
        # UI mostra estado válido falso, execuções subsequentes falham tardio.
        # _invalidate_dependents_of já salva as definitions afetadas e
        # dispara os refreshes visuais (✅/🔒 + tree + produced resources).
        for unique_id in output_unique_ids:
            self._invalidate_dependents_of(unique_id)

        # Agora remove placeholders dos outputs de `definition.resources`.
        # Diferente de Apply Alterações (que invalida outputs mas mantém os
        # placeholders na expectativa de reexecução), aqui a instrução
        # deixa de existir — então também removemos os placeholders dos
        # recursos produzidos. Senão eles ficariam órfãos na árvore, sem
        # instrução pai pra reproduzí-los.
        output_id_set = set(output_ids)
        self._current_definition.resources = [
            r for r in self._current_definition.resources
            if r.resource_id not in output_id_set
        ]

        # Remove também do contexto vivo (se existirem).
        for output_id in output_ids:
            if (
                self._execution_context is not None
                and self._execution_context.has_resource(output_id)
            ):
                try:
                    self._execution_context.remove_resource(output_id)
                except Exception:
                    pass

        self._executed_instruction_ids.discard(instruction.instruction_id)

        # NÃO chamamos `_save_current_definition()` aqui — o save virá em
        # seguida via `_on_strategy_modified` (emitido pelo widget logo
        # depois desse signal). Salvar duas vezes seria redundante.

    def _on_embedding_column_change_requested(
        self, old_value: str, new_value: Optional[str],
    ) -> None:
        """Disparado quando o usuário escolhe outra coluna de embedding no
        combo do header. Mostra diálogo de confirmação listando datasets
        primários incompatíveis e quantidade de derivados que serão
        invalidados. Aceita → cascade total + atualiza definition. Cancela
        → reverte o combo pro `old_value`.
        """
        defn = self._current_definition
        if defn is None or not new_value:
            # Sem definition ou tentativa de "desselecionar" — só reverte.
            self._strategy_editor.set_embedding_column_silent(
                old_value or None
            )
            return

        # Datasets primários referenciados por instruções da estratégia
        # que NÃO têm a nova coluna.
        incompatible: List[str] = []
        used_dataset_ids = self._used_primary_dataset_ids(defn)
        for ds_id in used_dataset_ids:
            res = defn.get_resource_by_id(ds_id)
            if res is None:
                continue
            cols = self._resource_embedding_columns(res)
            if new_value not in cols:
                incompatible.append(ds_id)

        # Conta derivados a invalidar (todos os que têm artifact_ref).
        derived_with_files = [
            r for r in defn.resources
            if r.category == ResourceCategory.DERIVED and r.artifact_ref
        ]

        # Estratégia limpa (sem derivados nem incompatíveis) → troca direto
        # sem diálogo. UX: não atrapalha quem está só configurando.
        if not derived_with_files and not incompatible:
            self._strategy_editor.set_embedding_column_silent(new_value)
            self._save_current_definition()
            self._refresh_left_panel()
            return

        # Monta mensagem do diálogo
        lines: List[str] = [
            f"Mudar o embedding de "
            f"<b>{old_value or '(nenhum)'}</b> para <b>{new_value}</b>?",
            "",
        ]
        if incompatible:
            sample = ", ".join(incompatible[:5])
            extra = f" (+{len(incompatible) - 5})" if len(incompatible) > 5 else ""
            lines.append(
                f"⚠ <b>{len(incompatible)} dataset(s) em uso</b> não têm a "
                f"coluna '{new_value}': {sample}{extra}. Eles ficarão "
                "indisponíveis nos dropdowns das instruções até serem "
                "trocados ou re-enriquecidos."
            )
            lines.append("")
        if derived_with_files:
            lines.append(
                f"⚠ <b>{len(derived_with_files)} recurso(s) derivado(s)</b> "
                "produzidos pelas instruções desta estratégia serão "
                "invalidados (arquivos apagados, ✅ removidos)."
            )
            lines.append("")
        lines.append("Confirmar a troca?")

        reply = QtWidgets.QMessageBox.question(
            self._view,
            "Trocar embedding da estratégia",
            "<br>".join(lines),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            self._strategy_editor.set_embedding_column_silent(
                old_value or None
            )
            return

        # Aceito: cascade total + grava nova coluna.
        self._invalidate_all_derived(defn)
        self._strategy_editor.set_embedding_column_silent(new_value)
        self._save_current_definition()
        self._refresh_left_panel()
        self._refresh_instruction_states()
        self._refresh_produced_resources()

    @staticmethod
    def _used_primary_dataset_ids(defn: LRStrategyDefinition) -> List[str]:
        """IDs de DATASETs PRIMÁRIOS referenciados como input por alguma
        instrução da estratégia. Usado pra detectar incompatibilidades
        ao trocar embedding."""
        resources_by_id = {r.resource_id: r for r in (defn.resources or [])}
        seen: Set[str] = set()
        ids: List[str] = []
        for instr in (defn.instructions or []):
            for rid in (getattr(instr, "input_resource_ids", None) or []):
                if rid in seen:
                    continue
                res = resources_by_id.get(rid)
                if res is None:
                    continue
                if (
                    res.resource_type == StrategyResourceType.DATASET
                    and res.category == ResourceCategory.PRIMARY
                ):
                    ids.append(rid)
                    seen.add(rid)
        return ids

    def _invalidate_all_derived(
        self, defn: LRStrategyDefinition,
    ) -> int:
        """Invalida TODOS os recursos derivados de uma estratégia: apaga
        arquivos, zera artifact_ref/payload, tira o ✅ das instruções.
        Usado pelo cascade ao trocar embedding (afeta toda a estratégia,
        não só a partir de uma raiz).

        Também zera o `payload` dos PRIMARY DATASETs (sem deletar arquivo —
        arquivos primários não pertencem à estratégia). Sem isso, payload
        hidratado em run anterior com `embedding_column` ANTIGO permanece
        em memória; a próxima run reusa via early-return de `ensure_payload`
        ignorando o novo `embedding_column`. Caso típico: usuário começa
        run com coluna errada, cancela, troca coluna, re-roda — sem essa
        invalidação, o segundo run usa silenciosamente os vetores da coluna
        errada.

        Diferente do `_invalidate_dependents_of` (BFS a partir de um
        resource_id) — aqui invalidamos sem percorrer grafo, já que
        embedding muda o universo inteiro.
        """
        if not self._project:
            return 0

        invalidated = 0
        for instr in (defn.instructions or []):
            output_ids = self._delete_instruction_output_files(
                instr, definition=defn,
            )
            for output_id in output_ids:
                res = defn.get_resource_by_id(output_id)
                if res is None:
                    continue
                res.artifact_ref = None
                res.payload = None
                invalidated += 1
                if (
                    self._execution_context is not None
                    and self._execution_context.has_resource(output_id)
                ):
                    try:
                        self._execution_context.remove_resource(output_id)
                    except Exception:
                        pass
            self._executed_instruction_ids.discard(instr.instruction_id)

        # Zera payload dos PRIMARY DATASETs (mantém artifact_ref —
        # arquivo continua no disco, só força rehidratação na próxima run
        # com a nova `embedding_column`).
        for res in (defn.resources or []):
            if (
                res.resource_type == StrategyResourceType.DATASET
                and res.category == ResourceCategory.PRIMARY
            ):
                res.payload = None
                if (
                    self._execution_context is not None
                    and self._execution_context.has_resource(res.resource_id)
                ):
                    ctx_res = self._execution_context.get_resource(res.resource_id)
                    if ctx_res is not None:
                        ctx_res.payload = None

        return invalidated

    def _migrate_legacy_artifacts_layout_if_needed(self) -> bool:
        """Detecta projeto com layout pré-reorg (`artifacts/` populada) e
        oferece migração wipe-and-rerun via diálogo.

        Retorna True se a migração foi executada (caller deve pular o
        `_process_integrity_results` do estado pré-migração — após wipe
        os results ficam stale e este método já roda check_integrity novo).
        Retorna False se não há layout legado OU se o usuário cancelou.

        Estratégia escolhida pelo usuário: NÃO mover arquivos pra paths
        novos (path-rewriting). Em vez disso, apagar tudo que é DERIVED
        e re-executar instruções — os outputs reescrevem nas novas
        subpastas (`models/<sid>/`, `scores/<sid>/`, `results/<sid>/`).
        Recursos PRIMARY (datasets/configs importados) permanecem
        intactos — não vivem em `artifacts/`.
        """
        if not self._project or not self._project.has_legacy_layout:
            return False

        import shutil

        legacy_dir = self._project.root / "artifacts"
        try:
            legacy_files = [p for p in legacy_dir.iterdir() if p.is_file()]
        except Exception:
            legacy_files = []
        n_files = len(legacy_files)
        n_strategies = len(self._definitions)

        # Lista até 3 nomes de estratégias afetadas pra exibir contexto.
        sample_names = [d.name for d in self._definitions[:3]]
        sample_str = ", ".join(sample_names)
        if n_strategies > 3:
            sample_str += f", ... (+{n_strategies - 3})"

        reply = QtWidgets.QMessageBox.question(
            self._view,
            "Migração de layout do projeto",
            (
                "Este projeto tem o layout antigo de pasta <code>artifacts/</code>. "
                "A nova versão organiza artefatos em três pastas hierárquicas:<br><br>"
                "&nbsp;&nbsp;• <code>models/&lt;estratégia&gt;/</code> "
                "(pipeline, similarity, calibration, typicality)<br>"
                "&nbsp;&nbsp;• <code>scores/&lt;estratégia&gt;/</code> "
                "(raw, mined, vectorized, calibrated)<br>"
                "&nbsp;&nbsp;• <code>results/&lt;estratégia&gt;/</code> "
                "(validation, casework, hubness, trajectory)<br><br>"
                f"<b>{n_files} arquivo(s)</b> serão apagados de "
                f"<code>artifacts/</code> e os recursos derivados das "
                f"<b>{n_strategies} estratégia(s)</b> serão invalidados — "
                "você precisará re-executar as instruções pra repopular os "
                f"outputs no novo layout.<br><br>"
                f"Estratégias afetadas: {sample_str}<br><br>"
                "Recursos primários (datasets, pipelines, configs importados) "
                "<b>não serão tocados</b>. Confirmar migração?"
            ),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,  # default conservador
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return False

        try:
            QtWidgets.QApplication.setOverrideCursor(
                QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
            )

            # 1. Pra cada definition: invalida TODOS os derivados
            #    (apaga arquivos das subpastas novas se existirem,
            #    zera artifact_ref/payload, remove do contexto vivo).
            for defn in self._definitions:
                self._current_definition = defn  # _invalidate_all_derived usa o atual pra cleanup do contexto
                self._invalidate_all_derived(defn)
                try:
                    self._project.definition_repository.save_definition(defn)
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(
                        self._view,
                        "Erro ao salvar definição",
                        f"Falha ao salvar '{defn.name}' após migração: {exc}",
                    )
            self._current_definition = None

            # 2. Remove a pasta `artifacts/` inteira — derivados antigos
            #    estavam todos lá; recursos primários nunca vivem em
            #    `artifacts/`. Atomicidade é best-effort.
            try:
                shutil.rmtree(legacy_dir, ignore_errors=True)
            except Exception:
                pass

            # 3. Atualiza state em memória + re-roda integridade.
            self._executed_instruction_ids.clear()
            results = self._project.check_integrity()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        # 4. Refresh UI completa.
        self._refresh_left_panel()
        self._refresh_instruction_states()
        self._refresh_produced_resources()

        QtWidgets.QMessageBox.information(
            self._view,
            "Migração concluída",
            (
                f"{n_files} arquivo(s) legados removidos e {n_strategies} "
                "estratégia(s) invalidadas. Execute as instruções pra "
                "repopular os outputs no novo layout."
            ),
        )

        # Reporta integridade pós-migração se ainda houver problemas
        # (raro, mas pode pegar arquivos órfãos em datasets/ ou outras
        # pastas que ficaram desreferenciados).
        if results and any(r.is_problem for r in results):
            self._process_integrity_results(results)

        return True

    def _compute_executed_ids_from_disk(self) -> set:
        """
        Retorna o conjunto de instruction_ids cujos outputs declarados estão
        TODOS persistidos no projeto (têm StrategyResource com artifact_ref
        na definição). Usado para reconstruir o estado "executada" ao abrir
        o projeto e para sobreviver a resets em memória.
        """
        executed: set = set()
        if not self._current_definition:
            return executed

        for instr in self._current_definition.instructions:
            try:
                specs = instr.get_resource_field_specs()
            except Exception:
                continue
            output_specs = [s for s in specs if s.is_output]
            if not output_specs:
                # Sem outputs declarados — não dá para inferir do disco.
                continue
            all_persisted = True
            for spec in output_specs:
                resource_id = getattr(instr, spec.field_name, None)
                if not resource_id:
                    all_persisted = False
                    break
                existing = self._current_definition.get_resource_by_id(resource_id)
                if existing is None or not existing.has_artifact:
                    all_persisted = False
                    break
            if all_persisted:
                executed.add(instr.instruction_id)
        return executed

    def _wait_cursor(self):
        """
        Context manager para mostrar cursor de espera durante operações
        bloqueantes (hidratar payloads grandes, abrir dialogs que renderizam
        gráficos). Usar como `with self._wait_cursor(): ...`. Garante
        restoreOverrideCursor mesmo em caso de exceção e força repaint
        antes de bloquear via processEvents().
        """
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            QtWidgets.QApplication.setOverrideCursor(
                QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
            )
            try:
                QtWidgets.QApplication.processEvents()
                yield
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        return _cm()

    def _ensure_executor_and_context(self, silent: bool = False) -> bool:
        """Cria executor e contexto se ainda não existem. Retorna False em caso de erro.

        Quando silent=True, falhas não exibem QMessageBox — útil para refreshes
        passivos (ex: cálculo de quais instruções são runnable enquanto o usuário
        ainda está editando).
        """
        if self._executor is None:
            try:
                self._executor = StrategyExecutorFactory.build(
                    artifact_storage=self._project.storage,
                )
                # Propaga o resource_loader do executor para o editor (e
                # daí para a results panel + cards). Sem isso, o
                # InstructionResultsPanel não consegue extrair project_root
                # e o ResourceInfoWidget(compact=True) embedado nos
                # outputs renderiza com paths/hashes/stats vazios.
                loader = getattr(self._executor, "_resource_loader", None)
                if loader is not None and hasattr(self._strategy_editor, "set_resource_loader"):
                    self._strategy_editor.set_resource_loader(loader)
            except Exception as exc:
                if not silent:
                    QtWidgets.QMessageBox.critical(
                        self._view, "Erro", f"Não foi possível inicializar executor:\n{exc}"
                    )
                return False

        if self._execution_context is None:
            try:
                self._execution_context = self._executor.prepare_context(
                    self._current_definition
                )
                # Registra recursos do projeto que não estão na definição da estratégia
                # (ex: recursos importados no disco ou de outras estratégias).
                # Os dropdowns do config panel mostram todos os recursos do projeto,
                # então o contexto precisa contê-los para que a validação e execução
                # possam encontrá-los.
                self._register_project_resources_in_context()

                # Armazena as dimensões do embedding da estratégia no contexto
                # para que regras de validação possam usar como fallback.
                # Lê do column_specs do primeiro dataset com a coluna
                # escolhida (substitui o lookup antigo via enum).
                if self._current_definition:
                    emb_dims = self._infer_embedding_dimensions(self._current_definition)
                    if emb_dims is not None:
                        self._execution_context.set_metadata(
                            "embedding_dimensions", emb_dims
                        )
            except Exception as exc:
                if not silent:
                    QtWidgets.QMessageBox.critical(
                        self._view, "Erro", f"Não foi possível preparar contexto:\n{exc}"
                    )
                self._execution_context = None
                return False

        return True

    def _register_project_resources_in_context(self) -> None:
        """
        Registra no contexto recursos do projeto que não estão na definição
        da estratégia, SEM hidratar.

        Esses resources existem no contexto apenas para popular dropdowns da
        UI (que precisam só de resource_id/resource_type). Quando uma instrução
        for executada e referenciar um deles, `context.ensure_payload(rid)`
        carrega do disco sob demanda — chamado por `_rehydrate_inputs_if_needed`.
        """
        if self._execution_context is None or self._executor is None:
            return

        for resource in self._collect_project_resources():
            if self._execution_context.has_resource(resource.resource_id):
                continue
            self._execution_context.set_resource(resource)

    def _refresh_instruction_states(self) -> None:
        """Recalcula quais instruções são executáveis e atualiza a UI."""
        if not self._current_definition:
            return

        # Garante executor + contexto silenciosamente. Sem isso, o gate de
        # dependência nunca pode classificar instruções como runnable e o
        # botão Run fica permanentemente desabilitado até o usuário clicar
        # em Validar/Run (chicken-and-egg).
        if self._project is not None:
            self._ensure_executor_and_context(silent=True)

        # REPLACEMENT (=) em vez de UNION (|=): o estado de disco é
        # sempre a fonte autoritativa. Antes esta linha era `|=`, que
        # corrompia o ✅ visual após edições de pipeline/mining: o
        # cascade descartava IDs em memória, depois esta linha re-
        # adicionava se `_compute_executed_ids_from_disk` retornasse
        # algo (caso raro mas observado), porque a união não tem como
        # "remover" — só agrega. Com replacement, a função sempre
        # reflete o estado de disco pós-cascade (artifact_refs zerados
        # → instrução excluída do set). Cenários que dependiam do
        # comportamento de união (Run All, Run Single) continuam
        # funcionando porque chamam `_sync_*_to_definition` ANTES desta
        # função, garantindo que o disco já tenha os artifact_refs
        # populados quando `_compute_executed_ids_from_disk` lê.
        self._executed_instruction_ids = self._compute_executed_ids_from_disk()

        runnable_ids: set = set()
        if self._execution_context is not None and self._executor is not None:
            for instr in self._current_definition.instructions:
                if instr.instruction_id in self._executed_instruction_ids:
                    continue
                can_run, _ = self._executor.can_execute_instruction(
                    instr, self._execution_context
                )
                if can_run:
                    runnable_ids.add(instr.instruction_id)

        self._strategy_editor.update_instruction_states(
            runnable_ids=runnable_ids,
            executed_ids=self._executed_instruction_ids,
        )

    def _on_validate_instruction(self, instruction: StrategyInstructionDefinition) -> None:
        if not self._project or not self._current_definition:
            return

        if not self._ensure_executor_and_context():
            return

        try:
            executor_instance = self._executor.instruction_executor_registry.resolve(
                instruction.instruction_type
            )
            result = executor_instance.validate(
                instruction=instruction,
                context=self._execution_context,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro de Validação", f"Erro ao validar instrução:\n{exc}"
            )
            return

        self._show_validation_result(result, instruction.name)

    def _show_validation_result(self, result: ValidationResult, instruction_name: str) -> None:
        if result.is_valid and not result.has_warnings:
            QtWidgets.QMessageBox.information(
                self._view,
                "Validação OK",
                f"Instrução '{instruction_name}' validada com sucesso.",
            )
            return

        parts = []
        if result.errors:
            parts.append("ERROS:\n" + "\n".join(f"  • {e}" for e in result.errors))
        if result.warnings:
            parts.append("AVISOS:\n" + "\n".join(f"  • {w}" for w in result.warnings))

        icon = (
            QtWidgets.QMessageBox.Icon.Critical if result.errors
            else QtWidgets.QMessageBox.Icon.Warning
        )
        msg = QtWidgets.QMessageBox(
            icon, "Resultado da Validação", "\n\n".join(parts), parent=self._view
        )
        msg.exec()

    def _on_view_trajectory(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Abre o TransformationTrajectoryViewerController (arch antiga, reusado)
        com a TransformationTrajectoryData associada à instrução PIPELINE_APPLICATION.

        Resolve o recurso na ordem:
          1. Contexto de execução (em memória).
          2. Definição da estratégia (com hidratação se vier do disco).
        """
        import traceback
        from faceaudit.models.instructions.pipeline_application_instruction import (
            PipelineApplicationInstruction,
        )

        if not self._project or not self._current_definition:
            return

        if not isinstance(instruction, PipelineApplicationInstruction):
            return

        trajectory_id = (
            instruction.trajectory_data_resource_id
            or f"{instruction.instruction_id}_trajectory"
        )

        trajectory_data = None
        hydrate_error = None

        with self._wait_cursor():
            # 1. Tentar do contexto vivo.
            if self._execution_context is not None and self._execution_context.has_resource(trajectory_id):
                ctx_resource = self._execution_context.get_resource(trajectory_id)
                if ctx_resource and ctx_resource.has_payload:
                    trajectory_data = ctx_resource.payload

            # 2. Tentar da definição (hidrata do disco se necessário).
            if trajectory_data is None:
                existing = self._current_definition.get_resource_by_id(trajectory_id)
                if existing is not None:
                    if existing.has_payload:
                        trajectory_data = existing.payload
                    elif existing.has_artifact:
                        loader = getattr(self._executor, "_resource_loader", None) if self._executor else None
                        if loader is None:
                            if self._ensure_executor_and_context(silent=True):
                                loader = getattr(self._executor, "_resource_loader", None)
                        if loader is not None:
                            try:
                                loader.hydrate(existing)
                                if existing.has_payload:
                                    trajectory_data = existing.payload
                            except Exception as hydrate_exc:
                                traceback.print_exc()
                                hydrate_error = str(hydrate_exc)

        if hydrate_error is not None:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao carregar trajetória",
                f"Falha ao carregar a trajetória do disco:\n{hydrate_error}",
            )
            return

        if trajectory_data is None:
            QtWidgets.QMessageBox.information(
                self._view,
                "Trajetória indisponível",
                f"Não foi encontrada uma trajetória 3D para '{trajectory_id}'. "
                "Execute o pipeline com 'Salvar dados de trajetória' ativo.",
            )
            return

        open_error = None
        with self._wait_cursor():
            try:
                from interface.tabs.strategy_manager.transformation_trajectory_viewer.transformation_trajectory_viewer_controller import (
                    TransformationTrajectoryViewerController,
                )
                self._trajectory_viewer = TransformationTrajectoryViewerController(
                    trajectory_data, parent=self._view
                )
                # Em macOS, dialogs criados em runtime podem nascer atrás da janela
                # principal. raise_() + activateWindow() força foco antes do exec().
                view = self._trajectory_viewer.view
                view.show()
                view.raise_()
                view.activateWindow()
            except Exception as exc:
                traceback.print_exc()
                open_error = str(exc)

        if open_error is not None:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao abrir visualizador",
                f"Não foi possível abrir o visualizador de trajetória:\n{open_error}",
            )

    # =========================================================================
    # PHASE 3 — Adição/save de regras de validação (template-only)
    # =========================================================================

    def _load_validation_rule_templates(self) -> list:
        """
        Lista templates de ValidationRuleTemplate disponíveis em duas
        fontes: biblioteca global + projeto atual. Falhas em arquivos
        individuais (JSON corrompido, schema antigo) são logadas no
        stdout e o template é pulado — não bloqueia o resto.
        """
        from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
        from faceaudit.models.validation.validation_rule_template import (
            ValidationRuleTemplate,
        )
        import json as _json

        templates: list = []

        def _load_from(storage, label: str) -> None:
            if storage is None:
                return
            try:
                refs = storage.list_artifacts(
                    StrategyArtifactType.VALIDATION_RULE_TEMPLATE
                )
            except Exception:
                return
            for ref in refs:
                abs_path = storage.resolve_artifact(ref)
                if abs_path is None or not abs_path.exists():
                    continue
                try:
                    data = _json.loads(abs_path.read_text(encoding="utf-8"))
                    tpl = ValidationRuleTemplate.from_dict(data)
                except Exception as exc:
                    print(
                        f"[validation_rule_template] Falha ao carregar "
                        f"'{ref.path}' ({label}): {exc}"
                    )
                    continue
                if tpl is not None:
                    templates.append(tpl)

        _load_from(self._global_storage, "global")
        if self._project is not None:
            _load_from(self._project.storage, "projeto")
        return templates

    def _resources_available_at_instruction_validation(
        self,
        instruction: StrategyInstructionDefinition,
        definition: Optional[LRStrategyDefinition],
    ) -> List[StrategyResource]:
        """
        Filtra os recursos que estarão **garantidamente** no contexto
        no momento em que a validação de `instruction` rodar.

        A regra é simples:
        1. PRIMARIES — registrados desde `prepare_context`, sempre lá.
        2. `instruction.input_resource_ids` — gate de dependência exige
           que estejam no contexto antes de X poder ser validada.
        3. `output_resource_ids` de instruções que vêm ANTES de
           `instruction` em ordem de execução (preparação → validação →
           análise de caso; dentro da fase, ordem da lista).

        Recursos **excluídos** intencionalmente:
        - Outputs da própria `instruction` (não existem ainda — X não
          rodou na hora da validação).
        - Outputs de instruções posteriores (ainda não produzidos).

        Sem isso, o usuário poderia anexar uma regra que referencia algo
        ainda não criado, e a regra falharia silenciosamente com "recurso
        não encontrado" a cada validação. Em vez de validar de verdade,
        viraria mensagem de erro espúria.
        """
        if definition is None or instruction is None:
            return []

        all_resources = self._collect_editor_resources(definition)
        by_id: Dict[str, StrategyResource] = {r.resource_id: r for r in all_resources}

        available_ids: Set[str] = set()

        # 1. PRIMARIES — sempre disponíveis.
        for r in all_resources:
            if r.category == ResourceCategory.PRIMARY:
                available_ids.add(r.resource_id)

        # 2. Inputs da própria instrução — gate de dependência garante.
        for rid in (instruction.input_resource_ids or []):
            if rid:
                available_ids.add(rid)

        # 3. Outputs de instruções anteriores na ordem de execução.
        #    Itera _phase_lists() (preparation → validation → casework)
        #    parando ao encontrar a instrução-alvo.
        try:
            phase_lists = definition._phase_lists()
        except AttributeError:
            phase_lists = []
        target_id = instruction.instruction_id
        for phase_list in phase_lists:
            stop = False
            for upstream in phase_list:
                if upstream.instruction_id == target_id:
                    stop = True
                    break
                for rid in (upstream.output_resource_ids or []):
                    if rid:
                        available_ids.add(rid)
            if stop:
                break

        # Filtra a lista mantendo apenas recursos que existem na pool de
        # `_collect_editor_resources`. Resource_ids declarados mas sem
        # entry em `_collect_editor_resources` (raro — ex: output_id
        # com nome mas que o spec do tipo não considera) ficam fora.
        return [by_id[rid] for rid in available_ids if rid in by_id]

    def _on_add_validation_rule_requested(
        self, instruction: StrategyInstructionDefinition,
    ) -> None:
        """
        Abre `AddValidationRuleDialog` populado com:
        - 11 tipos configuráveis (FISHER, MIN_IDENTITIES, etc.)
        - templates salvos (global + projeto)
        - recursos garantidos no contexto no momento de validação de X
          (PRIMARIES + inputs da X + outputs de instruções anteriores).
          NÃO inclui outputs da própria X nem de instruções posteriores
          — referenciá-los faria a regra falhar com "recurso não
          encontrado" na hora da validação.

        Em sucesso, anexa o `ValidationRuleConfig` retornado em
        `instruction.validation_rule_configs` via editor.
        """
        from interface.tabs.strategy_manager_v2.strategy_editor.add_validation_rule_dialog import (
            AddValidationRuleDialog,
        )

        if instruction is None:
            return

        # Filtragem rigorosa: só recursos garantidos no contexto na
        # ordem de execução. Evita footgun de referenciar recursos
        # ainda não produzidos.
        available = self._resources_available_at_instruction_validation(
            instruction, self._current_definition,
        )
        templates = self._load_validation_rule_templates()

        dlg = AddValidationRuleDialog(
            available_resources=available,
            templates=templates,
            parent=self._view,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        cfg = dlg.get_config()
        if cfg is None:
            return

        # Editor faz o append no panel correto e emite config_changed →
        # _on_strategy_modified → save.
        self._strategy_editor.append_validation_rule_to_instruction(
            instruction, cfg,
        )

    def _on_save_validation_rule_as_template_requested(self, cfg) -> None:
        """
        Abre `SaveValidationRuleTemplateDialog`, salva como JSON em
        `<global_library>/validation_rules/{template_id}.json` e/ou
        `<project>/validation_rules/{template_id}.json` conforme escolha.

        Mostra QMessageBox de erro se algum save falhar. Continua tentando
        o outro destino se for "ambos" e um falhar.
        """
        from interface.tabs.strategy_manager_v2.strategy_editor.save_validation_rule_template_dialog import (
            SaveValidationRuleTemplateDialog,
        )
        from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
        from faceaudit.models.validation.validation_rule_template import (
            ValidationRuleTemplate,
        )

        if cfg is None:
            return

        # Default name: o label do tipo da regra, pra dar dica visual.
        default_name = (
            cfg.rule_type.label if hasattr(cfg.rule_type, "label") else ""
        )

        dlg = SaveValidationRuleTemplateDialog(
            has_project=(self._project is not None),
            default_name=default_name,
            parent=self._view,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        result = dlg.get_result()
        if result is None:
            return
        template_id, name, description, destination = result

        template = ValidationRuleTemplate(
            template_id=template_id,
            name=name,
            description=description,
            rule_config=cfg,
        )

        errors: list = []
        saved_paths: list = []

        save_global = destination in (
            SaveValidationRuleTemplateDialog.DEST_GLOBAL,
            SaveValidationRuleTemplateDialog.DEST_BOTH,
        )
        save_project = destination in (
            SaveValidationRuleTemplateDialog.DEST_PROJECT,
            SaveValidationRuleTemplateDialog.DEST_BOTH,
        )

        if save_global:
            if self._global_storage is None:
                errors.append(
                    "Biblioteca global não está configurada. "
                    "Configure em Preferências antes de salvar."
                )
            else:
                try:
                    ref = self._global_storage.save_artifact(
                        artifact_type=StrategyArtifactType.VALIDATION_RULE_TEMPLATE,
                        data=template,
                        template_id=template_id,
                    )
                    saved_paths.append(f"Biblioteca global: {ref.path}")
                except Exception as exc:
                    errors.append(f"Falha ao salvar na biblioteca: {exc}")

        if save_project:
            if self._project is None:
                errors.append("Nenhum projeto aberto.")
            else:
                try:
                    ref = self._project.storage.save_artifact(
                        artifact_type=StrategyArtifactType.VALIDATION_RULE_TEMPLATE,
                        data=template,
                        template_id=template_id,
                    )
                    saved_paths.append(f"Projeto: {ref.path}")
                except Exception as exc:
                    errors.append(f"Falha ao salvar no projeto: {exc}")

        if errors:
            QtWidgets.QMessageBox.warning(
                self._view,
                "Salvar Template",
                "Alguns destinos falharam:\n\n" + "\n".join(errors)
                + (
                    "\n\nSalvos com sucesso em:\n" + "\n".join(saved_paths)
                    if saved_paths else ""
                ),
            )
        elif saved_paths:
            QtWidgets.QMessageBox.information(
                self._view,
                "Template salvo",
                "Salvo em:\n" + "\n".join(saved_paths),
            )

    def _on_hydrate_resource_requested(
        self,
        instruction: StrategyInstructionDefinition,
        resource_id: str,
    ) -> None:
        """
        Carrega o payload do recurso a partir do disco e refresca o painel
        de resultados para que o gráfico seja desenhado.

        Acionado pelo botão &quot;Carregar e visualizar&quot; na 3ª coluna do editor,
        que aparece quando o recurso está em disco mas não em memória (caso
        típico após reabrir um projeto sem ter executado nada nessa sessão).

        Cursor de espera (WaitCursor) é setado durante toda a operação —
        carregar parquets de milhões de pares trava a main thread por
        segundos e o usuário precisa de feedback visual imediato.
        """
        if not self._project or not self._current_definition:
            return

        if not self._ensure_executor_and_context(silent=True):
            QtWidgets.QMessageBox.warning(
                self._view,
                "Carregamento indisponível",
                "Não foi possível inicializar o executor para hidratar o recurso.",
            )
            return

        existing = self._current_definition.get_resource_by_id(resource_id)
        if existing is None:
            # Pode estar só no contexto (ex: produzido em sessão anterior).
            if self._execution_context is not None and self._execution_context.has_resource(resource_id):
                existing = self._execution_context.get_resource(resource_id)

        if existing is None:
            QtWidgets.QMessageBox.warning(
                self._view,
                "Recurso não encontrado",
                f"Recurso '{resource_id}' não está registrado nesta estratégia.",
            )
            return

        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            # Repinta a UI antes de bloquear na hidratação — sem isso o
            # cursor só atualiza depois do load terminar (event loop fica
            # parado durante leitura do parquet).
            QtWidgets.QApplication.processEvents()

            if not existing.has_payload and existing.has_artifact:
                loader = getattr(self._executor, "_resource_loader", None)
                if loader is not None:
                    try:
                        loader.hydrate(existing)
                    except Exception as exc:
                        QtWidgets.QApplication.restoreOverrideCursor()
                        QtWidgets.QMessageBox.critical(
                            self._view,
                            "Erro ao carregar",
                            f"Falha ao carregar '{resource_id}' do disco:\n{exc}",
                        )
                        return

            if not existing.has_payload:
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.information(
                    self._view,
                    "Nada a carregar",
                    f"O recurso '{resource_id}' não tem payload em memória nem em disco.",
                )
                return

            # Garante que o contexto vivo também enxergue o payload hidratado —
            # o card consulta o contexto antes de cair na definition.
            if self._execution_context is not None:
                self._execution_context.set_resource(existing)

            # Invalida o card cacheado e refresca para o card ser reconstruído
            # com o payload agora populado, disparando a renderização do plot.
            self._strategy_editor.invalidate_results_for_instruction(
                instruction.instruction_id
            )
            self._refresh_produced_resources()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_view_score_distribution(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Abre um dialog com gráfico de distribuição de scores genuínos vs
        impostores. Aplicável a:
        - SIMILARITY_SCORING (raw_score por par);
        - SCORE_CALIBRATION (log-LR calibrado por par).
        """
        from faceaudit.models.instructions.similarity_scoring_instruction import (
            SimilarityScoringInstruction,
        )
        from faceaudit.models.instructions.score_calibration_instruction import (
            ScoreCalibrationInstruction,
        )

        if not self._project or not self._current_definition:
            return

        scores_id: Optional[str] = None
        use_log_lr = False

        if isinstance(instruction, SimilarityScoringInstruction):
            scores_id = instruction.output_similarity_scores_resource_id
        elif isinstance(instruction, ScoreCalibrationInstruction):
            scores_id = instruction.output_calibrated_scores_resource_id
            use_log_lr = True
        else:
            return

        if not scores_id:
            return

        with self._wait_cursor():
            score_dataset = None
            missing_msg = None
            error_msg = None

            # 1. Tentar do contexto vivo.
            if self._execution_context is not None and self._execution_context.has_resource(scores_id):
                ctx_resource = self._execution_context.get_resource(scores_id)
                if ctx_resource and ctx_resource.has_payload:
                    score_dataset = ctx_resource.payload

            # 2. Tentar da definição (hidrata do disco se necessário).
            if score_dataset is None:
                existing = self._current_definition.get_resource_by_id(scores_id)
                if existing is not None:
                    if existing.has_payload:
                        score_dataset = existing.payload
                    elif existing.has_artifact and self._executor is not None:
                        loader = getattr(self._executor, "_resource_loader", None)
                        if loader is not None:
                            try:
                                loader.hydrate(existing)
                            except Exception:
                                pass
                            if existing.has_payload:
                                score_dataset = existing.payload

            if score_dataset is None:
                missing_msg = (
                    "Não foram encontrados scores salvos para esta instrução. "
                    "Execute a instrução de scoring primeiro."
                )
            else:
                try:
                    from interface.tabs.strategy_manager_v2.strategy_editor.graph_viewer_dialog import (
                        GraphViewerDialog,
                    )
                    title = (
                        f"Distribuição de log-LR: {instruction.name}"
                        if use_log_lr
                        else f"Distribuição de Scores: {instruction.name}"
                    )
                    dlg = GraphViewerDialog.from_draw_fn(
                        title=title,
                        draw_fn=lambda fig, ds=score_dataset, ull=use_log_lr:
                            self._draw_score_distribution(fig, ds, use_log_lr=ull),
                        parent=self._view,
                    )
                    self._distribution_viewer = dlg
                    dlg.show()
                except Exception as exc:
                    error_msg = f"Não foi possível abrir o visualizador de distribuição:\n{exc}"

        if missing_msg:
            QtWidgets.QMessageBox.information(self._view, "Scores indisponíveis", missing_msg)
        elif error_msg:
            QtWidgets.QMessageBox.critical(self._view, "Erro ao abrir visualizador", error_msg)

    def _draw_score_distribution(self, figure, score_dataset, use_log_lr: bool = False) -> None:
        """
        Pinta na figure histograma + KDE das distribuições genuínos vs
        impostores. Robusto a casos onde só uma das classes está presente.

        Quando `use_log_lr=True`, lê o eixo X de `sample.metadata["log_lr"]`
        em vez de `sample.raw_score` — usado pela instrução SCORE_CALIBRATION
        cujo output traz log-LR calibrado.
        """
        import numpy as np

        figure.clear()
        ax = figure.add_subplot(111)

        if use_log_lr:
            # Extrai log_lr de cada sample.metadata; samples sem log_lr
            # (não calibrados) entram como NaN e são filtrados depois.
            raw = []
            raw_labels = []
            for s in score_dataset.items:
                md = getattr(s, "metadata", None)
                v = md.get("log_lr") if isinstance(md, dict) else None
                if v is None:
                    continue
                raw.append(float(v))
                raw_labels.append(int(s.label))
            scores = np.asarray(raw, dtype=np.float64)
            labels = np.asarray(raw_labels, dtype=np.int32)
        else:
            scores = np.asarray(score_dataset.scores, dtype=np.float64)
            labels = np.asarray(score_dataset.labels, dtype=np.int32)

        if scores.size == 0:
            ax.text(
                0.5, 0.5, "(sem scores)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888",
            )
            return

        gen_mask = labels == 1
        imp_mask = labels == 0
        gen_scores = scores[gen_mask]
        imp_scores = scores[imp_mask]
        n_gen = int(gen_scores.size)
        n_imp = int(imp_scores.size)

        x_min = float(np.min(scores))
        x_max = float(np.max(scores))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        bins = np.linspace(x_min, x_max, 60)

        if n_imp > 0:
            ax.hist(
                imp_scores, bins=bins, density=True, alpha=0.45,
                color="#c62828", edgecolor="#7a0000", linewidth=0.5,
                label=f"Impostores  (n={n_imp:,})".replace(",", "."),
            )
        if n_gen > 0:
            ax.hist(
                gen_scores, bins=bins, density=True, alpha=0.45,
                color="#2e7d32", edgecolor="#0d3a10", linewidth=0.5,
                label=f"Genuínos  (n={n_gen:,})".replace(",", "."),
            )

        # Curva KDE leve via gaussian_kde (sem dependência adicional).
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(x_min, x_max, 400)
            if n_imp > 1:
                kde_i = gaussian_kde(imp_scores)
                ax.plot(xs, kde_i(xs), color="#7a0000", linewidth=1.4, alpha=0.9)
            if n_gen > 1:
                kde_g = gaussian_kde(gen_scores)
                ax.plot(xs, kde_g(xs), color="#0d3a10", linewidth=1.4, alpha=0.9)
        except Exception:
            pass  # Falta scipy ou amostras insuficientes: histograma ainda funciona.

        title_prefix = (
            "Distribuição de log-LR" if use_log_lr else "Distribuição de Scores"
        )
        ax.set_title(
            f"{title_prefix} — {score_dataset.name}\n"
            f"Total: {n_gen + n_imp:,} pares".replace(",", "."),
            fontsize=11,
        )
        ax.set_xlabel("log-LR calibrado" if use_log_lr else "Score bruto")
        ax.set_ylabel("Densidade")
        if use_log_lr:
            ax.axvline(0.0, color="#444", linestyle=":", linewidth=1.0,
                       alpha=0.6, label="log-LR = 0")
        ax.grid(True, linestyle="--", alpha=0.4)
        if n_gen > 0 or n_imp > 0:
            ax.legend(loc="best", fontsize=9)

        figure.tight_layout()

    def _on_resource_edit_requested(self, resource_id: str) -> None:
        """
        Despacha edição de um recurso de configuração para o dialog
        apropriado conforme o `resource_type`:
          - PIPELINE_CONFIG → PipelineCreatorController
          - MINING_CONFIG   → NewMiningConfigDialogController
        Hidrata o payload sob demanda, abre o editor, recebe a nova config,
        atualiza o payload em memória + contexto vivo + re-persiste sobre o
        mesmo path (preserva o nome do arquivo).
        """
        from faceaudit.enums.strategy_resource_type import StrategyResourceType

        if not self._project:
            return

        # Buscamos no pool completo (definitions + scan de disco) — mesma
        # fonte que popula o dropdown na UI.
        target_resource = None
        for res in self._collect_project_resources():
            if res.resource_id == resource_id:
                target_resource = res
                break
        if target_resource is None:
            QtWidgets.QMessageBox.warning(
                self._view,
                "Recurso não encontrado",
                f"O recurso '{resource_id}' não foi localizado no projeto.",
            )
            return

        rtype = target_resource.resource_type
        if rtype == StrategyResourceType.PIPELINE_CONFIG:
            self._edit_pipeline_config_resource(target_resource)
        elif rtype == StrategyResourceType.MINING_CONFIG:
            self._edit_mining_config_resource(target_resource)
        elif rtype == StrategyResourceType.SPLIT_CONFIG:
            self._edit_split_config_resource(target_resource)
        else:
            QtWidgets.QMessageBox.warning(
                self._view,
                "Edição não suportada",
                f"Tipo de recurso '{rtype.value}' não tem editor associado.",
            )

    def _hydrate_resource_if_needed(self, resource) -> bool:
        """
        Garante que `resource.payload` está hidratado (lazy-load do disco
        via loader). Retorna True em sucesso, False (com QMessageBox) em
        falha.
        """
        if resource.payload is not None:
            return True
        if resource.artifact_ref is None:
            return True

        loader = None
        if self._executor is not None:
            loader = getattr(self._executor, "_resource_loader", None)
        if loader is None:
            from faceaudit.services.strategy_resource_loader_service import (
                StrategyResourceLoaderService,
            )
            loader = StrategyResourceLoaderService(
                artifact_storage=self._project.storage
            )
        try:
            loader.hydrate(resource)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Falha ao carregar recurso",
                f"Não foi possível carregar o recurso do disco:\n{exc}",
            )
            return False
        return True

    @staticmethod
    def _definition_uses_resource(defn, resource_id: str) -> bool:
        """True se alguma instrução da definition referencia o recurso
        via `input_resource_ids`. Usado pelo upsert do
        `_persist_edited_config_resource` pra detectar quando uma
        config global (PIPELINE/MINING/SPLIT) precisa ser registrada
        em `defn.resources` com hash atualizado mesmo que ela só viva
        no scan de disco do projeto.
        """
        try:
            instructions = list(defn.instructions)
        except Exception:
            return False
        for instr in instructions:
            inputs = getattr(instr, "input_resource_ids", None) or []
            if resource_id in inputs:
                return True
        return False

    def _persist_edited_config_resource(
        self,
        target_resource,
        new_config,
        artifact_type,
    ) -> bool:
        """
        Atualiza payload em memória + contexto vivo, re-persiste o artefato
        sobre o mesmo path (preserva o nome do arquivo) e propaga o
        `artifact_ref` novo para qualquer definition que referencie o
        recurso. Retorna True em sucesso, False (com QMessageBox) em falha.
        """
        from pathlib import Path

        resource_id = target_resource.resource_id

        target_resource.payload = new_config
        if self._execution_context is not None and self._execution_context.has_resource(resource_id):
            ctx_resource = self._execution_context.get_resource(resource_id)
            if ctx_resource is not None:
                ctx_resource.payload = new_config

        try:
            old_ref = target_resource.artifact_ref
            config_id = (
                Path(old_ref.path).stem if old_ref and old_ref.path else resource_id
            )
            new_ref = self._project.storage.save_artifact(
                artifact_type,
                new_config,
                config_id=config_id,
            )
            target_resource.artifact_ref = new_ref
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao salvar",
                f"Não foi possível salvar a configuração editada:\n{exc}",
            )
            return False

        # Propaga novo hash a TODAS as definitions que referenciam o
        # recurso — direto via `defn.resources` ou indireto via
        # `instruction.input_resource_ids`. UPSERT cobre o caso comum
        # de configs globais (PIPELINE/MINING/SPLIT) que vivem só nas
        # pastas do projeto e são referenciadas por id (string) pelas
        # instruções, sem placeholder em `defn.resources`. Sem o UPSERT,
        # o hash novo se perdia: target_resource vinha sintético do
        # `_collect_project_resources` (scan de disco), nenhuma defn
        # tinha o recurso pra atualizar, e o próximo check_integrity
        # via "MODIFICADO" indevidamente.
        from faceaudit.models.strategy.strategy_resource import StrategyResource
        from faceaudit.enums.resource_category import ResourceCategory

        save_errors: List[str] = []
        n_updated = 0
        n_upserted = 0
        for defn in self._definitions:
            existing_res = defn.get_resource_by_id(resource_id)
            if existing_res is not None:
                # (a) Defn já declara o recurso — atualiza in-place.
                existing_res.payload = new_config
                existing_res.artifact_ref = new_ref
                touched = True
                n_updated += 1
            elif self._definition_uses_resource(defn, resource_id):
                # (b) Defn usa o recurso via input_resource_ids de alguma
                # instrução mas não declara em resources — adiciona como
                # primary pra registrar o hash sob essa estratégia.
                # Propaga o `id` técnico do target pra manter consistência
                # cross-definition (mesma config = mesmo id em todas).
                defn.resources.append(StrategyResource(
                    resource_id=resource_id,
                    name=target_resource.name or resource_id,
                    resource_type=target_resource.resource_type,
                    payload=new_config,
                    artifact_ref=new_ref,
                    category=ResourceCategory.PRIMARY,
                    id=getattr(target_resource, "id", None),
                ))
                touched = True
                n_upserted += 1
            else:
                touched = False

            if touched:
                try:
                    self._project.definition_repository.save_definition(defn)
                except Exception as exc:
                    save_errors.append(
                        f"  • {defn.name} ({defn.strategy_id}): {exc}"
                    )

        if save_errors:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao persistir definição",
                "A configuração foi salva em disco mas o hash atualizado "
                "não pôde ser propagado a uma ou mais estratégias. Ao "
                "reabrir o projeto, o arquivo aparecerá como MODIFICADO:\n\n"
                + "\n".join(save_errors),
            )
        else:
            print(
                f"[edit config] hash propagado: "
                f"{n_updated} defn(s) atualizada(s), "
                f"{n_upserted} defn(s) com recurso novo (upsert) — "
                f"id={resource_id}"
            )

        # Invalida em cascata qualquer instrução que dependa do recurso
        # editado — em qualquer estratégia do projeto. Sem isso, modelos
        # treinados / datasets transformados com a versão ANTIGA do
        # pipeline/mining ficariam marcados como ✅ apesar de estarem
        # desatualizados. Cascade também atinge outputs derivados que
        # virem inputs de outras instruções (transitivo).
        # Recurso PRIMARY (config compartilhada) tem o MESMO `id` em
        # todas as estratégias que o referenciam (não regenerado no
        # duplicate), então o cascade naturalmente atinge cross-strategy.
        primary_uid = getattr(target_resource, "id", None)
        if primary_uid:
            self._invalidate_dependents_of(primary_uid)

        # Defesa adicional: RESETA `_executed_instruction_ids` para o estado
        # autoritativo derivado de disco. Necessário porque
        # `_refresh_instruction_states` faz
        # `self._executed_instruction_ids |= _compute_executed_ids_from_disk()`
        # — esse `|=` é UNIÃO. Se o cascade descartou IDs em memória mas
        # `_compute_executed_ids_from_disk` retorna esses IDs (por alguma
        # inconsistência transiente entre `defn.resources` e o read), a
        # união RE-ADICIONA. Aqui forçamos REPLACEMENT (=, não |=) usando
        # o estado de disco direto, que reflete o estado pós-cascade
        # (foi acabado de ser salvo via `save_definition`).
        try:
            self._executed_instruction_ids = self._compute_executed_ids_from_disk()

            runnable_ids: Set[str] = set()
            if (
                self._current_definition is not None
                and self._execution_context is not None
                and self._executor is not None
            ):
                for instr in self._current_definition.instructions:
                    if instr.instruction_id in self._executed_instruction_ids:
                        continue
                    try:
                        can_run, _ = self._executor.can_execute_instruction(
                            instr, self._execution_context
                        )
                    except Exception:
                        can_run = False
                    if can_run:
                        runnable_ids.add(instr.instruction_id)

            self._strategy_editor.update_instruction_states(
                runnable_ids=runnable_ids,
                executed_ids=set(self._executed_instruction_ids),
            )
            self._refresh_produced_resources()
        except Exception as exc:
            # Surface erros ao invés de engolir silenciosamente.
            import traceback
            QtWidgets.QMessageBox.warning(
                self._view,
                "Falha ao atualizar UI pós-edição",
                f"Estado em memória foi atualizado, mas a UI pode "
                f"estar dessincronizada:\n{exc}\n\n{traceback.format_exc()}",
            )

        # Re-renderiza a árvore esquerda SEM recarregar definitions do disco
        # (o que descartaria seleções pendentes nos dropdowns do editor).
        # O payload/artifact_ref já foram atualizados in-place na lista
        # `self._definitions` em memória — só precisamos refletir isso na UI.
        # OBS: _invalidate_dependents_of já chama _refresh_left_panel, então
        # este bloco vira no-op quando houve invalidação. Mantemos como
        # fallback pro caso de nenhum dependente ter sido encontrado (edit
        # que não afetou estado executado).
        try:
            all_resources = self._collect_project_resources()
            resource_to_strategy = self._build_resource_to_strategy_map()
            self._left_panel.populate_resources(
                all_resources, resource_to_strategy=resource_to_strategy
            )
        except Exception:
            pass

        self._is_dirty = True
        return True

    def _edit_pipeline_config_resource(self, target_resource) -> None:
        from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
        from faceaudit.models.transformation_pipeline_config import (
            TransformationPipelineConfig,
        )
        from interface.tabs.strategy_manager.pipeline_creator.pipeline_creator_controller import (
            PipelineCreatorController,
        )

        if not self._hydrate_resource_if_needed(target_resource):
            return
        if not isinstance(target_resource.payload, TransformationPipelineConfig):
            QtWidgets.QMessageBox.warning(
                self._view,
                "Pipeline indisponível",
                "O payload carregado não é um TransformationPipelineConfig válido.",
            )
            return

        ctrl = PipelineCreatorController(
            parent=self._view,
            existing_config=target_resource.payload,
            lock_name=True,
        )
        new_config = ctrl.run()
        if new_config is None:
            return
        if not isinstance(new_config, TransformationPipelineConfig):
            QtWidgets.QMessageBox.warning(
                self._view,
                "Edição inválida",
                "A configuração editada não é um TransformationPipelineConfig válido.",
            )
            return

        if self._persist_edited_config_resource(
            target_resource, new_config, StrategyArtifactType.PIPELINE_CONFIG
        ):
            QtWidgets.QMessageBox.information(
                self._view,
                "Pipeline atualizado",
                f"A configuração de '{target_resource.resource_id}' foi atualizada e salva.",
            )

    def _edit_mining_config_resource(self, target_resource) -> None:
        from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
        from faceaudit.models.comparison_mining_config import ComparisonMiningConfig
        from interface.tabs.strategy_manager.new_mining_config_dialog.new_mining_config_dialog_controller import (
            NewMiningConfigDialogController,
        )

        if not self._hydrate_resource_if_needed(target_resource):
            return
        if not isinstance(target_resource.payload, ComparisonMiningConfig):
            QtWidgets.QMessageBox.warning(
                self._view,
                "Mining config indisponível",
                "O payload carregado não é um ComparisonMiningConfig válido.",
            )
            return

        ctrl = NewMiningConfigDialogController(parent=self._view, lock_name=True)
        ctrl.load_config(target_resource.payload)
        if ctrl.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        new_config = ctrl.get_config()
        if not isinstance(new_config, ComparisonMiningConfig):
            QtWidgets.QMessageBox.warning(
                self._view,
                "Edição inválida",
                "A configuração editada não é um ComparisonMiningConfig válido.",
            )
            return

        if self._persist_edited_config_resource(
            target_resource, new_config, StrategyArtifactType.MINING_CONFIG
        ):
            QtWidgets.QMessageBox.information(
                self._view,
                "Mining config atualizado",
                f"A configuração de '{target_resource.resource_id}' foi atualizada e salva.",
            )

    def _edit_split_config_resource(self, target_resource) -> None:
        """Edita um SPLIT_CONFIG via NewSplitConfigDialogController em
        modo edit + lock_name (mesmo padrão do mining/pipeline).
        Hidrata o payload sob demanda, salva via UPSERT na definition
        afetada, invalida em cascata os dependentes."""
        from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
        from faceaudit.models.split_config import SplitConfig
        from interface.tabs.strategy_manager.new_split_config_dialog.new_split_config_dialog_controller import (
            NewSplitConfigDialogController,
        )

        if not self._hydrate_resource_if_needed(target_resource):
            return
        if not isinstance(target_resource.payload, SplitConfig):
            QtWidgets.QMessageBox.warning(
                self._view,
                "Split config indisponível",
                "O payload carregado não é um SplitConfig válido.",
            )
            return

        ctrl = NewSplitConfigDialogController(
            parent=self._view,
            config_to_edit=target_resource.payload,
            lock_name=True,
        )
        new_config = ctrl.exec()
        if new_config is None:
            # Usuário cancelou.
            return
        if not isinstance(new_config, SplitConfig):
            QtWidgets.QMessageBox.warning(
                self._view,
                "Edição inválida",
                "A configuração editada não é um SplitConfig válido.",
            )
            return

        if self._persist_edited_config_resource(
            target_resource, new_config, StrategyArtifactType.SPLIT_CONFIG
        ):
            QtWidgets.QMessageBox.information(
                self._view,
                "Split config atualizado",
                f"A configuração de '{target_resource.resource_id}' foi atualizada e salva.",
            )

    def _on_view_validation_result(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Abre um dialog com 4 abas (Distribuição de LRs, Tippett, Curva DET,
        Calibração ECE) a partir do StrategyValidationResult produzido pela
        instrução STRATEGY_EVALUATION.
        """
        from faceaudit.models.instructions.strategy_evaluation_instruction import (
            StrategyEvaluationInstruction,
        )

        if not self._project or not self._current_definition:
            return
        if not isinstance(instruction, StrategyEvaluationInstruction):
            return

        result_id = instruction.output_validation_result_resource_id
        if not result_id:
            return

        with self._wait_cursor():
            validation_result = None

            # 1. Tentar do contexto vivo.
            if self._execution_context is not None and self._execution_context.has_resource(result_id):
                ctx_resource = self._execution_context.get_resource(result_id)
                if ctx_resource and ctx_resource.has_payload:
                    validation_result = ctx_resource.payload

            # 2. Tentar da definição (hidrata do disco se necessário).
            if validation_result is None:
                existing = self._current_definition.get_resource_by_id(result_id)
                if existing is not None:
                    if existing.has_payload:
                        validation_result = existing.payload
                    elif existing.has_artifact and self._executor is not None:
                        loader = getattr(self._executor, "_resource_loader", None)
                        if loader is not None:
                            try:
                                loader.hydrate(existing)
                            except Exception:
                                pass
                            if existing.has_payload:
                                validation_result = existing.payload

            if validation_result is None:
                missing_msg = (
                    "Não foi encontrado um resultado de validação salvo para esta "
                    "instrução. Execute STRATEGY_EVALUATION primeiro."
                )
                error_msg = None
            else:
                missing_msg = None
                error_msg = None
                try:
                    from interface.tabs.strategy_manager_v2.strategy_editor.validation_result_viewer_dialog import (
                        ValidationResultViewerDialog,
                    )
                    dlg = ValidationResultViewerDialog(validation_result, parent=self._view)
                    self._validation_viewer = dlg  # mantém referência forte
                    dlg.show()
                except Exception as exc:
                    error_msg = f"Não foi possível abrir o visualizador de validação:\n{exc}"

        # Mensagens disparadas FORA do with — cursor já voltou ao normal
        # quando o usuário interage com a caixa de diálogo.
        if missing_msg:
            QtWidgets.QMessageBox.information(
                self._view, "Resultado indisponível", missing_msg
            )
        elif error_msg:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro ao abrir visualizador", error_msg
            )

    def _on_view_casework_visualization(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Abre o CaseworkVisualizationViewerDialog a partir do
        CaseworkVisualizationResult produzido pela instrução.
        Padrão idêntico a _on_view_validation_result — tenta contexto vivo
        primeiro, depois hidrata do disco via loader.
        """
        from faceaudit.models.instructions.casework_visualization_instruction import (
            CaseworkVisualizationInstruction,
        )

        if not self._project or not self._current_definition:
            return
        if not isinstance(instruction, CaseworkVisualizationInstruction):
            return

        result_id = instruction.output_visualization_resource_id
        if not result_id:
            return

        with self._wait_cursor():
            casework_result = None
            missing_msg = None
            error_msg = None

            # 1. Tentar do contexto vivo.
            if self._execution_context is not None and self._execution_context.has_resource(result_id):
                ctx_resource = self._execution_context.get_resource(result_id)
                if ctx_resource and ctx_resource.has_payload:
                    casework_result = ctx_resource.payload

            # 2. Tentar da definição (hidrata do disco se necessário).
            if casework_result is None:
                existing = self._current_definition.get_resource_by_id(result_id)
                if existing is not None:
                    if existing.has_payload:
                        casework_result = existing.payload
                    elif existing.has_artifact and self._executor is not None:
                        loader = getattr(self._executor, "_resource_loader", None)
                        if loader is not None:
                            try:
                                loader.hydrate(existing)
                            except Exception:
                                pass
                            if existing.has_payload:
                                casework_result = existing.payload

            if casework_result is None:
                missing_msg = (
                    "Não foi encontrada uma visualização salva para esta "
                    "instrução. Execute CASEWORK_VISUALIZATION primeiro."
                )
            else:
                try:
                    from interface.tabs.strategy_manager_v2.strategy_editor.casework_visualization_viewer_dialog import (
                        CaseworkVisualizationViewerDialog,
                    )
                    dlg = CaseworkVisualizationViewerDialog(casework_result, parent=self._view)
                    self._casework_viewer = dlg
                    dlg.show()
                except Exception as exc:
                    error_msg = f"Não foi possível abrir o visualizador de caso:\n{exc}"

        if missing_msg:
            QtWidgets.QMessageBox.information(self._view, "Visualização indisponível", missing_msg)
        elif error_msg:
            QtWidgets.QMessageBox.critical(self._view, "Erro ao abrir visualizador", error_msg)

    def _on_view_hubness_report(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Abre o HubnessReportViewerDialog (5 abas curadas + galeria de
        16 plots) a partir do HubnessReport produzido pela instrução.
        Padrão idêntico aos demais view_*: tenta contexto vivo primeiro,
        depois hidrata do disco via loader.
        """
        from faceaudit.models.instructions.hubness_analysis_instruction import (
            HubnessAnalysisInstruction,
        )

        if not self._project or not self._current_definition:
            return
        if not isinstance(instruction, HubnessAnalysisInstruction):
            return

        result_id = instruction.output_report_resource_id
        if not result_id:
            return

        with self._wait_cursor():
            report = None
            missing_msg = None
            error_msg = None

            # 1. Tentar do contexto vivo.
            if (
                self._execution_context is not None
                and self._execution_context.has_resource(result_id)
            ):
                ctx_resource = self._execution_context.get_resource(result_id)
                if ctx_resource and ctx_resource.has_payload:
                    report = ctx_resource.payload

            # 2. Tentar da definição (hidrata do disco se necessário).
            if report is None:
                existing = self._current_definition.get_resource_by_id(result_id)
                if existing is not None:
                    if existing.has_payload:
                        report = existing.payload
                    elif existing.has_artifact and self._executor is not None:
                        loader = getattr(self._executor, "_resource_loader", None)
                        if loader is not None:
                            try:
                                loader.hydrate(existing)
                            except Exception:
                                pass
                            if existing.has_payload:
                                report = existing.payload

            if report is None:
                missing_msg = (
                    "Não foi encontrado um HubnessReport salvo para esta "
                    "instrução. Execute HUBNESS_ANALYSIS primeiro."
                )
            else:
                try:
                    from interface.tabs.strategy_manager_v2.strategy_editor.hubness_report_viewer_dialog import (
                        HubnessReportViewerDialog,
                    )
                    dlg = HubnessReportViewerDialog(report, parent=self._view)
                    self._hubness_viewer = dlg
                    dlg.show()
                except Exception as exc:
                    error_msg = f"Não foi possível abrir o relatório de hubness:\n{exc}"

        if missing_msg:
            QtWidgets.QMessageBox.information(self._view, "Relatório indisponível", missing_msg)
        elif error_msg:
            QtWidgets.QMessageBox.critical(self._view, "Erro ao abrir visualizador", error_msg)

    def _on_view_hubness_ablation(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Abre o HubnessAblationViewerDialog (single-plot full-screen) a
        partir do AblationStudyInfo produzido pela instrução. Mesmo
        padrão do _on_view_hubness_report — tenta contexto vivo, depois
        hidrata do disco.
        """
        from faceaudit.models.instructions.hubness_ablation_instruction import (
            HubnessAblationInstruction,
        )

        if not self._project or not self._current_definition:
            return
        if not isinstance(instruction, HubnessAblationInstruction):
            return

        result_id = instruction.output_ablation_resource_id
        if not result_id:
            return

        with self._wait_cursor():
            study = None
            missing_msg = None
            error_msg = None

            if (
                self._execution_context is not None
                and self._execution_context.has_resource(result_id)
            ):
                ctx_resource = self._execution_context.get_resource(result_id)
                if ctx_resource and ctx_resource.has_payload:
                    study = ctx_resource.payload

            if study is None:
                existing = self._current_definition.get_resource_by_id(result_id)
                if existing is not None:
                    if existing.has_payload:
                        study = existing.payload
                    elif existing.has_artifact and self._executor is not None:
                        loader = getattr(self._executor, "_resource_loader", None)
                        if loader is not None:
                            try:
                                loader.hydrate(existing)
                            except Exception:
                                pass
                            if existing.has_payload:
                                study = existing.payload

            if study is None:
                missing_msg = (
                    "Não foi encontrado um AblationStudyInfo salvo para esta "
                    "instrução. Execute HUBNESS_ABLATION primeiro."
                )
            else:
                try:
                    from interface.tabs.strategy_manager_v2.strategy_editor.hubness_ablation_viewer_dialog import (
                        HubnessAblationViewerDialog,
                    )
                    dlg = HubnessAblationViewerDialog(study, parent=self._view)
                    self._hubness_ablation_viewer = dlg
                    dlg.show()
                except Exception as exc:
                    error_msg = f"Não foi possível abrir o estudo de ablação:\n{exc}"

        if missing_msg:
            QtWidgets.QMessageBox.information(self._view, "Ablação indisponível", missing_msg)
        elif error_msg:
            QtWidgets.QMessageBox.critical(self._view, "Erro ao abrir visualizador", error_msg)

    def _on_view_typicality_distribution(
        self, instruction: StrategyInstructionDefinition,
    ) -> None:
        """
        Abre o TypicalityDistributionViewerDialog a partir do modelo treinado
        produzido por KNN_TYPICALITY_MODEL_TRAINING. Mesmo padrão de fetch
        dos demais viewers — contexto vivo primeiro, hidrata do disco como
        fallback.
        """
        from faceaudit.models.instructions.knn_typicality_model_training_instruction import (
            KNNTypicalityModelTrainingInstruction,
        )

        if not self._project or not self._current_definition:
            return
        if not isinstance(instruction, KNNTypicalityModelTrainingInstruction):
            return

        result_id = instruction.trained_typicality_model_resource_id
        if not result_id:
            return

        with self._wait_cursor():
            model = None
            missing_msg = None
            error_msg = None

            if (
                self._execution_context is not None
                and self._execution_context.has_resource(result_id)
            ):
                ctx_resource = self._execution_context.get_resource(result_id)
                if ctx_resource and ctx_resource.has_payload:
                    model = ctx_resource.payload

            if model is None:
                existing = self._current_definition.get_resource_by_id(result_id)
                if existing is not None:
                    if existing.has_payload:
                        model = existing.payload
                    elif existing.has_artifact and self._executor is not None:
                        loader = getattr(self._executor, "_resource_loader", None)
                        if loader is not None:
                            try:
                                loader.hydrate(existing)
                            except Exception:
                                pass
                            if existing.has_payload:
                                model = existing.payload

            if model is None:
                missing_msg = (
                    "Não foi encontrado um modelo de tipicidade treinado para "
                    "esta instrução. Execute KNN_TYPICALITY_MODEL_TRAINING primeiro."
                )
            else:
                try:
                    from interface.tabs.strategy_manager_v2.strategy_editor.typicality_distribution_viewer_dialog import (
                        TypicalityDistributionViewerDialog,
                    )
                    dlg = TypicalityDistributionViewerDialog(model, parent=self._view)
                    self._typicality_viewer = dlg
                    dlg.show()
                except Exception as exc:
                    error_msg = f"Não foi possível abrir o visualizador:\n{exc}"

        if missing_msg:
            QtWidgets.QMessageBox.information(
                self._view, "Modelo indisponível", missing_msg,
            )
        elif error_msg:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro ao abrir visualizador", error_msg,
            )

    def _on_run_instruction(self, instruction: StrategyInstructionDefinition) -> None:
        if not self._project or not self._current_definition:
            return

        if not self._ensure_executor_and_context():
            return

        can_run, missing = self._executor.can_execute_instruction(
            instruction, self._execution_context
        )
        if not can_run:
            QtWidgets.QMessageBox.warning(
                self._view,
                "Instrução bloqueada",
                "Recursos indisponíveis:\n" + "\n".join(f"• {m}" for m in missing),
            )
            return

        dialog = LogProgressDialog(
            f"Executando: {instruction.name}…", parent=self._view
        )
        # ApplicationModal bloqueia entrega de sinais queued de threads worker no
        # Windows (ver CLAUDE.md) — usar NonModal mantém o dialog visível sem
        # travar a delivery de log_message/finished.
        dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dialog.show()

        thread = QtCore.QThread()
        worker = InstructionExecutionWorker(
            self._executor,
            instruction,
            self._execution_context,
            strategy_id=self._current_definition.strategy_id,
        )
        worker.moveToThread(thread)

        # Mantém referência Python ao worker para evitar GC prematuro do wrapper.
        self._active_worker = worker

        # Estado de cancelamento (cooperative — fechamos o dialog assim que a
        # instrução terminar; não há ponto de interrupção dentro do executor).
        cancel_state = {"requested": False}

        thread.started.connect(worker.run)
        worker.log_message.connect(dialog.append_log)

        def on_cancel_requested():
            cancel_state["requested"] = True

        dialog.cancellation_requested.connect(on_cancel_requested)

        def close_dialog_if_cancelled():
            if cancel_state["requested"]:
                dialog.accept()

        def on_finished(result):
            if result.success:
                self._executed_instruction_ids.add(instruction.instruction_id)
                # A persistência (pickle.dump pesado) já rodou na thread do
                # worker — aqui só sincronizamos os artifact_refs já gravados
                # com a definição em memória (operação rápida, sem I/O).
                self._sync_persisted_resources_to_definition(result.produced_resource_ids)
                # Apaga recursos órfãos: outputs persistidos que deixaram
                # de ser declarados (ex: usuário renomeou A→B antes de
                # re-executar). Sem isso, A.<ext> + sidecars + PNG ficam
                # no disco como lixo, e A continua em definition.resources.
                self._cleanup_orphan_persisted_resources()
                dialog.finish(True)
            else:
                dialog.finish(False)
            self._refresh_instruction_states()
            self._refresh_produced_resources()
            # Repopula a árvore de recursos (esquerda) e os combos do editor
            # com os recursos novos produzidos pela instrução.
            self._refresh_left_panel()
            close_dialog_if_cancelled()

        def on_error(msg):
            dialog.finish(False)
            self._refresh_instruction_states()
            close_dialog_if_cancelled()

        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._active_thread = thread
        thread.start()

    # ── Global toolbar slots ──────────────────────────────────────────────────

    def _on_pipeline(self) -> None:
        ctrl = PipelineCreatorController(parent=self._view)
        config = ctrl.run()
        self._save_config_with_destination(
            config=config,
            artifact_type=StrategyArtifactType.PIPELINE_CONFIG,
            label="pipeline",
        )

    def _on_mining(self) -> None:
        ctrl = NewMiningConfigDialogController(parent=self._view)
        if ctrl.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return  # usuário cancelou — sem feedback (esperado)
        self._save_config_with_destination(
            config=ctrl.get_config(),
            artifact_type=StrategyArtifactType.MINING_CONFIG,
            label="mining",
        )

    def _on_split(self) -> None:
        ctrl = NewSplitConfigDialogController(parent=self._view)
        config = ctrl.exec()  # retorna SplitConfig em sucesso, None se cancelou
        self._save_config_with_destination(
            config=config,
            artifact_type=StrategyArtifactType.SPLIT_CONFIG,
            label="split",
        )

    def _choose_config_destination(self, label: str) -> Optional[str]:
        """Diálogo perguntando onde salvar a config recém-criada.

        Retorna 'global', 'project', 'both' ou None se o usuário cancelar.
        Quando não há projeto aberto, força 'global' (única opção viável)
        sem mostrar diálogo — sem projeto não há para onde mais salvar.
        """
        has_project = self._project is not None
        if not has_project:
            # Sem projeto aberto → biblioteca é o único destino possível.
            # Não vale a pena mostrar diálogo com uma opção só.
            return "global"

        dlg = QtWidgets.QDialog(self._view)
        dlg.setWindowTitle(f"Salvar configuração de {label}")
        dlg.setMinimumWidth(420)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(8)

        prompt = QtWidgets.QLabel(
            f"Onde salvar a configuração de <b>{label}</b>?"
        )
        prompt.setWordWrap(True)
        layout.addWidget(prompt)

        rb_both = QtWidgets.QRadioButton(
            "Biblioteca de configurações e pasta do projeto"
        )
        rb_global = QtWidgets.QRadioButton(
            "Apenas biblioteca de configurações"
        )
        rb_project = QtWidgets.QRadioButton(
            "Apenas pasta do projeto"
        )
        rb_both.setChecked(True)  # default conservador: tudo

        for rb in (rb_both, rb_global, rb_project):
            rb.setStyleSheet("padding: 4px 0;")
            layout.addWidget(rb)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        if rb_both.isChecked():
            return "both"
        if rb_global.isChecked():
            return "global"
        return "project"

    def _save_config_with_destination(
        self,
        config,
        artifact_type: StrategyArtifactType,
        label: str,
    ) -> None:
        """
        Persiste uma config recém criada (Pipeline / Mining / Split) no
        destino escolhido pelo usuário: biblioteca de configurações, pasta
        do projeto, ou ambos.

        - `config is None` → cancelamento no dialog de criação. Sai silencioso.
        - Sem projeto aberto → vai direto pra biblioteca (sem mostrar chooser).
        - Falha de IO num dos dois destinos → critical message com a exceção;
          aborta sem tentar o segundo destino (UX: explícito sobre o que
          aconteceu).
        - Sucesso → information message listando os locais persistidos.
        """
        if config is None:
            return

        destination = self._choose_config_destination(label)
        if destination is None:
            return  # cancelado no chooser

        config_id = self._derive_config_id(config)
        save_global = destination in ("global", "both")
        save_project = destination in ("project", "both")

        saved_locations: List[str] = []

        if save_global:
            if not self._global_storage:
                QtWidgets.QMessageBox.warning(
                    self._view,
                    "Biblioteca de configurações indisponível",
                    "Não foi possível acessar a biblioteca para salvar a "
                    f"configuração de {label}.\n\nPossíveis causas:\n"
                    "• A pasta padrão (~/.faceaudit/library) não pôde ser criada — "
                    "verifique permissões.\n"
                    "• Você definiu paths.library no config.json com um caminho "
                    "inválido — corrija ou remova essa entrada para usar o padrão.\n\n"
                    "A configuração NÃO foi persistida.",
                )
                return
            try:
                ref = self._global_storage.save_artifact(
                    artifact_type, config, config_id=config_id,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self._view,
                    "Erro ao salvar configuração",
                    f"Falha ao salvar a configuração de {label} na "
                    f"biblioteca:\n{exc}",
                )
                return
            library_root = getattr(self._global_storage, "library_root", None)
            if library_root:
                saved_locations.append(
                    f"• Biblioteca: {library_root}/{ref.path}"
                )
            else:
                saved_locations.append(f"• Biblioteca: {ref.path}")

        if save_project:
            if not self._project:
                # Não deveria acontecer (chooser bloqueia esse caso), mas
                # protegemos defensivamente.
                QtWidgets.QMessageBox.warning(
                    self._view,
                    "Projeto não disponível",
                    "Nenhum projeto aberto — não foi possível salvar na "
                    "pasta do projeto.",
                )
                return
            try:
                ref = self._project.storage.save_artifact(
                    artifact_type, config, config_id=config_id,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self._view,
                    "Erro ao salvar configuração",
                    f"Falha ao salvar a configuração de {label} na pasta do "
                    f"projeto:\n{exc}",
                )
                return
            project_root = self._project.storage.project_root
            saved_locations.append(f"• Projeto: {project_root}/{ref.path}")

        # Atualiza ANTES do messagebox pra que a árvore/dropdowns já
        # mostrem a nova config quando o usuário ler a confirmação.
        # `_refresh_left_panel` recarrega `_definitions` do disco e
        # repopula tanto a árvore esquerda quanto os dropdowns do editor
        # (via `_strategy_editor.update_available_resources`).
        self._refresh_left_panel()

        QtWidgets.QMessageBox.information(
            self._view,
            f"Configuração de {label} salva",
            f"'{config_id}' persistido em:\n\n"
            + "\n".join(saved_locations),
        )

    @staticmethod
    def _derive_config_id(config) -> str:
        """
        Deriva um config_id legível a partir do `name` do config (sanitizado
        como filename), com fallback para UUID curto se inviável.

        Sanitização: troca caracteres inválidos por `_`, colapsa repetições
        e remove `_` nas pontas. Limite de 64 chars para evitar paths longos.
        """
        import re

        raw_name = (getattr(config, "name", None) or "").strip()
        if not raw_name:
            return str(uuid.uuid4())[:8]

        # Sanitiza: mantém alfanuméricos, _ , - e espaços viram _.
        sanitized = re.sub(r"[^A-Za-z0-9._\- ]+", "_", raw_name)
        sanitized = re.sub(r"\s+", "_", sanitized)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_.- ")
        sanitized = sanitized[:64]

        return sanitized or str(uuid.uuid4())[:8]

    def _on_manage_library(self) -> None:
        ctrl = ManageLibraryController(parent=self._view, global_storage=self._global_storage)
        ctrl.exec()

    def _on_add_resources_from_library(self) -> None:
        """Adição em lote de recursos da biblioteca de configurações pro
        projeto atual.

        Abre um diálogo com 4 abas (Datasets / Splits / Minings / Pipelines),
        cada uma listando os recursos disponíveis na biblioteca. O usuário
        seleciona um ou mais em qualquer combinação de abas e confirma.

        Conflitos: se o resource_id (derivado do stem do arquivo na biblioteca)
        já existe no projeto, o item é **pulado** e listado no resumo
        final — não sobrescreve sem aviso.
        """
        from .dialogs.add_resources_from_library import (
            AddResourcesFromLibraryView,
        )

        if not self._project:
            QtWidgets.QMessageBox.warning(
                self._view,
                "Projeto não disponível",
                "Abra um projeto antes de adicionar recursos da biblioteca.",
            )
            return
        if not self._global_storage:
            QtWidgets.QMessageBox.warning(
                self._view,
                "Biblioteca indisponível",
                "Não foi possível acessar a biblioteca de configurações.",
            )
            return

        try:
            datasets = self._global_storage.list_artifacts(
                StrategyArtifactType.GLOBAL_DATASET
            )
            splits = self._global_storage.list_artifacts(
                StrategyArtifactType.SPLIT_CONFIG
            )
            minings = self._global_storage.list_artifacts(
                StrategyArtifactType.MINING_CONFIG
            )
            pipelines = self._global_storage.list_artifacts(
                StrategyArtifactType.PIPELINE_CONFIG
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao listar biblioteca",
                f"Falha ao consultar a biblioteca de configurações:\n{exc}",
            )
            return

        if not (datasets or splits or minings or pipelines):
            QtWidgets.QMessageBox.information(
                self._view,
                "Biblioteca vazia",
                "Não há recursos na biblioteca de configurações para "
                "adicionar. Crie pipelines, mining ou split configs primeiro.",
            )
            return

        dlg = AddResourcesFromLibraryView(parent=self._view)
        dlg.populate(
            datasets=datasets,
            splits=splits,
            minings=minings,
            pipelines=pipelines,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        selections = dlg.selected_refs()
        # Mapeia chave do dialog → tipo de artefato + kwarg do template
        spec_by_key = {
            "datasets":  (StrategyArtifactType.GLOBAL_DATASET,  "dataset_id"),
            "splits":    (StrategyArtifactType.SPLIT_CONFIG,    "config_id"),
            "minings":   (StrategyArtifactType.MINING_CONFIG,   "config_id"),
            "pipelines": (StrategyArtifactType.PIPELINE_CONFIG, "config_id"),
        }

        lib_root = getattr(self._global_storage, "library_root", None)
        added: List[str] = []
        skipped_conflicts: List[str] = []
        failed: List[str] = []

        renamed_in_conflict: List[tuple[str, str]] = []  # (original, sufixado)

        for key, refs in selections.items():
            if not refs:
                continue
            artifact_type, id_kwarg = spec_by_key[key]
            for ref in refs:
                # `resource_id` desejado = stem do arquivo da biblioteca.
                # Quando a biblioteca tem subpastas com mesmos stems
                # (ex: grupo_a/foo.parquet + grupo_b/foo.parquet), os dois
                # têm `desired_id="foo"` e o segundo colidiria no projeto.
                # Aqui auto-sufixamos numericamente (foo, foo_2, foo_3...)
                # pra preservar ambos como recursos distintos. O `name`
                # do recurso fica como o stem original (sem sufixo) —
                # o user pode editar depois pra desambiguar.
                desired_id = Path(ref.path).stem
                source_path = (lib_root / ref.path) if lib_root else None
                if source_path is None or not source_path.exists():
                    failed.append(f"{desired_id} (arquivo de origem ausente)")
                    continue

                # Detecta conflito de destino físico e resolve por sufixo.
                try:
                    from faceaudit.enums.strategy_file_type import (
                        StrategyFileType,
                    )
                    dest_file_type = StrategyFileType.for_artifact(artifact_type)
                    resource_id = self._resolve_unique_resource_id(
                        desired_stem=desired_id,
                        dest_file_type=dest_file_type,
                        id_kwarg=id_kwarg,
                    )
                    dest_rel = dest_file_type.build_path(**{id_kwarg: resource_id})
                    dest_abs = self._project.storage.project_root / dest_rel
                except Exception:
                    resource_id = desired_id
                    dest_abs = None

                if resource_id != desired_id:
                    renamed_in_conflict.append((desired_id, resource_id))

                try:
                    if artifact_type == StrategyArtifactType.GLOBAL_DATASET:
                        # Reusa o caminho específico de dataset (lida com
                        # path_mode RELATIVE → ABSOLUTE quando precisa).
                        self._project.storage.import_dataset(
                            source_path=source_path,
                            dataset_id=resource_id,
                        )
                        # Sidecar de stats também é útil pra dataset.
                        self._compute_and_save_dataset_stats(
                            self._project.storage.project_root
                            / dest_rel
                        )
                    else:
                        self._project.storage.copy_artifact_to_project(
                            source_path=source_path,
                            artifact_type=artifact_type,
                            **{id_kwarg: resource_id},
                        )
                    added.append(resource_id)
                except Exception as exc:
                    failed.append(f"{resource_id} ({exc})")

        # Refresh ANTES do summary pra que árvore/dropdowns já reflitam
        # os novos recursos quando o usuário ler a mensagem.
        if added:
            self._refresh_left_panel()

        # Resumo final
        parts: List[str] = []
        if added:
            parts.append(f"✓ {len(added)} adicionado(s): {', '.join(added)}")
        if renamed_in_conflict:
            renames_str = ", ".join(
                f"{orig}→{final}" for orig, final in renamed_in_conflict
            )
            parts.append(
                f"ℹ {len(renamed_in_conflict)} renomeado(s) por conflito de "
                f"nome no projeto: {renames_str}"
            )
        if skipped_conflicts:
            parts.append(
                f"⚠ {len(skipped_conflicts)} pulado(s) (já existem no "
                f"projeto): {', '.join(skipped_conflicts)}"
            )
        if failed:
            parts.append(
                f"✗ {len(failed)} falharam: {', '.join(failed)}"
            )

        if not parts:
            return  # nada a reportar

        icon = (
            QtWidgets.QMessageBox.Icon.Warning
            if (skipped_conflicts or failed)
            else QtWidgets.QMessageBox.Icon.Information
        )
        msg = QtWidgets.QMessageBox(self._view)
        msg.setIcon(icon)
        msg.setWindowTitle("Adicionar Recursos — Resumo")
        msg.setText("\n\n".join(parts))
        msg.exec()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_unique_resource_id(
        self,
        desired_stem: str,
        dest_file_type,
        id_kwarg: str,
        max_attempts: int = 999,
    ) -> str:
        """Resolve um resource_id que não colide com arquivos existentes no projeto.

        Quando dois recursos da biblioteca têm o mesmo stem (ex:
        `grupo_a/foo.parquet` e `grupo_b/foo.parquet`), o segundo colide
        com o primeiro no destino. Aqui auto-sufixamos numericamente
        (foo, foo_2, foo_3, ...) até achar um nome livre. Mantém ambos
        os recursos como entradas distintas no projeto sem precisar
        interromper o user.

        O `name` do recurso (em `StrategyResource.name`) continua sendo
        o stem ORIGINAL (sem sufixo) — o user vê "foo" duas vezes na
        árvore mas o `id` (DS-xxxxxxxx) e o `resource_id` (com sufixo)
        os diferenciam.

        Args:
            desired_stem: stem original (ex: "foo").
            dest_file_type: StrategyFileType pra calcular o path.
            id_kwarg: nome do kwarg do template ("dataset_id", "config_id").
            max_attempts: limite de sufixos antes de abortar (segurança).

        Returns:
            Stem único — `desired_stem` se livre, senão `{stem}_2`, `_3`, etc.

        Raises:
            RuntimeError se exceder `max_attempts` (cenário patológico).
        """
        project_root = self._project.storage.project_root
        candidate = desired_stem
        for attempt in range(1, max_attempts + 1):
            rel = dest_file_type.build_path(**{id_kwarg: candidate})
            if not (project_root / rel).exists():
                return candidate
            candidate = f"{desired_stem}_{attempt + 1}"
        raise RuntimeError(
            f"Não foi possível resolver nome único pra '{desired_stem}' "
            f"em {max_attempts} tentativas — diretório com colisões "
            "incompatíveis."
        )

    def _compute_and_save_dataset_stats(self, abs_path: Path) -> None:
        """
        Computa estatísticas básicas do dataset (leitura colunar, sem carregar todos os
        embeddings) e salva em `{stem}_stats.json` na mesma pasta.
        Silencia erros — stats são opcionais.
        """
        try:
            import pyarrow.parquet as pq

            svc = DatasetStatsService()
            stats = svc._read_parquet_stats(abs_path)

            # Detecta colunas de embedding pelo tipo (list<*>) — independente do nome
            embedding_columns = []
            try:
                import pyarrow as pa
                schema = pq.ParquetFile(abs_path).schema_arrow
                emb_indices = [
                    i for i, field in enumerate(schema)
                    if pa.types.is_list(field.type) or pa.types.is_large_list(field.type)
                ]
                if emb_indices:
                    pq_file2 = pq.ParquetFile(abs_path)
                    for batch in pq_file2.iter_batches(batch_size=1):
                        for pos, idx in enumerate(emb_indices):
                            val = batch.column(idx)[0].as_py()
                            dim = len(val) if (val is not None and hasattr(val, "__len__")) else 0
                            col_name = schema.field(idx).name
                            embedding_columns.append({
                                "index": pos,          # índice entre colunas lista
                                "parquet_index": idx,  # índice real no parquet
                                "name": col_name,
                                "dim": dim,
                            })
                        break
            except Exception:
                pass

            stats_path = abs_path.parent / (abs_path.stem + "_stats.json")
            payload = {
                "n_imgs": stats.n_imgs,
                "n_ids": stats.n_ids,
                "embedding_columns": embedding_columns,
            }
            stats_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass  # stats são opcionais; não bloquear importação

    def _on_rename_resource(
        self, resource: StrategyResource, new_name: str,
    ) -> None:
        """Renomeia o `name` de um recurso (Fase 4 da migração id/name).

        O `id` técnico permanece imutável — nenhuma referência interna a
        estratégias quebra. Os campos `input_resource_id`/`output_resource_id`
        das instruções continuam apontando pro mesmo `resource_id` legado.

        Salva todas as definitions que contêm a INSTÂNCIA `resource` (via
        `id(res)` no resource_to_strategy map). Em projeto fresh é só uma,
        mas recursos PRIMARY podem aparecer em várias estratégias quando
        coletados via _collect_project_resources com mesma identidade.
        """
        if not self._project or not resource:
            return
        new_name = (new_name or "").strip()
        if not new_name or new_name == resource.name:
            return

        # Mutação direta da instância — todas as definitions que a têm
        # em `resources` veem a mudança imediata.
        resource.name = new_name

        # Persiste todas as definitions que contêm essa instância.
        touched_strategy_ids: list = []
        for defn in self._definitions:
            if any(res is resource for res in defn.resources):
                try:
                    self._project.definition_repository.save_definition(defn)
                    touched_strategy_ids.append(defn.strategy_id)
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(
                        self._view,
                        "Falha ao salvar renomeação",
                        f"Erro ao salvar a estratégia '{defn.name}':\n{exc}",
                    )

        # Refresh do left panel pra mostrar o novo name (id) imediatamente.
        self._refresh_left_panel()

    def _on_remove_resource(self, resource: StrategyResource) -> None:
        """
        Remove o arquivo do recurso do disco, retira da definição da estratégia
        que o produziu e limpa os campos de output da instrução geradora.

        Sem essa limpeza completa, o recurso voltaria a aparecer na árvore no
        próximo refresh (porque ainda estaria em `definition.resources` com
        artifact_ref preenchido apontando para um arquivo já apagado).
        """
        if not self._project or not resource.artifact_ref:
            return

        reply = QtWidgets.QMessageBox.question(
            self._view,
            "Remover recurso",
            f"Remover o recurso '{resource.resource_id}' do projeto?\n"
            f"O arquivo será excluído do disco e não poderá ser recuperado.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            # 1. Apagar arquivo do disco (e stats JSON se houver).
            abs_path = self._project.storage.project_root / resource.artifact_ref.path
            if abs_path.exists():
                abs_path.unlink()

            stats_path = abs_path.parent / (abs_path.stem + "_stats.json")
            if stats_path.exists():
                stats_path.unlink()

            # 2. Em todas as estratégias, limpar referências e remover da lista
            #    de recursos. Salvar cada definição modificada.
            current_strategy_modified = False
            instructions_unexecuted: list = []

            # Identifica o recurso por `id` único — `resource_id` (label)
            # pode repetir entre estratégias quando duplicate preservou
            # nomes. Pra cada def, buscamos o resource com mesmo id;
            # DERIVED só vai casar na def dona; PRIMARY casa em todas
            # as defs que referenciam o mesmo arquivo físico.
            removed_uid = resource.id
            for defn in self._definitions:
                modified = False

                # Encontra o resource_id (label) LOCAL desta def que
                # corresponde ao mesmo `id` único do recurso removido.
                local_rid: Optional[str] = None
                for r in defn.resources:
                    if r.id and r.id == removed_uid:
                        local_rid = r.resource_id
                        break
                if local_rid is None:
                    continue

                # Limpa TODOS os campos (inputs e outputs) das instruções
                # que apontam pra esse label local. Inputs apontando pro
                # recurso fantasma ficavam no DTO mesmo após o combo ser
                # repopulado sem ele — agora limpa em cascata.
                for instr in defn.instructions:
                    try:
                        specs = instr.get_resource_field_specs()
                    except Exception:
                        specs = []
                    instr_modified = False
                    for spec in specs:
                        if getattr(instr, spec.field_name, None) == local_rid:
                            setattr(instr, spec.field_name, None)
                            instr_modified = True
                    if instr_modified and hasattr(instr, "_sync_declared_io"):
                        try:
                            instr._sync_declared_io()
                        except Exception:
                            pass
                    if instr_modified:
                        instructions_unexecuted.append(instr.instruction_id)
                        modified = True

                # Remove o recurso da lista de resources da definição.
                if defn.get_resource_by_id(local_rid) is not None:
                    defn.remove_resource(local_rid)
                    modified = True

                if modified:
                    self._project.definition_repository.save_definition(defn)
                    self._is_dirty = False
                    if defn is self._current_definition:
                        current_strategy_modified = True

            # 3. Se a estratégia ativa foi modificada, marca as instruções
            #    afetadas como não-executadas e atualiza a UI do editor.
            if current_strategy_modified:
                for instr_id in instructions_unexecuted:
                    self._executed_instruction_ids.discard(instr_id)
                # Reseta contexto — o recurso removido pode estar no contexto
                # vivo, e o gate de dependência precisa ser recalculado.
                self._execution_context = None
                self._executor = None

            # 4. Limpa o painel de propriedades e atualiza tudo.
            self._left_panel.resource_info.clear()
            self._refresh_left_panel()
            if self._current_definition:
                self._refresh_instruction_states()
                self._refresh_produced_resources()

        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro ao remover", f"Não foi possível remover o recurso:\n{exc}"
            )

    # ── Output placeholder sync & persistence ──────────────────────────────────

    def _sync_all_output_placeholders(self) -> None:
        """Garante que cada output declarado nas instruções tem um placeholder em resources."""
        if not self._current_definition:
            return

        # Coletar todos os output resource IDs declarados nas instruções
        declared_outputs: dict = {}
        for instr in self._current_definition.instructions:
            for placeholder in instr.get_output_resource_placeholders():
                declared_outputs[placeholder.resource_id] = placeholder

        # Adicionar placeholders que não existem; sincronizar resource_type
        # quando o spec atual da instrução o mudou (migração leve para
        # recursos persistidos com tipo legado).
        for resource_id, placeholder in declared_outputs.items():
            existing = self._current_definition.get_resource_by_id(resource_id)
            if existing is None:
                self._current_definition.add_resource(placeholder)
            elif existing.resource_type != placeholder.resource_type:
                existing.resource_type = placeholder.resource_type

        # Remover placeholders órfãos (output renomeado ou instrução removida)
        # Apenas para recursos sem artifact_ref e sem payload (são placeholders puros)
        _project_resource_types = {
            StrategyResourceType.DATASET,
            StrategyResourceType.PIPELINE_CONFIG,
            StrategyResourceType.SPLIT_CONFIG,
            StrategyResourceType.MERGE_CONFIG,
            StrategyResourceType.SIMILARITY_MODEL_CONFIG,
            StrategyResourceType.CALIBRATION_MODEL_CONFIG,
        }
        for resource in list(self._current_definition.resources):
            if resource.resource_type in _project_resource_types:
                continue
            if resource.resource_id not in declared_outputs:
                if not resource.has_payload and not resource.has_artifact:
                    self._current_definition.remove_resource(resource.resource_id)

    def _cleanup_orphan_persisted_resources(self) -> None:
        """Apaga do disco e remove de `definition.resources` recursos
        persistidos que não são mais output declarado de NENHUMA instrução
        da estratégia atual.

        Cobre três cenários equivalentes:
          1. Output de instrução renomeado (A→B): "A" sobrevivia ao
             `_sync_all_output_placeholders` (guard exige `not has_artifact`).
          2. Instrução removida cujos outputs estavam persistidos.
          3. `apply_to_training_datasets` em PIPELINE_TRAINING desligado
             depois de já ter rodado: outputs deixam de ser declarados.

        NUNCA toca em recursos com `category=PRIMARY` (datasets/configs
        importados da biblioteca do projeto) — são primários compartilhados,
        não outputs de instrução. Antes o filtro era por `resource_type`
        (DATASET, PIPELINE_CONFIG, ...), mas isso conflitava tipo com
        categoria: datasets DERIVADOS de PIPELINE_APPLICATION/DATASET_SPLIT
        também têm `resource_type=DATASET` e ficavam imunes à limpeza,
        deixando o output antigo de uma instrução renomeada pra sempre
        em `definition.resources`.

        NUNCA apaga recurso cujo `resource_id` ainda é declarado por
        OUTRA instrução da estratégia (edge case raro mas possível com
        DTOs de projetos antigos). A união cross-instrução em
        `declared_output_ids` cobre isso.

        Deve rodar na main thread (faz disco I/O e mutaciona estado).
        Chamado pelos handlers de pós-execução (single + Run All) — não
        chamar em `_on_strategy_modified` porque o usuário pode estar
        digitando o novo nome do output mid-keystroke.
        """
        if not self._project or not self._current_definition:
            return

        definition = self._current_definition

        declared_output_ids: Set[str] = set()
        for instr in definition.instructions:
            try:
                for placeholder in instr.get_output_resource_placeholders():
                    declared_output_ids.add(placeholder.resource_id)
            except Exception:
                pass

        removed_any = False
        for resource in list(definition.resources):
            if resource.category == ResourceCategory.PRIMARY:
                continue
            if resource.resource_id in declared_output_ids:
                continue

            self._delete_persisted_resource_files(resource)
            if (
                self._execution_context is not None
                and self._execution_context.has_resource(resource.resource_id)
            ):
                try:
                    self._execution_context.remove_resource(resource.resource_id)
                except Exception:
                    pass
            definition.remove_resource(resource.resource_id)
            removed_any = True

        if removed_any:
            self._save_current_definition()

    def _sync_trained_model_to_definition(self, model) -> None:
        """
        Sincroniza recursos do LRStrategyTrainedModel já persistidos pelo
        StrategyExecutionWorker com a definição em memória (Run All).

        Não faz I/O — apenas copia artifact_ref/payload do `model.trained_resources`
        para a `definition.resources`.
        """
        if not self._current_definition or model is None:
            return

        trained_resources = list(getattr(model, "trained_resources", []) or [])
        for resource in trained_resources:
            if not getattr(resource, "has_payload", False):
                continue

            existing = self._current_definition.get_resource_by_id(resource.resource_id)
            old_ref = existing.artifact_ref if existing else None

            if existing:
                existing.resource_type = resource.resource_type
                if resource.artifact_ref is not None:
                    existing.artifact_ref = resource.artifact_ref
                existing.payload = resource.payload
            elif resource.artifact_ref is not None:
                # Só adiciona se foi efetivamente persistido em disco;
                # caso contrário não é um recurso "produzido" do projeto.
                self._current_definition.add_resource(resource)

            # Limpa arquivo órfão quando o path mudou.
            # Normaliza '\' para '/' antes de comparar — projetos abertos do
            # disco têm path normalizado para POSIX em from_dict; já o
            # save_artifact agora também emite POSIX, mas mantemos o guard
            # para tolerar refs herdadas de versões anteriores.
            if (
                resource.artifact_ref is not None
                and old_ref is not None
                and old_ref.path
                and old_ref.path.replace("\\", "/")
                    != resource.artifact_ref.path.replace("\\", "/")
            ):
                try:
                    old_abs = self._project.storage.project_root / old_ref.path
                    if old_abs.exists():
                        old_abs.unlink()
                except Exception:
                    pass

        # Run All: limpa órfãos por rename/remoção/toggle antes do save
        # final (mesma lógica que o caminho single-instruction usa via
        # `on_finished` no `_run_instruction_worker`).
        self._cleanup_orphan_persisted_resources()
        self._save_current_definition()

    def _sync_persisted_resources_to_definition(self, resource_ids: list) -> None:
        """
        Sincroniza recursos já persistidos pelo worker com a definição em memória.

        Esta versão NÃO faz I/O (a persistência aconteceu na thread do worker
        via InstructionExecutionWorker._persist_produced_resources_inline).
        Aqui só copiamos artifact_ref/resource_type/payload do contexto vivo
        para o `existing` na definição da estratégia, e tratamos o caso de
        arquivo antigo órfão quando o caminho mudou.
        """
        if not self._execution_context or not self._current_definition:
            return

        for resource_id in resource_ids or []:
            resource = self._execution_context.get_resource(resource_id)
            if resource is None:
                continue

            existing = self._current_definition.get_resource_by_id(resource_id)
            old_ref = existing.artifact_ref if existing else None

            artifact_ref = resource.artifact_ref
            if existing:
                existing.resource_type = resource.resource_type
                if artifact_ref is not None:
                    existing.artifact_ref = artifact_ref
                existing.payload = resource.payload
            else:
                self._current_definition.add_resource(resource)

            # Limpa arquivo órfão quando o path mudou (ex: tipo do recurso
            # mudou e a extensão/pasta de destino também). Normaliza '\' → '/'
            # nos dois lados — old_ref pode vir de from_dict (POSIX) e o novo
            # ref também é POSIX, mas mantemos o guard contra refs antigas.
            if (
                artifact_ref is not None
                and old_ref is not None
                and old_ref.path
                and old_ref.path.replace("\\", "/")
                    != artifact_ref.path.replace("\\", "/")
            ):
                try:
                    old_abs = self._project.storage.project_root / old_ref.path
                    if old_abs.exists():
                        old_abs.unlink()
                except Exception:
                    pass

        self._save_current_definition()

    def _persist_produced_resources(self, resource_ids: list) -> None:
        """Persiste recursos produzidos em disco e atualiza a definição."""
        if not self._execution_context or not self._executor or not self._current_definition:
            return

        for resource_id in resource_ids:
            resource = self._execution_context.get_resource(resource_id)
            if resource is None or not resource.has_payload:
                continue

            try:
                # Captura o artifact_ref ANTERIOR no `existing` antes de
                # sobrescrever. Se o tipo do recurso mudou (ex: instrução
                # passou a registrar como SIMILARITY_SCORES em vez de
                # INTERMEDIATE_DATASET), o caminho físico muda, e o arquivo
                # antigo precisa ser apagado para não virar órfão na árvore.
                existing = self._current_definition.get_resource_by_id(resource_id)
                old_ref = existing.artifact_ref if existing else None

                artifact_ref = self._executor.persist_produced_resource(
                    resource=resource,
                    strategy_id=self._current_definition.strategy_id,
                )
                if artifact_ref:
                    resource.artifact_ref = artifact_ref
                    if existing:
                        # Sincroniza tipo (pode ter mudado entre execuções
                        # quando a definição da instrução foi atualizada).
                        existing.resource_type = resource.resource_type
                        existing.artifact_ref = artifact_ref
                        existing.payload = resource.payload
                        # Re-execução: preserva o `id` antigo da definição
                        # propagando-o de volta pro resource recém-produzido
                        # no contexto. Sem isso, defn fica com id DS-aaa
                        # (estável) e context fica com id DS-bbb (novo a
                        # cada execução), causando drift visual no tree.
                        # O `set_resource` abaixo re-indexa corretamente
                        # — limpa o id órfão automaticamente (ver
                        # StrategyExecutionContext.set_resource).
                        if getattr(existing, "id", None):
                            resource.id = existing.id
                            if self._execution_context is not None:
                                # Re-registra pro index refletir o id final.
                                self._execution_context.set_resource(resource)
                    else:
                        self._current_definition.add_resource(resource)

                    # Limpa arquivo antigo órfão quando o caminho mudou
                    # (tipicamente porque a extensão/pasta mudou junto com
                    # o resource_type). Normaliza '\' → '/' nos dois lados
                    # antes de comparar — old_ref vem de from_dict (POSIX)
                    # e o novo ref também é POSIX, mas o guard tolera refs
                    # antigas que ainda tenham '\' Windows.
                    if (
                        old_ref is not None
                        and old_ref.path
                        and old_ref.path.replace("\\", "/")
                            != artifact_ref.path.replace("\\", "/")
                    ):
                        try:
                            old_abs = self._project.storage.project_root / old_ref.path
                            if old_abs.exists():
                                old_abs.unlink()
                        except Exception:
                            pass  # Limpeza opcional — não pode quebrar a execução.
            except Exception as exc:
                print(f"Erro ao persistir recurso '{resource_id}': {exc}")

        self._save_current_definition()

    def _save_current_definition(self) -> None:
        """Salva a definição da estratégia atual no repositório.

        Erros de persistência são surfaceados como QMessageBox — antes
        eram silenciosamente engolidos via `except: pass`, o que mascarava
        bugs onde mutações in-memory (ex: novo hash após edição inline de
        pipeline/mining) nunca chegavam ao disco e o `check_integrity` do
        próximo `open` flagava o arquivo como MODIFICADO.
        """
        if self._project and self._current_definition:
            try:
                self._project.definition_repository.save_definition(
                    self._current_definition
                )
                self._is_dirty = False
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self._view,
                    "Erro ao salvar estratégia",
                    f"Não foi possível persistir a estratégia "
                    f"'{self._current_definition.name}':\n{exc}\n\n"
                    "Alterações em memória não foram salvas. Tente "
                    "novamente; se o problema persistir, verifique "
                    "permissões de escrita na pasta do projeto.",
                )

    def _refresh_produced_resources(self) -> None:
        """Atualiza o widget de recursos produzidos em todas as abas."""
        if self._current_definition:
            self._strategy_editor.update_produced_resources(
                definition=self._current_definition,
                executed_ids=self._executed_instruction_ids,
                context=self._execution_context,
            )

    def _persist_trained_model_resources(self, model, executor) -> None:
        """Persiste recursos do LRStrategyTrainedModel (fluxo Run All) em disco."""
        if not self._current_definition:
            return

        for resource in model.trained_resources:
            if not resource.has_payload:
                continue
            # Pular recursos do projeto (já estão em disco)
            artifact_type = resource.resource_type.to_artifact_type()
            if artifact_type is None:
                continue

            try:
                artifact_ref = executor.persist_produced_resource(
                    resource=resource,
                    strategy_id=self._current_definition.strategy_id,
                )
                if artifact_ref:
                    resource.artifact_ref = artifact_ref
                    existing = self._current_definition.get_resource_by_id(
                        resource.resource_id
                    )
                    if existing:
                        existing.artifact_ref = artifact_ref
                        existing.payload = resource.payload
                    else:
                        self._current_definition.add_resource(resource)
            except Exception as exc:
                print(f"Erro ao persistir recurso '{resource.resource_id}': {exc}")

        self._save_current_definition()

    def _has_unsaved_changes(self) -> bool:
        """
        Retorna True apenas quando há edição em andamento que ainda não foi
        persistida.

        - `_is_dirty` é setada para True por `_on_strategy_modified` (sinal
          emitido pelo editor durante edições da UI).
        - É zerada em todo save bem-sucedido (`_save_current_definition`,
          `save_project`, criação/duplicação/renomeação de estratégia,
          remoção de recurso).
        - É zerada ao abrir e ao fechar projeto.

        Sem projeto aberto, nunca há prompt.
        """
        return self._project is not None and self._is_dirty
