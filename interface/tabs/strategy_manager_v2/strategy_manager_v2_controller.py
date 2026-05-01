# interface/tabs/strategy_manager_v2/strategy_manager_v2_controller.py

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import List, Optional

from PyQt6 import QtWidgets, QtCore

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

from .comparison_tab.comparison_tab_widget import ComparisonTabWidget
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


class StrategyManagerV2Controller(QtCore.QObject):
    """
    Controlador principal da aba Strategy Manager V2 (nova arquitetura declarativa).

    Estado central:
        _project: FaceAuditProject | None — projeto atualmente aberto
        _current_definition: LRStrategyDefinition | None — estratégia em edição
        _definitions: List[LRStrategyDefinition] — todas as definições do projeto
    """

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

        # Comparison tab
        self._comparison_widget = ComparisonTabWidget()
        self.view.main_tabs.removeTab(1)
        self.view.main_tabs.addTab(self._comparison_widget, "Comparação")

    def _connect_signals(self) -> None:
        # Toolbar unificada
        self._project_toolbar.new_project_requested.connect(self.new_project)
        self._project_toolbar.open_project_requested.connect(self.open_project)
        self._project_toolbar.save_project_requested.connect(self.save_project)
        self._project_toolbar.close_project_requested.connect(self.close_project)
        self._project_toolbar.pipeline_requested.connect(self._on_pipeline)
        self._project_toolbar.mining_requested.connect(self._on_mining)
        self._project_toolbar.split_requested.connect(self._on_split)
        self._project_toolbar.manage_library_requested.connect(self._on_manage_library)
        # Left panel — estratégias
        self._left_panel.strategy_selected.connect(self._on_strategy_opened)
        self._left_panel.strategy_opened.connect(self._on_strategy_opened)
        self._left_panel.new_strategy_requested.connect(self._on_new_strategy)
        self._left_panel.duplicate_requested.connect(self._on_duplicate_strategy)
        self._left_panel.rename_requested.connect(self._on_rename_strategy)
        self._left_panel.remove_requested.connect(self._on_remove_strategy)
        self._left_panel.run_strategy_requested.connect(self.run_strategy)
        self._left_panel.save_as_template_requested.connect(self.save_as_template)

        # Left panel — importação/remoção de recursos
        self._left_panel.add_dataset_requested.connect(self._add_dataset)
        self._left_panel.add_split_config_requested.connect(self._add_split_config)
        self._left_panel.add_mining_config_requested.connect(self._add_mining_config)
        self._left_panel.add_pipeline_config_requested.connect(self._add_pipeline_config)
        self._left_panel.remove_resource_requested.connect(self._on_remove_resource)

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
        self._strategy_editor.resource_edit_requested.connect(
            self._on_resource_edit_requested
        )
        self._strategy_editor.instruction_applied.connect(
            self._on_instruction_applied
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

        # Avisa sobre problemas de integridade, mas não bloqueia
        if project.has_integrity_problems:
            problems = [
                r.describe()
                for r in (project.integrity_results or [])
                if r.is_problem
            ]
            QtWidgets.QMessageBox.warning(
                self._view,
                "Problemas de integridade",
                "Projeto aberto, mas foram encontrados problemas:\n\n"
                + "\n".join(problems),
            )

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

        try:
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
            QtWidgets.QMessageBox.information(
                self._view,
                "Recurso adicionado",
                f"Recurso '{resource_id}' importado com sucesso.\nPath: {ref.path}",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro ao importar", f"Não foi possível importar o recurso:\n{exc}"
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

        thread = QtCore.QThread()
        worker = StrategyExecutionWorker(executor, self._current_definition)
        worker.moveToThread(thread)

        # Mantém referência Python ao worker para evitar GC prematuro.
        self._active_worker = worker

        cancel_state = {"requested": False}

        thread.started.connect(worker.run)
        worker.log_message.connect(dialog.append_log)

        def on_cancel_requested():
            cancel_state["requested"] = True

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

    def save_as_template(self) -> None:
        if not self._current_definition or not self._global_storage:
            return

        template = copy.deepcopy(self._current_definition)
        template.strategy_id = str(uuid.uuid4())
        for resource in template.resources:
            resource.artifact_ref = None

        try:
            self._global_storage.save_artifact(
                StrategyArtifactType.STRATEGY_TEMPLATE,
                template,
                template_id=template.strategy_id,
            )
            QtWidgets.QMessageBox.information(
                self._view,
                "Template salvo",
                f"Estratégia '{template.name}' salva como template na biblioteca global.",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro", f"Não foi possível salvar o template:\n{exc}"
            )

    # ── Internal slots ────────────────────────────────────────────────────────

    def _set_project(self, project: FaceAuditProject) -> None:
        self._project = project
        self._definitions = list(project.definition_repository.list_definitions())
        self._current_definition = None
        self._project_toolbar.set_project_open(True)
        self._left_panel.set_strategy_actions_enabled(False)
        self._left_panel.set_project_root(project.root)
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
        # Árvore esquerda mostra TODOS os recursos do projeto (agrupados por
        # estratégia para tipos produzidos). Editor recebe apenas os recursos
        # acessíveis à estratégia atual: globais (datasets, configs) + os
        # produzidos pela própria estratégia.
        all_resources = self._collect_project_resources()
        resource_to_strategy = self._build_resource_to_strategy_map()
        self._left_panel.populate_strategies(self._definitions)
        self._left_panel.populate_resources(all_resources, resource_to_strategy=resource_to_strategy)

        editor_resources = self._collect_editor_resources(self._current_definition)
        self._strategy_editor.update_available_resources(editor_resources)

    def _collect_editor_resources(self, current_definition) -> List[StrategyResource]:
        """
        Lista de recursos disponíveis aos dropdowns do editor para a estratégia
        atualmente aberta:

        - Recursos globais (DATASET, PIPELINE_CONFIG, SPLIT_CONFIG, MINING_CONFIG,
          MERGE_CONFIG, SIMILARITY_MODEL_CONFIG, CALIBRATION_MODEL_CONFIG, ARTIFACT)
          — visíveis a qualquer estratégia, vindos de _collect_project_resources
          (definições + disk-scan).
        - Recursos produzidos (INTERMEDIATE_DATASET, TRAINED_PIPELINE,
          TRAINED_SIMILARITY_MODEL, TRAINED_CALIBRATION_MODEL, SIMILARITY_SCORES,
          TRAJECTORY_DATA, etc) — APENAS os declarados pela própria estratégia
          atual. Recursos produzidos por outras estratégias ficam invisíveis aqui.

        Sem `current_definition` (nenhuma estratégia aberta) retorna lista vazia.
        """
        if current_definition is None:
            return []

        _GLOBAL_TYPES = {
            StrategyResourceType.DATASET,
            StrategyResourceType.PIPELINE_CONFIG,
            StrategyResourceType.SPLIT_CONFIG,
            StrategyResourceType.MINING_CONFIG,
            StrategyResourceType.MERGE_CONFIG,
            StrategyResourceType.SIMILARITY_MODEL_CONFIG,
            StrategyResourceType.CALIBRATION_MODEL_CONFIG,
            StrategyResourceType.ARTIFACT,
        }

        seen_ids: set = set()
        resources: List[StrategyResource] = []

        # 1. Globais (vindos de qualquer estratégia ou do disk-scan).
        for r in self._collect_project_resources():
            if r.resource_type in _GLOBAL_TYPES and r.resource_id not in seen_ids:
                seen_ids.add(r.resource_id)
                resources.append(r)

        # 2. Produzidos: somente os da estratégia atual.
        for r in current_definition.resources:
            if r.resource_type not in _GLOBAL_TYPES and r.resource_id not in seen_ids:
                seen_ids.add(r.resource_id)
                resources.append(r)

        return resources

    def _build_resource_to_strategy_map(self) -> dict:
        """
        Mapeia resource_id → nome da estratégia que o produziu/declara.

        Usado pela árvore de recursos para sub-agrupar por estratégia os
        recursos produzidos por instruções (INTERMEDIATE_DATASET,
        TRAINED_PIPELINE, etc).
        """
        mapping: dict = {}
        for defn in self._definitions:
            for res in defn.resources:
                if res.resource_id not in mapping:
                    mapping[res.resource_id] = defn.name
        return mapping

    def _collect_project_resources(self) -> List[StrategyResource]:
        """
        Coleta recursos para exibição na árvore:
        1. Recursos declarados nas definições de estratégia.
        2. Arquivos importados em disco mas ainda não referenciados em nenhuma definição.

        A deduplicação entre (1) e (2) é feita por **path do artefato** — a
        pasta `datasets/` contém tanto datasets globais importados quanto
        intermediários produzidos por instruções; sem dedup por path, os
        intermediários aparecem em duplicidade (uma vez como DATASET via
        disk scan e outra como INTERMEDIATE_DATASET via definição).
        """
        seen_ids: set = set()
        seen_paths: set = set()
        resources: List[StrategyResource] = []

        # 1. Recursos declarados nas definições
        for defn in self._definitions:
            for res in defn.resources:
                if res.resource_id in seen_ids:
                    continue
                seen_ids.add(res.resource_id)
                if res.artifact_ref and res.artifact_ref.path:
                    seen_paths.add(res.artifact_ref.path)
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
                # Skip se o arquivo já está declarado como recurso de alguma estratégia.
                if ref.path in seen_paths:
                    continue
                resource_id = Path(ref.path).stem
                if resource_id in seen_ids:
                    continue

                # Heurística defensiva: arquivos cujo stem contém algum
                # resource_id já declarado (ex: prefixo "{strategy_id}_") são
                # considerados órfãos de uma versão anterior do recurso e
                # não devem aparecer novamente como "Datasets" soltos.
                if any(rid and rid in resource_id for rid in seen_ids):
                    continue

                seen_ids.add(resource_id)
                resources.append(StrategyResource(
                    resource_id=resource_id,
                    name=resource_id,
                    resource_type=resource_type,
                    artifact_ref=ref,
                ))

        return resources

    def _on_strategy_opened(self, definition: LRStrategyDefinition) -> None:
        self._current_definition = definition
        self._left_panel.set_strategy_actions_enabled(True)
        # Editor recebe apenas recursos acessíveis à estratégia atual:
        # globais + produzidos pela própria estratégia. Produzidos por
        # outras estratégias não aparecem nos dropdowns.
        editor_resources = self._collect_editor_resources(definition)
        self._strategy_editor.load_strategy(definition, available_resources=editor_resources)
        self._reset_execution_state()
        self._refresh_produced_resources()

        # Informa ao left panel as dimensões do embedding da estratégia selecionada
        emb_dims = None
        if definition.embedding_type is not None:
            emb_dims = definition.embedding_type.dimensions
        self._left_panel.set_embedding_dimensions(emb_dims)

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
        cloned = copy.deepcopy(definition)
        cloned.strategy_id = str(uuid.uuid4())
        cloned.name = f"{definition.name} (cópia)"
        try:
            self._project.definition_repository.save_definition(cloned)
            self._is_dirty = False
            self._refresh_left_panel()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view, "Erro", f"Não foi possível duplicar a estratégia:\n{exc}"
            )

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
        if not self._project:
            return
        reply = QtWidgets.QMessageBox.question(
            self._view,
            "Remover estratégia",
            f"Remover a estratégia '{definition.name}'?\nEssa ação não pode ser desfeita.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            # O repositório não tem delete_definition — remove o arquivo via storage
            from faceaudit.enums.strategy_file_type import StrategyFileType
            from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
            from faceaudit.enums.strategy_artifact_type import StrategyArtifactType as SAT
            rel = StrategyFileType.STRATEGY_DEFINITION.build_path(strategy_id=definition.strategy_id)
            ref = StrategyArtifactReference(artifact_type=SAT.STRATEGY_DEFINITION, path=str(rel), hash=None)
            abs_path = self._project.storage.resolve_artifact(ref)
            if abs_path and abs_path.exists():
                abs_path.unlink()
            if self._current_definition and self._current_definition.strategy_id == definition.strategy_id:
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

    def _on_instruction_applied(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Disparado quando o usuário clica "Aplicar alterações" numa instrução.
        Invalida os outputs persistidos: limpa artifact_ref/payload do recurso
        e apaga o arquivo do disco. Resultado: a instrução perde o ✅ até ser
        reexecutada.
        """
        if not self._project or not self._current_definition or instruction is None:
            return

        try:
            specs = instruction.get_resource_field_specs()
        except Exception:
            specs = []

        any_invalidated = False
        for spec in specs:
            if not spec.is_output:
                continue
            output_id = getattr(instruction, spec.field_name, None)
            if not output_id:
                continue

            existing = self._current_definition.get_resource_by_id(output_id)
            if existing is None:
                continue

            old_ref = existing.artifact_ref
            existing.artifact_ref = None
            existing.payload = None
            any_invalidated = True

            # Apaga arquivo do disco — o output anterior está stale relativo
            # à nova configuração e não deve continuar sendo claimed como o
            # output da instrução.
            if old_ref is not None and old_ref.path:
                try:
                    abs_path = self._project.storage.project_root / old_ref.path
                    if abs_path.exists():
                        abs_path.unlink()
                    # Stats JSON paralelo, se existir.
                    stats_path = abs_path.parent / (abs_path.stem + "_stats.json")
                    if stats_path.exists():
                        stats_path.unlink()
                except Exception:
                    pass

            # Remove também do contexto vivo se estiver lá — força nova execução.
            if (
                self._execution_context is not None
                and self._execution_context.has_resource(output_id)
            ):
                try:
                    self._execution_context.remove_resource(output_id)
                except Exception:
                    pass

        if any_invalidated:
            # Marca a instrução como não-executada e salva.
            self._executed_instruction_ids.discard(instruction.instruction_id)
            self._save_current_definition()
            # Recursos foram invalidados — instruções DOWNSTREAM que dependem
            # delas precisam ser revalidadas pelo gate de dependência.
            self._refresh_instruction_states()
            self._refresh_produced_resources()
            self._refresh_left_panel()

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
                if (
                    self._current_definition
                    and getattr(self._current_definition, "embedding_type", None) is not None
                ):
                    self._execution_context.set_metadata(
                        "embedding_dimensions",
                        self._current_definition.embedding_type.dimensions,
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

        # Une o set in-memory (instruções executadas nesta sessão) com o
        # set derivado do disco (instruções cujos outputs já estão
        # persistidos). Isso restaura ✅ ao reabrir o projeto e mantém ✅
        # mesmo após resets acidentais do contexto.
        self._executed_instruction_ids |= self._compute_executed_ids_from_disk()

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
                elif existing.has_artifact and self._executor is not None:
                    loader = getattr(self._executor, "_resource_loader", None)
                    if loader is not None:
                        try:
                            loader.hydrate(existing)
                        except Exception:
                            pass
                        if existing.has_payload:
                            trajectory_data = existing.payload

        if trajectory_data is None:
            QtWidgets.QMessageBox.information(
                self._view,
                "Trajetória indisponível",
                "Não foi encontrada uma trajetória 3D salva para esta instrução. "
                "Execute o pipeline com 'Salvar dados de trajetória' ativo.",
            )
            return

        try:
            from interface.tabs.strategy_manager.panels.preparation.cards.transformation_trajectory_viewer.transformation_trajectory_viewer_controller import (
                TransformationTrajectoryViewerController,
            )
            self._trajectory_viewer = TransformationTrajectoryViewerController(
                trajectory_data, parent=self._view
            )
            self._trajectory_viewer.show()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao abrir visualizador",
                f"Não foi possível abrir o visualizador de trajetória:\n{exc}",
            )

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

        score_dataset = None

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
            QtWidgets.QMessageBox.information(
                self._view,
                "Scores indisponíveis",
                "Não foram encontrados scores salvos para esta instrução. "
                "Execute a instrução de scoring primeiro.",
            )
            return

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
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao abrir visualizador",
                f"Não foi possível abrir o visualizador de distribuição:\n{exc}",
            )

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

        # Propaga novo hash a qualquer definition que referencie o recurso.
        for defn in self._definitions:
            touched = False
            for res in defn.resources:
                if res.resource_id == resource_id:
                    res.payload = new_config
                    res.artifact_ref = new_ref
                    touched = True
            if touched:
                try:
                    self._project.definition_repository.save_definition(defn)
                except Exception:
                    pass

        # Re-renderiza a árvore esquerda SEM recarregar definitions do disco
        # (o que descartaria seleções pendentes nos dropdowns do editor).
        # O payload/artifact_ref já foram atualizados in-place na lista
        # `self._definitions` em memória — só precisamos refletir isso na UI.
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
            QtWidgets.QMessageBox.information(
                self._view,
                "Resultado indisponível",
                "Não foi encontrado um resultado de validação salvo para esta "
                "instrução. Execute STRATEGY_EVALUATION primeiro.",
            )
            return

        try:
            from interface.tabs.strategy_manager_v2.strategy_editor.validation_result_viewer_dialog import (
                ValidationResultViewerDialog,
            )
            dlg = ValidationResultViewerDialog(validation_result, parent=self._view)
            self._validation_viewer = dlg  # mantém referência forte
            dlg.show()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao abrir visualizador",
                f"Não foi possível abrir o visualizador de validação:\n{exc}",
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

        casework_result = None

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
            QtWidgets.QMessageBox.information(
                self._view,
                "Visualização indisponível",
                "Não foi encontrada uma visualização salva para esta "
                "instrução. Execute CASEWORK_VISUALIZATION primeiro.",
            )
            return

        try:
            from interface.tabs.strategy_manager_v2.strategy_editor.casework_visualization_viewer_dialog import (
                CaseworkVisualizationViewerDialog,
            )
            dlg = CaseworkVisualizationViewerDialog(casework_result, parent=self._view)
            self._casework_viewer = dlg  # mantém referência forte
            dlg.show()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._view,
                "Erro ao abrir visualizador",
                f"Não foi possível abrir o visualizador de caso:\n{exc}",
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
        if config and self._global_storage:
            try:
                self._global_storage.save_artifact(
                    StrategyArtifactType.PIPELINE_CONFIG,
                    config,
                    config_id=str(uuid.uuid4())[:8],
                )
                QtWidgets.QMessageBox.information(
                    self._view, "Pipeline salvo", "Configuração de pipeline salva na biblioteca global."
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self._view, "Erro", f"Não foi possível salvar o pipeline:\n{exc}"
                )

    def _on_mining(self) -> None:
        ctrl = NewMiningConfigDialogController(parent=self._view)
        if ctrl.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            config = ctrl.get_config()
            if config and self._global_storage:
                try:
                    self._global_storage.save_artifact(
                        StrategyArtifactType.MINING_CONFIG,
                        config,
                        config_id=str(uuid.uuid4())[:8],
                    )
                    QtWidgets.QMessageBox.information(
                        self._view, "Mining salvo", "Configuração de mining salva na biblioteca global."
                    )
                except Exception as exc:
                    QtWidgets.QMessageBox.critical(
                        self._view, "Erro", f"Não foi possível salvar a config de mining:\n{exc}"
                    )

    def _on_split(self) -> None:
        ctrl = NewSplitConfigDialogController(parent=self._view)
        result = ctrl.exec()
        if result and self._global_storage:
            try:
                self._global_storage.save_artifact(
                    StrategyArtifactType.SPLIT_CONFIG,
                    result,
                    config_id=str(uuid.uuid4())[:8],
                )
                QtWidgets.QMessageBox.information(
                    self._view, "Split salvo", "Configuração de split salva na biblioteca global."
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self._view, "Erro", f"Não foi possível salvar a config de split:\n{exc}"
                )

    def _on_manage_library(self) -> None:
        ctrl = ManageLibraryController(parent=self._view, global_storage=self._global_storage)
        ctrl.exec()

    # ── Helpers ───────────────────────────────────────────────────────────────

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

            for defn in self._definitions:
                modified = False

                # Limpa campos `output_*_resource_id` das instruções que apontam
                # para este resource_id, mantendo os campos como None.
                for instr in defn.instructions:
                    try:
                        specs = instr.get_resource_field_specs()
                    except Exception:
                        specs = []
                    for spec in specs:
                        if not spec.is_output:
                            continue
                        if getattr(instr, spec.field_name, None) == resource.resource_id:
                            setattr(instr, spec.field_name, None)
                            modified = True
                    if modified and hasattr(instr, "_sync_declared_io"):
                        try:
                            instr._sync_declared_io()
                        except Exception:
                            pass
                    if modified:
                        instructions_unexecuted.append(instr.instruction_id)

                # Remove o recurso da lista de resources da definição.
                if defn.get_resource_by_id(resource.resource_id) is not None:
                    defn.remove_resource(resource.resource_id)
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
            if (
                resource.artifact_ref is not None
                and old_ref is not None
                and old_ref.path
                and old_ref.path != resource.artifact_ref.path
            ):
                try:
                    old_abs = self._project.storage.project_root / old_ref.path
                    if old_abs.exists():
                        old_abs.unlink()
                except Exception:
                    pass

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
            # mudou e a extensão/pasta de destino também).
            if (
                artifact_ref is not None
                and old_ref is not None
                and old_ref.path
                and old_ref.path != artifact_ref.path
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
                    else:
                        self._current_definition.add_resource(resource)

                    # Limpa arquivo antigo órfão quando o caminho mudou
                    # (tipicamente porque a extensão/pasta mudou junto com
                    # o resource_type).
                    if (
                        old_ref is not None
                        and old_ref.path
                        and old_ref.path != artifact_ref.path
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
        """Salva a definição da estratégia atual no repositório."""
        if self._project and self._current_definition:
            try:
                self._project.definition_repository.save_definition(
                    self._current_definition
                )
                self._is_dirty = False
            except Exception:
                pass

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
