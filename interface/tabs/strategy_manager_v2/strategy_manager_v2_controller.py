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

        thread.started.connect(worker.run)
        worker.log_message.connect(dialog.append_log)

        def on_finished(model):
            for instr in self._current_definition.instructions:
                self._executed_instruction_ids.add(instr.instruction_id)

            # Persistir todos os recursos produzidos pelo Run All
            self._persist_trained_model_resources(model, executor)

            self._refresh_instruction_states()
            self._refresh_produced_resources()
            dialog.finish(True)

        worker.finished.connect(on_finished)
        worker.error.connect(lambda msg: dialog.finish(False))
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
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

    def _refresh_left_panel(self) -> None:
        if not self._project:
            return
        self._definitions = list(self._project.definition_repository.list_definitions())
        resources = self._collect_project_resources()
        self._left_panel.populate_strategies(self._definitions)
        self._left_panel.populate_resources(resources)
        # Propaga recursos atualizados ao editor para que dropdowns reflitam novos recursos
        self._strategy_editor.update_available_resources(resources)

    def _collect_project_resources(self) -> List[StrategyResource]:
        """
        Coleta recursos para exibição na árvore:
        1. Recursos declarados nas definições de estratégia.
        2. Arquivos importados em disco mas ainda não referenciados em nenhuma definição.
        """
        seen_ids: set = set()
        resources: List[StrategyResource] = []

        # 1. Recursos declarados nas definições
        for defn in self._definitions:
            for res in defn.resources:
                if res.resource_id not in seen_ids:
                    seen_ids.add(res.resource_id)
                    resources.append(res)

        # 2. Arquivos importados no disco (não vinculados a nenhuma definição ainda)
        _import_scan = [
            (StrategyArtifactType.GLOBAL_DATASET,  StrategyResourceType.DATASET),
            (StrategyArtifactType.SPLIT_CONFIG,    StrategyResourceType.SPLIT_CONFIG),
            (StrategyArtifactType.MINING_CONFIG,   StrategyResourceType.ARTIFACT),
            (StrategyArtifactType.PIPELINE_CONFIG, StrategyResourceType.PIPELINE_CONFIG),
        ]
        for artifact_type, resource_type in _import_scan:
            for ref in self._project.storage.list_artifacts(artifact_type):
                resource_id = Path(ref.path).stem
                if resource_id not in seen_ids:
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
        resources = self._collect_project_resources()
        self._strategy_editor.load_strategy(definition, available_resources=resources)
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

    # ── Execution state management ─────────────────────────────────────────────

    def _reset_execution_state(self) -> None:
        """Reseta contexto e estados visuais quando a estratégia muda."""
        self._execution_context = None
        self._executed_instruction_ids.clear()
        self._executor = None
        self._refresh_instruction_states()

    def _ensure_executor_and_context(self) -> bool:
        """Cria executor e contexto se ainda não existem. Retorna False em caso de erro."""
        if self._executor is None:
            try:
                self._executor = StrategyExecutorFactory.build(
                    artifact_storage=self._project.storage,
                )
            except Exception as exc:
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
                QtWidgets.QMessageBox.critical(
                    self._view, "Erro", f"Não foi possível preparar contexto:\n{exc}"
                )
                return False

        return True

    def _register_project_resources_in_context(self) -> None:
        """Registra no contexto recursos do projeto que não estão na definição da estratégia."""
        if self._execution_context is None or self._executor is None:
            return

        embedding_column = None
        if (
            self._current_definition
            and getattr(self._current_definition, "embedding_type", None) is not None
        ):
            embedding_column = self._current_definition.embedding_type.parquet_column

        for resource in self._collect_project_resources():
            if self._execution_context.has_resource(resource.resource_id):
                continue
            if self._executor._resource_loader is not None:
                try:
                    self._executor._resource_loader.hydrate(
                        resource, embedding_column=embedding_column
                    )
                except Exception:
                    pass  # Recurso sem payload — validação reportará se necessário
            self._execution_context.set_resource(resource)

    def _refresh_instruction_states(self) -> None:
        """Recalcula quais instruções são executáveis e atualiza a UI."""
        if not self._current_definition:
            return

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
        dialog.show()

        thread = QtCore.QThread()
        worker = InstructionExecutionWorker(
            self._executor, instruction, self._execution_context
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.log_message.connect(dialog.append_log)

        def on_finished(result):
            if result.success:
                self._executed_instruction_ids.add(instruction.instruction_id)
                self._persist_produced_resources(result.produced_resource_ids)
                dialog.finish(True)
            else:
                dialog.finish(False)
            self._refresh_instruction_states()
            self._refresh_produced_resources()

        def on_error(msg):
            dialog.finish(False)
            self._refresh_instruction_states()

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
        """Remove o arquivo do recurso do disco e atualiza a árvore."""
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
            abs_path = self._project.storage.project_root / resource.artifact_ref.path
            if abs_path.exists():
                abs_path.unlink()

            # Remove também o stats JSON se existir
            stats_path = abs_path.parent / (abs_path.stem + "_stats.json")
            if stats_path.exists():
                stats_path.unlink()

            self._left_panel.resource_info.clear()
            self._refresh_left_panel()
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

        # Adicionar placeholders que não existem
        for resource_id, placeholder in declared_outputs.items():
            existing = self._current_definition.get_resource_by_id(resource_id)
            if existing is None:
                self._current_definition.add_resource(placeholder)

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

    def _persist_produced_resources(self, resource_ids: list) -> None:
        """Persiste recursos produzidos em disco e atualiza a definição."""
        if not self._execution_context or not self._executor or not self._current_definition:
            return

        for resource_id in resource_ids:
            resource = self._execution_context.get_resource(resource_id)
            if resource is None or not resource.has_payload:
                continue

            try:
                artifact_ref = self._executor.persist_produced_resource(
                    resource=resource,
                    strategy_id=self._current_definition.strategy_id,
                )
                if artifact_ref:
                    resource.artifact_ref = artifact_ref
                    existing = self._current_definition.get_resource_by_id(resource_id)
                    if existing:
                        existing.artifact_ref = artifact_ref
                        existing.payload = resource.payload
                    else:
                        self._current_definition.add_resource(resource)
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
        """Heurística simples: sempre assume que pode haver mudanças não salvas."""
        return self._project is not None
