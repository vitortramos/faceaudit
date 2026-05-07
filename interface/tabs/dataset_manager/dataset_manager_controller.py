import os
from pathlib import Path
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication

# --- COMPONENTES VISUAIS (SUB-CONTROLLERS) ---
from .toolbar.manager_toolbar_controller import ManagerToolbarController
from .table.manager_table_controller import ManagerTableController
from .detail_panel.detail_panel_controller import DetailPanelController

# --- DIÁLOGOS DE FUNCIONALIDADE ---
from .new_dataset_dialog.new_dataset_controller import NewDatasetController
from .filter_dialog.filter_dialog_controller import FilterDialogController
from .stats_dialog.stats_dialog_controller import StatsDialogController
from .compare_dialog.compare_dialog_controller import CompareDialogController
from .enrich_dataset_dialog.enrich_dataset_controller import EnrichDatasetController
from .split_dataset_dialog.split_dataset_dialog_controller import SplitDatasetDialogController

# [NOVO] Controlador do Diálogo de Análise Forense
from .dataset_analysis.dataset_analysis_controller import DatasetAnalysisController

# --- WORKERS (PROCESSAMENTO) ---
from .workers.load_dataset_worker import LoadDatasetWorker
from .workers.localize_dataset_worker import LocalizeDatasetWorker  # PR L1d
from .workers.filter_worker import FilterWorker
from .workers.new_dataset_worker import NewDatasetWorker
from .workers.enrich_worker import EnrichWorker
from .workers.worker_progress_runner import WorkerProgressRunner

# --- MODELOS ---
from faceaudit.models import FaceDataset
from faceaudit.models.filter_config import FilterConfig
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig

# PR 25 (Tema 5.4) — audit log forense / chain of custody
from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.services.audit_log_service import AuditLogService
from faceaudit.services.hash_service import HashService


class DatasetManagerController(QtCore.QObject):
    """
    Controlador Mestre do Gerenciamento de Datasets.
    """
    
    dataset_changed = QtCore.pyqtSignal()
    # PR 17: emitido quando o usuário pede pra enviar o dataset corrente
    # pra V2. Carrega o caminho absoluto do parquet salvo. Capturado pelo
    # MainGUIController, que troca de aba e chama a V2.
    send_to_strategy_requested = QtCore.pyqtSignal(str)

    # RECEBE O CONTAINER AQUI
    def __init__(self, view, pipeline, config, container):
        super().__init__()
        self.view = view
        self.pipeline = pipeline
        self.config = config
        self.container = container # Guardamos a referência do container para injeção explícita

        # Estado Global
        self.dataset = None
        self.dataset_original = None
        self.current_path = None
        self.current_filter_config = FilterConfig()
        self._is_dirty = False

        # Ref forte para o runner ativo — o WorkerProgressRunner gerencia o
        # próprio QThread + QProgressDialog; só precisamos manter vivo até
        # ele finalizar (quando finished/error disparam, é descartado).
        self._active_runner: WorkerProgressRunner | None = None

        # --- 1. SUB-CONTROLADORES ---
        self.toolbar_ctrl = ManagerToolbarController(parent_layout=None)
        self.table_ctrl = ManagerTableController(parent_layout=None)
        self.detail_ctrl = DetailPanelController(parent_layout=None)

        # PR 15: galeria embutida em QStackedWidget junto com a tabela.
        # Toggle na toolbar troca o índice (0=tabela, 1=galeria).
        from .gallery.gallery_widget import GalleryWidget
        self.gallery_widget = GalleryWidget()
        self._view_stack = QtWidgets.QStackedWidget()
        self._view_stack.addWidget(self.table_ctrl.view)   # index 0
        self._view_stack.addWidget(self.gallery_widget)    # index 1

        if hasattr(self.view, 'toolbar_layout'):
            self.view.toolbar_layout.addWidget(self.toolbar_ctrl.view)
        if hasattr(self.view, 'table_layout'):
            self.view.table_layout.addWidget(self._view_stack)
        if hasattr(self.view, 'detail_layout'):
            self.view.detail_layout.addWidget(self.detail_ctrl.view)

        # --- 2. SINAIS ---
        self._connect_submodules()

    def _connect_submodules(self):
        self.toolbar_ctrl.action_new.connect(self.fluxo_novo_dataset)
        self.toolbar_ctrl.action_load.connect(self.fluxo_abrir_dataset)
        self.toolbar_ctrl.action_close.connect(self.fluxo_fechar_dataset)
        self.toolbar_ctrl.action_save.connect(self.fluxo_salvar)
        self.toolbar_ctrl.action_save_as.connect(self.fluxo_salvar_como)
        self.toolbar_ctrl.action_merge.connect(self.fluxo_merge)
        self.toolbar_ctrl.action_split.connect(self.fluxo_split)
        self.toolbar_ctrl.action_compare.connect(self.fluxo_comparar)

        self.toolbar_ctrl.action_stats.connect(self.fluxo_estatisticas)
        # [NOVO] Conecta o botão Analisar ao novo fluxo
        self.toolbar_ctrl.action_analyze.connect(self.fluxo_analise_dataset)
        # PR 24 (Tema 5.3 + 5.5) — relatório de schema + export CSV/Excel
        self.toolbar_ctrl.action_schema_report.connect(self.fluxo_schema_report)
        # PR 25 (Tema 5.4) — audit log forense
        if hasattr(self.toolbar_ctrl, "action_audit_log"):
            self.toolbar_ctrl.action_audit_log.connect(self.fluxo_audit_log)
        # PR L1d — Localização de paths
        if hasattr(self.toolbar_ctrl, "action_localize"):
            self.toolbar_ctrl.action_localize.connect(self.fluxo_localizar)

        self.toolbar_ctrl.action_filter.connect(self.fluxo_filtrar)
        self.toolbar_ctrl.action_clear_filter.connect(self.fluxo_limpar_filtro)
        self.table_ctrl.face_selected.connect(self.detail_ctrl.load_face_data)

        # Edits no modelo da tabela marcam dirty; sem isso, alterações de célula
        # ficavam invisíveis até um flush manual e podiam ser perdidas no fechamento.
        if hasattr(self.table_ctrl, "model"):
            self.table_ctrl.model.dataChanged.connect(self._on_table_data_changed)

        # PR 8 — Spreadsheet UX
        if hasattr(self.toolbar_ctrl, "action_add_column"):
            self.toolbar_ctrl.action_add_column.connect(self.fluxo_adicionar_coluna)
        if hasattr(self.table_ctrl, "schema_changed"):
            self.table_ctrl.schema_changed.connect(self._on_schema_changed)

        # PR 15 — Modo galeria
        if hasattr(self.toolbar_ctrl, "action_toggle_gallery"):
            self.toolbar_ctrl.action_toggle_gallery.connect(self._on_toggle_gallery)
        # Selection sync: galeria → detail panel + tabela
        self.gallery_widget.face_selected.connect(self._on_gallery_face_selected)

        # PR 15.2 — Comparar par lado a lado (tabela + galeria emitem
        # com a mesma assinatura: compare_pair_requested(int, int))
        if hasattr(self.table_ctrl, "compare_pair_requested"):
            self.table_ctrl.compare_pair_requested.connect(self._on_compare_pair)
        self.gallery_widget.compare_pair_requested.connect(self._on_compare_pair)

        # PR 16 — Adicionar à Biblioteca Global
        if hasattr(self.toolbar_ctrl, "action_save_to_library"):
            self.toolbar_ctrl.action_save_to_library.connect(self.fluxo_salvar_biblioteca)

        # PR 17 — Enviar pra Estratégia (V2)
        if hasattr(self.toolbar_ctrl, "action_send_to_strategy"):
            self.toolbar_ctrl.action_send_to_strategy.connect(self.fluxo_enviar_para_estrategia)

        if hasattr(self.toolbar_ctrl, 'action_enrich'):
            self.toolbar_ctrl.action_enrich.connect(self.fluxo_enriquecer)

    def _on_table_data_changed(self, *_args):
        if self.dataset is not None:
            self.set_dirty(True)

    # ── PR 15: Modo galeria ──────────────────────────────────────────

    def _on_toggle_gallery(self, is_gallery: bool) -> None:
        """Toggle entre tabela (index 0) e galeria (index 1)."""
        if is_gallery:
            # Repopula a galeria com as faces atuais — barato pra cache hit
            # (mesmo dataset, mesmos paths). Garante que edições recentes
            # na tabela aparecem no grid.
            faces = list(self.dataset.faces) if self.dataset else []
            self.gallery_widget.populate(faces)
            self._view_stack.setCurrentIndex(1)
        else:
            self._view_stack.setCurrentIndex(0)

    def _on_gallery_face_selected(self, face_idx: int) -> None:
        """Click no grid → atualiza detail panel (mesma semântica que tabela).

        A tabela e a galeria compartilham o mesmo dataframe via
        `table_ctrl.model`, então reusamos `get_row_data(face_idx)` pra
        propagar pra detail panel — mesmo path que o slot da tabela.
        """
        if self.dataset is None:
            return
        if hasattr(self.table_ctrl, "model"):
            row_data = self.table_ctrl.model.get_row_data(face_idx)
            if row_data is not None:
                self.detail_ctrl.load_face_data(row_data)

    # ── PR 15.2: comparar par de faces lado a lado ────────────────────

    def _on_compare_pair(self, idx_a: int, idx_b: int) -> None:
        """Abre o ComparePairController com as duas faces selecionadas.

        Disparado tanto pelo context menu da tabela quanto da galeria.
        Recebe índices no `dataset.faces` (mesma indexação que o
        dataframe, por construção). Usa `comparison_service` quando
        disponível pra calcular cosine similarity.
        """
        if self.dataset is None:
            return
        n = len(self.dataset.faces)
        if not (0 <= idx_a < n and 0 <= idx_b < n):
            return
        if idx_a == idx_b:
            QMessageBox.warning(self.view, "Comparar par",
                                "Os dois índices apontam pra mesma face.")
            return

        face_a = self.dataset.faces[idx_a]
        face_b = self.dataset.faces[idx_b]

        # comparison_service vem do container (já usado em outras flows
        # como `_abrir_analise_real`); ausência é tolerada — o dialog
        # só não mostra similaridade calculada.
        comp_svc = None
        try:
            comp_svc = self.container.comparison_service()
        except Exception:
            pass

        from .compare_pair_dialog.compare_pair_controller import ComparePairController
        cmp_ctrl = ComparePairController(
            face_a=face_a, face_b=face_b,
            comparison_service=comp_svc,
            parent_view=self.view,
        )
        cmp_ctrl.exec()

    # ── PR 8: Spreadsheet UX (add column + show/hide) ─────────────────

    def _on_schema_changed(self):
        """Disparado quando o usuário muda visibilidade/adiciona coluna via
        header context-menu ou toolbar. Propaga o novo column_specs (que o
        model já mutou in-place) para o FaceDataset e marca dirty.
        """
        if self.dataset is None:
            return
        # column_specs é o mesmo objeto referenciado pelo model — mas
        # garantimos a atribuição explícita em caso de refactor futuro.
        self.dataset.column_specs = dict(self.table_ctrl.model.column_specs)
        self.set_dirty(True)

    def fluxo_adicionar_coluna(self):
        """Abre o AddColumnDialog. Se aceito, adiciona a coluna ao model
        (via add_custom_column do DatasetTableModel) e propaga ao dataset.
        """
        if self.dataset is None:
            return
        if not self.table_ctrl.model.column_specs:
            QMessageBox.warning(
                self.view, "Schema legado",
                "Este dataset foi carregado de um parquet sem column_specs "
                "na metadata. Salve-o primeiro (Salvar) para atualizar para "
                "o novo formato — depois você pode adicionar colunas custom.",
            )
            return

        from .column_dialogs.add_column_dialog import AddColumnDialog
        existing = set(self.table_ctrl.model.column_specs.keys())
        dialog = AddColumnDialog(existing_column_names=existing, parent=self.view)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        name, label, dtype, default, computed_expr = dialog.get_result()
        ok = self.table_ctrl.model.add_custom_column(
            name=name, label=label, dtype=dtype, default_value=default,
            editable=True, computed_expr=computed_expr,
        )
        if not ok:
            QMessageBox.warning(self.view, "Erro", "Falha ao adicionar coluna.")
            return

        # Re-propaga + marca dirty
        self._on_schema_changed()
        self.table_ctrl.view.resizeColumnsToContents()
        # PR 25 (Tema 5.4) — registra ADD_COLUMN.
        self._audit(
            AuditOperation.ADD_COLUMN,
            f"Coluna '{name}' adicionada (rótulo: '{label}', dtype: {dtype}).",
            metadata={
                "column": name,
                "label": label,
                "dtype": str(dtype),
                "computed_expr": computed_expr,
            },
        )

    def _get_initial_dir(self) -> str:
        if self.pipeline and hasattr(self.pipeline, 'storage_path'):
            path = str(self.pipeline.storage_path)
            if os.path.isdir(path): return path
        return ""

    # ── PR 25 (Tema 5.4) — Audit log forense / chain of custody ──────────
    def _audit(
        self,
        operation: AuditOperation,
        description: str,
        *,
        metadata: dict | None = None,
        path_override: str | None = None,
        hash_file: bool = False,
    ) -> None:
        """Append best-effort no audit log do dataset corrente.

        - `path_override`: para SAVE_AS, onde `current_path` ainda
          aponta para o path antigo no momento da chamada.
        - `hash_file`: quando True, calcula SHA-256 do parquet via
          `HashService.calculate_file_hash` e grava em `dataset_hash`.
          Pular em arquivos > 500 MB (nesses casos grava
          `metadata['hash_skipped']=True` com o tamanho).
        - Falhas no append ficam em stderr e NÃO interrompem o fluxo
          chamado — auditoria é observabilidade, não bloqueio.
        """
        path = path_override or self.current_path
        if not path:
            return  # dataset em memória ainda — sem sidecar pra escrever
        try:
            ds_hash = None
            extra_meta = dict(metadata or {})
            if hash_file:
                try:
                    p = Path(path)
                    if p.exists():
                        size_mb = p.stat().st_size / (1024 * 1024)
                        if size_mb > 500.0:
                            extra_meta["hash_skipped"] = True
                            extra_meta["file_size_mb"] = round(size_mb, 1)
                        else:
                            ds_hash = HashService().calculate_file_hash(p)
                except Exception as e:
                    extra_meta["hash_error"] = str(e)
            AuditLogService.append(
                path,
                operation,
                description,
                metadata=extra_meta,
                dataset_hash=ds_hash,
            )
        except Exception as e:
            # Best-effort — não abortar o fluxo principal
            import sys as _sys
            print(f"[AuditLog] Falha ao registrar {operation.value}: {e}",
                  file=_sys.stderr)

    def carregar_dataset_na_tela(self, dataset, path=None):
        self.dataset = dataset
        self.current_path = path

        # Schema-by-role (PR 6): se o FaceDataset trouxe column_specs (parquet
        # pós-PR 3), o table_ctrl propaga pra model + instala delegates de
        # vocabulário. Caso contrário (parquet legado ou dataset criado em
        # memória), o model cai no enum estático.
        column_specs = getattr(self.dataset, "column_specs", None) or {}
        if hasattr(self.table_ctrl, "apply_column_specs"):
            self.table_ctrl.apply_column_specs(column_specs)
        elif hasattr(self.table_ctrl, "model") and hasattr(self.table_ctrl.model, "set_column_specs"):
            self.table_ctrl.model.set_column_specs(column_specs)

        self.table_ctrl.load_dataframe(self.dataset.to_pandas())

        # PR 15: invalida cache da galeria (dataset trocou) e re-popula só
        # se o usuário estiver no modo galeria. No modo tabela, a galeria
        # popula on-demand quando o usuário ativar o toggle.
        self.gallery_widget.clear_cache()
        if self._view_stack.currentIndex() == 1:
            self.gallery_widget.populate(list(self.dataset.faces))
        else:
            # Repopula com lista vazia pra resetar o estado interno do widget
            # (evita seleção velha quando o user trocar pra modo galeria).
            self.gallery_widget.populate([])

        self.set_dirty(False)
        
        qtd_imgs = len(self.dataset)
        qtd_ids = self.dataset.number_of_identities
        nome = self.dataset.name if self.dataset.name else "Sem Nome"
        status_txt = f"📂 <b>{nome}</b> | 🖼️ {qtd_imgs} Imagens | 👤 {qtd_ids} IDs"
        self.toolbar_ctrl.set_dataset_loaded_state(True, status_txt)

    def sync_changes_from_table(self):
        if not self.dataset: return
        df_edited = self.table_ctrl.get_current_dataframe()
        if df_edited is None or df_edited.empty: return

        dataset_visual_atualizado = FaceDataset.from_dataframe(df_edited, name=self.dataset.name)
        # PR 21 (Tema 4.2) — `from_dataframe` constrói um FaceDataset novo
        # sem column_specs (default {}). Preserva os specs vindos do
        # parquet/UI pra que custom columns continuem visíveis no filter
        # dialog e na próxima save (parquet metadata).
        if self.dataset.column_specs:
            dataset_visual_atualizado.column_specs = dict(self.dataset.column_specs)
        self.dataset = dataset_visual_atualizado

        if self.dataset_original:
            print(f"🔄 Sync: Propagando {len(self.dataset)} registros editados para o Dataset Original...")
            mapa_edicoes = {face.image_path: face for face in self.dataset.faces}
            count_updates = 0
            for i, face_orig in enumerate(self.dataset_original.faces):
                if face_orig.image_path in mapa_edicoes:
                    self.dataset_original.faces[i] = mapa_edicoes[face_orig.image_path]
                    count_updates += 1
            print(f"✅ Sync Concluído: {count_updates} faces atualizadas no backup.")
        else:
            print(f"✅ Sync Simples: Dataset atualizado ({len(self.dataset)} registros).")
        # NOTE: dirty é marcado por _on_table_data_changed quando o usuário edita
        # uma célula. Sync apenas propaga; não inventa estado dirty quando o df
        # estiver sem alterações reais.

    def is_dirty(self) -> bool:
        return self._is_dirty

    def set_dirty(self, state: bool):
        self._is_dirty = state
        if self.dataset:
            qtd_imgs = len(self.dataset)
            qtd_ids = self.dataset.number_of_identities
            nome = self.dataset.name if self.dataset.name else "Sem Nome"
            asterisco = "*" if state else ""
            status_txt = f"📂 <b>{nome}{asterisco}</b> | 🖼️ {qtd_imgs} Imagens | 👤 {qtd_ids} IDs"
            self.toolbar_ctrl.set_dataset_loaded_state(True, status_txt)

    def fluxo_abrir_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Abrir Dataset", self._get_initial_dir(), "Parquet (*.parquet)"
        )
        if not path:
            return
        self._run_worker(
            LoadDatasetWorker(path),
            label="Lendo arquivo Parquet...",
            allow_cancel=False,
            determinate=False,  # pyarrow não emite progresso
            wait_cursor=True,
            on_finished=lambda ds: self._on_load_success(ds, path),
        )

    def _run_worker(
        self,
        worker,
        *,
        label,
        allow_cancel=False,
        determinate=True,
        wait_cursor=False,
        use_log_dialog=False,
        on_finished=None,
    ):
        """Encapsula a criação do WorkerProgressRunner + ref forte.

        `use_log_dialog=True` ativa o LogProgressDialog (área de texto
        rolando os logs) — adequado pra fluxos multi-etapa onde o
        usuário precisa ver cada fase (ingestão de dataset, enriquecimento).
        Default é `False` (QProgressDialog compacto).

        Só permite uma operação ativa de cada vez — se o usuário disparar
        outra antes da anterior terminar, a anterior é cancelada
        cooperativamente.
        """
        if self._active_runner is not None:
            self._active_runner.stop_and_wait()
            self._active_runner = None

        def _wrap_finished(payload):
            self._active_runner = None
            if on_finished is not None:
                on_finished(payload)

        def _wrap_error(msg):
            self._active_runner = None
            QMessageBox.critical(self.view, "Erro", msg)

        runner = WorkerProgressRunner(
            self.view,
            worker,
            label=label,
            allow_cancel=allow_cancel,
            determinate=determinate,
            wait_cursor=wait_cursor,
            use_log_dialog=use_log_dialog,
            on_finished=_wrap_finished,
            on_error=_wrap_error,
        )
        self._active_runner = runner
        runner.start()
        return runner

    def _on_load_success(self, dataset, path):
        try:
            self.carregar_dataset_na_tela(dataset, path)
            # PR 25 (Tema 5.4) — registra OPEN no audit log do dataset.
            # hash_file=True calcula SHA-256 do parquet (eficiente via
            # hashlib.file_digest) — anchor da cadeia de custódia.
            self._audit(
                AuditOperation.OPEN,
                f"Dataset '{dataset.name}' aberto.",
                metadata={
                    "rows": len(dataset),
                    "identities": dataset.number_of_identities,
                },
                hash_file=True,
            )
        except Exception as e:
            QMessageBox.critical(self.view, "Erro de Renderização", f"Falha ao exibir tabela: {e}")

    def fluxo_salvar(self):
        if not self.dataset: return
        self.sync_changes_from_table()

        if self.current_path:
            try:
                self.dataset.to_parquet(self.current_path)
                QMessageBox.information(self.view, "Salvo", "Dataset salvo com sucesso.")
                self.set_dirty(False)
                # PR 25 (Tema 5.4) — registra SAVE com hash do arquivo
                # recém-escrito.
                self._audit(
                    AuditOperation.SAVE,
                    f"Dataset salvo em '{Path(self.current_path).name}'.",
                    metadata={
                        "rows": len(self.dataset),
                        "identities": self.dataset.number_of_identities,
                    },
                    hash_file=True,
                )
            except Exception as e:
                QMessageBox.critical(self.view, "Erro", f"Falha ao salvar: {e}")
        else:
            self.fluxo_salvar_como()

    def fluxo_salvar_como(self):
        if not self.dataset: return
        self.sync_changes_from_table()

        default_name = f"{self.dataset.name}.parquet"
        init_path = os.path.join(self._get_initial_dir(), default_name)

        path, _ = QFileDialog.getSaveFileName(self.view, "Salvar Como", init_path, "Parquet (*.parquet)")
        if path:
            old_path = self.current_path
            self.dataset.name = Path(path).stem
            self.dataset.to_parquet(path)
            # PR 25 (Tema 5.4) — preserva o sidecar do dataset original
            # quando o usuário faz Save As: copia o `.audit.json` antigo
            # para o novo path antes de adicionar a entrada SAVE_AS,
            # mantendo a cadeia de custódia ininterrupta.
            if old_path and old_path != path:
                try:
                    AuditLogService.migrate_sidecar(old_path, path)
                except Exception:
                    pass  # se a migração falhar, segue com chain nova
            self.current_path = path
            self.set_dirty(False)
            self.carregar_dataset_na_tela(self.dataset, path)
            self._audit(
                AuditOperation.SAVE_AS,
                f"Dataset salvo em novo arquivo '{Path(path).name}'.",
                metadata={
                    "previous_path": old_path,
                    "rows": len(self.dataset),
                    "identities": self.dataset.number_of_identities,
                },
                path_override=path,
                hash_file=True,
            )

    # ── PR 16: Adicionar dataset à Biblioteca Global ──────────────────────

    def fluxo_salvar_biblioteca(self):
        """Copia o parquet salvo do dataset atual pra biblioteca global,
        ficando reusável entre projetos do Strategy Manager V2.

        Pré-condições:
          - Dataset carregado e salvo em disco (current_path setado).
          - Se houver edições não-salvas, prompt pra salvar antes (a
            biblioteca só recebe o que está em disco).
          - Library path configurado (`config.paths.library`).
        """
        if not self.dataset:
            return

        # 1. Garante que o dataset foi salvo primeiro — a biblioteca
        # global aceita só arquivos em disco; edições em memória ficam de fora.
        self.sync_changes_from_table()
        if self._is_dirty or not self.current_path or not Path(self.current_path).exists():
            res = QMessageBox.question(
                self.view, "Salvar antes",
                "O dataset precisa estar salvo em disco antes de ir pra "
                "biblioteca global. Salvar agora?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Cancel,
            )
            if res != QMessageBox.StandardButton.Save:
                return
            self.fluxo_salvar()
            if self._is_dirty or not self.current_path:
                return  # falhou ou usuário cancelou o save dialog

        # 2. Resolve o root da biblioteca via app_config.paths.library
        library_root = getattr(getattr(self.config, "paths", None), "library", None)
        if not library_root:
            QMessageBox.warning(
                self.view, "Biblioteca não configurada",
                "Configure 'paths.library' no config.json antes de "
                "adicionar datasets à biblioteca global.",
            )
            return

        # 3. Pede o dataset_id (default = nome atual). Permite versionar
        # ao salvar (ex: "passaportes_2025_01" sem afetar o nome interno
        # do dataset).
        from PyQt6.QtWidgets import QInputDialog
        default_id = self.dataset.name or Path(self.current_path).stem
        dataset_id, ok = QInputDialog.getText(
            self.view, "Adicionar à Biblioteca",
            "Identificador (sem extensão) — será usado como nome do "
            "arquivo na biblioteca:",
            text=default_id,
        )
        if not ok:
            return
        dataset_id = (dataset_id or "").strip()
        if not dataset_id:
            QMessageBox.warning(self.view, "Inválido",
                                "O identificador não pode ser vazio.")
            return

        # 4. Confirma overwrite
        target_path = Path(library_root) / "datasets" / f"{dataset_id}.parquet"
        if target_path.exists():
            res = QMessageBox.question(
                self.view, "Já existe",
                f"Já existe um dataset '{dataset_id}.parquet' na biblioteca. "
                "Sobrescrever?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                defaultButton=QMessageBox.StandardButton.No,
            )
            if res != QMessageBox.StandardButton.Yes:
                return

        # 5. Copia via GlobalLibraryStorage — preserva schema metadata
        # (PR 3) e calcula hash SHA-256 automaticamente.
        try:
            from faceaudit.storage.global_library_storage import GlobalLibraryStorage
            from faceaudit.enums.strategy_artifact_type import StrategyArtifactType

            storage = GlobalLibraryStorage(library_root=library_root)
            storage.create_library_structure()  # idempotente
            ref = storage.copy_artifact_to_project(
                source_path=self.current_path,
                artifact_type=StrategyArtifactType.GLOBAL_DATASET,
                dataset_id=dataset_id,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self.view, "Erro",
                f"Falha ao copiar pra biblioteca: {e}",
            )
            return

        QMessageBox.information(
            self.view, "Adicionado à biblioteca",
            f"Dataset salvo na biblioteca global como '{dataset_id}.parquet'.\n\n"
            f"Path: {target_path}\n"
            f"Hash: {ref.hash[:16]}…",
        )

    # ── PR 17: Enviar dataset pra Strategy Manager V2 ─────────────────────

    def fluxo_enviar_para_estrategia(self):
        """Garante dataset salvo, então emite `send_to_strategy_requested`
        com o path absoluto. O `MainGUIController` captura, troca de aba
        e delega o import pra V2 (criando/abrindo projeto se necessário).

        Pré-condições:
          - Dataset carregado e salvo em disco (current_path setado).
          - Edições não salvas → prompt pra salvar antes; aborta se cancelar.
        """
        if not self.dataset:
            return

        self.sync_changes_from_table()
        if self._is_dirty or not self.current_path or not Path(self.current_path).exists():
            res = QMessageBox.question(
                self.view, "Salvar antes",
                "O dataset precisa estar salvo em disco antes de ir pra V2. "
                "Salvar agora?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Cancel,
            )
            if res != QMessageBox.StandardButton.Save:
                return
            self.fluxo_salvar()
            if self._is_dirty or not self.current_path:
                return

        # Emite o signal pro MainGUIController fazer a ponte cross-tab
        abs_path = str(Path(self.current_path).resolve())
        self.send_to_strategy_requested.emit(abs_path)

    def request_close(self) -> bool:
        if not self.dataset:
            return True

        # Flush pending in-table edits para self.dataset antes de avaliar dirty —
        # caso contrário, edições só no QAbstractTableModel são perdidas silenciosamente.
        self.sync_changes_from_table()

        if self.is_dirty():
            res = QMessageBox.question(self.view, "Salvar Alterações?", 
                                       "O dataset possui alterações não salvas. Deseja salvar antes de fechar?", 
                                       QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            
            if res == QMessageBox.StandardButton.Cancel:
                return False
            if res == QMessageBox.StandardButton.Save:
                self.fluxo_salvar()
                if self.is_dirty(): return False

        # PR 25 (Tema 5.4) — registra CLOSE antes de zerar current_path.
        # Sem hash do arquivo (não muda no fechamento) — só timestamp/operator.
        self._audit(
            AuditOperation.CLOSE,
            f"Dataset '{self.dataset.name}' fechado.",
        )

        self.dataset = None
        self.current_path = None
        self.dataset_original = None
        self.table_ctrl.load_dataframe(None)
        self.detail_ctrl.load_face_data(None)
        self.toolbar_ctrl.set_dataset_loaded_state(False, "Nenhum dataset carregado.")
        self.set_dirty(False)
        return True

    def fluxo_fechar_dataset(self):
        self.request_close()

    def fluxo_novo_dataset(self):
        templates_path = Path(self.config.paths.dir_templates)
        dialog = NewDatasetController(self.view, templates_path=templates_path)
        
        if dialog.exec():
            ingestion_config: DatasetIngestionConfig = dialog.get_config()
            storage_path = Path(self.pipeline.storage_path)
            cache_file = storage_path / f"{ingestion_config.dataset_name}.parquet"
            
            if cache_file.exists():
                msg = f"O dataset '{ingestion_config.dataset_name}' já existe. Sobrescrever?"
                res = QMessageBox.question(self.view, "Confirmar", msg)
                if res == QMessageBox.StandardButton.No: return

            self._iniciar_worker_criacao(ingestion_config)

    def _iniciar_worker_criacao(self, config):
        self._run_worker(
            NewDatasetWorker(self.pipeline, config),
            label="Criando Dataset",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,
            on_finished=self._on_criacao_concluida,
        )

    def _on_criacao_concluida(self, dataset):
        if dataset and len(dataset) > 0:
            path_cache = self.pipeline.storage_path / f"{dataset.name}.parquet"
            self.carregar_dataset_na_tela(dataset, path=str(path_cache))
            # PR 25 (Tema 5.4) — registra CREATE como primeira entrada
            # da chain. hash_file=True dá um anchor inicial: o hash do
            # parquet recém-ingerido.
            self._audit(
                AuditOperation.CREATE,
                f"Dataset '{dataset.name}' criado a partir de ingestão.",
                metadata={
                    "rows": len(dataset),
                    "identities": dataset.number_of_identities,
                },
                hash_file=True,
            )
            QMessageBox.information(self.view, "Sucesso", "Dataset Criado!")
        else:
            QMessageBox.warning(self.view, "Aviso", "Processo finalizado sem dados.")

    def fluxo_stats(self): pass 

    def fluxo_estatisticas(self):
        if not self.dataset: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(50, self._abrir_estatisticas_real)

    def _abrir_estatisticas_real(self):
        try:
            self.sync_changes_from_table()
            stats_ctrl = StatsDialogController(self.dataset, parent_view=self.view)
            QApplication.restoreOverrideCursor()
            stats_ctrl.exec()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Falha ao gerar estatísticas: {e}")

    # =================================================================================
    # PR 24 (Tema 5.3 + 5.5) — Relatório de schema + Export CSV/Excel
    # =================================================================================
    def fluxo_schema_report(self):
        """Abre o SchemaReportDialog com a cobertura por coluna do
        dataset atual. O dialog inclui um botão de export CSV/Excel."""
        if not self.dataset:
            return
        # Sincroniza edições in-table antes — garante que custom columns
        # editadas ali já apareçam no relatório.
        self.sync_changes_from_table()
        from .schema_report_dialog import SchemaReportDialog
        # PR 25 (Tema 5.4) — passa callback de audit pro dialog. Quando
        # o usuário clica "Exportar para CSV/Excel" lá dentro e
        # confirma, o callback registra um evento EXPORT na chain.
        def _audit_export(meta: dict) -> None:
            self._audit(
                AuditOperation.EXPORT,
                f"Dataset exportado para {meta.get('format', 'arquivo')} "
                f"({meta.get('rows', '?')} linhas, "
                f"{meta.get('cols', '?')} colunas).",
                metadata=meta,
            )
        dialog = SchemaReportDialog(
            self.dataset, parent=self.view,
            on_export=_audit_export,
        )
        dialog.exec()

    def fluxo_audit_log(self):
        """Abre o AuditLogDialog com a chain de custódia do dataset
        atual. Requer dataset salvo em disco — log é per-arquivo."""
        if not self.dataset:
            return
        if not self.current_path:
            QMessageBox.information(
                self.view, "Audit log indisponível",
                "Salve o dataset primeiro — a cadeia de custódia é "
                "anexada ao arquivo parquet (sidecar .audit.json).",
            )
            return
        from .audit_log_dialog import AuditLogDialog
        dialog = AuditLogDialog(self.current_path, parent=self.view)
        dialog.exec()

    # =================================================================================
    # PR L1d — FLUXO DE LOCALIZAÇÃO (paths absolutos → relativos / cópia)
    # =================================================================================
    def fluxo_localizar(self):
        """Abre o LocalizeDatasetDialog. Após exec aceito, dispara o
        worker de localização e, ao final, substitui o dataset
        corrente + propõe Save As (consistência com Enrich/Filter)."""
        if not self.dataset:
            return
        self.sync_changes_from_table()

        # Sugere target_dir = <parquet_dir_atual>/<dataset_name>__localized/
        if self.current_path:
            base = Path(self.current_path).parent
        else:
            base = Path(self._get_initial_dir() or os.getcwd())
        suggested = base / f"{self.dataset.name}__localized"

        from .localize_dataset_dialog import LocalizeDatasetDialog
        dialog = LocalizeDatasetDialog(
            dataset_name=self.dataset.name,
            suggested_target_dir=suggested,
            parent=self.view,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        try:
            opts = dialog.get_options()
        except ValueError as e:
            QMessageBox.warning(self.view, "Configuração inválida", str(e))
            return

        self._iniciar_worker_localize(
            target_dir=opts["target_dir"],
            mode=opts["mode"],
            naming=opts["naming"],
            skip_existing=opts["skip_existing"],
        )

    def _iniciar_worker_localize(
        self, target_dir: Path, mode, naming, skip_existing: bool,
    ):
        # PR L3 — passa ff_service do pipeline pra habilitar COPY_ALIGNED.
        # Quando o modo é REWRITE_RELATIVE / COPY_ORIGINALS o
        # ff_service é ignorado pelo service. Pra COPY_ALIGNED, é
        # necessário (caso contrário service levanta ValueError).
        ff_svc = getattr(self.pipeline, "ff_service", None) if self.pipeline else None
        worker = LocalizeDatasetWorker(
            dataset=self.dataset,
            target_dir=target_dir,
            mode=mode,
            naming=naming,
            skip_existing=skip_existing,
            ff_service=ff_svc,
        )
        # Guarda contexto pra usar no on_finished
        self._localize_target_dir = target_dir
        self._localize_mode = mode
        self._localize_naming = naming
        self._run_worker(
            worker,
            label=f"Localizando dataset: {self.dataset.name}",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,  # feedback constante (decisão #3 do usuário)
            on_finished=self._on_localize_concluido,
        )

    def _on_localize_concluido(self, result):
        """Result é `LocalizationResult` ou None (cancelado/erro)."""
        if result is None:
            QMessageBox.warning(
                self.view, "Cancelado",
                "Localização cancelada ou finalizada sem resultado.",
            )
            return

        # Resumo na barra de status / popup
        if result.n_failed:
            failures_preview = "\n".join(
                f"  • {p}: {msg}" for p, msg in list(result.failures)[:5]
            )
            extra = (
                f"\n\n⚠ {result.n_failed} falha(s):\n{failures_preview}"
                + ("\n  …" if result.n_failed > 5 else "")
            )
        else:
            extra = ""

        # PR 25 (Tema 5.4) — registra no audit log do dataset original
        # ANTES de trocar o dataset corrente. Anchor da cadeia de
        # custódia: depois de localizar, o original perde 'rastreio
        # vivo' (continua existindo no disco mas o usuário fica com o
        # localizado). O audit no original preserva essa transição.
        try:
            self._audit(
                AuditOperation.COPY_IMAGES,
                (
                    f"Localização concluída: {result.summary()}. "
                    f"Destino: {result.target_dir}"
                ),
                metadata={
                    "mode": result.mode.value,
                    "naming": result.naming.value,
                    "target_dir": str(result.target_dir),
                    "n_processed": result.n_processed,
                    "n_copied": result.n_copied,
                    "n_skipped": result.n_skipped,
                    "n_failed": result.n_failed,
                    "duration_seconds": round(result.duration_seconds, 3),
                },
            )
        except Exception:
            pass  # audit é best-effort

        # Substitui o dataset corrente pelo localizado e força Save As
        # (paths e schema mudaram — current_path do original não vale).
        self.dataset = result.dataset
        self.dataset_original = None
        self.current_path = None
        self.set_dirty(True)
        self.carregar_dataset_na_tela(self.dataset, path=None)

        QMessageBox.information(
            self.view, "Localização concluída",
            f"Dataset localizado em memória.\n\n{result.summary()}{extra}\n\n"
            f"Próximo passo: salvar o dataset (sugerido em "
            f"{result.target_dir / (self.dataset.name + '.parquet')}).",
        )

        # Sugere Save As com path pré-preenchido pra dentro do target_dir
        suggested_save = result.target_dir / f"{self.dataset.name}.parquet"
        path, _ = QFileDialog.getSaveFileName(
            self.view, "Salvar dataset localizado",
            str(suggested_save), "Parquet (*.parquet)",
        )
        if path:
            self.dataset.name = Path(path).stem
            self.dataset.to_parquet(path)
            self.current_path = path
            self.set_dirty(False)
            self.carregar_dataset_na_tela(self.dataset, path)

    # =================================================================================
    # [NOVO] FLUXO DE ANÁLISE DO DATASET (AUDITORIA E MINERAÇÃO BIOMÉTRICA)
    # =================================================================================
    def fluxo_analise_dataset(self):
        if not self.dataset: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(10, self._abrir_analise_real)
        
    def _abrir_analise_real(self):
        try:
            # Salva qualquer edição de rótulo antes de rodar a auditoria
            self.sync_changes_from_table()

            # Pós-PR 11: o controller delega pro DatasetAuditService (cheap audits
            # automáticos no abrir) e o ComparisonService (hard mining sob demanda).
            comp_svc = self.container.comparison_service()
            audit_svc = self.container.dataset_audit_service()

            analysis_ctrl = DatasetAnalysisController(
                dataset=self.dataset,
                audit_service=audit_svc,
                comparison_service=comp_svc,
                parent_view=self.view,
            )
            # Quando o usuário usa as ações de sanitização (excluir, ...),
            # o dialog muta self.dataset em memória. Refrescamos a tabela
            # principal e marcamos dirty.
            analysis_ctrl.dataset_modified.connect(self._on_dataset_modified_by_audit)

            QApplication.restoreOverrideCursor()
            analysis_ctrl.exec()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Falha ao abrir Auditoria: {e}")

    def _on_dataset_modified_by_audit(self) -> None:
        """Hook chamado pelo DatasetAnalysisController quando faces foram
        deletadas/modificadas via o painel de auditoria."""
        if self.dataset is None:
            return
        self.table_ctrl.load_dataframe(self.dataset.to_pandas())
        self.set_dirty(True)

    def fluxo_comparar(self):
        if not self.dataset: return
        path, _ = QFileDialog.getOpenFileName(self.view, "Comparar com...", self._get_initial_dir(), "Parquet (*.parquet)")
        if path:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QTimer.singleShot(50, lambda: self._abrir_comparacao_real(path))

    def _abrir_comparacao_real(self, path):
        try:
            self.sync_changes_from_table()
            other = FaceDataset.from_parquet(path)
            QApplication.restoreOverrideCursor()
            
            # Buscamos os Singletons diretamente do Container
            stats_svc = self.container.stats_service()
            comp_svc = self.container.dataset_comparison_service()
            
            # Injetamos manualmente no construtor do Diálogo
            cmp_ctrl = CompareDialogController(
                dataset_a=self.dataset, 
                dataset_b=other, 
                stats_service=stats_svc, 
                comparison_service=comp_svc, 
                parent_view=self.view
            )
            cmp_ctrl.exec()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Falha na comparação: {e}")

    def fluxo_merge(self):
        if not self.dataset: return
        self.sync_changes_from_table()
        path, _ = QFileDialog.getOpenFileName(self.view, "Juntar com...", self._get_initial_dir(), "Parquet (*.parquet)")
        if path:
            other = FaceDataset.from_parquet(path)
            new_ds = self.dataset.concat(other)
            self.carregar_dataset_na_tela(new_ds)
            QMessageBox.information(self.view, "Merge", f"Datasets unidos. Total: {len(new_ds)} imagens.")
        self.set_dirty(True)

    def fluxo_split(self):
        if not self.dataset: return
        self.sync_changes_from_table()
        
        split_dialog = SplitDatasetDialogController(self.dataset, parent_view=self.view)
        
        if split_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            config = split_dialog.get_split_config()
            
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                if config["mode"] == "ratio":
                    ds_a, ds_b = self.dataset.split_by_identity(
                        split_ratio=config["ratio"]
                    )
                else:
                    ds_a, ds_b = self.dataset.split_by_identity_list(
                        ids_for_dataset_b=config["manual_ids_b"]
                    )
                
                ds_a.name = config["name_a"]
                ds_b.name = config["name_b"]
                
                path_a = self.pipeline.storage_path / f"{ds_a.name}.parquet"
                path_b = self.pipeline.storage_path / f"{ds_b.name}.parquet"
                
                ds_a.to_parquet(path_a)
                ds_b.to_parquet(path_b)
                
                QApplication.restoreOverrideCursor()
                
                QMessageBox.information(
                    self.view, 
                    "Sucesso", 
                    f"Divisão concluída com sucesso!\n\nSalvos em:\n- {path_a.name}\n- {path_b.name}"
                )

                action = config["post_action"]
                if action == 1:
                    self.carregar_dataset_na_tela(ds_a, path=str(path_a))
                elif action == 2:
                    self.carregar_dataset_na_tela(ds_b, path=str(path_b))
                
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self.view, "Erro", f"Erro durante a divisão do dataset:\n{e}")

    def fluxo_filtrar(self):
        if not self.dataset: return
        # PR 21 (Tema 4.2) — sincroniza edições in-table (incluindo
        # colunas custom no model._df) pra `dataset.faces[i].extra_data`
        # antes de abrir o dialog. Sem isso, custom columns adicionadas/
        # editadas via UI ficariam só no model e seriam invisíveis pra
        # `apply_filter`.
        self.sync_changes_from_table()
        if not self.dataset_original: self.dataset_original = self.dataset
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(10, self._abrir_dialogo_filtro_real)

    def _abrir_dialogo_filtro_real(self):
        try:
            filter_ctrl = FilterDialogController(
                self.dataset_original, 
                self.current_filter_config, 
                parent=self.view
            )
            QApplication.restoreOverrideCursor()
            if filter_ctrl.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                QApplication.restoreOverrideCursor()
                self.current_filter_config = filter_ctrl.get_config()
                self._iniciar_worker_filtro()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Erro ao abrir filtros: {e}")

    def _iniciar_worker_filtro(self):
        self._run_worker(
            FilterWorker(self.dataset_original, self.current_filter_config),
            label="Aplicando Filtros...",
            allow_cancel=True,
            determinate=True,
            wait_cursor=True,
            on_finished=self._on_filtro_concluido,
        )

    def _on_filtro_concluido(self, filtered_ds):
        if filtered_ds is None:
            # Cancelamento cooperativo do usuário.
            return
        try:
            if len(filtered_ds) > 0:
                self.dataset = filtered_ds
                self.table_ctrl.load_dataframe(self.dataset.to_pandas())
                qtd_imgs = len(self.dataset)
                qtd_ids = self.dataset.number_of_identities
                nome = self.dataset.name if self.dataset.name else "Sem Nome"
                status_txt = f"📂 <b>{nome}</b> | 🖼️ {qtd_imgs} Imagens | 👤 {qtd_ids} IDs (Filtrado)"
                self.toolbar_ctrl.update_filter_status(status_txt, is_filtered=True)
            else:
                QMessageBox.warning(self.view, "Filtro", "Nenhum resultado encontrado com estes critérios.")
        except Exception as e:
            QMessageBox.critical(self.view, "Erro", f"Erro ao aplicar filtro: {e}")

    def fluxo_limpar_filtro(self):
        if self.dataset_original:
            self.dataset = self.dataset_original
            self.dataset_original = None
            self.current_filter_config = FilterConfig() 
            self.carregar_dataset_na_tela(self.dataset, self.current_path)
            
            qtd_imgs = len(self.dataset)
            qtd_ids = self.dataset.number_of_identities
            nome = self.dataset.name
            status_txt = f"📂 <b>{nome}</b> | 🖼️ {qtd_imgs} Imagens | 👤 {qtd_ids} IDs"
            self.toolbar_ctrl.update_filter_status(status_txt, is_filtered=False)

    def fluxo_enriquecer(self):
        if not self.dataset: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(10, self._abrir_enrich_real)

    def _abrir_enrich_real(self):
        try:
            self.sync_changes_from_table()
            
            enrich_ctrl = EnrichDatasetController(self.dataset, parent=self.view)
            
            QApplication.restoreOverrideCursor()
            
            if enrich_ctrl.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                config = enrich_ctrl.get_config()
                self._iniciar_worker_enrich(config)
                
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Erro ao abrir janela de enriquecimento: {e}")

    def _iniciar_worker_enrich(self, config: DatasetIngestionConfig):
        self._run_worker(
            EnrichWorker(self.pipeline, self.dataset, config),
            label="Enriquecendo Dataset",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,
            on_finished=self._on_enrich_concluido,
        )

    def _on_enrich_concluido(self, dataset):
        if dataset:
            self.dataset = dataset
            self.table_ctrl.load_dataframe(self.dataset.to_pandas())
            # PR 25 (Tema 5.4) — registra ENRICH (sem hash; o parquet
            # ainda não foi resalvo, o save subsequente fecha o ciclo).
            self._audit(
                AuditOperation.ENRICH,
                f"Enriquecimento aplicado em '{dataset.name}'.",
                metadata={
                    "rows": len(self.dataset),
                    "identities": self.dataset.number_of_identities,
                },
            )
            QMessageBox.information(self.view, "Sucesso", "Enriquecimento concluído com sucesso!")
            self.set_dirty(True)
        else:
            QMessageBox.warning(self.view, "Cancelado", "Operação cancelada ou finalizada sem dados.")