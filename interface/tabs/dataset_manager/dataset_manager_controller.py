import os
from pathlib import Path
from typing import Optional
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
from .workers.modify_paths_worker import ModifyPathsWorker  # PR L1d
from .workers.filter_worker import FilterWorker
from .workers.new_dataset_worker import NewDatasetWorker
from .workers.enrich_worker import EnrichWorker
from .workers.worker_progress_runner import WorkerProgressRunner

# --- MODELOS ---
from faceaudit.models import FaceDataset
from faceaudit.models.filter import FilterConfig
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig

# PR 25 (Tema 5.4) — audit log forense / chain of custody
from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.path_mode import PathMode
from faceaudit.services.image_path_resolver import (
    normalize_stored_path,
    resolve_image_path,
)

# Feedback visual em operações síncronas pesadas
from interface.wait_cursor import wait_cursor
from faceaudit.services.audit_log_service import AuditLogService
from faceaudit.services.hash_service import HashService


# Quantos identificadores de face a entrada de auditoria lista antes de
# truncar. O hash do conjunto completo continua gravado, então a prova
# de pertinência não se perde com o truncamento.
from faceaudit.services import dataset_custody_service as custody
from faceaudit.services.audit_log_service import (
    JustificationRequiredError,
    require_justification,
)

_MAX_AUDIT_IDENTIFIERS = custody.MAX_AUDIT_IDENTIFIERS

# Operações que não descrevem o CONTEÚDO do dataset: sem arquivo, não há o
# que guardar para depois (fechar um dataset em memória não deixa história).
_NAO_ADIAR_SEM_ARQUIVO = frozenset({
    AuditOperation.OPEN, AuditOperation.SAVE, AuditOperation.SAVE_AS,
    AuditOperation.CLOSE,
})


def _sem_infinito(valor):
    """Troca ±inf por "inf"/"-inf" (texto) numa estrutura JSON.

    Intervalo aberto de filtro numérico é guardado como `float('inf')`, e o
    `json.dumps` o escreve como `Infinity` — que não é JSON válido e quebra
    a leitura da cadeia de custódia por qualquer ferramenta fora do Python.
    """
    if isinstance(valor, float) and valor in (float("inf"), float("-inf")):
        return "inf" if valor > 0 else "-inf"
    if isinstance(valor, dict):
        return {k: _sem_infinito(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_sem_infinito(v) for v in valor]
    return valor


class DatasetManagerController(QtCore.QObject):
    """
    Controlador Mestre do Gerenciamento de Datasets.
    """
    
    dataset_changed = QtCore.pyqtSignal()
    # Save concluído sobrescrevendo o path atual (fluxo_salvar). Path é
    # absoluto. Subscriber natural: Strategy Manager V2 — quando o path
    # casa com um primary do projeto aberto, atualiza hash + cascade.
    dataset_saved_to_project = QtCore.pyqtSignal(str)
    # PR 17: emitido quando o usuário pede pra enviar o dataset corrente
    # pra V2. Carrega o caminho absoluto do parquet salvo. Capturado pelo
    # MainGUIController, que troca de aba e chama a V2.
    send_to_strategy_requested = QtCore.pyqtSignal(str)

    # RECEBE O CONTAINER AQUI
    def __init__(
        self,
        view,
        pipeline,
        config,
        container,
        ff_available: bool = True,
        ofiq_available: bool = True,
    ):
        super().__init__()
        self.view = view
        self.pipeline = pipeline
        self.config = config
        self.container = container # Guardamos a referência do container para injeção explícita
        # Flags de disponibilidade dos serviços externos. Propagadas aos
        # diálogos Novo/Enriquecer Dataset pra desabilitar checkboxes
        # quando o respectivo serviço não pôde ser inicializado (ex:
        # modelo ForensicFace não instalado, ou OFIQ exe não encontrado).
        # O resto da aba (carregar, filtrar, comparar, balancear, pipeline
        # de imagens, exportar) funciona normalmente sem esses serviços.
        self.ff_available = bool(ff_available)
        self.ofiq_available = bool(ofiq_available)

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
        self.toolbar_ctrl.action_new_from_video.connect(self.fluxo_novo_dataset_video)
        self.toolbar_ctrl.action_import_video_run.connect(self.fluxo_importar_run_video)
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
        # PR L1d — Modificação de caminhos das imagens
        if hasattr(self.toolbar_ctrl, "action_modify_paths"):
            self.toolbar_ctrl.action_modify_paths.connect(self.fluxo_modificar_caminhos)
        # PR I2 — Editor de pipeline de imagens
        if hasattr(self.toolbar_ctrl, "action_image_pipeline_editor"):
            self.toolbar_ctrl.action_image_pipeline_editor.connect(
                self.fluxo_image_pipeline_editor
            )
        # PR I5 — Aplicar pipeline em batch
        if hasattr(self.toolbar_ctrl, "action_image_pipeline_apply"):
            self.toolbar_ctrl.action_image_pipeline_apply.connect(
                self.fluxo_aplicar_image_pipeline
            )

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
        if hasattr(self.toolbar_ctrl, "action_metadata"):
            self.toolbar_ctrl.action_metadata.connect(self.fluxo_editar_metadados)
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

        # Ações de seleção do menu de contexto da galeria (N faces).
        self.gallery_widget.remove_faces_requested.connect(self._on_gallery_remove_faces)
        self.gallery_widget.create_dataset_from_selection_requested.connect(
            self._on_gallery_create_dataset_from_selection)
        self.gallery_widget.set_column_value_requested.connect(
            self._on_gallery_set_column_value)

        # As MESMAS ações pela tabela (que emite caminhos, não índices) e
        # pela toolbar (que lê a seleção da visão ativa). Um handler só por
        # ação — implementações paralelas divergem, e a que divergir vai
        # esquecer a cadeia de custódia.
        if hasattr(self.table_ctrl, "remove_faces_requested"):
            self.table_ctrl.remove_faces_requested.connect(self.excluir_selecionados)
        if hasattr(self.table_ctrl, "save_selection_requested"):
            self.table_ctrl.save_selection_requested.connect(
                lambda _paths: self.fluxo_salvar_selecionados())
        self.toolbar_ctrl.action_remove_selected.connect(
            lambda: self.excluir_selecionados())
        self.toolbar_ctrl.action_save_selected.connect(
            self.fluxo_salvar_selecionados)
        self.toolbar_ctrl.action_extract_face_images.connect(
            self.fluxo_gerar_imagens_de_face)

        # PR 16 — Adicionar à Biblioteca Global
        if hasattr(self.toolbar_ctrl, "action_save_to_library"):
            self.toolbar_ctrl.action_save_to_library.connect(self.fluxo_salvar_biblioteca)

        # PR 17 — Enviar pra Estratégia (V2)
        if hasattr(self.toolbar_ctrl, "action_send_to_strategy"):
            self.toolbar_ctrl.action_send_to_strategy.connect(self.fluxo_enviar_para_estrategia)

        if hasattr(self.toolbar_ctrl, 'action_enrich'):
            self.toolbar_ctrl.action_enrich.connect(self.fluxo_enriquecer)

        if hasattr(self.toolbar_ctrl, 'action_balance'):
            self.toolbar_ctrl.action_balance.connect(self.fluxo_balancear)

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
            dataset=self.dataset,
        )
        cmp_ctrl.exec()

    # ── Ações de seleção do menu de contexto da galeria ───────────────
    #
    # As ações recebem face_idx (índices em dataset.faces na ordem em que a
    # galeria foi populada). Ordenar a TABELA reordena o dataframe — e o
    # `sync_changes_from_table` reconstrói dataset.faces a partir dele —, então
    # índices posicionais podem deixar de bater após um sync. Por isso cada
    # handler captura os `image_path` ALVO ANTES de sincronizar e, após o sync,
    # recupera as faces por caminho (mesma chave que o próprio sync usa).
    #
    # IMPORTANTE — NÃO-DESTRUTIVO: "Remover" só tira a face do dataset (em
    # memória, efetivado ao Salvar). NENHUM arquivo de imagem é apagado do
    # disco, em nenhum modo. `path_mode==RELATIVE` NÃO garante que as imagens
    # sejam cópias do dataset (REWRITE_RELATIVE referencia os ORIGINAIS; o
    # parquet_dir pode ser None → cwd), então apagar arquivo arriscaria destruir
    # evidência original. Cópias órfãs num autocontido apenas permanecem em disco.

    def _gallery_target_paths(self, idxs) -> list:
        """`image_path` das faces selecionadas, em ordem crescente de índice de
        dataset.faces (a seleção já vem ordenada da galeria). Capturar ANTES de
        qualquer sync — depois a ordem de dataset.faces pode mudar."""
        if not self.dataset:
            return []
        faces = self.dataset.faces
        n = len(faces)
        out = []
        for i in sorted(set(idxs)):
            if isinstance(i, int) and 0 <= i < n:
                p = getattr(faces[i], "image_path", None)
                if p:
                    out.append(p)
        return out

    def _current_selection_paths(self) -> list:
        """`image_path` do que está selecionado na visão ATIVA.

        Tabela e galeria têm espaços de índice diferentes (a tabela pode
        estar ordenada), então a moeda comum entre elas é o caminho da
        imagem — que é também a chave usada para casar de volta com
        `dataset.faces`. Sem este ponto único, cada botão precisaria saber
        em qual visão o usuário está, e isso vazaria para os handlers.
        """
        if self._view_stack.currentIndex() == 1:
            return self._gallery_target_paths(self.gallery_widget.selected_indexes())
        return self.table_ctrl.selected_image_paths()

    def _faces_from_paths(self, paths, action_label: str):
        """Faces de `dataset.faces` correspondentes a `paths`, ou None.

        Mesma proteção do caminho da galeria: se um `image_path` casar com
        MAIS faces do que as selecionadas (imagens de conteúdo idêntico
        apontando para o mesmo arquivo), aborta em vez de afetar faces que
        o usuário não escolheu.
        """
        if not paths:
            QMessageBox.information(
                self.view, action_label,
                "Selecione ao menos um registro.",
            )
            return None
        n_selected = len(paths)
        self.sync_changes_from_table()          # não perder edições da tabela
        path_set = set(paths)
        matched = [f for f in self.dataset.faces
                   if getattr(f, "image_path", None) in path_set]
        if len(matched) > n_selected:
            QMessageBox.warning(
                self.view, action_label,
                "Algumas faces selecionadas compartilham o caminho de imagem com "
                "faces NÃO selecionadas (imagens de conteúdo idêntico podem "
                "apontar para o mesmo arquivo). Para não afetar faces que você "
                "não escolheu, a operação foi cancelada.")
            return None
        return matched

    def _gallery_match_after_sync(self, idxs, action_label: str):
        """Captura os caminhos da seleção, sincroniza a tabela e devolve as
        faces correspondentes em self.dataset.faces. Aborta (devolve None) se um
        caminho casar com MAIS faces do que as selecionadas — sinal de
        `image_path` duplicado (raro; ex.: localização HASH com conteúdo
        idêntico). Operar por caminho aí afetaria faces NÃO selecionadas, então
        paramos por segurança em vez de adivinhar."""
        paths = self._gallery_target_paths(idxs)
        if not paths:
            return None
        n_selected = len(paths)
        self.sync_changes_from_table()              # não perder edições da tabela
        path_set = set(paths)
        matched = [f for f in self.dataset.faces
                   if getattr(f, "image_path", None) in path_set]
        if len(matched) > n_selected:
            QMessageBox.warning(
                self.view, action_label,
                "Algumas faces selecionadas compartilham o caminho de imagem com "
                "faces NÃO selecionadas (imagens de conteúdo idêntico podem "
                "apontar para o mesmo arquivo). Para não afetar faces que você "
                "não escolheu, a operação foi cancelada.")
            return None
        return matched

    def _refresh_after_structural_change(self) -> None:
        """Reconstrói tabela + galeria a partir de self.dataset (fonte de
        verdade) e marca modificado. Reseta a ordenação da tabela para a ordem
        do dataset (efeito colateral aceitável de uma mutação estrutural)."""
        self.table_ctrl.load_dataframe(self.dataset.to_pandas(include_embeddings=False))
        self.gallery_widget.populate(list(self.dataset.faces))
        self.detail_ctrl.load_face_data(None)
        self.set_dirty(True)

    def _on_gallery_remove_faces(self, idxs) -> None:
        """Entrada vinda do menu de contexto da galeria (índices da galeria)."""
        self.excluir_selecionados(self._gallery_target_paths(idxs))

    def excluir_selecionados(self, paths=None) -> None:
        """Exclui do dataset os registros selecionados.

        `paths` vem do menu de contexto que disparou a ação; sem ele, lê a
        seleção da visão ativa (é o caminho do botão da toolbar).
        """
        if not self.dataset:
            return
        rotulo = "Excluir registros"
        matched = self._faces_from_paths(
            paths if paths is not None else self._current_selection_paths(), rotulo,
        )
        if not matched:
            return

        from .remove_faces_dialog import RemoveFacesDialog

        dlg = RemoveFacesDialog(
            self.view, n_faces=len(matched), nome_dataset=self.dataset.name or "",
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        # O diálogo já não confirma sem texto; a regra, porém, é do backend
        # e é conferida ANTES de mexer no dataset — exclusão aplicada sem
        # registro possível é o pior desfecho.
        try:
            justificativa = require_justification(dlg.justificativa())
        except JustificationRequiredError as e:
            QMessageBox.warning(self.view, rotulo, str(e))
            return

        n = len(matched)
        matched_ids = {id(f) for f in matched}      # remove EXATAMENTE as faces casadas
        self.dataset.faces = [f for f in self.dataset.faces if id(f) not in matched_ids]
        # Com filtro ativo, o recorte na tela e o dataset de origem
        # compartilham os MESMOS objetos FaceInfo. Remover só da lista do
        # recorte deixava as faces vivas no original — e "Limpar filtro"
        # as trazia de volta, desfazendo a exclusão sem avisar.
        if self.dataset_original is not None and self.dataset_original is not self.dataset:
            self.dataset_original.faces = [
                f for f in self.dataset_original.faces if id(f) not in matched_ids
            ]

        self._audit_remocao_de_faces(matched, justificativa)
        self._refresh_after_structural_change()
        QMessageBox.information(
            self.view, "Registros excluídos",
            f"{n} registro(s) excluído(s) do dataset. Salve para efetivar.")

    def _audit_remocao_de_faces(self, faces, justificativa: str) -> None:
        """Registra a exclusão na cadeia de custódia.

        Grava no ATO da exclusão — é o momento e o autor da decisão. Como
        a remoção só vai para o arquivo no save seguinte, a entrada nasce
        marcada como pendente; o save posterior grava o hash do arquivo
        resultante e fecha o elo.

        Identificar o que saiu importa, mas listar 5 mil linhas não serve a
        ninguém: guarda os primeiros hashes e um hash do conjunto inteiro,
        que permite provar depois se um registro específico estava ou não
        entre os excluídos. O conteúdo da entrada é montado no backend
        (`dataset_custody_service.removal_entry`), que recusa justificativa
        vazia.
        """
        try:
            descricao, meta = custody.removal_entry(faces, justificativa)
            self._audit(AuditOperation.REMOVE_FACES, descricao, metadata=meta)
        except Exception:
            pass  # audit é best-effort — nunca impede a operação do perito

    def _on_gallery_create_dataset_from_selection(self, idxs) -> None:
        import copy as _copy
        from PyQt6.QtWidgets import QInputDialog
        matched = self._gallery_match_after_sync(idxs, "Criar dataset a partir da seleção")
        if not matched:
            return
        suggested = f"{self.dataset.name}_sel"
        name, ok = QInputDialog.getText(
            self.view, "Criar dataset a partir da seleção",
            f"Nome do novo dataset ({len(matched)} face(s)):", text=suggested)
        if not ok or not name.strip():
            return
        name = name.strip()
        if self.is_dirty():
            res = QMessageBox.question(
                self.view, "Dataset modificado",
                "O dataset atual tem alterações não salvas; criar o subset "
                "substitui o que está na tela.\nContinuar?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if res != QMessageBox.StandardButton.Yes:
                return
        subset = FaceDataset(
            name=name, description=getattr(self.dataset, "description", "") or "",
            faces=[_copy.copy(f) for f in matched],
            column_specs=dict(self.dataset.column_specs) if self.dataset.column_specs else {},
            path_mode=self.dataset.path_mode, path_root=self.dataset.path_root,
            is_aligned=self.dataset.is_aligned,
        )
        subset._parquet_dir = self.dataset._parquet_dir
        subset.hash = None
        # O subset SUBSTITUI o dataset na tela — zera o estado de filtro do
        # anterior, senão "Limpar filtro" restauraria o dataset antigo por cima
        # do subset (dataset_original ficaria obsoleto).
        self.dataset_original = None
        self.current_filter_config = FilterConfig()
        self.carregar_dataset_na_tela(subset, path=None)
        self.set_dirty(True)
        QMessageBox.information(
            self.view, "Subset criado",
            f"Dataset '{name}' com {len(subset.faces)} face(s) carregado na tela.\n"
            "Use Salvar Como para gravá-lo num arquivo.")

    def _on_gallery_set_column_value(self, idxs) -> None:
        matched = self._gallery_match_after_sync(idxs, "Definir valor de coluna")
        if not matched:
            return
        # Alinha a ordem da tabela à de dataset.faces (df row == faces index).
        self.table_ctrl.load_dataframe(self.dataset.to_pandas(include_embeddings=False))
        model = self.table_ctrl.model
        # Colunas EDITÁVEIS (label, key, vocabulary, dtype) para o diálogo —
        # dtype faz o campo de valor se adaptar (bool→combo, numérico→validação).
        cols = []
        for c in model.visible_columns:
            if getattr(c, "editable", False):
                spec = (self.dataset.column_specs or {}).get(getattr(c, "key", None))
                vocab = getattr(spec, "vocabulary", None) if spec else None
                dtype = getattr(spec, "dtype", None) if spec else None
                dtype_str = getattr(dtype, "value", None) or (dtype if isinstance(dtype, str) else None)
                cols.append((getattr(c, "label", None) or c.key, c.key, vocab, dtype_str))
        if not cols:
            QMessageBox.information(
                self.view, "Definir valor de coluna",
                "Este dataset não tem colunas editáveis.")
            return
        matched_ids = {id(f) for f in matched}
        rows = [r for r, f in enumerate(self.dataset.faces) if id(f) in matched_ids]
        if not rows:
            return
        from .set_column_value_dialog import SetColumnValueDialog
        dlg = SetColumnValueDialog(cols, len(rows), parent=self.view)
        if not dlg.exec():
            return
        result = dlg.get_result()
        if not result:
            return
        key, value = result
        n_ok = self._apply_column_value_to_rows(rows, key, value)
        if n_ok == 0:
            QMessageBox.information(
                self.view, "Definir valor de coluna",
                f"Nenhuma face alterada — '{value}' pode estar fora do "
                "vocabulário da coluna.")
            return
        # Propaga df→dataset e repinta a galeria (caption pode mudar — ex. identidade).
        self.sync_changes_from_table()
        self.gallery_widget.populate(list(self.dataset.faces))
        self.set_dirty(True)
        msg = f"'{key}' = '{value}' aplicado a {n_ok} face(s)."
        if n_ok < len(rows):
            msg += (f"\n{len(rows) - n_ok} rejeitada(s) — valor fora do "
                    "vocabulário da coluna?")
        QMessageBox.information(self.view, "Valor definido", msg)

    def _apply_column_value_to_rows(self, rows, key, value) -> int:
        """Seta `value` na coluna `key` para as `rows` (índices do model, já
        alinhados a dataset.faces) via model.setData — que valida vocabulário.
        Retorna quantas foram aceitas."""
        model = self.table_ctrl.model
        col_idx = next((i for i, c in enumerate(model.visible_columns)
                        if getattr(c, "key", None) == key), None)
        if col_idx is None:
            return 0
        model.layoutAboutToBeChanged.emit()
        n_ok = 0
        for r in rows:
            if 0 <= r < model.rowCount():
                if model.setData(model.index(r, col_idx), value, Qt.ItemDataRole.EditRole):
                    n_ok += 1
        model.layoutChanged.emit()
        return n_ok

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

    def fluxo_editar_metadados(self):
        """Abre o DatasetMetadataEditorDialog (visão global).

        Aplica em sucesso:
        - Campos do dataset (name, description, is_aligned).
        - column_specs alterados → `update_column_spec(name, new_spec)` por coluna.

        Em seguida: reinstala vocabulary delegates + emite schema_changed
        (sincroniza specs do model com o dataset + marca dirty).
        """
        if self.dataset is None:
            return
        from PyQt6.QtWidgets import QMessageBox

        if not self.table_ctrl.model.column_specs:
            QMessageBox.warning(
                self.view, "Schema legado",
                "Este dataset foi carregado de um parquet sem column_specs "
                "na metadata. Salve-o primeiro (Salvar) para atualizar para "
                "o novo formato — depois você pode editar metadados.",
            )
            return

        from .column_dialogs.dataset_metadata_editor_dialog import (
            DatasetMetadataEditorDialog,
        )
        from .table.vocabulary_combo_delegate import install_vocabulary_delegates

        dialog = DatasetMetadataEditorDialog(self.dataset, parent=self.view)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        ds_changes, col_changes = dialog.get_result()

        # Aplica dataset-level
        for field_name, new_value in ds_changes.items():
            setattr(self.dataset, field_name, new_value)

        # Aplica por-coluna via model (mantém modelo em sync e reset adequado)
        applied: List[str] = []
        for name, new_spec in col_changes.items():
            if self.table_ctrl.model.update_column_spec(name, new_spec):
                applied.append(name)

        # Re-instala delegates (vocabularies podem ter mudado), refresca
        # specs no dataset e marca dirty.
        install_vocabulary_delegates(self.table_ctrl.view, self.table_ctrl.model)
        self.table_ctrl.view.resizeColumnsToContents()
        self._on_schema_changed()

        # PR 25 (Tema 5.4) — registra a edição se houve mudanças.
        if ds_changes or applied:
            summary_parts = []
            if ds_changes:
                summary_parts.append(
                    "dataset: " + ", ".join(sorted(ds_changes.keys()))
                )
            if applied:
                summary_parts.append(f"colunas: {', '.join(sorted(applied))}")
            self._audit(
                AuditOperation.EDIT_SCHEMA,
                "Metadados editados — " + "; ".join(summary_parts),
                metadata={
                    "dataset_fields": sorted(ds_changes.keys()),
                    "columns": sorted(applied),
                },
            )

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
        # Pasta sugerida por uma operação anterior (ex: as imagens acabaram
        # de ser copiadas para a pasta própria do dataset — é lá que o
        # parquet precisa ser gravado). Consumida uma vez.
        sugerida = getattr(self, "_suggested_save_dir", None)
        if sugerida and os.path.isdir(sugerida):
            self._suggested_save_dir = None
            return sugerida
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
            # Dataset só em memória (resultado de Juntar, subset da galeria,
            # vídeo sem parquet): ainda não há sidecar. A entrada fica
            # pendente no dataset e é escrita quando ele for gravado — antes
            # ela simplesmente se perdia. Fica no dataset RAIZ: com filtro
            # ativo, o recorte na tela é descartável.
            if operation in _NAO_ADIAR_SEM_ARQUIVO:
                return
            raiz = getattr(self, "dataset_original", None) or getattr(self, "dataset", None)
            if raiz is None:
                return
            try:
                custody.queue_pending(raiz, operation, description, metadata)
            except Exception as e:
                import sys as _sys
                print(f"[AuditLog] Falha ao adiar {operation.value}: {e}",
                      file=_sys.stderr)
            return
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

    def _escrever_pendencias_de_custodia(self, path, gravado, *origens) -> None:
        """Escreve na cadeia de `path` as entradas adiadas de `gravado` e das
        `origens` (de onde o gravado deriva) e esvazia as do `gravado`, que
        agora tem arquivo. As das origens ficam: uma origem que continua
        aberta sem arquivo ainda as escreverá quando for gravada."""
        try:
            itens = custody.pending_entries(gravado, *origens)
            if itens:
                custody.write_pending(path, itens)
            custody.clear_pending(gravado)
        except Exception as e:
            import sys as _sys
            print(f"[AuditLog] Falha ao escrever entradas adiadas: {e}",
                  file=_sys.stderr)

    def carregar_dataset_na_tela(self, dataset, path=None, display_df=None):
        self.dataset = dataset
        self.current_path = path
        # Propaga referência pra todos os widgets que precisam resolver
        # paths em modo RELATIVE (sem isso, datasets do Image Pipeline
        # mostram "Imagem não encontrada" / thumbnails vazias).
        self.detail_ctrl.set_dataset(dataset)
        if hasattr(self.gallery_widget, "set_dataset"):
            self.gallery_widget.set_dataset(dataset)
        if hasattr(self.table_ctrl, "set_dataset"):
            self.table_ctrl.set_dataset(dataset)

        # Schema-by-role (PR 6): se o FaceDataset trouxe column_specs (parquet
        # pós-PR 3), o table_ctrl propaga pra model + instala delegates de
        # vocabulário. Caso contrário (parquet legado ou dataset criado em
        # memória), o model cai no enum estático.
        column_specs = getattr(self.dataset, "column_specs", None) or {}
        if hasattr(self.table_ctrl, "apply_column_specs"):
            self.table_ctrl.apply_column_specs(column_specs)
        elif hasattr(self.table_ctrl, "model") and hasattr(self.table_ctrl.model, "set_column_specs"):
            self.table_ctrl.model.set_column_specs(column_specs)

        # df de EXIBIÇÃO é metadata-only (sem embeddings). Quando o worker de
        # abertura já o construiu fora da GUI, reusa; senão (refresh de
        # datasets em memória — filtro/merge/split) constrói on-demand.
        df_exibicao = (
            display_df if display_df is not None
            else self.dataset.to_pandas(include_embeddings=False)
        )
        self.table_ctrl.load_dataframe(df_exibicao)

        # Reseta o painel direito — sem isso, ao trocar de dataset
        # (ou limpar filtro) a face selecionada antes continua
        # exibida, mesmo podendo não estar mais no dataset corrente.
        self.detail_ctrl.load_face_data(None)

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

        self._refresh_dataset_status()

    def _is_filter_active(self) -> bool:
        """Há filtro ativo? O que está na tela é recorte de um dataset maior."""
        return (
            self.dataset_original is not None
            and self.dataset_original is not self.dataset
        )

    def _refresh_dataset_status(self) -> None:
        """Reescreve a status bar a partir do estado REAL de filtro.

        Antes o estado era fixado em `False` aqui e em `set_dirty`, então
        qualquer recarga de tela com filtro ativo (salvar, editar célula)
        apagava o indicador e DESABILITAVA o "Limpar filtro" — deixando o
        recorte na tela sem como voltar ao dataset de origem.
        """
        if not self.dataset:
            return
        filtrado = self._is_filter_active()
        text, tooltip = self._build_dataset_status(filtered=filtrado)
        self.toolbar_ctrl.set_dataset_loaded_state(
            True, text, tooltip, is_filtered=filtrado,
        )

    def sync_changes_from_table(self):
        if not self.dataset: return
        # Sem edições pendentes, self.dataset já reflete a tabela — reconstruir
        # via from_dataframe (iterrows sobre N faces, com laço de colunas
        # aninhado) seria O(N) puro desperdício. TODA edição de célula/coluna
        # seta dirty (set_dirty(True)); se não-dirty, nada mudou desde o load.
        # Isso barateia dramaticamente save / filtro / troca de view / abrir
        # diálogos em datasets grandes quando o usuário só está navegando.
        if not self._is_dirty:
            return
        df_edited = self.table_ctrl.get_current_dataframe()
        if df_edited is None or df_edited.empty:
            return
        if "image_path" not in df_edited.columns:
            # image_path é a chave de junção (coluna essencial, sempre presente).
            # Sem ela não há como casar linhas com faces — aborta em vez de
            # arriscar um sync incorreto.
            return

        # APLICAÇÃO IN-PLACE (substitui o from_dataframe/iterrows, que custava
        # ~35s em 40k). Só metadados editáveis + colunas custom podem ter mudado
        # na tabela; aplicamos esses valores DIRETO nas faces existentes,
        # casadas por image_path, com extração vetorizada das colunas. O
        # enriquecimento pesado (FF/OFIQ/SER-FIQ/embeddings) permanece intacto
        # nas faces — não é editável e nunca é tocado. `column_specs` é mantido
        # pelo `_on_schema_changed`; path_mode/path_root/_parquet_dir/is_aligned
        # já estão corretos porque o dataset NÃO é recriado.
        self._apply_table_edits_inplace(df_edited)

        # Propaga pro backup. Após um filtro, self.dataset (subset) e
        # dataset_original (full) compartilham as MESMAS FaceInfo, então a
        # mutação in-place já se reflete lá; a substituição por image_path
        # abaixo é defensiva para o caso (raro) de objetos separados.
        if self.dataset_original is not None and self.dataset_original is not self.dataset:
            by_path = {f.image_path: f for f in self.dataset.faces}
            for i, face_orig in enumerate(self.dataset_original.faces):
                repl = by_path.get(face_orig.image_path)
                if repl is not None:
                    self.dataset_original.faces[i] = repl
        # NOTE: dirty é marcado por _on_table_data_changed quando o usuário edita
        # uma célula. Sync apenas propaga; não reseta dirty (o "*" de não-salvo
        # permanece até o save). Como o apply-in-place é barato (<1s), re-syncs
        # redundantes enquanto dirty=True não pesam.

    def _apply_table_edits_inplace(self, df_edited) -> None:
        """Aplica edições da tabela (metadados de FaceInfo + colunas custom)
        direto nas faces existentes, casadas por image_path. Extração
        VETORIZADA por coluna (`df[col].tolist()`) + laço enxuto — evita o
        `from_dataframe` (iterrows sobre N faces, com laço de colunas aninhado)
        que custava ~35s em 40k. NÃO toca no enriquecimento pesado
        (FF/OFIQ/SER-FIQ/embeddings), que não é editável e permanece nas faces.
        Replica a semântica de FaceInfo de `from_dataframe` (mesma normalização
        de erro / `clean` / extra_data)."""
        import pandas as pd
        import numpy as np

        faces_by_path = {f.image_path: f for f in self.dataset.faces}

        cols = list(df_edited.columns)
        col_lists = {c: df_edited[c].tolist() for c in cols}
        raw_paths = col_lists.get("image_path", [])
        # Normaliza como FaceInfo.__post_init__ (\\ -> /) pra casar as chaves.
        paths = [str(p).replace("\\", "/") if p is not None else "" for p in raw_paths]

        # Metadados de FaceInfo aplicáveis presentes no df (colunas pesadas —
        # embeddings/ff/ofiq/serfiq — não entram no df metadata-only e/ou não
        # são editáveis, então são ignoradas e preservadas nas sub-estruturas).
        attr_cols = [
            c for c in ("file_name", "identity", "ethnicity", "gender",
                        "hash", "origin", "error_stage", "error_message")
            if c in cols
        ]
        source_col = "source" if ("source" in cols and "origin" not in cols) else None
        has_error = "error" in cols
        # Custom columns → extra_data (não-canônicas; array/NaN excluídos, igual from_dataframe)
        from faceaudit.models.face_dataset import _is_canonical_face_column
        custom_cols = [c for c in cols if not _is_canonical_face_column(c)]

        def _clean(v):
            try:
                if pd.isna(v):
                    return None
            except (TypeError, ValueError):
                pass
            if v == "":
                return None
            return v

        for i in range(len(paths)):
            face = faces_by_path.get(paths[i])
            if face is None:
                continue
            for c in attr_cols:
                setattr(face, c, _clean(col_lists[c][i]))
            if source_col is not None:
                face.origin = _clean(col_lists[source_col][i])
            if has_error:
                raw = _clean(col_lists["error"][i])
                if raw is None:
                    face.error = None
                elif isinstance(raw, str):
                    face.error = raw.strip().lower() in ("true", "1", "yes")
                else:
                    face.error = bool(raw)
            extra: Dict[str, Any] = {}
            for c in custom_cols:
                v = col_lists[c][i]
                if isinstance(v, (list, np.ndarray, dict)):
                    continue
                try:
                    if pd.isna(v):
                        continue
                except (TypeError, ValueError):
                    pass
                extra[c] = _clean(v)
            face.extra_data = extra

    def is_dirty(self) -> bool:
        return self._is_dirty

    def set_dirty(self, state: bool):
        self._is_dirty = state
        if self.dataset:
            self._refresh_dataset_status()

    # ── Status bar: linha principal + tooltip rico ───────────────────
    def _build_dataset_status(
        self, *, filtered: bool = False,
    ) -> tuple[str, str]:
        """Constrói (texto-HTML, tooltip-HTML) pra status bar.

        Linha principal compacta:
          📂 nome★ | 🖼️ N Imagens | 👤 N IDs | 🎯 alinhado |
          🧬 N FF · 📐 N OFIQ · 🎲 N SER-FIQ | ⚖ verdict (min–max/id) |
          ⚠️ N erros

        Badges omitidos quando 0 ou inaplicáveis. ★ aparece se dirty.

        Tooltip rich-text detalha: path do parquet, descrição,
        path_mode, hash, mtime, cobertura de enriquecimento e métricas
        de balanceamento (CV, ratio, min/max/mean/median).
        """
        if self.dataset is None:
            return ("Nenhum dataset carregado.", "")
        ds = self.dataset
        n_imgs = len(ds)
        # Passada ÚNICA sobre as faces — antes eram 5+ varreduras O(N)
        # separadas (number_of_identities ×2 + 4×sum) na thread da GUI, o que
        # somava segundos em datasets de 1M+. Aqui acumulamos tudo num loop só
        # e reusamos as contagens no verdict de balanceamento.
        from collections import Counter as _Counter
        _id_counts = _Counter()
        n_ff = n_ofiq = n_serfiq = n_err = 0
        for _f in ds.faces:
            _idv = _f.identity
            if _idv:
                _id_counts[_idv] += 1
            if _f.ff_info is not None:
                n_ff += 1
            if _f.ofiq_info is not None:
                n_ofiq += 1
            if _f.serfiq_info is not None:
                n_serfiq += 1
            if getattr(_f, "error", False):
                n_err += 1
        n_ids = len(_id_counts)
        nome = ds.name or "Sem Nome"
        star = "*" if self._is_dirty else ""
        suffix = " (Filtrado)" if filtered else ""

        parts: list[str] = [
            f"📂 <b>{nome}{star}</b>",
            f"🖼️ {n_imgs} Imagens",
            f"👤 {n_ids} IDs",
        ]

        # is_aligned — só mostra quando True (a maioria é "originais").
        if getattr(ds, "is_aligned", False):
            parts.append("🎯 alinhado")

        # Coberturas de enriquecimento (já contadas na passada única acima).
        enrich_badges = []
        if n_ff:
            enrich_badges.append(f"🧬 {n_ff} FF")
        if n_ofiq:
            enrich_badges.append(f"📐 {n_ofiq} OFIQ")
        if n_serfiq:
            enrich_badges.append(f"🎲 {n_serfiq} SER-FIQ")
        if enrich_badges:
            parts.append(" · ".join(enrich_badges))

        # Balanceamento — usa BalanceMetrics consistente com o resto
        # do app (CV + imbalance ratio, worst-of-two).
        balance_main, balance_detail = self._compute_balance_for_status(
            ds, id_counts=_id_counts,
        )
        if balance_main:
            parts.append(balance_main)

        # Erros (já contados na passada única acima).
        if n_err:
            parts.append(f"⚠️ {n_err} erro(s)")

        text = " | ".join(parts) + suffix

        # ── Tooltip rich-text ──
        tooltip_lines: list[str] = []
        if self.current_path:
            tooltip_lines.append(f"<b>Path:</b> {self.current_path}")
        if ds.description:
            tooltip_lines.append(f"<b>Descrição:</b> {ds.description}")
        tooltip_lines.append(
            f"<b>Modo de path:</b> {ds.path_mode.value}"
            + (f" (root={ds.path_root})" if ds.path_root else "")
        )
        # Sempre mostra o estado de alinhamento no tooltip — útil quando
        # o usuário cria com "Alinhar durante criação" e quer confirmar
        # que o dataset ficou marcado.
        if getattr(ds, "is_aligned", False):
            tooltip_lines.append("<b>Conteúdo:</b> faces alinhadas 112×112")
        else:
            tooltip_lines.append("<b>Conteúdo:</b> fotos originais (não alinhadas)")
        # Coverage % por estágio (mais informativo que count).
        if n_imgs > 0:
            tooltip_lines.append(
                f"<b>Enriquecimento:</b> "
                f"FF {n_ff}/{n_imgs} ({100*n_ff//n_imgs}%), "
                f"OFIQ {n_ofiq}/{n_imgs} ({100*n_ofiq//n_imgs}%), "
                f"SER-FIQ {n_serfiq}/{n_imgs} ({100*n_serfiq//n_imgs}%)"
            )
        if balance_detail:
            tooltip_lines.append(f"<b>Balanceamento:</b> {balance_detail}")
        if n_err:
            tooltip_lines.append(
                f"<b>Erros:</b> {n_err} face(s) marcadas — "
                f"filtre por <code>error=True</code> na tabela."
            )
        # File stats — best effort (parquet pode não existir em
        # datasets em memória).
        if self.current_path:
            try:
                p = Path(self.current_path)
                if p.exists():
                    from datetime import datetime
                    size_mb = p.stat().st_size / (1024 * 1024)
                    mtime = datetime.fromtimestamp(
                        p.stat().st_mtime,
                    ).strftime("%Y-%m-%d %H:%M")
                    tooltip_lines.append(
                        f"<b>Tamanho:</b> {size_mb:.1f} MB · "
                        f"<b>Modificado:</b> {mtime}"
                    )
            except Exception:
                pass

        tooltip = "<br>".join(tooltip_lines)
        return (text, tooltip)

    @staticmethod
    def _compute_balance_for_status(ds, id_counts=None) -> tuple[str, str]:
        """Calcula verdict de balanceamento + detalhe.
        Retorna ("", "") quando inaplicável (≤1 identidade).

        `id_counts` (Counter identidade→contagem) pode ser passado já
        computado pela passada única de `_build_dataset_status` — evita
        uma segunda varredura O(N) sobre as faces.
        """
        from collections import Counter
        from faceaudit.models.dataset_stats import DatasetStats
        if id_counts is not None:
            counts = id_counts
        else:
            counts = Counter(f.identity for f in ds.faces if f.identity)
        if len(counts) < 2:
            return ("", "")
        # Reusa a lógica de BalanceMetrics via DatasetStats inline.
        stub = DatasetStats(
            n_imgs=len(ds), n_ids=len(counts),
            samples_per_identity=dict(counts),
        )
        bm = stub.balance_metrics
        if bm is None:
            return ("", "")
        main = (
            f"{bm.verdict} ({bm.min_samples}–{bm.max_samples}/id)"
        )
        detail = (
            f"{bm.verdict} · "
            f"CV={bm.cv:.2f} · max/min={bm.imbalance_ratio:.1f}:1 · "
            f"min={bm.min_samples}, max={bm.max_samples}, "
            f"mean={bm.mean_samples:.1f}, median={bm.median_samples:.0f}"
        )
        return (main, detail)

    def fluxo_abrir_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Abrir Dataset", self._get_initial_dir(), "Parquet (*.parquet)"
        )
        if not path:
            return
        self.open_dataset_from_path(path)

    def open_dataset_from_path(self, path: str) -> None:
        """Carrega um dataset diretamente de um path absoluto, pulando o
        QFileDialog. Entry point usado por chamadas cross-tab (ex:
        Strategy Manager V2 "Abrir no Gerenciador de Datasets")."""
        if not path:
            return
        self._run_worker(
            LoadDatasetWorker(path),
            label="Lendo arquivo Parquet...",
            allow_cancel=False,
            determinate=False,
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

    def _on_load_success(self, payload, path):
        try:
            # O worker de abertura emite (dataset, display_df) com o df de
            # exibição metadata-only já construído fora da GUI. Defensivo:
            # aceita também um dataset puro (compat com chamadas legadas).
            if isinstance(payload, tuple):
                dataset, display_df = payload
            else:
                dataset, display_df = payload, None
            self.carregar_dataset_na_tela(dataset, path, display_df=display_df)
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
                self._escrever_pendencias_de_custodia(self.current_path, self.dataset)
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
                # Notifica subscribers (cross-tab) que o arquivo no path
                # atual foi sobrescrito. Strategy Manager V2 usa pra
                # detectar mudança em primary do projeto e propagar
                # hash + cascade.
                self.dataset_saved_to_project.emit(str(self.current_path))
            except Exception as e:
                QMessageBox.critical(self.view, "Erro", f"Falha ao salvar: {e}")
        else:
            self.fluxo_salvar_como()

    def fluxo_gerar_imagens_de_face(self) -> None:
        """Gera, numa pasta nova, a face de cada imagem de uma pasta.

        Independe de haver dataset aberto: a entrada é a pasta escolhida
        no diálogo. Não cria dataset — só imagens e o manifesto.
        """
        from .extract_face_images_dialog import ExtractFaceImagesDialog

        ff_svc = getattr(self.pipeline, "ff_service", None) if self.pipeline else None
        if ff_svc is None:
            QMessageBox.warning(
                self.view, "Serviço indisponível",
                "A detecção facial não está disponível nesta sessão, então "
                "não há como localizar as faces para recortar ou alinhar.",
            )
            return

        dlg = ExtractFaceImagesDialog(self.view, initial_dir=self._get_initial_dir())
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        from .workers.materialize_images_worker import MaterializeImagesWorker

        kind = dlg.image_kind()
        worker = MaterializeImagesWorker(
            dlg.source_dir(), dlg.target_dir(),
            kind=kind,
            margin_factor=dlg.margin_factor(),
            recursive=dlg.recursive(),
            ff_service=ff_svc,
        )
        self._run_worker(
            worker,
            label=f"Gerando imagens ({kind.label_pt})",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,
            on_finished=self._on_gerar_imagens_concluido,
        )

    def _on_gerar_imagens_concluido(self, resultado) -> None:
        """Relata o saldo. As falhas precisam ser ditas, não só contadas."""
        if resultado is None:
            QMessageBox.warning(
                self.view, "Cancelado",
                "A geração de imagens foi cancelada ou terminou sem resultado.",
            )
            return

        detalhe = ""
        if resultado.n_failed:
            primeiras = "\n".join(
                f"  • {caminho}" for caminho, _motivo in resultado.failures[:5]
            )
            if resultado.n_failed > 5:
                primeiras += f"\n  • (… e mais {resultado.n_failed - 5})"
            detalhe = (
                f"\n\n{resultado.n_failed} imagem(ns) ficaram de fora por não "
                f"ter face detectada:\n{primeiras}"
            )

        QMessageBox.information(
            self.view, "Imagens geradas",
            f"{resultado.summary_pt()}\n\nPasta: {resultado.target_dir}\n"
            f"Manifesto: {resultado.manifest_path.name if resultado.manifest_path else '—'}"
            f"{detalhe}",
        )

    def _current_dataset_dir(self) -> Optional[Path]:
        """Pasta onde o dataset vive hoje — âncora dos caminhos relativos."""
        if self.current_path:
            return Path(self.current_path).parent.resolve()
        origem = getattr(self.dataset, "_parquet_dir", None)
        return Path(origem).resolve() if origem else None

    def _ask_save_as_options(
        self, destination: str, *, ask_images: bool, ask_which_stays: bool,
        a_gravar=None, origem=None,
    ):
        """Abre o diálogo de opções do Salvar Como.

        Retorna `(portatil, manter_origem)` ou None se o usuário cancelou.
        `portatil` vem None quando a pergunta sobre imagens não se aplica
        (salvando na mesma pasta).
        """
        from .save_as_options_dialog import SaveAsOptionsDialog

        dlg = SaveAsOptionsDialog(
            self.view,
            ask_images=ask_images,
            ask_which_stays=ask_which_stays,
            nome_destino=Path(destination).stem,
            nome_origem=getattr(origem, "name", "") or "dataset de origem",
            n_salvar=len(a_gravar) if a_gravar else 0,
            n_origem=len(origem) if origem else 0,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return dlg.portable(), dlg.keep_original()

    def fluxo_salvar_como(self):
        """Salvar Como do dataset aberto.

        Com filtro ativo, a origem é o dataset de onde o recorte veio — é
        ela que pode continuar na tela depois de gravar.
        """
        if not self.dataset:
            return
        self.sync_changes_from_table()
        self._executar_salvar_como(
            self.dataset, origem=self.dataset_original, titulo="Salvar Como",
        )

    def fluxo_salvar_selecionados(self):
        """Grava a seleção como um dataset novo, sem sair do que está aberto.

        É o mesmo Salvar Como com a seleção no lugar do filtro: mesmo
        diálogo de opções (autocontido × referenciar, e qual dataset fica
        aberto), mesma cópia de imagens em worker, mesma finalização com
        audit e sidecar. O padrão continua sendo permanecer no dataset de
        origem — deriva-se vários recortes do mesmo dataset grande.
        """
        subset = self._montar_subset_da_selecao()
        if subset is None:
            return
        self._executar_salvar_como(
            subset, origem=self.dataset, titulo="Salvar Selecionados",
        )

    def _montar_subset_da_selecao(self):
        """FaceDataset com as faces selecionadas, ou None se desistir.

        Não toca na tela: quem decide o que fica aberto é o fluxo de
        gravação, depois de perguntar.
        """
        import copy as _copy
        from PyQt6.QtWidgets import QInputDialog

        if not self.dataset:
            return None
        matched = self._faces_from_paths(
            self._current_selection_paths(), "Salvar Selecionados",
        )
        if not matched:
            return None

        sugerido = f"{self.dataset.name}_sel"
        nome, ok = QInputDialog.getText(
            self.view, "Salvar Selecionados",
            f"Nome do novo dataset ({len(matched)} registro(s)):", text=sugerido)
        if not ok or not nome.strip():
            return None

        subset = FaceDataset(
            name=nome.strip(),
            description=getattr(self.dataset, "description", "") or "",
            faces=[_copy.copy(f) for f in matched],
            column_specs=dict(self.dataset.column_specs) if self.dataset.column_specs else {},
            path_mode=self.dataset.path_mode, path_root=self.dataset.path_root,
            is_aligned=self.dataset.is_aligned,
        )
        subset._parquet_dir = self.dataset._parquet_dir
        subset.hash = None
        return subset

    def _executar_salvar_como(self, a_gravar, origem, *, titulo: str):
        default_name = f"{a_gravar.name}.parquet"
        init_path = os.path.join(self._get_initial_dir(), default_name)

        path, _ = QFileDialog.getSaveFileName(self.view, titulo, init_path, "Parquet (*.parquet)")
        if not path:
            return

        origem_dir = self._current_dataset_dir()
        # Imagens: só há o que decidir quando elas ficam para trás. Salvar
        # na mesma pasta preserva a âncora e não merece pergunta.
        perguntar_imagens = (
            origem_dir is not None and Path(path).parent.resolve() != origem_dir
        )
        # Dataset aberto: só há duas opções quando existe um dataset de
        # origem por trás do que será gravado (filtro ativo, ou seleção).
        perguntar_qual_fica = origem is not None

        manter_origem = False
        portatil = None
        if perguntar_imagens or perguntar_qual_fica:
            escolha = self._ask_save_as_options(
                path,
                ask_images=perguntar_imagens,
                ask_which_stays=perguntar_qual_fica,
                a_gravar=a_gravar, origem=origem,
            )
            if escolha is None:
                return
            portatil, manter_origem = escolha

        a_gravar.name = Path(path).stem
        if portatil:
            # Cópia de imagens é pesada — vai para worker, e o parquet é
            # gravado quando ela termina.
            self._pending_save_as_path = path
            self._pending_save_as_keep_original = manter_origem
            self._pending_save_as_dataset = a_gravar
            self._pending_save_as_origin = origem
            self._iniciar_worker_salvar_como_portatil(path, a_gravar)
            return
        self._last_path_rebase = None
        if portatil is False:
            from faceaudit.services.dataset_localization_service import (
                prepare_dataset_for_destination,
            )
            a_gravar, relatorio = prepare_dataset_for_destination(
                a_gravar, path, portable=False,
            )
            # Referenciar não é só "absolutizar": o serviço tenta primeiro
            # reapontar o path_root para a MESMA pasta de imagens, o que
            # mantém o dataset relativo e sobrevive a mover a árvore
            # inteira para outra máquina.
            self._last_path_rebase = relatorio.rebase

        self._gravar_e_finalizar_salvar_como(
            path, a_gravar, origem, manter_origem=manter_origem,
        )

    def _descrever_conteudo_salvo(self, a_gravar) -> dict:
        """O que está sendo gravado: o dataset inteiro, a vista filtrada ou
        uma seleção — e, havendo filtro ativo, o critério dele.

        Com filtro, o dataset na tela É o recorte e `dataset_original` é o
        inteiro; a seleção chega como outro objeto (cópias das faces).
        """
        filtro = getattr(self, "current_filter_config", None)
        filtrado = (
            getattr(self, "dataset_original", None) is not None
            and bool(getattr(filtro, "filters", None))
        )
        if a_gravar is not self.dataset:
            tipo = "selection"
        elif filtrado:
            tipo = "filtered_view"
        else:
            tipo = "full"
        out = {"content": tipo}
        if filtrado:
            try:
                out["filter"] = _sem_infinito(filtro.to_dict())
            except Exception as e:
                out["filter_error"] = str(e)
        return out

    def _gravar_e_finalizar_salvar_como(
        self, path: str, a_gravar=None, origem=None, *,
        manter_origem: bool = False,
    ) -> None:
        """Grava o parquet no destino e fecha o Save As (sidecar, audit, UI).

        Compartilhado pelos dois caminhos (direto e portátil, que chega aqui
        depois que o worker copia as imagens) e pelas duas origens de
        conteúdo: o dataset aberto e um recorte da seleção.

        `a_gravar` é o que vai para o disco; `origem` é o dataset que pode
        continuar na tela. Com `manter_origem`, o recorte é gravado mas quem
        fica é a origem — o fluxo típico é derivar vários recortes do mesmo
        dataset grande, e trocar para o recém-salvo obrigava a fechar e
        reabrir o grande a cada um.
        """
        if a_gravar is None:
            a_gravar = self.dataset
        old_path = self.current_path
        rebase = getattr(self, "_last_path_rebase", None)
        self._last_path_rebase = None
        # O QUE foi gravado vai para a cadeia de custódia — capturado antes
        # de os ramos abaixo desfazerem o estado de filtro. Um dataset
        # filtrado gravado com só "rows: 83" não permite reconstruir depois
        # qual critério produziu aquelas 83 linhas.
        conteudo = self._descrever_conteudo_salvo(a_gravar)
        a_gravar.name = Path(path).stem
        a_gravar.to_parquet(path)
        # PR 25 (Tema 5.4) — o arquivo novo herda a cadeia de custódia do
        # original: COPIA o `.audit.json` antes de adicionar a entrada
        # SAVE_AS. Copiar, não mover — mover deixava o dataset de origem
        # sem cadeia nenhuma.
        if old_path and old_path != path:
            try:
                AuditLogService.copy_sidecar(old_path, path)
            except Exception:
                pass  # se a cópia falhar, segue com chain nova
        # Entradas adiadas (dataset que ainda não tinha arquivo — ex.: o
        # resultado de um Juntar) vão para a cadeia do arquivo novo ANTES do
        # SAVE_AS, que traz o hash e fecha o elo. Vêm de todos os objetos de
        # que o gravado deriva (recorte de filtro, seleção, dataset raiz).
        self._escrever_pendencias_de_custodia(
            path, a_gravar, origem,
            getattr(self, "dataset", None), getattr(self, "dataset_original", None),
        )
        salvo_nome = a_gravar.name
        salvo_linhas = len(a_gravar)
        salvo_ids = a_gravar.number_of_identities

        if manter_origem and origem is not None:
            # A origem continua na tela, no arquivo DELA. O `current_path`
            # tem de acompanhar: se ficasse apontando para o recorte, um
            # "Salvar" seguinte gravaria a origem por cima dele.
            if origem is self.dataset_original:
                # Era um recorte de filtro: a origem volta inteira, então o
                # estado de filtro se desfaz.
                self.dataset_original = None
                self.current_filter_config = FilterConfig()
            self.set_dirty(False)
            # `carregar_dataset_na_tela` deriva o estado de filtro e
            # atualiza a status bar — preserva o filtro quando o que foi
            # gravado era uma seleção e a origem já era um recorte.
            self.carregar_dataset_na_tela(origem, old_path)
        else:
            # O recorte salvo passa a ser um dataset por si — desligar o
            # vínculo com a origem evita que um "Limpar filtro" depois
            # traga o dataset grande de volta POR CIMA do arquivo do
            # recorte (o `current_path` agora é o dele).
            self.dataset_original = None
            self.current_filter_config = FilterConfig()
            self.current_path = path
            self.set_dirty(False)
            self.carregar_dataset_na_tela(a_gravar, path)

        self._audit(
            AuditOperation.SAVE_AS,
            f"Dataset salvo em novo arquivo '{Path(path).name}'."
            + (" Origem permaneceu aberta." if manter_origem else ""),
            metadata={
                "previous_path": old_path,
                "rows": salvo_linhas,
                "identities": salvo_ids,
                "dataset_saved": salvo_nome,
                "kept_source_open": manter_origem,
                # O que aconteceu com os caminhos das imagens faz parte da
                # cadeia de custódia: o mesmo dataset gravado noutra pasta
                # pode ter mudado de ponteiro sem mudar de conteúdo.
                "path_rebase": (
                    rebase.action.value if rebase is not None else "not_needed"
                ),
                "path_root": rebase.path_root if rebase is not None else None,
                **conteudo,
            },
            path_override=path,
            hash_file=True,
        )
        # A cadeia da origem foi copiada, não movida — e ela pode conter
        # alterações registradas "até gravar" (exclusões, edições) que
        # foram gravadas no arquivo NOVO. Sem esta entrada, a cadeia da
        # origem afirmaria exclusões que o arquivo dela não sofreu.
        if old_path and old_path != path and Path(old_path).exists():
            self._audit(
                AuditOperation.SAVE_AS,
                f"Conteúdo gravado em novo arquivo '{Path(path).name}'. "
                "Este arquivo não foi alterado; alterações registradas "
                "como pendentes nesta cadeia, se houver, foram gravadas no "
                "arquivo novo.",
                metadata={
                    "saved_to": str(path),
                    "this_file_unchanged": True,
                    **conteudo,
                },
                path_override=old_path,
                hash_file=True,
            )

        if rebase is not None and rebase.changed:
            QMessageBox.information(
                self.view, "Caminhos das imagens", rebase.message,
            )

    def _iniciar_worker_salvar_como_portatil(self, path: str, a_gravar=None) -> None:
        """Copia as imagens para junto do destino, em worker.

        Reusa o worker de "Modificar Caminhos" — é exatamente a mesma
        operação (COPY_ORIGINALS com subpasta por dataset). O parquet é
        gravado em `_on_salvar_como_portatil_concluido`.
        """
        from faceaudit.enums.localization_mode import LocalizationMode
        from faceaudit.enums.localization_naming import NamingStrategy

        worker = ModifyPathsWorker(
            dataset=a_gravar if a_gravar is not None else self.dataset,
            target_dir=Path(path).parent,
            mode=LocalizationMode.COPY_ORIGINALS,
            naming=NamingStrategy.PRESERVE,
            skip_existing=True,
        )
        self._run_worker(
            worker,
            label=f"Copiando imagens: {worker.dataset.name}",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,
            on_finished=self._on_salvar_como_portatil_concluido,
        )

    def _on_salvar_como_portatil_concluido(self, result) -> None:
        """Fecha o Save As autocontido depois da cópia das imagens."""
        path = getattr(self, "_pending_save_as_path", None)
        self._pending_save_as_path = None
        if path is None:
            return
        if result is None:
            QMessageBox.warning(
                self.view, "Cancelado",
                "A cópia das imagens foi cancelada — nada foi gravado.",
            )
            return

        # Faces cuja imagem não foi encontrada são DESCARTADAS pela
        # localização. Gravar sem avisar produz um dataset menor (ou
        # vazio) sem que ninguém perceba.
        if not len(result.dataset):
            falhas = "\n".join(
                f"  • {caminho}: {motivo}"
                for caminho, motivo in list(result.failures)[:5]
            )
            QMessageBox.critical(
                self.view, "Nenhuma imagem encontrada",
                f"Nenhuma das {result.n_processed} imagens foi encontrada — "
                f"o arquivo NÃO foi gravado.\n\nPrimeiras falhas:\n{falhas}",
            )
            return
        if result.n_failed:
            falhas = "\n".join(
                f"  • {caminho}" for caminho, _m in list(result.failures)[:5]
            )
            resposta = QMessageBox.warning(
                self.view, "Imagens não encontradas",
                f"{result.n_failed} de {result.n_processed} imagens não foram "
                f"encontradas e ficarão DE FORA do dataset salvo.\n\n"
                f"Primeiras:\n{falhas}\n\n"
                f"Gravar assim mesmo, com {len(result.dataset)} face(s)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if resposta != QMessageBox.StandardButton.Yes:
                return

        from faceaudit.services.dataset_localization_service import (
            adopt_identity_from,
        )
        # O worker criou a pasta própria do dataset (`<escolhida>/<nome>/`)
        # e copiou as imagens para `images/` dentro dela. O parquet vai
        # para lá também — é o que mantém `path_root` válido.
        destino = Path(result.target_dir) / Path(path).name
        original = getattr(self, "_pending_save_as_dataset", None) or self.dataset
        a_gravar = adopt_identity_from(result.dataset, original, destino.parent)
        origem = getattr(self, "_pending_save_as_origin", None)
        manter_origem = getattr(self, "_pending_save_as_keep_original", False)
        self._pending_save_as_keep_original = False
        self._pending_save_as_dataset = None
        self._pending_save_as_origin = None
        self._gravar_e_finalizar_salvar_como(
            str(destino), a_gravar, origem, manter_origem=manter_origem,
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
        # Limpa a galeria também — senão, ao fechar em modo galeria, as
        # miniaturas do dataset anterior continuam aparecendo.
        self.gallery_widget.clear_cache()
        self.gallery_widget.populate([])
        self.detail_ctrl.load_face_data(None)
        self.toolbar_ctrl.set_dataset_loaded_state(False, "Nenhum dataset carregado.")
        self.set_dirty(False)
        return True

    def fluxo_fechar_dataset(self):
        self.request_close()

    def fluxo_novo_dataset_video(self):
        """🎬 Abre o fluxo de criação de dataset a partir de vídeo."""
        self._video_ctrl_lazy()  # cursor de espera no import pesado da 1ª vez
        self._video_ctrl.run_new_from_video()

    def fluxo_importar_run_video(self):
        """📥 Importa um run de extração de vídeo já produzido (revisar + dataset)."""
        self._video_ctrl_lazy()
        self._video_ctrl.run_import_video_run()

    def _video_ctrl_lazy(self):
        """Cria o VideoDatasetController sob demanda. O import puxa cv2 +
        FFService + serviços de vídeo — pesado na 1ª vez —, então mostra o
        cursor de ocupado para dar feedback ao clicar no botão."""
        if getattr(self, "_video_ctrl", None) is not None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()  # paint imediato do cursor
        try:
            from .video_dataset_dialog.video_dataset_controller import VideoDatasetController
            self._video_ctrl = VideoDatasetController(self)
        finally:
            QApplication.restoreOverrideCursor()

    def fluxo_novo_dataset(self):
        templates_path = Path(self.config.paths.dir_templates)
        default_dataset_dir = str(getattr(self.config.paths, "datasets", "") or "")
        ff_cfg = getattr(self.config, "ff", None)
        dialog = NewDatasetController(
            self.view,
            templates_path=templates_path,
            ff_available=self.ff_available,
            ofiq_available=self.ofiq_available,
            default_dataset_dir=default_dataset_dir,
            default_use_gpu=bool(getattr(ff_cfg, "use_gpu", False)),
            default_batch_size=int(getattr(ff_cfg, "batch_size", 32)),
        )
        
        if dialog.exec():
            ingestion_config: DatasetIngestionConfig = dialog.get_config()
            # O caminho final vem do backend — no modo autocontido o parquet
            # mora em `Destino/{nome}/{nome}.parquet`, não em `Destino/`.
            cache_file = self.pipeline.destination_parquet(ingestion_config)

            if cache_file.exists():
                msg = f"O dataset '{ingestion_config.dataset_name}' já existe. Sobrescrever?"
                res = QMessageBox.question(self.view, "Confirmar", msg)
                if res == QMessageBox.StandardButton.No: return

            self._iniciar_worker_criacao(ingestion_config)

    def _iniciar_worker_criacao(self, config):
        # A gravação pode acontecer numa segunda etapa (após a revisão das
        # imagens sem face detectada), e ali o config é indispensável para
        # gravar no mesmo destino.
        self._ingestion_config = config
        self._run_worker(
            NewDatasetWorker(self.pipeline, config),
            label="Criando Dataset",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,
            on_finished=self._on_criacao_concluida,
        )

    def _on_criacao_concluida(self, dataset):
        # Ingestão parada antes de gravar porque há imagens sem face
        # detectada esperando decisão. A tela de progresso já fechou
        # (o runner desmonta a UI antes de chamar este retorno), então a
        # revisão pode ser modal sem risco de prender sinais do worker.
        if dataset is not None and getattr(dataset, "_save_deferred", False):
            self._revisar_nao_detectadas_e_gravar(dataset)
            return
        self._concluir_criacao(dataset)

    def _revisar_nao_detectadas_e_gravar(self, dataset):
        """Abre a revisão, aplica a decisão e dispara a gravação."""
        from .undetected_review_dialog import UndetectedReviewController

        pendentes = list(getattr(dataset, "_undetected_faces", []) or [])
        config = getattr(self, "_ingestion_config", None)
        if not pendentes or config is None:
            # Não deveria acontecer; gravar é melhor que abandonar o
            # dataset processado em memória.
            self._finalizar_e_gravar(dataset, incluidas=[], descartadas=[])
            return

        ctrl = UndetectedReviewController(
            self.view,
            faces=pendentes,
            config=config,
            dataset=dataset,
            n_total=len(dataset) + len(pendentes),
            ff_service_factory=self._fabrica_de_ff_service(config),
        )
        ctrl.exec()
        incluidas, descartadas = ctrl.decision()
        recuperadas = ctrl.recovered_faces()

        for face in incluidas:
            dataset.add(face)

        if len(dataset) == 0:
            # Nada sobrou. Gravar aqui produziria um parquet de zero linhas,
            # que a interface aceita, lista como recurso e some dos
            # seletores sem explicação.
            QMessageBox.warning(
                self.view, "Nada a gravar",
                f"Nenhuma das {len(descartadas)} imagens rendeu face detectada "
                "e todas foram descartadas na revisão.\n\n"
                "O dataset não foi criado — um arquivo vazio passaria por "
                "dataset válido e desapareceria dos seletores sem explicação.",
            )
            return

        self._finalizar_e_gravar(dataset, incluidas, descartadas, recuperadas)

    def _fabrica_de_ff_service(self, config):
        """Devolve uma função que constrói um FFService na resolução pedida.

        Injetada na revisão para que ela possa repetir a detecção em
        resolução maior sem saber como um FFService é construído — e para
        que, nos testes, a fábrica seja um duplo.
        """
        from faceaudit.services.ff_service import FFService
        from faceaudit.enums.sepael_model import SepaelModel

        modelos = list(
            getattr(config, "ff_models", None) or [SepaelModel.default().ff_name]
        )
        singleton = getattr(self.pipeline, "ff_service", None)
        usar_gpu = config.use_gpu
        if usar_gpu is None:
            usar_gpu = bool(getattr(singleton, "_use_gpu", False))
        lote = config.batch_size or int(getattr(singleton, "_batch_size", 32) or 32)

        def _construir(det_size: int):
            # Sempre uma instância nova: os modelos ONNX são compilados para
            # a resolução pedida, então o singleton (em outra resolução) não
            # serve. Quem chama guarda em cache — a carga é paga uma vez por
            # resolução, não por imagem.
            return FFService(
                models=modelos,
                det_size=int(det_size),
                use_gpu=usar_gpu,
                extended=bool(config.ff_extended),
                batch_size=lote,
            )

        return _construir

    def _finalizar_e_gravar(self, dataset, incluidas, descartadas, recuperadas=None):
        """Segunda etapa: copiar imagens e gravar, em worker."""
        from .workers.finalize_dataset_worker import FinalizeDatasetWorker

        self._revisao_incluidas = list(incluidas)
        self._revisao_descartadas = list(descartadas)
        self._revisao_recuperadas = list(recuperadas or [])
        self._run_worker(
            FinalizeDatasetWorker(
                self.pipeline, dataset, self._ingestion_config,
                recovered_faces=self._revisao_recuperadas,
            ),
            label="Gravando Dataset",
            determinate=True,
            use_log_dialog=True,
            on_finished=self._concluir_criacao,
        )

    def _concluir_criacao(self, dataset):
        if dataset and len(dataset) > 0:
            # `_parquet_dir` é populado pelo pipeline com o destino efetivo
            # (mesmo padrão do fluxo de vídeo); storage_path é só o fallback.
            base_dir = Path(getattr(dataset, "_parquet_dir", None) or self.pipeline.storage_path)
            path_cache = base_dir / f"{dataset.name}.parquet"
            self.carregar_dataset_na_tela(dataset, path=str(path_cache))
            # PR 25 (Tema 5.4) — registra CREATE como primeira entrada
            # da chain. hash_file=True dá um anchor inicial: o hash do
            # parquet recém-ingerido.
            self._audit(
                AuditOperation.CREATE,
                self._descricao_da_criacao(dataset),
                metadata={
                    "rows": len(dataset),
                    "identities": dataset.number_of_identities,
                    **self._metadados_da_deteccao(dataset),
                    **self._metadados_do_ofiq(
                        dataset,
                        rodou=getattr(self._ingestion_config, "compute_ofiq", False),
                    ),
                },
                hash_file=True,
            )
            QMessageBox.information(self.view, "Sucesso", "Dataset Criado!")
        else:
            QMessageBox.warning(self.view, "Aviso", "Processo finalizado sem dados.")

    # -----------------------------------------------------------------
    # Proveniência da detecção (vai para a cadeia de custódia)
    # -----------------------------------------------------------------

    def _descricao_da_criacao(self, dataset) -> str:
        """Texto da entrada CREATE, citando o que a detecção exigiu.

        Quantas imagens só renderam face depois de girar, e quantas ficaram
        sem face nenhuma, é informação de proveniência: sem ela, ninguém
        reconstrói depois por que o dataset tem menos linhas que a pasta de
        origem tinha arquivos.
        """
        partes = [f"Dataset '{dataset.name}' criado a partir de ingestão."]
        n_giradas = self._contar_giradas(dataset)
        if n_giradas:
            partes.append(
                f"{n_giradas} face(s) só foram detectadas com a imagem girada."
            )
        incluidas = getattr(self, "_revisao_incluidas", None) or []
        descartadas = getattr(self, "_revisao_descartadas", None) or []
        recuperadas = getattr(self, "_revisao_recuperadas", None) or []
        if incluidas or descartadas:
            partes.append(
                f"Revisão de imagens sem face detectada: {len(incluidas)} "
                f"incluída(s) por decisão do operador, {len(descartadas)} "
                f"descartada(s)."
            )
        if recuperadas:
            tamanhos = sorted({
                int(getattr(getattr(f, "ff_info", None), "detection_det_size", 0) or 0)
                for f in recuperadas
            })
            partes.append(
                f"{len(recuperadas)} dessas face(s) foram encontradas numa "
                f"segunda tentativa, com o detector em "
                f"{', '.join(str(t) for t in tamanhos if t)} px."
            )
        return " ".join(partes)

    def _metadados_da_deteccao(self, dataset) -> dict:
        """Contagens e nomes de arquivo da revisão, para a custódia."""
        incluidas = getattr(self, "_revisao_incluidas", None) or []
        descartadas = getattr(self, "_revisao_descartadas", None) or []
        recuperadas = getattr(self, "_revisao_recuperadas", None) or []
        meta: dict = {}
        if recuperadas:
            # Recuperada é caso diferente de incluída-sem-biometria: esta
            # TEM embedding, obtido numa segunda tentativa em resolução
            # maior. Registrar as duas juntas apagaria a distinção.
            meta["recovered_on_retry"] = len(recuperadas)
            meta["recovered_files"] = self._nomes_de_arquivo(recuperadas)
            meta["recovered_det_sizes"] = sorted({
                int(getattr(getattr(f, "ff_info", None), "detection_det_size", 0) or 0)
                for f in recuperadas
            })
        n_giradas = self._contar_giradas(dataset)
        if n_giradas:
            meta["faces_detected_after_rotation"] = n_giradas
        if not (incluidas or descartadas):
            return meta
        meta["undetected_total"] = len(incluidas) + len(descartadas)
        meta["undetected_included"] = len(incluidas)
        meta["undetected_discarded"] = len(descartadas)
        # Os nomes importam mais que a contagem para quem confere depois;
        # limitados pelo mesmo teto das outras entradas de custódia.
        meta["undetected_included_files"] = self._nomes_de_arquivo(incluidas)
        meta["undetected_discarded_files"] = self._nomes_de_arquivo(descartadas)
        return meta

    def _metadados_do_ofiq(self, dataset, *, rodou: bool) -> dict:
        """De qual OFIQ vieram as métricas de qualidade, para a custódia.

        `ofiq_versions` conta as faces por origem e vale mesmo quando o
        OFIQ não rodou nesta operação (as métricas antigas continuam no
        dataset). Versão, modo e modelos do serviço só entram quando ele
        de fato rodou — senão a entrada atribuiria à operação um OFIQ que
        ela não usou.
        """
        from faceaudit.services.ofiq_provenance_service import summarize_ofiq_versions

        meta: dict = {}
        resumo = summarize_ofiq_versions(getattr(dataset, "faces", []) or [])
        if resumo:
            meta["ofiq_versions"] = resumo
        servico = getattr(self.pipeline, "ofiq_service", None) if self.pipeline else None
        if rodou and servico is not None and hasattr(servico, "provenance"):
            try:
                meta.update(servico.provenance())
            except Exception as e:  # noqa: BLE001 — custódia é best-effort
                meta["ofiq_provenance_error"] = str(e)
        return meta

    @staticmethod
    def _contar_giradas(dataset) -> int:
        return sum(
            1 for f in getattr(dataset, "faces", []) or []
            if getattr(getattr(f, "ff_info", None), "detection_rotation", None)
        )

    @staticmethod
    def _nomes_de_arquivo(faces) -> list:
        nomes = [
            Path(str(getattr(f, "image_path", "") or "")).name for f in faces
        ]
        return nomes[:_MAX_AUDIT_IDENTIFIERS]

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
        # editadas ali já apareçam no relatório. WaitCursor cobre todo
        # o setup pesado (sync + computação de stats de cobertura no
        # construtor do dialog); restaurado antes do `exec()` pra que
        # o usuário interaja com cursor normal.
        from .schema_report_dialog import SchemaReportDialog

        # PR 25 (Tema 5.4) — callback de audit pro dialog.
        def _audit_export(meta: dict) -> None:
            self._audit(
                AuditOperation.EXPORT,
                f"Dataset exportado para {meta.get('format', 'arquivo')} "
                f"({meta.get('rows', '?')} linhas, "
                f"{meta.get('cols', '?')} colunas).",
                metadata=meta,
            )

        # Callback de visibilidade — chamado a cada toggle do checkbox
        # da coluna "Visível" no dialog. Aplica imediatamente no model
        # da tabela (mesmo caminho do menu de contexto do header).
        def _on_visibility_change(col_name: str, visible: bool) -> None:
            try:
                if self.table_ctrl.model.set_column_visibility(col_name, visible):
                    from .table.vocabulary_combo_delegate import (
                        install_vocabulary_delegates,
                    )
                    install_vocabulary_delegates(
                        self.table_ctrl.view, self.table_ctrl.model,
                    )
                    self.table_ctrl.view.resizeColumnsToContents()
                    self.set_dirty(True)
            except Exception as e:
                print(f"[warn] visibility toggle falhou: {e}")

        with wait_cursor():
            self.sync_changes_from_table()
            dialog = SchemaReportDialog(
                self.dataset, parent=self.view,
                on_export=_audit_export,
                on_visibility_change=_on_visibility_change,
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
    # PR L1d — FLUXO "MODIFICAR CAMINHOS" (corrige path_root / relativiza / copia)
    # =================================================================================
    def fluxo_modificar_caminhos(self):
        """Abre o ModifyPathsDialog. `FIX_PATH_ROOT` é aplicado direto
        (só-metadados, sem mover arquivos); as migrações pesadas
        (REWRITE/COPY/REBASE) disparam o worker e, ao final, substituem
        o dataset corrente + propõem Save As (consistência com
        Enrich/Filter)."""
        if not self.dataset:
            return

        # sync + detect_path_prefix + construção do dialog são síncronos
        # e podem demorar em datasets grandes → wait cursor cobre tudo
        # até o dialog estar pronto pra interagir.
        with wait_cursor():
            self.sync_changes_from_table()

            # Sugere target_dir = <parquet_dir_atual>/<dataset_name>__localized/
            if self.current_path:
                base = Path(self.current_path).parent
            else:
                base = Path(self._get_initial_dir() or os.getcwd())
            suggested = base / f"{self.dataset.name}__localized"

            # Auto-detecta o prefixo comum dos image_paths atuais — só
            # é útil pro modo REBASE_PREFIX, mas computar antes evita
            # uma round-trip do dialog ao service.
            from faceaudit.services.dataset_localization_service import (
                DatasetLocalizationService,
            )
            detected = DatasetLocalizationService.detect_path_prefix(self.dataset)

            # Pasta do parquet: ancora o picker de subpasta e a validação
            # de path_root. `_parquet_dir` é runtime (from_parquet); cai
            # no diretório do current_path quando ausente.
            parquet_dir = getattr(self.dataset, "_parquet_dir", None)
            if parquet_dir is None and self.current_path:
                parquet_dir = Path(self.current_path).parent

            from .modify_paths_dialog import ModifyPathsDialog, PathOp
            dialog = ModifyPathsDialog(
                dataset_name=self.dataset.name,
                suggested_target_dir=suggested,
                parent=self.view,
                detected_prefix=detected,
                path_mode=self.dataset.path_mode,
                path_root=self.dataset.path_root,
                parquet_dir=parquet_dir,
                dataset=self.dataset,
            )
        # exec() fora do wait_cursor — dialog precisa de cursor normal
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        try:
            result = dialog.get_result()
        except ValueError as e:
            QMessageBox.warning(self.view, "Configuração inválida", str(e))
            return

        # Correção leve de path_root — só-metadados, aplicada em memória
        # sem worker (não move nenhum arquivo).
        if result["op"] == PathOp.FIX_PATH_ROOT:
            self._aplicar_path_root(result["path_root"])
            return

        self._iniciar_worker_modificar_caminhos(
            target_dir=result["target_dir"],
            mode=result["mode"],
            naming=result["naming"],
            skip_existing=result["skip_existing"],
            old_prefix=result.get("old_prefix"),
            image_kind=result.get("image_kind"),
            crop_margin_factor=result.get("crop_margin_factor", 2.0),
        )

    # ── Correção de path_root (só-metadados, sem mover arquivos) ─────────
    def _aplicar_path_root(self, new_root: str):
        """Repontar a subpasta (`path_root`) de um dataset RELATIVE sem
        mover arquivos. Valida por amostragem; se falhar, exige
        confirmação explícita antes de aplicar."""
        if not self.dataset:
            return
        old_root = self.dataset.path_root or ""
        new_root = normalize_stored_path((new_root or "").strip())

        # "", ".", None são todos equivalentes à raiz — early-out se
        # não houve mudança efetiva.
        if self._canon_root(new_root) == self._canon_root(old_root):
            QMessageBox.information(
                self.view, "Sem alteração",
                "A subpasta (path_root) já está com esse valor.",
            )
            return

        ok, msg = self._validate_path_root(new_root)
        if not ok:
            resp = QMessageBox.question(
                self.view, "Validação falhou",
                f"{msg}\n\nAplicar mesmo assim?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        # Aplica em memória. path_mode defensivamente RELATIVE (path_root
        # só faz sentido em RELATIVE; o combo só oferece FIX nesse modo,
        # mas garantimos coerência).
        self.dataset.path_root = new_root or None
        self.dataset.path_mode = PathMode.RELATIVE

        # Refresh mínimo ANTES do set_dirty (o refresh não toca no dirty;
        # carregar_dataset_na_tela tocaria — por isso NÃO é usado aqui).
        self._refresh_after_path_root_change()
        self.set_dirty(True)

        try:
            self._audit(
                AuditOperation.COPY_IMAGES,
                (
                    f"path_root alterado de '{old_root or '(raiz)'}' para "
                    f"'{new_root or '(raiz)'}' (edição de metadados, sem "
                    "mover arquivos)."
                ),
                metadata={
                    "operation": "fix_path_root",
                    "old_path_root": old_root,
                    "new_path_root": new_root,
                },
            )
        except Exception:
            pass  # audit é best-effort

        QMessageBox.information(
            self.view, "Subpasta atualizada",
            "O path_root foi atualizado em memória. Use 'Salvar' pra "
            "gravar a mudança no arquivo do dataset.",
        )

    @staticmethod
    def _canon_root(root: Optional[str]) -> str:
        """Canonicaliza um path_root pra comparação: '', '.', None → ''."""
        r = (root or "").strip()
        return "" if r in ("", ".") else normalize_stored_path(r)

    def _validate_path_root(self, candidate: str) -> tuple:
        """Valida um path_root candidato resolvendo a 1ª face com imagem,
        SEM mutar o dataset. Retorna (ok: bool, msg: str). Indisponível
        (retorna ok=True) quando o dataset ainda não tem pasta em disco."""
        parquet_dir = getattr(self.dataset, "_parquet_dir", None)
        if parquet_dir is None and self.current_path:
            parquet_dir = Path(self.current_path).parent
        if parquet_dir is None:
            return True, "Validação indisponível (dataset ainda não salvo)."
        faces = getattr(self.dataset, "faces", None) or []
        sample = next(
            (f for f in faces if getattr(f, "image_path", None)), None,
        )
        if sample is None:
            return True, "Nenhuma face com caminho de imagem para validar."
        resolved = resolve_image_path(
            sample,
            parquet_dir=parquet_dir,
            path_mode=PathMode.RELATIVE,
            path_root=candidate,
        )
        if resolved.exists():
            return True, f"OK — o arquivo de exemplo existe: {resolved}"
        return False, f"O arquivo de exemplo NÃO existe: {resolved}"

    def _refresh_after_path_root_change(self):
        """Refresh mínimo após mudar path_root: re-resolve imagens (só a
        RESOLUÇÃO mudou, não as células). NÃO chama
        `carregar_dataset_na_tela` — este termina em `set_dirty(False)` e
        apagaria o estado dirty que acabamos de marcar."""
        if not self.dataset:
            return
        self.detail_ctrl.set_dataset(self.dataset)
        if hasattr(self.gallery_widget, "set_dataset"):
            self.gallery_widget.set_dataset(self.dataset)
        if hasattr(self.table_ctrl, "set_dataset"):
            self.table_ctrl.set_dataset(self.dataset)
        # Reseta o painel direito (a face antes selecionada pode agora
        # resolver pra um arquivo diferente).
        self.detail_ctrl.load_face_data(None)
        # Invalida o cache de miniaturas — thumbnails são resolvidas via
        # path e mudaram de origem.
        self.gallery_widget.clear_cache()
        if self._view_stack.currentIndex() == 1:
            self.gallery_widget.populate(list(self.dataset.faces))

    def _iniciar_worker_modificar_caminhos(
        self,
        target_dir: Path,
        mode,
        naming,
        skip_existing: bool,
        old_prefix: Optional[str] = None,
        image_kind=None,
        crop_margin_factor: float = 2.0,
    ):
        # Recortar/alinhar exige detecção — o worker só materializa quando
        # o FFService está disponível (senão copia as imagens como estão).
        ff_svc = getattr(self.pipeline, "ff_service", None) if self.pipeline else None
        worker = ModifyPathsWorker(
            dataset=self.dataset,
            target_dir=target_dir,
            mode=mode,
            naming=naming,
            skip_existing=skip_existing,
            old_prefix=old_prefix,
            image_kind=image_kind,
            crop_margin_factor=crop_margin_factor,
            ff_service=ff_svc,
        )
        # Guarda contexto pra usar no on_finished
        self._localize_target_dir = target_dir
        self._localize_mode = mode
        self._localize_naming = naming
        self._run_worker(
            worker,
            label=f"Modificando caminhos: {self.dataset.name}",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,  # feedback constante (decisão #3 do usuário)
            on_finished=self._on_modificar_caminhos_concluido,
        )

    def _on_modificar_caminhos_concluido(self, result):
        """Result é `LocalizationResult` ou None (cancelado/erro)."""
        if result is None:
            QMessageBox.warning(
                self.view, "Cancelado",
                "Modificação de caminhos cancelada ou finalizada sem "
                "resultado.",
            )
            return

        # Zero faces: nada a aplicar. Trocar o dataset aberto por um vazio
        # e propor "Salvar Como" produziria um parquet de zero linhas, que
        # a UI aceita e depois some dos seletores sem explicação.
        recusa = result.refusal_message()
        if recusa is not None:
            QMessageBox.critical(self.view, "Caminhos não modificados", recusa)
            return

        # Resumo na barra de status / popup
        if result.n_failed:
            extra = (
                f"\n\n⚠ {result.n_failed} falha(s):\n{result.failures_preview()}"
            )
            # Falha parcial REMOVE faces do resultado — exige decisão.
            perdidas = result.n_processed - result.n_faces_remaining
            resp = QMessageBox.question(
                self.view, "Faces sem imagem",
                f"{result.n_failed} face(s) falharam"
                + (f" e {perdidas} ficariam de fora do dataset" if perdidas else "")
                + f".{extra}\n\nAplicar o resultado com "
                f"{result.n_faces_remaining} face(s)?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return
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
                    f"Modificação de caminhos concluída: {result.summary()}. "
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

        # Substitui o dataset corrente pelo localizado.
        # Em REBASE_PREFIX o arquivo conceitualmente é o mesmo (só os
        # paths foram reescritos), então preservamos current_path —
        # assim "Salvar" sobrescreve o parquet original e "Salvar Como"
        # permite gravar como cópia. Nos demais modos (COPY_*, REWRITE),
        # imagens foram movidas/transformadas e o original perde
        # significado — current_path=None força Save As.
        is_rebase = result.mode == LocalizationMode.REBASE_PREFIX
        self.dataset = result.dataset
        self.dataset_original = None
        if not is_rebase:
            self.current_path = None
            # O dataset autocontido só continua resolvendo se o parquet for
            # gravado na pasta onde as imagens foram copiadas. Sugere essa
            # pasta no Salvar como seguinte, em vez de deixar o perito
            # adivinhar.
            self._suggested_save_dir = str(result.target_dir)
        self.set_dirty(True)
        self.carregar_dataset_na_tela(self.dataset, path=self.current_path)

        QMessageBox.information(
            self.view, "Caminhos modificados",
            f"Dataset atualizado em memória.\n\n{result.summary()}{extra}\n\n"
            "Use 'Salvar' pra sobrescrever o arquivo original ou "
            "'Salvar Como' pra gravar com outro nome.",
        )

    # =================================================================================
    # PR I2 — FLUXO DE EDITOR DE PIPELINE DE IMAGENS
    # =================================================================================
    def fluxo_image_pipeline_editor(self):
        """Abre o ImagePipelineDialogController em modal.

        Imagem inicial: face única selecionada na tabela, ou aleatória
        quando 0 ou múltiplas seleções (decisão #2 do tema).
        """
        if not self.dataset or len(self.dataset) == 0:
            QMessageBox.information(
                self.view, "Sem dataset",
                "Abra ou crie um dataset antes de usar o editor de "
                "pipeline de imagens.",
            )
            return

        # Determina face inicial baseado na seleção da tabela.
        # Decisão #2 do tema: 1 face selecionada → abre com ela;
        # 0 ou múltiplas → aleatória (passamos -1 e o controller do
        # editor escolhe).
        initial_face_idx = -1
        try:
            selection_model = self.table_ctrl.view.selectionModel()
            if selection_model is not None:
                rows = {idx.row() for idx in selection_model.selectedIndexes()}
                if len(rows) == 1:
                    initial_face_idx = next(iter(rows))
        except Exception:
            pass  # API mudou ou estado não inicializado — cai pro aleatório

        # FFService do pipeline pai pra suportar ALIGNED_FACE +
        # CropMode.FACE_DETECTED.
        ff_svc = (
            getattr(self.pipeline, "ff_service", None)
            if self.pipeline
            else None
        )

        # PR I4 — library_root pra Salvar/Carregar pipelines da
        # biblioteca global. None desabilita esses 2 botões.
        library_root = getattr(
            getattr(self.config, "paths", None), "library", None,
        )

        # Construção do controller é pesada (carrega face inicial,
        # inicializa pipeline + view com preview). WaitCursor cobre.
        from .image_pipeline_dialog import ImagePipelineDialogController
        with wait_cursor():
            ctrl = ImagePipelineDialogController(
                dataset=self.dataset,
                initial_face_index=initial_face_idx,
                ff_service=ff_svc,
                library_root=library_root,
                parent=self.view,
            )
        ctrl.exec()

    # =================================================================================
    # PR I5 — FLUXO DE BATCH PROCESSING DE PIPELINE DE IMAGENS
    # =================================================================================
    def fluxo_aplicar_image_pipeline(self):
        """Abre o ApplyImagePipelineDialog. Após confirmação, dispara
        worker que aplica pipeline em todo dataset."""
        if not self.dataset or len(self.dataset) == 0:
            QMessageBox.information(
                self.view, "Sem dataset",
                "Abra ou crie um dataset antes de aplicar pipelines.",
            )
            return

        library_root = getattr(
            getattr(self.config, "paths", None), "library", None,
        )
        if not library_root:
            QMessageBox.warning(
                self.view, "Biblioteca indisponível",
                "Configure 'paths.library' no config.json antes de "
                "aplicar pipelines de imagens.",
            )
            return

        # Pasta sugerida: a que CONTÉM o dataset aberto. Num dataset
        # autocontido o parquet mora em `<base>/<nome>/<nome>.parquet`;
        # sugerir a pasta dele aninharia o derivado dentro da origem.
        if self.current_path:
            pasta_do_parquet = Path(self.current_path).parent
            if pasta_do_parquet.name == Path(self.current_path).stem:
                base = pasta_do_parquet.parent
            else:
                base = pasta_do_parquet
        else:
            base = Path(self._get_initial_dir() or os.getcwd())

        from .apply_image_pipeline_dialog import ApplyImagePipelineDialog
        # Construção lista pipelines disponíveis da biblioteca — I/O.
        with wait_cursor():
            dialog = ApplyImagePipelineDialog(
                library_root=Path(library_root),
                dataset_name=self.dataset.name,
                suggested_target_parent=base,
                parent=self.view,
            )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        pipeline_cfg = dialog.selected_pipeline()
        if pipeline_cfg is None:
            QMessageBox.warning(
                self.view, "Pipeline inválido",
                "Nenhum pipeline foi selecionado.",
            )
            return

        target_dir = dialog.target_dir()
        if not str(target_dir).strip() or str(target_dir) == ".":
            QMessageBox.warning(
                self.view, "Destino vazio",
                "Pasta destino não pode estar vazia.",
            )
            return

        skip_existing = dialog.skip_existing()
        output_filename = dialog.dataset_name()
        open_new_after = dialog.open_new_after()

        # Se o usuário quer abrir o derivado e o dataset atual está
        # dirty, oferece salvar antes (mesmo padrão do close).
        if open_new_after and self._is_dirty:
            res = QMessageBox.question(
                self.view, "Salvar antes?",
                "O dataset atual tem alterações não salvas. "
                "Deseja salvar antes de abrir o derivado?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if res == QMessageBox.StandardButton.Cancel:
                return
            if res == QMessageBox.StandardButton.Save:
                self.fluxo_salvar()
                if self._is_dirty:
                    return  # save falhou

        self._iniciar_worker_aplicar_image_pipeline(
            pipeline_cfg, target_dir, skip_existing,
            output_filename, open_new_after,
        )

    def _iniciar_worker_aplicar_image_pipeline(
        self, pipeline_cfg, target_dir, skip_existing: bool,
        output_filename: str, open_new_after: bool,
    ):
        from .workers.apply_image_pipeline_worker import (
            ApplyImagePipelineWorker,
        )

        ff_svc = (
            getattr(self.pipeline, "ff_service", None)
            if self.pipeline
            else None
        )

        # O derivado nasce autocontido em `<target_dir>/<nome>/`. A origem
        # vai junto para a cadeia de custódia dele registrar de onde veio.
        worker = ApplyImagePipelineWorker(
            dataset=self.dataset,
            pipeline_config=pipeline_cfg,
            base_dir=Path(target_dir),
            dataset_name=output_filename or None,
            ff_service=ff_svc,
            skip_existing=skip_existing,
            source_parquet_path=(
                Path(self.current_path) if self.current_path else None
            ),
            source_has_unsaved_changes=bool(self._is_dirty),
        )
        # Guarda contexto pra usar no on_finished
        self._batch_target_dir = Path(target_dir)
        self._batch_pipeline_name = pipeline_cfg.name
        self._batch_open_new_after = open_new_after
        self._run_worker(
            worker,
            label=(
                f"Aplicando pipeline '{pipeline_cfg.name}' em "
                f"'{self.dataset.name}'…"
            ),
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,  # feedback constante
            on_finished=self._on_aplicar_image_pipeline_concluido,
        )

    @staticmethod
    def _texto_da_comparacao_com_a_questionada(result) -> str:
        """Questionada × mediana da base adaptada, medida a medida."""
        from faceaudit.services.image_degradation_profile_service import (
            MEASURES_BY_KEY,
        )

        deg = getattr(result, "degradation", None) or {}
        questionada = deg.get("questioned")
        linhas = deg.get("comparison") or []
        if not questionada or not linhas:
            return ""
        saida = [
            f"\n\nQuestionada {questionada.get('file_name')} × base adaptada "
            "(mediana · imagens dentro da tolerância, uma a uma):"
        ]
        for linha in linhas:
            m = MEASURES_BY_KEY.get(linha["measure"])
            if m is None:
                continue
            marca = {True: "✓", False: "✗"}.get(linha.get("close"), "–")
            frac = linha.get("images_close_fraction")
            parcela = "" if frac is None else f" · {100 * frac:.0f}% das imagens"
            saida.append(
                f"  {marca} {m.label}: {m.format(linha.get('questioned'))} × "
                f"{m.format(linha.get('set_median'))}{parcela}"
            )
        conc = deg.get("agreement") or {}
        if conc:
            n_med = len(conc.get("measures", []))
            saida.append(
                f"  Imagens com as {n_med} medidas dentro: "
                f"{100 * conc['all_close_fraction']:.0f}%; com {n_med - 1} ou mais: "
                f"{100 * conc['all_but_one_close_fraction']:.0f}%."
            )
        return "\n".join(saida)

    def _on_aplicar_image_pipeline_concluido(self, result):
        if result is None:
            QMessageBox.warning(
                self.view, "Cancelado",
                "Aplicação de pipeline cancelada ou finalizada sem "
                "resultado.",
            )
            return

        # Audit log (PR 25) — registra COPY_IMAGES no parquet original
        try:
            self._audit(
                AuditOperation.COPY_IMAGES,
                (
                    f"Pipeline de imagens '{result.pipeline_name}' "
                    f"aplicado em todo o dataset. {result.summary()}"
                ),
                metadata={
                    "operation": "image_pipeline_batch",
                    "pipeline_name": result.pipeline_name,
                    "dataset_dir": str(result.dataset_dir),
                    "images_dir": str(result.images_dir),
                    "parquet_path": str(result.parquet_path),
                    "n_processed": result.n_processed,
                    "n_succeeded": result.n_succeeded,
                    "n_skipped": result.n_skipped,
                    "n_failed": result.n_failed,
                    "duration_seconds": round(result.duration_seconds, 3),
                },
            )
        except Exception:
            pass  # audit é best-effort

        # Resumo + falhas
        if result.n_failed:
            failures_preview = "\n".join(
                f"  • {p}: {msg}"
                for p, msg in list(result.failures)[:5]
            )
            extra = (
                f"\n\n⚠ {result.n_failed} falha(s):\n{failures_preview}"
                + ("\n  …" if result.n_failed > 5 else "")
            )
        else:
            extra = ""

        error_line = ""
        if result.error_parquet_path is not None:
            error_line = (
                f"\nDataset de erros: {result.error_parquet_path}\n"
                f"  → abra como dataset normal pra inspecionar a "
                f"coluna `error_message`."
            )

        QMessageBox.information(
            self.view, "Pipeline aplicado",
            f"{result.summary()}{extra}"
            f"{self._texto_da_comparacao_com_a_questionada(result)}\n\n"
            f"Dataset derivado: {result.dataset_dir}\n"
            f"  parquet: {result.parquet_path.name}\n"
            f"  imagens: {result.images_dir.name}/\n"
            f"  manifesto: {result.manifest_path.name}"
            f"{error_line}",
        )

        # Abre o derivado quando o usuário pediu. Fecha o atual sem
        # prompt (já tratamos dirty antes de iniciar o worker).
        if getattr(self, "_batch_open_new_after", False):
            try:
                # Reusa o caminho normal de open — fluxo idêntico a
                # clicar Abrir Dataset apontando pro novo parquet.
                self._run_worker(
                    LoadDatasetWorker(str(result.parquet_path)),
                    label="Abrindo dataset derivado…",
                    allow_cancel=False,
                    determinate=False,
                    wait_cursor=True,
                    on_finished=(
                        lambda ds: self._on_load_success(
                            ds, str(result.parquet_path),
                        )
                    ),
                )
            except Exception as e:
                QMessageBox.warning(
                    self.view, "Não foi possível abrir",
                    f"O derivado foi salvo, mas falhou abrir: {e}\n"
                    f"Use 'Abrir Dataset' apontando pra "
                    f"{result.parquet_path}.",
                )
            finally:
                self._batch_open_new_after = False

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
            try:
                analysis_ctrl.exec()
            finally:
                # O que ficou aplicado ao fechar vai para a cadeia — mesmo
                # se o diálogo fechou pelo X ou por exceção.
                self._registrar_acoes_da_auditoria(analysis_ctrl)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Falha ao abrir Auditoria: {e}")

    def _registrar_acoes_da_auditoria(self, analysis_ctrl) -> None:
        """Registra na cadeia as ações do diálogo Auditar que ficaram aplicadas.

        O Auditar tem desfazer (Ctrl+Z e ↶ por linha, inclusive no meio da
        pilha). Registrar no ato e registrar cada desfazer encheria a cadeia
        de afirmações que se anulam e nenhuma chega ao arquivo; registrar só
        o saldo ao fechar faz a cadeia dizer exatamente o que foi aplicado
        ao dataset — cada ação com a própria justificativa e o instante em
        que foi aplicada (`applied_at`), e a sessão diz quantas foram
        desfeitas. Exclusões → REMOVE_FACES; renomear/mesclar → EDIT_LABEL.

        Com filtro ativo, a exclusão também sai do dataset de origem (como
        na Tabela/Galeria) — senão "Limpar filtro" desfaria em silêncio a
        exclusão que a cadeia acabou de afirmar.
        """
        try:
            acoes = analysis_ctrl.applied_actions()
        except Exception:
            return
        if not acoes:
            return
        sessao = {
            "origin": "audit_dialog",
            "audit_session_id": getattr(analysis_ctrl, "session_id", None),
            "audit_session_actions_applied": len(acoes),
            "audit_session_actions_undone": int(analysis_ctrl.undone_count()),
        }
        original = getattr(self, "dataset_original", None)
        for acao in acoes:
            try:
                extra = {**sessao, "applied_at": acao.applied_at}
                if acao.kind == "delete":
                    if original is not None and original is not self.dataset:
                        ids = {id(f) for f in acao.deleted_faces}
                        original.faces = [f for f in original.faces if id(f) not in ids]
                    descricao, meta = custody.removal_entry(
                        acao.deleted_faces, acao.justification,
                        origin="audit_dialog", extra=extra,
                    )
                    self._audit(AuditOperation.REMOVE_FACES, descricao, metadata=meta)
                elif acao.kind in ("rename", "merge"):
                    descricao, meta = custody.identity_edit_entry(
                        action=(
                            "rename_identity" if acao.kind == "rename"
                            else "merge_identities"
                        ),
                        old_identities=acao.old_identities,
                        new_identity=acao.new_identity,
                        changed_identifiers=acao.changed_identifiers,
                        justification=acao.justification,
                        extra=extra,
                    )
                    self._audit(AuditOperation.EDIT_LABEL, descricao, metadata=meta)
            except Exception as e:
                import sys as _sys
                print(f"[AuditLog] Falha ao registrar ação do Auditar: {e}",
                      file=_sys.stderr)

    def _on_dataset_modified_by_audit(self) -> None:
        """Hook chamado pelo DatasetAnalysisController quando faces foram
        deletadas/modificadas via o painel de auditoria."""
        if self.dataset is None:
            return
        self.table_ctrl.load_dataframe(self.dataset.to_pandas(include_embeddings=False))
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

    _AMOSTRA_VERIFICACAO_MERGE = 200

    def fluxo_merge(self):
        if not self.dataset:
            return
        self.sync_changes_from_table()
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Juntar com...", self._get_initial_dir(),
            "Parquet (*.parquet)",
        )
        if not path:
            return

        with wait_cursor():
            other = FaceDataset.from_parquet(path)

        # Misturar faces alinhadas com fotos originais num mesmo dataset é
        # armadilha silenciosa: o dataset fica marcado como não-alinhado, e
        # num enriquecimento futuro os recortes 112×112 passam pelo detector,
        # que costuma falhar neles. Pergunta ANTES, porque depois de unido
        # não há como separar sem refazer.
        alinhamento_misto = bool(self.dataset.is_aligned) != bool(other.is_aligned)
        if alinhamento_misto:
            alinhado = ("o atual" if self.dataset.is_aligned else "o escolhido")
            resposta = QMessageBox.question(
                self.view, "Tipos de imagem diferentes",
                f"Apenas {alinhado} contém faces já alinhadas (112×112); o "
                f"outro contém fotos originais.\n\n"
                f"O dataset unido ficará marcado como NÃO alinhado, e num "
                f"enriquecimento futuro os recortes alinhados passarão pelo "
                f"detector, que costuma falhar neles.\n\nUnir mesmo assim?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resposta != QMessageBox.StandardButton.Yes:
                return

        with wait_cursor():
            base = self.dataset
            new_ds = base.concat(other)
            self._registrar_juntar(
                new_ds, base, other, path,
                alinhamento_misto=alinhamento_misto,
            )
            ausentes, verificadas = self._amostrar_imagens_ausentes(new_ds)
            # O unido é um dataset por si: o estado de filtro da base não
            # vale para ele ("Limpar filtro" traria a base de volta por cima).
            self.dataset_original = None
            self.current_filter_config = FilterConfig()
            self.carregar_dataset_na_tela(new_ds)

        msg = f"Datasets unidos. Total: {len(new_ds)} imagens."
        # Compara com a BASE: depois de carregar, `self.dataset` já é o unido.
        if new_ds.path_mode is not base.path_mode:
            # Explica a mudança de modo antes que o usuário a descubra ao
            # mover a pasta: os dois lados tinham âncoras diferentes e os
            # caminhos foram resolvidos para absolutos.
            msg += (
                "\n\nOs dois datasets apontavam para pastas de imagens "
                "diferentes, então os caminhos do resultado foram gravados "
                "como absolutos. Para voltar a ter um dataset portátil, use "
                "\"Modificar Caminhos\" ou salve como autocontido."
            )
        if ausentes:
            proporcao = ausentes / max(verificadas, 1)
            estimativa = int(round(proporcao * len(new_ds)))
            QMessageBox.warning(
                self.view, "Imagens não encontradas",
                f"{msg}\n\nATENÇÃO: em uma amostra de {verificadas} "
                f"imagens, {ausentes} não foram encontradas no disco "
                f"(cerca de {estimativa} do total).\n\nIsso costuma "
                f"significar que a pasta de imagens de um dos datasets foi "
                f"movida ou renomeada. Confira antes de usar este dataset.",
            )
        else:
            QMessageBox.information(self.view, "Juntar", msg)
        self.set_dirty(True)

    def _descrever_origem(self, dataset, path, *, sem_alteracoes: bool, conteudo=None) -> dict:
        """Descrição da origem de um derivado (arquivo, SHA-256, contagens)."""
        return custody.describe_source(
            dataset, path,
            has_unsaved_changes=not sem_alteracoes,
            content=conteudo,
        )

    def _registrar_juntar(self, unido, base, outro, outro_path, *, alinhamento_misto: bool) -> None:
        """MERGE na cadeia do dataset unido.

        O unido nasce em memória, sem arquivo: a entrada fica pendente nele
        e é escrita na cadeia do arquivo quando ele for gravado (Salvar
        Como), antes do SAVE_AS. Herda também as entradas ainda pendentes
        da base — se ela própria nunca foi gravada, é a única história que
        ela tem. Best-effort: falha de registro não desfaz a união.
        """
        try:
            base_desc = self._descrever_origem(
                base, self.current_path,
                sem_alteracoes=bool(self.current_path) and not self.is_dirty(),
                conteudo=self._descrever_conteudo_salvo(base),
            )
            outro_desc = self._descrever_origem(outro, outro_path, sem_alteracoes=True)
            descricao, meta = custody.merge_entry(
                base_desc, outro_desc, unido,
                mixed_alignment=alinhamento_misto,
                # Só se chega aqui com alinhamento misto se o perito disse Sim.
                mixed_alignment_confirmed=alinhamento_misto,
            )
            custody.inherit_pending(unido, base, getattr(self, "dataset_original", None))
            custody.queue_pending(unido, AuditOperation.MERGE, descricao, meta)
        except Exception as e:
            import sys as _sys
            print(f"[AuditLog] Falha ao registrar MERGE: {e}", file=_sys.stderr)

    def _registrar_divisao(self, config, partes) -> None:
        """SPLIT na cadeia de cada metade gravada.

        `partes` = [(rótulo, dataset, path), ...] — as duas metades já em
        disco. Cada uma recebe as entradas ainda pendentes da origem (se a
        origem nunca foi gravada) e a sua entrada SPLIT com o hash do
        próprio arquivo. Best-effort: falha de registro não desfaz a divisão.
        """
        try:
            origem = self.dataset
            fonte = self._descrever_origem(
                origem, self.current_path,
                sem_alteracoes=bool(self.current_path) and not self.is_dirty(),
                conteudo=self._descrever_conteudo_salvo(origem),
            )
            modo = config.get("mode")
            semente = (
                config.get("random_seed", custody.default_split_seed())
                if modo == "ratio" else None
            )
            pendentes = custody.pending_entries(
                origem, getattr(self, "dataset_original", None),
            )
            for i, (rotulo, parte, caminho) in enumerate(partes):
                _r, irma, caminho_irma = partes[1 - i]
                if pendentes:
                    custody.write_pending(caminho, pendentes)
                descricao, meta = custody.split_entry(
                    fonte, part_label=rotulo, part=parte,
                    sibling=irma, sibling_path=caminho_irma,
                    mode=modo, ratio=config.get("ratio"), random_seed=semente,
                    ids_for_b=config.get("manual_ids_b"),
                )
                self._audit(
                    AuditOperation.SPLIT, descricao, metadata=meta,
                    path_override=str(caminho), hash_file=True,
                )
        except Exception as e:
            import sys as _sys
            print(f"[AuditLog] Falha ao registrar SPLIT: {e}", file=_sys.stderr)

    @classmethod
    def _amostrar_imagens_ausentes(cls, dataset) -> tuple:
        """(ausentes, verificadas) numa amostra. Nunca varre tudo.

        Conferir existência é I/O por face, e em dataset de milhões seria
        inviável — em pasta sincronizada por nuvem, pior ainda. A amostra
        responde à pergunta que importa, que é se a união quebrou os
        caminhos, sem travar a interface.
        """
        from pathlib import Path as _Path
        import random

        faces = getattr(dataset, "faces", None) or []
        if not faces:
            return 0, 0
        amostra = (
            faces if len(faces) <= cls._AMOSTRA_VERIFICACAO_MERGE
            else random.sample(faces, cls._AMOSTRA_VERIFICACAO_MERGE)
        )
        ausentes = 0
        for face in amostra:
            try:
                if not _Path(dataset.resolve_image_path(face)).is_file():
                    ausentes += 1
            except Exception:
                ausentes += 1
        return ausentes, len(amostra)

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
                
                # As metades referenciam as imagens onde estão: o backend
                # reaponta `path_root` (ou absolutiza) para que continuem
                # resolvendo na pasta de destino, e recusa metade vazia.
                from faceaudit.services.dataset_split_save_service import (
                    EmptySplitHalfError, save_split_halves,
                )
                try:
                    salvo_a, salvo_b = save_split_halves(
                        ds_a, ds_b, self.pipeline.storage_path,
                    )
                except EmptySplitHalfError as e:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(self.view, "Divisão não gravada", str(e))
                    return
                path_a, path_b = salvo_a.path, salvo_b.path
                # Cadeia de custódia de cada metade (origem, critério, contagens).
                self._registrar_divisao(
                    config, [("A", salvo_a.dataset, path_a), ("B", salvo_b.dataset, path_b)]
                )

                QApplication.restoreOverrideCursor()

                avisos = "".join(
                    f"\n\n{s.dataset.name}: {s.rebase.message}"
                    for s in (salvo_a, salvo_b)
                    if s.rebase is not None and s.rebase.changed
                )
                QMessageBox.information(
                    self.view,
                    "Sucesso",
                    f"Divisão concluída com sucesso!\n\nSalvos em:\n- {path_a}\n- {path_b}"
                    f"{avisos}"
                )

                action = config["post_action"]
                if action == 1:
                    self.carregar_dataset_na_tela(salvo_a.dataset, path=str(path_a))
                elif action == 2:
                    self.carregar_dataset_na_tela(salvo_b.dataset, path=str(path_b))
                
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self.view, "Erro", f"Erro durante a divisão do dataset:\n{e}")

    def fluxo_filtrar(self):
        if not self.dataset: return
        # IMPORTANTE: setOverrideCursor PRIMEIRO + processEvents pra
        # forçar o repaint do cursor ANTES de qualquer trabalho síncrono.
        # `sync_changes_from_table()` em datasets grandes (1M+ rows)
        # bloqueia a main thread por segundos — sem o defer abaixo, o
        # cursor só virava ampulheta lá no fim, parecendo que o clique
        # não foi recebido.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()  # repaint imediato do cursor
        # Defer TODO o trabalho síncrono pro próximo tick — sync + abrir
        # dialog vão acontecer com a ampulheta já visível.
        QTimer.singleShot(0, self._abrir_dialogo_filtro_continuar)

    def _abrir_dialogo_filtro_continuar(self):
        # PR 21 (Tema 4.2) — sincroniza edições in-table (incluindo
        # colunas custom no model._df) pra `dataset.faces[i].extra_data`
        # antes de abrir o dialog. Sem isso, custom columns adicionadas/
        # editadas via UI ficariam só no model e seriam invisíveis pra
        # `apply_filter`.
        self.sync_changes_from_table()
        if not self.dataset_original:
            self.dataset_original = self.dataset
        self._abrir_dialogo_filtro_real()

    def _abrir_dialogo_filtro_real(self):
        try:
            # library_root vem de app_config.paths.library — usado pelo
            # filter_dialog pra Salvar/Abrir filtros na biblioteca global.
            library_root = getattr(
                getattr(self.config, "paths", None), "library", None,
            )
            filter_ctrl = FilterDialogController(
                self.dataset_original,
                self.current_filter_config,
                parent=self.view,
                library_root=library_root,
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
                self.table_ctrl.load_dataframe(self.dataset.to_pandas(include_embeddings=False))
                # Propaga o NOVO dataset filtrado pros widgets que
                # resolvem paths/atributos contra ele. Sem isso, o
                # detail_panel/gallery continuam apontando pro dataset
                # original (e a face selecionada antes do filtro pode
                # nem estar mais no resultado).
                self.detail_ctrl.set_dataset(self.dataset)
                # Reseta a face exibida no painel direito — sem isso
                # o usuário vê uma face que pode nem estar no
                # resultado filtrado (ex: filtrou só M e ficou
                # mostrando a F que estava selecionada antes).
                self.detail_ctrl.load_face_data(None)
                # Idem pra galeria
                if hasattr(self.gallery_widget, "set_dataset"):
                    self.gallery_widget.set_dataset(self.dataset)
                self.gallery_widget.clear_cache()
                if self._view_stack.currentIndex() == 1:
                    self.gallery_widget.populate(list(self.dataset.faces))
                else:
                    self.gallery_widget.populate([])
                text, tooltip = self._build_dataset_status(filtered=True)
                self.toolbar_ctrl.update_filter_status(
                    text, is_filtered=True, tooltip=tooltip,
                )
            else:
                QMessageBox.warning(self.view, "Filtro", "Nenhum resultado encontrado com estes critérios.")
        except Exception as e:
            QMessageBox.critical(self.view, "Erro", f"Erro ao aplicar filtro: {e}")

    def fluxo_limpar_filtro(self):
        if not self.dataset_original:
            return
        # Recarregar a tabela com o dataset original pode levar segundos
        # em datasets grandes (BFW 20k → 1-2s; webface 1M → minutos).
        # WaitCursor cobre o trabalho síncrono na main thread.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.dataset = self.dataset_original
            self.dataset_original = None
            self.current_filter_config = FilterConfig()
            self.carregar_dataset_na_tela(self.dataset, self.current_path)

            text, tooltip = self._build_dataset_status(filtered=False)
            self.toolbar_ctrl.update_filter_status(
                text, is_filtered=False, tooltip=tooltip,
            )
        finally:
            QApplication.restoreOverrideCursor()

    def fluxo_enriquecer(self):
        if not self.dataset: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(10, self._abrir_enrich_real)

    def _abrir_enrich_real(self):
        try:
            self.sync_changes_from_table()
            
            ff_cfg = getattr(self.config, "ff", None)
            enrich_ctrl = EnrichDatasetController(
                self.dataset,
                parent=self.view,
                ff_available=self.ff_available,
                ofiq_available=self.ofiq_available,
                default_use_gpu=bool(getattr(ff_cfg, "use_gpu", False)),
                default_batch_size=int(getattr(ff_cfg, "batch_size", 32)),
            )
            
            QApplication.restoreOverrideCursor()
            
            if enrich_ctrl.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                config = enrich_ctrl.get_config()
                self._iniciar_worker_enrich(config)
                
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Erro ao abrir janela de enriquecimento: {e}")

    def _iniciar_worker_enrich(self, config: DatasetIngestionConfig):
        # A gravação pode acontecer numa segunda etapa, depois da revisão
        # das faces que ficaram sem biometria.
        self._enrich_config = config
        self._run_worker(
            EnrichWorker(self.pipeline, self.dataset, config),
            label="Enriquecendo Dataset",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,
            on_finished=self._on_enrich_concluido,
        )

    def _on_enrich_concluido(self, dataset):
        # Enriquecimento parado antes de gravar porque há faces sem
        # biometria a revisar. A tela de progresso já fechou.
        if dataset is not None and getattr(dataset, "_save_deferred", False):
            self._revisar_sem_biometria_e_gravar(dataset)
            return
        self._concluir_enriquecimento(dataset)

    def _revisar_sem_biometria_e_gravar(self, dataset):
        """Mostra as faces sem biometria e grava depois.

        Aqui a revisão NÃO decide inclusão: as faces já estão no dataset, e
        removê-las seria edição destrutiva com caminho próprio no aplicativo
        (Excluir selecionados, que registra justificativa). O que esta
        janela oferece é inspecionar e tentar numa resolução maior.
        """
        from faceaudit.enums.undetected_review_mode import UndetectedReviewMode

        from .undetected_review_dialog import UndetectedReviewController

        pendentes = list(getattr(dataset, "_undetected_faces", []) or [])
        config = getattr(self, "_enrich_config", None)
        if not pendentes or config is None:
            self._finalizar_enriquecimento_e_gravar(dataset, [])
            return

        ctrl = UndetectedReviewController(
            self.view,
            faces=pendentes,
            config=config,
            dataset=dataset,
            n_total=len(dataset),
            ff_service_factory=self._fabrica_de_ff_service(config),
            modo=UndetectedReviewMode.ENRICHMENT,
        )
        ctrl.exec()
        self._finalizar_enriquecimento_e_gravar(dataset, ctrl.recovered_faces())

    def _finalizar_enriquecimento_e_gravar(self, dataset, recuperadas):
        from .workers.finalize_dataset_worker import FinalizeDatasetWorker

        self._revisao_recuperadas = list(recuperadas or [])
        self._run_worker(
            FinalizeDatasetWorker(
                self.pipeline, dataset, self._enrich_config,
                recovered_faces=self._revisao_recuperadas,
                enrichment=True,
            ),
            label="Gravando Enriquecimento",
            determinate=True,
            use_log_dialog=True,
            on_finished=self._concluir_enriquecimento,
        )

    def _concluir_enriquecimento(self, dataset):
        if dataset:
            # Reusa `carregar_dataset_na_tela` pra refresh completo:
            # propaga column_specs atualizados (que agora incluem
            # embedding_v2/fiqa_score/ofiq_*/serfiq_score) pro table
            # model + delegates, repopula gallery, e propaga o dataset
            # pros widgets que resolvem paths em modo RELATIVE. Sem
            # isso, o "Mostrar/Ocultar Colunas" ficava cego pras
            # colunas adicionadas pelo enrichment.
            self.carregar_dataset_na_tela(dataset, self.current_path)
            # PR 25 (Tema 5.4) — registra ENRICH (sem hash; o parquet
            # já foi resalvo pelo enrich_existing_dataset, mas a próxima
            # save manual rehasha).
            recuperadas = getattr(self, "_revisao_recuperadas", None) or []
            descricao = f"Enriquecimento aplicado em '{dataset.name}'."
            meta = {
                "rows": len(self.dataset),
                "identities": self.dataset.number_of_identities,
            }
            if recuperadas:
                tamanhos = sorted({
                    int(getattr(
                        getattr(f, "ff_info", None), "detection_det_size", 0,
                    ) or 0)
                    for f in recuperadas
                })
                descricao += (
                    f" {len(recuperadas)} face(s) sem biometria foram "
                    f"recuperadas numa segunda tentativa, com o detector em "
                    f"{', '.join(str(t) for t in tamanhos if t)} px."
                )
                meta["recovered_on_retry"] = len(recuperadas)
                meta["recovered_files"] = self._nomes_de_arquivo(recuperadas)
                meta["recovered_det_sizes"] = tamanhos
            n_giradas = self._contar_giradas(dataset)
            if n_giradas:
                meta["faces_detected_after_rotation"] = n_giradas
            from faceaudit.services.ofiq_provenance_service import mixed_versions_message
            meta.update(self._metadados_do_ofiq(
                dataset,
                rodou=getattr(self._enrich_config, "compute_ofiq", False),
            ))
            aviso_ofiq = mixed_versions_message(meta.get("ofiq_versions") or {})
            if aviso_ofiq:
                descricao += " " + aviso_ofiq
            self._audit(AuditOperation.ENRICH, descricao, metadata=meta)
            # Consumido: não pode vazar para a próxima operação.
            self._revisao_recuperadas = []
            # `enrich_existing_dataset` SOBRESCREVE o parquet em disco
            # diretamente — não passa por `fluxo_salvar`. Pra que o
            # Strategy Manager V2 (cross-tab) detecte a mudança e
            # propague atualização de hash + cascade dos derivados,
            # disparamos o signal aqui manualmente. Crítico pro fluxo
            # de enriquecimento (caso mais comum de "dataset modificado
            # externamente"): sem isso, derivados continuariam ✅
            # apontando pra entrada de uma versão pré-enriquecimento.
            if self.current_path:
                self.dataset_saved_to_project.emit(str(self.current_path))
            if aviso_ofiq:
                # Concluiu, mas o dataset ficou heterogêneo: dizer na hora,
                # e não deixar só na custódia, onde ninguém olha antes do laudo.
                QMessageBox.warning(
                    self.view, "Enriquecimento concluído",
                    "Enriquecimento concluído.\n\n" + aviso_ofiq,
                )
            else:
                QMessageBox.information(self.view, "Sucesso", "Enriquecimento concluído com sucesso!")
            self.set_dirty(True)
        else:
            QMessageBox.warning(self.view, "Cancelado", "Operação cancelada ou finalizada sem dados.")

    # ── Balanceamento por identidade (botão "Balancear" da toolbar) ──
    def fluxo_balancear(self):
        if not self.dataset:
            return
        with wait_cursor():
            self.sync_changes_from_table()
            from .balance_dataset_dialog import BalanceDatasetDialog
            dialog = BalanceDatasetDialog(
                dataset_name=self.dataset.name or "Sem Nome",
                parent=self.view,
            )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        from .workers.balance_dataset_worker import BalanceDatasetWorker
        worker = BalanceDatasetWorker(
            dataset=self.dataset,
            mode=dialog.mode(),
            target_n=dialog.target_n(),
            selection=dialog.selection(),
            random_seed=dialog.random_seed(),
            quality_metric=dialog.quality_metric(),
            output_name=None,  # service usa default <name>_Balanced
        )
        self._balance_replace_current = dialog.replace_current()
        self._run_worker(
            worker,
            label="Balanceando dataset…",
            allow_cancel=True,
            determinate=False,
            use_log_dialog=False,
            on_finished=self._on_balance_concluido,
        )

    def _on_balance_concluido(self, result):
        if result is None:
            QMessageBox.warning(
                self.view, "Cancelado",
                "Balanceamento cancelado ou finalizado sem resultado.",
            )
            return

        # `result` é um DatasetBalancingResult; `.dataset` é o
        # FaceDataset balanceado.
        balanced = result.dataset
        summary_extra = []
        if result.discarded_identities > 0:
            summary_extra.append(
                f"{result.discarded_identities} ID(s) descartada(s) "
                f"(FLOOR_AND_CAP)"
            )
        if result.quality_fallback_count > 0:
            summary_extra.append(
                f"{result.quality_fallback_count} sample(s) sem métrica "
                "(usados em ordem original)"
            )
        extra_msg = (
            "\n\n⚠ " + " · ".join(summary_extra) if summary_extra else ""
        )

        replace = getattr(self, "_balance_replace_current", False)
        try:
            self._balance_replace_current = False
            if replace:
                # Substitui o dataset atual pelo balanceado, marcando
                # dirty (perito decide se Salva ou Salva Como).
                self.carregar_dataset_na_tela(
                    balanced, path=None,
                )
                self.set_dirty(True)
                QMessageBox.information(
                    self.view, "Balanceamento aplicado",
                    f"Dataset substituído pelo balanceado.\n\n"
                    f"{result.summary()}{extra_msg}\n\n"
                    f"Use 'Salvar Como' pra gravar como novo parquet "
                    f"(o original em disco continua intacto)."
                )
            else:
                # Modo "manter atual": só mostra o resumo, não muta a
                # tela. Útil pra ver o impacto sem comprometer o
                # dataset corrente.
                QMessageBox.information(
                    self.view, "Pré-visualização de balanceamento",
                    f"Balanceamento simulado (não aplicado ao dataset "
                    f"corrente):\n\n{result.summary()}{extra_msg}\n\n"
                    f"Marque 'Substituir o dataset corrente' no "
                    f"diálogo se quiser aplicar.",
                )
        except Exception as e:
            QMessageBox.critical(
                self.view, "Erro ao aplicar balanceamento", str(e),
            )