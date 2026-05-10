"""ImagePipelineDialogController — orquestra estado do editor (PR I2).

Mantém em RAM:
  - `_pipeline_config`: o pipeline em construção
  - `_current_image_idx`: posição atual no dataset (pra navegação)
  - `_current_step_idx`: step selecionado na lista (pra config panel)
  - `_cached_original`: ndarray BGR da imagem original carregada
    (cv2.imread + ALIGNED_FACE via FFService quando aplicável)
  - `_cached_processed`: ndarray BGR após `pipeline.process_image`

Decisões de design:
  - Pipeline em RAM até o usuário fechar (sem auto-save em I2).
  - Aplicar é síncrono no I2 — performance OK pra steps simples.
    Worker thread entra em I3 quando scores de qualidade exigirem.
  - Imagem carregada via `cv2.imread` direto (já temos cv2 instalado).
    ALIGNED_FACE delega ao `FFService.align_face_only` (PR L3).
"""

from __future__ import annotations

import random as random_module
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.enums.image_input_source import ImageInputSource
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.image_processing.blur_config import BlurConfig
from faceaudit.models.image_processing.crop_config import CropConfig
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.jpeg_compression_config import (
    JpegCompressionConfig,
)
from faceaudit.models.image_processing.occlusion_config import OcclusionConfig
from faceaudit.models.image_processing.webp_compression_config import (
    WebpCompressionConfig,
)
from faceaudit.pipelines.image_processing_pipeline import (
    ImageProcessingPipeline,
)

from .image_pipeline_dialog_view import ImagePipelineDialogView
from .workers.quality_scores_worker import QualityScoresWorker


# Defaults pra steps recém-adicionados — refletem casos típicos
# (não aleatórios) pra que o usuário veja efeito imediatamente após
# adicionar.
_DEFAULT_FACTORIES = {
    ImageProcessingStepType.BLUR: lambda: BlurConfig(kernel_size=7, sigma=2.0),
    ImageProcessingStepType.JPEG: lambda: JpegCompressionConfig(quality=50),
    ImageProcessingStepType.WEBP: lambda: WebpCompressionConfig(quality=50),
    ImageProcessingStepType.CROP: lambda: CropConfig(),
    ImageProcessingStepType.OCCLUSION: lambda: OcclusionConfig(),
}


class ImagePipelineDialogController:
    """Entry point da janela do editor.

    Construa com `dataset` (FaceDataset corrente) +
    `initial_face_index` (-1 = aleatória) + `ff_service` (do
    pipeline pai).
    """

    def __init__(
        self,
        dataset: FaceDataset,
        initial_face_index: int = -1,
        ff_service=None,
        library_root: Optional[Path] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        self._dataset = dataset
        self._ff_service = ff_service
        # PR I4 — root da biblioteca global. None = save/load
        # desabilitados (config sem `paths.library`).
        self._library_root: Optional[Path] = (
            Path(library_root) if library_root else None
        )

        self._pipeline_config = ImageProcessingPipelineConfig()
        self._current_step_idx: int = -1
        # PR I4 — dirty tracking. Setado em todas as mudanças do
        # pipeline; limpado em save/load.
        self._is_dirty: bool = False

        # Imagem corrente
        if not dataset or len(dataset) == 0:
            self._current_image_idx = -1
        elif initial_face_index >= 0 and initial_face_index < len(dataset):
            self._current_image_idx = initial_face_index
        else:
            self._current_image_idx = random_module.randrange(len(dataset))

        self._cached_original: Optional[np.ndarray] = None
        self._cached_processed: Optional[np.ndarray] = None

        # PR I3 — worker thread pra scores de qualidade. Mantém
        # referências fortes pra evitar GC durante a execução.
        self._quality_thread: Optional[QtCore.QThread] = None
        self._quality_worker: Optional[QualityScoresWorker] = None
        # Threads ainda rodando que foram "canceladas" (resultado
        # ignorado). Precisam continuar com referência Python viva
        # até `thread.finished` disparar — caso contrário, o sip
        # wrapper destrói o QThread ainda em execução e o Qt aborta
        # com qFatal("QThread: Destroyed while thread is still running").
        self._pending_quality_threads: List[
            tuple[QtCore.QThread, QualityScoresWorker]
        ] = []

        # View
        self.view = ImagePipelineDialogView(parent=parent)
        self._wire_signals()
        self._sync_initial_state()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def exec(self) -> int:
        """Mostra o dialog modal. Retorna o exit code do QDialog."""
        return self.view.exec()

    # ── Wiring ─────────────────────────────────────────────────────────────

    def _wire_signals(self):
        # Top meta
        self.view.name_changed.connect(self._on_name_changed)
        self.view.input_source_changed.connect(self._on_input_source_changed)
        self.view.seed_changed.connect(self._on_seed_changed)

        # PR I4 — Save/Load + close hook
        self.view.save_clicked.connect(self._on_save_clicked)
        self.view.load_clicked.connect(self._on_load_clicked)
        self.view.confirm_close_callback = self._on_close_attempt

        # Steps list
        self.view.steps_list.step_added.connect(self._on_step_added)
        self.view.steps_list.step_removed.connect(self._on_step_removed)
        self.view.steps_list.selection_changed.connect(
            self._on_step_selection_changed
        )
        self.view.steps_list.steps_reordered.connect(
            self._on_steps_reordered
        )

        # Config panel
        self.view.step_config.config_changed.connect(
            self._on_step_config_changed
        )
        # PR I6 — botão "Desenhar máscara" no form de OCCLUSION
        self.view.step_config.mask_canvas_requested.connect(
            self._on_mask_canvas_requested
        )

        # Image compare
        self.view.image_compare.prev_clicked.connect(self._on_prev)
        self.view.image_compare.next_clicked.connect(self._on_next)
        self.view.image_compare.random_clicked.connect(self._on_random)
        self.view.image_compare.apply_clicked.connect(self._on_apply)

    def _sync_initial_state(self):
        # Topo
        self.view.set_pipeline_meta(
            self._pipeline_config.name,
            self._pipeline_config.input_source,
            self._pipeline_config.seed,
        )
        # PR I4 — habilita/desabilita Save/Load conforme library_root
        if self._library_root is None:
            self.view.set_save_load_enabled(
                False,
                "biblioteca global não configurada "
                "(config.paths.library)",
            )
        else:
            self.view.set_save_load_enabled(True)

        # Lista
        self.view.steps_list.populate(
            [s.step_type for s in self._pipeline_config.steps]
        )
        # Config panel (lista vazia → nada selecionado)
        self.view.step_config.show_config(None)

        # Imagem inicial
        if self._current_image_idx >= 0:
            self._reload_current_image()
        else:
            self.view.image_compare.show_index(0, 0, None)
            self.view.image_compare.set_navigation_enabled(False)

    # ── Top meta ────────────────────────────────────────────────────────────

    def _on_name_changed(self, name: str):
        new = name or "untitled"
        if new != self._pipeline_config.name:
            self._pipeline_config.name = new
            self._mark_dirty()

    def _on_input_source_changed(self, src: ImageInputSource):
        if src != self._pipeline_config.input_source:
            self._pipeline_config.input_source = src
            self._mark_dirty()
        # Re-carrega a imagem corrente (mudou o que mostrar)
        self._reload_current_image()

    def _on_seed_changed(self, seed: int):
        if int(seed) != self._pipeline_config.seed:
            self._pipeline_config.seed = int(seed)
            self._mark_dirty()
        # Não força re-aplicação — usuário clica Aplicar quando quiser

    # ── Steps ──────────────────────────────────────────────────────────────

    def _on_step_added(self, step_type: ImageProcessingStepType):
        new_config = _DEFAULT_FACTORIES[step_type]()
        self._pipeline_config.steps.append(new_config)
        new_idx = len(self._pipeline_config.steps) - 1
        self.view.steps_list.populate(
            [s.step_type for s in self._pipeline_config.steps]
        )
        self.view.steps_list.select_index(new_idx)
        self._mark_dirty()

    def _on_step_removed(self, idx: int):
        if 0 <= idx < len(self._pipeline_config.steps):
            self._pipeline_config.steps.pop(idx)
            self.view.steps_list.populate(
                [s.step_type for s in self._pipeline_config.steps]
            )
            # Mantém seleção válida
            if self._pipeline_config.steps:
                new_idx = min(idx, len(self._pipeline_config.steps) - 1)
                self.view.steps_list.select_index(new_idx)
            else:
                self._current_step_idx = -1
                self.view.step_config.show_config(None)
            self._mark_dirty()

    def _on_step_selection_changed(self, idx: int):
        self._current_step_idx = idx
        if 0 <= idx < len(self._pipeline_config.steps):
            self.view.step_config.show_config(
                self._pipeline_config.steps[idx]
            )
        else:
            self.view.step_config.show_config(None)

    def _on_steps_reordered(
        self, new_order: List[ImageProcessingStepType],
    ):
        """Lista foi reordenada via drag-and-drop. Reconstrói
        `_pipeline_config.steps` na nova ordem usando matching por
        identidade (cada step_type pode aparecer múltiplas vezes —
        precisa preservar a config DTO específica de cada um).
        """
        # Estratégia: pega os configs antigos, distribui pra nova
        # ordem na sequência de occurrence de cada step_type.
        old = list(self._pipeline_config.steps)
        new_steps = []
        used = [False] * len(old)
        for st in new_order:
            for i, oc in enumerate(old):
                if not used[i] and oc.step_type == st:
                    new_steps.append(oc)
                    used[i] = True
                    break
        self._pipeline_config.steps = new_steps
        self._mark_dirty()

    def _on_step_config_changed(
        self, _config: ImageProcessingStepConfig,
    ):
        # `_config` é o mesmo objeto já dentro de `_pipeline_config.steps`
        # (form mutates in place). Marca dirty pra UX consistente.
        self._mark_dirty()

    # ── Navegação de imagem ────────────────────────────────────────────────

    def _on_prev(self):
        if not self._dataset or len(self._dataset) == 0:
            return
        self._current_image_idx = (
            (self._current_image_idx - 1) % len(self._dataset)
        )
        self._reload_current_image()

    def _on_next(self):
        if not self._dataset or len(self._dataset) == 0:
            return
        self._current_image_idx = (
            (self._current_image_idx + 1) % len(self._dataset)
        )
        self._reload_current_image()

    def _on_random(self):
        if not self._dataset or len(self._dataset) == 0:
            return
        self._current_image_idx = random_module.randrange(len(self._dataset))
        self._reload_current_image()

    # ── Carregamento + processamento ───────────────────────────────────────

    def _reload_current_image(self):
        """Lê a imagem corrente do disco (ou align_face_only) e
        atualiza o painel esquerdo. Limpa o painel processada (precisa
        clicar Aplicar pra atualizar).
        """
        if (
            not self._dataset
            or self._current_image_idx < 0
            or self._current_image_idx >= len(self._dataset)
        ):
            self._cached_original = None
            self._cached_processed = None
            self.view.image_compare.show_original(None)
            self.view.image_compare.show_processed(None)
            self.view.image_compare.show_status("Sem imagem disponível.", is_error=True)
            return

        face = self._dataset.faces[self._current_image_idx]

        # Resolve path absoluto
        try:
            img_path = self._dataset.resolve_image_path(face)
        except Exception as e:
            self._cached_original = None
            self.view.image_compare.show_status(
                f"Erro ao resolver path: {e}", is_error=True,
            )
            return

        # Lê imagem
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        except Exception as e:
            self._cached_original = None
            self.view.image_compare.show_status(
                f"Erro ao ler imagem: {e}", is_error=True,
            )
            return

        if img is None:
            self._cached_original = None
            self.view.image_compare.show_status(
                f"Falha ao decodificar: {img_path.name}", is_error=True,
            )
            self.view.image_compare.show_original(None)
            self.view.image_compare.show_processed(None)
            return

        # Se input_source = ALIGNED_FACE, alinha via FFService
        src = self._pipeline_config.input_source
        if src == ImageInputSource.ALIGNED_FACE:
            if self._ff_service is None:
                self.view.image_compare.show_status(
                    "ALIGNED_FACE requer FFService — não disponível "
                    "neste contexto.", is_error=True,
                )
                self.view.image_compare.show_original(None)
                self.view.image_compare.show_processed(None)
                return
            try:
                aligned = self._ff_service.align_face_only(face, dataset=self._dataset)
            except Exception as e:
                self.view.image_compare.show_status(
                    f"align_face_only falhou: {e}", is_error=True,
                )
                self.view.image_compare.show_original(None)
                self.view.image_compare.show_processed(None)
                return
            if aligned is None:
                self.view.image_compare.show_status(
                    f"Nenhuma face detectada em {img_path.name}.",
                    is_error=True,
                )
                self.view.image_compare.show_original(None)
                self.view.image_compare.show_processed(None)
                return
            img = aligned

        self._cached_original = img
        self._cached_processed = None  # força re-aplicação

        pixmap = self._ndarray_to_pixmap(img)
        h, w = img.shape[:2]
        self.view.image_compare.show_original(pixmap, dims=f"{w}×{h}")
        # Painel processada limpo — usuário clica Aplicar pra ver
        self.view.image_compare.show_processed(None)
        self.view.image_compare.show_index(
            self._current_image_idx, len(self._dataset),
            getattr(face, "file_name", None) or img_path.name,
        )
        self.view.image_compare.set_navigation_enabled(True)
        self.view.image_compare.show_status(
            "Clique ▶ Aplicar pra processar a imagem com o pipeline atual."
        )
        # PR I3 — limpa scores quando a imagem muda (navegação)
        self.view.quality_scores.clear()
        # Cancela worker em andamento (resultado seria pra imagem antiga)
        self._cancel_quality_worker()

    def _on_apply(self):
        """Aplica o pipeline corrente na imagem em `_cached_original`."""
        if self._cached_original is None:
            self.view.image_compare.show_status(
                "Sem imagem carregada.", is_error=True,
            )
            return
        if not self._pipeline_config.steps:
            # Pipeline vazio = no-op; mostra mesma imagem
            self._cached_processed = self._cached_original.copy()
            pixmap = self._ndarray_to_pixmap(self._cached_processed)
            h, w = self._cached_processed.shape[:2]
            self.view.image_compare.show_processed(pixmap, dims=f"{w}×{h}")
            self.view.image_compare.show_status(
                "Pipeline vazio — saída idêntica à entrada."
            )
            # Limpa scores em vez de calcular (pipeline vazio = scores
            # idênticos, não vale o custo de extrair embeddings 2×).
            self.view.quality_scores.clear()
            return

        try:
            pipeline = ImageProcessingPipeline(
                self._pipeline_config, ff_service=self._ff_service,
            )
            face = self._dataset.faces[self._current_image_idx]
            jitter = (face.image_path or "") if face else ""
            self._cached_processed = pipeline.process_image(
                self._cached_original, seed_jitter=jitter,
            )
        except Exception as e:
            self.view.image_compare.show_status(
                f"Erro ao aplicar pipeline: {e}", is_error=True,
            )
            self.view.image_compare.show_processed(None)
            return

        pixmap = self._ndarray_to_pixmap(self._cached_processed)
        h, w = self._cached_processed.shape[:2]
        self.view.image_compare.show_processed(pixmap, dims=f"{w}×{h}")
        self.view.image_compare.show_status(
            f"Pipeline aplicado: {len(self._pipeline_config.steps)} step(s)."
        )

        # PR I3 — dispara worker pra calcular FIQA + ‖emb‖ + cosine.
        # Worker thread é cancelado/recriado a cada Apply pra refletir
        # o estado mais recente.
        self._dispatch_quality_scores_worker()

    # ── PR I4 — Persistência na biblioteca global ──────────────────────────

    def _mark_dirty(self):
        if not self._is_dirty:
            self._is_dirty = True
            self.view.set_dirty_indicator(True)

    def _mark_clean(self):
        if self._is_dirty:
            self._is_dirty = False
            self.view.set_dirty_indicator(False)

    def _on_close_attempt(self) -> bool:
        """Hook injetado em `view.confirm_close_callback`. Retorna
        True pra fechar, False pra cancelar."""
        if not self._is_dirty:
            return True
        res = QtWidgets.QMessageBox.question(
            self.view,
            "Mudanças não salvas",
            f"O pipeline '{self._pipeline_config.name}' tem mudanças "
            f"não salvas.\n\nDescartar mudanças e fechar?",
            QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return res == QtWidgets.QMessageBox.StandardButton.Discard

    def _on_save_clicked(self):
        """Salva o pipeline atual em
        `<library_root>/image_pipelines/<id>.json`.

        Sequência:
          1. Pede confirmação do `config_id` (default = config.name).
          2. Se já existe, pergunta sobreescrever.
          3. Persiste via GlobalLibraryStorage.
          4. Marca clean.
        """
        if self._library_root is None:
            QtWidgets.QMessageBox.warning(
                self.view, "Biblioteca indisponível",
                "Configure 'paths.library' no config.json antes de "
                "salvar pipelines.",
            )
            return

        suggested = self._pipeline_config.name or "untitled"
        config_id, ok = QtWidgets.QInputDialog.getText(
            self.view,
            "Salvar pipeline",
            "ID do pipeline (vai virar o nome do arquivo):",
            text=suggested,
        )
        if not ok or not config_id.strip():
            return
        config_id = self._sanitize_id(config_id.strip())

        from faceaudit.storage.global_library_storage import (
            GlobalLibraryStorage,
        )
        storage = GlobalLibraryStorage(library_root=self._library_root)

        # Verifica se já existe
        from faceaudit.enums.global_library_file_type import (
            GlobalLibraryFileType,
        )
        rel_path = GlobalLibraryFileType.IMAGE_PIPELINE_CONFIG.build_path(
            config_id=config_id,
        )
        abs_path = self._library_root / rel_path
        if abs_path.exists():
            res = QtWidgets.QMessageBox.question(
                self.view,
                "Pipeline existente",
                f"Já existe um pipeline com ID '{config_id}' "
                f"na biblioteca global. Sobrescrever?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            if res != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        # Atualiza name no config pra refletir o ID escolhido
        if config_id != self._pipeline_config.name:
            self._pipeline_config.name = config_id
            self.view.input_name.setText(config_id)

        try:
            ref = storage.save_artifact(
                StrategyArtifactType.IMAGE_PIPELINE_CONFIG,
                self._pipeline_config,
                config_id=config_id,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.view, "Erro ao salvar",
                f"Falha ao salvar pipeline: {e}",
            )
            return

        self._mark_clean()
        QtWidgets.QMessageBox.information(
            self.view, "Pipeline salvo",
            f"Salvo em:\n{ref.path}",
        )

    def _on_load_clicked(self):
        """Carrega um pipeline salvo da biblioteca global, substituindo
        o pipeline atual em RAM. Pergunta confirmação se há mudanças
        não salvas.
        """
        if self._library_root is None:
            QtWidgets.QMessageBox.warning(
                self.view, "Biblioteca indisponível",
                "Configure 'paths.library' no config.json antes de "
                "carregar pipelines.",
            )
            return

        # Confirma descarte de mudanças não salvas
        if self._is_dirty:
            res = QtWidgets.QMessageBox.question(
                self.view,
                "Mudanças não salvas",
                "Carregar vai descartar as mudanças não salvas no "
                "pipeline atual.\n\nContinuar?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if res != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        # Lista pipelines disponíveis
        from faceaudit.enums.global_library_folder import GlobalLibraryFolder
        pipelines_dir = (
            self._library_root / GlobalLibraryFolder.IMAGE_PIPELINES.value
        )
        if not pipelines_dir.exists():
            QtWidgets.QMessageBox.information(
                self.view, "Sem pipelines",
                "Nenhum pipeline salvo na biblioteca global ainda.",
            )
            return

        files = sorted(pipelines_dir.glob("*.json"))
        if not files:
            QtWidgets.QMessageBox.information(
                self.view, "Sem pipelines",
                "Nenhum pipeline salvo na biblioteca global ainda.",
            )
            return

        ids = [f.stem for f in files]
        chosen, ok = QtWidgets.QInputDialog.getItem(
            self.view,
            "Carregar pipeline",
            f"Pipelines disponíveis em {pipelines_dir}:",
            ids, 0, False,
        )
        if not ok:
            return

        target = pipelines_dir / f"{chosen}.json"
        try:
            raw = target.read_text(encoding="utf-8")
            new_config = ImageProcessingPipelineConfig.from_json_string(raw)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.view, "Erro ao carregar",
                f"Falha ao ler {target.name}: {e}",
            )
            return

        # Substitui pipeline atual
        self._pipeline_config = new_config
        self._current_step_idx = -1
        self._cached_processed = None  # invalida — usuário roda Apply
        self._cancel_quality_worker()
        self.view.quality_scores.clear()
        self.view.image_compare.show_processed(None)
        self.view.image_compare.show_status(
            f"Pipeline '{chosen}' carregado. Clique ▶ Aplicar."
        )

        # Re-sincroniza UI
        self.view.set_pipeline_meta(
            self._pipeline_config.name,
            self._pipeline_config.input_source,
            self._pipeline_config.seed,
        )
        self.view.steps_list.populate(
            [s.step_type for s in self._pipeline_config.steps]
        )
        self.view.step_config.show_config(None)

        self._mark_clean()

    @staticmethod
    def _sanitize_id(raw: str) -> str:
        """Remove caracteres problemáticos pra nome de arquivo
        cross-platform. Substitui `\\/:*?"<>|` por `_`."""
        bad = '\\/:*?"<>|'
        out = raw
        for ch in bad:
            out = out.replace(ch, "_")
        return out.strip() or "untitled"

    # ── PR I6 — Canvas de máscara de oclusão ──────────────────────────────

    def _on_mask_canvas_requested(self, occlusion_config):
        """Abre o `OcclusionMaskCanvas` com a imagem corrente do
        preview. O usuário desenha shapes; ao confirmar, atualiza
        `occlusion_config.mask` in-place e marca dirty.

        `occlusion_config` é o mesmo objeto já dentro de
        `_pipeline_config.steps` (form mutates in-place — padrão dos
        outros configs).
        """
        if self._cached_original is None:
            QtWidgets.QMessageBox.warning(
                self.view, "Sem imagem",
                "Carregue uma imagem no preview antes de desenhar a máscara.",
            )
            return

        # Converte ndarray BGR → QPixmap
        pixmap = self._ndarray_to_pixmap(self._cached_original)

        from .occlusion_mask_canvas import OcclusionMaskCanvas
        dlg = OcclusionMaskCanvas(
            pixmap=pixmap,
            initial_mask=occlusion_config.mask,
            parent=self.view,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        new_mask = dlg.get_mask()
        occlusion_config.mask = new_mask
        # Refresh do botão pra mostrar contagem de shapes
        if hasattr(self.view.step_config, "_refresh_mask_button"):
            self.view.step_config._refresh_mask_button()
        # Marca dirty (mesma semântica de outras edições do form)
        self._mark_dirty()

    # ── PR I3 — Worker thread de scores de qualidade ──────────────────────

    def _dispatch_quality_scores_worker(self):
        """Cria worker + thread pra calcular scores em background.

        Cancela qualquer worker anterior (resultado seria stale).
        Mostra "loading" no painel imediatamente; resultado chega via
        sinal `finished` ou `error`.
        """
        if self._cached_original is None or self._cached_processed is None:
            return
        if self._ff_service is None:
            self.view.quality_scores.show_error(
                "FFService não disponível neste contexto."
            )
            return

        self._cancel_quality_worker()
        self.view.quality_scores.show_loading()

        thread = QtCore.QThread()
        worker = QualityScoresWorker(
            ff_service=self._ff_service,
            original_bgr=self._cached_original,
            processed_bgr=self._cached_processed,
            input_source=self._pipeline_config.input_source,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_quality_finished)
        worker.error.connect(self._on_quality_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._quality_thread = thread
        self._quality_worker = worker
        thread.start()

    def _cancel_quality_worker(self):
        """Cancela worker em andamento. Não tem ponto de cancelamento
        granular — desconecta as callbacks (resultado fica ignorado)
        e move thread+worker pra lista de pendentes pra continuar
        viva até terminar naturalmente. Sem isso, o sip wrapper
        destruiria o QThread em execução."""
        if self._quality_worker is None and self._quality_thread is None:
            return
        if self._quality_worker is not None:
            try:
                self._quality_worker.finished.disconnect(self._on_quality_finished)
                self._quality_worker.error.disconnect(self._on_quality_error)
            except (TypeError, RuntimeError):
                pass  # já desconectado ou worker GC'd
        if self._quality_thread is not None:
            thread = self._quality_thread
            worker = self._quality_worker
            entry = (thread, worker)
            self._pending_quality_threads.append(entry)
            thread.finished.connect(
                lambda e=entry: self._discard_pending_thread(e)
            )
        self._quality_worker = None
        self._quality_thread = None

    def _discard_pending_thread(self, entry):
        """Remove uma entrada (thread, worker) da lista de pendentes
        após `thread.finished` — a partir daqui a thread já não está
        rodando, então é seguro deixar o GC limpar."""
        try:
            self._pending_quality_threads.remove(entry)
        except ValueError:
            pass

    def _on_quality_finished(self, result):
        """Resultado do worker chegou — atualiza panel."""
        self.view.quality_scores.show_results(result)
        self._move_current_quality_to_pending()

    def _on_quality_error(self, msg: str):
        """Worker falhou — mostra erro no panel."""
        self.view.quality_scores.show_error(msg)
        self._move_current_quality_to_pending()

    def _move_current_quality_to_pending(self):
        """Move o (thread, worker) corrente pra lista de pendentes
        quando o resultado chega — `worker.finished` ainda tem slots
        `thread.quit` e `worker.deleteLater` pendentes na fila do
        event loop, então a thread C++ ainda está rodando seu event
        loop nesse instante. Dropar `self._quality_thread = None`
        agora destruiria o sip wrapper → `~QThread()` em thread
        ainda rodando → qFatal."""
        if self._quality_thread is None:
            return
        thread = self._quality_thread
        worker = self._quality_worker
        entry = (thread, worker)
        self._pending_quality_threads.append(entry)
        thread.finished.connect(
            lambda e=entry: self._discard_pending_thread(e)
        )
        self._quality_worker = None
        self._quality_thread = None

    @staticmethod
    def _ndarray_to_pixmap(bgr: np.ndarray) -> QtGui.QPixmap:
        """Converte ndarray BGR uint8 (cv2) em QPixmap."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        bytes_per_line = 3 * w
        # Cópia explícita pra evitar referência a buffer numpy que
        # pode ser GC'd antes do QImage ser usado.
        rgb_contig = np.ascontiguousarray(rgb)
        qimage = QtGui.QImage(
            rgb_contig.data, w, h, bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        )
        return QtGui.QPixmap.fromImage(qimage.copy())
