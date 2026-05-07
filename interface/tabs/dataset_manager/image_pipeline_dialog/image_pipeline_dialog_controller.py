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
from typing import List, Optional

import cv2
import numpy as np
from PyQt6 import QtGui, QtWidgets

from faceaudit.enums.image_input_source import ImageInputSource
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
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
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        self._dataset = dataset
        self._ff_service = ff_service

        self._pipeline_config = ImageProcessingPipelineConfig()
        self._current_step_idx: int = -1

        # Imagem corrente
        if not dataset or len(dataset) == 0:
            self._current_image_idx = -1
        elif initial_face_index >= 0 and initial_face_index < len(dataset):
            self._current_image_idx = initial_face_index
        else:
            self._current_image_idx = random_module.randrange(len(dataset))

        self._cached_original: Optional[np.ndarray] = None
        self._cached_processed: Optional[np.ndarray] = None

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
        self._pipeline_config.name = name or "untitled"

    def _on_input_source_changed(self, src: ImageInputSource):
        self._pipeline_config.input_source = src
        # Re-carrega a imagem corrente (mudou o que mostrar)
        self._reload_current_image()

    def _on_seed_changed(self, seed: int):
        self._pipeline_config.seed = int(seed)
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

    def _on_step_config_changed(
        self, _config: ImageProcessingStepConfig,
    ):
        # `_config` é o mesmo objeto já dentro de `_pipeline_config.steps`
        # (form mutates in place). Não precisa fazer nada aqui — a
        # próxima chamada de "Aplicar" reflete a mudança.
        pass

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
