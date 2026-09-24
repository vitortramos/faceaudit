"""ModifyPathsWorker — roda DatasetLocalizationService em thread
separada (renomeado de LocalizeDatasetWorker — PR L1d).

Cobre as três operações *pesadas* de "Modificar Caminhos"
(REWRITE_RELATIVE / COPY_ORIGINALS / REBASE_PREFIX). A correção leve de
`path_root` (FIX_PATH_ROOT) é só-metadados e é aplicada direto no
controller, sem passar por aqui.

Sinais Qt seguindo o padrão dos outros workers do dataset_manager
(EnrichWorker, FilterWorker, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtCore

from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.localization_naming import NamingStrategy
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.localization_result import LocalizationResult
from faceaudit.services.dataset_localization_service import (
    SELF_CONTAINED_IMAGES_SUBDIR,
    DatasetLocalizationService,
    self_contained_dataset_dir,
)


class ModifyPathsWorker(QtCore.QObject):
    """Worker assíncrono para modificação de caminhos (REWRITE_RELATIVE /
    COPY_ORIGINALS / REBASE_PREFIX).

    Emite progress periodicamente — uma vez a cada `_PROGRESS_THROTTLE_FACES`
    faces ou ao chegar no final. Sem throttle, datasets de 100k faces
    saturam o event loop Qt com 100k signal emissions.
    """

    finished = QtCore.pyqtSignal(object)  # LocalizationResult ou None
    error = QtCore.pyqtSignal(str)
    progress_update = QtCore.pyqtSignal(str, int)

    _PROGRESS_THROTTLE_FACES = 50

    def __init__(
        self,
        dataset: FaceDataset,
        target_dir: Path,
        mode: LocalizationMode,
        naming: NamingStrategy = NamingStrategy.PRESERVE,
        skip_existing: bool = True,
        old_prefix: Optional[str] = None,
        image_kind=None,
        crop_margin_factor: float = 2.0,
        ff_service=None,
    ):
        super().__init__()
        # Forma da imagem a gravar no destino. `None`/ORIGINAL preserva o
        # comportamento histórico (copiar as imagens como estão); CROP e
        # ALIGNED produzem a face a partir de cada original.
        self.image_kind = image_kind
        self.crop_margin_factor = float(crop_margin_factor)
        self.ff_service = ff_service
        self.dataset = dataset
        self.target_dir = target_dir
        self.mode = mode
        self.naming = naming
        self.skip_existing = skip_existing
        # Só relevante em REBASE_PREFIX (substituição de prefixo de path).
        self.old_prefix = old_prefix
        self._is_running = True

    def run(self):
        """Executado na thread. Roda o service e emite finished com o
        `LocalizationResult` (ou None em cancelamento)."""
        try:
            self.progress_update.emit(
                f"Iniciando {self.mode.value} para '{self.dataset.name}'…", 0,
            )
            if self._should_materialize():
                self._run_materialization()
                return

            service = DatasetLocalizationService()
            target = self.target_dir
            if self.mode == LocalizationMode.COPY_ORIGINALS:
                # Copiar imagens produz um dataset autocontido, e ele ganha
                # pasta própria: `<destino>/<nome>/` com `images/` dentro.
                # O parquet precisa ser gravado NESSA pasta — o controller
                # sugere isso no Salvar como seguinte.
                target = self_contained_dataset_dir(target, self.dataset.name)
            result = service.localize(
                self.dataset,
                target,
                mode=self.mode,
                naming=self.naming,
                skip_existing=self.skip_existing,
                old_prefix=self.old_prefix,
                # Subpasta fixa dentro da pasta do dataset. Só tem
                # efeito em COPY_ORIGINALS.
                copy_subdir=SELF_CONTAINED_IMAGES_SUBDIR,
                progress_callback=self._on_progress,
            )
            if not self._is_running:
                self.finished.emit(None)
                return
            # Mensagem final detalhada
            self.progress_update.emit(result.summary(), 100)
            self.finished.emit(result)
        except InterruptedError:
            self.finished.emit(None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
            self.finished.emit(None)

    def _should_materialize(self) -> bool:
        """Copiar imagens + forma diferente da original = materializar."""
        from faceaudit.enums.dataset_image_kind import DatasetImageKind

        return (
            self.mode == LocalizationMode.COPY_ORIGINALS
            and self.image_kind is not None
            and DatasetImageKind.from_str(self.image_kind).requires_face_detection
            and self.ff_service is not None
        )

    def _run_materialization(self) -> None:
        """Produz recortes/alinhadas na pasta do dataset e emite o resultado.

        Reusa o mesmo serviço do botão "Gerar Imagens" — a diferença é a
        porta de entrada: aqui o conjunto de imagens vem do dataset aberto,
        cujas faces podem apontar para pastas diferentes.
        """
        from faceaudit.enums.dataset_image_kind import DatasetImageKind
        from faceaudit.services.dataset_localization_service import (
            SELF_CONTAINED_IMAGES_SUBDIR,
            self_contained_dataset_dir,
        )
        from faceaudit.services.face_image_materialization_service import (
            FaceImageMaterializationService,
        )
        from faceaudit.services.ff_geometry_adapter import FFServiceGeometryAdapter

        kind = DatasetImageKind.from_str(self.image_kind)
        destino = self_contained_dataset_dir(self.target_dir, self.dataset.name)
        self.progress_update.emit(
            f"Gerando imagens ({kind.label_pt}) em {destino}…", 0,
        )
        servico = FaceImageMaterializationService(
            FFServiceGeometryAdapter(self.ff_service)
        )
        novo_dataset, resultado = servico.materialize_dataset(
            self.dataset, destino,
            kind=kind,
            margin_factor=self.crop_margin_factor,
            images_subdir=SELF_CONTAINED_IMAGES_SUBDIR,
            progress_callback=self._on_progress,
        )
        if not self._is_running:
            self.finished.emit(None)
            return

        # A UI a jusante consome `LocalizationResult`; empacotar aqui evita
        # um segundo contrato só para este caminho.
        from faceaudit.models.localization_result import LocalizationResult

        self.progress_update.emit(resultado.summary_pt(), 100)
        self.finished.emit(LocalizationResult(
            dataset=novo_dataset,
            target_dir=destino,
            mode=self.mode,
            naming=self.naming,
            n_processed=resultado.n_processed,
            n_copied=resultado.n_written,
            n_skipped=resultado.n_skipped,
            n_failed=resultado.n_failed,
            failures=tuple(resultado.failures),
            duration_seconds=0.0,
        ))

    def stop(self):
        """Sinaliza cancelamento cooperativo. O service não tem ponto
        de cancelamento granular hoje — quem rodar Cancel pode esperar
        terminar a face atual e a chamada toda completa, mas ignora o
        resultado."""
        self._is_running = False

    def _on_progress(self, idx: int, total: int, msg: str):
        """Recebe (idx, total, msg) do service e converte em
        (mensagem, pct) pro Qt signal."""
        if not self._is_running:
            # Cancelamento cooperativo — service não tem como abortar,
            # mas podemos pular emissões pra reduzir footprint.
            return
        # Throttle: emite só a cada N faces ou na última
        is_last = (idx + 1) >= total
        if not is_last and (idx % self._PROGRESS_THROTTLE_FACES) != 0:
            return
        pct = int(((idx + 1) / max(total, 1)) * 100)
        # Trim de mensagens de path muito longas pra não inflar o log
        short = msg if len(msg) <= 80 else "…" + msg[-77:]
        self.progress_update.emit(
            f"[{idx + 1}/{total}] {short}", pct,
        )
