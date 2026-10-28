"""Avaliação de um pipeline de imagens numa AMOSTRA da base, contra a
questionada.

Calibrar olhando uma face de cada vez engana: o caso que motivou isto
tinha mediana perfeita e só 5% das imagens parecidas com a questionada.
Aqui o pipeline roda numa amostra (sem gravar nada) e devolve o resumo,
a comparação da mediana e a concordância imagem a imagem — a mesma conta
que o Processar grava no manifesto, para o perito ver ANTES de processar.

A amostra é carregada uma vez (`load_sample`) e reaproveitada a cada
avaliação: o ajuste automático avalia dezenas de configurações sobre as
mesmas imagens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from faceaudit.enums.image_input_source import ImageInputSource
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.pipelines.image_processing_pipeline import ImageProcessingPipeline
from faceaudit.services.image_degradation_profile_service import (
    DegradationMeasures,
    agreement,
    compare_summary,
    measure,
    read_image_bgr,
    summarize,
)
from faceaudit.services.image_pipeline_batch_service import geometria_do_registro

DEFAULT_SAMPLE_SIZE = 40


@dataclass(frozen=True)
class SampleItem:
    image: np.ndarray
    face: Optional[ImageFaceGeometry]
    jitter: str
    file_name: str


@dataclass(frozen=True)
class SampleEvaluation:
    measures: List[DegradationMeasures]
    summary: Dict[str, Any]
    comparison: List[Dict[str, Any]]
    agreement: Optional[Dict[str, Any]]
    n_failed: int
    failures: List[str]

    @property
    def n_measured(self) -> int:
        return len(self.measures)


def sample_indices(n_total: int, n: int) -> List[int]:
    """Índices espaçados uniformemente — determinístico e cobre a base
    inteira (as bases costumam vir ordenadas por identidade)."""
    if n_total <= 0:
        return []
    n = max(1, min(int(n), n_total))
    return sorted({int(round(i)) for i in np.linspace(0, n_total - 1, n)})


def load_sample(
    dataset: FaceDataset,
    input_source: ImageInputSource,
    *,
    n: int = DEFAULT_SAMPLE_SIZE,
    ff_service=None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[SampleItem]:
    """Lê (e, se for o caso, alinha) as imagens da amostra. Imagem que não
    abre ou não alinha fica de fora."""
    itens: List[SampleItem] = []
    indices = sample_indices(len(dataset), n)
    for pos, idx in enumerate(indices):
        if progress:
            progress(pos, len(indices))
        face = dataset.faces[idx]
        try:
            img = read_image_bgr(dataset.resolve_image_path(face))
        except Exception:  # noqa: BLE001 — imagem ilegível sai da amostra
            continue
        if input_source == ImageInputSource.ALIGNED_FACE:
            if ff_service is None:
                raise RuntimeError("Origem 'Face alinhada' exige o detector.")
            alinhada, kps = ff_service.align_face_from_array(img)
            if alinhada is None:
                continue
            img, geom = alinhada, ImageFaceGeometry.from_values(None, kps)
        else:
            geom = geometria_do_registro(face, dataset)
        itens.append(SampleItem(img, geom, str(face.image_path or ""),
                                str(face.file_name or "")))
    return itens


def evaluate(
    sample: Sequence[SampleItem],
    config: ImageProcessingPipelineConfig,
    *,
    ff_service=None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> SampleEvaluation:
    """Roda o pipeline na amostra e mede contra `config.questioned`."""
    pipeline = ImageProcessingPipeline(config, ff_service=ff_service)
    medidas: List[DegradationMeasures] = []
    falhas: List[str] = []
    for item in sample:
        if should_stop and should_stop():
            break
        try:
            out = pipeline.process(item.image, seed_jitter=item.jitter, face=item.face)
        except Exception as e:  # noqa: BLE001 — a falha é contada e listada
            falhas.append(f"{item.file_name}: {e}")
            continue
        medidas.append(measure(out.image, out.face))
    alvo = None if config.questioned is None else config.questioned.measures
    resumo = summarize(medidas) if medidas else {}
    return SampleEvaluation(
        measures=medidas,
        summary=resumo,
        comparison=compare_summary(alvo, resumo, medidas) if medidas else [],
        agreement=agreement(alvo, medidas),
        n_failed=len(falhas),
        failures=falhas,
    )
