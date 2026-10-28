"""Ajuste automático dos parâmetros de um pipeline de imagens à questionada.

Procura, entre os campos numéricos marcados como ajustáveis
(`StepParamSpec.tunable`), os valores que deixam as imagens da base, UMA
A UMA, mais parecidas com a questionada. O objetivo é a distância média
por imagem (`per_image_distance`: soma das distâncias das medidas, em
múltiplos da tolerância) — não a mediana do conjunto, que já se mostrou
enganosa.

Busca por coordenadas com passo decrescente: a cada rodada, cada campo é
testado em 6 valores em torno do atual, e o passo cai à metade na rodada
seguinte. Simples, determinística e auditável: o resultado lista o que
mudou, de quanto para quanto.

O que o ajuste NÃO decide: a ordem dos passos, a técnica de meio-tom, o
tipo de oclusão, qualquer escolha qualitativa. Esses ficam com o perito.

Duas amostras disjuntas: o ajuste é feito na de TREINO e o resultado é
relatado também numa de VALIDAÇÃO, com imagens que o ajuste não viu —
sem isso o número relatado seria otimista por construção.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.models.image_processing.step_param_spec import (
    StepParamKind,
    StepParamSpec,
)
from faceaudit.services.image_degradation_profile_service import (
    MEASURES,
    per_image_distance,
)
from faceaudit.services.image_pipeline_evaluation_service import (
    SampleEvaluation,
    SampleItem,
    evaluate,
    load_sample,
    sample_indices,
)

DEFAULT_TRAIN_SIZE = 40
DEFAULT_VALIDATION_SIZE = 40
DEFAULT_ROUNDS = 4
_CANDIDATOS_POR_LADO = 3


class FitCancelled(Exception):
    pass


@dataclass(frozen=True)
class ParamChange:
    step_index: int
    step_label: str
    param: str
    param_label: str
    before: Any
    after: Any


@dataclass(frozen=True)
class FitResult:
    config: ImageProcessingPipelineConfig
    changes: List[ParamChange]
    train_before: SampleEvaluation
    train_after: SampleEvaluation
    validation_before: Optional[SampleEvaluation]
    validation_after: Optional[SampleEvaluation]
    n_evaluations: int

    def to_dict(self) -> Dict[str, Any]:
        """Registro do ajuste (vai para a descrição do pipeline/log)."""
        def _r(ev: Optional[SampleEvaluation]):
            return None if ev is None else {"n_images": ev.n_measured,
                                             "agreement": ev.agreement}
        return {
            "changes": [dataclasses.asdict(c) for c in self.changes],
            "train_before": _r(self.train_before),
            "train_after": _r(self.train_after),
            "validation_before": _r(self.validation_before),
            "validation_after": _r(self.validation_after),
            "n_evaluations": self.n_evaluations,
        }


def tunable_params(config: ImageProcessingPipelineConfig) -> List[Tuple[int, StepParamSpec]]:
    """(índice do passo, campo) dos campos que o ajuste pode mexer."""
    out = []
    for i, step in enumerate(config.steps):
        for spec in step.param_specs():
            if (spec.tunable and spec.kind in (StepParamKind.INT, StepParamKind.FLOAT)
                    and spec.is_visible(step)):
                out.append((i, spec))
    return out


def objective(ev: SampleEvaluation, config: ImageProcessingPipelineConfig, n_imagens: int) -> float:
    """Distância média por imagem; imagem que falhou conta como a pior."""
    alvo = config.questioned.measures
    pior = 10.0 * sum(1 for m in MEASURES if getattr(alvo, m.key) is not None)
    soma = sum(per_image_distance(alvo, m) for m in ev.measures)
    soma += pior * (n_imagens - len(ev.measures))
    return soma / max(1, n_imagens)


def _copia(config: ImageProcessingPipelineConfig) -> ImageProcessingPipelineConfig:
    return ImageProcessingPipelineConfig.from_dict(config.to_dict())


def _candidatos(spec: StepParamSpec, atual: float, rodada: int) -> List[float]:
    lo, hi = spec.search_range()
    passo = (hi - lo) / (2 * _CANDIDATOS_POR_LADO) / (2 ** rodada)
    if spec.kind == StepParamKind.INT:
        passo = max(1.0, round(passo))
    vistos, out = {atual}, []
    for k in range(-_CANDIDATOS_POR_LADO, _CANDIDATOS_POR_LADO + 1):
        if k == 0:
            continue
        v = min(hi, max(lo, atual + k * passo))
        v = spec.clamp(v)
        if v not in vistos:
            vistos.add(v)
            out.append(v)
    return out


def _com_valor(config, i: int, spec: StepParamSpec, valor) -> ImageProcessingPipelineConfig:
    c = _copia(config)
    setattr(c.steps[i], spec.name, valor)
    c.steps[i].normalize()
    return c


def split_indices(n_total: int, n_train: int, n_validation: int) -> Tuple[List[int], List[int]]:
    """Treino espaçado pela base; validação espaçada no que sobrou."""
    treino = sample_indices(n_total, n_train)
    resto = [i for i in range(n_total) if i not in set(treino)]
    validacao = [resto[j] for j in sample_indices(len(resto), n_validation)] if resto else []
    return treino, validacao


def fit_to_questioned(
    config: ImageProcessingPipelineConfig,
    dataset: FaceDataset,
    *,
    n_train: int = DEFAULT_TRAIN_SIZE,
    n_validation: int = DEFAULT_VALIDATION_SIZE,
    rounds: int = DEFAULT_ROUNDS,
    ff_service=None,
    progress: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    train_sample: Optional[Sequence[SampleItem]] = None,
    validation_sample: Optional[Sequence[SampleItem]] = None,
) -> FitResult:
    """Ajusta os campos numéricos ajustáveis do pipeline à questionada.

    Raises:
        ValueError: pipeline sem questionada medida ou sem campo ajustável.
        FitCancelled: `should_stop()` devolveu True.
    """
    if config.questioned is None or config.questioned.measures is None:
        raise ValueError("O ajuste precisa de uma imagem questionada no pipeline.")
    params = tunable_params(config)
    if not params:
        raise ValueError("Nenhum passo do pipeline tem parâmetro numérico ajustável.")

    def _parar():
        if should_stop and should_stop():
            raise FitCancelled()

    def _avisar(msg: str):
        if progress:
            progress(msg)

    if train_sample is None or validation_sample is None:
        _avisar("Carregando as amostras de treino e de validação…")
        i_treino, i_val = split_indices(len(dataset), n_train, n_validation)
        if train_sample is None:
            train_sample = load_sample(dataset, config.input_source,
                                       indices=i_treino, ff_service=ff_service)
        if validation_sample is None:
            validation_sample = load_sample(dataset, config.input_source,
                                            indices=i_val, ff_service=ff_service)
    if not train_sample:
        raise ValueError("A amostra de treino ficou vazia (imagens ilegíveis?).")

    n_aval = 0

    def _avaliar(c):
        nonlocal n_aval
        _parar()
        n_aval += 1
        ev = evaluate(train_sample, c, ff_service=ff_service)
        return ev, objective(ev, c, len(train_sample))

    atual = _copia(config)
    ev0, melhor = _avaliar(atual)
    ev_atual = ev0
    total_previsto = rounds * len(params) * 2 * _CANDIDATOS_POR_LADO
    for rodada in range(rounds):
        for i, spec in params:
            valor_atual = getattr(atual.steps[i], spec.name)
            for v in _candidatos(spec, float(valor_atual), rodada):
                cand = _com_valor(atual, i, spec, v)
                ev, o = _avaliar(cand)
                if o < melhor - 1e-9:
                    atual, melhor, ev_atual = cand, o, ev
                _avisar(
                    f"Rodada {rodada + 1}/{rounds} · {spec.label} · avaliação "
                    f"{n_aval}/~{total_previsto} · distância média {melhor:.2f}"
                )

    mudancas = []
    for i, spec in params:
        antes = getattr(config.steps[i], spec.name)
        depois = getattr(atual.steps[i], spec.name)
        if antes != depois:
            mudancas.append(ParamChange(i, config.steps[i].step_type.label_pt,
                                        spec.name, spec.label, antes, depois))

    val_antes = val_depois = None
    if validation_sample:
        _avisar("Medindo na amostra de validação…")
        val_antes = evaluate(validation_sample, config, ff_service=ff_service)
        val_depois = evaluate(validation_sample, atual, ff_service=ff_service)

    return FitResult(
        config=atual,
        changes=mudancas,
        train_before=ev0,
        train_after=ev_atual,
        validation_before=val_antes,
        validation_after=val_depois,
        n_evaluations=n_aval,
    )
