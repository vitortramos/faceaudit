"""Ajuste automático dos parâmetros de um pipeline de imagens à questionada.

Procura, entre os campos numéricos marcados como ajustáveis
(`StepParamSpec.tunable`), os valores que deixam as imagens da base, UMA
A UMA, dentro da tolerância da questionada.

O objetivo é o EXCESSO além da tolerância, por imagem e medida
(`excess_objective`): o que já está dentro não conta. A primeira versão
minimizava a distância média — e no caso real a distância caiu enquanto
a parcela de imagens dentro da tolerância, que é o que se relata, CAIU
(18% → 12%): aproximar imagens que já estavam dentro "pagava" empurrar
outras para fora. Com o objetivo alinhado ao relatório, partindo do
mesmo ponto, a parcela subiu (12% → 21%, em 231 imagens de validação).

Busca por coordenadas com passo decrescente: a cada rodada, cada campo é
testado em 6 valores em torno do atual, e o passo cai à metade na rodada
seguinte. Simples, determinística e auditável: o resultado lista o que
mudou, de quanto para quanto.

Duas partidas: o pipeline como está e um ponto calculado das medidas da
questionada (`start_from_questioned`). A busca fica presa perto de onde
começa, e nenhum dos dois pontos ganhou em todas as questionadas
testadas; fica o que termina com menor excesso no TREINO.

O que o ajuste NÃO decide: a ordem dos passos, a técnica de meio-tom, o
tipo de oclusão, qualquer escolha qualitativa. Esses ficam com o perito.

Duas amostras disjuntas: o ajuste é feito na de TREINO e o resultado é
relatado numa de VALIDAÇÃO com todas as outras imagens da base (até
`DEFAULT_VALIDATION_MAX`) — o número de treino seria otimista, e 40
imagens de validação deixavam a contagem oscilar vários pontos com duas
imagens a mais ou a menos. `improved_on_validation` diz se vale aplicar.
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
)
from faceaudit.services.image_pipeline_evaluation_service import (
    SampleEvaluation,
    SampleItem,
    evaluate,
    load_sample,
    sample_indices,
)

DEFAULT_TRAIN_SIZE = 40
DEFAULT_VALIDATION_MAX = 400
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
    validation_objective_before: Optional[float] = None
    validation_objective_after: Optional[float] = None
    # De onde partiu a busca que venceu: "pipeline atual" ou "medidas da
    # questionada" (ver `start_from_questioned`).
    start: str = "pipeline atual"

    @property
    def improved_on_validation(self) -> Optional[bool]:
        """O ajuste melhorou nas imagens que não usou? None sem validação.
        Melhorou = menos excesso além da tolerância E pelo menos tantas
        imagens com todas as medidas, ou todas menos uma, dentro."""
        if self.validation_objective_before is None or self.validation_objective_after is None:
            return None
        a = (self.validation_before.agreement or {}) if self.validation_before else {}
        b = (self.validation_after.agreement or {}) if self.validation_after else {}
        nao_piorou = (
            b.get("all_close_fraction", 0) >= a.get("all_close_fraction", 0)
            and b.get("all_but_one_close_fraction", 0) >= a.get("all_but_one_close_fraction", 0)
        )
        return self.validation_objective_after < self.validation_objective_before and nao_piorou

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
            "start": self.start,
            "validation_objective_before": self.validation_objective_before,
            "validation_objective_after": self.validation_objective_after,
            "improved_on_validation": self.improved_on_validation,
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


def excess_objective(ev: SampleEvaluation, config: ImageProcessingPipelineConfig,
                     n_imagens: int) -> float:
    """Excesso médio além da tolerância, por imagem.

    Para cada imagem e medida: `max(0, distância − 1)` (distância em
    múltiplos da tolerância), limitado a 10. Medida dentro da tolerância
    contribui zero. Imagem que falhou no pipeline conta como a pior.
    """
    alvo = config.questioned.measures
    medidas = [m for m in MEASURES if getattr(alvo, m.key) is not None]
    pior = 10.0 * len(medidas)
    soma = 0.0
    for x in ev.measures:
        for m in medidas:
            d = m.distance(getattr(alvo, m.key), getattr(x, m.key))
            soma += 10.0 if d is None else min(max(0.0, d - 1.0), 10.0)
    soma += pior * (n_imagens - len(ev.measures))
    return soma / max(1, n_imagens)


objective = excess_objective


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


def start_from_questioned(
    config: ImageProcessingPipelineConfig,
) -> Optional[ImageProcessingPipelineConfig]:
    """Cópia do pipeline com os passos levados ao ponto de partida que as
    medidas da questionada sugerem (`start_from_questioned` de cada
    passo), ou None quando nenhum passo sabe partir delas.

    Existe porque a busca por coordenadas muda um campo por vez e fica
    presa perto de onde começa: numa questionada clara (nível médio 219,
    52% de branco), partindo do modelo calibrado noutra cópia, o ajuste
    chegava a 10% das imagens com 8 de 9 medidas dentro; partindo das
    medidas, a 26%. Noutra questionada era o contrário — por isso o
    ajuste parte dos DOIS e fica com o melhor no treino.
    """
    alvo = config.questioned.measures if config.questioned is not None else None
    if alvo is None:
        return None
    copia = _copia(config)
    mudou = False
    for step in copia.steps:
        mudou = step.start_from_questioned(alvo) or mudou
    return copia if mudou else None


def split_indices(n_total: int, n_train: int,
                  n_validation: int = DEFAULT_VALIDATION_MAX) -> Tuple[List[int], List[int]]:
    """Treino espaçado pela base; validação espaçada no que sobrou (todo o
    resto, até `n_validation`)."""
    treino = sample_indices(n_total, n_train)
    resto = [i for i in range(n_total) if i not in set(treino)]
    validacao = [resto[j] for j in sample_indices(len(resto), n_validation)] if resto else []
    return treino, validacao


def fit_to_questioned(
    config: ImageProcessingPipelineConfig,
    dataset: FaceDataset,
    *,
    n_train: int = DEFAULT_TRAIN_SIZE,
    n_validation: int = DEFAULT_VALIDATION_MAX,
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

    partidas = [("pipeline atual", _copia(config))]
    pelas_medidas = start_from_questioned(config)
    if pelas_medidas is not None:
        partidas.append(("medidas da questionada", pelas_medidas))
    total_previsto = len(partidas) * rounds * len(params) * 2 * _CANDIDATOS_POR_LADO

    ev0 = None
    vencedor = None  # (objetivo, rótulo, config, avaliação)
    for n_partida, (rotulo, atual) in enumerate(partidas, start=1):
        ev_atual, melhor = _avaliar(atual)
        if ev0 is None:
            ev0 = ev_atual
        for rodada in range(rounds):
            for i, spec in params:
                valor_atual = getattr(atual.steps[i], spec.name)
                for v in _candidatos(spec, float(valor_atual), rodada):
                    cand = _com_valor(atual, i, spec, v)
                    ev, o = _avaliar(cand)
                    if o < melhor - 1e-9:
                        atual, melhor, ev_atual = cand, o, ev
                    _avisar(
                        f"Partida {n_partida}/{len(partidas)} ({rotulo}) · rodada "
                        f"{rodada + 1}/{rounds} · {spec.label} · avaliação "
                        f"{n_aval}/~{total_previsto} · excesso {melhor:.2f}"
                    )
        if vencedor is None or melhor < vencedor[0] - 1e-9:
            vencedor = (melhor, rotulo, atual, ev_atual)
    _, partida_vencedora, atual, ev_atual = vencedor

    mudancas = []
    for i, spec in params:
        antes = getattr(config.steps[i], spec.name)
        depois = getattr(atual.steps[i], spec.name)
        if antes != depois:
            mudancas.append(ParamChange(i, config.steps[i].step_type.label_pt,
                                        spec.name, spec.label, antes, depois))

    val_antes = val_depois = obj_antes = obj_depois = None
    if validation_sample:
        _avisar(f"Medindo nas {len(validation_sample)} imagens de validação…")
        val_antes = evaluate(validation_sample, config, ff_service=ff_service)
        val_depois = evaluate(validation_sample, atual, ff_service=ff_service)
        obj_antes = objective(val_antes, config, len(validation_sample))
        obj_depois = objective(val_depois, atual, len(validation_sample))

    return FitResult(
        config=atual,
        changes=mudancas,
        train_before=ev0,
        train_after=ev_atual,
        validation_before=val_antes,
        validation_after=val_depois,
        n_evaluations=n_aval,
        validation_objective_before=obj_antes,
        validation_objective_after=obj_depois,
        start=partida_vencedora,
    )
