"""Regra EMBEDDINGS_COMPLETE — face sem embedding barra o cálculo de escores.

O caso real que motivou a regra: um dataset de vídeo com faces sem
`embedding_v2` chegou ao SIMILARITY_SCORING e o erro foi do numpy
("array at index 82 has size 1"), sem dizer qual dataset nem qual imagem.
O valor nulo do parquet vira `np.array(None, float32)` no carregador — um
escalar `nan` —, e é esse o formato que a regra tem de pegar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from faceaudit.contexts.strategy_execution_context import StrategyExecutionContext
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.factories.validation_rule_factory import ValidationRuleFactory
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.models.instructions.similarity_scoring_instruction import (
    SimilarityScoringInstruction,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig
from faceaudit.validation_rules.embeddings_complete_rule import EmbeddingsCompleteRule


def _face(i: int, embedding) -> EmbeddingInfo:
    return EmbeddingInfo(
        embedding=embedding, face_hash=f"h{i:03d}", image_path=f"/img/{i}.png",
        identity="A", file_name=f"face_{i}.png",
    )


def _dataset(embeddings) -> EmbeddingDataset:
    return EmbeddingDataset(
        name="ds", items=[_face(i, e) for i, e in enumerate(embeddings)],
    )


def _contexto(dataset, rid="ds") -> StrategyExecutionContext:
    ctx = StrategyExecutionContext()
    ctx.set_resource(StrategyResource(
        resource_id=rid, name=rid,
        resource_type=StrategyResourceType.DATASET, payload=dataset,
    ))
    return ctx


def _validar(dataset, **kw):
    rule = EmbeddingsCompleteRule(rule_id="r", dataset_resource_id="ds", **kw)
    return rule.validate(None, _contexto(dataset))


def _vetor(seed, dim=8):
    return np.random.RandomState(seed).rand(dim).astype(np.float32)


def test_dataset_completo_passa():
    res = _validar(_dataset([_vetor(i) for i in range(5)]))
    assert res.is_valid
    assert not res.has_warnings


def test_nulo_do_parquet_vira_escalar_nan_e_e_barrado():
    # Exatamente o que o carregador produz para a célula nula.
    nulo = np.array(None, dtype=np.float32)
    assert nulo.ndim == 0
    res = _validar(_dataset([_vetor(0), _vetor(1), nulo]))
    assert not res.is_valid
    msg = " ".join(res.errors)
    assert "1 de 3" in msg
    assert "face_2.png" in msg
    assert "ausente" in msg
    assert "dimensão esperada: 8" in msg


@pytest.mark.parametrize("ruim, motivo", [
    (None, "ausente"),
    (np.array([], dtype=np.float32), "ausente"),
    (np.full(8, np.nan, dtype=np.float32), "NaN"),
    (np.ones((2, 4), dtype=np.float32), "ausente"),   # não é 1-D
    (np.ones(4, dtype=np.float32), "dimensão 4"),     # dimensão divergente
])
def test_formas_invalidas(ruim, motivo):
    res = _validar(_dataset([_vetor(0), _vetor(1), _vetor(2), ruim]))
    assert not res.is_valid
    assert motivo in " ".join(res.errors)


def test_mensagem_limita_os_nomes_e_conta_o_resto():
    nulo = np.array(None, dtype=np.float32)
    res = _validar(_dataset([_vetor(0)] + [nulo] * 15), max_named=3)
    msg = " ".join(res.errors)
    assert "15 de 16" in msg
    assert "e mais 12" in msg


def test_recurso_ausente_e_erro_nao_excecao():
    rule = EmbeddingsCompleteRule(rule_id="r", dataset_resource_id="nao_existe")
    res = rule.validate(None, StrategyExecutionContext())
    assert not res.is_valid


def test_parquet_com_embedding_nulo_carregado_pelo_loader_real(tmp_path):
    """Ponta a ponta do carregador: célula nula no parquet → regra barra."""
    linhas = []
    for i in range(3):
        linhas.append({
            "embedding_v2": None if i == 2 else _vetor(i).astype(float).tolist(),
            "hash": f"h{i}", "image_path": f"/img/{i}.png",
            "file_name": f"face_{i}.png", "identity": "A",
        })
    p = tmp_path / "ds.parquet"
    pd.DataFrame(linhas).to_parquet(p)
    ds = EmbeddingDataset.from_parquet(p, embedding_column="embedding_v2")
    assert len(ds.items) == 3  # o carregador NÃO recusa — por isso a regra
    res = _validar(ds)
    assert not res.is_valid
    assert "face_2.png" in " ".join(res.errors)


def test_fabrica_constroi_a_partir_do_config():
    cfg = ValidationRuleConfig(
        rule_type=ValidationRuleType.EMBEDDINGS_COMPLETE,
        params={"dataset_resource_id": "ds"},
    )
    rule = ValidationRuleFactory.create(cfg, instruction_id="i1")
    assert isinstance(rule, EmbeddingsCompleteRule)


def _scoring(**kw) -> SimilarityScoringInstruction:
    return SimilarityScoringInstruction(
        instruction_id="s1", name="",
        instruction_type=StrategyInstructionType.SIMILARITY_SCORING, **kw,
    )


def test_scoring_aplica_a_regra_ao_probe_e_ao_reference():
    instr = _scoring(
        trained_similarity_model_resource_id="m",
        probe_dataset_resource_id="p", reference_dataset_resource_id="q",
    )
    alvos = sorted(
        r._dataset_id for r in instr.build_validation_rules()
        if isinstance(r, EmbeddingsCompleteRule)
    )
    assert alvos == ["p", "q"]


def test_scoring_aplica_a_regra_mesmo_sem_modelo_escolhido():
    instr = _scoring(probe_dataset_resource_id="p")
    assert any(
        isinstance(r, EmbeddingsCompleteRule) for r in instr.build_validation_rules()
    )


def test_auto_comparacao_nao_duplica_a_regra():
    instr = _scoring(
        trained_similarity_model_resource_id="m",
        probe_dataset_resource_id="p", reference_dataset_resource_id="p",
    )
    n = sum(isinstance(r, EmbeddingsCompleteRule) for r in instr.build_validation_rules())
    assert n == 1
