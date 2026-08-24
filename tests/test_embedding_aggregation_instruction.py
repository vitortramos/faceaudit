"""Instrução EMBEDDING_AGGREGATION — DTO, registro e executor.

Instrução única stateless: a agregação não estima parâmetro nenhum a partir
dos dados, então não existe o par PIPELINE_TRAINING + PIPELINE_APPLICATION —
é uma função pura DATASET→DATASET.

Cobre os pontos que, se esquecidos, fazem a instrução falhar em SILÊNCIO:
- dispatch ausente em `StrategyInstructionDefinition.from_dict` → a instrução
  desaparece da definition ao reabrir o projeto, sem erro;
- registro ausente na factory → o executor não resolve;
- `.value` de Smart Enum vazando para `ValidationRuleConfig.params` → o save
  da definition quebra com "Object of type function is not JSON serializable".
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from faceaudit.contexts.strategy_execution_context import StrategyExecutionContext
from faceaudit.enums.aggregation_method import AggregationMethod
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.executors.instructions.embedding_aggregation_instruction_executor import (
    EmbeddingAggregationInstructionExecutor,
)
from faceaudit.factories.strategy_executor_factory import StrategyExecutorFactory
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.models.instructions.embedding_aggregation_instruction import (
    EmbeddingAggregationInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource


def _face(i: int, identity: str) -> EmbeddingInfo:
    return EmbeddingInfo(
        embedding=np.random.RandomState(i).rand(8), face_hash=f"real{i:03d}",
        image_path=f"/fotos/{i}.jpg", identity=identity, file_name=f"{i}.jpg",
        quality_scores={"fiqa": 50.0 + i},
    )


def _contexto(dataset: EmbeddingDataset) -> StrategyExecutionContext:
    ctx = StrategyExecutionContext()
    ctx.set_resource(StrategyResource(
        resource_id="ds_in", name="entrada",
        resource_type=StrategyResourceType.DATASET, payload=dataset,
    ))
    return ctx


def _instrucao(**kw) -> EmbeddingAggregationInstruction:
    # `name` e `instruction_type` são posicionais na base (sem default) — a UI
    # os fornece via `StrategyInstructionDefinition.from_dict`. O
    # `__post_init__` reescreve ambos, mas o dataclass ainda os exige.
    kw.setdefault("dataset_resource_id", "ds_in")
    kw.setdefault("output_dataset_resource_id", "ds_out")
    return EmbeddingAggregationInstruction(
        instruction_id="i1",
        name="",
        instruction_type=StrategyInstructionType.EMBEDDING_AGGREGATION,
        **kw,
    )


def _dataset(n_por_id=3, ids=("A", "B")) -> EmbeddingDataset:
    itens = []
    k = 0
    for ident in ids:
        for _ in range(n_por_id):
            itens.append(_face(k, ident))
            k += 1
    return EmbeddingDataset(name="base", items=itens)


# ───────────────────────────── registro ─────────────────────────────

def test_tipo_existe_no_enum_e_resolve_por_string():
    t = StrategyInstructionType.EMBEDDING_AGGREGATION
    assert t.value == "EMBEDDING_AGGREGATION"
    assert StrategyInstructionType.from_str("EMBEDDING_AGGREGATION") is t
    assert t.label_pt and t.description


def test_dispatch_hidrata_a_classe_concreta():
    """Sem o branch no dispatch, `from_dict` devolve a base genérica e a
    instrução some da definition ao reabrir o projeto — sem erro nenhum."""
    instr = StrategyInstructionDefinition.from_dict({
        "instruction_id": "i1", "name": "",
        "instruction_type": "EMBEDDING_AGGREGATION",
    })
    assert isinstance(instr, EmbeddingAggregationInstruction)
    assert instr.instruction_type is StrategyInstructionType.EMBEDDING_AGGREGATION
    assert instr.name, "deve receber nome default"
    assert instr.aggregation_config is not None, "config não pode nascer None"


def test_executor_esta_registrado_na_factory():
    registry = StrategyExecutorFactory.build_instruction_executor_registry()
    executor = registry.resolve(StrategyInstructionType.EMBEDDING_AGGREGATION)
    assert isinstance(executor, EmbeddingAggregationInstructionExecutor)


# ─────────────────────────────── DTO ───────────────────────────────

def test_io_declarado_reflete_os_campos():
    instr = _instrucao()
    assert instr.input_resource_ids == ["ds_in"]
    assert instr.output_resource_ids == ["ds_out"]


def test_saida_e_dataset_e_nao_um_tipo_extinto():
    """INTERMEDIATE_DATASET não existe mais; a distinção primário/derivado
    vive em `category`, cujo default DERIVED é o que permite persistir."""
    spec = next(s for s in _instrucao().get_resource_field_specs() if s.is_output)
    assert spec.accepted_types[0] is StrategyResourceType.DATASET


def test_uma_unica_identidade_e_caso_valido():
    """Casework: fundir as fotos de referência de UM indivíduo é o uso mais
    natural da agregação. O default da regra (min_required=2) vem do fit do
    PLDA e barraria esse caso — por isso a instrução o sobrescreve."""
    ctx = _contexto(_dataset(n_por_id=3, ids=("A",)))
    resultado = EmbeddingAggregationInstructionExecutor().execute(_instrucao(), ctx)
    assert resultado.success, resultado.errors
    assert len(ctx.get_resource("ds_out").payload.items) == 1


def test_nome_de_saida_nao_e_duplicado_na_secao_de_configuracao():
    """O nome da saída se define em "Configurações de saída". Expô-lo também
    como ValueFieldSpec deixava dois campos de nome na mesma tela — um em
    Configuração e outro em Saída. Mesma escolha da DatasetBalancingInstruction,
    que tem o campo e não o expõe."""
    instr = _instrucao()
    nomes = {s.field_name for s in (instr.get_value_field_specs() or [])}
    assert "output_dataset_name" not in nomes

    # E o campo continua existindo no DTO — o executor o honra quando definido.
    assert hasattr(instr, "output_dataset_name")


def test_ajuda_contextual_e_dica_curta_e_nao_documentacao():
    """A caixa âmbar fica no topo da seção e é a primeira coisa a ser cortada
    quando a coluna estreita. A explicação longa vive no `description` do tipo
    de instrução, que o painel usa como tooltip."""
    ajuda = _instrucao().get_configuration_help()
    assert ajuda and len(ajuda) < 320, f"ajuda longa demais ({len(ajuda)} chars)"
    # O essencial permanece: a saída não é dataset de fotos.
    assert "templates" in ajuda
    # E o detalhamento continua disponível onde há espaço para ele.
    assert len(StrategyInstructionType.EMBEDDING_AGGREGATION.description) > 500


def test_regras_default_acompanham_a_configuracao():
    instr = _instrucao()
    tipos = [c.rule_type for c in instr.validation_rule_configs]
    assert ValidationRuleType.MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES in tipos
    assert ValidationRuleType.QUALITY_METRIC_AVAILABLE not in tipos

    instr.aggregation_config.use_quality_weighting = True
    instr.aggregation_config.method = AggregationMethod.COMBINATORIAL
    instr._sync_declared_io()
    tipos = [c.rule_type for c in instr.validation_rule_configs]
    assert ValidationRuleType.QUALITY_METRIC_AVAILABLE in tipos
    assert ValidationRuleType.MAX_SAMPLES_WARNING in tipos


def test_params_de_regra_sao_serializaveis_em_json():
    """`QualityScoreType.value` é a tupla (nome, faixa, LAMBDA). Usar `.value`
    em params vaza a função pro JSON e quebra o save da definition."""
    instr = _instrucao()
    instr.aggregation_config.use_quality_weighting = True
    instr.aggregation_config.quality_weight_type = QualityScoreType.OFIQ
    instr._sync_declared_io()

    json.dumps([c.to_dict() for c in instr.validation_rule_configs])  # não pode levantar
    cfg = next(c for c in instr.validation_rule_configs
               if c.rule_type == ValidationRuleType.QUALITY_METRIC_AVAILABLE)
    assert cfg.params["quality_metric"] == "OFIQ"


def test_round_trip_json_preserva_a_configuracao():
    instr = _instrucao()
    instr.aggregation_config.method = AggregationMethod.COMBINATORIAL
    instr.aggregation_config.combination_group_size = 4

    voltou = StrategyInstructionDefinition.from_dict(json.loads(json.dumps(instr.to_dict())))
    assert isinstance(voltou, EmbeddingAggregationInstruction)
    assert voltou.aggregation_config.method is AggregationMethod.COMBINATORIAL
    assert voltou.aggregation_config.combination_group_size == 4


def test_respeita_exclusao_de_regra_pelo_usuario():
    instr = _instrucao()
    instr.validation_rule_configs = []
    instr.validation_rules_user_managed = True
    instr._sync_declared_io()
    assert instr.validation_rule_configs == []


# ───────────────────────────── executor ─────────────────────────────

def test_executa_e_produz_recurso_de_dataset():
    ctx = _contexto(_dataset())
    resultado = EmbeddingAggregationInstructionExecutor().execute(_instrucao(), ctx)

    assert resultado.success, resultado.errors
    assert resultado.produced_resource_ids == ["ds_out"]

    saida = ctx.get_resource("ds_out")
    assert saida.resource_type is StrategyResourceType.DATASET
    assert len(saida.payload.items) == 2                    # CENTROID: 1 por identidade
    assert {i.identity for i in saida.payload.items} == {"A", "B"}


def test_execucao_registra_metadados_de_auditoria():
    ctx = _contexto(_dataset())
    meta = EmbeddingAggregationInstructionExecutor().execute(_instrucao(), ctx).metadata
    assert meta["aggregation_method"] == "centroid"
    assert meta["input_size"] == 6 and meta["input_n_ids"] == 2
    assert meta["output_size"] == 2


def test_entrada_nao_e_mutada():
    ds = _dataset()
    antes = [i.face_hash for i in ds.items]
    EmbeddingAggregationInstructionExecutor().execute(_instrucao(), _contexto(ds))
    assert [i.face_hash for i in ds.items] == antes
    assert len(ds.items) == 6


def test_nome_de_saida_customizado_vence():
    ctx = _contexto(_dataset())
    instr = _instrucao(output_dataset_name="templates_periciais")
    EmbeddingAggregationInstructionExecutor().execute(instr, ctx)
    assert ctx.get_resource("ds_out").payload.name == "templates_periciais"


def test_validacao_rejeita_identidade_ausente_antes_de_executar():
    """A checagem vive TAMBÉM no executor (e não só no service) para que a
    falha apareça na validação da UI, antes de o usuário disparar a execução."""
    ds = _dataset()
    ds.items[0].identity = None
    resultado = EmbeddingAggregationInstructionExecutor().validate(
        instruction=_instrucao(), context=_contexto(ds),
    )
    assert not resultado.is_valid
    assert any("identidade" in e.lower() for e in resultado.errors)


def test_validacao_rejeita_k_invalido_em_combinatoria():
    instr = _instrucao()
    instr.aggregation_config.method = AggregationMethod.COMBINATORIAL
    instr.aggregation_config.combination_group_size = 1
    resultado = EmbeddingAggregationInstructionExecutor().validate(
        instruction=instr, context=_contexto(_dataset()),
    )
    assert not resultado.is_valid
    assert any("combination_group_size" in e for e in resultado.errors)


def test_validacao_avisa_quando_saida_sobrescreve_a_entrada():
    instr = _instrucao()
    instr.output_dataset_resource_id = "ds_in"
    instr._sync_declared_io()
    resultado = EmbeddingAggregationInstructionExecutor().validate(
        instruction=instr, context=_contexto(_dataset()),
    )
    assert resultado.has_warnings


def test_validacao_rejeita_recurso_inexistente():
    instr = _instrucao(dataset_resource_id="nao_existe")
    resultado = EmbeddingAggregationInstructionExecutor().validate(
        instruction=instr, context=_contexto(_dataset()),
    )
    assert not resultado.is_valid


def test_combinatoria_pelo_executor_gera_multiplos_templates():
    ctx = _contexto(_dataset(n_por_id=3, ids=("A",)))
    instr = _instrucao()
    instr.aggregation_config.method = AggregationMethod.COMBINATORIAL
    instr.aggregation_config.combination_group_size = 2
    instr._sync_declared_io()

    resultado = EmbeddingAggregationInstructionExecutor().execute(instr, ctx)
    assert resultado.success, resultado.errors
    itens = ctx.get_resource("ds_out").payload.items
    assert len(itens) == 3                                  # C(3,2)
    assert len({i.face_hash for i in itens}) == 3


def test_saida_nao_carrega_metadado_de_foto_e_anota_proveniencia():
    ctx = _contexto(_dataset())
    EmbeddingAggregationInstructionExecutor().execute(_instrucao(), ctx)
    saida = ctx.get_resource("ds_out").payload

    for t in saida.items:
        assert t.image_path == "" and t.pose == {} and t.quality_scores == {}
    assert saida.loaded_metadata["aggregation_applied"] == "true"


def test_executor_rejeita_instrucao_de_outro_tipo():
    from faceaudit.models.instructions.dataset_balancing_instruction import (
        DatasetBalancingInstruction,
    )
    outra = DatasetBalancingInstruction(
        instruction_id="x", name="",
        instruction_type=StrategyInstructionType.DATASET_BALANCING,
        dataset_resource_id="ds_in",
    )
    resultado = EmbeddingAggregationInstructionExecutor().execute(outra, _contexto(_dataset()))
    assert not resultado.success
