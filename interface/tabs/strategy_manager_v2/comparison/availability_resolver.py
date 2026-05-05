# interface/tabs/strategy_manager_v2/comparison/availability_resolver.py

"""
Resolve quais (estratégia, recurso) têm dado disponível para cada tipo de
comparação. Single source of truth para o filtro do left panel + para a
seleção de itens consumida pelas views.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.models.instructions.casework_visualization_instruction import (
    CaseworkVisualizationInstruction,
)
from faceaudit.models.instructions.strategy_evaluation_instruction import (
    StrategyEvaluationInstruction,
)
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition

from .comparison_kind import ComparisonKind


@dataclass(frozen=True)
class ComparisonItem:
    """
    Identifica unicamente uma instância passível de comparação.

    Um par (estratégia, recurso). Para uma mesma estratégia podem existir
    múltiplos itens do mesmo kind (ex: dois STRATEGY_EVALUATION com
    VALIDATION_RESULTs diferentes).
    """
    strategy_id: str
    strategy_name: str
    resource_id: str
    # Nome amigável do recurso. Se a definition não tiver nome distinto,
    # cai no resource_id.
    resource_name: str
    # Caminho ABSOLUTO do artefato em disco (já resolvido por project_root).
    # None se o recurso ainda não foi persistido — nesse caso o item NÃO
    # entra na lista (é filtrado fora pelo resolver).
    artifact_path: Optional[Path]


def list_available_items(
    kind: ComparisonKind,
    definitions: List[LRStrategyDefinition],
    project_root: Path,
) -> List[ComparisonItem]:
    """
    Retorna a lista de itens disponíveis para o `kind` informado.

    Itens disponíveis = (estratégia, recurso) onde:
    - a estratégia tem ao menos uma instrução do tipo correspondente, E
    - o recurso de output dessa instrução está persistido em disco
      (artifact_ref existente + arquivo presente).

    Ordem: por nome de estratégia, depois por nome de recurso (case-insensitive).
    """
    if kind == ComparisonKind.CALIBRATION_DATASET_DISTRIBUTION:
        items = _list_calibration_distribution_items(definitions, project_root)
    elif kind == ComparisonKind.VALIDATION_RESULTS:
        items = _list_validation_items(definitions, project_root)
    elif kind == ComparisonKind.CASEWORK_RESULTS:
        items = _list_casework_items(definitions, project_root)
    else:
        items = []

    items.sort(
        key=lambda it: (it.strategy_name.lower(), it.resource_name.lower())
    )
    return items


# ── Por kind ──────────────────────────────────────────────────────────────────


def _list_calibration_distribution_items(
    definitions: List[LRStrategyDefinition],
    project_root: Path,
) -> List[ComparisonItem]:
    """
    Para cada CALIBRATION_MODEL_TRAINING em qualquer estratégia, pega o
    `input_resource_id` (dataset de scores que alimentou o calibrador) e
    valida que o parquet está no disco. Cada calibrador vira um item.
    """
    items: List[ComparisonItem] = []
    for defn in definitions:
        for instr in defn.instructions:
            if instr.instruction_type != StrategyInstructionType.CALIBRATION_MODEL_TRAINING:
                continue
            input_id = getattr(instr, "input_resource_id", None)
            if not input_id:
                continue
            res = defn.get_resource_by_id(input_id)
            if res is None or not res.has_artifact:
                continue
            abs_path = _resolve_artifact_path(res, project_root)
            if abs_path is None:
                continue
            items.append(ComparisonItem(
                strategy_id=defn.strategy_id,
                strategy_name=defn.name,
                resource_id=input_id,
                resource_name=res.name or input_id,
                artifact_path=abs_path,
            ))
    return items


def _list_validation_items(
    definitions: List[LRStrategyDefinition],
    project_root: Path,
) -> List[ComparisonItem]:
    """Cada STRATEGY_EVALUATION com VALIDATION_RESULT persistido vira um item."""
    items: List[ComparisonItem] = []
    for defn in definitions:
        for instr in defn.instructions:
            if not isinstance(instr, StrategyEvaluationInstruction):
                continue
            output_id = instr.output_validation_result_resource_id
            if not output_id:
                continue
            res = defn.get_resource_by_id(output_id)
            if res is None or not res.has_artifact:
                continue
            abs_path = _resolve_artifact_path(res, project_root)
            if abs_path is None:
                continue
            items.append(ComparisonItem(
                strategy_id=defn.strategy_id,
                strategy_name=defn.name,
                resource_id=output_id,
                resource_name=res.name or output_id,
                artifact_path=abs_path,
            ))
    return items


def _list_casework_items(
    definitions: List[LRStrategyDefinition],
    project_root: Path,
) -> List[ComparisonItem]:
    """Cada CASEWORK_VISUALIZATION com result persistido vira um item."""
    items: List[ComparisonItem] = []
    for defn in definitions:
        for instr in defn.instructions:
            if not isinstance(instr, CaseworkVisualizationInstruction):
                continue
            output_id = instr.output_visualization_resource_id
            if not output_id:
                continue
            res = defn.get_resource_by_id(output_id)
            if res is None or not res.has_artifact:
                continue
            abs_path = _resolve_artifact_path(res, project_root)
            if abs_path is None:
                continue
            items.append(ComparisonItem(
                strategy_id=defn.strategy_id,
                strategy_name=defn.name,
                resource_id=output_id,
                resource_name=res.name or output_id,
                artifact_path=abs_path,
            ))
    return items


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_artifact_path(res, project_root: Path) -> Optional[Path]:
    """
    Resolve o path absoluto e confirma que o arquivo existe. Normaliza
    `\\` → `/` (legacy Windows refs) — pathlib aceita `/` em qualquer SO.
    """
    if not res.artifact_ref or not res.artifact_ref.path:
        return None
    rel = res.artifact_ref.path.replace("\\", "/")
    abs_path = Path(project_root) / rel
    return abs_path if abs_path.exists() else None
