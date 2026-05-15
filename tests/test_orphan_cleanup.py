"""Teste isolado para `_cleanup_orphan_persisted_resources` e
`_delete_persisted_resource_files` do `StrategyManagerV2Controller`.

Não instancia a controller (que constrói widgets Qt). Em vez disso,
constrói um objeto stub com os atributos que os métodos consultam
(`_project`, `_current_definition`, `_execution_context`) e binda os
métodos do `StrategyManagerV2Controller` ao stub via `__get__`.

Cobre os 3 cenários do fix:
  T1: output de instrução renomeado (A→B) — arquivo A vira órfão.
  T2: instrução removida — todos os outputs viram órfãos.
  T3: recurso PROJECT (DATASET/PIPELINE_CONFIG/etc.) NUNCA é apagado.

Mais um caso defensivo:
  T4: mesmo resource_id declarado por OUTRA instrução continua ativo
      (cross-instruction union em `declared_output_ids`).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from types import SimpleNamespace

from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.storage.strategy_artifact_reference import (
    StrategyArtifactReference,
)
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource

from interface.tabs.strategy_manager_v2.strategy_manager_v2_controller import (
    StrategyManagerV2Controller,
)


@dataclass
class _StubInstruction:
    """Instrução-stub: a única coisa que `_cleanup_orphan_persisted_resources`
    consulta é `get_output_resource_placeholders()`. Não precisamos de DTO real."""

    declared_output_ids: List[str] = field(default_factory=list)

    def get_output_resource_placeholders(self) -> List[StrategyResource]:
        return [
            StrategyResource(
                resource_id=rid,
                name=rid,
                resource_type=StrategyResourceType.SIMILARITY_SCORES,
            )
            for rid in self.declared_output_ids
        ]


def _make_stub_controller(project_root: Path, definition: LRStrategyDefinition):
    """Stub que expõe só os atributos+métodos que os dois métodos
    sob teste leem. `_save_current_definition` é noop."""
    stub = SimpleNamespace()
    stub._project = SimpleNamespace(
        storage=SimpleNamespace(project_root=project_root),
    )
    stub._current_definition = definition
    stub._execution_context = None
    stub._save_current_definition = lambda: None

    stub._delete_persisted_resource_files = (
        StrategyManagerV2Controller._delete_persisted_resource_files.__get__(stub)
    )
    stub._cleanup_orphan_persisted_resources = (
        StrategyManagerV2Controller._cleanup_orphan_persisted_resources.__get__(stub)
    )
    return stub


def _make_artifact_file(project_root: Path, rel_path: str) -> Path:
    abs_path = project_root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(b"fake artifact")
    sidecar = abs_path.parent / (abs_path.stem + "_stats.json")
    sidecar.write_text("{}")
    return abs_path


def _make_resource(
    rid: str,
    rel_path: str,
    rtype: StrategyResourceType = StrategyResourceType.SIMILARITY_SCORES,
) -> StrategyResource:
    return StrategyResource(
        resource_id=rid,
        name=rid,
        resource_type=rtype,
        artifact_ref=StrategyArtifactReference(
            artifact_type=StrategyArtifactType.SIMILARITY_SCORES,
            path=rel_path,
        ),
    )


def test_t1_renamed_output_orphan_is_cleaned_up():
    """T1: output renomeado A→B — arquivo A é apagado e resource sai do definition."""
    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)
        scores_old_file = _make_artifact_file(project_root, "scores/scores_old.parquet")
        scores_new_file = _make_artifact_file(project_root, "scores/scores_new.parquet")
        assert scores_old_file.exists() and scores_new_file.exists()

        definition = LRStrategyDefinition(strategy_id="strat-1", name="Test")
        definition.preparation_instructions.append(
            _StubInstruction(declared_output_ids=["scores_new"])
        )
        definition.resources.append(_make_resource("scores_old", "scores/scores_old.parquet"))
        definition.resources.append(_make_resource("scores_new", "scores/scores_new.parquet"))

        stub = _make_stub_controller(project_root, definition)
        stub._cleanup_orphan_persisted_resources()

        assert not scores_old_file.exists(), "Arquivo scores_old.parquet deveria ter sido apagado"
        assert not (project_root / "scores" / "scores_old_stats.json").exists(), \
            "Sidecar _stats.json do órfão deveria ter sido apagado"
        assert scores_new_file.exists(), "Arquivo scores_new.parquet (declarado) deve permanecer"
        assert definition.get_resource_by_id("scores_old") is None, \
            "Resource scores_old deveria sair do definition.resources"
        assert definition.get_resource_by_id("scores_new") is not None, \
            "Resource scores_new (declarado) deve permanecer"
        print("[OK] T1: rename A->B limpou A do disco e do definition.")


def test_t2_removed_instruction_outputs_are_cleaned_up():
    """T2: instrução removida — todos os outputs viram órfãos e são limpos."""
    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)
        out1 = _make_artifact_file(project_root, "scores/out1.parquet")
        out2 = _make_artifact_file(project_root, "scores/out2.parquet")

        # Estratégia sem nenhuma instrução, mas com 2 resources persistidos.
        definition = LRStrategyDefinition(strategy_id="strat-2", name="Test")
        definition.resources.append(_make_resource("out1", "scores/out1.parquet"))
        definition.resources.append(_make_resource("out2", "scores/out2.parquet"))

        stub = _make_stub_controller(project_root, definition)
        stub._cleanup_orphan_persisted_resources()

        assert not out1.exists() and not out2.exists()
        assert definition.get_resource_by_id("out1") is None
        assert definition.get_resource_by_id("out2") is None
        print("[OK] T2: instrucao removida -> todos os outputs orfaos foram limpos.")


def test_t3_project_resources_are_never_deleted():
    """T3: DATASET / PIPELINE_CONFIG / etc. nunca são apagados, mesmo
    sem declaração de output."""
    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)
        dataset_file = _make_artifact_file(project_root, "datasets/d1.parquet")
        pipeline_file = _make_artifact_file(project_root, "pipelines/p1.json")

        definition = LRStrategyDefinition(strategy_id="strat-3", name="Test")
        # Nenhuma instrução declara esses recursos como output — mas eles
        # são tipo PROJECT, então devem sobreviver.
        ds = _make_resource("d1", "datasets/d1.parquet", StrategyResourceType.DATASET)
        pp = _make_resource("p1", "pipelines/p1.json", StrategyResourceType.PIPELINE_CONFIG)
        definition.resources.append(ds)
        definition.resources.append(pp)

        stub = _make_stub_controller(project_root, definition)
        stub._cleanup_orphan_persisted_resources()

        assert dataset_file.exists(), "DATASET não pode ser apagado pelo cleanup"
        assert pipeline_file.exists(), "PIPELINE_CONFIG não pode ser apagado pelo cleanup"
        assert definition.get_resource_by_id("d1") is not None
        assert definition.get_resource_by_id("p1") is not None
        print("[OK] T3: recursos PROJECT preservados.")


def test_t4_multi_instruction_same_output_id_is_preserved():
    """T4: se duas instruções declaram o mesmo output_id, e apenas uma
    foi 'removida' (substituída), o recurso deve permanecer porque
    a OUTRA ainda o declara."""
    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)
        shared = _make_artifact_file(project_root, "scores/shared.parquet")

        definition = LRStrategyDefinition(strategy_id="strat-4", name="Test")
        definition.preparation_instructions.append(
            _StubInstruction(declared_output_ids=["other_output"])
        )
        definition.validation_instructions.append(
            _StubInstruction(declared_output_ids=["shared"])  # ainda declara "shared"
        )
        definition.resources.append(_make_resource("shared", "scores/shared.parquet"))

        stub = _make_stub_controller(project_root, definition)
        stub._cleanup_orphan_persisted_resources()

        assert shared.exists(), \
            "Recurso ainda declarado por OUTRA instrução não pode ser apagado"
        assert definition.get_resource_by_id("shared") is not None
        print("[OK] T4: cross-instruction union preservou recurso compartilhado.")



def test_delete_persisted_resource_files_skips_missing_artifact_ref():
    """Sanity: recurso sem artifact_ref não causa erro nem tenta deletar nada."""
    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)
        definition = LRStrategyDefinition(strategy_id="strat-5", name="Test")
        resource = StrategyResource(
            resource_id="placeholder",
            name="placeholder",
            resource_type=StrategyResourceType.SIMILARITY_SCORES,
        )
        stub = _make_stub_controller(project_root, definition)
        # Não deve lançar.
        stub._delete_persisted_resource_files(resource)
        print("[OK] _delete_persisted_resource_files é no-op pra placeholder puro.")


if __name__ == "__main__":
    test_t1_renamed_output_orphan_is_cleaned_up()
    test_t2_removed_instruction_outputs_are_cleaned_up()
    test_t3_project_resources_are_never_deleted()
    test_t4_multi_instruction_same_output_id_is_preserved()
    test_delete_persisted_resource_files_skips_missing_artifact_ref()
    print("\nTodos os testes passaram.")
