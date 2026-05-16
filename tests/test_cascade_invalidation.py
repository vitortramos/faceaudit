"""Teste isolado pra cascade transitivo na exclusão de instrução
do `StrategyManagerV2Controller`.

Setup: cadeia de instruções A → B → C onde cada uma consome o output
da anterior. User remove A → verifica que TODA a cadeia downstream é
invalidada corretamente:

  - Arquivos físicos de TODOS os outputs apagados em disco.
  - `out_a` removido de `defn.resources` (instrução geradora deixou de
    existir).
  - `out_b`, `out_c` mantidos como placeholders em `defn.resources`
    mas com `artifact_ref=None`/`payload=None` (B e C ainda existem
    como instruções, podem ser re-executadas).
  - `_executed_instruction_ids` perde ✅ de B e C via cascade.
  - Context vivo limpo de todos os outputs.

Não instancia a controller (que constrói widgets Qt). Usa subclasse
com `__init__` minimalista e mocks pros refresh methods.

Cobre também:
  - Cadeia linear (A → B → C).
  - Diamond dependency: A produz out_a; B e C ambos consomem out_a;
    D consome out_b e out_c. Remover A invalida B, C, D.
  - Múltiplos outputs por instrução: A produz out_a1 + out_a2;
    B consome só out_a1 (cascade pega), C não consome nada de A
    (cascade pula).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

from faceaudit.contexts.strategy_execution_context import StrategyExecutionContext
from faceaudit.enums.resource_category import ResourceCategory
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
)
from faceaudit.models.storage.strategy_artifact_reference import (
    StrategyArtifactReference,
)
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource

from interface.tabs.strategy_manager_v2.strategy_manager_v2_controller import (
    StrategyManagerV2Controller,
)


# ─── Mock de instrução ────────────────────────────────────────────────────


class _MockInstr:
    """Instrução-stub: o cascade consulta `get_resource_field_specs()` pra
    descobrir output fields + `input_resource_ids` pra alimentar o BFS.

    Não usamos DTO real (`StrategyInstructionDefinition`) pra evitar
    dependência dos schemas concretos no teste."""

    def __init__(self, instr_id: str, inputs: List[str], outputs: List[str]):
        self.instruction_id = instr_id
        self.input_resource_ids = list(inputs)
        self.output_resource_ids = list(outputs)
        self._outputs = list(outputs)

    def get_resource_field_specs(self) -> List[ResourceFieldSpec]:
        return [
            ResourceFieldSpec(
                field_name=f"_out_{i}",
                label=f"out {i}",
                accepted_types=(StrategyResourceType.SIMILARITY_SCORES,),
                is_output=True,
            )
            for i in range(len(self._outputs))
        ]

    def __getattr__(self, name):
        # Permite `getattr(instr, spec.field_name)` retornar o output_id.
        if name.startswith("_out_"):
            idx = int(name[5:])
            return self._outputs[idx] if idx < len(self._outputs) else None
        raise AttributeError(name)


# ─── Controller-stub que herda do real ───────────────────────────────────


class _TestController(StrategyManagerV2Controller):
    """Subclasse minimalista: skip do __init__ do real (que toca Qt) e
    setup dos atributos que o cascade usa.
    """

    def __init__(self, project_root: Path, defn: LRStrategyDefinition):
        # NÃO chamamos super().__init__ — toca Qt e wireia signals.
        self._project = MagicMock()
        self._project.storage.project_root = project_root
        self._project.definition_repository.save_definition = MagicMock()
        self._current_definition = defn
        self._definitions = [defn]
        self._executed_instruction_ids = {
            instr.instruction_id for instr in defn.instructions
        }
        self._execution_context = StrategyExecutionContext()
        self._view = MagicMock()
        self._is_dirty = False
        self._strategy_editor = MagicMock()
        self._left_panel = MagicMock()
        self._left_panel.resource_info = MagicMock()

    # Override dos refresh methods — são chamados ao fim do cascade mas
    # não têm o que rodar sem widget real. No-ops.
    def _refresh_instruction_states(self) -> None:
        pass

    def _refresh_produced_resources(self) -> None:
        pass

    def _refresh_left_panel(self) -> None:
        pass


# ─── Helpers ──────────────────────────────────────────────────────────────


def _mk_resource(rid: str, rel_path: str) -> StrategyResource:
    """Constrói um StrategyResource DERIVED com artifact_ref + payload populados."""
    res = StrategyResource(
        resource_id=rid,
        name=rid,
        resource_type=StrategyResourceType.SIMILARITY_SCORES,
        category=ResourceCategory.DERIVED,
    )
    res.artifact_ref = StrategyArtifactReference(
        artifact_type=StrategyArtifactType.SIMILARITY_SCORES,
        path=rel_path,
        hash="dummy_hash",
    )
    res.payload = "dummy_payload"
    return res


def _build_chain_scenario(tmp_path: Path):
    """Setup compartilhado: cria 3 arquivos físicos + defn com cadeia
    A → B → C registrando 3 outputs e adicionando ao context."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    file_a = artifacts_dir / "out_a.pkl"
    file_b = artifacts_dir / "out_b.pkl"
    file_c = artifacts_dir / "out_c.pkl"
    for f in (file_a, file_b, file_c):
        f.write_text("dummy")

    defn = LRStrategyDefinition(strategy_id="ST_cascade", name="Cascade Test")
    res_a = _mk_resource("out_a", "artifacts/out_a.pkl")
    res_b = _mk_resource("out_b", "artifacts/out_b.pkl")
    res_c = _mk_resource("out_c", "artifacts/out_c.pkl")
    defn.add_resource(res_a)
    defn.add_resource(res_b)
    defn.add_resource(res_c)

    instr_A = _MockInstr("instr_A", inputs=[], outputs=["out_a"])
    instr_B = _MockInstr("instr_B", inputs=["out_a"], outputs=["out_b"])
    instr_C = _MockInstr("instr_C", inputs=["out_b"], outputs=["out_c"])
    defn._get_phase_list("preparation").extend([instr_A, instr_B, instr_C])

    ctrl = _TestController(tmp_path, defn)
    ctrl._execution_context.set_resource(res_a)
    ctrl._execution_context.set_resource(res_b)
    ctrl._execution_context.set_resource(res_c)

    return ctrl, defn, instr_A, instr_B, instr_C, file_a, file_b, file_c


# ─── Tests ────────────────────────────────────────────────────────────────


def test_cascade_linear_chain():
    """Cadeia A → B → C. Remover A deve invalidar tudo recursivamente."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (ctrl, defn, instr_A, instr_B, instr_C,
         file_a, file_b, file_c) = _build_chain_scenario(tmp_path)

        # Pre-condições
        assert file_a.exists() and file_b.exists() and file_c.exists()
        assert {"instr_A", "instr_B", "instr_C"} == ctrl._executed_instruction_ids

        # Simula remoção via widget: A sai da fase_list ANTES do callback.
        defn._get_phase_list("preparation").remove(instr_A)
        ctrl._on_instruction_removed(instr_A)

        # out_a (output direto da instrução removida): SUMIU da defn
        assert defn.get_resource_by_id("out_a") is None

        # out_b, out_c (cascade): placeholders MANTIDOS mas refs zeradas
        rb = defn.get_resource_by_id("out_b")
        rc = defn.get_resource_by_id("out_c")
        assert rb is not None and rc is not None
        assert rb.artifact_ref is None and rb.payload is None
        assert rc.artifact_ref is None and rc.payload is None

        # TODOS os arquivos físicos apagados (cascade transitivo)
        assert not file_a.exists()
        assert not file_b.exists()
        assert not file_c.exists()

        # Context vivo limpo
        assert not ctrl._execution_context.has_resource("out_a")
        assert not ctrl._execution_context.has_resource("out_b")
        assert not ctrl._execution_context.has_resource("out_c")

        # ✅ removido de B e C via cascade (e de A pela remoção direta)
        assert ctrl._executed_instruction_ids == set()

        # Definition persistida
        assert ctrl._project.definition_repository.save_definition.called


def test_cascade_diamond_dependency():
    """Diamond: A produz out_a; B e C ambos consomem out_a (paralelos);
    D consome out_b E out_c (re-converge). Remover A → invalida B, C, D."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        # 4 outputs: out_a, out_b, out_c, out_d
        files = {
            "out_a": artifacts / "out_a.pkl",
            "out_b": artifacts / "out_b.pkl",
            "out_c": artifacts / "out_c.pkl",
            "out_d": artifacts / "out_d.pkl",
        }
        for f in files.values():
            f.write_text("dummy")

        defn = LRStrategyDefinition(strategy_id="ST_diamond", name="Diamond")
        for rid, f in files.items():
            defn.add_resource(_mk_resource(rid, f"artifacts/{f.name}"))

        instr_A = _MockInstr("A", inputs=[], outputs=["out_a"])
        instr_B = _MockInstr("B", inputs=["out_a"], outputs=["out_b"])
        instr_C = _MockInstr("C", inputs=["out_a"], outputs=["out_c"])
        instr_D = _MockInstr("D", inputs=["out_b", "out_c"], outputs=["out_d"])
        defn._get_phase_list("preparation").extend([instr_A, instr_B, instr_C, instr_D])

        ctrl = _TestController(tmp_path, defn)
        for rid in files:
            ctrl._execution_context.set_resource(defn.get_resource_by_id(rid))

        defn._get_phase_list("preparation").remove(instr_A)
        ctrl._on_instruction_removed(instr_A)

        # out_a removido da defn
        assert defn.get_resource_by_id("out_a") is None
        # out_b/c/d placeholders mantidos, refs zeradas
        for rid in ("out_b", "out_c", "out_d"):
            r = defn.get_resource_by_id(rid)
            assert r is not None, f"placeholder {rid} sumiu"
            assert r.artifact_ref is None
            assert r.payload is None
        # Todos os arquivos físicos apagados
        for rid, f in files.items():
            assert not f.exists(), f"{rid}.pkl deveria estar apagado"
        # Context limpo
        for rid in files:
            assert not ctrl._execution_context.has_resource(rid)
        # ✅ removido de todos
        assert ctrl._executed_instruction_ids == set()


def test_cascade_multiple_outputs_partial_dependency():
    """A produz 2 outputs (out_a1, out_a2). B consome só out_a1.
    C é independente (não consome nada de A). Remover A:
      - apaga out_a1, out_a2 do disco.
      - cascade pega B (consome out_a1) → invalida out_b.
      - C permanece com ✅ (não dependente).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        files = {
            "out_a1": artifacts / "out_a1.pkl",
            "out_a2": artifacts / "out_a2.pkl",
            "out_b":  artifacts / "out_b.pkl",
            "out_c":  artifacts / "out_c.pkl",
        }
        for f in files.values():
            f.write_text("dummy")

        defn = LRStrategyDefinition(strategy_id="ST_partial", name="Partial")
        for rid, f in files.items():
            defn.add_resource(_mk_resource(rid, f"artifacts/{f.name}"))

        instr_A = _MockInstr("A", inputs=[], outputs=["out_a1", "out_a2"])
        instr_B = _MockInstr("B", inputs=["out_a1"], outputs=["out_b"])
        # C consome out_c_input que não é nada produzido aqui — independente.
        instr_C = _MockInstr("C", inputs=["external_input"], outputs=["out_c"])
        defn._get_phase_list("preparation").extend([instr_A, instr_B, instr_C])

        ctrl = _TestController(tmp_path, defn)
        for rid in files:
            ctrl._execution_context.set_resource(defn.get_resource_by_id(rid))

        defn._get_phase_list("preparation").remove(instr_A)
        ctrl._on_instruction_removed(instr_A)

        # Outputs de A removidos da defn
        assert defn.get_resource_by_id("out_a1") is None
        assert defn.get_resource_by_id("out_a2") is None
        # B invalidado por cascade (out_b mantido como placeholder)
        rb = defn.get_resource_by_id("out_b")
        assert rb is not None and rb.artifact_ref is None
        # C NÃO afetado — placeholder intacto
        rc = defn.get_resource_by_id("out_c")
        assert rc is not None
        assert rc.artifact_ref is not None, \
            "C não consome nada de A — deveria continuar intacto"
        # Arquivos de A + out_b apagados; out_c intacto
        assert not files["out_a1"].exists()
        assert not files["out_a2"].exists()
        assert not files["out_b"].exists()
        assert files["out_c"].exists(), "out_c.pkl deveria estar intacto"
        # ✅ removido de A e B; C preservado
        assert "B" not in ctrl._executed_instruction_ids
        assert "A" not in ctrl._executed_instruction_ids
        assert "C" in ctrl._executed_instruction_ids


def test_cascade_via_id_or_resource_id():
    """Regressão da Fase 4: remove_resource via id técnico OU resource_id
    legado limpa ambas as chaves do contexto + _id_to_resource_id_index."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ctrl, defn, instr_A, *_, file_a, file_b, file_c = _build_chain_scenario(
            tmp_path
        )

        res_a = defn.get_resource_by_id("out_a")
        legacy_id = res_a.resource_id  # "out_a"
        technical_id = res_a.id        # "SS-xxxxxxxx"

        # Lookup duplo confirmado
        assert ctrl._execution_context.has_resource(legacy_id)
        assert ctrl._execution_context.has_resource(technical_id)

        # Remove via id técnico — ambas chaves limpas
        ctrl._execution_context.remove_resource(technical_id)
        assert not ctrl._execution_context.has_resource(legacy_id)
        assert not ctrl._execution_context.has_resource(technical_id)
        assert technical_id not in ctrl._execution_context._id_to_resource_id_index
