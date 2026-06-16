"""
Testes do `InstructionExecutionFrame` e do
`StrategyExecutionContext.with_frame`.

Cobre:
- Frame imutável (frozen).
- with_frame retorna view com mesmo estado mutável (resources etc.).
- Mutações via uma view se propagam à outra (compartilhamento de estado).
- `current_frame` é excluído de to_dict/from_dict (efêmero).
"""

from __future__ import annotations

import pytest

from faceaudit.contexts.strategy_execution_context import StrategyExecutionContext
from faceaudit.enums.strategy_phase import StrategyPhase
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.strategy.instruction_execution_frame import InstructionExecutionFrame
from faceaudit.models.strategy.strategy_resource import StrategyResource


class TestFrameImmutability:
    def test_frame_is_frozen(self):
        f = InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="abc",
        )
        with pytest.raises(Exception):
            f.phase = StrategyPhase.VALIDATION

    def test_frame_carries_fields(self):
        f = InstructionExecutionFrame(
            phase=StrategyPhase.CASEWORK, instruction_index=3, strategy_id="abc-123",
        )
        assert f.phase == StrategyPhase.CASEWORK
        assert f.instruction_index == 3
        assert f.strategy_id == "abc-123"


class TestContextWithFrame:
    def test_with_frame_returns_view_with_shared_resources(self):
        ctx = StrategyExecutionContext()
        res = StrategyResource(
            resource_id="a", name="A",
            resource_type=StrategyResourceType.DATASET, payload=None,
        )
        ctx.set_resource(res)
        f = InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s",
        )
        ctx_with = ctx.with_frame(f)
        assert ctx_with.frame == f
        # Resources dict é o MESMO objeto — view compartilha estado mutável
        assert ctx_with.resources is ctx.resources

    def test_mutation_via_view_propagates_to_root(self):
        ctx = StrategyExecutionContext()
        f = InstructionExecutionFrame(
            phase=StrategyPhase.VALIDATION, instruction_index=1, strategy_id="s",
        )
        ctx_with = ctx.with_frame(f)
        new_res = StrategyResource(
            resource_id="x", name="X",
            resource_type=StrategyResourceType.DATASET, payload=None,
        )
        ctx_with.set_resource(new_res)
        # Resource adicionado via view aparece no contexto original
        assert ctx.get_resource("x") is new_res

    def test_root_context_has_none_frame(self):
        ctx = StrategyExecutionContext()
        assert ctx.frame is None
        assert ctx.current_frame is None


class TestSerialization:
    def test_to_dict_excludes_current_frame(self):
        ctx = StrategyExecutionContext()
        f = InstructionExecutionFrame(
            phase=StrategyPhase.CASEWORK, instruction_index=0, strategy_id="s",
        )
        ctx_with = ctx.with_frame(f)
        d = ctx_with.to_dict()
        assert "current_frame" not in d

    def test_from_dict_ignores_current_frame_if_present(self):
        ctx = StrategyExecutionContext()
        d = ctx.to_dict()
        d["current_frame"] = {"phase": "PREPARATION", "instruction_index": 0, "strategy_id": "x"}
        back = StrategyExecutionContext.from_dict(d)
        assert back.current_frame is None
