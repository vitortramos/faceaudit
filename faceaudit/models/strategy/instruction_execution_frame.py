# faceaudit/models/strategy/instruction_execution_frame.py
"""
Frame de execução de uma instrução: fase + índice + identificador de
estratégia.

Imutável (`frozen=True`). Construído pelo `StrategyExecutor` antes de
cada `executor.execute(instr, context)` e injetado via
`context.with_frame(frame)` — view do contexto que carrega o frame sem
mutar o estado global.

Executores legados ignoram o frame (assinatura `execute(instr, context)`
inalterada). O novo `LRArchitectureVectorizationInstructionExecutor`
fail-closes quando `context.frame is None` para evitar ajuste silencioso
de s₀/r₀ em casework por falha do orquestrador.
"""

from __future__ import annotations

from dataclasses import dataclass

from faceaudit.enums.strategy_phase import StrategyPhase


@dataclass(frozen=True)
class InstructionExecutionFrame:
    """
    Identifica onde uma instrução está rodando dentro do fluxo da
    estratégia: fase (PREPARATION/VALIDATION/CASEWORK), índice
    sequencial dentro da fase, e UUID da estratégia.

    Usado por executores que precisam de comportamento dependente da
    fase (e.g., o novo `LR_ARCHITECTURE_VECTORIZATION` bloqueia ajuste
    de parâmetros em VALIDATION/CASEWORK).
    """

    phase: StrategyPhase
    instruction_index: int
    strategy_id: str
