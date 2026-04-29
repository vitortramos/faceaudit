# faceaudit/executors/instructions/pair_mining_instruction_executor.py

from __future__ import annotations

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.pair_mining_instruction import PairMiningInstruction
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.comparison_mining_service import ComparisonMiningService


class PairMiningInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de curadoria de pares (PAIR_MINING).

    Recebe um SimilarityScoreDataset e delega ao ComparisonMiningService,
    que aplica as regras configuradas (ComparisonMiningConfig) e devolve
    um único SimilarityScoreDataset com os pares minerados (impostores +
    genuínos concatenados). O output é tipicamente consumido por
    CALIBRATION_MODEL_TRAINING.
    """

    def __init__(self) -> None:
        self._mining_service = ComparisonMiningService()

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, PairMiningInstruction):
            return self._failure_result(
                "O executor PairMiningInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando curadoria de pares: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            scores_resource = context.get_resource(
                instruction.input_similarity_scores_resource_id
            )
            score_dataset: SimilarityScoreDataset = scores_resource.payload

            n_in = len(score_dataset.items)
            self._log(
                context,
                f"[{instruction.name}] ⛏️ Minerando a partir de {n_in:,} pares "
                "(input)…".replace(",", ".")
            )

            mined: SimilarityScoreDataset = self._mining_service.mine_pairs(
                dataset=score_dataset,
                config=instruction.mining_config,
            )

            output_name = (
                instruction.output_mined_scores_name
                or mined.name
            )
            mined.name = output_name

            output_resource_id = instruction.output_mined_scores_resource_id
            mined_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.SIMILARITY_SCORES,
                payload=mined,
                artifact_ref=None,
            )
            context.set_resource(mined_resource)

            n_gen = int(mined.metadata.get("n_genuines_mined", mined.total_genuines))
            n_imp = int(mined.metadata.get("n_impostors_mined", mined.total_impostors))
            n_total = len(mined.items)

            cfg = instruction.mining_config
            mining_summary = {
                "seed": cfg.seed,
                "exclude_self_comparisons": cfg.exclude_self_comparisons,
                "genuines_strategy": (
                    cfg.genuines_rule.strategy.value
                    if cfg.genuines_rule and hasattr(cfg.genuines_rule.strategy, "value")
                    else None
                ),
                "genuines_max_global_pairs": (
                    cfg.genuines_rule.max_global_pairs if cfg.genuines_rule else None
                ),
                "impostors_strategy": (
                    cfg.impostors_rule.strategy.value
                    if cfg.impostors_rule and hasattr(cfg.impostors_rule.strategy, "value")
                    else None
                ),
                "impostors_max_global_pairs": (
                    cfg.impostors_rule.max_global_pairs if cfg.impostors_rule else None
                ),
            }

            self._log(
                context,
                f"[{instruction.name}] ✅ Mineração concluída — "
                f"impostores={n_imp:,} | genuínos={n_gen:,} | total={n_total:,}"
                .replace(",", ".")
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "input_similarity_scores_resource_id":
                        instruction.input_similarity_scores_resource_id,
                    "output_mined_scores_resource_id": output_resource_id,
                    "n_input": int(n_in),
                    "n_total": int(n_total),
                    "n_genuine": int(n_gen),
                    "n_impostor": int(n_imp),
                    "mining_config": mining_summary,
                },
            )

        except Exception as exc:
            self._log(
                context,
                f"❌ Erro ao executar curadoria de pares "
                f"{instruction.instruction_id}: {exc}"
            )
            return self._failure_result(str(exc))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, PairMiningInstruction):
            result.add_error(
                "A instrução informada não é do tipo PairMiningInstruction."
            )
            return result

        # ─── Input: SIMILARITY_SCORES ─────────────────────────────────────
        scores_id = instruction.input_similarity_scores_resource_id
        if not scores_id:
            result.add_error(
                "A instrução de pair mining deve definir 'input_similarity_scores_resource_id'."
            )
        elif not context.has_resource(scores_id):
            result.add_error(
                f"O recurso de scores de entrada não foi encontrado no contexto: "
                f"'{scores_id}'."
            )
        else:
            scores_resource = context.get_resource(scores_id)
            if scores_resource is None:
                result.add_error(
                    f"O recurso '{scores_id}' está registrado mas é nulo no contexto."
                )
            elif scores_resource.payload is None:
                ref_path = (
                    scores_resource.artifact_ref.path
                    if scores_resource.artifact_ref is not None else None
                )
                if ref_path:
                    result.add_error(
                        f"O recurso '{scores_id}' tem artifact_ref ('{ref_path}') "
                        "mas o payload está vazio — falha ao carregar os scores "
                        "do disco. Re-execute a instrução SIMILARITY_SCORING "
                        "que produz estes scores."
                    )
                else:
                    result.add_error(
                        f"O recurso '{scores_id}' ainda não foi produzido. "
                        "Execute SIMILARITY_SCORING antes."
                    )
            elif not isinstance(scores_resource.payload, SimilarityScoreDataset):
                result.add_error(
                    f"O recurso '{scores_id}' contém um payload do tipo "
                    f"{type(scores_resource.payload).__name__}, esperado "
                    "SimilarityScoreDataset."
                )

        # ─── Mining config ─────────────────────────────────────────────────
        if instruction.mining_config is None:
            result.add_error(
                "A instrução de pair mining deve definir 'mining_config'."
            )

        # ─── Output ────────────────────────────────────────────────────────
        if not instruction.output_mined_scores_resource_id:
            result.add_error(
                "A instrução de pair mining deve definir 'output_mined_scores_resource_id'."
            )

        if (
            instruction.output_mined_scores_resource_id
            and instruction.output_mined_scores_resource_id
            == instruction.input_similarity_scores_resource_id
        ):
            result.add_warning(
                "output_mined_scores_resource_id coincide com o recurso de "
                "scores de entrada. O contexto irá sobrescrever esse recurso."
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result
