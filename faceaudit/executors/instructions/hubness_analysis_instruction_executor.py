# faceaudit/executors/instructions/hubness_analysis_instruction_executor.py

from __future__ import annotations

from typing import Optional

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.hubness_report import HubnessReport
from faceaudit.models.instructions.hubness_analysis_instruction import (
    HubnessAnalysisInstruction,
)
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


class HubnessAnalysisInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor da instrução HUBNESS_ANALYSIS.

    Recebe um EmbeddingDataset, instancia o HubnessAnalysisService (com o
    ComparisonService injetado), executa compute_analysis + os 4 estudos
    derivados (stratification, paired_study, quality_diagnostics,
    optimization_study) e empacota tudo num HubnessReport compósito
    persistido como pickle.

    O ComparisonService é injetado via construtor — quem monta o executor
    (StrategyExecutorFactory) é responsável por fornecê-lo.
    """

    def __init__(self, comparison_service=None):
        """
        Args:
            comparison_service: instância de ComparisonService injetada pela
                factory. Se None, é instanciado on-demand com defaults.
        """
        self._comparison_service = comparison_service

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, HubnessAnalysisInstruction):
            return self._failure_result(
                "Executor HubnessAnalysisInstructionExecutor recebeu uma "
                "instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando análise de hubness: {instruction.name} "
            f"({instruction.instruction_id}) | K={instruction.k_neighbors}"
        )

        try:
            ds_resource = context.get_resource(instruction.dataset_resource_id)
            dataset: EmbeddingDataset = ds_resource.payload

            service = self._build_service(instruction.k_neighbors)
            log_cb = lambda msg: self._log(context, msg)

            # Análise base — caro O(N²).
            analysis = service.compute_analysis(
                dataset=dataset,
                batch_size=instruction.batch_size,
                chunk_size=instruction.chunk_size,
                logger_callback=log_cb,
            )

            # Estudos derivados — todos baratos em cima do resultado base.
            # `stratification` e `paired_study` precisam ser injetados de volta
            # no `analysis` porque os métodos plot_stratified_hubness e
            # plot_paired_* do HubnessPlotService leem desses atributos.
            # `quality_diagnostics` e `optimization_study` são passados como
            # arg direto para seus respectivos plots — não precisam de atribuição.
            self._log(context, "  ↳ Computando estratificação por pose...")
            stratification = service.compute_stratification(analysis)
            analysis.stratification = stratification

            self._log(context, "  ↳ Computando estudo pareado...")
            paired = service.compute_paired_study(
                analysis,
                threshold_nf=instruction.paired_study_threshold_nf,
            )
            analysis.paired_study = paired

            self._log(context, "  ↳ Computando diagnóstico de qualidade...")
            quality_diag = service.compute_quality_diagnostics(analysis)

            self._log(context, "  ↳ Computando estudo de otimização (sweet spot)...")
            optimization = service.compute_optimization_study(analysis)

            report = HubnessReport(
                analysis=analysis,
                stratification=stratification,
                paired_study=paired,
                quality_diagnostics=quality_diag,
                optimization_study=optimization,
                dataset_name=getattr(dataset, "name", "") or "",
                k_neighbors=int(instruction.k_neighbors),
                n_samples=int(analysis.sample_count),
            )

            output_id = instruction.output_report_resource_id
            output = StrategyResource(
                resource_id=output_id,
                name=output_id,
                resource_type=StrategyResourceType.HUBNESS_REPORT,
                payload=report,
                artifact_ref=None,
            )
            context.set_resource(output)

            self._log(
                context,
                f"✅ HubnessReport gerado: {report.n_samples} samples, "
                f"K={report.k_neighbors}, max Nk={analysis.max_nk}, "
                f"mean Nk={analysis.mean_nk:.2f} (esperado ≈ K)"
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "k_neighbors": int(report.k_neighbors),
                    "n_samples": int(report.n_samples),
                    "max_nk": int(analysis.max_nk),
                    "mean_nk": float(analysis.mean_nk),
                    "output_resource_id": output_id,
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro na análise de hubness {instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    def _build_service(self, k_neighbors: int):
        """Instancia HubnessAnalysisService com o comparison_service injetado."""
        from faceaudit.services.hubness_analysis_service import HubnessAnalysisService
        comp = self._comparison_service or self._build_default_comparison_service()
        return HubnessAnalysisService(comparison_service=comp, k_neighbors=k_neighbors)

    @staticmethod
    def _build_default_comparison_service():
        """Fallback se a factory não injetou — instancia ComparisonService default."""
        from faceaudit.services.comparison_service import ComparisonService
        return ComparisonService()

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, HubnessAnalysisInstruction):
            result.add_error("Instrução não é HubnessAnalysisInstruction.")
            return result

        if not instruction.dataset_resource_id:
            result.add_error("HUBNESS_ANALYSIS exige 'dataset_resource_id'.")
        elif not context.has_resource(instruction.dataset_resource_id):
            result.add_error(
                f"Recurso de entrada não encontrado: '{instruction.dataset_resource_id}'."
            )

        if not instruction.output_report_resource_id:
            result.add_error("HUBNESS_ANALYSIS exige 'output_report_resource_id'.")

        if instruction.k_neighbors < 1:
            result.add_error("k_neighbors deve ser ≥ 1.")

        if instruction.batch_size < 1 or instruction.chunk_size < 1:
            result.add_error("batch_size e chunk_size devem ser ≥ 1.")

        # Tipo do payload — precisa ser EmbeddingDataset.
        if (
            instruction.dataset_resource_id
            and context.has_resource(instruction.dataset_resource_id)
        ):
            payload = context.get_resource(instruction.dataset_resource_id).payload
            if payload is None:
                result.add_error("Payload do dataset não pôde ser hidratado.")
            elif not isinstance(payload, EmbeddingDataset):
                result.add_error(
                    f"HUBNESS_ANALYSIS exige EmbeddingDataset (tem identidades, "
                    f"poses e quality_scores). Recebido: {type(payload).__name__}."
                )
            else:
                # Aviso: identidades insuficientes degradam todos os estudos.
                n_ids = len(set(
                    item.identity for item in payload.items
                    if item.identity is not None
                ))
                if n_ids < 5:
                    result.add_warning(
                        f"Apenas {n_ids} identidades distintas — estudos pareado e "
                        "de otimização provavelmente não terão amostras suficientes."
                    )
                if instruction.k_neighbors >= len(payload.items):
                    result.add_error(
                        f"k_neighbors ({instruction.k_neighbors}) deve ser menor que "
                        f"o número de samples ({len(payload.items)})."
                    )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()
        return result
