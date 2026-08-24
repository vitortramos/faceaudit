# faceaudit/executors/instructions/embedding_aggregation_instruction_executor.py

from __future__ import annotations

from faceaudit.enums.aggregation_method import AggregationMethod
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.aggregation_config import AggregationConfig
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.instructions.embedding_aggregation_instruction import (
    EmbeddingAggregationInstruction,
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


# Quantas identidades ausentes listar na mensagem de erro antes de truncar.
_MAX_IDS_NA_MENSAGEM = 5


class EmbeddingAggregationInstructionExecutor(StrategyInstructionExecutorBase):
    """Executor da agregação de embeddings por identidade.

    Stateless: delega a matemática a `EmbeddingMathService.apply_smart_aggregation`,
    o mesmo caminho usado pelo step `IDENTITY_AGGREGATION` do pipeline de
    transformação — uma implementação só, dois pontos de entrada.

    Opera sobre `EmbeddingDataset`. Um `FaceDataset` é convertido antes (o
    loader de recursos tenta EmbeddingDataset primeiro, então o caso comum já
    chega no formato certo).
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, EmbeddingAggregationInstruction):
            return self._failure_result(
                "O executor EmbeddingAggregationInstructionExecutor recebeu uma "
                "instrução incompatível."
            )

        cfg = instruction.aggregation_config or AggregationConfig()
        detalhe = f"método={cfg.method.value}"
        if cfg.method == AggregationMethod.COMBINATORIAL:
            detalhe += f", k={cfg.combination_group_size}, teto/ID={cfg.max_synthetic_templates_per_id}"
        if cfg.use_quality_weighting:
            detalhe += f", ponderação={cfg.quality_key_str}"

        self._log(
            context,
            f"▶ Iniciando agregação: {instruction.name} "
            f"({instruction.instruction_id}) — {detalhe}"
        )

        try:
            input_dataset = context.get_resource(instruction.dataset_resource_id).payload

            if isinstance(input_dataset, FaceDataset):
                self._log(
                    context,
                    "ℹ FaceDataset recebido — convertendo para EmbeddingDataset "
                    "antes de agregar."
                )
                input_dataset = input_dataset.to_embedding_dataset()

            n_entrada = len(input_dataset.items)
            ids_entrada = len({str(i.identity) for i in input_dataset.items})

            from faceaudit.services.embedding_math_service import EmbeddingMathService

            resultado = EmbeddingMathService().apply_smart_aggregation(
                input_dataset,
                config=cfg,
                logger_callback=lambda msg: self._log(context, str(msg)),
            )

            if instruction.output_dataset_name:
                resultado.name = instruction.output_dataset_name

            output_resource_id = instruction.output_dataset_resource_id
            context.set_resource(StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.DATASET,
                payload=resultado,
                artifact_ref=None,
            ))

            n_saida = len(resultado.items)
            self._log(
                context,
                f"✅ Agregação concluída. Entrada: {n_entrada} amostras / "
                f"{ids_entrada} identidades → Saída: {n_saida} templates. "
                f"Metadado de foto (image_path, pose, qualidade) NÃO é "
                f"transportado — o template não corresponde a imagem nenhuma. "
                f"Recurso gerado: '{output_resource_id}'."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "aggregation_method": cfg.method.value,
                    "combination_group_size": (
                        cfg.combination_group_size
                        if cfg.method == AggregationMethod.COMBINATORIAL else None
                    ),
                    "max_synthetic_templates_per_id": (
                        cfg.max_synthetic_templates_per_id
                        if cfg.method == AggregationMethod.COMBINATORIAL else None
                    ),
                    "quality_weighting": cfg.quality_key_str,
                    "input_size": n_entrada,
                    "input_n_ids": ids_entrada,
                    "output_size": n_saida,
                    "output_dataset_name": resultado.name,
                },
            )

        except Exception as e:
            self._log(context, f"❌ Erro na agregação {instruction.instruction_id}: {e}")
            return self._failure_result(str(e))

    # ── Validação específica do executor ─────────────────────────────

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, EmbeddingAggregationInstruction):
            result.add_error(
                "A instrução informada não é do tipo EmbeddingAggregationInstruction."
            )
            return result

        if not instruction.dataset_resource_id:
            result.add_error("EMBEDDING_AGGREGATION exige 'dataset_resource_id'.")
        elif not context.has_resource(instruction.dataset_resource_id):
            result.add_error(
                f"Recurso de entrada não encontrado: "
                f"'{instruction.dataset_resource_id}'."
            )

        if not instruction.output_dataset_resource_id:
            result.add_error("EMBEDDING_AGGREGATION exige 'output_dataset_resource_id'.")

        if (
            instruction.output_dataset_resource_id
            and instruction.output_dataset_resource_id == instruction.dataset_resource_id
        ):
            result.add_warning(
                "output_dataset_resource_id coincide com a entrada — o contexto "
                "irá sobrescrever o recurso original, e o dataset de faces "
                "deixaria de existir para as instruções seguintes."
            )

        cfg = instruction.aggregation_config
        if cfg is None:
            result.add_error("Configuração de agregação ausente.")
        elif cfg.method == AggregationMethod.COMBINATORIAL:
            if cfg.combination_group_size < 2:
                result.add_error(
                    f"combination_group_size deve ser ≥ 2 em COMBINATORIAL "
                    f"(recebido {cfg.combination_group_size}) — subgrupo de uma "
                    f"face só não agrega nada."
                )
            if cfg.max_synthetic_templates_per_id < 1:
                result.add_error(
                    f"max_synthetic_templates_per_id deve ser ≥ 1 "
                    f"(recebido {cfg.max_synthetic_templates_per_id})."
                )

        if cfg is not None and cfg.use_quality_weighting and cfg.quality_weight_type is None:
            result.add_error(
                "Ponderação por qualidade ativada sem métrica escolhida."
            )

        # Payload: tipo suportado + identidade em todas as faces. A checagem de
        # identidade é feita AQUI, e não só no service, para que a falha
        # apareça na validação da UI — antes de o usuário disparar a execução.
        if (
            instruction.dataset_resource_id
            and context.has_resource(instruction.dataset_resource_id)
        ):
            payload = context.get_resource(instruction.dataset_resource_id).payload

            if payload is None:
                result.add_error(
                    "Payload do dataset de entrada não pôde ser hidratado."
                )
            elif not isinstance(payload, (FaceDataset, EmbeddingDataset)):
                result.add_error(
                    f"Tipo de dataset não suportado para agregação: "
                    f"{type(payload).__name__}. Esperado FaceDataset ou "
                    f"EmbeddingDataset."
                )
            else:
                itens = getattr(payload, "items", None) or getattr(payload, "faces", [])
                sem_id = [
                    idx for idx, it in enumerate(itens)
                    if getattr(it, "identity", None) is None
                    or not str(getattr(it, "identity", "")).strip()
                ]
                if sem_id:
                    amostra = ", ".join(str(i) for i in sem_id[:_MAX_IDS_NA_MENSAGEM])
                    reticencias = "…" if len(sem_id) > _MAX_IDS_NA_MENSAGEM else ""
                    result.add_error(
                        f"Agregação exige identidade em todas as faces: "
                        f"{len(sem_id)} de {len(itens)} estão sem identidade "
                        f"(índices {amostra}{reticencias}). Faces sem identidade "
                        f"não podem ser agregadas — corrija o dataset ou "
                        f"filtre-as antes."
                    )
                elif not itens:
                    result.add_error("Dataset de entrada está vazio.")

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result
