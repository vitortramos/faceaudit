# faceaudit/executors/instructions/dataset_merge_instruction_executor.py

from __future__ import annotations

import copy

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.instructions.dataset_merge_instruction import (
    DatasetMergeInstruction,
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


class DatasetMergeInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de combinação/atualização de datasets.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, DatasetMergeInstruction):
            return self._failure_result(
                "O executor DatasetMergeInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando execução da instrução de merge: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            base_resource = context.get_resource(instruction.base_dataset_resource_id)
            secondary_resource = context.get_resource(
                instruction.secondary_dataset_resource_id
            )

            base_dataset = base_resource.payload
            secondary_dataset = secondary_resource.payload

            merged_dataset = self._merge_datasets(
                base_dataset=base_dataset,
                secondary_dataset=secondary_dataset,
                instruction=instruction,
            )

            output_resource_id = instruction.output_dataset_resource_id
            output_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.DATASET,
                payload=merged_dataset,
                artifact_ref=None,
            )
            context.set_resource(output_resource)

            self._log(
                context,
                f"✅ Merge concluído. Recurso gerado: '{output_resource_id}'."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "merge_mode": instruction.merge_mode.to_str(),
                    "base_dataset_resource_id": instruction.base_dataset_resource_id,
                    "secondary_dataset_resource_id": instruction.secondary_dataset_resource_id,
                    "output_dataset_resource_id": output_resource_id,
                    "output_dataset_size": len(merged_dataset),
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao executar a instrução de merge "
                f"{instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, DatasetMergeInstruction):
            result.add_error(
                "A instrução informada não é do tipo DatasetMergeInstruction."
            )
            return result

        if not instruction.base_dataset_resource_id:
            result.add_error(
                "A instrução de merge deve definir 'base_dataset_resource_id'."
            )
        elif not context.has_resource(instruction.base_dataset_resource_id):
            result.add_error(
                f"O recurso base não foi encontrado no contexto: "
                f"'{instruction.base_dataset_resource_id}'."
            )

        if not instruction.secondary_dataset_resource_id:
            result.add_error(
                "A instrução de merge deve definir 'secondary_dataset_resource_id'."
            )
        elif not context.has_resource(instruction.secondary_dataset_resource_id):
            result.add_error(
                f"O recurso secundário não foi encontrado no contexto: "
                f"'{instruction.secondary_dataset_resource_id}'."
            )

        if (
            instruction.base_dataset_resource_id
            and instruction.secondary_dataset_resource_id
            and instruction.base_dataset_resource_id == instruction.secondary_dataset_resource_id
        ):
            result.add_error(
                "Os recursos base e secundário do merge devem ser diferentes."
            )

        if not instruction.output_dataset_resource_id:
            result.add_error(
                "A instrução de merge deve definir 'output_dataset_resource_id'."
            )

        if (
            instruction.output_dataset_resource_id
            and (
                instruction.output_dataset_resource_id == instruction.base_dataset_resource_id
                or instruction.output_dataset_resource_id == instruction.secondary_dataset_resource_id
            )
        ):
            result.add_warning(
                "output_dataset_resource_id coincide com um recurso de entrada. "
                "O contexto irá sobrescrever esse recurso."
            )

        if instruction.uses_concat_preserve_order:
            result.add_error(
                "CONCAT_PRESERVE_ORDER ainda não foi implementado de forma semântica."
            )

        if (
            instruction.uses_replace_by_index_list
            or instruction.uses_update_by_index_list
        ):
            if not instruction.index_list:
                result.add_error(
                    "Os modos por índice exigem 'index_list' não vazia."
                )

            if len(instruction.index_list) != len(set(instruction.index_list)):
                result.add_error(
                    "index_list não pode conter índices duplicados."
                )

        if (
            context.has_resource(instruction.base_dataset_resource_id)
            and context.has_resource(instruction.secondary_dataset_resource_id)
        ):
            base_dataset = context.get_resource(
                instruction.base_dataset_resource_id
            ).payload
            secondary_dataset = context.get_resource(
                instruction.secondary_dataset_resource_id
            ).payload

            if type(base_dataset) is not type(secondary_dataset):
                result.add_error(
                    "Os datasets base e secundário devem ter o mesmo tipo."
                )
            elif not isinstance(base_dataset, (FaceDataset, EmbeddingDataset)):
                result.add_error(
                    f"Tipo de dataset não suportado para merge: {type(base_dataset)}"
                )
            elif (
                instruction.uses_replace_by_index_list
                or instruction.uses_update_by_index_list
            ):
                secondary_size = self._dataset_size(secondary_dataset)

                if secondary_size != len(instruction.index_list):
                    result.add_error(
                        "Nos modos por índice, o número de itens do dataset secundário "
                        "deve ser igual ao número de índices informados."
                    )

                invalid = [
                    idx
                    for idx in instruction.index_list
                    if idx < 0 or idx >= self._dataset_size(base_dataset)
                ]
                if invalid:
                    result.add_error(
                        f"Foram encontrados índices inválidos para o dataset base: "
                        f"{sorted(invalid)[:10]}"
                    )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    def _merge_datasets(
        self,
        base_dataset,
        secondary_dataset,
        instruction: DatasetMergeInstruction,
    ):
        if isinstance(base_dataset, FaceDataset):
            return self._merge_face_datasets(base_dataset, secondary_dataset, instruction)

        if isinstance(base_dataset, EmbeddingDataset):
            return self._merge_embedding_datasets(
                base_dataset, secondary_dataset, instruction
            )

        raise TypeError(
            f"Tipo de dataset não suportado para merge: {type(base_dataset)}"
        )

    def _merge_face_datasets(
        self,
        base_dataset: FaceDataset,
        secondary_dataset: FaceDataset,
        instruction: DatasetMergeInstruction,
    ) -> FaceDataset:
        if instruction.uses_concat:
            merged = base_dataset + secondary_dataset
            merged.name = instruction.output_dataset_name or merged.name
            merged.description = self._resolve_description(
                base_dataset.description,
                f"Merge ({instruction.merge_mode.to_str()}) de "
                f"{base_dataset.name} + {secondary_dataset.name}",
                instruction.preserve_base_description,
            )
            return merged

        if (
            instruction.uses_replace_by_index_list
            or instruction.uses_update_by_index_list
        ):
            new_faces = list(base_dataset.faces)
            for idx, replacement_face in zip(instruction.index_list, secondary_dataset.faces):
                new_faces[idx] = copy.deepcopy(replacement_face)

            return FaceDataset(
                name=instruction.output_dataset_name or f"{base_dataset.name}_Merged",
                description=self._resolve_description(
                    base_dataset.description,
                    f"Merge ({instruction.merge_mode.to_str()}) de {base_dataset.name}",
                    instruction.preserve_base_description,
                ),
                hash=base_dataset.hash,
                faces=new_faces,
            )

        raise ValueError(f"Modo de merge não suportado: {instruction.merge_mode}")

    def _merge_embedding_datasets(
        self,
        base_dataset: EmbeddingDataset,
        secondary_dataset: EmbeddingDataset,
        instruction: DatasetMergeInstruction,
    ) -> EmbeddingDataset:
        if instruction.uses_concat:
            merged = base_dataset + secondary_dataset
            merged.name = instruction.output_dataset_name or merged.name
            merged.description = self._resolve_description(
                base_dataset.description,
                f"Merge ({instruction.merge_mode.to_str()}) de "
                f"{base_dataset.name} + {secondary_dataset.name}",
                instruction.preserve_base_description,
            )
            merged.loaded_metadata = base_dataset.loaded_metadata.copy()
            return merged

        if (
            instruction.uses_replace_by_index_list
            or instruction.uses_update_by_index_list
        ):
            new_items = list(base_dataset.items)
            for idx, replacement_item in zip(instruction.index_list, secondary_dataset.items):
                new_items[idx] = copy.deepcopy(replacement_item)

            return EmbeddingDataset(
                name=instruction.output_dataset_name or f"{base_dataset.name}_Merged",
                description=self._resolve_description(
                    base_dataset.description,
                    f"Merge ({instruction.merge_mode.to_str()}) de {base_dataset.name}",
                    instruction.preserve_base_description,
                ),
                items=new_items,
                loaded_metadata=base_dataset.loaded_metadata.copy(),
            )

        raise ValueError(f"Modo de merge não suportado: {instruction.merge_mode}")

    def _dataset_size(self, dataset) -> int:
        if isinstance(dataset, FaceDataset):
            return len(dataset.faces)
        if isinstance(dataset, EmbeddingDataset):
            return len(dataset.items)
        return len(dataset)

    def _resolve_description(
        self,
        base_description: str | None,
        fallback_description: str,
        preserve_base_description: bool,
    ) -> str:
        if preserve_base_description and base_description:
            return base_description
        return fallback_description