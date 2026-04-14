# faceaudit/executors/instructions/dataset_split_instruction_executor.py

from __future__ import annotations

import random
from typing import List, Set

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.instructions.dataset_split_instruction import (
    DatasetSplitInstruction,
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


class DatasetSplitInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de split de dataset.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, DatasetSplitInstruction):
            return self._failure_result(
                "O executor DatasetSplitInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando execução da instrução de split: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            dataset_resource = context.get_resource(instruction.dataset_resource_id)
            dataset = dataset_resource.payload

            dataset_a, dataset_b = self._split_dataset(
                dataset=dataset,
                instruction=instruction,
            )

            resource_a_id = instruction.output_dataset_a_resource_id
            resource_b_id = instruction.output_dataset_b_resource_id

            resource_a = StrategyResource(
                resource_id=resource_a_id,
                name=resource_a_id,
                resource_type=StrategyResourceType.INTERMEDIATE_DATASET,
                payload=dataset_a,
                artifact_ref=None,
            )
            resource_b = StrategyResource(
                resource_id=resource_b_id,
                name=resource_b_id,
                resource_type=StrategyResourceType.INTERMEDIATE_DATASET,
                payload=dataset_b,
                artifact_ref=None,
            )

            context.set_resource(resource_a)
            context.set_resource(resource_b)

            self._log(
                context,
                f"✅ Split concluído. Recursos gerados: '{resource_a_id}' e '{resource_b_id}'."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[resource_a_id, resource_b_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "split_mode": instruction.split_mode.to_str(),
                    "input_dataset_resource_id": instruction.dataset_resource_id,
                    "output_dataset_a_resource_id": resource_a_id,
                    "output_dataset_b_resource_id": resource_b_id,
                    "output_dataset_a_size": len(dataset_a),
                    "output_dataset_b_size": len(dataset_b),
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao executar a instrução de split "
                f"{instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, DatasetSplitInstruction):
            result.add_error(
                "A instrução informada não é do tipo DatasetSplitInstruction."
            )
            return result

        if not instruction.dataset_resource_id:
            result.add_error("A instrução de split deve definir 'dataset_resource_id'.")
        elif not context.has_resource(instruction.dataset_resource_id):
            result.add_error(
                f"O recurso de dataset não foi encontrado no contexto: "
                f"'{instruction.dataset_resource_id}'."
            )

        if not instruction.output_dataset_a_resource_id:
            result.add_error(
                "A instrução de split deve definir 'output_dataset_a_resource_id'."
            )

        if not instruction.output_dataset_b_resource_id:
            result.add_error(
                "A instrução de split deve definir 'output_dataset_b_resource_id'."
            )

        if (
            instruction.output_dataset_a_resource_id
            and instruction.output_dataset_b_resource_id
            and instruction.output_dataset_a_resource_id == instruction.output_dataset_b_resource_id
        ):
            result.add_error(
                "Os recursos de saída A e B do split devem ser diferentes."
            )

        if instruction.uses_identity_ratio or instruction.uses_index_ratio:
            if not (0.0 < instruction.split_ratio < 1.0):
                result.add_error(
                    "split_ratio deve estar estritamente entre 0.0 e 1.0."
                )

        if instruction.uses_identity_list and not instruction.identity_list:
            result.add_error(
                "split_mode=BY_IDENTITY_LIST exige 'identity_list' não vazia."
            )

        if instruction.uses_index_list and not instruction.index_list:
            result.add_error(
                "split_mode=BY_INDEX_LIST exige 'index_list' não vazia."
            )

        if context.has_resource(instruction.dataset_resource_id):
            dataset = context.get_resource(instruction.dataset_resource_id).payload

            if not isinstance(dataset, (FaceDataset, EmbeddingDataset)):
                result.add_error(
                    f"Tipo de dataset não suportado para split: {type(dataset)}"
                )

            if isinstance(dataset, EmbeddingDataset) and instruction.uses_identity_ratio:
                result.add_error(
                    "BY_IDENTITY_RATIO ainda não foi implementado para EmbeddingDataset."
                )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    def _split_dataset(
        self,
        dataset,
        instruction: DatasetSplitInstruction,
    ):
        if isinstance(dataset, FaceDataset):
            return self._split_face_dataset(dataset, instruction)

        if isinstance(dataset, EmbeddingDataset):
            return self._split_embedding_dataset(dataset, instruction)

        raise TypeError(
            f"Tipo de dataset não suportado para split: {type(dataset)}"
        )

    def _split_face_dataset(
        self,
        dataset: FaceDataset,
        instruction: DatasetSplitInstruction,
    ):
        if instruction.uses_identity_ratio:
            ds_a, ds_b = dataset.split_by_identity(
                split_ratio=instruction.split_ratio,
                random_seed=instruction.random_seed,
            )
            return self._rename_face_outputs(ds_a, ds_b, instruction)

        if instruction.uses_identity_list:
            ds_a, ds_b = dataset.split_by_identity_list(
                ids_for_dataset_b=set(instruction.identity_list),
                name_a=instruction.output_dataset_a_name or "",
                name_b=instruction.output_dataset_b_name or "",
            )
            return ds_a, ds_b

        if instruction.uses_index_ratio:
            return self._split_face_dataset_by_index_ratio(dataset, instruction)

        if instruction.uses_index_list:
            return self._split_face_dataset_by_index_list(dataset, instruction)

        raise ValueError(f"Modo de split não suportado: {instruction.split_mode}")

    def _split_face_dataset_by_index_ratio(
        self,
        dataset: FaceDataset,
        instruction: DatasetSplitInstruction,
    ):
        total = len(dataset.faces)
        indices = list(range(total))

        rng = random.Random(instruction.random_seed)
        rng.shuffle(indices)

        split_idx = int(total * instruction.split_ratio)

        indices_a = set(indices[:split_idx])
        faces_a = [face for idx, face in enumerate(dataset.faces) if idx in indices_a]
        faces_b = [face for idx, face in enumerate(dataset.faces) if idx not in indices_a]

        ds_a = FaceDataset(
            name=instruction.output_dataset_a_name or f"{dataset.name}_A",
            description=f"Split por índice (ratio) de {dataset.name}",
            faces=faces_a,
        )
        ds_b = FaceDataset(
            name=instruction.output_dataset_b_name or f"{dataset.name}_B",
            description=f"Split por índice (ratio) de {dataset.name}",
            faces=faces_b,
        )
        return ds_a, ds_b

    def _split_face_dataset_by_index_list(
        self,
        dataset: FaceDataset,
        instruction: DatasetSplitInstruction,
    ):
        index_set = self._sanitize_index_list(
            instruction.index_list,
            dataset_size=len(dataset.faces),
        )

        faces_b = [face for idx, face in enumerate(dataset.faces) if idx in index_set]
        faces_a = [face for idx, face in enumerate(dataset.faces) if idx not in index_set]

        ds_a = FaceDataset(
            name=instruction.output_dataset_a_name or f"{dataset.name}_A",
            description=f"Split por lista de índices de {dataset.name}",
            faces=faces_a,
        )
        ds_b = FaceDataset(
            name=instruction.output_dataset_b_name or f"{dataset.name}_B",
            description=f"Split por lista de índices de {dataset.name}",
            faces=faces_b,
        )
        return ds_a, ds_b

    def _rename_face_outputs(
        self,
        ds_a: FaceDataset,
        ds_b: FaceDataset,
        instruction: DatasetSplitInstruction,
    ):
        if instruction.output_dataset_a_name:
            ds_a.name = instruction.output_dataset_a_name
        if instruction.output_dataset_b_name:
            ds_b.name = instruction.output_dataset_b_name
        return ds_a, ds_b

    def _split_embedding_dataset(
        self,
        dataset: EmbeddingDataset,
        instruction: DatasetSplitInstruction,
    ):
        if instruction.uses_identity_ratio:
            raise NotImplementedError(
                "BY_IDENTITY_RATIO ainda não foi implementado para EmbeddingDataset."
            )

        if instruction.uses_identity_list:
            ds_b, ds_a = dataset.split_by_identities(set(instruction.identity_list))
            ds_a.name = instruction.output_dataset_a_name or f"{dataset.name}_A"
            ds_b.name = instruction.output_dataset_b_name or f"{dataset.name}_B"
            return ds_a, ds_b

        if instruction.uses_index_ratio:
            return self._split_embedding_dataset_by_index_ratio(dataset, instruction)

        if instruction.uses_index_list:
            return self._split_embedding_dataset_by_index_list(dataset, instruction)

        raise ValueError(f"Modo de split não suportado: {instruction.split_mode}")

    def _split_embedding_dataset_by_index_ratio(
        self,
        dataset: EmbeddingDataset,
        instruction: DatasetSplitInstruction,
    ):
        total = len(dataset.items)
        indices = list(range(total))

        rng = random.Random(instruction.random_seed)
        rng.shuffle(indices)

        split_idx = int(total * instruction.split_ratio)

        indices_a = set(indices[:split_idx])
        items_a = [item for idx, item in enumerate(dataset.items) if idx in indices_a]
        items_b = [item for idx, item in enumerate(dataset.items) if idx not in indices_a]

        ds_a = EmbeddingDataset(
            name=instruction.output_dataset_a_name or f"{dataset.name}_A",
            description=f"Split por índice (ratio) de {dataset.name}",
            items=items_a,
            loaded_metadata=dataset.loaded_metadata.copy(),
        )
        ds_b = EmbeddingDataset(
            name=instruction.output_dataset_b_name or f"{dataset.name}_B",
            description=f"Split por índice (ratio) de {dataset.name}",
            items=items_b,
            loaded_metadata=dataset.loaded_metadata.copy(),
        )
        return ds_a, ds_b

    def _split_embedding_dataset_by_index_list(
        self,
        dataset: EmbeddingDataset,
        instruction: DatasetSplitInstruction,
    ):
        index_set = self._sanitize_index_list(
            instruction.index_list,
            dataset_size=len(dataset.items),
        )

        items_b = [item for idx, item in enumerate(dataset.items) if idx in index_set]
        items_a = [item for idx, item in enumerate(dataset.items) if idx not in index_set]

        ds_a = EmbeddingDataset(
            name=instruction.output_dataset_a_name or f"{dataset.name}_A",
            description=f"Split por lista de índices de {dataset.name}",
            items=items_a,
            loaded_metadata=dataset.loaded_metadata.copy(),
        )
        ds_b = EmbeddingDataset(
            name=instruction.output_dataset_b_name or f"{dataset.name}_B",
            description=f"Split por lista de índices de {dataset.name}",
            items=items_b,
            loaded_metadata=dataset.loaded_metadata.copy(),
        )
        return ds_a, ds_b

    def _sanitize_index_list(
        self,
        indices: List[int],
        dataset_size: int,
    ) -> Set[int]:
        sanitized = set(indices)

        invalid = [idx for idx in sanitized if idx < 0 or idx >= dataset_size]
        if invalid:
            raise ValueError(
                f"Foram encontrados índices inválidos para o tamanho do dataset "
                f"({dataset_size}): {sorted(invalid)[:10]}"
            )

        return sanitized