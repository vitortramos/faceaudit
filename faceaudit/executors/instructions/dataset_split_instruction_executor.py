# faceaudit/executors/instructions/dataset_split_instruction_executor.py

from __future__ import annotations

import random
from typing import List, Optional, Set, Tuple

from faceaudit.enums.plda_training_strategy import PLDATrainingStrategy
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
from faceaudit.models.split_config import GroupingStrategy, SplitConfig, SplitMode
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.plda_split_service import PLDASplitService


class DatasetSplitInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de split de dataset.

    Modos triviais (BY_INDEX_*, BY_IDENTITY_LIST, BY_IDENTITY_RATIO sem
    clustering nem pisos rígidos) usam o caminho declarativo direto sobre
    FaceDataset/EmbeddingDataset.

    Modos com semântica de treino+calibração (BY_IDENTITY_ABSOLUTE,
    BY_SUPERVISED_ADAPTATION, ou BY_IDENTITY_RATIO + PRESERVE_LOOKALIKES)
    delegam ao PLDASplitService — que implementa interseção source∩target,
    clustering de sósias e busca por pisos rígidos.
    """

    def __init__(
        self,
        plda_split_service: Optional[PLDASplitService] = None,
    ):
        self._plda_split_service = plda_split_service or PLDASplitService()

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
                context=context,
            )

            resource_a_id = instruction.output_dataset_a_resource_id
            resource_b_id = instruction.output_dataset_b_resource_id

            resource_a = StrategyResource(
                resource_id=resource_a_id,
                name=resource_a_id,
                resource_type=StrategyResourceType.DATASET,
                payload=dataset_a,
                artifact_ref=None,
            )
            resource_b = StrategyResource(
                resource_id=resource_b_id,
                name=resource_b_id,
                resource_type=StrategyResourceType.DATASET,
                payload=dataset_b,
                artifact_ref=None,
            )

            context.set_resource(resource_a)
            context.set_resource(resource_b)

            self._log(
                context,
                f"✅ Split concluído. Recursos gerados: '{resource_a_id}' e '{resource_b_id}'."
            )

            metadata = {
                "instruction_id": instruction.instruction_id,
                "instruction_name": instruction.name,
                "split_mode": instruction.split_mode.to_str(),
                "input_dataset_resource_id": instruction.dataset_resource_id,
                "output_dataset_a_resource_id": resource_a_id,
                "output_dataset_b_resource_id": resource_b_id,
                "output_dataset_a_size": len(dataset_a),
                "output_dataset_b_size": len(dataset_b),
            }
            if instruction.uses_supervised_adaptation:
                metadata["target_dataset_resource_id"] = (
                    instruction.target_dataset_resource_id
                )
            if instruction.uses_identity_grouping_capable:
                # grouping_strategy pode chegar como GroupingStrategy ou string crua
                # (depende de quem construiu o DTO). Serializa de forma defensiva.
                gs = instruction.grouping_strategy
                metadata["grouping_strategy"] = getattr(gs, "value", str(gs))
                if instruction.uses_lookalikes:
                    metadata["cluster_size"] = instruction.cluster_size

            return InstructionExecutionResult.ok(
                produced_resource_ids=[resource_a_id, resource_b_id],
                metadata=metadata,
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

        if (
            instruction.uses_identity_ratio
            or instruction.uses_index_ratio
            or instruction.uses_supervised_adaptation
        ):
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

        # Pisos rígidos exigidos em ABSOLUTE.
        if instruction.uses_identity_absolute:
            if instruction.min_calibration_pairs <= 0:
                result.add_error(
                    "BY_IDENTITY_ABSOLUTE exige 'min_calibration_pairs' > 0."
                )
            if instruction.min_calibration_ids <= 0:
                result.add_error(
                    "BY_IDENTITY_ABSOLUTE exige 'min_calibration_ids' > 0."
                )

        # cluster_size só faz sentido em modos por identidade com LOOKALIKES.
        if instruction.uses_lookalikes:
            if not instruction.uses_identity_grouping_capable:
                result.add_error(
                    "PRESERVE_LOOKALIKES só é aplicável a modos por identidade "
                    "(BY_IDENTITY_RATIO, BY_IDENTITY_ABSOLUTE, BY_IDENTITY_LIST, "
                    "BY_SUPERVISED_ADAPTATION)."
                )
            if instruction.cluster_size < 2:
                result.add_error(
                    "PRESERVE_LOOKALIKES exige 'cluster_size' >= 2."
                )

        # SDA: target obrigatório, ambos EmbeddingDataset.
        if instruction.uses_supervised_adaptation:
            if not instruction.target_dataset_resource_id:
                result.add_error(
                    "BY_SUPERVISED_ADAPTATION exige 'target_dataset_resource_id'."
                )
            elif not context.has_resource(instruction.target_dataset_resource_id):
                result.add_error(
                    "Recurso target não encontrado no contexto: "
                    f"'{instruction.target_dataset_resource_id}'."
                )
            else:
                target_payload = context.get_resource(
                    instruction.target_dataset_resource_id
                ).payload
                if not isinstance(target_payload, EmbeddingDataset):
                    result.add_error(
                        "BY_SUPERVISED_ADAPTATION exige target EmbeddingDataset; "
                        f"recebido: {type(target_payload).__name__}."
                    )

        if context.has_resource(instruction.dataset_resource_id):
            dataset = context.get_resource(instruction.dataset_resource_id).payload

            if not isinstance(dataset, (FaceDataset, EmbeddingDataset)):
                result.add_error(
                    f"Tipo de dataset não suportado para split: {type(dataset)}"
                )

            # Modos que delegam ao PLDASplitService só operam sobre EmbeddingDataset.
            if (
                self._uses_plda_split_path(instruction)
                and not isinstance(dataset, EmbeddingDataset)
            ):
                result.add_error(
                    f"split_mode={instruction.split_mode.to_str()} exige EmbeddingDataset "
                    f"(recebido: {type(dataset).__name__})."
                )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    def _split_dataset(
        self,
        dataset,
        instruction: DatasetSplitInstruction,
        context: StrategyExecutionContextProtocol,
    ):
        # Caminho PLDA: ABSOLUTE, SDA ou RATIO + LOOKALIKES — exige
        # EmbeddingDataset (PLDASplitService só opera sobre embeddings).
        if self._uses_plda_split_path(instruction):
            if not isinstance(dataset, EmbeddingDataset):
                raise TypeError(
                    "Modos BY_IDENTITY_ABSOLUTE, BY_SUPERVISED_ADAPTATION e "
                    "PRESERVE_LOOKALIKES exigem EmbeddingDataset; "
                    f"recebido: {type(dataset).__name__}."
                )
            return self._split_via_plda_service(
                source_dataset=dataset,
                instruction=instruction,
                context=context,
            )

        if isinstance(dataset, FaceDataset):
            return self._split_face_dataset(dataset, instruction)

        if isinstance(dataset, EmbeddingDataset):
            return self._split_embedding_dataset(dataset, instruction)

        raise TypeError(
            f"Tipo de dataset não suportado para split: {type(dataset)}"
        )

    @staticmethod
    def _uses_plda_split_path(instruction: DatasetSplitInstruction) -> bool:
        """
        Determina se a instrução exige o motor do PLDASplitService.

        Verdadeiro quando:
        - modo é BY_IDENTITY_ABSOLUTE,
        - modo é BY_SUPERVISED_ADAPTATION,
        - modo é BY_IDENTITY_RATIO E grouping_strategy é PRESERVE_LOOKALIKES.
        """
        if instruction.uses_identity_absolute:
            return True
        if instruction.uses_supervised_adaptation:
            return True
        if instruction.uses_identity_ratio and instruction.uses_lookalikes:
            return True
        return False

    def _split_via_plda_service(
        self,
        source_dataset: EmbeddingDataset,
        instruction: DatasetSplitInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> Tuple[EmbeddingDataset, EmbeddingDataset]:
        """
        Delega ao PLDASplitService. Mapeia (plda_set, calib_set) →
        (output_a, output_b) e aplica nomes amigáveis.
        """
        target_dataset: Optional[EmbeddingDataset] = None
        strategy_type = PLDATrainingStrategy.STANDARD

        if instruction.uses_supervised_adaptation:
            strategy_type = PLDATrainingStrategy.SUPERVISED_ADAPTATION
            target_resource = context.get_resource(instruction.target_dataset_resource_id)
            target_payload = target_resource.payload if target_resource else None
            if not isinstance(target_payload, EmbeddingDataset):
                raise TypeError(
                    "BY_SUPERVISED_ADAPTATION exige um target EmbeddingDataset; "
                    f"recebido: {type(target_payload).__name__}."
                )
            target_dataset = target_payload

        split_config = self._build_split_config(instruction)

        # Aviso explícito: quantas identidades cairão pelo filtro
        # min_samples_per_id antes de qualquer sorteio. Inclui o pool completo
        # (source ∪ target em SDA, só source caso contrário).
        self._log_pre_split_filter(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            min_samples_per_id=instruction.min_samples_per_id,
            context=context,
        )

        plda_set, calib_set = self._plda_split_service.perform_split(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            strategy_type=strategy_type,
            config=split_config,
            logger_callback=context.log,
        )

        # Convenção: A = treino (plda_set), B = calibração (calib_set).
        plda_set.name = (
            instruction.output_dataset_a_name or f"{source_dataset.name}_train"
        )
        plda_set.description = (
            f"Split {instruction.split_mode.to_str()} (treino) de {source_dataset.name}"
        )
        calib_set.name = (
            instruction.output_dataset_b_name or f"{source_dataset.name}_calib"
        )
        calib_set.description = (
            f"Split {instruction.split_mode.to_str()} (calibração) de {source_dataset.name}"
        )

        return plda_set, calib_set

    def _log_pre_split_filter(
        self,
        source_dataset: EmbeddingDataset,
        target_dataset: Optional[EmbeddingDataset],
        min_samples_per_id: int,
        context: StrategyExecutionContextProtocol,
    ) -> None:
        """
        Loga quantas identidades serão descartadas pelo filtro
        min_samples_per_id antes do sorteio. Em SDA o pool é source∪target.

        Filtro silencioso é fonte comum de surpresa para o perito — este
        log torna a perda explícita no LogProgressDialog.
        """
        from collections import Counter

        if min_samples_per_id <= 1:
            return

        counts: Counter = Counter()
        for item in source_dataset.items:
            ident = item.identity
            if ident is None or not str(ident).strip():
                continue
            counts[str(ident).strip()] += 1

        if target_dataset is not None:
            for item in target_dataset.items:
                ident = item.identity
                if ident is None or not str(ident).strip():
                    continue
                counts[str(ident).strip()] += 1

        total_ids = len(counts)
        if total_ids == 0:
            return

        dropped_ids = [idn for idn, k in counts.items() if k < min_samples_per_id]
        n_dropped = len(dropped_ids)
        n_dropped_samples = sum(counts[idn] for idn in dropped_ids)

        if n_dropped == 0:
            context.log(
                f"ℹ️ Filtro min_samples_per_id={min_samples_per_id}: "
                f"nenhuma identidade descartada ({total_ids} elegíveis)."
            )
        else:
            pct = 100.0 * n_dropped / total_ids
            context.log(
                f"⚠️ Filtro min_samples_per_id={min_samples_per_id}: "
                f"{n_dropped} de {total_ids} identidade(s) "
                f"({pct:.1f}%, {n_dropped_samples} amostra(s)) "
                f"DESCARTADAS — não vão para A nem para B."
            )

    def _build_split_config(
        self,
        instruction: DatasetSplitInstruction,
    ) -> SplitConfig:
        """
        Mapeia campos do DTO da instrução para SplitConfig (arch antiga).

        Convenção: A = treino, B = calibração.
        - calibration_ratio = 1 - split_ratio (split_ratio é a fração de A).
        - mode=ABSOLUTE em BY_IDENTITY_ABSOLUTE; senão RATIO.
        """
        if instruction.uses_identity_absolute:
            split_mode = SplitMode.ABSOLUTE
            calibration_ratio = 0.0  # ignorado em ABSOLUTE
        else:
            split_mode = SplitMode.RATIO
            ratio = instruction.split_ratio
            calibration_ratio = max(0.0, min(1.0, 1.0 - ratio))

        return SplitConfig(
            name=f"DTO_split_{instruction.instruction_id}",
            description=f"SplitConfig derivada de {instruction.name}",
            mode=split_mode,
            grouping_strategy=instruction.grouping_strategy,
            cluster_size=instruction.cluster_size,
            calibration_ratio=calibration_ratio,
            min_calibration_pairs=instruction.min_calibration_pairs,
            min_calibration_ids=instruction.min_calibration_ids,
            min_samples_per_id=instruction.min_samples_per_id,
            seed=instruction.random_seed,
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
            return self._split_embedding_dataset_by_identity_ratio(dataset, instruction)

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

    def _split_embedding_dataset_by_identity_ratio(
        self,
        dataset: EmbeddingDataset,
        instruction: DatasetSplitInstruction,
    ):
        """
        Divide EmbeddingDataset por identidades distintas na proporção informada.

        ratio×100% das identidades vão para Dataset A; o complemento vai para B.
        Evita vazamento de identidade entre os subsets — todas as amostras de
        uma mesma identidade ficam num único subset.
        """
        identities = sorted({
            item.identity for item in dataset.items if item.identity is not None
        })

        rng = random.Random(instruction.random_seed)
        rng.shuffle(identities)

        n_a = int(len(identities) * instruction.split_ratio)
        ids_a = set(identities[:n_a])

        # split_by_identities retorna (items_in, items_out) — o "in" é o subset
        # cujas identidades estão no conjunto passado. Aqui ids_a → subset A.
        ds_a, ds_b = dataset.split_by_identities(ids_a)

        ds_a.name = instruction.output_dataset_a_name or f"{dataset.name}_A"
        ds_a.description = (
            f"Split por identidade (ratio={instruction.split_ratio}) de {dataset.name}"
        )
        ds_b.name = instruction.output_dataset_b_name or f"{dataset.name}_B"
        ds_b.description = (
            f"Split por identidade (ratio={instruction.split_ratio}) de {dataset.name}"
        )
        return ds_a, ds_b

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