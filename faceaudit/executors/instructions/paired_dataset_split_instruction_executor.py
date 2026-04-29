# faceaudit/executors/instructions/paired_dataset_split_instruction_executor.py

from __future__ import annotations

from typing import Optional, Set

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.paired_dataset_split_instruction import (
    PairedDatasetSplitInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.split_config import SplitConfig, SplitMode
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.plda_split_service import PLDASplitService


class PairedDatasetSplitInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor da PAIRED_DATASET_SPLIT.

    Estratégia:
    1. Calcula a interseção de identidades entre primary e secondary.
    2. Reusa `PLDASplitService._select_ids` no pool combinado (primary +
       secondary) restrito aos IDs da interseção, com PRESERVE_LOOKALIKES
       opcional. O método retorna o conjunto de IDs alocadas ao split B
       (calibração); A é o complemento.
    3. Aplica `dataset.split_by_identities(ids_b)` em PRIMARY e SECONDARY
       separadamente — produzindo 4 datasets cujas IDs em A são idênticas
       e cujas IDs em B são idênticas (entre primary e secondary).
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

        if not isinstance(instruction, PairedDatasetSplitInstruction):
            return self._failure_result(
                "O executor PairedDatasetSplitInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando paired split: {instruction.name} "
            f"({instruction.instruction_id})"
        )

        try:
            primary: EmbeddingDataset = context.get_resource(
                instruction.primary_dataset_resource_id
            ).payload
            secondary: EmbeddingDataset = context.get_resource(
                instruction.secondary_dataset_resource_id
            ).payload

            primary_ids = self._sanitize_ids(primary.identities)
            secondary_ids = self._sanitize_ids(secondary.identities)
            intersection = primary_ids & secondary_ids

            n_int = len(intersection)
            n_p_only = len(primary_ids - secondary_ids)
            n_s_only = len(secondary_ids - primary_ids)
            self._log(
                context,
                f"🔗 Interseção de IDs: {n_int} (exclusivas em primary: "
                f"{n_p_only}, em secondary: {n_s_only})."
            )

            if n_int == 0:
                return self._failure_result(
                    "Interseção de identidades é vazia. Os datasets não "
                    "compartilham nenhuma identidade — paired split impossível."
                )

            # Pool combinado (concatena items dos dois datasets) para que o
            # _select_ids veja todas as amostras de cada ID na contagem de
            # min_samples_per_id e na matriz de confusão (LOOKALIKES).
            full_pool = primary + secondary

            # Configura SplitConfig em modo RATIO usando ratio "B" =
            # 1 - split_ratio (lembrando que _select_ids retorna o split
            # de calibração — aqui chamado de B).
            calib_ratio = max(0.0, min(1.0, 1.0 - float(instruction.split_ratio)))
            split_cfg = SplitConfig(
                name=f"PAIRED_{instruction.instruction_id}",
                description=f"SplitConfig derivada de {instruction.name}",
                mode=SplitMode.RATIO,
                grouping_strategy=instruction.grouping_strategy,
                cluster_size=int(instruction.cluster_size),
                calibration_ratio=calib_ratio,
                min_calibration_pairs=int(instruction.min_calibration_pairs),
                min_calibration_ids=int(instruction.min_calibration_ids),
                min_samples_per_id=int(instruction.min_samples_per_id),
                seed=int(instruction.random_seed),
            )

            ids_b, metrics = self._plda_split_service._select_ids(
                pool_dataset=full_pool,
                candidate_ids=intersection,
                config=split_cfg,
                logger_callback=context.log,
            )

            ids_b = set(ids_b)
            n_b = len(ids_b)
            n_a = max(0, n_int - n_b)
            self._log(
                context,
                f"✂ IDs alocadas: A (treino) = {n_a}, B (calibração) = {n_b}. "
                f"Pares genuínos no pool B: {metrics.get('total_pairs', '—')}."
            )

            # Aplica o mesmo conjunto de IDs B nos dois datasets.
            primary_b, primary_a = primary.split_by_identities(ids_b)
            secondary_b, secondary_a = secondary.split_by_identities(ids_b)

            # Renomeia (fallback) e descreve.
            primary_a.name = f"{primary.name}_A"
            primary_a.description = (
                f"Split A (treino, {n_a} IDs) de {primary.name} — paired"
            )
            primary_b.name = f"{primary.name}_B"
            primary_b.description = (
                f"Split B (calibração, {n_b} IDs) de {primary.name} — paired"
            )
            secondary_a.name = f"{secondary.name}_A"
            secondary_a.description = (
                f"Split A (treino, {n_a} IDs) de {secondary.name} — paired"
            )
            secondary_b.name = f"{secondary.name}_B"
            secondary_b.description = (
                f"Split B (calibração, {n_b} IDs) de {secondary.name} — paired"
            )

            outputs = [
                (instruction.output_primary_a_resource_id,   primary_a),
                (instruction.output_primary_b_resource_id,   primary_b),
                (instruction.output_secondary_a_resource_id, secondary_a),
                (instruction.output_secondary_b_resource_id, secondary_b),
            ]
            for oid, ds in outputs:
                context.set_resource(StrategyResource(
                    resource_id=oid,
                    name=oid,
                    resource_type=StrategyResourceType.INTERMEDIATE_DATASET,
                    payload=ds,
                    artifact_ref=None,
                ))

            self._log(
                context,
                f"✅ Paired split concluído. 4 recursos produzidos: "
                f"'{outputs[0][0]}', '{outputs[1][0]}', "
                f"'{outputs[2][0]}', '{outputs[3][0]}'."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[oid for oid, _ in outputs],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "primary_dataset_resource_id": instruction.primary_dataset_resource_id,
                    "secondary_dataset_resource_id": instruction.secondary_dataset_resource_id,
                    "n_intersection_ids": int(n_int),
                    "n_primary_only_ids": int(n_p_only),
                    "n_secondary_only_ids": int(n_s_only),
                    "n_a_ids": int(n_a),
                    "n_b_ids": int(n_b),
                    "n_b_pairs": int(metrics.get("total_pairs", 0)),
                    "grouping_strategy": instruction.grouping_strategy.value,
                    "cluster_size": int(instruction.cluster_size)
                        if instruction.uses_lookalikes else None,
                    "primary_a_size": len(primary_a),
                    "primary_b_size": len(primary_b),
                    "secondary_a_size": len(secondary_a),
                    "secondary_b_size": len(secondary_b),
                },
            )

        except Exception as exc:
            self._log(
                context,
                f"❌ Erro ao executar paired split {instruction.instruction_id}: {exc}"
            )
            return self._failure_result(str(exc))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, PairedDatasetSplitInstruction):
            result.add_error(
                "A instrução informada não é do tipo PairedDatasetSplitInstruction."
            )
            return result

        # ─── Inputs ───────────────────────────────────────────────────────
        for fname, fid in (
            ("primary_dataset_resource_id", instruction.primary_dataset_resource_id),
            ("secondary_dataset_resource_id", instruction.secondary_dataset_resource_id),
        ):
            if not fid:
                result.add_error(f"A instrução deve definir '{fname}'.")
                continue
            if not context.has_resource(fid):
                result.add_error(
                    f"O recurso '{fid}' ({fname}) não foi encontrado no contexto."
                )
                continue
            res = context.get_resource(fid)
            if res is None or res.payload is None:
                ref_path = (
                    res.artifact_ref.path
                    if (res is not None and res.artifact_ref is not None) else None
                )
                if ref_path:
                    result.add_error(
                        f"O recurso '{fid}' tem artifact_ref ('{ref_path}') mas "
                        "o payload está vazio — falha ao carregar do disco. "
                        "Re-execute a instrução que produz este recurso."
                    )
                else:
                    result.add_error(
                        f"O recurso '{fid}' ainda não foi produzido."
                    )
                continue
            if not isinstance(res.payload, EmbeddingDataset):
                result.add_error(
                    f"O recurso '{fid}' contém payload do tipo "
                    f"{type(res.payload).__name__}, esperado EmbeddingDataset."
                )

        # ─── Ratio ────────────────────────────────────────────────────────
        if not (0.0 < instruction.split_ratio < 1.0):
            result.add_error("split_ratio deve estar estritamente entre 0.0 e 1.0.")

        # ─── Outputs (4 obrigatórios e distintos) ─────────────────────────
        out_ids = [
            instruction.output_primary_a_resource_id,
            instruction.output_primary_b_resource_id,
            instruction.output_secondary_a_resource_id,
            instruction.output_secondary_b_resource_id,
        ]
        for oid, fname in zip(out_ids, [
            "output_primary_a_resource_id",
            "output_primary_b_resource_id",
            "output_secondary_a_resource_id",
            "output_secondary_b_resource_id",
        ]):
            if not oid:
                result.add_error(f"A instrução deve definir '{fname}'.")

        valid_ids = [oid for oid in out_ids if oid]
        if len(valid_ids) != len(set(valid_ids)):
            result.add_error(
                "Os 4 resource_ids de saída devem ser DISTINTOS entre si."
            )

        # ─── LOOKALIKES exige cluster_size sano ───────────────────────────
        if instruction.uses_lookalikes and instruction.cluster_size < 2:
            result.add_error("PRESERVE_LOOKALIKES exige cluster_size >= 2.")

        # ─── Warning se interseção pequena (heurístico, requer payloads) ──
        p_id = instruction.primary_dataset_resource_id
        s_id = instruction.secondary_dataset_resource_id
        if p_id and s_id and context.has_resource(p_id) and context.has_resource(s_id):
            p_res = context.get_resource(p_id)
            s_res = context.get_resource(s_id)
            if (
                p_res is not None and s_res is not None
                and isinstance(p_res.payload, EmbeddingDataset)
                and isinstance(s_res.payload, EmbeddingDataset)
            ):
                p_ids = self._sanitize_ids(p_res.payload.identities)
                s_ids = self._sanitize_ids(s_res.payload.identities)
                inter = len(p_ids & s_ids)
                if inter == 0:
                    result.add_error(
                        "Interseção de identidades entre primary e secondary é "
                        "VAZIA — paired split impossível. Verifique se os "
                        "datasets compartilham as mesmas identidades."
                    )
                elif inter < instruction.min_calibration_ids:
                    result.add_warning(
                        f"Interseção tem apenas {inter} identidades, abaixo do "
                        f"piso desejado de min_calibration_ids="
                        f"{instruction.min_calibration_ids}. O split B pode "
                        f"ficar com poucas identidades."
                    )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    @staticmethod
    def _sanitize_ids(ids_iter) -> Set[str]:
        out: Set[str] = set()
        for i in ids_iter:
            if i is None:
                continue
            s = str(i).strip()
            if s:
                out.add(s)
        return out
