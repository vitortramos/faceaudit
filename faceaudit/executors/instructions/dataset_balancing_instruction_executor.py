# faceaudit/executors/instructions/dataset_balancing_instruction_executor.py

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from faceaudit.enums.dataset_balancing_mode import DatasetBalancingMode
from faceaudit.enums.dataset_balancing_selection import DatasetBalancingSelection
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.instructions.dataset_balancing_instruction import (
    DatasetBalancingInstruction,
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


class DatasetBalancingInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de balanceamento por identidade.

    Suporta tanto `FaceDataset` (lista de `FaceInfo`) quanto `EmbeddingDataset`
    (lista de `EmbeddingInfo`). A lógica de cap/floor é compartilhada — o
    que muda é a função extratora de identidade e de score de qualidade.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, DatasetBalancingInstruction):
            return self._failure_result(
                "O executor DatasetBalancingInstructionExecutor recebeu uma "
                "instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando balanceamento: {instruction.name} "
            f"({instruction.instruction_id}) — modo={instruction.balancing_mode.to_str()}, "
            f"critério={instruction.selection_criterion.to_str()}"
        )

        try:
            input_resource = context.get_resource(instruction.dataset_resource_id)
            input_dataset = input_resource.payload

            balanced, stats = self._balance(
                input_dataset=input_dataset,
                instruction=instruction,
                logger=lambda msg: self._log(context, msg),
            )

            output_resource_id = instruction.output_dataset_resource_id
            output_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.INTERMEDIATE_DATASET,
                payload=balanced,
                artifact_ref=None,
            )
            context.set_resource(output_resource)

            self._log(
                context,
                f"✅ Balanceamento concluído. "
                f"Entrada: {stats['input_size']} samples / "
                f"{stats['input_n_ids']} IDs → "
                f"Saída: {stats['output_size']} samples / "
                f"{stats['output_n_ids']} IDs (N={stats['effective_n']}). "
                f"Recurso gerado: '{output_resource_id}'."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "balancing_mode": instruction.balancing_mode.to_str(),
                    "selection_criterion": instruction.selection_criterion.to_str(),
                    "quality_metric": (
                        instruction.quality_metric.name
                        if instruction.uses_quality_selection else None
                    ),
                    "effective_n": stats["effective_n"],
                    "input_size": stats["input_size"],
                    "input_n_ids": stats["input_n_ids"],
                    "output_size": stats["output_size"],
                    "output_n_ids": stats["output_n_ids"],
                    "discarded_identities": stats["discarded_identities"],
                    "quality_fallback_count": stats["quality_fallback_count"],
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro no balanceamento {instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    # ── Validação específica do executor ─────────────────────────────

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, DatasetBalancingInstruction):
            result.add_error(
                "A instrução informada não é do tipo DatasetBalancingInstruction."
            )
            return result

        if not instruction.dataset_resource_id:
            result.add_error(
                "DATASET_BALANCING exige 'dataset_resource_id'."
            )
        elif not context.has_resource(instruction.dataset_resource_id):
            result.add_error(
                f"Recurso de entrada não encontrado: "
                f"'{instruction.dataset_resource_id}'."
            )

        if not instruction.output_dataset_resource_id:
            result.add_error(
                "DATASET_BALANCING exige 'output_dataset_resource_id'."
            )

        if (
            instruction.output_dataset_resource_id
            and instruction.output_dataset_resource_id == instruction.dataset_resource_id
        ):
            result.add_warning(
                "output_dataset_resource_id coincide com a entrada — "
                "o contexto irá sobrescrever o recurso original."
            )

        if not instruction.uses_auto_min and instruction.target_n < 1:
            result.add_error(
                "target_n deve ser ≥ 1 nos modos CAP_ONLY e FLOOR_AND_CAP."
            )

        # Valida tipo do payload (apenas se o recurso estiver disponível).
        if (
            instruction.dataset_resource_id
            and context.has_resource(instruction.dataset_resource_id)
        ):
            payload = context.get_resource(
                instruction.dataset_resource_id
            ).payload

            if payload is None:
                result.add_error(
                    "Payload do dataset de entrada não pôde ser hidratado."
                )
            elif not isinstance(payload, (FaceDataset, EmbeddingDataset)):
                result.add_error(
                    f"Tipo de dataset não suportado para balanceamento: "
                    f"{type(payload).__name__}. Esperado FaceDataset ou "
                    f"EmbeddingDataset."
                )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    # ── Lógica de balanceamento — agnóstica ao tipo de dataset ──────

    def _balance(
        self,
        input_dataset: Any,
        instruction: DatasetBalancingInstruction,
        logger: Callable[[str], None],
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Aplica o balanceamento. Retorna (dataset_balanceado, stats_dict).

        Estratégia agnóstica ao tipo:
        1. Itera nos itens, agrupa por identidade
        2. Determina N efetivo segundo o modo
        3. Para cada ID, aplica o cap usando o critério de seleção
        4. Para FLOOR_AND_CAP, descarta IDs com < N
        5. Reconstrói o dataset com os itens selecionados
        """
        items = self._get_items(input_dataset)
        items_by_id = self._group_by_identity(items, input_dataset)

        input_size = len(items)
        input_n_ids = len(items_by_id)

        # Determina N efetivo conforme o modo.
        if instruction.uses_auto_min:
            counts = [len(v) for v in items_by_id.values()]
            effective_n = min(counts) if counts else 0
            logger(
                f"  ↳ AUTO_MIN_BALANCE: N computado = {effective_n} "
                f"(menor entre {input_n_ids} identidades)."
            )
        else:
            effective_n = instruction.target_n

        # Aplica floor (descarta IDs inteiros) apenas em FLOOR_AND_CAP.
        if instruction.uses_floor_and_cap:
            before = len(items_by_id)
            items_by_id = {
                k: v for k, v in items_by_id.items() if len(v) >= effective_n
            }
            discarded_identities = before - len(items_by_id)
            if discarded_identities > 0:
                logger(
                    f"  ↳ FLOOR_AND_CAP: {discarded_identities} identidade(s) "
                    f"descartada(s) por terem < {effective_n} amostras."
                )
        else:
            discarded_identities = 0

        # Aplica cap por identidade.
        rng = random.Random(instruction.random_seed)
        quality_fallback_count = 0
        is_face_dataset = isinstance(input_dataset, FaceDataset)
        kept_items: List[Any] = []

        for identity, id_items in items_by_id.items():
            if len(id_items) <= effective_n:
                kept_items.extend(id_items)
                continue

            if instruction.uses_random_selection:
                # Cópia local para não mutar o input.
                shuffled = list(id_items)
                rng.shuffle(shuffled)
                kept_items.extend(shuffled[:effective_n])
            else:
                # BY_QUALITY: ordena por score (desc), tie-break determinístico.
                # Itens sem métrica disponível caem para o final, com warning.
                sorted_items, fallbacks = self._sort_by_quality(
                    id_items=id_items,
                    quality_metric=instruction.quality_metric,
                    is_face_dataset=is_face_dataset,
                    rng_for_ties=rng,
                )
                quality_fallback_count += fallbacks
                kept_items.extend(sorted_items[:effective_n])

        if quality_fallback_count > 0:
            logger(
                f"  ↳ ⚠ {quality_fallback_count} sample(s) sem "
                f"`{instruction.quality_metric.name}` — usados na ordem "
                f"original como fallback."
            )

        # Reconstrói o dataset preservando o tipo.
        balanced = self._build_output_dataset(
            input_dataset=input_dataset,
            kept_items=kept_items,
            instruction=instruction,
        )

        output_size = len(self._get_items(balanced))
        output_n_ids = len({
            self._extract_identity(it, is_face_dataset)
            for it in self._get_items(balanced)
            if self._extract_identity(it, is_face_dataset) is not None
        })

        return balanced, {
            "input_size": input_size,
            "input_n_ids": input_n_ids,
            "output_size": output_size,
            "output_n_ids": output_n_ids,
            "effective_n": effective_n,
            "discarded_identities": discarded_identities,
            "quality_fallback_count": quality_fallback_count,
        }

    # ── Helpers de tipo (FaceDataset vs EmbeddingDataset) ───────────

    @staticmethod
    def _get_items(dataset: Any) -> List[Any]:
        if isinstance(dataset, FaceDataset):
            return list(dataset.faces)
        if isinstance(dataset, EmbeddingDataset):
            return list(dataset.items)
        return list(dataset)

    @staticmethod
    def _extract_identity(item: Any, is_face_dataset: bool) -> Optional[str]:
        # FaceDataset → FaceInfo.identity ; EmbeddingDataset → EmbeddingInfo.identity
        identity = getattr(item, "identity", None)
        if identity is None:
            return None
        return str(identity)

    def _group_by_identity(
        self, items: List[Any], input_dataset: Any
    ) -> Dict[str, List[Any]]:
        """Agrupa itens por identidade. Itens sem identity são ignorados."""
        is_face_dataset = isinstance(input_dataset, FaceDataset)
        groups: Dict[str, List[Any]] = defaultdict(list)
        for item in items:
            identity = self._extract_identity(item, is_face_dataset)
            if identity is None:
                continue
            groups[identity].append(item)
        return dict(groups)

    def _sort_by_quality(
        self,
        id_items: List[Any],
        quality_metric: QualityScoreType,
        is_face_dataset: bool,
        rng_for_ties: random.Random,
    ) -> Tuple[List[Any], int]:
        """
        Ordena `id_items` decrescentemente pela métrica (maior = melhor).

        Itens sem score (None) são empurrados para o final mas mantém ordem
        relativa estável (tie-break por hash do identificador). Retorna a
        lista ordenada + contagem de fallbacks (samples sem métrica).
        """
        scored: List[Tuple[float, str, Any]] = []
        fallback_count = 0

        for item in id_items:
            score = self._extract_quality_score(
                item, quality_metric, is_face_dataset
            )
            if score is None:
                fallback_count += 1
                # Score = -inf garante que vai para o final.
                score_for_sort = float("-inf")
            else:
                score_for_sort = float(score)

            # Tie-breaker determinístico por hash do file_name/face_hash.
            tie_key = self._stable_tie_key(item, is_face_dataset)
            scored.append((score_for_sort, tie_key, item))

        # Sort: maior score primeiro; em empate, hash menor primeiro.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [t[2] for t in scored], fallback_count

    @staticmethod
    def _extract_quality_score(
        item: Any,
        quality_metric: QualityScoreType,
        is_face_dataset: bool,
    ) -> Optional[float]:
        """
        Extrai score de qualidade de um item, agnóstico ao tipo de dataset.

        - FaceDataset (FaceInfo): usa `quality_metric.extract_from(face)`.
        - EmbeddingDataset (EmbeddingInfo):
            * EMBEDDING_NORM: calcula `np.linalg.norm(item.embedding)` on-the-fly
            * Outras: lê de `item.quality_scores[metric.name.lower()]`
        """
        if is_face_dataset:
            try:
                return quality_metric.extract_from(item)
            except Exception:
                return None

        # EmbeddingDataset path
        if quality_metric == QualityScoreType.EMBEDDING_NORM:
            emb = getattr(item, "embedding", None)
            if emb is None:
                return None
            try:
                return float(np.linalg.norm(emb))
            except Exception:
                return None

        scores = getattr(item, "quality_scores", None)
        if not isinstance(scores, dict):
            return None
        val = scores.get(quality_metric.name.lower())
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _stable_tie_key(item: Any, is_face_dataset: bool) -> str:
        """
        Chave estável para desempate em ordenação por qualidade.

        Prioridade: face_hash → file_name → image_path → repr.
        """
        for attr in ("face_hash", "hash", "file_name", "image_path"):
            v = getattr(item, attr, None)
            if v:
                return hashlib.sha1(str(v).encode("utf-8")).hexdigest()[:16]
        return hashlib.sha1(repr(item).encode("utf-8")).hexdigest()[:16]

    def _build_output_dataset(
        self,
        input_dataset: Any,
        kept_items: List[Any],
        instruction: DatasetBalancingInstruction,
    ) -> Any:
        """Reconstrói um dataset do mesmo tipo do input contendo apenas kept_items."""
        new_name = (
            instruction.output_dataset_name
            or f"{getattr(input_dataset, 'name', 'dataset')}_Balanced"
        )
        new_desc = (
            f"Balanceado ({instruction.balancing_mode.to_str()} / "
            f"{instruction.selection_criterion.to_str()})"
        )

        if isinstance(input_dataset, FaceDataset):
            return FaceDataset(
                name=new_name,
                description=new_desc,
                hash=getattr(input_dataset, "hash", None),
                faces=kept_items,
            )

        if isinstance(input_dataset, EmbeddingDataset):
            ds = EmbeddingDataset(
                name=new_name,
                description=new_desc,
                items=kept_items,
            )
            # Preserva metadados se a classe expuser.
            loaded_md = getattr(input_dataset, "loaded_metadata", None)
            if isinstance(loaded_md, dict):
                ds.loaded_metadata = loaded_md.copy()
            return ds

        raise TypeError(
            f"Tipo de dataset não suportado: {type(input_dataset).__name__}"
        )
