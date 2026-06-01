# faceaudit/executors/instructions/score_interaction_feature_instruction_executor.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from faceaudit.enums.s0_mode import S0Mode
from faceaudit.enums.score_interaction_mode import ScoreInteractionMode
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_combiner import TypicalityCombiner
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.calibration.logreg_s0_config import LogRegS0Config
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.score_interaction_feature_instruction import (
    ScoreInteractionFeatureInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)

OrderedPairKey = Tuple[str, str]
UnorderedPairKey = frozenset
PairKey = Union[OrderedPairKey, UnorderedPairKey]

CV_LEAKAGE_WARN = "n_pairs<1000"
AUTO_LOGREG_MIN_PAIRS_FOR_NO_WARNING = 1000
COEF_ZERO_EPSILON = 1e-6


class ScoreInteractionFeatureInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor da instrução SCORE_INTERACTION_FEATURE.

    Constrói a feature derivada `(base − s0)·rarity` (modos
    LINEAR/ONE_SIDED_POSITIVE/HINGE_BASE) a partir de dois
    `SimilarityScoreDataset` alinháveis por par. Resolve `s0` via
    AUTO_LOGREG (calibrador auxiliar univariado), MANUAL ou INHERIT.

    A instrução é agnóstica à origem da raridade. O alinhamento usa
    chave orientada `(probe_hash, ref_hash)` quando o modifier vem
    de combiner unilateral (REFERENCE_ONLY/PROBE_ONLY); senão usa
    `frozenset({probe, ref})` (compat com simétricos).

    A saída é um `SimilarityScoreDataset` com `raw_score` 1D (feature
    derivada), consumível por `SCORE_VECTORIZATION` para empilhar com
    o cosseno e treinar calibrador LogReg multivariado.
    """

    # ─────────────────────────── execute ────────────────────────────

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, ScoreInteractionFeatureInstruction):
            return self._failure_result(
                "Executor recebeu instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando SCORE_INTERACTION_FEATURE: "
            f"{instruction.name} ({instruction.instruction_id})",
        )

        try:
            base_ds = self._fetch_dataset(
                instruction.base_score_resource_id, context, "base",
            )
            self._reject_recursive_base(base_ds, instruction.base_score_resource_id)

            modifier_ds: Optional[SimilarityScoreDataset] = None
            if instruction.interaction_mode is not ScoreInteractionMode.HINGE_BASE:
                modifier_ds = self._fetch_dataset(
                    instruction.modifier_score_resource_id, context, "modifier",
                )
                self._validate_modifier_family(
                    modifier_ds, instruction.modifier_score_resource_id,
                )

            # Alinhamento e validações entre base e modifier (orientado vs
            # frozenset; checa duplicatas; checa same_person quando não
            # casework).
            alignment_orientation = "unordered"
            aligned_modifier_items: Optional[List[SimilarityScoreSample]] = None
            if modifier_ds is not None:
                combiner_str = (modifier_ds.metadata or {}).get("combiner")
                combiner = self._safe_parse_combiner(combiner_str)
                orientation = (
                    "ordered"
                    if combiner is not None and not combiner.is_symmetric
                    else "unordered"
                )
                alignment_orientation = orientation
                aligned_modifier_items = self._align_pairs(
                    base_ds=base_ds,
                    modifier_ds=modifier_ds,
                    orientation=orientation,
                    base_id=instruction.base_score_resource_id,
                    modifier_id=instruction.modifier_score_resource_id,
                    context=context,
                )
                if not instruction.is_casework:
                    self._validate_same_person_consistency(
                        base_ds.items, aligned_modifier_items,
                        instruction.base_score_resource_id,
                        instruction.modifier_score_resource_id,
                    )

            # Vetores base e modifier alinhados.
            base_scores = np.fromiter(
                (float(s.raw_score) for s in base_ds.items),
                dtype=np.float64,
                count=len(base_ds.items),
            )
            modifier_raw: Optional[np.ndarray] = None
            modifier_score_type: Optional[TypicalityScoreType] = None
            modifier_combiner: Optional[TypicalityCombiner] = None
            rarity_transform_label: Optional[str] = None
            rarity_values: Optional[np.ndarray] = None
            if modifier_ds is not None and aligned_modifier_items is not None:
                modifier_raw = np.fromiter(
                    (float(s.raw_score) for s in aligned_modifier_items),
                    dtype=np.float64,
                    count=len(aligned_modifier_items),
                )
                modifier_score_type = self._require_score_type(
                    modifier_ds, instruction.modifier_score_resource_id,
                )
                combiner_str = (modifier_ds.metadata or {}).get("combiner")
                modifier_combiner = self._safe_parse_combiner(combiner_str)
                rarity_values, rarity_transform_label = self._to_rarity(
                    typicality=modifier_raw,
                    score_type=modifier_score_type,
                    invert=instruction.invert_rarity,
                )

            # Resolução de s0 + diagnósticos.
            s0_meta = self._resolve_s0(
                instruction=instruction,
                base_scores=base_scores,
                base_ds=base_ds,
                context=context,
            )
            s0 = s0_meta["interaction_s0"]

            # Centralização opcional da raridade por r0 (princípio
            # hierárquico de Tukey). Quando ativada, transforma `rarity`
            # em `rarity − r0` antes de aplicar a fórmula da interação.
            # Sem efeito em HINGE_BASE (que ignora rarity). r0 é
            # calculado/herdado em paralelo a s0 — mesma source no INHERIT.
            r0_meta = self._resolve_r0(
                instruction=instruction,
                rarity_values=rarity_values,
                context=context,
            )
            if (
                instruction.center_rarity
                and rarity_values is not None
                and r0_meta["interaction_r0"] is not None
            ):
                rarity_values = rarity_values - float(r0_meta["interaction_r0"])

            # Construção da feature derivada.
            feature = self._build_feature(
                mode=instruction.interaction_mode,
                base_scores=base_scores,
                rarity=rarity_values,
                s0=s0,
            )

            # Diagnóstico de extrapolação / fração ativa do hinge.
            score_min = float(np.min(base_scores)) if base_scores.size else float("nan")
            score_max = float(np.max(base_scores)) if base_scores.size else float("nan")
            inside_range = bool(
                base_scores.size > 0 and score_min <= s0 <= score_max
            )
            fraction_above_s0 = (
                float(np.mean(base_scores > s0))
                if base_scores.size else 0.0
            )
            if instruction.interaction_mode in (
                ScoreInteractionMode.ONE_SIDED_POSITIVE,
                ScoreInteractionMode.HINGE_BASE,
            ):
                fraction_hinge_active = fraction_above_s0
            else:
                fraction_hinge_active = 1.0  # LINEAR sempre "ativo"

            if not inside_range and base_scores.size:
                self._log(
                    context,
                    f"⚠ s0={s0:.6g} está fora do intervalo de base "
                    f"[{score_min:.6g}, {score_max:.6g}]. A feature "
                    "unilateral pode estar quase toda em zero ou quase "
                    "toda ativa — verifique a representatividade do "
                    "domínio de calibração.",
                )

            # Constrói o SimilarityScoreDataset de saída reusando os
            # samples da base (preserva probe/reference/identidades).
            # Em casework, zera same_person e marca o dataset.
            output_name = (
                instruction.output_dataset_name
                or instruction.output_resource_id
                or f"{base_ds.name}_interaction"
            )
            output_metadata = self._build_output_metadata(
                instruction=instruction,
                base_ds=base_ds,
                modifier_ds=modifier_ds,
                modifier_score_type=modifier_score_type,
                modifier_combiner=modifier_combiner,
                rarity_transform_label=rarity_transform_label,
                alignment_orientation=alignment_orientation,
                s0_meta=s0_meta,
                score_min=score_min,
                score_max=score_max,
                inside_range=inside_range,
                fraction_above_s0=fraction_above_s0,
                fraction_hinge_active=fraction_hinge_active,
            )
            # Propaga r0 (quando center_rarity ativo) — incluído em
            # paralelo às chaves s0 pra auditoria e INHERIT downstream.
            output_metadata.update(r0_meta)

            output_samples = self._build_output_samples(
                base_items=base_ds.items,
                feature=feature,
                is_casework=instruction.is_casework,
            )

            output_dataset = SimilarityScoreDataset(
                name=output_name,
                description=(
                    f"Feature de interação ({instruction.interaction_mode.value}) "
                    f"sobre '{instruction.base_score_resource_id}'"
                    + (
                        f" × '{instruction.modifier_score_resource_id}'"
                        if instruction.modifier_score_resource_id
                        else ""
                    )
                    + f" (s0={s0:.6g}, mode={instruction.s0_mode.value})."
                ),
                items=output_samples,
                metadata=output_metadata,
            )

            output_id = instruction.output_resource_id
            output_resource = StrategyResource(
                resource_id=output_id,
                name=output_id,
                resource_type=StrategyResourceType.SIMILARITY_SCORES,
                payload=output_dataset,
                artifact_ref=None,
            )
            context.set_resource(output_resource)

            n_pairs = len(output_samples)
            self._log(
                context,
                f"✅ Feature de interação produzida: '{output_id}' "
                f"(n_pairs={n_pairs:,}, s0={s0:.6g}, "
                f"mode={instruction.interaction_mode.value}, "
                f"s0_mode={instruction.s0_mode.value}).".replace(",", "."),
            )
            if instruction.s0_mode is S0Mode.AUTO_LOGREG:
                self._log(
                    context,
                    "⚠ AUTO_LOGREG: s0 foi estimado no conjunto completo. "
                    "Métricas de CV interno do calibrador final neste "
                    "mesmo dataset podem ser OTIMISTAS — use holdout "
                    "externo com s0_mode=INHERIT para avaliação honesta.",
                )

            execution_metadata: Dict[str, Any] = {
                "instruction_id": instruction.instruction_id,
                "instruction_name": instruction.name,
                "output_resource_id": output_id,
                "n_pairs": int(n_pairs),
                "is_interaction_feature": True,
                "interaction_mode": instruction.interaction_mode.to_str(),
                "interaction_s0": float(s0),
                "interaction_s0_mode": instruction.s0_mode.to_str(),
                "alignment_orientation": alignment_orientation,
                "fraction_hinge_active": float(fraction_hinge_active),
                "interaction_s0_inside_score_range": inside_range,
                "is_casework": bool(instruction.is_casework),
            }
            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_id],
                metadata=execution_metadata,
            )

        except Exception as exc:
            self._log(
                context,
                f"❌ Erro em SCORE_INTERACTION_FEATURE "
                f"({instruction.instruction_id}): {exc}",
            )
            return self._failure_result(str(exc))

    # ─────────────────────── instruction-specific ─────────────────

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, ScoreInteractionFeatureInstruction):
            result.add_error(
                "A instrução informada não é do tipo ScoreInteractionFeatureInstruction."
            )
            return result

        if not instruction.base_score_resource_id:
            result.add_error("A instrução deve definir 'base_score_resource_id'.")
        elif not context.has_resource(instruction.base_score_resource_id):
            result.add_error(
                f"Recurso base '{instruction.base_score_resource_id}' "
                "não encontrado no contexto."
            )

        # Modifier obrigatório exceto em HINGE_BASE.
        if instruction.interaction_mode is not ScoreInteractionMode.HINGE_BASE:
            if not instruction.modifier_score_resource_id:
                result.add_error(
                    "A instrução deve definir 'modifier_score_resource_id' "
                    "(exceto em HINGE_BASE)."
                )
            elif not context.has_resource(instruction.modifier_score_resource_id):
                result.add_error(
                    f"Recurso modifier '{instruction.modifier_score_resource_id}' "
                    "não encontrado no contexto."
                )

        if not instruction.output_resource_id:
            result.add_error("A instrução deve definir 'output_resource_id'.")

        # Tipos dos payloads + bloqueio anti-recursão no base.
        base_id = instruction.base_score_resource_id
        if base_id and context.has_resource(base_id):
            payload = context.get_resource(base_id).payload
            if payload is not None:
                if not isinstance(payload, SimilarityScoreDataset):
                    result.add_error(
                        f"Recurso base '{base_id}' não é SimilarityScoreDataset "
                        f"(tem {type(payload).__name__})."
                    )
                elif bool((payload.metadata or {}).get("is_interaction_feature")):
                    result.add_error(
                        f"Recurso base '{base_id}' já é uma "
                        "interaction feature — anti-recursão: o base deve "
                        "ser score puro (cosseno/PLDA), não derivado."
                    )

        modifier_id = instruction.modifier_score_resource_id
        if (
            instruction.interaction_mode is not ScoreInteractionMode.HINGE_BASE
            and modifier_id
            and context.has_resource(modifier_id)
        ):
            payload = context.get_resource(modifier_id).payload
            if payload is not None:
                if not isinstance(payload, SimilarityScoreDataset):
                    result.add_error(
                        f"Recurso modifier '{modifier_id}' não é "
                        f"SimilarityScoreDataset (tem {type(payload).__name__})."
                    )
                else:
                    metadata = payload.metadata or {}
                    score_kind = metadata.get("score_kind")
                    score_type = metadata.get("score_type")
                    valid_types = {t.value for t in TypicalityScoreType}
                    if not (
                        score_kind == "typicality_pair"
                        or (score_type and str(score_type) in valid_types)
                    ):
                        result.add_error(
                            f"Recurso modifier '{modifier_id}' não é de "
                            "família tipicidade. Esperado: output de "
                            "TYPICALITY_PAIR_SCORING (metadata "
                            "`score_kind=typicality_pair` ou `score_type` "
                            "∈ {TYPICALITY_PERCENTILE, MEAN_KNN_SIMILARITY, "
                            "LOG_DENSITY_KNN})."
                        )

        # MANUAL / INHERIT validations.
        if instruction.s0_mode is S0Mode.MANUAL:
            if instruction.s0_manual_value is None:
                result.add_error(
                    "s0_mode=MANUAL requer 's0_manual_value' preenchido."
                )

        if instruction.s0_mode is S0Mode.INHERIT:
            inherit_id = instruction.inherit_s0_from_resource_id
            if not inherit_id:
                result.add_error(
                    "s0_mode=INHERIT requer 'inherit_s0_from_resource_id'."
                )
            elif not context.has_resource(inherit_id):
                result.add_error(
                    f"Recurso para herdar s0 '{inherit_id}' não encontrado."
                )
            else:
                ensure_payload = getattr(context, "ensure_payload", None)
                if callable(ensure_payload):
                    try:
                        ensure_payload(inherit_id)
                    except Exception:
                        pass
                resource = context.get_resource(inherit_id)
                payload = resource.payload if resource is not None else None
                if isinstance(payload, SimilarityScoreDataset):
                    meta = payload.metadata or {}
                    if "interaction_s0" not in meta:
                        result.add_error(
                            f"Recurso '{inherit_id}' não carrega "
                            "'interaction_s0' no metadata — não foi "
                            "produzido por SCORE_INTERACTION_FEATURE."
                        )
                    else:
                        self._maybe_warn_inherit_base_mismatch(
                            inherit_meta=meta,
                            base_id=instruction.base_score_resource_id,
                            context=context,
                            result=result,
                            inherit_id=inherit_id,
                        )

        # AUTO_LOGREG em casework é proibido.
        if (
            instruction.s0_mode is S0Mode.AUTO_LOGREG
            and instruction.is_casework
        ):
            result.add_error(
                "s0_mode=AUTO_LOGREG é incompatível com is_casework=True "
                "(sem ground truth para ajustar o calibrador auxiliar). "
                "Use s0_mode=INHERIT em casework."
            )

        # AUTO_LOGREG com poucos pares → warning não-crítico.
        if instruction.s0_mode is S0Mode.AUTO_LOGREG and base_id and context.has_resource(base_id):
            payload = context.get_resource(base_id).payload
            if isinstance(payload, SimilarityScoreDataset):
                if 0 < len(payload.items) < AUTO_LOGREG_MIN_PAIRS_FOR_NO_WARNING:
                    result.add_warning(
                        f"AUTO_LOGREG com poucos pares "
                        f"(n={len(payload.items)}) — s0 pode ser instável. "
                        "Considere MANUAL ou aumente o dataset de "
                        "calibração."
                    )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()
        return result

    # ─────────────────────── helpers (dataset) ────────────────────

    @staticmethod
    def _fetch_dataset(
        resource_id: Optional[str],
        context: StrategyExecutionContextProtocol,
        role: str,
    ) -> SimilarityScoreDataset:
        if not resource_id:
            raise ValueError(f"Recurso '{role}' não foi informado.")
        ensure_payload = getattr(context, "ensure_payload", None)
        if callable(ensure_payload):
            try:
                ensure_payload(resource_id)
            except Exception:
                pass
        resource = context.get_resource(resource_id)
        if resource is None or resource.payload is None:
            raise ValueError(
                f"Recurso '{resource_id}' ({role}) ausente ou sem payload."
            )
        payload = resource.payload
        if not isinstance(payload, SimilarityScoreDataset):
            raise TypeError(
                f"Recurso '{resource_id}' ({role}) não é SimilarityScoreDataset "
                f"(tem {type(payload).__name__})."
            )
        return payload

    @staticmethod
    def _reject_recursive_base(
        base_ds: SimilarityScoreDataset,
        base_id: Optional[str],
    ) -> None:
        meta = base_ds.metadata or {}
        if meta.get("is_interaction_feature"):
            raise ValueError(
                f"Recurso base '{base_id}' já é uma interaction feature "
                "— anti-recursão: use o score puro (cosseno/PLDA)."
            )

    @staticmethod
    def _validate_modifier_family(
        modifier_ds: SimilarityScoreDataset,
        modifier_id: Optional[str],
    ) -> None:
        meta = modifier_ds.metadata or {}
        score_kind = meta.get("score_kind")
        score_type = meta.get("score_type")
        valid_types = {t.value for t in TypicalityScoreType}
        if score_kind == "typicality_pair":
            return
        if score_type and str(score_type) in valid_types:
            return
        raise ValueError(
            f"Modifier '{modifier_id}' não é de família tipicidade. "
            "Esperado output de TYPICALITY_PAIR_SCORING "
            "(metadata `score_kind=typicality_pair` ou `score_type` em "
            "{TYPICALITY_PERCENTILE, MEAN_KNN_SIMILARITY, LOG_DENSITY_KNN})."
        )

    @staticmethod
    def _require_score_type(
        modifier_ds: SimilarityScoreDataset,
        modifier_id: Optional[str],
    ) -> TypicalityScoreType:
        meta = modifier_ds.metadata or {}
        raw = meta.get("score_type")
        if not raw:
            raise ValueError(
                f"Modifier '{modifier_id}' não tem 'score_type' no "
                "metadata. Regenere o modifier via TYPICALITY_PAIR_SCORING "
                "para garantir a presença do score_type — sem ele a "
                "inversão tipicidade→raridade fica ambígua."
            )
        return TypicalityScoreType.from_str(str(raw))

    @staticmethod
    def _safe_parse_combiner(raw: Optional[str]) -> Optional[TypicalityCombiner]:
        if not raw:
            return None
        try:
            return TypicalityCombiner.from_str(str(raw))
        except Exception:
            return None

    # ─────────────────────── alignment ────────────────────────────

    @staticmethod
    def _pair_key_unordered(sample: SimilarityScoreSample) -> UnorderedPairKey:
        probe_hash = getattr(sample.probe, "face_hash", None) or ""
        ref_hash = getattr(sample.reference, "face_hash", None) or ""
        return frozenset({str(probe_hash), str(ref_hash)})

    @staticmethod
    def _pair_key_ordered(sample: SimilarityScoreSample) -> OrderedPairKey:
        probe_hash = getattr(sample.probe, "face_hash", None) or ""
        ref_hash = getattr(sample.reference, "face_hash", None) or ""
        return (str(probe_hash), str(ref_hash))

    def _align_pairs(
        self,
        base_ds: SimilarityScoreDataset,
        modifier_ds: SimilarityScoreDataset,
        orientation: str,
        base_id: Optional[str],
        modifier_id: Optional[str],
        context: StrategyExecutionContextProtocol,
    ) -> List[SimilarityScoreSample]:
        """Reordena `modifier.items` para casar com `base.items`.

        Em modo `ordered`, a chave é `(probe_hash, reference_hash)` —
        sensível à orientação do par (necessário em REFERENCE_ONLY/
        PROBE_ONLY). Em modo `unordered`, usa `frozenset({p, r})`
        (compat simétrico).

        Detecta **chaves duplicadas** em qualquer lado e levanta ERRO
        explícito (evita silenciamento por sobrescrita em dict).
        """
        if orientation == "ordered":
            key_fn = self._pair_key_ordered
        else:
            key_fn = self._pair_key_unordered

        if len(modifier_ds.items) != len(base_ds.items):
            raise ValueError(
                f"Modifier '{modifier_id}' tem {len(modifier_ds.items)} "
                f"pares; base '{base_id}' tem {len(base_ds.items)}. "
                "SCORE_INTERACTION_FEATURE exige conjuntos idênticos "
                "de pares — verifique se há PAIR_MINING em apenas um "
                "dos lados."
            )

        # Anti-duplicata (base).
        base_keys = [key_fn(s) for s in base_ds.items]
        seen_base: dict = {}
        for idx, k in enumerate(base_keys):
            if k in seen_base:
                raise ValueError(
                    f"Pares duplicados detectados no base '{base_id}': "
                    f"chave repetida em índices {seen_base[k]} e {idx}. "
                    "Verifique geração de pares (auto-comparação mal "
                    "construída ou duplicação acidental)."
                )
            seen_base[k] = idx

        # Anti-duplicata (modifier).
        modifier_keys = [key_fn(s) for s in modifier_ds.items]
        seen_mod: dict = {}
        for idx, k in enumerate(modifier_keys):
            if k in seen_mod:
                raise ValueError(
                    f"Pares duplicados detectados no modifier "
                    f"'{modifier_id}': chave repetida em índices "
                    f"{seen_mod[k]} e {idx}."
                )
            seen_mod[k] = idx

        # Fast path: já alinhado posicionalmente.
        if base_keys == modifier_keys:
            return list(modifier_ds.items)

        self._log(
            context,
            f"  ↻ Modifier '{modifier_id}' não está na mesma ordem que "
            f"base '{base_id}' ({orientation}) — reordenando…",
        )

        index_modifier = dict(zip(modifier_keys, modifier_ds.items))
        aligned: List[SimilarityScoreSample] = []
        missing: List[PairKey] = []
        for k in base_keys:
            sample = index_modifier.get(k)
            if sample is None:
                missing.append(k)
                if len(missing) > 5:
                    break
            else:
                aligned.append(sample)
        if missing:
            sample_str = ", ".join(self._fmt_pair_key(m) for m in missing[:5])
            raise ValueError(
                f"{len(missing)}+ pares do base '{base_id}' não estão "
                f"em '{modifier_id}' (ex: {sample_str}). Verifique que "
                "ambos processaram o mesmo conjunto de pares com a "
                f"mesma orientação ({orientation})."
            )
        return aligned

    @staticmethod
    def _fmt_pair_key(key: PairKey) -> str:
        if isinstance(key, frozenset):
            return "{" + ", ".join(h[:8] + ".." for h in sorted(key)) + "}"
        return "(" + ", ".join(h[:8] + ".." for h in key) + ")"

    @staticmethod
    def _validate_same_person_consistency(
        base_items: List[SimilarityScoreSample],
        modifier_items: List[SimilarityScoreSample],
        base_id: Optional[str],
        modifier_id: Optional[str],
    ) -> None:
        """Em modo não-casework, exige `same_person` idêntico sample-a-sample.

        Divergência indica que os datasets vieram de splits diferentes
        ou foram corrompidos no pipeline (chaves alinharam por hash
        mas labels não batem — bug perigoso e silencioso sem essa
        checagem).
        """
        max_mismatches = 5
        mismatches: List[int] = []
        for idx, (b, m) in enumerate(zip(base_items, modifier_items)):
            if b.same_person != m.same_person:
                mismatches.append(idx)
                if len(mismatches) >= max_mismatches:
                    break
        if mismatches:
            raise ValueError(
                f"Rótulos `same_person` divergem entre base '{base_id}' "
                f"e modifier '{modifier_id}' em pelo menos "
                f"{len(mismatches)} pares (índices: {mismatches}). "
                "Verifique que ambos os datasets vieram do MESMO split "
                "ou marque `is_casework=True` (em casework ambos devem "
                "ser None)."
            )

    # ─────────────────────── transforms / s0 ──────────────────────

    @staticmethod
    def _to_rarity(
        typicality: np.ndarray,
        score_type: TypicalityScoreType,
        invert: bool,
    ) -> Tuple[np.ndarray, str]:
        """Converte tipicidade → raridade conforme `score_type`.

        Inversão = True (default):
        - TYPICALITY_PERCENTILE: o `TYPICALITY_PAIR_SCORING` upstream já
          normaliza o percentil para [0,1] (via
          `get_percentiles(matrix)/100.0`), então a entrada aqui já
          está em [0,1]. Aplica `1 − v` direto. Label =
          `"1 - percentile_0_1"`. Dividir por 100 novamente neste passo
          (como uma versão anterior fazia) produzia raridade saturada
          em ~0.99 para todos os pares — bug histórico já corrigido.
        - MEAN_KNN_SIMILARITY ([-1,1]): `−v`. Label = `"−mean_knn_similarity"`.
        - LOG_DENSITY_KNN (ℝ): `−v`. Label = `"−log_density_knn"`.

        Inversão = False: passthrough da tipicidade na escala em que
        veio — útil quando o usuário quer que o sinal do coeficiente
        do LogReg multivariado reflita tipicidade diretamente
        (não-cosmético no caso unilateral com offset, mas permitido
        para auditoria).
        """
        if score_type is TypicalityScoreType.TYPICALITY_PERCENTILE:
            # Bugfix: TYPICALITY_PAIR_SCORING já normaliza percentile pra [0,1]
            # (linha 369 do typicality_pair_scoring_instruction_executor.py:
            # `return model.get_percentiles(matrix) / 100.0`). Dividir por 100
            # de novo aqui produzia raridade saturada em ~0.99 pra todos os
            # pares, neutralizando a feature de interação. Mantemos o valor
            # como recebido (já em [0,1]).
            v01 = typicality.astype(np.float64)
            if invert:
                return 1.0 - v01, "1 - percentile_0_1"
            return v01, "percentile_0_1"
        if score_type is TypicalityScoreType.MEAN_KNN_SIMILARITY:
            arr = typicality.astype(np.float64)
            if invert:
                return -arr, "−mean_knn_similarity"
            return arr, "mean_knn_similarity"
        if score_type is TypicalityScoreType.LOG_DENSITY_KNN:
            arr = typicality.astype(np.float64)
            if invert:
                return -arr, "−log_density_knn"
            return arr, "log_density_knn"
        raise ValueError(f"score_type não suportado: {score_type}")

    def _resolve_s0(
        self,
        instruction: ScoreInteractionFeatureInstruction,
        base_scores: np.ndarray,
        base_ds: SimilarityScoreDataset,
        context: StrategyExecutionContextProtocol,
    ) -> Dict[str, Any]:
        """Resolve s0 conforme `s0_mode` e devolve dict pronto pra metadata."""
        mode = instruction.s0_mode

        if mode is S0Mode.MANUAL:
            return {
                "interaction_s0": float(instruction.s0_manual_value),
                "interaction_s0_mode": mode.to_str(),
                "interaction_s0_source_resource_id": None,
                "interaction_s0_coef": None,
                "interaction_s0_intercept": None,
                "interaction_s0_C": None,
                "interaction_s0_training_method": "manual",
                "interaction_s0_coef_sign": None,
            }

        if mode is S0Mode.INHERIT:
            inherit_id = instruction.inherit_s0_from_resource_id
            resource = context.get_resource(inherit_id) if inherit_id else None
            payload = resource.payload if resource is not None else None
            if not isinstance(payload, SimilarityScoreDataset):
                raise ValueError(
                    f"Recurso para herdar s0 '{inherit_id}' não é "
                    "SimilarityScoreDataset."
                )
            inherited_meta = payload.metadata or {}
            if "interaction_s0" not in inherited_meta:
                raise ValueError(
                    f"Recurso '{inherit_id}' não carrega 'interaction_s0' "
                    "no metadata."
                )
            return {
                "interaction_s0": float(inherited_meta["interaction_s0"]),
                "interaction_s0_mode": mode.to_str(),
                "interaction_s0_source_resource_id": inherit_id,
                "interaction_s0_coef": inherited_meta.get("interaction_s0_coef"),
                "interaction_s0_intercept": inherited_meta.get(
                    "interaction_s0_intercept"
                ),
                "interaction_s0_C": inherited_meta.get("interaction_s0_C"),
                "interaction_s0_training_method": "inherit",
                "interaction_s0_coef_sign": inherited_meta.get(
                    "interaction_s0_coef_sign"
                ),
            }

        # AUTO_LOGREG
        labels = self._collect_labels(base_ds.items)
        if labels is None:
            raise ValueError(
                "AUTO_LOGREG: o dataset base não tem ground truth "
                "binário em todos os samples. Use s0_mode=MANUAL ou "
                "INHERIT."
            )
        unique = np.unique(labels)
        if unique.size < 2:
            raise ValueError(
                "AUTO_LOGREG: o dataset base só tem uma classe "
                f"(labels={list(unique)}). Use MANUAL ou aumente o "
                "dataset de calibração para incluir gen+imp."
            )

        from sklearn.linear_model import LogisticRegression

        cfg = instruction.s0_logreg_config_override or LogRegS0Config()
        clf = LogisticRegression(
            C=cfg.C,
            solver=cfg.solver,
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
        )
        X = base_scores.reshape(-1, 1)
        clf.fit(X, labels)
        coef = float(clf.coef_[0, 0])
        intercept = float(clf.intercept_[0])

        if abs(coef) < COEF_ZERO_EPSILON:
            raise ValueError(
                f"AUTO_LOGREG: calibrador auxiliar degenerou "
                f"(coef={coef:.3e} ≈ 0). O score base não separa as "
                "classes — verifique o pipeline e o domínio de "
                "calibração."
            )
        if coef < 0:
            raise ValueError(
                f"AUTO_LOGREG: coeficiente negativo (coef={coef:.6g}). "
                "s0 perde a interpretação 'maior base = maior evidência "
                "Hp'. Verifique a direção do score base (talvez seja "
                "distância em vez de similaridade)."
            )

        s0 = -intercept / coef
        return {
            "interaction_s0": float(s0),
            "interaction_s0_mode": mode.to_str(),
            "interaction_s0_source_resource_id": None,
            "interaction_s0_coef": coef,
            "interaction_s0_intercept": intercept,
            "interaction_s0_C": cfg.C,
            "interaction_s0_training_method": "auto_logreg",
            "interaction_s0_coef_sign": "+" if coef > 0 else ("-" if coef < 0 else "0"),
        }

    def _resolve_r0(
        self,
        instruction: ScoreInteractionFeatureInstruction,
        rarity_values: Optional[np.ndarray],
        context: StrategyExecutionContextProtocol,
    ) -> Dict[str, Any]:
        """Resolve r0 (centralização da raridade) em paralelo a s0.

        Convenção: piggy-back no `s0_mode` da própria instrução —
        AUTO_LOGREG → calcula r0 = mean(rarity) no dataset atual e o
        persiste; INHERIT → lê `interaction_r0` do mesmo recurso
        apontado por `inherit_s0_from_resource_id`. MANUAL não suporta
        centralização (r0 ficaria undefined em runtime).

        Quando `center_rarity=False`, retorna o dict com r0=None (no-op).
        Idempotente para HINGE_BASE (onde rarity é None).
        """
        if not instruction.center_rarity:
            return {
                "interaction_r0": None,
                "interaction_r0_mode": None,
                "interaction_r0_source_resource_id": None,
            }
        if rarity_values is None:
            # HINGE_BASE ignora rarity — centralização vira no-op.
            return {
                "interaction_r0": None,
                "interaction_r0_mode": None,
                "interaction_r0_source_resource_id": None,
            }

        mode = instruction.s0_mode  # piggy-back
        if mode is S0Mode.MANUAL:
            raise ValueError(
                "center_rarity=True com s0_mode=MANUAL não é suportado — "
                "r0 ficaria indefinido. Use AUTO_LOGREG (calcula r0 = "
                "mean(rarity) no dataset atual) ou INHERIT (lê de execução "
                "anterior)."
            )

        if mode is S0Mode.INHERIT:
            inherit_id = instruction.inherit_s0_from_resource_id
            resource = context.get_resource(inherit_id) if inherit_id else None
            payload = resource.payload if resource is not None else None
            if not isinstance(payload, SimilarityScoreDataset):
                raise ValueError(
                    f"center_rarity: recurso para herdar r0 '{inherit_id}' "
                    "não é SimilarityScoreDataset."
                )
            inherited_meta = payload.metadata or {}
            if "interaction_r0" not in inherited_meta or inherited_meta.get("interaction_r0") is None:
                raise ValueError(
                    f"center_rarity: recurso '{inherit_id}' não carrega "
                    "'interaction_r0' no metadata. Garanta que a "
                    "execução de calibração rodou com center_rarity=True."
                )
            return {
                "interaction_r0": float(inherited_meta["interaction_r0"]),
                "interaction_r0_mode": mode.to_str(),
                "interaction_r0_source_resource_id": inherit_id,
            }

        # AUTO_LOGREG → calcula r0 in-situ.
        r0 = float(np.mean(rarity_values))
        return {
            "interaction_r0": r0,
            "interaction_r0_mode": mode.to_str(),
            "interaction_r0_source_resource_id": None,
        }

    @staticmethod
    def _collect_labels(
        items: List[SimilarityScoreSample],
    ) -> Optional[np.ndarray]:
        labels: List[int] = []
        for s in items:
            if s.same_person is None:
                return None
            labels.append(1 if s.same_person else 0)
        return np.asarray(labels, dtype=np.int32)

    # ─────────────────────── feature build ───────────────────────

    @staticmethod
    def _build_feature(
        mode: ScoreInteractionMode,
        base_scores: np.ndarray,
        rarity: Optional[np.ndarray],
        s0: float,
    ) -> np.ndarray:
        if mode is ScoreInteractionMode.HINGE_BASE:
            return np.maximum(base_scores - s0, 0.0)
        if rarity is None:
            raise ValueError(
                f"Modo {mode.value} requer modifier (raridade) — "
                "rarity_values é None."
            )
        if mode is ScoreInteractionMode.LINEAR:
            return (base_scores - s0) * rarity
        if mode is ScoreInteractionMode.ONE_SIDED_POSITIVE:
            return np.maximum(base_scores - s0, 0.0) * rarity
        raise ValueError(f"interaction_mode não suportado: {mode}")

    @staticmethod
    def _build_output_samples(
        base_items: List[SimilarityScoreSample],
        feature: np.ndarray,
        is_casework: bool,
    ) -> List[SimilarityScoreSample]:
        out: List[SimilarityScoreSample] = []
        for src, val in zip(base_items, feature):
            sample = SimilarityScoreSample(
                probe=src.probe,
                reference=src.reference,
                raw_score=float(val),
                same_person=None if is_casework else src.same_person,
                metadata={},
            )
            # Em casework `__post_init__` re-infere `same_person` a
            # partir das identidades quando o caller passa None — força
            # de volta após a construção (mesmo padrão do
            # TypicalityPairScoringInstructionExecutor).
            if is_casework:
                sample.same_person = None
            out.append(sample)
        return out

    # ─────────────────────── metadata build ──────────────────────

    @staticmethod
    def _build_output_metadata(
        instruction: ScoreInteractionFeatureInstruction,
        base_ds: SimilarityScoreDataset,
        modifier_ds: Optional[SimilarityScoreDataset],
        modifier_score_type: Optional[TypicalityScoreType],
        modifier_combiner: Optional[TypicalityCombiner],
        rarity_transform_label: Optional[str],
        alignment_orientation: str,
        s0_meta: Dict[str, Any],
        score_min: float,
        score_max: float,
        inside_range: bool,
        fraction_above_s0: float,
        fraction_hinge_active: float,
    ) -> Dict[str, Any]:
        base_meta = base_ds.metadata or {}
        modifier_meta = (modifier_ds.metadata or {}) if modifier_ds is not None else {}

        metadata: Dict[str, Any] = {
            "is_interaction_feature": True,
            "interaction_mode": instruction.interaction_mode.to_str(),
            "invert_rarity": bool(instruction.invert_rarity),
            "base_source_resource_id": instruction.base_score_resource_id,
            "base_score_type": (
                "plda" if "plda" in str(base_meta.get("similarity_model_id", "")).lower()
                else (
                    "cosine"
                    if "cos" in str(base_meta.get("similarity_model_id", "")).lower()
                    else base_meta.get("similarity_model_id")
                )
            ),
            "base_score_name": base_ds.name,
            "base_similarity_model_resource_id": base_meta.get("similarity_model_id"),
            "base_embedding_column": base_meta.get("embedding_column"),
            "modifier_source_resource_id": instruction.modifier_score_resource_id,
            "modifier_score_type": (
                modifier_score_type.to_str() if modifier_score_type is not None else None
            ),
            "modifier_combiner": (
                modifier_combiner.to_str() if modifier_combiner is not None else None
            ),
            "rarity_transform": rarity_transform_label,
            "alignment_orientation": alignment_orientation,
            "interaction_base_score_min": score_min,
            "interaction_base_score_max": score_max,
            "fraction_scores_above_s0": float(fraction_above_s0),
            "fraction_hinge_active": float(fraction_hinge_active),
            "is_casework": bool(instruction.is_casework),
            # Marca como `score_kind` para diferenciar do "typicality_pair"
            # — útil para futuras regras que queiram filtrar por tipo.
            "score_kind": "interaction_feature",
        }
        metadata.update(s0_meta)
        # Compat: garante `s0` está marcado também como inside_range
        # mesmo que tenha vindo de outro DTO via INHERIT.
        metadata["interaction_s0_inside_score_range"] = bool(inside_range)
        return metadata

    @staticmethod
    def _maybe_warn_inherit_base_mismatch(
        inherit_meta: Dict[str, Any],
        base_id: Optional[str],
        context: StrategyExecutionContextProtocol,
        result: ValidationResult,
        inherit_id: str,
    ) -> None:
        """WARNING quando o s0 herdado foi aprendido sobre um base
        diferente do atual (tipo, modelo, ou coluna de embedding).
        Promovível a ERRO em versão futura.
        """
        if not base_id or not context.has_resource(base_id):
            return
        base_payload = context.get_resource(base_id).payload
        if not isinstance(base_payload, SimilarityScoreDataset):
            return
        base_meta = base_payload.metadata or {}

        keys = [
            "base_score_type",
            "base_similarity_model_resource_id",
            "base_embedding_column",
        ]
        # `inherit_meta` carrega esses keys diretamente (gerados pela
        # execução anterior); `base_meta` (cosine output) carrega o
        # `similarity_model_id`/`embedding_column` no formato do SCORING.
        actual = {
            "base_score_type": (
                "plda" if "plda" in str(base_meta.get("similarity_model_id", "")).lower()
                else (
                    "cosine"
                    if "cos" in str(base_meta.get("similarity_model_id", "")).lower()
                    else base_meta.get("similarity_model_id")
                )
            ),
            "base_similarity_model_resource_id": base_meta.get("similarity_model_id"),
            "base_embedding_column": base_meta.get("embedding_column"),
        }
        diffs: List[str] = []
        for k in keys:
            inherited = inherit_meta.get(k)
            current = actual.get(k)
            if inherited is None and current is None:
                continue
            if inherited != current:
                diffs.append(f"{k}: herdado={inherited!r} vs atual={current!r}")

        if diffs:
            result.add_warning(
                f"s0 herdado de '{inherit_id}' foi aprendido para base "
                f"diferente do atual ({'; '.join(diffs)}). Reuso pode "
                "invalidar a definição de 'evidência neutra' — em "
                "versão futura isso será promovido a ERRO."
            )
