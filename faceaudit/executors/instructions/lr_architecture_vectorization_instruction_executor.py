# faceaudit/executors/instructions/lr_architecture_vectorization_instruction_executor.py
"""
Executor da instrução `LR_ARCHITECTURE_VECTORIZATION`.

Fail-closed: levanta `MissingExecutionFrameError` quando
`context.frame is None`. Sem frame não há como derivar fase, e ajustar
s₀/r₀ silenciosamente em casework seria uma falha de segurança.

Fluxo:

  1. Validar frame e estrutura do DTO.
  2. Carregar base + (se aplicável) rarity_probe / rarity_reference.
  3. Verificar score_kind / score_type / combiner / pair_role_orientation
     dos inputs com base no que a arquitetura exige.
  4. Construir array canônico de raridade via `RarityTransformRegistry`.
  5. Resolver state:
       - PREPARATION + FIT_FULL_CALIBRATION_LEGACY: AUTO_LOGREG sobre todo
         D_cal + r̄ sobre D_cal.
       - PREPARATION + MANUAL: usa valores manuais (validação per-arq).
       - PREPARATION + INHERIT (raro): carrega state + valida compat.
       - VALIDATION/CASEWORK: obrigatório INHERIT; carrega state e valida.
       - PREPARATION + FIT_PER_FOLD: erro (v2 only).
  6. Construir matriz via `ArchitectureFeatureBuilder.build`.
  7. Emitir SimilarityScoreDataset com metadata completo (incluindo
     `vectorization_contract_fingerprint`, ambos os hashes, e
     `architecture_state`).
  8. Opcionalmente persistir LR_ARCHITECTURE_RAW_INPUT no contexto.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from faceaudit.enums.architecture_fit_scope import ArchitectureFitScope
from faceaudit.enums.architecture_parameter_mode import ArchitectureParameterMode
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.pair_key_mode import PairKeyMode
from faceaudit.enums.pair_role_orientation import PairRoleOrientation
from faceaudit.enums.s0_resolution_method import S0ResolutionMethod
from faceaudit.enums.strategy_phase import StrategyPhase
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.instructions.lr_architecture_vectorization_instruction import (
    LRArchitectureVectorizationInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.lr.lr_architecture_raw_feature_schema import (
    LRArchitectureRawFeatureSchema,
    schema_for_base_and_rarities,
    schema_for_base_only,
)
from faceaudit.models.lr.lr_architecture_raw_input_payload import (
    LRArchitectureRawInputPayloadFactory,
    _compute_pair_key_order_hash,
)
from faceaudit.models.lr.lr_architecture_spec import LR_ARCHITECTURE_SPECS
from faceaudit.models.lr.lr_architecture_state import (
    LRArchitectureState,
    LRArchitectureStateFactory,
    StateIntegrityError,
)
from faceaudit.models.lr._canonical_hash import sha256_hex
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services import architecture_feature_builder as builder
from faceaudit.services.architecture_state_compat_service import (
    validate_state_compatibility,
)
from faceaudit.services.rarity_transform_service import (
    RarityTransformRegistry,
    to_canonical_rarity,
)
from faceaudit.services.s0_resolution_service import (
    S0FitSummary,
    S0ResolutionError,
    resolve_s0_auto_logreg,
    resolve_s0_manual,
)


logger = logging.getLogger(__name__)


class MissingExecutionFrameError(RuntimeError):
    """
    Levantada quando o executor é invocado sem `context.frame`.

    A nova instrução `LR_ARCHITECTURE_VECTORIZATION` não existia em JSONs
    históricos — não precisa aceitar execução fora do orquestrador.
    Fail-closed evita ajuste silencioso de s₀/r₀ em casework por
    descuido do caller.
    """


class LRArchitectureVectorizationInstructionExecutor(
    StrategyInstructionExecutorBase
):
    """Executor da instrução `LR_ARCHITECTURE_VECTORIZATION`."""

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        if not isinstance(instruction, LRArchitectureVectorizationInstruction):
            return ValidationResult.critical(
                f"Esperado LRArchitectureVectorizationInstruction; "
                f"recebido {type(instruction).__name__}"
            )

        # Frame é obrigatório — fail-closed
        frame = getattr(context, "frame", None)
        if frame is None:
            raise MissingExecutionFrameError(
                "LR_ARCHITECTURE_VECTORIZATION exige InstructionExecutionFrame no "
                "contexto. O orquestrador deve chamar `context.with_frame(frame)` "
                "antes de invocar este executor."
            )

        spec = LR_ARCHITECTURE_SPECS.get(instruction.architecture)
        if spec is None:
            return ValidationResult.critical(
                f"Arquitetura desconhecida: {instruction.architecture}"
            )

        # Fase × parameter_mode
        phase: StrategyPhase = frame.phase
        mode = instruction.parameter_mode
        if phase != StrategyPhase.PREPARATION:
            if mode != ArchitectureParameterMode.INHERIT:
                return ValidationResult.critical(
                    f"Fase {phase.value} exige parameter_mode=INHERIT; "
                    f"recebido {mode.value}"
                )
        # FIT_PER_FOLD habilitado no PR 3: o ajuste real por fold acontece
        # dentro do CALIBRATION_MODEL_TRAINING via Pipeline+GridSearchCV.
        # Esta instrução apenas marca o output como "defer to calibrator
        # per-fold" — ela emite o LR_ARCHITECTURE_RAW_INPUT e um vetor
        # placeholder construído com s₀/r̄ provisórios sobre todo D_cal
        # para preservar o contrato (calibrador troca pelos valores
        # corretos pós-CV).

        # base obrigatório
        if not instruction.base_score_resource_id:
            return ValidationResult.critical(
                "base_score_resource_id é obrigatório"
            )

        # tipicidade exigida pela arquitetura?
        if spec.requires_typicality_probe and not instruction.rarity_probe_resource_id:
            return ValidationResult.critical(
                f"Arquitetura {spec.architecture.value} exige "
                "rarity_probe_resource_id"
            )
        if spec.requires_typicality_reference and not instruction.rarity_reference_resource_id:
            return ValidationResult.critical(
                f"Arquitetura {spec.architecture.value} exige "
                "rarity_reference_resource_id"
            )

        # MANUAL: validação per-arquitetura
        if mode == ArchitectureParameterMode.MANUAL:
            if spec.requires_s0 and instruction.manual_s0 is None:
                return ValidationResult.critical(
                    f"parameter_mode=MANUAL: manual_s0 é obrigatório para "
                    f"{spec.architecture.value}"
                )
            if (
                spec.requires_typicality_probe
                and spec.centering_mode.value == "MEAN"
                and instruction.manual_r0_probe is None
            ):
                return ValidationResult.critical(
                    f"parameter_mode=MANUAL: manual_r0_probe é obrigatório "
                    f"para {spec.architecture.value} (centering=MEAN)"
                )
            if (
                spec.requires_typicality_reference
                and spec.centering_mode.value == "MEAN"
                and instruction.manual_r0_reference is None
            ):
                return ValidationResult.critical(
                    f"parameter_mode=MANUAL: manual_r0_reference é obrigatório "
                    f"para {spec.architecture.value} (centering=MEAN)"
                )

        # INHERIT requer recurso
        if mode == ArchitectureParameterMode.INHERIT:
            if not instruction.inherit_architecture_state_from_resource_id:
                return ValidationResult.critical(
                    "parameter_mode=INHERIT exige "
                    "inherit_architecture_state_from_resource_id"
                )

        # Output obrigatório
        if not instruction.output_resource_id:
            return ValidationResult.critical("output_resource_id é obrigatório")

        return ValidationResult.ok()

    # ─────────────────────────────────────────────────────────────────────
    # Execução
    # ─────────────────────────────────────────────────────────────────────

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return InstructionExecutionResult.failure(errors=validation.errors)

        assert isinstance(instruction, LRArchitectureVectorizationInstruction)
        spec = LR_ARCHITECTURE_SPECS[instruction.architecture]
        frame = context.frame
        assert frame is not None  # garantido pela validação

        try:
            return self._execute_inner(instruction, context, spec, frame.phase)
        except (S0ResolutionError, StateIntegrityError, ValueError) as exc:
            return InstructionExecutionResult.failure(errors=[str(exc)])

    def _execute_inner(
        self,
        instr: LRArchitectureVectorizationInstruction,
        context: StrategyExecutionContextProtocol,
        spec,
        phase: StrategyPhase,
    ) -> InstructionExecutionResult:
        # Hidratação de inputs
        context.ensure_payload(instr.base_score_resource_id)
        base_res = context.get_resource(instr.base_score_resource_id)
        if base_res is None or base_res.payload is None:
            return InstructionExecutionResult.failure(
                errors=[f"Recurso base ausente: {instr.base_score_resource_id}"]
            )
        base_ds: SimilarityScoreDataset = base_res.payload
        if not isinstance(base_ds, SimilarityScoreDataset):
            return InstructionExecutionResult.failure(
                errors=[
                    f"Payload de {instr.base_score_resource_id} não é "
                    f"SimilarityScoreDataset (tipo={type(base_ds).__name__})"
                ]
            )

        rarity_probe_ds = self._load_optional_payload(
            context, instr.rarity_probe_resource_id
        )
        rarity_reference_ds = self._load_optional_payload(
            context, instr.rarity_reference_resource_id
        )

        if spec.requires_typicality_probe and rarity_probe_ds is None:
            return InstructionExecutionResult.failure(
                errors=[
                    f"Arquitetura {spec.architecture.value} exige rarity_probe, "
                    "mas o recurso não foi encontrado/hidratado"
                ]
            )
        if spec.requires_typicality_reference and rarity_reference_ds is None:
            return InstructionExecutionResult.failure(
                errors=[
                    f"Arquitetura {spec.architecture.value} exige rarity_reference, "
                    "mas o recurso não foi encontrado/hidratado"
                ]
            )

        # Alinhamento
        try:
            (
                base_scores, rarity_probe_raw, rarity_reference_raw,
                pair_keys, same_person, probe_ids, reference_ids,
                probe_rarity_meta, reference_rarity_meta,
            ) = self._align_inputs(
                base_ds, rarity_probe_ds, rarity_reference_ds, spec,
            )
        except ValueError as exc:
            return InstructionExecutionResult.failure(errors=[str(exc)])

        # Resolver state
        try:
            state, vec_dataset_metadata_extra = self._resolve_state(
                instr=instr, spec=spec, phase=phase, context=context,
                base_scores=base_scores, same_person=same_person,
                rarity_probe_raw=rarity_probe_raw,
                rarity_reference_raw=rarity_reference_raw,
                base_ds_meta=base_ds.metadata,
                probe_rarity_meta=probe_rarity_meta,
                reference_rarity_meta=reference_rarity_meta,
                pair_keys=pair_keys,
            )
        except (S0ResolutionError, StateIntegrityError, ValueError) as exc:
            return InstructionExecutionResult.failure(errors=[str(exc)])

        # Construir matriz vetorizada
        matrix = builder.build(
            spec, state, base_scores,
            rarity_probe_raw, rarity_reference_raw,
        )

        # Fingerprints da matriz
        pair_key_order_hash = _compute_pair_key_order_hash(pair_keys)
        numeric_fp = self._score_vector_numeric_fingerprint(
            matrix, pair_key_order_hash,
        )
        contract_fp = self._score_vector_contract_fingerprint(
            spec, numeric_fp,
        )

        # Construir payload bruto (canonicalizado + read-only)
        schema = self._build_raw_input_schema(spec)
        raw_matrix = self._stack_raw_matrix(
            base_scores, rarity_probe_raw, rarity_reference_raw, schema,
        )
        raw_payload = LRArchitectureRawInputPayloadFactory.create(
            matrix=raw_matrix,
            schema=schema,
            pair_keys=pair_keys,
            same_person=same_person if same_person is not None else None,
            probe_identity_ids=probe_ids if probe_ids else None,
            reference_identity_ids=reference_ids if reference_ids else None,
            pair_role_orientation=state.pair_role_orientation,
            pair_key_mode=state.pair_key_mode,
            phase=phase,
        )

        # Empacotar SimilarityScoreDataset vetorizado.
        # `raw_score` preservado do base (compat com consumidores escalares).
        vec_metadata = {
            "is_score_vector": True,
            "score_vector_matrix": matrix,
            "score_vector_dim": spec.output_dim,
            "score_vector_n_features": spec.output_dim,
            "score_vector_source_resources": list(filter(None, [
                instr.base_score_resource_id,
                instr.rarity_probe_resource_id,
                instr.rarity_reference_resource_id,
            ])),
            # Identidade da arquitetura
            "lr_architecture": spec.architecture.value,
            "lr_architecture_formula_version": spec.formula_version,
            "architecture_spec_fingerprint": spec.fingerprint(),
            "feature_names": list(spec.feature_names),
            "feature_terms": [t.value for t in spec.feature_terms],
            "feature_builder_version": "1.0",
            # State serializado (para INHERIT em VALIDATION/CASEWORK)
            "architecture_state": state.to_dict(),
            # Fingerprints binding ↔ calibrador
            "vectorization_contract_fingerprint": state.vectorization_contract_fingerprint,
            "score_vector_numeric_fingerprint": numeric_fp,
            "score_vector_contract_fingerprint": contract_fp,
            # Provenance dos arrays brutos (auditoria + v2)
            "raw_input_resource_id": instr.output_raw_input_resource_id,
            "raw_input_matrix_fingerprint": raw_payload.raw_input_matrix_fingerprint,
            "raw_input_schema_version": schema.schema_version,
            "raw_input_shape": list(raw_payload.matrix.shape),
            "pair_orientation": state.pair_role_orientation.value,
            "pair_role_orientation": state.pair_role_orientation.value,
            "pair_key_mode": state.pair_key_mode.value,
            "pair_key_order_hash": pair_key_order_hash,
        }
        if vec_dataset_metadata_extra:
            vec_metadata.update(vec_dataset_metadata_extra)

        out_name = f"{instr.name or instr.output_resource_id} (vetorizado)"
        vec_dataset = SimilarityScoreDataset(
            name=out_name,
            description=(
                f"Score vector D={spec.output_dim} pela arquitetura "
                f"{spec.architecture.value}"
            ),
            items=list(base_ds.items),
            metadata=vec_metadata,
        )

        # Registrar output(s) no contexto
        vec_resource = StrategyResource(
            resource_id=instr.output_resource_id,
            name=instr.output_resource_id,
            resource_type=StrategyResourceType.SIMILARITY_SCORES,
            payload=vec_dataset,
        )
        context.set_resource(vec_resource)

        if instr.output_raw_input_resource_id:
            raw_resource = StrategyResource(
                resource_id=instr.output_raw_input_resource_id,
                name=instr.output_raw_input_resource_id,
                resource_type=StrategyResourceType.LR_ARCHITECTURE_RAW_INPUT,
                payload=raw_payload,
            )
            context.set_resource(raw_resource)

        self._log(
            context,
            f"  ✓ {spec.architecture.value} (D={spec.output_dim}) "
            f"vetorizado para {len(matrix)} pares; contract_fp="
            f"{state.vectorization_contract_fingerprint[:8]}…",
        )

        return self._success_result(metadata={
            "lr_architecture": spec.architecture.value,
            "output_dim": spec.output_dim,
            "n_pairs": len(matrix),
            "vectorization_contract_fingerprint": state.vectorization_contract_fingerprint,
        })

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _load_optional_payload(
        self,
        context: StrategyExecutionContextProtocol,
        resource_id: Optional[str],
    ) -> Optional[SimilarityScoreDataset]:
        if not resource_id:
            return None
        context.ensure_payload(resource_id)
        res = context.get_resource(resource_id)
        if res is None or res.payload is None:
            return None
        if not isinstance(res.payload, SimilarityScoreDataset):
            return None
        return res.payload

    def _align_inputs(
        self,
        base_ds: SimilarityScoreDataset,
        rarity_probe_ds: Optional[SimilarityScoreDataset],
        rarity_reference_ds: Optional[SimilarityScoreDataset],
        spec,
    ) -> Tuple[
        np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
        Tuple[Tuple[str, str], ...],
        Optional[np.ndarray],
        Tuple[str, ...], Tuple[str, ...],
        Dict[str, Any], Dict[str, Any],
    ]:
        """
        Constrói arrays alinhados pela ordem de `base_ds.items`. Para
        cada item do base, busca o par correspondente nos datasets de
        raridade por (probe_face_hash, reference_face_hash). Falha se
        algum par não for encontrado nos recursos de raridade
        requeridos pela arquitetura.

        Retorna também:
          - same_person (bool array) quando disponível em todos os items
          - probe_ids, reference_ids (tuplas)
          - metadata "incoming" para uso na validação de compat (state INHERIT)
        """
        n = len(base_ds.items)
        if n == 0:
            raise ValueError("base_ds está vazio")

        base_scores = np.empty(n, dtype=np.float64)
        pair_keys: List[Tuple[str, str]] = []
        same_person_list: List[bool] = []
        any_same_person_missing = False
        probe_ids: List[str] = []
        reference_ids: List[str] = []

        for i, item in enumerate(base_ds.items):
            base_scores[i] = item.raw_score
            probe_hash = getattr(item.probe, "face_hash", "") or ""
            ref_hash = getattr(item.reference, "face_hash", "") or ""
            pair_keys.append((probe_hash, ref_hash))
            sp = item.same_person
            if sp is None:
                any_same_person_missing = True
                same_person_list.append(False)  # placeholder
            else:
                same_person_list.append(bool(sp))
            probe_ids.append(item.probe.identity or "")
            reference_ids.append(item.reference.identity or "")

        same_person_arr = (
            None if any_same_person_missing
            else np.array(same_person_list, dtype=np.bool_)
        )

        # Indexar e alinhar raridades por chave (probe, reference)
        rarity_probe_raw = self._extract_canonical_rarity(
            rarity_probe_ds, pair_keys, side="probe",
        ) if rarity_probe_ds is not None else None
        rarity_reference_raw = self._extract_canonical_rarity(
            rarity_reference_ds, pair_keys, side="reference",
        ) if rarity_reference_ds is not None else None

        # Metadata "incoming" — usado em validate_state_compatibility
        base_meta = {
            "score_kind": base_ds.metadata.get("score_kind") or base_ds.metadata.get("base_score_kind") or "cosine",
            "pair_role_orientation": (
                base_ds.metadata.get("pair_role_orientation")
                or base_ds.metadata.get("pair_orientation")
            ),
        }
        probe_meta = (
            self._collect_rarity_meta(rarity_probe_ds)
            if rarity_probe_ds is not None else {}
        )
        reference_meta = (
            self._collect_rarity_meta(rarity_reference_ds)
            if rarity_reference_ds is not None else {}
        )

        return (
            base_scores, rarity_probe_raw, rarity_reference_raw,
            tuple(pair_keys), same_person_arr,
            tuple(probe_ids), tuple(reference_ids),
            probe_meta, reference_meta,
        )

    def _extract_canonical_rarity(
        self,
        ds: SimilarityScoreDataset,
        pair_keys: List[Tuple[str, str]],
        side: str,
    ) -> np.ndarray:
        """
        Indexa o ds por (probe_hash, reference_hash) e retorna array
        alinhado a `pair_keys`. Cobre depois a conversão canônica via
        `RarityTransformRegistry`.
        """
        score_type_str = ds.metadata.get("score_type") or ds.metadata.get("typicality_score_type")
        if not score_type_str:
            raise ValueError(
                f"Recurso de raridade {side}: metadata.score_type ausente — "
                "TYPICALITY_PAIR_SCORING deve persistir este campo"
            )
        try:
            score_type = TypicalityScoreType.from_str(score_type_str)
        except Exception as exc:
            raise ValueError(
                f"score_type desconhecido em recurso de raridade {side}: "
                f"{score_type_str}"
            ) from exc

        # Indexa por chave orientada
        index: Dict[Tuple[str, str], float] = {}
        for item in ds.items:
            ph = getattr(item.probe, "face_hash", "") or ""
            rh = getattr(item.reference, "face_hash", "") or ""
            key = (ph, rh)
            if key in index:
                raise ValueError(
                    f"Recurso de raridade {side}: chave de par duplicada "
                    f"({ph[:8]}..., {rh[:8]}...)"
                )
            index[key] = item.raw_score

        typicality_arr = np.empty(len(pair_keys), dtype=np.float64)
        missing = 0
        for i, key in enumerate(pair_keys):
            v = index.get(key)
            if v is None:
                missing += 1
                if missing < 5:  # log apenas os primeiros pra não floodar
                    pass
                typicality_arr[i] = np.nan
            else:
                typicality_arr[i] = float(v)
        if missing:
            raise ValueError(
                f"Recurso de raridade {side}: {missing} pares de base não "
                "encontrados — verifique alinhamento (TYPICALITY_PAIR_SCORING "
                "deve produzir saída para todos os pares avaliados em "
                "SIMILARITY_SCORING)"
            )

        # Conversão canônica → raridade
        rarity, _ = to_canonical_rarity(typicality_arr, score_type)
        return rarity

    def _collect_rarity_meta(self, ds: SimilarityScoreDataset) -> Dict[str, Any]:
        m = ds.metadata
        return {
            "score_type": m.get("score_type") or m.get("typicality_score_type"),
            "transform_id": m.get("transform_id"),
            "transform_formula_version": m.get("transform_formula_version"),
            "transform_fingerprint": m.get("transform_fingerprint"),
            "combiner": m.get("combiner") or m.get("typicality_combiner"),
            "model_fingerprint": m.get("model_fingerprint") or m.get("typicality_model_fingerprint"),
            "model_resource_id": m.get("model_resource_id") or m.get("typicality_model_resource_id"),
        }

    # ─── State resolution ────────────────────────────────────────────────

    def _resolve_state(
        self,
        *,
        instr: LRArchitectureVectorizationInstruction,
        spec,
        phase: StrategyPhase,
        context: StrategyExecutionContextProtocol,
        base_scores: np.ndarray,
        same_person: Optional[np.ndarray],
        rarity_probe_raw: Optional[np.ndarray],
        rarity_reference_raw: Optional[np.ndarray],
        base_ds_meta: Dict[str, Any],
        probe_rarity_meta: Dict[str, Any],
        reference_rarity_meta: Dict[str, Any],
        pair_keys: Tuple[Tuple[str, str], ...],
    ) -> Tuple[LRArchitectureState, Optional[Dict[str, Any]]]:
        mode = instr.parameter_mode

        # Determinar pair_role_orientation (estrita: do metadata da base)
        role_orientation = self._determine_pair_role_orientation(
            spec, base_ds_meta,
        )
        pair_key_mode = self._determine_pair_key_mode(base_ds_meta)

        base_score_kind = (
            base_ds_meta.get("score_kind") or base_ds_meta.get("base_score_kind")
            or "cosine"
        )

        if mode == ArchitectureParameterMode.INHERIT:
            return self._resolve_inherit(
                instr=instr, spec=spec, context=context,
                base_meta_for_compat={
                    "score_kind": base_score_kind,
                    "pair_role_orientation": role_orientation.value,
                },
                probe_meta_for_compat=probe_rarity_meta,
                reference_meta_for_compat=reference_rarity_meta,
            )

        if mode == ArchitectureParameterMode.MANUAL:
            state = LRArchitectureStateFactory.create(
                spec=spec,
                s0=instr.manual_s0 if spec.requires_s0 else None,
                r0_probe=(
                    instr.manual_r0_probe
                    if (spec.centering_mode.value == "MEAN" and spec.requires_typicality_probe)
                    else None
                ),
                r0_reference=(
                    instr.manual_r0_reference
                    if (spec.centering_mode.value == "MEAN" and spec.requires_typicality_reference)
                    else None
                ),
                base_score_kind=base_score_kind,
                pair_role_orientation=role_orientation,
                pair_key_mode=pair_key_mode,
                rarity_probe_score_type=probe_rarity_meta.get("score_type"),
                rarity_reference_score_type=reference_rarity_meta.get("score_type"),
                rarity_probe_transform_id=probe_rarity_meta.get("transform_id"),
                rarity_reference_transform_id=reference_rarity_meta.get("transform_id"),
                rarity_probe_combiner=probe_rarity_meta.get("combiner"),
                rarity_reference_combiner=reference_rarity_meta.get("combiner"),
                rarity_probe_model_resource_id=probe_rarity_meta.get("model_resource_id"),
                rarity_reference_model_resource_id=reference_rarity_meta.get("model_resource_id"),
                rarity_probe_model_fingerprint=probe_rarity_meta.get("model_fingerprint"),
                rarity_reference_model_fingerprint=reference_rarity_meta.get("model_fingerprint"),
                fit_scope=ArchitectureFitScope.MANUAL,
                s0_resolution_method=(
                    S0ResolutionMethod.MANUAL if spec.requires_s0
                    else S0ResolutionMethod.NOT_APPLICABLE
                ),
                r0_probe_estimator=None,
                r0_reference_estimator=None,
            )
            return state, None

        # FIT_FULL_CALIBRATION_LEGACY (v1) e FIT_PER_FOLD (v2/PR 3).
        # Ambos os modos exercem o mesmo cálculo provisório de s₀/r̄ aqui
        # (sobre todo D_cal). A diferença é o fit_scope persistido no
        # state e a flag `defer_to_calibrator_per_fold` no metadata do
        # vetor produzido — o CALIBRATION_MODEL_TRAINING consome essa
        # flag para trocar para o Pipeline+GridSearchCV com transformer,
        # produzindo um state final ajustado por fold (PR 3).
        if mode in (
            ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY,
            ArchitectureParameterMode.FIT_PER_FOLD,
        ):
            return self._resolve_fit_full_legacy(
                instr=instr, spec=spec, phase=phase,
                base_scores=base_scores, same_person=same_person,
                rarity_probe_raw=rarity_probe_raw,
                rarity_reference_raw=rarity_reference_raw,
                base_score_kind=base_score_kind,
                role_orientation=role_orientation,
                pair_key_mode=pair_key_mode,
                probe_meta=probe_rarity_meta,
                reference_meta=reference_rarity_meta,
                fitted_on_resource_id=instr.base_score_resource_id,
                pair_keys=pair_keys,
            )

        raise ValueError(f"parameter_mode não suportado: {mode}")

    def _resolve_inherit(
        self, *,
        instr: LRArchitectureVectorizationInstruction,
        spec,
        context: StrategyExecutionContextProtocol,
        base_meta_for_compat: Dict[str, Any],
        probe_meta_for_compat: Dict[str, Any],
        reference_meta_for_compat: Dict[str, Any],
    ) -> Tuple[LRArchitectureState, Optional[Dict[str, Any]]]:
        rid = instr.inherit_architecture_state_from_resource_id
        context.ensure_payload(rid)
        res = context.get_resource(rid)
        if res is None or res.payload is None:
            raise ValueError(
                f"Recurso de herança ausente: {rid}"
            )
        meta = getattr(res.payload, "metadata", {}) or {}
        state_dict = meta.get("architecture_state")
        if not state_dict:
            raise ValueError(
                f"Recurso {rid} não tem metadata.architecture_state — "
                "INHERIT precisa de uma execução anterior de "
                "LR_ARCHITECTURE_VECTORIZATION"
            )
        state = LRArchitectureStateFactory.from_dict(state_dict)

        issues = validate_state_compatibility(
            state, spec,
            incoming_base_meta=base_meta_for_compat,
            incoming_rarity_probe_meta=probe_meta_for_compat or None,
            incoming_rarity_reference_meta=reference_meta_for_compat or None,
        )
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            joined = "; ".join(f"{i.field}: {i.message}" for i in errors)
            raise ValueError(
                f"State herdado incompatível com contexto atual: {joined}"
            )
        return state, None

    def _resolve_fit_full_legacy(
        self, *,
        instr: LRArchitectureVectorizationInstruction,
        spec, phase: StrategyPhase,
        base_scores: np.ndarray,
        same_person: Optional[np.ndarray],
        rarity_probe_raw: Optional[np.ndarray],
        rarity_reference_raw: Optional[np.ndarray],
        base_score_kind: str,
        role_orientation: PairRoleOrientation,
        pair_key_mode: PairKeyMode,
        probe_meta: Dict[str, Any],
        reference_meta: Dict[str, Any],
        fitted_on_resource_id: Optional[str],
        pair_keys: Tuple[Tuple[str, str], ...],
    ) -> Tuple[LRArchitectureState, Optional[Dict[str, Any]]]:
        # s0
        s0_value = None
        s0_summary_dict: Optional[Dict[str, Any]] = None
        s0_method = S0ResolutionMethod.NOT_APPLICABLE
        if spec.requires_s0:
            if same_person is None:
                raise ValueError(
                    "FIT_FULL_CALIBRATION_LEGACY exige same_person nos items do "
                    "base; nenhum par tem rótulo"
                )
            cfg = instr.s0_logreg_config_override or {}
            summary = resolve_s0_auto_logreg(
                base_scores,
                same_person.astype(int),
                C=float(cfg.get("C", 1.0)),
                solver=str(cfg.get("solver", "liblinear")),
                class_weight=cfg.get("class_weight", "balanced"),
                max_iter=int(cfg.get("max_iter", 1000)),
                coef_tolerance=float(cfg.get("coef_tolerance", 1e-8)),
            )
            s0_value = summary.s0_value
            s0_summary_dict = summary.to_dict_hex()
            s0_method = S0ResolutionMethod.AUTO_LOGREG

        # r0 (apenas se centering=MEAN)
        r0_probe = None
        r0_reference = None
        r0_probe_estimator = None
        r0_reference_estimator = None
        if spec.centering_mode.value == "MEAN":
            if spec.requires_typicality_probe and rarity_probe_raw is not None:
                r0_probe = float(rarity_probe_raw.mean())
                r0_probe_estimator = "MEAN"
            if spec.requires_typicality_reference and rarity_reference_raw is not None:
                r0_reference = float(rarity_reference_raw.mean())
                r0_reference_estimator = "MEAN"

        n_gen = int(same_person.sum()) if same_person is not None else None
        n_imp = int((~same_person).sum()) if same_person is not None else None

        state = LRArchitectureStateFactory.create(
            spec=spec,
            s0=s0_value,
            r0_probe=r0_probe,
            r0_reference=r0_reference,
            base_score_kind=base_score_kind,
            pair_role_orientation=role_orientation,
            pair_key_mode=pair_key_mode,
            rarity_probe_score_type=probe_meta.get("score_type"),
            rarity_reference_score_type=reference_meta.get("score_type"),
            rarity_probe_transform_id=probe_meta.get("transform_id"),
            rarity_reference_transform_id=reference_meta.get("transform_id"),
            rarity_probe_combiner=probe_meta.get("combiner"),
            rarity_reference_combiner=reference_meta.get("combiner"),
            rarity_probe_model_resource_id=probe_meta.get("model_resource_id"),
            rarity_reference_model_resource_id=reference_meta.get("model_resource_id"),
            rarity_probe_model_fingerprint=probe_meta.get("model_fingerprint"),
            rarity_reference_model_fingerprint=reference_meta.get("model_fingerprint"),
            fit_scope=(
                ArchitectureFitScope.PER_FOLD_REFIT_FULL_CALIBRATION
                if instr.parameter_mode == ArchitectureParameterMode.FIT_PER_FOLD
                else ArchitectureFitScope.FULL_CALIBRATION_LEGACY
            ),
            s0_resolution_method=s0_method,
            r0_probe_estimator=r0_probe_estimator,
            r0_reference_estimator=r0_reference_estimator,
            fitted_on_resource_id=fitted_on_resource_id,
            n_fitted_pairs=len(base_scores),
            n_fitted_genuine_pairs=n_gen,
            n_fitted_impostor_pairs=n_imp,
            fitted_dataset_pair_key_order_hash=_compute_pair_key_order_hash(pair_keys),
            s0_logreg_config=instr.s0_logreg_config_override,
            s0_fit_summary=s0_summary_dict,
        )
        # PR 3: em FIT_PER_FOLD, sinaliza ao CALIBRATION_MODEL_TRAINING
        # downstream que deve assumir o ajuste por fold via Pipeline+
        # GridSearchCV em vez de aceitar a matriz pré-vetorizada.
        # O `vec_dataset_metadata_extra` é mesclado no metadata do
        # SimilarityScoreDataset vetorizado emitido pelo executor.
        extra: Optional[Dict[str, Any]] = None
        if instr.parameter_mode == ArchitectureParameterMode.FIT_PER_FOLD:
            extra = {"defer_to_calibrator_per_fold": True}
        return state, extra

    # ─── Orientation/key mode derivation ─────────────────────────────────

    def _determine_pair_role_orientation(
        self, spec, base_meta: Dict[str, Any],
    ) -> PairRoleOrientation:
        """
        Política do plano §3 (4 ajuste obrigatório):
          - Orientação explícita no metadata → usa.
          - Arquitetura simétrica → permite NOT_APPLICABLE como default.
          - Ambiguidade + arquitetura ordenada → erro.
        """
        explicit = (
            base_meta.get("pair_role_orientation")
            or base_meta.get("pair_orientation")
        )
        if explicit:
            try:
                return PairRoleOrientation.from_str(explicit)
            except Exception:
                pass
        # Não há orientação explícita
        if spec.requires_ordered_pair_roles:
            raise ValueError(
                f"Arquitetura {spec.architecture.value} exige orientação "
                "PROBE_REFERENCE explícita no metadata do recurso base, mas "
                "nenhuma foi declarada. SIMILARITY_SCORING (cosseno/PLDA) "
                "deve persistir pair_role_orientation."
            )
        return PairRoleOrientation.NOT_APPLICABLE

    def _determine_pair_key_mode(
        self, base_meta: Dict[str, Any],
    ) -> PairKeyMode:
        explicit = base_meta.get("pair_key_mode")
        if explicit:
            try:
                return PairKeyMode.from_str(explicit)
            except Exception:
                pass
        return PairKeyMode.ORDERED

    # ─── Fingerprints ────────────────────────────────────────────────────

    @staticmethod
    def _score_vector_numeric_fingerprint(
        matrix: np.ndarray, pair_key_order_hash: str,
    ) -> str:
        h = hashlib.sha256()
        h.update(str(matrix.dtype.str).encode("utf-8"))
        h.update(b"\n")
        h.update(",".join(str(d) for d in matrix.shape).encode("utf-8"))
        h.update(b"\n")
        h.update(np.ascontiguousarray(matrix, dtype="<f8").tobytes(order="C"))
        h.update(b"\n")
        h.update(pair_key_order_hash.encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _score_vector_contract_fingerprint(spec, numeric_fp: str) -> str:
        payload = {
            "architecture_spec_fingerprint": spec.fingerprint(),
            "feature_terms": [t.value for t in spec.feature_terms],
            "feature_names": list(spec.feature_names),
            "centering_mode": spec.centering_mode.value,
            "numeric_fingerprint": numeric_fp,
        }
        from faceaudit.models.lr._canonical_hash import hash_canonical_payload
        return hash_canonical_payload(payload)

    @staticmethod
    def _build_raw_input_schema(spec) -> LRArchitectureRawFeatureSchema:
        if not spec.requires_typicality_probe and not spec.requires_typicality_reference:
            return schema_for_base_only()
        return schema_for_base_and_rarities()

    @staticmethod
    def _stack_raw_matrix(
        base: np.ndarray,
        rarity_probe: Optional[np.ndarray],
        rarity_reference: Optional[np.ndarray],
        schema: LRArchitectureRawFeatureSchema,
    ) -> np.ndarray:
        n = base.shape[0]
        if schema.rarity_probe_column is None and schema.rarity_reference_column is None:
            return base.reshape(-1, 1).astype(np.float64)
        # Layout fixo: [base, rarity_probe, rarity_reference]
        cols = [base.astype(np.float64)]
        if rarity_probe is not None:
            cols.append(rarity_probe.astype(np.float64))
        elif schema.rarity_probe_column is not None:
            cols.append(np.zeros(n, dtype=np.float64))
        if rarity_reference is not None:
            cols.append(rarity_reference.astype(np.float64))
        elif schema.rarity_reference_column is not None:
            cols.append(np.zeros(n, dtype=np.float64))
        return np.column_stack(cols).astype(np.float64)
