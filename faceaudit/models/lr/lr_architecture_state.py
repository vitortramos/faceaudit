# faceaudit/models/lr/lr_architecture_state.py
"""
`LRArchitectureState`: estado frozen com parâmetros ajustados + 2
fingerprints + proveniência completa.

Construção somente via `LRArchitectureStateFactory` — nunca direta.
Permite que o construtor da dataclass aceite os fingerprints como
strings (frozen dataclasses não permitem propriedades calculadas em
__init__), enquanto a factory cuida da consistência (fingerprint do
state nunca inclui o próprio fingerprint).

Dois fingerprints separados:

- `vectorization_contract_fingerprint`: cobre tudo que altera a matriz
  produzida + a semântica + a proveniência dos scores. É o que vincula
  vetorização ao calibrador. Mismatch entre dataset.fp e calibrator.fp na
  aplicação → erro.

- `provenance_record_fingerprint`: cobre IDs de recursos, contagens, fit
  summary, etc. Pode mudar entre runs com matemática idêntica. Apenas
  documenta proveniência.

`from_dict` recalcula ambos os fingerprints e compara com os persistidos
— mismatch levanta `StateIntegrityError` (adulteração do JSON detectada).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from faceaudit.enums.architecture_fit_scope import ArchitectureFitScope
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.lr_feature_term import LRFeatureTerm
from faceaudit.enums.pair_key_mode import PairKeyMode
from faceaudit.enums.pair_role_orientation import PairRoleOrientation
from faceaudit.enums.rarity_centering_mode import RarityCenteringMode
from faceaudit.enums.s0_resolution_method import S0ResolutionMethod
from faceaudit.models.lr._canonical_hash import hash_canonical_payload, hexify_float


class StateIntegrityError(Exception):
    """
    Levantada quando `LRArchitectureStateFactory.from_dict()` detecta
    que os fingerprints recalculados não batem com os persistidos.

    Indica adulteração manual do JSON, corrupção, ou bug de
    serialização. NUNCA deve ser silenciado — a aplicação posterior do
    state é insegura sem garantia de integridade.
    """


@dataclass(frozen=True)
class LRArchitectureState:
    """
    Estado completo de uma arquitetura de vetorização LR.

    Construir apenas via `LRArchitectureStateFactory.create(...)` ou
    `LRArchitectureStateFactory.from_dict(...)`. Não instanciar
    diretamente — a factory garante consistência dos fingerprints.
    """

    # ── Identidade da arquitetura ──
    architecture: LRCalibrationArchitecture
    formula_version: str
    architecture_spec_fingerprint: str

    # ── Parâmetros ajustados (None onde não aplicável) ──
    s0: Optional[float]
    r0_probe: Optional[float]
    r0_reference: Optional[float]

    # ── Estrutura derivada ──
    feature_terms: Tuple[LRFeatureTerm, ...]
    feature_names: Tuple[str, ...]
    output_dim: int
    centering_mode: RarityCenteringMode

    # ── Proveniência da base ──
    base_score_kind: str  # "cosine" | "plda" | etc. — string livre por compat
    pair_role_orientation: PairRoleOrientation
    pair_key_mode: PairKeyMode

    # ── Proveniência da tipicidade (None onde não aplicável) ──
    rarity_probe_score_type: Optional[str]
    rarity_reference_score_type: Optional[str]
    rarity_probe_transform_id: Optional[str]
    rarity_reference_transform_id: Optional[str]
    rarity_probe_transform_formula_version: Optional[str]
    rarity_reference_transform_formula_version: Optional[str]
    rarity_probe_transform_fingerprint: Optional[str]
    rarity_reference_transform_fingerprint: Optional[str]
    rarity_probe_combiner: Optional[str]
    rarity_reference_combiner: Optional[str]
    rarity_probe_model_resource_id: Optional[str]
    rarity_reference_model_resource_id: Optional[str]
    rarity_probe_model_fingerprint: Optional[str]
    rarity_reference_model_fingerprint: Optional[str]

    # ── Detalhes do ajuste ──
    fit_scope: ArchitectureFitScope
    s0_resolution_method: S0ResolutionMethod
    r0_probe_estimator: Optional[str]      # "MEAN" ou None
    r0_reference_estimator: Optional[str]

    # ── Proveniência do ajuste ──
    fitted_on_resource_id: Optional[str]
    fitted_on_resource_fingerprint: Optional[str]
    n_fitted_pairs: Optional[int]
    n_fitted_genuine_pairs: Optional[int]
    n_fitted_impostor_pairs: Optional[int]
    fitted_dataset_pair_key_order_hash: Optional[str]
    s0_logreg_config: Optional[Dict[str, Any]] = field(default=None)
    s0_fit_summary: Optional[Dict[str, Any]] = field(default=None)

    # ── ONE_FACE (V2.14.1) — None em arquiteturas non-ONE_FACE ──
    r0_single: Optional[float] = field(default=None)
    r0_single_estimator: Optional[str] = field(default=None)
    rarity_single_score_type: Optional[str] = field(default=None)
    rarity_single_transform_id: Optional[str] = field(default=None)
    rarity_single_transform_formula_version: Optional[str] = field(default=None)
    rarity_single_transform_fingerprint: Optional[str] = field(default=None)
    rarity_single_combiner: Optional[str] = field(default=None)
    rarity_single_model_resource_id: Optional[str] = field(default=None)
    rarity_single_model_fingerprint: Optional[str] = field(default=None)

    # ── Fingerprints (preenchidos pela factory) ──
    vectorization_contract_fingerprint: str = ""
    provenance_record_fingerprint: str = ""

    @property
    def has_internal_cv_parameter_leakage_risk(self) -> bool:
        """
        True quando o state foi ajustado no modo legado (todo D_cal antes da
        CV interna) E a arquitetura usa parâmetros ajustáveis. COSINE_ONLY
        com fit_scope=NOT_APPLICABLE retorna False.
        """
        architecture_uses_fitted_params = (
            self.s0 is not None
            or self.r0_probe is not None
            or self.r0_reference is not None
            or self.r0_single is not None
        )
        return (
            self.fit_scope == ArchitectureFitScope.FULL_CALIBRATION_LEGACY
            and architecture_uses_fitted_params
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialização canônica para persistência em JSON."""
        return {
            "architecture": self.architecture.value,
            "formula_version": self.formula_version,
            "architecture_spec_fingerprint": self.architecture_spec_fingerprint,
            "s0": self.s0,
            "r0_probe": self.r0_probe,
            "r0_reference": self.r0_reference,
            "r0_single": self.r0_single,
            "feature_terms": [t.value for t in self.feature_terms],
            "feature_names": list(self.feature_names),
            "output_dim": self.output_dim,
            "centering_mode": self.centering_mode.value,
            "base_score_kind": self.base_score_kind,
            "pair_role_orientation": self.pair_role_orientation.value,
            "pair_key_mode": self.pair_key_mode.value,
            "rarity_probe_score_type": self.rarity_probe_score_type,
            "rarity_reference_score_type": self.rarity_reference_score_type,
            "rarity_single_score_type": self.rarity_single_score_type,
            "rarity_probe_transform_id": self.rarity_probe_transform_id,
            "rarity_reference_transform_id": self.rarity_reference_transform_id,
            "rarity_single_transform_id": self.rarity_single_transform_id,
            "rarity_probe_transform_formula_version": self.rarity_probe_transform_formula_version,
            "rarity_reference_transform_formula_version": self.rarity_reference_transform_formula_version,
            "rarity_single_transform_formula_version": self.rarity_single_transform_formula_version,
            "rarity_probe_transform_fingerprint": self.rarity_probe_transform_fingerprint,
            "rarity_reference_transform_fingerprint": self.rarity_reference_transform_fingerprint,
            "rarity_single_transform_fingerprint": self.rarity_single_transform_fingerprint,
            "rarity_probe_combiner": self.rarity_probe_combiner,
            "rarity_reference_combiner": self.rarity_reference_combiner,
            "rarity_single_combiner": self.rarity_single_combiner,
            "rarity_probe_model_resource_id": self.rarity_probe_model_resource_id,
            "rarity_reference_model_resource_id": self.rarity_reference_model_resource_id,
            "rarity_single_model_resource_id": self.rarity_single_model_resource_id,
            "rarity_probe_model_fingerprint": self.rarity_probe_model_fingerprint,
            "rarity_reference_model_fingerprint": self.rarity_reference_model_fingerprint,
            "rarity_single_model_fingerprint": self.rarity_single_model_fingerprint,
            "fit_scope": self.fit_scope.value,
            "s0_resolution_method": self.s0_resolution_method.value,
            "r0_probe_estimator": self.r0_probe_estimator,
            "r0_reference_estimator": self.r0_reference_estimator,
            "r0_single_estimator": self.r0_single_estimator,
            "fitted_on_resource_id": self.fitted_on_resource_id,
            "fitted_on_resource_fingerprint": self.fitted_on_resource_fingerprint,
            "n_fitted_pairs": self.n_fitted_pairs,
            "n_fitted_genuine_pairs": self.n_fitted_genuine_pairs,
            "n_fitted_impostor_pairs": self.n_fitted_impostor_pairs,
            "fitted_dataset_pair_key_order_hash": self.fitted_dataset_pair_key_order_hash,
            "s0_logreg_config": self.s0_logreg_config,
            "s0_fit_summary": self.s0_fit_summary,
            "vectorization_contract_fingerprint": self.vectorization_contract_fingerprint,
            "provenance_record_fingerprint": self.provenance_record_fingerprint,
        }


# ─────────────────────────────────────────────────────────────────────
# Helpers de canonicalização de dicts voláteis (para hash de provenance)
# ─────────────────────────────────────────────────────────────────────


def _normalize_dict_recursive(d: Any) -> Any:
    """
    Normaliza recursivamente um dict para uso em hash canônico.

    - Floats viram `float.hex(value)` (bit-exact).
    - Chaves stringificadas e ordenadas.
    - None e bool preservados.
    - Listas/tuples normalizadas pelo conteúdo.
    """
    if d is None or isinstance(d, bool):
        return d
    if isinstance(d, float):
        return d.hex()
    if isinstance(d, (int, str)):
        return d
    if isinstance(d, dict):
        return {str(k): _normalize_dict_recursive(v) for k, v in sorted(d.items())}
    if isinstance(d, (list, tuple)):
        return [_normalize_dict_recursive(v) for v in d]
    # Fallback — tenta converter via repr (último recurso)
    return repr(d)


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────


class LRArchitectureStateFactory:
    """
    Única forma de construir `LRArchitectureState`. Garante consistência
    de fingerprints e validações estruturais com o spec.

    `create(...)` recebe spec + parâmetros + proveniência e calcula os
    dois fingerprints. `from_dict(...)` desserializa, recalcula os
    fingerprints e compara com os persistidos — `StateIntegrityError`
    em mismatch.
    """

    @staticmethod
    def _compute_contract_fingerprint(state_partial: Dict[str, Any]) -> str:
        """
        Calcula `vectorization_contract_fingerprint` — cobre tudo que
        altera a matriz, semântica e proveniência dos scores.
        Explicitamente NÃO inclui IDs de recurso, contagens, fit_summary.

        ONE_FACE (V2.14.1): inclui r0_single + campos rarity_single_* na
        proveniência. Para arquiteturas não-ONE_FACE, todos esses ficam
        None — mantém compat com state legado (None é serializado
        consistentemente).
        """
        payload = {
            "architecture": state_partial["architecture"],
            "formula_version": state_partial["formula_version"],
            "architecture_spec_fingerprint": state_partial["architecture_spec_fingerprint"],
            "feature_terms": state_partial["feature_terms"],
            "centering_mode": state_partial["centering_mode"],
            "s0_hex": hexify_float(state_partial["s0"]),
            "r0_probe_hex": hexify_float(state_partial["r0_probe"]),
            "r0_reference_hex": hexify_float(state_partial["r0_reference"]),
            "r0_single_hex": hexify_float(state_partial.get("r0_single")),
            "base_score_kind": state_partial["base_score_kind"],
            "pair_role_orientation": state_partial["pair_role_orientation"],
            # Tipicidade — só campos contratuais (não inclui resource_id)
            "rarity_probe_score_type": state_partial["rarity_probe_score_type"],
            "rarity_reference_score_type": state_partial["rarity_reference_score_type"],
            "rarity_single_score_type": state_partial.get("rarity_single_score_type"),
            "rarity_probe_transform_id": state_partial["rarity_probe_transform_id"],
            "rarity_reference_transform_id": state_partial["rarity_reference_transform_id"],
            "rarity_single_transform_id": state_partial.get("rarity_single_transform_id"),
            "rarity_probe_transform_formula_version": state_partial["rarity_probe_transform_formula_version"],
            "rarity_reference_transform_formula_version": state_partial["rarity_reference_transform_formula_version"],
            "rarity_single_transform_formula_version": state_partial.get("rarity_single_transform_formula_version"),
            "rarity_probe_transform_fingerprint": state_partial["rarity_probe_transform_fingerprint"],
            "rarity_reference_transform_fingerprint": state_partial["rarity_reference_transform_fingerprint"],
            "rarity_single_transform_fingerprint": state_partial.get("rarity_single_transform_fingerprint"),
            "rarity_probe_combiner": state_partial["rarity_probe_combiner"],
            "rarity_reference_combiner": state_partial["rarity_reference_combiner"],
            "rarity_single_combiner": state_partial.get("rarity_single_combiner"),
            "rarity_probe_model_fingerprint": state_partial["rarity_probe_model_fingerprint"],
            "rarity_reference_model_fingerprint": state_partial["rarity_reference_model_fingerprint"],
            "rarity_single_model_fingerprint": state_partial.get("rarity_single_model_fingerprint"),
        }
        return hash_canonical_payload(payload)

    @staticmethod
    def _compute_provenance_fingerprint(state_partial: Dict[str, Any]) -> str:
        """Cobre IDs, contagens, fit_summary, hashes de dataset de treino."""
        payload = {
            "fit_scope": state_partial["fit_scope"],
            "s0_resolution_method": state_partial["s0_resolution_method"],
            "r0_probe_estimator": state_partial["r0_probe_estimator"],
            "r0_reference_estimator": state_partial["r0_reference_estimator"],
            "r0_single_estimator": state_partial.get("r0_single_estimator"),
            "fitted_on_resource_id": state_partial["fitted_on_resource_id"],
            "fitted_on_resource_fingerprint": state_partial["fitted_on_resource_fingerprint"],
            "n_fitted_pairs": state_partial["n_fitted_pairs"],
            "n_fitted_genuine_pairs": state_partial["n_fitted_genuine_pairs"],
            "n_fitted_impostor_pairs": state_partial["n_fitted_impostor_pairs"],
            "fitted_dataset_pair_key_order_hash": state_partial["fitted_dataset_pair_key_order_hash"],
            "rarity_probe_model_resource_id": state_partial["rarity_probe_model_resource_id"],
            "rarity_reference_model_resource_id": state_partial["rarity_reference_model_resource_id"],
            "rarity_single_model_resource_id": state_partial.get("rarity_single_model_resource_id"),
            "s0_logreg_config": _normalize_dict_recursive(state_partial["s0_logreg_config"]),
            "s0_fit_summary": _normalize_dict_recursive(state_partial["s0_fit_summary"]),
        }
        return hash_canonical_payload(payload)

    @classmethod
    def create(
        cls,
        *,
        spec,  # LRArchitectureSpec (import indireto para evitar ciclo)
        s0: Optional[float] = None,
        r0_probe: Optional[float] = None,
        r0_reference: Optional[float] = None,
        r0_single: Optional[float] = None,
        base_score_kind: str = "cosine",
        pair_role_orientation: PairRoleOrientation = PairRoleOrientation.NOT_APPLICABLE,
        pair_key_mode: PairKeyMode = PairKeyMode.ORDERED,
        rarity_probe_score_type: Optional[str] = None,
        rarity_reference_score_type: Optional[str] = None,
        rarity_single_score_type: Optional[str] = None,
        rarity_probe_transform_id: Optional[str] = None,
        rarity_reference_transform_id: Optional[str] = None,
        rarity_single_transform_id: Optional[str] = None,
        rarity_probe_transform_formula_version: Optional[str] = None,
        rarity_reference_transform_formula_version: Optional[str] = None,
        rarity_single_transform_formula_version: Optional[str] = None,
        rarity_probe_transform_fingerprint: Optional[str] = None,
        rarity_reference_transform_fingerprint: Optional[str] = None,
        rarity_single_transform_fingerprint: Optional[str] = None,
        rarity_probe_combiner: Optional[str] = None,
        rarity_reference_combiner: Optional[str] = None,
        rarity_single_combiner: Optional[str] = None,
        rarity_probe_model_resource_id: Optional[str] = None,
        rarity_reference_model_resource_id: Optional[str] = None,
        rarity_single_model_resource_id: Optional[str] = None,
        rarity_probe_model_fingerprint: Optional[str] = None,
        rarity_reference_model_fingerprint: Optional[str] = None,
        rarity_single_model_fingerprint: Optional[str] = None,
        fit_scope: ArchitectureFitScope = ArchitectureFitScope.NOT_APPLICABLE,
        s0_resolution_method: S0ResolutionMethod = S0ResolutionMethod.NOT_APPLICABLE,
        r0_probe_estimator: Optional[str] = None,
        r0_reference_estimator: Optional[str] = None,
        r0_single_estimator: Optional[str] = None,
        fitted_on_resource_id: Optional[str] = None,
        fitted_on_resource_fingerprint: Optional[str] = None,
        n_fitted_pairs: Optional[int] = None,
        n_fitted_genuine_pairs: Optional[int] = None,
        n_fitted_impostor_pairs: Optional[int] = None,
        fitted_dataset_pair_key_order_hash: Optional[str] = None,
        s0_logreg_config: Optional[Dict[str, Any]] = None,
        s0_fit_summary: Optional[Dict[str, Any]] = None,
    ) -> LRArchitectureState:
        """
        Constrói um state validando coerência com o spec.

        Falha com `ValueError`:
        - Centering mode == NONE mas r0_* populado, ou vice-versa.
        - Arquitetura exige s₀ mas s₀ é None (a menos que fit_scope=NOT_APPLICABLE,
          permite construir state vazio antes do fit).
        - pair_role_orientation = NOT_APPLICABLE em arquitetura ordenada,
          ou vice-versa.
        - ONE_FACE: r0_single + family flag inconsistentes.
        """
        cls._validate_against_spec(
            spec, s0, r0_probe, r0_reference, r0_single,
            pair_role_orientation, fit_scope,
        )

        state_partial = {
            "architecture": spec.architecture.value,
            "formula_version": spec.formula_version,
            "architecture_spec_fingerprint": spec.fingerprint(),
            "s0": s0,
            "r0_probe": r0_probe,
            "r0_reference": r0_reference,
            "r0_single": r0_single,
            "feature_terms": [t.value for t in spec.feature_terms],
            "centering_mode": spec.centering_mode.value,
            "base_score_kind": base_score_kind,
            "pair_role_orientation": pair_role_orientation.value,
            "pair_key_mode": pair_key_mode.value,
            "rarity_probe_score_type": rarity_probe_score_type,
            "rarity_reference_score_type": rarity_reference_score_type,
            "rarity_single_score_type": rarity_single_score_type,
            "rarity_probe_transform_id": rarity_probe_transform_id,
            "rarity_reference_transform_id": rarity_reference_transform_id,
            "rarity_single_transform_id": rarity_single_transform_id,
            "rarity_probe_transform_formula_version": rarity_probe_transform_formula_version,
            "rarity_reference_transform_formula_version": rarity_reference_transform_formula_version,
            "rarity_single_transform_formula_version": rarity_single_transform_formula_version,
            "rarity_probe_transform_fingerprint": rarity_probe_transform_fingerprint,
            "rarity_reference_transform_fingerprint": rarity_reference_transform_fingerprint,
            "rarity_single_transform_fingerprint": rarity_single_transform_fingerprint,
            "rarity_probe_combiner": rarity_probe_combiner,
            "rarity_reference_combiner": rarity_reference_combiner,
            "rarity_single_combiner": rarity_single_combiner,
            "rarity_probe_model_resource_id": rarity_probe_model_resource_id,
            "rarity_reference_model_resource_id": rarity_reference_model_resource_id,
            "rarity_single_model_resource_id": rarity_single_model_resource_id,
            "rarity_probe_model_fingerprint": rarity_probe_model_fingerprint,
            "rarity_reference_model_fingerprint": rarity_reference_model_fingerprint,
            "rarity_single_model_fingerprint": rarity_single_model_fingerprint,
            "fit_scope": fit_scope.value,
            "s0_resolution_method": s0_resolution_method.value,
            "r0_probe_estimator": r0_probe_estimator,
            "r0_reference_estimator": r0_reference_estimator,
            "r0_single_estimator": r0_single_estimator,
            "fitted_on_resource_id": fitted_on_resource_id,
            "fitted_on_resource_fingerprint": fitted_on_resource_fingerprint,
            "n_fitted_pairs": n_fitted_pairs,
            "n_fitted_genuine_pairs": n_fitted_genuine_pairs,
            "n_fitted_impostor_pairs": n_fitted_impostor_pairs,
            "fitted_dataset_pair_key_order_hash": fitted_dataset_pair_key_order_hash,
            "s0_logreg_config": s0_logreg_config,
            "s0_fit_summary": s0_fit_summary,
        }

        contract_fp = cls._compute_contract_fingerprint(state_partial)
        provenance_fp = cls._compute_provenance_fingerprint(state_partial)

        return LRArchitectureState(
            architecture=spec.architecture,
            formula_version=spec.formula_version,
            architecture_spec_fingerprint=spec.fingerprint(),
            s0=s0,
            r0_probe=r0_probe,
            r0_reference=r0_reference,
            feature_terms=spec.feature_terms,
            feature_names=spec.feature_names,
            output_dim=spec.output_dim,
            centering_mode=spec.centering_mode,
            base_score_kind=base_score_kind,
            pair_role_orientation=pair_role_orientation,
            pair_key_mode=pair_key_mode,
            rarity_probe_score_type=rarity_probe_score_type,
            rarity_reference_score_type=rarity_reference_score_type,
            rarity_probe_transform_id=rarity_probe_transform_id,
            rarity_reference_transform_id=rarity_reference_transform_id,
            rarity_probe_transform_formula_version=rarity_probe_transform_formula_version,
            rarity_reference_transform_formula_version=rarity_reference_transform_formula_version,
            rarity_probe_transform_fingerprint=rarity_probe_transform_fingerprint,
            rarity_reference_transform_fingerprint=rarity_reference_transform_fingerprint,
            rarity_probe_combiner=rarity_probe_combiner,
            rarity_reference_combiner=rarity_reference_combiner,
            rarity_probe_model_resource_id=rarity_probe_model_resource_id,
            rarity_reference_model_resource_id=rarity_reference_model_resource_id,
            rarity_probe_model_fingerprint=rarity_probe_model_fingerprint,
            rarity_reference_model_fingerprint=rarity_reference_model_fingerprint,
            fit_scope=fit_scope,
            s0_resolution_method=s0_resolution_method,
            r0_probe_estimator=r0_probe_estimator,
            r0_reference_estimator=r0_reference_estimator,
            fitted_on_resource_id=fitted_on_resource_id,
            fitted_on_resource_fingerprint=fitted_on_resource_fingerprint,
            n_fitted_pairs=n_fitted_pairs,
            n_fitted_genuine_pairs=n_fitted_genuine_pairs,
            n_fitted_impostor_pairs=n_fitted_impostor_pairs,
            fitted_dataset_pair_key_order_hash=fitted_dataset_pair_key_order_hash,
            s0_logreg_config=s0_logreg_config,
            s0_fit_summary=s0_fit_summary,
            # ONE_FACE (V2.14.1)
            r0_single=r0_single,
            r0_single_estimator=r0_single_estimator,
            rarity_single_score_type=rarity_single_score_type,
            rarity_single_transform_id=rarity_single_transform_id,
            rarity_single_transform_formula_version=rarity_single_transform_formula_version,
            rarity_single_transform_fingerprint=rarity_single_transform_fingerprint,
            rarity_single_combiner=rarity_single_combiner,
            rarity_single_model_resource_id=rarity_single_model_resource_id,
            rarity_single_model_fingerprint=rarity_single_model_fingerprint,
            vectorization_contract_fingerprint=contract_fp,
            provenance_record_fingerprint=provenance_fp,
        )

    @staticmethod
    def _validate_against_spec(
        spec,
        s0: Optional[float],
        r0_probe: Optional[float],
        r0_reference: Optional[float],
        r0_single: Optional[float],
        pair_role_orientation: PairRoleOrientation,
        fit_scope: ArchitectureFitScope,
    ) -> None:
        requires_single = getattr(spec, "requires_typicality_single", False)
        if spec.centering_mode == RarityCenteringMode.NONE:
            if r0_probe is not None or r0_reference is not None or r0_single is not None:
                raise ValueError(
                    f"Arquitetura {spec.architecture.value} usa centering=NONE; "
                    f"r0_probe/r0_reference/r0_single devem ser None"
                )
        elif spec.centering_mode == RarityCenteringMode.MEAN:
            # Quando há ajuste, r0_* devem estar presentes para os lados consumidos.
            if fit_scope != ArchitectureFitScope.NOT_APPLICABLE:
                if spec.requires_typicality_probe and r0_probe is None:
                    raise ValueError(
                        f"Arquitetura {spec.architecture.value} centra raridade e "
                        f"consome r_F; r0_probe é obrigatório quando fit_scope != NOT_APPLICABLE"
                    )
                if spec.requires_typicality_reference and r0_reference is None:
                    raise ValueError(
                        f"Arquitetura {spec.architecture.value} centra raridade e "
                        f"consome r_P; r0_reference é obrigatório quando fit_scope != NOT_APPLICABLE"
                    )
                if requires_single and r0_single is None:
                    raise ValueError(
                        f"Arquitetura {spec.architecture.value} centra raridade e "
                        f"consome r (single); r0_single é obrigatório quando "
                        f"fit_scope != NOT_APPLICABLE"
                    )

        # ONE_FACE: rejeita combinações cruzadas
        if requires_single:
            if r0_probe is not None or r0_reference is not None:
                raise ValueError(
                    f"Arquitetura ONE_FACE {spec.architecture.value} não aceita "
                    f"r0_probe ou r0_reference; use r0_single"
                )
        else:
            if r0_single is not None:
                raise ValueError(
                    f"Arquitetura {spec.architecture.value} não é ONE_FACE; "
                    f"r0_single deve ser None"
                )

        if spec.requires_s0 and s0 is None and fit_scope != ArchitectureFitScope.NOT_APPLICABLE:
            raise ValueError(
                f"Arquitetura {spec.architecture.value} requer s₀; s0 é obrigatório "
                f"quando fit_scope != NOT_APPLICABLE"
            )

        if not spec.requires_s0 and s0 is not None:
            raise ValueError(
                f"Arquitetura {spec.architecture.value} não usa s₀; "
                f"s0 deve ser None (recebido {s0})"
            )

        if spec.requires_ordered_pair_roles:
            if pair_role_orientation == PairRoleOrientation.NOT_APPLICABLE:
                raise ValueError(
                    f"Arquitetura {spec.architecture.value} distingue probe/reference; "
                    f"pair_role_orientation não pode ser NOT_APPLICABLE"
                )
        # Para arquiteturas simétricas (requires_ordered_pair_roles=False),
        # aceitamos qualquer pair_role_orientation — o spec não distingue, então
        # NOT_APPLICABLE ou PROBE_REFERENCE são ambos válidos.

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LRArchitectureState:
        """
        Desserializa um state e VERIFICA INTEGRIDADE recalculando ambos
        os fingerprints. `StateIntegrityError` em mismatch.

        Aceita state com fingerprints ausentes (`""` ou None) apenas
        quando ambos estão ausentes — emite warning e segue como state
        legado. Mismatch parcial sempre falha.
        """
        try:
            arch = LRCalibrationArchitecture.from_str(data["architecture"])
            feature_terms = tuple(LRFeatureTerm.from_str(t) for t in data["feature_terms"])
            centering = RarityCenteringMode.from_str(data["centering_mode"])
            role = PairRoleOrientation.from_str(data["pair_role_orientation"])
            key_mode = PairKeyMode.from_str(data["pair_key_mode"])
            fit_scope = ArchitectureFitScope.from_str(data["fit_scope"])
            s0_method = S0ResolutionMethod.from_str(data["s0_resolution_method"])
        except (KeyError, ValueError) as e:
            raise StateIntegrityError(f"Falha desserializando enums: {e}") from e

        state_partial = {
            "architecture": arch.value,
            "formula_version": data["formula_version"],
            "architecture_spec_fingerprint": data["architecture_spec_fingerprint"],
            "s0": data.get("s0"),
            "r0_probe": data.get("r0_probe"),
            "r0_reference": data.get("r0_reference"),
            "r0_single": data.get("r0_single"),
            "feature_terms": [t.value for t in feature_terms],
            "centering_mode": centering.value,
            "base_score_kind": data["base_score_kind"],
            "pair_role_orientation": role.value,
            "pair_key_mode": key_mode.value,
            "rarity_probe_score_type": data.get("rarity_probe_score_type"),
            "rarity_reference_score_type": data.get("rarity_reference_score_type"),
            "rarity_single_score_type": data.get("rarity_single_score_type"),
            "rarity_probe_transform_id": data.get("rarity_probe_transform_id"),
            "rarity_reference_transform_id": data.get("rarity_reference_transform_id"),
            "rarity_single_transform_id": data.get("rarity_single_transform_id"),
            "rarity_probe_transform_formula_version": data.get("rarity_probe_transform_formula_version"),
            "rarity_reference_transform_formula_version": data.get("rarity_reference_transform_formula_version"),
            "rarity_single_transform_formula_version": data.get("rarity_single_transform_formula_version"),
            "rarity_probe_transform_fingerprint": data.get("rarity_probe_transform_fingerprint"),
            "rarity_reference_transform_fingerprint": data.get("rarity_reference_transform_fingerprint"),
            "rarity_single_transform_fingerprint": data.get("rarity_single_transform_fingerprint"),
            "rarity_probe_combiner": data.get("rarity_probe_combiner"),
            "rarity_reference_combiner": data.get("rarity_reference_combiner"),
            "rarity_single_combiner": data.get("rarity_single_combiner"),
            "rarity_probe_model_resource_id": data.get("rarity_probe_model_resource_id"),
            "rarity_reference_model_resource_id": data.get("rarity_reference_model_resource_id"),
            "rarity_single_model_resource_id": data.get("rarity_single_model_resource_id"),
            "rarity_probe_model_fingerprint": data.get("rarity_probe_model_fingerprint"),
            "rarity_reference_model_fingerprint": data.get("rarity_reference_model_fingerprint"),
            "rarity_single_model_fingerprint": data.get("rarity_single_model_fingerprint"),
            "fit_scope": fit_scope.value,
            "s0_resolution_method": s0_method.value,
            "r0_probe_estimator": data.get("r0_probe_estimator"),
            "r0_reference_estimator": data.get("r0_reference_estimator"),
            "r0_single_estimator": data.get("r0_single_estimator"),
            "fitted_on_resource_id": data.get("fitted_on_resource_id"),
            "fitted_on_resource_fingerprint": data.get("fitted_on_resource_fingerprint"),
            "n_fitted_pairs": data.get("n_fitted_pairs"),
            "n_fitted_genuine_pairs": data.get("n_fitted_genuine_pairs"),
            "n_fitted_impostor_pairs": data.get("n_fitted_impostor_pairs"),
            "fitted_dataset_pair_key_order_hash": data.get("fitted_dataset_pair_key_order_hash"),
            "s0_logreg_config": data.get("s0_logreg_config"),
            "s0_fit_summary": data.get("s0_fit_summary"),
        }

        expected_contract_fp = cls._compute_contract_fingerprint(state_partial)
        expected_provenance_fp = cls._compute_provenance_fingerprint(state_partial)

        stored_contract_fp = data.get("vectorization_contract_fingerprint", "")
        stored_provenance_fp = data.get("provenance_record_fingerprint", "")

        if stored_contract_fp == "" and stored_provenance_fp == "":
            warnings.warn(
                "LRArchitectureState carregado SEM fingerprints — modo legado; "
                "integridade não pode ser verificada.",
                stacklevel=2,
            )
        else:
            if stored_contract_fp != expected_contract_fp:
                raise StateIntegrityError(
                    f"vectorization_contract_fingerprint não bate: "
                    f"persistido={stored_contract_fp[:16]}... "
                    f"recalculado={expected_contract_fp[:16]}..."
                )
            if stored_provenance_fp != expected_provenance_fp:
                raise StateIntegrityError(
                    f"provenance_record_fingerprint não bate: "
                    f"persistido={stored_provenance_fp[:16]}... "
                    f"recalculado={expected_provenance_fp[:16]}..."
                )

        return LRArchitectureState(
            architecture=arch,
            formula_version=data["formula_version"],
            architecture_spec_fingerprint=data["architecture_spec_fingerprint"],
            s0=data.get("s0"),
            r0_probe=data.get("r0_probe"),
            r0_reference=data.get("r0_reference"),
            feature_terms=feature_terms,
            feature_names=tuple(data["feature_names"]),
            output_dim=int(data["output_dim"]),
            centering_mode=centering,
            base_score_kind=data["base_score_kind"],
            pair_role_orientation=role,
            pair_key_mode=key_mode,
            rarity_probe_score_type=data.get("rarity_probe_score_type"),
            rarity_reference_score_type=data.get("rarity_reference_score_type"),
            rarity_probe_transform_id=data.get("rarity_probe_transform_id"),
            rarity_reference_transform_id=data.get("rarity_reference_transform_id"),
            rarity_probe_transform_formula_version=data.get("rarity_probe_transform_formula_version"),
            rarity_reference_transform_formula_version=data.get("rarity_reference_transform_formula_version"),
            rarity_probe_transform_fingerprint=data.get("rarity_probe_transform_fingerprint"),
            rarity_reference_transform_fingerprint=data.get("rarity_reference_transform_fingerprint"),
            rarity_probe_combiner=data.get("rarity_probe_combiner"),
            rarity_reference_combiner=data.get("rarity_reference_combiner"),
            rarity_probe_model_resource_id=data.get("rarity_probe_model_resource_id"),
            rarity_reference_model_resource_id=data.get("rarity_reference_model_resource_id"),
            rarity_probe_model_fingerprint=data.get("rarity_probe_model_fingerprint"),
            rarity_reference_model_fingerprint=data.get("rarity_reference_model_fingerprint"),
            fit_scope=fit_scope,
            s0_resolution_method=s0_method,
            r0_probe_estimator=data.get("r0_probe_estimator"),
            r0_reference_estimator=data.get("r0_reference_estimator"),
            fitted_on_resource_id=data.get("fitted_on_resource_id"),
            fitted_on_resource_fingerprint=data.get("fitted_on_resource_fingerprint"),
            n_fitted_pairs=data.get("n_fitted_pairs"),
            n_fitted_genuine_pairs=data.get("n_fitted_genuine_pairs"),
            n_fitted_impostor_pairs=data.get("n_fitted_impostor_pairs"),
            fitted_dataset_pair_key_order_hash=data.get("fitted_dataset_pair_key_order_hash"),
            s0_logreg_config=data.get("s0_logreg_config"),
            s0_fit_summary=data.get("s0_fit_summary"),
            # ONE_FACE (V2.14.1)
            r0_single=data.get("r0_single"),
            r0_single_estimator=data.get("r0_single_estimator"),
            rarity_single_score_type=data.get("rarity_single_score_type"),
            rarity_single_transform_id=data.get("rarity_single_transform_id"),
            rarity_single_transform_formula_version=data.get("rarity_single_transform_formula_version"),
            rarity_single_transform_fingerprint=data.get("rarity_single_transform_fingerprint"),
            rarity_single_combiner=data.get("rarity_single_combiner"),
            rarity_single_model_resource_id=data.get("rarity_single_model_resource_id"),
            rarity_single_model_fingerprint=data.get("rarity_single_model_fingerprint"),
            vectorization_contract_fingerprint=stored_contract_fp or expected_contract_fp,
            provenance_record_fingerprint=stored_provenance_fp or expected_provenance_fp,
        )
