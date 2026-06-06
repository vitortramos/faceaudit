# faceaudit/models/lr/lr_architecture_spec.py
"""
Spec declarativo das arquiteturas de calibração LR.

Fonte única de verdade. `LR_ARCHITECTURE_SPECS` mapeia cada membro de
`LRCalibrationArchitecture` para um `LRArchitectureSpec` que declara:

- `feature_terms`: tupla ordenada de `LRFeatureTerm`. A ordem é a ordem das
  colunas da matriz produzida.
- `centering_mode`: política de centragem (NONE ou MEAN).
- `requires_typicality_probe` / `requires_typicality_reference`: True se a
  arquitetura consome r_F e/ou r_P.
- `requires_s0`: True se algum termo precisa de s₀.
- `requires_ordered_pair_roles`: True quando a arquitetura distingue
  semanticamente probe vs reference. Falso em arquiteturas simétricas
  (COSINE_ONLY, HINGE_ONLY, TWO_SIDED_HINGE_ONLY).
- `formula_version`: bumpada quando a composição de termos da arquitetura
  muda. Entra no fingerprint do spec.

`spec.fingerprint()` é um SHA-256 estável dos campos acima + da
`formula_version` de cada termo registrado em `LR_FEATURE_TERM_SPECS` —
detecta drift de implementação dos termos individualmente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.lr_feature_term import LRFeatureTerm
from faceaudit.enums.rarity_centering_mode import RarityCenteringMode
from faceaudit.models.lr._canonical_hash import hash_canonical_payload
from faceaudit.models.lr.lr_feature_term_spec import LR_FEATURE_TERM_SPECS


@dataclass(frozen=True)
class LRArchitectureSpec:
    """Declaração estrutural de uma arquitetura de vetorização LR."""

    architecture: LRCalibrationArchitecture
    feature_terms: Tuple[LRFeatureTerm, ...]
    centering_mode: RarityCenteringMode
    requires_typicality_probe: bool
    requires_typicality_reference: bool
    requires_s0: bool
    requires_ordered_pair_roles: bool
    formula_version: str
    # V2.14.1 — família ONE_FACE
    requires_typicality_single: bool = False

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Identificadores estáveis das colunas (= value dos termos)."""
        return tuple(t.value for t in self.feature_terms)

    @property
    def output_dim(self) -> int:
        return len(self.feature_terms)

    @property
    def is_one_face(self) -> bool:
        """True para arquiteturas da família ONE_FACE (V2.14.1)."""
        return self.requires_typicality_single

    def fingerprint(self) -> str:
        """
        SHA-256 do spec — cobre composição, centering, requires e
        formula_version individual de cada termo. Bumpar a formula_version
        de QUALQUER termo usado por esta arquitetura muda o fingerprint
        — detecta drift de implementação dos termos.
        """
        term_versions = {
            term.value: LR_FEATURE_TERM_SPECS[term].formula_version
            for term in self.feature_terms
        }
        payload = {
            "architecture": self.architecture.value,
            "feature_terms": [t.value for t in self.feature_terms],
            "centering_mode": self.centering_mode.value,
            "requires_typicality_probe": bool(self.requires_typicality_probe),
            "requires_typicality_reference": bool(self.requires_typicality_reference),
            "requires_typicality_single": bool(self.requires_typicality_single),
            "requires_s0": bool(self.requires_s0),
            "requires_ordered_pair_roles": bool(self.requires_ordered_pair_roles),
            "formula_version": self.formula_version,
            "term_formula_versions": term_versions,
        }
        return hash_canonical_payload(payload)


# Constantes locais — abreviações para legibilidade da tabela
_T = LRFeatureTerm
_C = RarityCenteringMode


LR_ARCHITECTURE_SPECS: Mapping[LRCalibrationArchitecture, LRArchitectureSpec] = {
    LRCalibrationArchitecture.COSINE_ONLY: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.COSINE_ONLY,
        feature_terms=(_T.BASE_SCORE,),
        centering_mode=_C.NONE,
        requires_typicality_probe=False,
        requires_typicality_reference=False,
        requires_s0=False,
        requires_ordered_pair_roles=False,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.HINGE_ONLY: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.HINGE_ONLY,
        feature_terms=(_T.BASE_SCORE, _T.HINGE_POSITIVE),
        centering_mode=_C.NONE,
        requires_typicality_probe=False,
        requires_typicality_reference=False,
        requires_s0=True,
        requires_ordered_pair_roles=False,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.TWO_SIDED_HINGE_ONLY: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.TWO_SIDED_HINGE_ONLY,
        feature_terms=(_T.BASE_SCORE, _T.HINGE_POSITIVE, _T.HINGE_NEGATIVE),
        centering_mode=_C.NONE,
        requires_typicality_probe=False,
        requires_typicality_reference=False,
        requires_s0=True,
        requires_ordered_pair_roles=False,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED,
        feature_terms=(
            _T.BASE_SCORE,
            _T.HINGE_POSITIVE_X_RARITY_PROBE_RAW,
            _T.HINGE_POSITIVE_X_RARITY_REFERENCE_RAW,
        ),
        centering_mode=_C.NONE,
        requires_typicality_probe=True,
        requires_typicality_reference=True,
        requires_s0=True,
        requires_ordered_pair_roles=True,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.ONE_SIDED_CENTERED: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.ONE_SIDED_CENTERED,
        feature_terms=(
            _T.BASE_SCORE,
            _T.HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN,
            _T.HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN,
        ),
        centering_mode=_C.MEAN,
        requires_typicality_probe=True,
        requires_typicality_reference=True,
        requires_s0=True,
        requires_ordered_pair_roles=True,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.LINEAR_NON_CENTERED: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.LINEAR_NON_CENTERED,
        feature_terms=(
            _T.BASE_SCORE,
            _T.LINEAR_X_RARITY_PROBE_RAW,
            _T.LINEAR_X_RARITY_REFERENCE_RAW,
        ),
        centering_mode=_C.NONE,
        requires_typicality_probe=True,
        requires_typicality_reference=True,
        requires_s0=True,
        requires_ordered_pair_roles=True,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.LINEAR_CENTERED: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.LINEAR_CENTERED,
        feature_terms=(
            _T.BASE_SCORE,
            _T.LINEAR_X_RARITY_PROBE_CENTERED_MEAN,
            _T.LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN,
        ),
        centering_mode=_C.MEAN,
        requires_typicality_probe=True,
        requires_typicality_reference=True,
        requires_s0=True,
        requires_ordered_pair_roles=True,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.HIERARCHICAL_CENTERED: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.HIERARCHICAL_CENTERED,
        feature_terms=(
            _T.BASE_SCORE,
            _T.HINGE_POSITIVE,
            _T.HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN,
            _T.HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN,
        ),
        centering_mode=_C.MEAN,
        requires_typicality_probe=True,
        requires_typicality_reference=True,
        requires_s0=True,
        requires_ordered_pair_roles=True,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.BILATERAL_FULL_CENTERED: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
        feature_terms=(
            _T.BASE_SCORE,
            _T.HINGE_POSITIVE,
            _T.HINGE_NEGATIVE,
            _T.HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN,
            _T.HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN,
            _T.HINGE_NEGATIVE_X_RARITY_PROBE_CENTERED_MEAN,
            _T.HINGE_NEGATIVE_X_RARITY_REFERENCE_CENTERED_MEAN,
        ),
        centering_mode=_C.MEAN,
        requires_typicality_probe=True,
        requires_typicality_reference=True,
        requires_s0=True,
        requires_ordered_pair_roles=True,
        formula_version="1.0",
    ),
    # ── Família ONE_FACE (V2.14.1) ───────────────────────────────────────
    # Variantes que consomem raridade apenas de UMA face única.
    # `requires_ordered_pair_roles=False` porque não há "lado" a distinguir
    # quando apenas uma face contribui com tipicidade.
    LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED_ONE_FACE: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED_ONE_FACE,
        feature_terms=(
            _T.BASE_SCORE,
            _T.H_POS_TIMES_SINGLE_RARITY,
        ),
        centering_mode=_C.NONE,
        requires_typicality_probe=False,
        requires_typicality_reference=False,
        requires_typicality_single=True,
        requires_s0=True,
        requires_ordered_pair_roles=False,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.ONE_SIDED_CENTERED_ONE_FACE: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.ONE_SIDED_CENTERED_ONE_FACE,
        feature_terms=(
            _T.BASE_SCORE,
            _T.H_POS_TIMES_SINGLE_RARITY_CENTERED,
        ),
        centering_mode=_C.MEAN,
        requires_typicality_probe=False,
        requires_typicality_reference=False,
        requires_typicality_single=True,
        requires_s0=True,
        requires_ordered_pair_roles=False,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.LINEAR_NON_CENTERED_ONE_FACE: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.LINEAR_NON_CENTERED_ONE_FACE,
        feature_terms=(
            _T.BASE_SCORE,
            _T.LINEAR_TIMES_SINGLE_RARITY,
        ),
        centering_mode=_C.NONE,
        requires_typicality_probe=False,
        requires_typicality_reference=False,
        requires_typicality_single=True,
        requires_s0=True,
        requires_ordered_pair_roles=False,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.LINEAR_CENTERED_ONE_FACE: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.LINEAR_CENTERED_ONE_FACE,
        feature_terms=(
            _T.BASE_SCORE,
            _T.LINEAR_TIMES_SINGLE_RARITY_CENTERED,
        ),
        centering_mode=_C.MEAN,
        requires_typicality_probe=False,
        requires_typicality_reference=False,
        requires_typicality_single=True,
        requires_s0=True,
        requires_ordered_pair_roles=False,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.HIERARCHICAL_CENTERED_ONE_FACE: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.HIERARCHICAL_CENTERED_ONE_FACE,
        feature_terms=(
            _T.BASE_SCORE,
            _T.HINGE_POSITIVE,
            _T.H_POS_TIMES_SINGLE_RARITY_CENTERED,
        ),
        centering_mode=_C.MEAN,
        requires_typicality_probe=False,
        requires_typicality_reference=False,
        requires_typicality_single=True,
        requires_s0=True,
        requires_ordered_pair_roles=False,
        formula_version="1.0",
    ),
    LRCalibrationArchitecture.BILATERAL_FULL_CENTERED_ONE_FACE: LRArchitectureSpec(
        architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED_ONE_FACE,
        feature_terms=(
            _T.BASE_SCORE,
            _T.HINGE_POSITIVE,
            _T.HINGE_NEGATIVE,
            _T.H_POS_TIMES_SINGLE_RARITY_CENTERED,
            _T.H_NEG_TIMES_SINGLE_RARITY_CENTERED,
        ),
        centering_mode=_C.MEAN,
        requires_typicality_probe=False,
        requires_typicality_reference=False,
        requires_typicality_single=True,
        requires_s0=True,
        requires_ordered_pair_roles=False,
        formula_version="1.0",
    ),
}


def get_architecture_spec(architecture: LRCalibrationArchitecture) -> LRArchitectureSpec:
    """Lookup do spec. Levanta KeyError se a arquitetura não estiver registrada."""
    return LR_ARCHITECTURE_SPECS[architecture]
