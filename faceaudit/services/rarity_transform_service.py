# faceaudit/services/rarity_transform_service.py
"""
Conversão canônica de scores de tipicidade em raridade.

Convenção: higher rarity = mais rara. Para cada `TypicalityScoreType`,
o registry expõe uma transformação que produz raridade satisfazendo
essa convenção:

  - TYPICALITY_PERCENTILE (v ∈ [0,1]): r = 1 − v.
  - MEAN_KNN_SIMILARITY (v ∈ [-1,1]): r = −v.
  - LOG_DENSITY_KNN (v ∈ ℝ): r = −v.

Extensibilidade: o registry aceita `register(spec, override=False)`.
Novas tipicidades (MoVMF, etc.) registram um `RarityTransformSpec`
descrevendo identificador estável, faixa válida e função. O
`ArchitectureFeatureBuilder` NUNCA conhece tipos específicos — só chama
`to_rarity(typicality, score_type)`.

Diferença entre as duas APIs:

- `to_canonical_rarity(...)`: nova instrução declarativa. Valida `min_input`,
  `max_input`, `higher_means_rarer`, NaN/inf. Falha estrita.
- `legacy_transform_typicality(...)`: SCORE_INTERACTION_FEATURE histórica.
  Preserva semântica antiga (sem validar ranges) para tolerar artefatos
  de estratégias legadas. Não é exportada pelo registry — é uma função
  separada com contrato distinto.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.models.lr._canonical_hash import hash_canonical_payload


@dataclass(frozen=True)
class RarityTransformSpec:
    """
    Especificação de uma transformação tipicidade → raridade canônica.

    `transform_id` é um identificador estável em CAPS, persistido em
    estados/fingerprints. Mudanças cosméticas em `transform_label` (rótulo
    de UI) não afetam fingerprints. Mudanças em `fn` exigem bump de
    `formula_version`.

    `fingerprint()` cobre transform_id + formula_version + min_input +
    max_input + higher_means_rarer — uma alteração silenciosa de `fn`
    sem bump não é detectada (mitigado por testes sentinela com vetores
    fixos no PR 1).
    """

    score_type: TypicalityScoreType
    transform_id: str
    formula_version: str
    transform_label: str
    fn: Callable[[np.ndarray], np.ndarray]
    min_input: Optional[float]
    max_input: Optional[float]
    higher_means_rarer: bool

    def fingerprint(self) -> str:
        payload = {
            "transform_id": self.transform_id,
            "formula_version": self.formula_version,
            "min_input_hex": (
                None if self.min_input is None else float(self.min_input).hex()
            ),
            "max_input_hex": (
                None if self.max_input is None else float(self.max_input).hex()
            ),
            "higher_means_rarer": bool(self.higher_means_rarer),
        }
        return hash_canonical_payload(payload)


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────


class RarityTransformRegistry:
    """
    Registry plugável de `RarityTransformSpec`.

    Inicialização determinística: chamar `_ensure_default_registrations()`
    é idempotente (default registrations já presentes não reentram).
    """

    _registry: Dict[TypicalityScoreType, RarityTransformSpec] = {}
    _defaults_registered: bool = False

    @classmethod
    def register(cls, spec: RarityTransformSpec, *, override: bool = False) -> None:
        """
        Registra uma transformação para o `score_type` correspondente.

        Por padrão, rejeita duplicado (`ValueError`). `override=True`
        substitui o registro existente com warning explícito — usar com
        cuidado, idealmente apenas em testes.
        """
        existing = cls._registry.get(spec.score_type)
        if existing is not None and not override:
            raise ValueError(
                f"RarityTransformSpec para {spec.score_type.value} já registrado "
                f"(transform_id={existing.transform_id}). Use override=True "
                "para substituir."
            )
        if existing is not None and override:
            warnings.warn(
                f"RarityTransformRegistry: SUBSTITUINDO transform de "
                f"{spec.score_type.value} ({existing.transform_id} → "
                f"{spec.transform_id})",
                stacklevel=2,
            )
        cls._registry[spec.score_type] = spec

    @classmethod
    def get(cls, score_type: TypicalityScoreType) -> RarityTransformSpec:
        """Retorna o spec registrado. Levanta `KeyError` se ausente."""
        cls._ensure_default_registrations()
        if score_type not in cls._registry:
            raise KeyError(
                f"RarityTransformSpec não registrado para {score_type.value}. "
                "Use RarityTransformRegistry.register(...) ou aguarde a inicialização."
            )
        return cls._registry[score_type]

    @classmethod
    def to_canonical_rarity(
        cls,
        typicality: np.ndarray,
        score_type: TypicalityScoreType,
    ) -> Tuple[np.ndarray, str]:
        """
        Converte tipicidade em raridade canônica via spec registrada.

        Valida: shape == 1D; finitude (NaN/inf rejeitados); range
        [min_input, max_input] se declarados.

        Retorna `(rarity, transform_id)`.
        """
        spec = cls.get(score_type)
        arr = np.asarray(typicality, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"typicality deve ser 1D; shape recebido {arr.shape}")
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"typicality contém {n_bad} valor(es) NaN/inf — rejeitado "
                "pela transformação canônica"
            )
        if spec.min_input is not None and float(arr.min()) < spec.min_input:
            raise ValueError(
                f"typicality.min={arr.min():.6f} abaixo de min_input "
                f"{spec.min_input} para transform {spec.transform_id}"
            )
        if spec.max_input is not None and float(arr.max()) > spec.max_input:
            raise ValueError(
                f"typicality.max={arr.max():.6f} acima de max_input "
                f"{spec.max_input} para transform {spec.transform_id}"
            )
        rarity = spec.fn(arr).astype(np.float64, copy=False)
        return rarity, spec.transform_id

    @classmethod
    def list_registered(cls) -> Tuple[RarityTransformSpec, ...]:
        cls._ensure_default_registrations()
        return tuple(cls._registry.values())

    @classmethod
    def _ensure_default_registrations(cls) -> None:
        """Idempotente. Registra os 3 transforms canônicos."""
        if cls._defaults_registered:
            return
        # Marcar antes pra evitar recursão inadvertida
        cls._defaults_registered = True

        if TypicalityScoreType.TYPICALITY_PERCENTILE not in cls._registry:
            cls._registry[TypicalityScoreType.TYPICALITY_PERCENTILE] = RarityTransformSpec(
                score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
                transform_id="ONE_MINUS_PERCENTILE_0_1",
                formula_version="1.0",
                transform_label="1 − percentil (em [0,1])",
                fn=lambda v: 1.0 - v,
                min_input=0.0,
                max_input=1.0,
                higher_means_rarer=True,
            )
        if TypicalityScoreType.MEAN_KNN_SIMILARITY not in cls._registry:
            cls._registry[TypicalityScoreType.MEAN_KNN_SIMILARITY] = RarityTransformSpec(
                score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
                transform_id="NEGATE_MEAN_KNN_SIMILARITY",
                formula_version="1.0",
                transform_label="−similaridade média kNN",
                fn=lambda v: -v,
                min_input=-1.0,
                max_input=1.0,
                higher_means_rarer=True,
            )
        if TypicalityScoreType.LOG_DENSITY_KNN not in cls._registry:
            cls._registry[TypicalityScoreType.LOG_DENSITY_KNN] = RarityTransformSpec(
                score_type=TypicalityScoreType.LOG_DENSITY_KNN,
                transform_id="NEGATE_LOG_DENSITY_KNN",
                formula_version="1.0",
                transform_label="−log-densidade kNN",
                fn=lambda v: -v,
                min_input=None,
                max_input=None,
                higher_means_rarer=True,
            )

    @classmethod
    def _reset_for_tests(cls) -> None:
        """Limpa o registry. APENAS para uso em testes (não exportado)."""
        cls._registry.clear()
        cls._defaults_registered = False


def to_canonical_rarity(
    typicality: np.ndarray,
    score_type: TypicalityScoreType,
) -> Tuple[np.ndarray, str]:
    """Atalho funcional para `RarityTransformRegistry.to_canonical_rarity`."""
    return RarityTransformRegistry.to_canonical_rarity(typicality, score_type)


def legacy_transform_typicality(
    typicality: np.ndarray,
    score_type: TypicalityScoreType,
    invert: bool = True,
) -> Tuple[np.ndarray, str]:
    """
    Transformação histórica usada por `SCORE_INTERACTION_FEATURE` legacy.

    Preserva o comportamento literal do executor antigo: sem validação
    de min/max nem rejeição de NaN/inf. A flag `invert` controla a
    inversão, mantendo compatibilidade com JSONs antigos. Não use na
    nova instrução declarativa — use `to_canonical_rarity` em vez.
    """
    arr = np.asarray(typicality, dtype=np.float64)
    if score_type is TypicalityScoreType.TYPICALITY_PERCENTILE:
        # Upstream já normaliza para [0,1]; aplicar 1 - v direto.
        if invert:
            return 1.0 - arr, "1 - percentile_0_1"
        return arr, "percentile_0_1"
    if score_type is TypicalityScoreType.MEAN_KNN_SIMILARITY:
        if invert:
            return -arr, "-mean_knn_similarity"
        return arr, "mean_knn_similarity"
    if score_type is TypicalityScoreType.LOG_DENSITY_KNN:
        if invert:
            return -arr, "-log_density_knn"
        return arr, "log_density_knn"
    raise ValueError(f"score_type desconhecido em legacy_transform_typicality: {score_type}")
