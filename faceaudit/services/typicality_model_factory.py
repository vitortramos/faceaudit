# faceaudit/services/typicality_model_factory.py

"""Factory de serialização para modelos de tipicidade.

`to_dict(model)` produz envelope versionado discriminado por
`TypicalityModelType`. `from_dict(payload)` resolve a classe concreta
via `TypicalityModelBackendRegistry` e devolve uma instância pronta para
scoring (não-Fittable — quem quer treinar usa o trainer do registry).

Envelope V1:

    {
        "schema_version": 1,
        "model_type": "KNN" | "MOVMF",
        "legacy_to_dict": {...},          # to_dict() interno do modelo concreto
        "fingerprints": {
            "model_state_fingerprint": "...",
            "model_contract_fingerprint": "...",
        },
        "provenance": {
            "faceaudit_envelope_version": 1,
        },
    }

Notas:
- `legacy_to_dict` reusa os métodos `to_dict()`/`from_dict()` já
  implementados nos modelos. Centralizar o envelope nesta camada permite
  evolução do schema sem mexer no formato interno do modelo concreto.
- Em PR 1b o envelope ainda **não** canoniza componentes MoVMF — isso
  entra no PR 4.
- `from_dict` retorna `TypicalityScoringModelProtocol`; refit é
  responsabilidade do trainer do registry.
"""

from __future__ import annotations

from typing import Any, Dict

from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.protocols.typicality_scoring_model_protocol import (
    TypicalityScoringModelProtocol,
)
from faceaudit.services.typicality_model_backend_registry import (
    TypicalityModelBackendRegistry,
)


ENVELOPE_SCHEMA_VERSION = 1


class TypicalityModelFactory:
    """Serialização discriminada via `TypicalityModelType`."""

    @staticmethod
    def to_dict(model: TypicalityScoringModelProtocol) -> Dict[str, Any]:
        """Encapsula o modelo num envelope versionado."""
        # Modelo deve estar treinado para serialização (legacy to_dict exige)
        if hasattr(model, "is_fitted") and callable(model.is_fitted):
            if not model.is_fitted():
                raise RuntimeError(
                    "TypicalityModelFactory.to_dict: modelo não treinado."
                )
        legacy = model.to_dict()  # type: ignore[attr-defined]
        return {
            "schema_version": ENVELOPE_SCHEMA_VERSION,
            "model_type": model.model_type.to_str(),
            "legacy_to_dict": legacy,
            "fingerprints": {
                "model_state_fingerprint": model.model_state_fingerprint,
                "model_contract_fingerprint": model.model_contract_fingerprint,
            },
            "provenance": {
                "faceaudit_envelope_version": ENVELOPE_SCHEMA_VERSION,
            },
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> TypicalityScoringModelProtocol:
        """Resolve a classe concreta via registry e reconstrói o modelo.

        Valida `schema_version` e `model_type`. Re-verifica os fingerprints
        após reconstrução — divergência levanta `EnvelopeFingerprintMismatch`.
        """
        if not isinstance(payload, dict):
            raise TypeError(
                f"TypicalityModelFactory.from_dict: payload deve ser dict, "
                f"recebeu {type(payload).__name__}."
            )

        sv = payload.get("schema_version")
        if sv != ENVELOPE_SCHEMA_VERSION:
            raise ValueError(
                f"TypicalityModelFactory.from_dict: schema_version "
                f"{sv!r} não suportada (esperado "
                f"{ENVELOPE_SCHEMA_VERSION})."
            )

        type_str = payload.get("model_type")
        if not type_str:
            raise ValueError(
                "TypicalityModelFactory.from_dict: envelope sem 'model_type'."
            )
        model_type = TypicalityModelType.from_str(type_str)
        cls = TypicalityModelBackendRegistry.resolve_class(model_type)

        if "legacy_to_dict" not in payload:
            raise ValueError(
                "TypicalityModelFactory.from_dict: envelope sem "
                "'legacy_to_dict'."
            )
        if not hasattr(cls, "from_dict"):
            raise TypeError(
                f"Classe {cls.__name__} registrada como backend de "
                "tipicidade não expõe `from_dict` classmethod."
            )
        model = cls.from_dict(payload["legacy_to_dict"])

        # P0-R6-4 (revisão externa round 6): no schema atual, ambos
        # os fingerprints são OBRIGATÓRIOS. Antes, `expected_state` e
        # `expected_contract` eram verificados apenas quando presentes,
        # permitindo bypass via remoção do bloco `fingerprints`.
        expected = payload.get("fingerprints", {}) or {}
        expected_state = expected.get("model_state_fingerprint")
        expected_contract = expected.get("model_contract_fingerprint")
        if not expected_state:
            raise ValueError(
                "TypicalityModelFactory.from_dict: envelope sem "
                "`fingerprints.model_state_fingerprint`. No schema atual "
                "ambos os fingerprints (state + contract) são obrigatórios "
                "— sua ausência desativaria a proteção contra adulteração."
            )
        if not expected_contract:
            raise ValueError(
                "TypicalityModelFactory.from_dict: envelope sem "
                "`fingerprints.model_contract_fingerprint`. Idem regra "
                "acima: ambos os fingerprints são obrigatórios."
            )
        if model.model_state_fingerprint != expected_state:
            raise EnvelopeFingerprintMismatch(
                kind="state",
                expected=expected_state,
                actual=model.model_state_fingerprint,
            )
        if model.model_contract_fingerprint != expected_contract:
            raise EnvelopeFingerprintMismatch(
                kind="contract",
                expected=expected_contract,
                actual=model.model_contract_fingerprint,
            )
        return model


class EnvelopeFingerprintMismatch(ValueError):
    """Fingerprint declarado no envelope divergente do recomputado pós-load."""

    def __init__(self, *, kind: str, expected: str, actual: str) -> None:
        super().__init__(
            f"TypicalityModelFactory: fingerprint {kind!r} divergente. "
            f"Esperado={expected[:16]}..., atual={actual[:16]}..."
        )
        self.kind = kind
        self.expected = expected
        self.actual = actual
