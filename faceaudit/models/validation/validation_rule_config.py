# faceaudit/models/validation/validation_rule_config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar

from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="ValidationRuleConfig")


@dataclass
class ValidationRuleConfig(JSONSerializableMixin):
    """
    Configuração serializável de uma regra de validação plugável.

    O `rule_type` identifica qual classe concreta de regra instanciar;
    `params` carrega os parâmetros de construção (ex: dataset_resource_id,
    min_genuine, safety_margin_ratio). A factory
    `ValidationRuleFactory.create(config, instruction_id)` consome esta
    config para produzir a regra concreta.

    Persistido inline em `StrategyInstructionDefinition.validation_rule_configs`.
    """

    rule_type: ValidationRuleType
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None

        processed = data.copy()
        raw_type = processed.get("rule_type")
        if raw_type is None:
            return None

        if isinstance(raw_type, ValidationRuleType):
            rule_type = raw_type
        else:
            try:
                rule_type = ValidationRuleType.from_str(raw_type)
            except ValueError:
                # Tipo desconhecido (provavelmente de uma versão futura ou
                # renomeada). Descartado silenciosamente — o caller decide
                # se isso é fatal ao montar a lista de configs.
                return None

        params = processed.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        return cls(rule_type=rule_type, params=dict(params))
