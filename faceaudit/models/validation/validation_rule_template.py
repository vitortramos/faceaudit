# faceaudit/models/validation/validation_rule_template.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="ValidationRuleTemplate")


@dataclass
class ValidationRuleTemplate(JSONSerializableMixin):
    """
    Template salvo de regra de validação — wrap em torno de `ValidationRuleConfig`
    com metadados editoriais (nome, descrição) pra UX de "import template".

    Persistido em:
    - global library: `<library_root>/validation_rules/{template_id}.json`
    - projeto: `<project>/validation_rules/{template_id}.json`

    O `rule_config` é a fonte de verdade técnica (rule_type + params) — quando
    o usuário importa um template, o config é embedded em
    `instruction.validation_rule_configs`. Metadata (`name`, `description`)
    serve só pra UI de seleção: aparecem no dropdown do `AddValidationRuleDialog`
    pra o usuário identificar qual template é qual.

    template_id é o nome do arquivo (sem extensão). Define-se na hora do save:
    o usuário escolhe um nome amigável tipo "FISHER_BFW_strict" e o arquivo
    vai pra `validation_rules/FISHER_BFW_strict.json`.
    """

    template_id: str
    name: str
    rule_config: ValidationRuleConfig
    description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "rule_config": self.rule_config.to_dict() if self.rule_config else None,
        }

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None

        processed = dict(data)
        if "rule_config" in processed and isinstance(processed["rule_config"], dict):
            processed["rule_config"] = ValidationRuleConfig.from_dict(processed["rule_config"])

        if processed.get("rule_config") is None:
            return None

        return cls(
            template_id=processed.get("template_id", ""),
            name=processed.get("name", processed.get("template_id", "")),
            description=processed.get("description"),
            rule_config=processed["rule_config"],
        )
