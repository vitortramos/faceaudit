# faceaudit/models/validation/validation_result.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Type, TypeVar, Optional

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.validation_severity import ValidationSeverity


T = TypeVar("T", bound="ValidationResult")


@dataclass
class ValidationResult(JSONSerializableMixin):
    """
    Representa o resultado estruturado de uma validação.

    Pode ser usado tanto por regras individuais quanto por validadores globais
    da estratégia. Suporta agregação incremental de erros e avisos.
    """

    severity: ValidationSeverity = field(default=ValidationSeverity.NEUTRAL)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """
        Retorna True se não houver erro crítico bloqueante.
        """
        return self.severity != ValidationSeverity.CRITICAL and not self.errors

    @property
    def has_warnings(self) -> bool:
        """
        Retorna True se houver avisos registrados.
        """
        return bool(self.warnings)

    def add_error(self, message: str) -> None:
        """
        Adiciona um erro bloqueante ao resultado.
        """
        if message:
            self.errors.append(message)
            self.severity = ValidationSeverity.CRITICAL

    def add_warning(self, message: str) -> None:
        """
        Adiciona um aviso não bloqueante ao resultado.
        """
        if message:
            self.warnings.append(message)
            if self.severity not in (ValidationSeverity.CRITICAL,):
                self.severity = ValidationSeverity.WARNING

    def mark_ok(self) -> None:
        """
        Marca explicitamente o resultado como OK caso não existam erros nem avisos.
        """
        if not self.errors and not self.warnings:
            self.severity = ValidationSeverity.OK

    def merge(self, other: Optional["ValidationResult"]) -> "ValidationResult":
        """
        Mescla outro ValidationResult neste objeto, agregando erros, avisos
        e ajustando a severidade final.
        """
        if other is None:
            return self

        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

        if self.errors or other.severity == ValidationSeverity.CRITICAL:
            self.severity = ValidationSeverity.CRITICAL
        elif self.warnings or other.severity == ValidationSeverity.WARNING:
            self.severity = ValidationSeverity.WARNING
        elif (
            self.severity == ValidationSeverity.NEUTRAL
            and other.severity == ValidationSeverity.OK
        ):
            self.severity = ValidationSeverity.OK

        return self

    @classmethod
    def ok(cls: Type[T]) -> T:
        """
        Fábrica para resultado válido e sem pendências.
        """
        return cls(severity=ValidationSeverity.OK)

    @classmethod
    def neutral(cls: Type[T]) -> T:
        """
        Fábrica para estado inicial/neutro.
        """
        return cls(severity=ValidationSeverity.NEUTRAL)

    @classmethod
    def critical(cls: Type[T], message: str) -> T:
        """
        Fábrica para erro crítico único.
        """
        obj = cls(severity=ValidationSeverity.CRITICAL)
        obj.add_error(message)
        return obj

    @classmethod
    def warning(cls: Type[T], message: str) -> T:
        """
        Fábrica para aviso único.
        """
        obj = cls(severity=ValidationSeverity.WARNING)
        obj.add_warning(message)
        return obj

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO convertendo corretamente a severidade.
        """
        if not data:
            return cls.neutral()

        processed_data = data.copy()

        if "severity" in processed_data and processed_data["severity"]:
            processed_data["severity"] = ValidationSeverity.from_str(processed_data["severity"])

        return cls(**processed_data)