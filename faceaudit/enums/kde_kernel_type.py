from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class KernelTypeInfo:
    label: str
    sklearn_name: str # Nome exato que o Scikit-Learn espera

class KDEKernelType(Enum):
    """
    Tipo de Kernel para a Estimativa de Densidade (KDE).
    Afeta a suavidade da curva gerada e a modelagem estatística dos scores.
    """
    GAUSSIAN = "gaussian"
    TOPHAT = "tophat"
    EPANECHNIKOV = "epanechnikov"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"

    @property
    def info(self) -> KernelTypeInfo:
        """Retorna metadados descritivos e o mapeamento para o scikit-learn."""
        if self is KDEKernelType.GAUSSIAN:
            return KernelTypeInfo("Gaussiano (Normal)", "gaussian")
        if self is KDEKernelType.TOPHAT:
            return KernelTypeInfo("Tophat (Box)", "tophat")
        if self is KDEKernelType.EPANECHNIKOV:
            return KernelTypeInfo("Epanechnikov (Parabólico)", "epanechnikov")
        if self is KDEKernelType.EXPONENTIAL:
            return KernelTypeInfo("Exponencial", "exponential")
        if self is KDEKernelType.LINEAR:
            return KernelTypeInfo("Linear", "linear")
        
        return KernelTypeInfo(self.name, self.value)

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna o valor string do kernel para persistência JSON."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> KDEKernelType:
        """
        Reconstrói o Enum a partir de uma string. 
        Retorna GAUSSIAN como fallback padrão do sistema.
        """
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.GAUSSIAN