from __future__ import annotations
from enum import Enum
from typing import Type, TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from faceaudit.protocols.strategy_manager_protocol import StrategyManagerProtocol

class StrategyType(Enum):
    """
    Enum que define os tipos de estratégias e mapeia para suas respectivas classes.
    Atua como um Factory Registry para Managers e Configurações.
    """
    PLDA = "PLDA"
    KDE = "KDE"

    @property
    def strategy_class(self) -> Type['StrategyManagerProtocol']:
        """Lookup do Manager associado — removido na Fase 3 da migração.
        Os managers (PLDAManager/KDEManager) foram movidos para
        `Código Velho/`; estratégias agora usam o pipeline declarativo
        da nova arquitetura."""
        raise NotImplementedError(
            "Managers da arquitetura antiga foram movidos. Use a nova "
            "arquitetura via StrategyExecutorFactory."
        )

    @property
    def config_class(self) -> Type:
        """
        Retorna a classe de DTO de Configuração correspondente ao tipo.
        Fundamental para a reconstrução polimórfica via JSON.
        """
        if self is StrategyType.PLDA:
            from faceaudit.models.plda_config import PLDAConfig
            return PLDAConfig
        elif self is StrategyType.KDE:
            from faceaudit.models.kde_config import KDEConfig
            return KDEConfig
        raise ValueError(f"Config class desconhecida para: {self.value}")

    def __str__(self) -> str:
        return self.value

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna a string identificadora (ex: 'PLDA')."""
        return self.value

    @classmethod
    def from_str(cls, value: Any) -> StrategyType:
        """
        Reconstrói o Enum a partir do valor lido do JSON.
        Normaliza para maiúsculas e remove espaços.
        """
        if value is None:
            return cls.KDE # Fallback seguro ou erro, conforme regra de negócio
        
        try:
            val_norm = str(value).strip().upper()
            return cls(val_norm)
        except ValueError:
            return cls.KDE # Fallback padrão

    @classmethod
    def from_json(cls, value: Any) -> StrategyType:
        """Mantido para compatibilidade com chamadas legadas do WorkspaceConfig."""
        return cls.from_str(value)