from __future__ import annotations
from enum import Enum
from typing import Optional

class ValidationSeverity(Enum):
    """
    Define os níveis de gravidade para os resultados de validação do sistema.
    Utilizado para controlar a lógica de bloqueio de execução e feedback visual (cores).
    """
    OK = "ok"             # Verde: Seguro para prosseguir
    WARNING = "warning"   # Amarelo: Atenção (Risco estatístico ou metodológico)
    CRITICAL = "critical" # Vermelho: Erro Matemático ou Lógico (Bloqueia execução)
    NEUTRAL = "neutral"   # Cinza: Estado inicial / Aguardando dados

    def __str__(self) -> str:
        return self.value

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna o identificador de severidade para o JSON (ex: 'critical')."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> ValidationSeverity:
        """
        Reconstrói o Enum a partir da string.
        Retorna NEUTRAL como fallback seguro.
        """
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.NEUTRAL