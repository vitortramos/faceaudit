"""Config DTO para a transformação Radial Gaussianization (RG).

Baseado em Lyu & Simoncelli (2009), "Nonlinear Extraction of
Independent Components of Natural Images Using Radial Gaussianization",
Neural Computation 21(6):1485–1519.

A transformação opera em dados **whitened** (idealmente após uma
etapa de Whitening no pipeline). Recompõe a marginal radial das
amostras via mapeamento `g(r) = F_χ⁻¹(F_r(r))` onde:
  - F_r é a CDF empírica das normas dos vetores de treino
  - F_χ é a CDF da distribuição chi com d graus de liberdade
    (dimensionalidade dos embeddings)

Output: cada vetor `x' = x · g(‖x‖) / ‖x‖`. Direção preservada,
norma reposicionada para a marginal de uma gaussiana esférica unitária.

⚠ Importante (paper §3.4): RG só remove dependências quando o source
é elipticamente simétrico (ESD). Em sources factoriais, RG pode
AUMENTAR a dependência. Caso típico em embeddings faciais: joint
local de filtros bandpass é ESD; embeddings de modelos treinados
podem variar — convém medir mutual information antes/depois.
"""

from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

T = TypeVar("T", bound="RadialGaussianizationConfig")


@dataclass
class RadialGaussianizationConfig(JSONSerializableMixin):
    """Configuração para Radial Gaussianization.

    Attributes:
        eps_clip: clamp dos quantis empíricos antes do `chi.ppf`.
            Sem isso, u=1.0 produziria r=∞ no inverso da chi.
            Default 1e-6 (≈ 1 amostra a cada 1M).
        assume_whitened: quando True, emite warning no fit se a
            covariância do source desviar muito de I (proxy de
            "não whitened"). Não impede a execução — RG é definido
            mesmo sem whitening, apenas perde a propriedade
            matemática garantida do paper.
        whitening_check_tolerance: tolerância pra detectar desvio de
            whitening (mean da diagonal da covariância vs 1.0).
            Default 0.1 = ±10%.
    """

    eps_clip: float = field(default=1e-6)
    assume_whitened: bool = field(default=True)
    whitening_check_tolerance: float = field(default=0.1)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return cls()
        return cls(
            eps_clip=float(data.get("eps_clip", 1e-6)),
            assume_whitened=bool(data.get("assume_whitened", True)),
            whitening_check_tolerance=float(
                data.get("whitening_check_tolerance", 0.1)
            ),
        )
