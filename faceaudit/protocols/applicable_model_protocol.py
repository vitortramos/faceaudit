# faceaudit/protocols/applicable_model_protocol.py

from __future__ import annotations
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ApplicableModelProtocol(Protocol):
    """
    Protocolo genérico para modelos aplicáveis da nova arquitetura.

    Um modelo aplicável é capaz de operar sobre dados de entrada já na fase
    de uso/inferência, produzindo um resultado derivado.
    """

    def apply(self, *args, **kwargs) -> Any:
        """
        Aplica o modelo sobre a entrada informada.

        A assinatura concreta depende da natureza do modelo.
        Exemplos:
        - pipeline: apply(dataset)
        - calibrador: apply(scores)
        - scorer: apply(comparisons)
        """
        ...

    def apply_single(self, item: Any) -> Any:
        """
        Aplica o modelo a um único item.

        Este método é opcional do ponto de vista conceitual, mas faz sentido
        como parte do contrato para modelos que precisem suportar aplicação
        unitária além da aplicação em lote.
        """
        ...