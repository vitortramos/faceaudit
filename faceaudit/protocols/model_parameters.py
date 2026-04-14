from typing import Protocol, runtime_checkable, Dict, Any

@runtime_checkable
class ModelParametersProtocol(Protocol):
    """
    Protocolo (Interface) para classes de parâmetros de modelos (Hiperparâmetros).
    Garante que qualquer configuração matemática possa ser serializada/logada.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Deve retornar uma representação em dicionário dos parâmetros.
        """
        ...