# faceaudit/services/typicality_model.py

"""
Shim de compatibilidade — DEPRECATED.

Este módulo foi renomeado para `faceaudit.services.movmf_typicality_model`
e a classe `TypicalityModel` foi renomeada para `MoVMFTypicalityModel` para
alinhar com `KNNTypicalityModel` (família de backends de tipicidade).

Imports antigos continuam funcionando até V2.16:

    from faceaudit.services.typicality_model import TypicalityModel
    from faceaudit.services.typicality_model import TypicalityInfo

Novo padrão:

    from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
    from faceaudit.models.typicality_info import TypicalityInfo
"""

import warnings

from faceaudit.models.typicality_info import TypicalityInfo
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel

warnings.warn(
    "`faceaudit.services.typicality_model` foi renomeado para "
    "`faceaudit.services.movmf_typicality_model`; a classe `TypicalityModel` "
    "agora se chama `MoVMFTypicalityModel`. Este alias é mantido por "
    "compatibilidade e será removido em V2.16.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias de classe para imports antigos
TypicalityModel = MoVMFTypicalityModel

__all__ = ["TypicalityModel", "MoVMFTypicalityModel", "TypicalityInfo"]
