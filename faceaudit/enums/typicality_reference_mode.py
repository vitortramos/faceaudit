# faceaudit/enums/typicality_reference_mode.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class TypicalityReferenceMode(Enum):
    """
    Como construir a referência do `KNNTypicalityModel` a partir do
    dataset de treino.

    PER_FACE: cada face é um ponto da referência. A vizinhança k-NN é
        medida no espaço de faces. Hubness intra-identidade (faces da
        mesma pessoa aparecem como vizinhos uns dos outros) é mascarada
        como "tipicidade alta". Útil pra diagnóstico de fotos atípicas
        dentro de identidades típicas, mas vulnerável a desbalanceamento
        de identidades.

    PER_IDENTITY (default): agrupa por identidade, calcula centroide
        L2-renormalizado por identidade. A referência tem 1 ponto por
        pessoa, não por foto. Tipicidade mede então "quão típica é esta
        PESSOA na população", não "quão típica é esta FOTO". Defensável
        em corte forense ("esta pessoa está no 73º percentil da população
        de referência"). Identidades singleton viram centroide igual à
        face única — não são descartadas.
    """

    PER_FACE = ("PER_FACE", "Por face")
    PER_IDENTITY = ("PER_IDENTITY", "Por identidade (centroide)")

    def __new__(cls, value: str, label_pt: str):
        # Definir `_value_` em `__new__` (não em `__init__`) é o pattern
        # canônico do CPython pra enums com valor composto (string id +
        # label PT). Em `__init__` o `_value2member_map_` já foi populado
        # pelo metaclass com a tupla original — sobrescrever `_value_`
        # depois quebra `MyEnum(value)` e o roundtrip de pickle:
        # `pickle.dumps(member)` salva `_value_` (a string), mas
        # `pickle.loads` chama `MyEnum('PER_IDENTITY')` que falha porque
        # o mapa interno só tem a tupla como chave. Modelos serializados
        # antes desse fix (knn_typicality_model.pkl) começam a carregar
        # de novo após esse ajuste.
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label_pt = label_pt
        return obj

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "TypicalityReferenceMode":
        if not value:
            return cls.PER_IDENTITY
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.PER_IDENTITY

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]
