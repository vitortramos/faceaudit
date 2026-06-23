# faceaudit/enums/video_extraction_mode.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class VideoExtractionMode(Enum):
    """
    Ponto de parada da pipeline de extração de faces de vídeo.

    A mesma pipeline produz os recortes + `_meta/`; o modo só decide o
    que acontece depois:

    - `PRODUCER`: para nos recortes `<rótulo>/...png` + `_meta/`. O
      perito importa depois (manual) via o importador de run de vídeo.
    - `END_TO_END`: após a extração e a revisão humana (invariante:
      parquet só pós-review), monta o `FaceDataset` e enriquece
      (FF/OFIQ/SERFIQ) → dataset registrado, numa tacada. Os recortes
      permanecem em disco (evidência).
    """

    PRODUCER = (
        "PRODUCER",
        "Produtor (só recortes + metadados)",
        "Para nos recortes + _meta/. Importação posterior pelo fluxo "
        "de importação de run de vídeo.",
    )
    END_TO_END = (
        "END_TO_END",
        "Ponta a ponta (gera o dataset)",
        "Após extração + revisão, monta o FaceDataset e enriquece "
        "(FF/OFIQ/SERFIQ) → dataset registrado.",
    )

    def __new__(cls, value: str, label_pt: str, description: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label_pt = label_pt
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value) -> "VideoExtractionMode":
        if not value:
            return cls.PRODUCER
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.PRODUCER

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]
