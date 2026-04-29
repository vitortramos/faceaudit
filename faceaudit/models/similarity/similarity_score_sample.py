# faceaudit/models/similarity/similarity_score_sample.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.embedding_info import EmbeddingInfo


T = TypeVar("T", bound="SimilarityScoreSample")


@dataclass(slots=True)
class SimilarityScoreSample(JSONSerializableMixin):
    """
    Representa uma amostra individual de score de similaridade entre dois embeddings.

    Esta classe é a ponte natural entre:
    - modelos de similaridade, que produzem score bruto
    - modelos de calibração, que aprendem a partir de scores e ground truth
    - rotinas de auditoria e análise de erro

    Diferentemente do DTO antigo preso ao cosseno, aqui usamos `raw_score`
    para manter neutralidade em relação ao modelo gerador:
    - cosine
    - PLDA
    - outros modelos futuros
    """

    probe: EmbeddingInfo = field(
        metadata={"help": "EmbeddingInfo do item probe."}
    )

    reference: EmbeddingInfo = field(
        metadata={"help": "EmbeddingInfo do item reference."}
    )

    raw_score: float = field(
        metadata={"help": "Score bruto de similaridade produzido pelo modelo."}
    )

    same_person: Optional[bool] = field(
        default=None,
        metadata={"help": "Ground truth da comparação: True, False ou None."}
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Metadados adicionais associados à amostra de score."}
    )

    def __post_init__(self) -> None:
        """
        Se same_person não for informado, tenta inferir a partir das identidades.
        """
        if self.same_person is None:
            id1 = self.probe.identity
            id2 = self.reference.identity

            if id1 is not None and id2 is not None:
                self.same_person = (id1 == id2)
            else:
                self.same_person = False

    @property
    def label(self) -> int:
        """
        Retorna o ground truth no formato binário:
        - 1 para genuíno / Hp
        - 0 para impostor / Hd
        """
        return 1 if self.same_person else 0

    @property
    def has_ground_truth(self) -> bool:
        """
        Retorna True se a amostra possui ground truth explícito ou inferido.
        """
        return self.same_person is not None

    def to_dict(self) -> Dict[str, Any]:
        """
        Retorna dicionário achatado para DataFrame, exportação e auditoria.
        """
        return {
            "probe_file_name": self.probe.file_name,
            "probe_image_path": self.probe.image_path,
            "probe_face_hash": self.probe.face_hash,
            "probe_identity": self.probe.identity,

            "reference_file_name": self.reference.file_name,
            "reference_image_path": self.reference.image_path,
            "reference_face_hash": self.reference.face_hash,
            "reference_identity": self.reference.identity,

            "same_person": self.same_person,
            "label": self.label,
            "raw_score": float(self.raw_score),

            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói a amostra a partir de dicionário.

        Espera que probe/reference sejam informados como subdicionários ou
        objetos EmbeddingInfo já hidratados.
        """
        if not data:
            return None

        processed_data = data.copy()

        probe_data = processed_data.get("probe")
        if isinstance(probe_data, dict):
            processed_data["probe"] = EmbeddingInfo.from_dict(probe_data)

        reference_data = processed_data.get("reference")
        if isinstance(reference_data, dict):
            processed_data["reference"] = EmbeddingInfo.from_dict(reference_data)

        return cls(**processed_data)