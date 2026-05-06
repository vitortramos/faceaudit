from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


class EmbeddingInfo:
    """Modelo simplificado contendo as informações essenciais para indexação,
    busca vetorial e pipelines de transformação.

    Pose vive em um único dict (`pose`) análogo a `quality_scores`. Isso é o
    análogo runtime do schema-by-role do `column_catalog`: cada entrada do
    dict é uma "axis-key → value" sem comprometer o modelo a um conjunto fixo
    de eixos. `yaw` e `pitch` continuam acessíveis como properties que
    delegam ao dict — backward-compat para os ~20 callers existentes.
    """

    __slots__ = (
        "embedding", "face_hash", "image_path", "identity", "file_name",
        "source", "origin", "face_image", "pose", "quality_scores",
    )

    def __init__(
        self,
        embedding: np.ndarray,
        face_hash: str,
        image_path: str,
        identity: Optional[str],
        file_name: str,
        source: Optional[str] = None,
        origin: Optional[str] = None,
        face_image: Optional[np.ndarray] = None,
        # Pose: ou via dict (preferido) ou via yaw/pitch kwargs (legacy).
        # Quando ambos são passados, valores explícitos no dict ganham.
        pose: Optional[Dict[str, float]] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        quality_scores: Optional[Dict[str, float]] = None,
    ):
        self.embedding = embedding
        self.face_hash = face_hash
        self.image_path = image_path
        self.identity = identity
        self.file_name = file_name
        self.source = source
        self.origin = origin
        self.face_image = face_image

        # Pose: começa do dict; legacy kwargs preenchem o que faltar.
        self.pose: Dict[str, float] = dict(pose) if pose else {}
        if yaw is not None and "yaw" not in self.pose:
            self.pose["yaw"] = float(yaw)
        if pitch is not None and "pitch" not in self.pose:
            self.pose["pitch"] = float(pitch)

        self.quality_scores: Dict[str, float] = dict(quality_scores) if quality_scores else {}

    # ── Properties: pose axes acessíveis pelo nome literal ──

    @property
    def yaw(self) -> Optional[float]:
        return self.pose.get("yaw")

    @yaw.setter
    def yaw(self, v: Optional[float]) -> None:
        if v is None:
            self.pose.pop("yaw", None)
        else:
            self.pose["yaw"] = float(v)

    @property
    def pitch(self) -> Optional[float]:
        return self.pose.get("pitch")

    @pitch.setter
    def pitch(self, v: Optional[float]) -> None:
        if v is None:
            self.pose.pop("pitch", None)
        else:
            self.pose["pitch"] = float(v)

    @property
    def roll(self) -> Optional[float]:
        return self.pose.get("roll")

    @roll.setter
    def roll(self, v: Optional[float]) -> None:
        if v is None:
            self.pose.pop("roll", None)
        else:
            self.pose["roll"] = float(v)

    # ── Propriedade de conveniência (somente leitura) ──

    @property
    def quality_score(self) -> Optional[float]:
        """Retorna o 'melhor' score disponível no dicionário para uso geral.
        Prioridade: FIQA > OFIQ > SERFIQ > Genérico.
        """
        if not self.quality_scores:
            return None
        for key in ("fiqa", "ofiq", "serfiq", "generic"):
            if key in self.quality_scores:
                return self.quality_scores[key]
        return next(iter(self.quality_scores.values()))

    @property
    def size(self) -> int:
        return self.embedding.size

    @property
    def ndim(self) -> int:
        return self.embedding.ndim

    @property
    def shape(self) -> tuple:
        return self.embedding.shape

    # ── Serialização ──

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingInfo":
        """Cria uma instância a partir de um dicionário, com tratamento para
        tipos NumPy. Aceita formato novo (`pose` dict) e legado (`yaw`/`pitch`)."""
        emb = data.get("embedding")
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)

        img = data.get("face_image")
        if isinstance(img, list):
            img = np.array(img, dtype=np.uint8)

        # Quality scores — migração do legado `quality_score` único
        q_scores = data.get("quality_scores", {}) or {}
        legacy_q = data.get("quality_score")
        if legacy_q is not None and not q_scores:
            q_scores = {"generic": float(legacy_q)}

        # Pose — formato novo `pose` dict tem prioridade
        pose_payload = data.get("pose")
        pose: Dict[str, float] = {}
        if isinstance(pose_payload, dict):
            for k, v in pose_payload.items():
                if v is not None:
                    try:
                        pose[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
        # Legado: yaw/pitch como campos top-level
        legacy_yaw = data.get("yaw")
        legacy_pitch = data.get("pitch")
        if legacy_yaw is not None and "yaw" not in pose:
            try:
                pose["yaw"] = float(legacy_yaw)
            except (TypeError, ValueError):
                pass
        if legacy_pitch is not None and "pitch" not in pose:
            try:
                pose["pitch"] = float(legacy_pitch)
            except (TypeError, ValueError):
                pass

        return cls(
            embedding=emb,
            face_hash=data.get("face_hash", ""),
            image_path=data.get("image_path", ""),
            identity=data.get("identity"),
            file_name=data.get("file_name", ""),
            source=data.get("source"),
            origin=data.get("origin"),
            face_image=img,
            pose=pose,
            quality_scores=q_scores,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável.

        Emite tanto `pose` (formato novo) quanto `yaw`/`pitch` (compat com
        leitores antigos que esperam essas chaves no schema do parquet).
        """
        emb_val = self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding
        img_val = self.face_image.tolist() if isinstance(self.face_image, np.ndarray) else self.face_image

        return {
            "embedding": emb_val,
            "face_hash": self.face_hash,
            "image_path": self.image_path,
            "identity": self.identity,
            "file_name": self.file_name,
            "source": self.source,
            "origin": self.origin,
            "face_image": img_val,
            "yaw": self.pose.get("yaw"),
            "pitch": self.pose.get("pitch"),
            "pose": dict(self.pose),
            "quality_scores": dict(self.quality_scores),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def __repr__(self) -> str:
        return (f"EmbeddingInfo(face_hash={self.face_hash!r}, "
                f"identity={self.identity!r}, dim={self.size}, "
                f"pose={self.pose}, quality_scores={list(self.quality_scores.keys())})")
