# faceaudit/models/storage/strategy_artifact_reference.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar, Any

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType


T = TypeVar("T", bound="StrategyArtifactReference")


@dataclass
class StrategyArtifactReference(JSONSerializableMixin):
    """
    DTO que representa a referência lógica a um artefato persistido
    produzido ou consumido pela nova arquitetura de estratégias.

    Ele não substitui diretamente o ProjectFile legado, mas cumpre papel
    semelhante na nova camada semântica, registrando:
    - o tipo do artefato
    - o caminho relativo/identificador físico
    - o hash de integridade
    """

    artifact_type: StrategyArtifactType = field(
        metadata={"help": "Tipo semântico do artefato na arquitetura de estratégia."}
    )
    path: str = field(
        metadata={"help": "Caminho relativo ou identificador físico do artefato persistido."}
    )
    hash: Optional[str] = field(
        default=None,
        metadata={"help": "Hash de integridade do artefato, se disponível."}
    )

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> Optional[T]:
        """
        Reconstrói o DTO, convertendo corretamente o enum de tipo de artefato.

        Normaliza o separador de caminho: paths gravados no Windows usam '\\'
        e são interpretados como nome de arquivo único pelo pathlib em
        macOS/Linux. Convertemos para '/' (aceito por todos os SO) para que
        projetos sejam portáveis entre plataformas.
        """
        if not data:
            return None

        processed_data = data.copy()

        if "artifact_type" in processed_data and isinstance(processed_data["artifact_type"], str):
            processed_data["artifact_type"] = StrategyArtifactType.from_str(
                processed_data["artifact_type"]
            )

        if "path" in processed_data and isinstance(processed_data["path"], str):
            processed_data["path"] = processed_data["path"].replace("\\", "/")

        return cls(**processed_data)