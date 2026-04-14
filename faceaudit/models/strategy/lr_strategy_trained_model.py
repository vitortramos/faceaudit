# faceaudit/models/strategy/lr_strategy_trained_model.py

from __future__ import annotations
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.models.strategy.strategy_resource import StrategyResource


T = TypeVar("T", bound="LRStrategyTrainedModel")


@dataclass
class LRStrategyTrainedModel(JSONSerializableMixin):
    """
    Representa o modelo treinado final da estratégia.

    Este objeto materializa o resultado da preparação da estratégia e pode
    conter:
    - recursos treinados/finais
    - artefatos físicos/lógicos produzidos
    - metadados consolidados da execução

    Ele não executa a estratégia; apenas representa o resultado persistível
    da sua preparação.
    """

    strategy_id: str
    trained_resources: List[StrategyResource] = field(default_factory=list)
    artifacts: List[StrategyArtifactReference] = field(default_factory=list)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_resource_by_id(self, resource_id: str) -> Optional[StrategyResource]:
        """
        Retorna o recurso treinado correspondente ao identificador informado.
        """
        for resource in self.trained_resources:
            if resource.resource_id == resource_id:
                return resource
        return None

    def get_artifact_by_path(self, path: str) -> Optional[StrategyArtifactReference]:
        """
        Retorna o artefato correspondente ao caminho informado.
        """
        for artifact in self.artifacts:
            if artifact.path == path:
                return artifact
        return None

    def add_trained_resource(self, resource: StrategyResource) -> None:
        """
        Registra um recurso treinado no modelo da estratégia.

        Se já existir um recurso com o mesmo resource_id, ele será substituído.
        """
        if resource is None:
            raise ValueError("O recurso treinado não pode ser nulo.")

        for index, current in enumerate(self.trained_resources):
            if current.resource_id == resource.resource_id:
                self.trained_resources[index] = resource
                return

        self.trained_resources.append(resource)

    def add_artifact(self, artifact: StrategyArtifactReference) -> None:
        """
        Registra um artefato associado ao modelo da estratégia.

        Se já existir um artefato com o mesmo path, ele será substituído.
        """
        if artifact is None:
            raise ValueError("O artefato não pode ser nulo.")

        for index, current in enumerate(self.artifacts):
            if current.path == artifact.path:
                self.artifacts[index] = artifact
                return

        self.artifacts.append(artifact)

    def set_execution_metadata(self, key: str, value: Any) -> None:
        """
        Registra ou atualiza um metadado de execução.
        """
        self.execution_metadata[key] = value

    def get_execution_metadata(self, key: str, default: Any = None) -> Any:
        """
        Recupera um metadado de execução.
        """
        return self.execution_metadata.get(key, default)

    def save(self, path: Union[str, Path]) -> None:
        """
        Persiste o modelo treinado da estratégia em disco.

        Nesta primeira versão, a persistência é feita por pickle do próprio DTO.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls: Type[T], path: Union[str, Path]) -> T:
        """
        Restaura o modelo treinado da estratégia a partir do disco.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Modelo treinado da estratégia não encontrado: {file_path}")

        with open(file_path, "rb") as f:
            loaded = pickle.load(f)

        if not isinstance(loaded, cls):
            raise TypeError(
                f"O arquivo carregado não contém uma instância de {cls.__name__}."
            )

        return loaded

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO, hidratando corretamente:
        - trained_resources
        - artifacts
        """
        if not data:
            return None

        processed_data = data.copy()

        if "trained_resources" in processed_data and isinstance(processed_data["trained_resources"], list):
            processed_data["trained_resources"] = [
                StrategyResource.from_dict(item) if isinstance(item, dict) else item
                for item in processed_data["trained_resources"]
            ]

        if "artifacts" in processed_data and isinstance(processed_data["artifacts"], list):
            processed_data["artifacts"] = [
                StrategyArtifactReference.from_dict(item) if isinstance(item, dict) else item
                for item in processed_data["artifacts"]
            ]

        return cls(**processed_data)