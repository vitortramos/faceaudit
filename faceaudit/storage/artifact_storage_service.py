# faceaudit/storage/artifact_storage_service.py

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from faceaudit.enums.project_file_type import ProjectFileType
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol
from faceaudit.services.file_system_service import FileSystemService


class ArtifactStorageService(ArtifactStorageProtocol):
    """
    Implementação concreta da camada de persistência física de artefatos da nova arquitetura.

    Esta classe atua como adaptadora entre:
    - a semântica nova (`StrategyArtifactType`)
    - a infraestrutura legada (`FileSystemService` + `ProjectFileType`)

    A ideia é permitir evolução da nova arquitetura sem quebrar o que já funciona.
    """

    _ARTIFACT_TYPE_MAP: Dict[StrategyArtifactType, ProjectFileType] = {
        StrategyArtifactType.STRATEGY_MODEL: ProjectFileType.TRAINED_STRATEGY,
        StrategyArtifactType.TRAINED_PIPELINE: ProjectFileType.FITTED_PIPELINE,
        StrategyArtifactType.CHECKPOINT: ProjectFileType.MODEL_CHECKPOINT,
        StrategyArtifactType.EXECUTION_LOG: ProjectFileType.EXECUTION_LOG,
        StrategyArtifactType.TRAJECTORY_DATA: ProjectFileType.TRAJECTORY_DATA,
        StrategyArtifactType.GRAPHIC: ProjectFileType.TRAJECTORY_PLOT,
        StrategyArtifactType.METRICS: ProjectFileType.PREPARATION_STATS,
    }

    def __init__(self, file_system_service: FileSystemService):
        self.file_system_service = file_system_service

    def save_artifact(
        self,
        artifact_type: StrategyArtifactType,
        data: Any,
        **kwargs,
    ) -> StrategyArtifactReference:
        """
        Persiste um artefato no projeto atual.

        Casos:
        - Tipos com mapeamento legado: delega ao FileSystemService
        - STRATEGY_DEFINITION: salva explicitamente como JSON em artifacts/
        """
        if artifact_type == StrategyArtifactType.STRATEGY_DEFINITION:
            return self._save_strategy_definition_artifact(data=data, **kwargs)

        project_file_type = self._map_artifact_type(artifact_type)

        abs_path, file_hash = self.file_system_service.save_file(
            file_type=project_file_type,
            data=data,
            **kwargs,
        )

        rel_path = self._to_relative_path(abs_path)

        return StrategyArtifactReference(
            artifact_type=artifact_type,
            path=str(rel_path),
            hash=file_hash,
        )

    def copy_artifact_to_project(
        self,
        source_path: Union[str, Path],
        artifact_type: StrategyArtifactType,
        **kwargs,
    ) -> StrategyArtifactReference:
        """
        Copia um artefato existente para a estrutura do projeto atual.
        """
        if artifact_type == StrategyArtifactType.STRATEGY_DEFINITION:
            abs_path = Path(source_path)
            if not abs_path.exists():
                raise FileNotFoundError(f"Arquivo de origem não encontrado: {abs_path}")

            project_root = self.file_system_service.current_project_root
            if project_root is None:
                raise RuntimeError("Raiz do projeto não definida no FileSystemService.")

            strategy_id = kwargs.get("strategy_id")
            if not strategy_id:
                raise ValueError(
                    "É obrigatório informar 'strategy_id' para copiar STRATEGY_DEFINITION."
                )

            dest_path = project_root / "artifacts" / f"{strategy_id}_definition.json"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(abs_path.read_bytes())

            file_hash = self.file_system_service.hash_service.calculate_file_hash(dest_path)

            return StrategyArtifactReference(
                artifact_type=artifact_type,
                path=str(self._to_relative_path(dest_path)),
                hash=file_hash,
            )

        project_file_type = self._map_artifact_type(artifact_type)

        abs_path, file_hash = self.file_system_service.copy_file_to_project(
            src_path=Path(source_path),
            file_type=project_file_type,
            **kwargs,
        )

        rel_path = self._to_relative_path(abs_path)

        return StrategyArtifactReference(
            artifact_type=artifact_type,
            path=str(rel_path),
            hash=file_hash,
        )

    def resolve_artifact(
        self,
        artifact_ref: StrategyArtifactReference,
    ) -> Optional[Path]:
        """
        Resolve a referência lógica do artefato para caminho físico.

        Nesta primeira versão:
        - se o path já existir relativo ao projeto atual, retorna o caminho completo
        - se o path for absoluto e existir, retorna diretamente
        """
        if artifact_ref is None or not artifact_ref.path:
            return None

        candidate = Path(artifact_ref.path)

        if candidate.is_absolute() and candidate.exists():
            return candidate

        project_root = self.file_system_service.current_project_root
        if project_root is not None:
            project_candidate = project_root / candidate
            if project_candidate.exists():
                return project_candidate

        return None

    def list_artifacts(
        self,
        artifact_type: Optional[StrategyArtifactType] = None,
    ) -> List[StrategyArtifactReference]:
        """
        Lista artefatos disponíveis.

        Observações:
        - para STRATEGY_DEFINITION, usa convenção própria em artifacts/
        - para tipos mapeados no legado, tenta reutilizar list_available_files()
        """
        artifact_types = (
            [artifact_type] if artifact_type is not None else [*self._ARTIFACT_TYPE_MAP.keys(), StrategyArtifactType.STRATEGY_DEFINITION]
        )

        found: List[StrategyArtifactReference] = []

        for current_type in artifact_types:
            if current_type == StrategyArtifactType.STRATEGY_DEFINITION:
                found.extend(self._list_strategy_definition_artifacts())
                continue

            project_file_type = self._ARTIFACT_TYPE_MAP.get(current_type)
            if project_file_type is None:
                continue

            try:
                global_names, local_names = self.file_system_service.list_available_files(
                    file_type=project_file_type
                )
            except Exception:
                continue

            all_names = list(dict.fromkeys(local_names + global_names))

            for name in all_names:
                resolved = self.file_system_service.resolve_file(
                    file_name=name,
                    file_type=project_file_type,
                )
                if resolved is None:
                    continue

                found.append(
                    StrategyArtifactReference(
                        artifact_type=current_type,
                        path=str(self._to_relative_path_or_abs(resolved)),
                        hash=None,
                    )
                )

        return found

    # =========================================================================
    # AUXILIARES ESPECÍFICOS
    # =========================================================================

    def _save_strategy_definition_artifact(
        self,
        data: Any,
        **kwargs,
    ) -> StrategyArtifactReference:
        """
        Salva a definição declarativa da estratégia como JSON em artifacts/.
        """
        project_root = self.file_system_service.current_project_root
        if project_root is None:
            raise RuntimeError("Raiz do projeto não definida no FileSystemService.")

        strategy_id = kwargs.get("strategy_id")
        if not strategy_id:
            raise ValueError(
                "É obrigatório informar 'strategy_id' para salvar STRATEGY_DEFINITION."
            )

        abs_path = project_root / "artifacts" / f"{strategy_id}_definition.json"
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        payload = data.to_dict() if hasattr(data, "to_dict") else data
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)

        file_hash = self.file_system_service.hash_service.calculate_file_hash(abs_path)

        return StrategyArtifactReference(
            artifact_type=StrategyArtifactType.STRATEGY_DEFINITION,
            path=str(self._to_relative_path(abs_path)),
            hash=file_hash,
        )

    def _list_strategy_definition_artifacts(self) -> List[StrategyArtifactReference]:
        """
        Lista definições declarativas de estratégia encontradas em artifacts/.
        """
        project_root = self.file_system_service.current_project_root
        if project_root is None:
            raise RuntimeError("Raiz do projeto não definida no FileSystemService.")

        artifacts_dir = project_root / "artifacts"
        if not artifacts_dir.exists():
            return []

        found: List[StrategyArtifactReference] = []

        for path in artifacts_dir.glob("*_definition.json"):
            found.append(
                StrategyArtifactReference(
                    artifact_type=StrategyArtifactType.STRATEGY_DEFINITION,
                    path=str(self._to_relative_path(path)),
                    hash=None,
                )
            )

        return found

    # =========================================================================
    # AUXILIARES GERAIS
    # =========================================================================

    def _map_artifact_type(self, artifact_type: StrategyArtifactType) -> ProjectFileType:
        project_file_type = self._ARTIFACT_TYPE_MAP.get(artifact_type)
        if project_file_type is None:
            raise ValueError(
                f"Não existe mapeamento entre StrategyArtifactType '{artifact_type}' "
                "e ProjectFileType no ArtifactStorageService."
            )
        return project_file_type

    def _to_relative_path(self, abs_path: Path) -> Path:
        """
        Tenta converter um caminho absoluto em relativo ao projeto atual.
        Se não for possível, retorna o próprio caminho absoluto.
        """
        project_root = self.file_system_service.current_project_root
        if project_root is None:
            return abs_path

        try:
            return abs_path.relative_to(project_root)
        except ValueError:
            return abs_path

    def _to_relative_path_or_abs(self, path: Path) -> Path:
        """
        Helper tolerante para exibição/persistência do path.
        """
        return self._to_relative_path(path)