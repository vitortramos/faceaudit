# faceaudit/enums/global_library_file_type.py

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from faceaudit.enums.global_library_folder import GlobalLibraryFolder
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType


class GlobalLibraryFileType(Enum):
    """
    Mapeia cada tipo de artefato da biblioteca global para sua localização física.

    Análogo a StrategyFileType, mas para recursos reutilizáveis entre projetos
    (datasets, split_configs, mining_configs, strategy_templates).

    Cada membro define:
    - extensão do arquivo
    - subpasta dentro da raiz da biblioteca (GlobalLibraryFolder)
    - template de nomenclatura com variáveis nomeadas

    Usado por GlobalLibraryStorage.
    """

    GLOBAL_DATASET = (
        ".parquet",
        GlobalLibraryFolder.DATASETS,
        "{dataset_id}",
    )

    SPLIT_CONFIG = (
        ".json",
        GlobalLibraryFolder.SPLIT_CONFIGS,
        "{config_id}",
    )

    MINING_CONFIG = (
        ".json",
        GlobalLibraryFolder.MINING_CONFIGS,
        "{config_id}",
    )

    PIPELINE_CONFIG = (
        ".json",
        GlobalLibraryFolder.PIPELINE_CONFIGS,
        "{config_id}",
    )

    STRATEGY_TEMPLATE = (
        ".json",
        GlobalLibraryFolder.STRATEGY_TEMPLATES,
        "{template_id}",
    )

    # PR I4 — pipeline de processamento de imagens (blur/JPEG/WebP/crop/oclusão).
    # Salvo via Editor de Pipeline de Imagens no Dataset Manager.
    IMAGE_PIPELINE_CONFIG = (
        ".json",
        GlobalLibraryFolder.IMAGE_PIPELINES,
        "{config_id}",
    )

    # FilterConfig polimórfico — uma lista de ColumnFilter aplicada via
    # `DatasetFilterService.apply(...)`. Salvo via diálogo de Filtro do
    # Dataset Manager (botão Salvar / Salvar Como).
    FILTER_CONFIG = (
        ".json",
        GlobalLibraryFolder.FILTER_CONFIGS,
        "{config_id}",
    )

    def __init__(self, extension: str, folder: GlobalLibraryFolder, name_template: str):
        self.extension = extension
        self.folder = folder
        self.name_template = name_template

    # =========================================================================
    # RESOLUÇÃO DE CAMINHOS
    # =========================================================================

    def build_path(self, **kwargs) -> Path:
        """
        Gera o caminho relativo (a partir da raiz da biblioteca) para este tipo.

        Exemplos:
            GlobalLibraryFileType.SPLIT_CONFIG.build_path(config_id="70_30_id")
            # → Path("split_configs/70_30_id.json")

            GlobalLibraryFileType.GLOBAL_DATASET.build_path(dataset_id="ds_passaportes")
            # → Path("datasets/ds_passaportes.parquet")

        Raises:
            ValueError: Se uma variável obrigatória do template estiver ausente.
        """
        try:
            file_name = self.name_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Variável {e} ausente para montar o caminho de '{self.name}'. "
                f"Template esperado: '{self.name_template}'."
            )

        full_name = (
            file_name
            if file_name.endswith(self.extension)
            else f"{file_name}{self.extension}"
        )

        if not self.folder.value:
            return Path(full_name)

        return Path(self.folder.value) / full_name

    # =========================================================================
    # LOOKUP POR StrategyArtifactType
    # =========================================================================

    @classmethod
    def for_artifact(cls, artifact_type: StrategyArtifactType) -> "GlobalLibraryFileType":
        """
        Retorna o GlobalLibraryFileType correspondente ao tipo de artefato global.

        Raises:
            ValueError: Se o tipo não tiver mapeamento na biblioteca global.
        """
        result = cls._artifact_map().get(artifact_type)
        if result is None:
            raise ValueError(
                f"Nenhum GlobalLibraryFileType mapeado para StrategyArtifactType.{artifact_type.name}. "
                "Este tipo de artefato não pertence à biblioteca global."
            )
        return result

    @classmethod
    def for_artifact_safe(
        cls, artifact_type: StrategyArtifactType
    ) -> Optional["GlobalLibraryFileType"]:
        """
        Versão tolerante de for_artifact() — retorna None se não houver mapeamento.
        """
        return cls._artifact_map().get(artifact_type)

    @classmethod
    def _artifact_map(cls) -> dict:
        return {
            StrategyArtifactType.GLOBAL_DATASET:        cls.GLOBAL_DATASET,
            StrategyArtifactType.SPLIT_CONFIG:           cls.SPLIT_CONFIG,
            StrategyArtifactType.MINING_CONFIG:          cls.MINING_CONFIG,
            StrategyArtifactType.PIPELINE_CONFIG:        cls.PIPELINE_CONFIG,
            StrategyArtifactType.IMAGE_PIPELINE_CONFIG:  cls.IMAGE_PIPELINE_CONFIG,
            StrategyArtifactType.FILTER_CONFIG:          cls.FILTER_CONFIG,
            StrategyArtifactType.STRATEGY_TEMPLATE:      cls.STRATEGY_TEMPLATE,
        }
