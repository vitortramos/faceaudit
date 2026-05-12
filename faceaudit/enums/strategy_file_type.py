# faceaudit/enums/strategy_file_type.py

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from faceaudit.enums.project_folder import ProjectFolder
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType


class StrategyFileType(Enum):
    """
    Mapeia cada tipo de artefato da nova arquitetura para sua localização física
    dentro de um projeto autocontido (Pendrive Paradigm).

    Cada membro define:
    - a extensão do arquivo
    - a subpasta dentro da raiz do projeto (ProjectFolder)
    - o template de nomenclatura com variáveis nomeadas

    Exclusivo da nova arquitetura — não depende de FileSystemService nem de
    ProjectFileType. Usado por LocalProjectArtifactStorage.
    """

    # --- DEFINIÇÃO DA ESTRATÉGIA ---
    STRATEGY_DEFINITION = (
        ".json",
        ProjectFolder.DEFINITIONS,
        "{strategy_id}_definition",
    )

    # --- MODELO TREINADO FINAL ---
    TRAINED_STRATEGY_MODEL = (
        ".pkl",
        ProjectFolder.MODELS,
        "{strategy_id}_trained_model",
    )

    # --- ARTEFATOS INTERMEDIÁRIOS ---
    TRAINED_PIPELINE = (
        ".pkl",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_{step_id}_pipeline",
    )
    TRAINED_SIMILARITY_MODEL = (
        ".pkl",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_similarity_model",
    )
    TRAINED_CALIBRATION_MODEL = (
        ".pkl",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_calibration_model",
    )

    # --- DATASETS ---
    # Path físico pra DATASETs DERIVADOS (produzidos por instrução).
    # Distinto de SOURCE_DATASET (cópia do dataset importado, nome
    # escolhido pelo usuário) — ver mapping em `for_artifact`.
    DERIVED_DATASET = (
        ".parquet",
        ProjectFolder.DATASETS,
        "{strategy_id}_{resource_id}",
    )
    TRANSFORMED_DATASET = (
        ".parquet",
        ProjectFolder.DATASETS,
        "{strategy_id}_{resource_id}_transformed",
    )
    SPLIT_RESULT = (
        ".parquet",
        ProjectFolder.DATASETS,
        "{strategy_id}_{resource_id}_split",
    )
    MERGE_RESULT = (
        ".parquet",
        ProjectFolder.DATASETS,
        "{strategy_id}_{resource_id}_merged",
    )

    # --- LOGS E MÉTRICAS ---
    EXECUTION_LOG = (
        ".log",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_execution_log",
    )
    METRICS = (
        ".json",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_metrics",
    )

    # --- VISUALIZAÇÕES E TRAJETÓRIA ---
    TRAJECTORY_DATA = (
        ".pkl",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_{resource_id}",
    )
    SIMILARITY_SCORES = (
        ".parquet",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_{resource_id}",
    )
    CALIBRATED_SCORES = (
        ".parquet",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_{resource_id}",
    )
    VALIDATION_RESULT = (
        ".pkl",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_{resource_id}",
    )
    CASEWORK_VISUALIZATION = (
        ".pkl",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_{resource_id}",
    )
    HUBNESS_REPORT = (
        ".pkl",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_{resource_id}",
    )
    HUBNESS_ABLATION_RESULT = (
        ".pkl",
        ProjectFolder.ARTIFACTS,
        "{strategy_id}_{resource_id}",
    )
    GRAPHIC = (
        ".png",
        ProjectFolder.GRAPHICS,
        "{strategy_id}_{graphic_name}",
    )

    # --- CHECKPOINTS ---
    CHECKPOINT = (
        ".ckpt",
        ProjectFolder.CHECKPOINTS,
        "{strategy_id}_checkpoint",
    )

    # --- RECURSOS IMPORTADOS DA BIBLIOTECA GLOBAL ---
    # Usados quando copy_artifact_to_project() recebe um tipo de biblioteca global.
    SOURCE_DATASET = (
        ".parquet",
        ProjectFolder.DATASETS,
        "{dataset_id}",
    )
    PROJECT_SPLIT_CONFIG = (
        ".json",
        ProjectFolder.SPLIT_CONFIGS,
        "{config_id}",
    )
    PROJECT_MINING_CONFIG = (
        ".json",
        ProjectFolder.MINING_CONFIGS,
        "{config_id}",
    )
    PROJECT_PIPELINE_CONFIG = (
        ".json",
        ProjectFolder.PIPELINES,
        "{config_id}",
    )

    def __init__(self, extension: str, folder: ProjectFolder, name_template: str):
        self.extension = extension
        self.folder = folder
        self.name_template = name_template

    # =========================================================================
    # RESOLUÇÃO DE CAMINHOS
    # =========================================================================

    def build_path(self, **kwargs) -> Path:
        """
        Gera o caminho relativo completo (a partir da raiz do projeto) para este
        tipo de artefato, aplicando as variáveis dinâmicas ao template de nome.

        O caminho retornado é sempre RELATIVO — garantia do Pendrive Paradigm.

        Exemplos:
            StrategyFileType.STRATEGY_DEFINITION.build_path(strategy_id="ST_01")
            # → Path("definitions/ST_01_definition.json")

            StrategyFileType.TRAINED_PIPELINE.build_path(strategy_id="ST_01", step_id="pipe_a")
            # → Path("artifacts/ST_01_pipe_a_pipeline.pkl")

        Args:
            **kwargs: Variáveis necessárias para preencher o name_template.

        Returns:
            Path: Caminho relativo pronto para ser combinado com project_root.

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
    def for_artifact(cls, artifact_type: StrategyArtifactType) -> "StrategyFileType":
        """
        Retorna o StrategyFileType correspondente ao tipo de artefato informado.

        Raises:
            ValueError: Se o tipo de artefato não tiver mapeamento definido.
        """
        result = cls._artifact_map().get(artifact_type)
        if result is None:
            raise ValueError(
                f"Nenhum StrategyFileType mapeado para StrategyArtifactType.{artifact_type.name}. "
                "Adicione a entrada no método _artifact_map()."
            )
        return result

    @classmethod
    def for_artifact_safe(
        cls, artifact_type: StrategyArtifactType
    ) -> Optional["StrategyFileType"]:
        """
        Versão tolerante de for_artifact() — retorna None em vez de lançar exceção
        se o tipo não tiver mapeamento. Útil para list_artifacts().
        """
        return cls._artifact_map().get(artifact_type)

    @classmethod
    def _artifact_map(cls) -> dict:
        """
        Tabela de mapeamento StrategyArtifactType → StrategyFileType.
        Centralizada aqui para facilitar manutenção.
        """
        return {
            StrategyArtifactType.STRATEGY_DEFINITION:       cls.STRATEGY_DEFINITION,
            StrategyArtifactType.STRATEGY_MODEL:            cls.TRAINED_STRATEGY_MODEL,
            StrategyArtifactType.TRAINED_PIPELINE:          cls.TRAINED_PIPELINE,
            StrategyArtifactType.TRAINED_SIMILARITY_MODEL:  cls.TRAINED_SIMILARITY_MODEL,
            StrategyArtifactType.TRAINED_CALIBRATION_MODEL: cls.TRAINED_CALIBRATION_MODEL,
            StrategyArtifactType.DERIVED_DATASET:           cls.DERIVED_DATASET,
            StrategyArtifactType.TRANSFORMED_DATASET:       cls.TRANSFORMED_DATASET,
            StrategyArtifactType.SPLIT_RESULT:              cls.SPLIT_RESULT,
            StrategyArtifactType.MERGE_RESULT:              cls.MERGE_RESULT,
            StrategyArtifactType.EXECUTION_LOG:             cls.EXECUTION_LOG,
            StrategyArtifactType.METRICS:                   cls.METRICS,
            StrategyArtifactType.TRAJECTORY_DATA:           cls.TRAJECTORY_DATA,
            StrategyArtifactType.SIMILARITY_SCORES:         cls.SIMILARITY_SCORES,
            StrategyArtifactType.CALIBRATED_SCORES:         cls.CALIBRATED_SCORES,
            StrategyArtifactType.VALIDATION_RESULT:         cls.VALIDATION_RESULT,
            StrategyArtifactType.CASEWORK_VISUALIZATION:    cls.CASEWORK_VISUALIZATION,
            StrategyArtifactType.HUBNESS_REPORT:            cls.HUBNESS_REPORT,
            StrategyArtifactType.HUBNESS_ABLATION_RESULT:   cls.HUBNESS_ABLATION_RESULT,
            StrategyArtifactType.GRAPHIC:                   cls.GRAPHIC,
            StrategyArtifactType.CHECKPOINT:                cls.CHECKPOINT,
            # Tipos de biblioteca global → destino dentro do projeto
            StrategyArtifactType.GLOBAL_DATASET:            cls.SOURCE_DATASET,
            StrategyArtifactType.SPLIT_CONFIG:              cls.PROJECT_SPLIT_CONFIG,
            StrategyArtifactType.MINING_CONFIG:             cls.PROJECT_MINING_CONFIG,
            StrategyArtifactType.PIPELINE_CONFIG:           cls.PROJECT_PIPELINE_CONFIG,
            # StrategyArtifactType.UNKNOWN e STRATEGY_TEMPLATE não têm mapeamento intencional
        }
