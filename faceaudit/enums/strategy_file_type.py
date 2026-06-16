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

    # --- MODELO TREINADO FINAL (wrapper Run All) ---
    # Layout hierárquico: `models/<strategy_id>/strategy_model.pkl`.
    # Demais modelos da estratégia vivem em subpastas irmãs
    # (pipeline/similarity/calibration/typicality).
    TRAINED_STRATEGY_MODEL = (
        ".pkl",
        ProjectFolder.MODELS,
        "{strategy_folder}/strategy_model",
    )

    # --- MODELOS COMPONENTES (subpastas em models/<strategy_id>/) ---
    # Filenames combinam `{resource_id}_{id}` pra evitar colisão quando
    # dois recursos compartilham nome (resource_id) — o id técnico
    # (`TP-abcd1234`, `TS-...`, etc.) é único globalmente.
    TRAINED_PIPELINE = (
        ".pkl",
        ProjectFolder.MODELS,
        "{strategy_folder}/pipeline/{step_id}_{id}",
    )
    TRAINED_SIMILARITY_MODEL = (
        ".pkl",
        ProjectFolder.MODELS,
        "{strategy_folder}/similarity/{resource_id}_{id}",
    )
    TRAINED_CALIBRATION_MODEL = (
        ".pkl",
        ProjectFolder.MODELS,
        "{strategy_folder}/calibration/{resource_id}_{id}",
    )
    TRAINED_TYPICALITY_MODEL = (
        ".pkl",
        ProjectFolder.MODELS,
        "{strategy_folder}/typicality/{resource_id}_{id}",
    )

    # --- DATASETS DERIVADOS (datasets/<strategy_folder>/) ---
    # Filename `{resource_id}_{id}` — `_{id}` desambigua nomes; subpasta
    # `<strategy_folder>` agrupa derivados por estratégia. Mesma convenção
    # de models/, scores/, results/.
    # Distinto de SOURCE_DATASET (importados pelo usuário, ficam em
    # `datasets/primary/`).
    DERIVED_DATASET = (
        ".parquet",
        ProjectFolder.DATASETS,
        "{strategy_folder}/{resource_id}_{id}",
    )
    TRANSFORMED_DATASET = (
        ".parquet",
        ProjectFolder.DATASETS,
        "{strategy_folder}/{resource_id}_{id}_transformed",
    )
    SPLIT_RESULT = (
        ".parquet",
        ProjectFolder.DATASETS,
        "{strategy_folder}/{resource_id}_{id}_split",
    )
    MERGE_RESULT = (
        ".parquet",
        ProjectFolder.DATASETS,
        "{strategy_folder}/{resource_id}_{id}_merged",
    )

    # --- LOGS E MÉTRICAS (results/<strategy_id>/) ---
    EXECUTION_LOG = (
        ".log",
        ProjectFolder.RESULTS,
        "{strategy_folder}/execution_log",
    )
    METRICS = (
        ".json",
        ProjectFolder.RESULTS,
        "{strategy_folder}/metrics",
    )

    # --- SCORE DATASETS (scores/<strategy_id>/<subtipo>/) ---
    # Filename: `{resource_id}_{id}` — disambiguação igual aos modelos.
    SIMILARITY_SCORES = (
        ".parquet",
        ProjectFolder.SCORES,
        "{strategy_folder}/raw/{resource_id}_{id}",
    )
    MINED_SCORES = (
        ".parquet",
        ProjectFolder.SCORES,
        "{strategy_folder}/mined/{resource_id}_{id}",
    )
    VECTORIZED_SCORES = (
        ".parquet",
        ProjectFolder.SCORES,
        "{strategy_folder}/vectorized/{resource_id}_{id}",
    )
    CALIBRATED_SCORES = (
        ".parquet",
        ProjectFolder.SCORES,
        "{strategy_folder}/calibrated/{resource_id}_{id}",
    )

    # --- RESULTADOS DE ANÁLISE (results/<strategy_id>/<tipo>/) ---
    TRAJECTORY_DATA = (
        ".pkl",
        ProjectFolder.RESULTS,
        "{strategy_folder}/trajectory/{resource_id}_{id}",
    )
    VALIDATION_RESULT = (
        ".pkl",
        ProjectFolder.RESULTS,
        "{strategy_folder}/validation/{resource_id}_{id}",
    )
    CASEWORK_VISUALIZATION = (
        ".pkl",
        ProjectFolder.RESULTS,
        "{strategy_folder}/casework/{resource_id}_{id}",
    )
    HUBNESS_REPORT = (
        ".pkl",
        ProjectFolder.RESULTS,
        "{strategy_folder}/hubness/{resource_id}_{id}",
    )
    HUBNESS_ABLATION_RESULT = (
        ".pkl",
        ProjectFolder.RESULTS,
        "{strategy_folder}/hubness/{resource_id}_{id}_ablation",
    )
    LR_ARCHITECTURE_RAW_INPUT = (
        ".pkl",
        ProjectFolder.SCORES,
        "{strategy_folder}/raw_input/{resource_id}_{id}",
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
    # PRIMARY datasets ficam em `datasets/primary/<dataset_id>.parquet` —
    # subfolder dedicado separa visualmente do que é DERIVADO (em
    # `datasets/<strategy_folder>/`).
    SOURCE_DATASET = (
        ".parquet",
        ProjectFolder.DATASETS,
        "primary/{dataset_id}",
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
    PROJECT_VALIDATION_RULE_TEMPLATE = (
        ".json",
        ProjectFolder.VALIDATION_RULES,
        "{template_id}",
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

            StrategyFileType.TRAINED_PIPELINE.build_path(
                strategy_id="ST_01", step_id="pipe_a", id="TP-abc12345",
                strategy_folder="My Strategy_ST_01",
            )
            # → Path("models/My Strategy_ST_01/pipeline/pipe_a_TP-abc12345.pkl")

        Auto-fill: se o template contém `{strategy_folder}` e o caller não
        passou esse kwarg, deriva via `strategy_folder_name(strategy_id)`
        (fallback sem nome, vira só `<strategy_id>`). Permite que callers
        legados que só conhecem `strategy_id` continuem funcionando.

        Args:
            **kwargs: Variáveis necessárias para preencher o name_template.

        Returns:
            Path: Caminho relativo pronto para ser combinado com project_root.

        Raises:
            ValueError: Se uma variável obrigatória do template estiver ausente.
        """
        # Auto-fill `strategy_folder` quando o template precisa e caller
        # não passou explicit (caso comum: repositórios que só recebem
        # `strategy_id`).
        if (
            "{strategy_folder}" in self.name_template
            and "strategy_folder" not in kwargs
        ):
            from faceaudit.models.strategy.strategy_folder import strategy_folder_name
            kwargs["strategy_folder"] = strategy_folder_name(
                kwargs.get("strategy_id", ""),
                kwargs.get("strategy_name", ""),
            )

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
            StrategyArtifactType.TRAINED_TYPICALITY_MODEL:  cls.TRAINED_TYPICALITY_MODEL,
            StrategyArtifactType.DERIVED_DATASET:           cls.DERIVED_DATASET,
            StrategyArtifactType.TRANSFORMED_DATASET:       cls.TRANSFORMED_DATASET,
            StrategyArtifactType.SPLIT_RESULT:              cls.SPLIT_RESULT,
            StrategyArtifactType.MERGE_RESULT:              cls.MERGE_RESULT,
            StrategyArtifactType.EXECUTION_LOG:             cls.EXECUTION_LOG,
            StrategyArtifactType.METRICS:                   cls.METRICS,
            StrategyArtifactType.TRAJECTORY_DATA:           cls.TRAJECTORY_DATA,
            StrategyArtifactType.SIMILARITY_SCORES:         cls.SIMILARITY_SCORES,
            StrategyArtifactType.MINED_SCORES:              cls.MINED_SCORES,
            StrategyArtifactType.VECTORIZED_SCORES:         cls.VECTORIZED_SCORES,
            StrategyArtifactType.CALIBRATED_SCORES:         cls.CALIBRATED_SCORES,
            StrategyArtifactType.VALIDATION_RESULT:         cls.VALIDATION_RESULT,
            StrategyArtifactType.CASEWORK_VISUALIZATION:    cls.CASEWORK_VISUALIZATION,
            StrategyArtifactType.HUBNESS_REPORT:            cls.HUBNESS_REPORT,
            StrategyArtifactType.HUBNESS_ABLATION_RESULT:   cls.HUBNESS_ABLATION_RESULT,
            StrategyArtifactType.LR_ARCHITECTURE_RAW_INPUT: cls.LR_ARCHITECTURE_RAW_INPUT,
            StrategyArtifactType.GRAPHIC:                   cls.GRAPHIC,
            StrategyArtifactType.CHECKPOINT:                cls.CHECKPOINT,
            # Tipos de biblioteca global → destino dentro do projeto
            StrategyArtifactType.GLOBAL_DATASET:            cls.SOURCE_DATASET,
            StrategyArtifactType.SPLIT_CONFIG:              cls.PROJECT_SPLIT_CONFIG,
            StrategyArtifactType.MINING_CONFIG:             cls.PROJECT_MINING_CONFIG,
            StrategyArtifactType.PIPELINE_CONFIG:           cls.PROJECT_PIPELINE_CONFIG,
            StrategyArtifactType.VALIDATION_RULE_TEMPLATE:  cls.PROJECT_VALIDATION_RULE_TEMPLATE,
            # StrategyArtifactType.UNKNOWN e STRATEGY_TEMPLATE não têm mapeamento intencional
        }
