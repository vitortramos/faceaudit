from __future__ import annotations
from enum import Enum
from pathlib import Path
from faceaudit.enums.project_folder import ProjectFolder

class ProjectFileType(Enum):
    """
    Mapeia os tipos de arquivos do sistema FaceAudit para suas respectivas
    extensões, pastas de destino e regras de nomenclatura (templates).
    
    Remove a responsabilidade de "dar nome aos arquivos" dos Managers.
    """
    # --- SISTEMA ---
    PROJECT_CONFIG = (".fap", ProjectFolder.ROOT, "project_config")
    
    # --- DADOS E PIPELINES ---
    DATASET_METADATA = (".parquet", ProjectFolder.DATASETS, "{dataset_name}")
    PIPELINE_CONFIG = (".json", ProjectFolder.PIPELINES, "{pipeline_name}")
    MINING_CONFIG = (".json", ProjectFolder.MINING_CONFIGS, "{config_name}")
    SPLIT_CONFIG = (".json", ProjectFolder.SPLIT_CONFIGS, "{config_name}")
    
    # --- ESTRATÉGIAS ---
    TRAINED_STRATEGY = (".pkl", ProjectFolder.STRATEGIES, "{strategy_id}_model")

    
    # --- ARTEFATOS E CHECKPOINTS ---
    MODEL_CHECKPOINT = (".ckpt", ProjectFolder.CHECKPOINTS, "{strategy_id}_checkpoint_epoch_{epoch}")
    KDE_TRAINING_ARTIFACT = (".pkl", ProjectFolder.ARTIFACTS, "{strategy_id}_kde_artifacts")
    FITTED_PIPELINE = (".pkl", ProjectFolder.ARTIFACTS, "{strategy_id}_{pipeline_name}_fitted")
    PREPARATION_STATS = (".json", ProjectFolder.ARTIFACTS, "{strategy_id}_prep_stats")
    SPLIT_SIMULATION_STATS = (".json", ProjectFolder.ARTIFACTS, "{strategy_id}_split_stats")
    EXECUTION_LOG = (".log", ProjectFolder.ARTIFACTS, "{process_name}_execution_log")
    
    # --- GRÁFICOS (VISUALIZAÇÕES) ---
    # --- TRAJETÓRIA DO PIPELINE PLDA ---
    TRAJECTORY_DATA = (".pkl", ProjectFolder.ARTIFACTS, "{strategy_id}_trajectory_data")
    TRAJECTORY_PLOT = (".png", ProjectFolder.GRAPHICS, "{strategy_id}_trajectory_step_{step}")
    # --- GRÁFICO DE DISTRIBUIÇÃO KDE ---
    DISTRIBUTION_GRAPH = (".png", ProjectFolder.GRAPHICS, "{strategy_id}_distribution_graph")
    
    # --- RELATÓRIOS DE VALIDAÇÃO (ARENA) ---
    # Guarda o DTO ValidationComparisonReport em formato binário (Pickle)
    VALIDATION_COMPARISON_DATA = (".pkl", ProjectFolder.ARTIFACTS, "validation_comparison_{report_name}")
    
    # Salva o DTO ValidationInputConfig em formato JSON (se o perito quiser salvar um "preset" de teste)
    VALIDATION_INPUT_PRESET = (".json", ProjectFolder.PIPELINES, "validation_preset_{preset_name}")
    
    # (Opcional) Se o serviço for gerar PNGs automáticos das métricas
    VALIDATION_TIPPETT_PLOT = (".png", ProjectFolder.GRAPHICS, "validation_tippett_{report_name}")
    VALIDATION_DET_PLOT = (".png", ProjectFolder.GRAPHICS, "validation_det_{report_name}")


    def __init__(self, extension: str, folder: ProjectFolder, name_template: str):
        self.extension = extension
        self.folder = folder
        self.name_template = name_template

    # =========================================================================
    # SERIALIZAÇÃO (JSON .fap)
    # =========================================================================

    def to_str(self) -> str:
        """
        Retorna o nome da constante (ex: 'DATASET_METADATA') para o JSON.
        Evita exportar a tupla complexa para o manifesto.
        """
        return self.name

    @classmethod
    def from_str(cls, value: str) -> ProjectFileType:
        """
        Reconstrói o Enum a partir do nome da constante string.
        """
        try:
            # Busca pelo nome da chave no dicionário interno do Enum
            return cls[value.upper() if value else ""]
        except KeyError:
            # Se o arquivo .fap for corrompido ou apontar para um tipo que não 
            # existe mais no software, cai de forma segura em um tipo neutro.
            return cls.EXECUTION_LOG

    # =========================================================================
    # RESOLUÇÃO DE CAMINHOS
    # =========================================================================

    def build_path(self, **kwargs) -> Path:
        """
        Gera o caminho relativo completo aplicando as variáveis dinâmicas ao template.
        
        Exemplo:
            path = ProjectFileType.TRAJECTORY_PLOT.build_path(strategy_id="ST_01", step=3)
            # Retorna: Path("graphics/ST_01_trajectory_step_3.png")
            
        Args:
            **kwargs: Variáveis necessárias para preencher o `name_template`.
            
        Returns:
            Path: O caminho relativo do arquivo pronto para ser salvo.
        """
        # Formata o nome base usando as variáveis fornecidas
        try:
            file_name = self.name_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Falta a variável {e} para montar o nome do arquivo {self.name}.")
        
        # Garante a extensão correta
        if not file_name.endswith(self.extension):
            full_name = f"{file_name}{self.extension}"
        else:
            full_name = file_name
            
        # Retorna o caminho juntando a pasta (se não for a raiz)
        if not self.folder.value:
            return Path(full_name)
            
        return Path(self.folder.value) / full_name