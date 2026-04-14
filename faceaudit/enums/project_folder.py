from enum import Enum

class ProjectFolder(Enum):
    """
    Define a taxonomia de diretórios (pastas) de um projeto FaceAudit.
    Garante que toda a estrutura de disco do projeto seja padronizada e previsível.
    """
    ROOT = ""                  # Raiz do projeto (onde fica o .fap)
    DATASETS = "datasets"      # Arquivos .parquet (metadados e embeddings)
    IMAGES = "images"          # Cópia das fotografias originais (opcional)
    PIPELINES = "pipelines"    # Arquivos .json de transformação linear
    SPLIT_CONFIGS = "split_configs" # Arquivos .json de configurações de split
    MINING_CONFIGS = "mining_configs" # Arquivos .json de regras de curadoria/mineração
    STRATEGIES = "strategies"  # Modelos matemáticos finais (.pkl)
    REPORTS = "reports"        # Laudos ou relatórios exportados (PDF/HTML)
    CHECKPOINTS = "checkpoints"# Estados intermediários de treinamento
    ARTIFACTS = "artifacts"    # Subprodutos de dados (ex: logs, pkls secundários)
    GRAPHICS = "graphics"      # Exclusivo para imagens exportadas (plots, 3D, curvas)

    # --- NOVA ARQUITETURA ---
    DEFINITIONS = "definitions" # Definições declarativas de estratégia (LRStrategyDefinition JSON)
    MODELS = "models"           # Modelos treinados da nova arch (LRStrategyTrainedModel pkl)

    def __str__(self) -> str:
        return self.value