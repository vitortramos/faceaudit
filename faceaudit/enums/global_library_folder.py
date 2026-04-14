# faceaudit/enums/global_library_folder.py

from enum import Enum


class GlobalLibraryFolder(Enum):
    """
    Define a taxonomia de diretórios da biblioteca global de recursos reutilizáveis.

    A biblioteca global é uma pasta da máquina (configurada em config.json) que
    armazena recursos compartilhados entre projetos FaceAudit. Cada subpasta
    corresponde a um tipo semântico de recurso.

    Diferente de ProjectFolder (que vive dentro de cada projeto autocontido),
    GlobalLibraryFolder vive num local fixo da máquina do perito.
    """

    ROOT = ""                               # Raiz da biblioteca global
    DATASETS = "datasets"                   # Datasets .parquet reutilizáveis
    SPLIT_CONFIGS = "split_configs"         # Configurações de split (SplitConfig JSON)
    MINING_CONFIGS = "mining_configs"       # Configurações de mineração de pares (ComparisonMiningConfig JSON)
    STRATEGY_TEMPLATES = "strategy_templates"  # Templates de estratégia (LRStrategyDefinition JSON)

    def __str__(self) -> str:
        return self.value
