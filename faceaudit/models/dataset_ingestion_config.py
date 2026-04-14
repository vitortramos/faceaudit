from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from faceaudit.models.dir_template import DirTemplate

@dataclass
class DatasetIngestionConfig:
    """
    DTO (Data Transfer Object) que contém todas as configurações 
    necessárias para criar um novo dataset via Pipeline.
    """
    # 1. Dados Básicos
    input_path: Path
    dataset_name: str
    description: str = ""
    
    # 2. Estratégia de Ingestão
    recursive: bool = False
    template: Optional[DirTemplate] = None
    
    # 3. Metadados Padrão (Fallback ou Force)
    default_identity: str = "Unknown"
    origin: str = "Local"
    default_gender: Optional[str] = None    # "Male", "Female" ou None
    default_ethnicity: Optional[str] = None
    
    # 4. Configurações de Processamento (Flags)
    compute_ff: bool = True      # ForensicFace (Embeddings)
    compute_ofiq: bool = True    # Qualidade ISO
    compute_serfiq: bool = True  # Qualidade Estocástica
    
    def __post_init__(self):
        # Garante que o path seja Path object
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)