import warnings
from dataclasses import dataclass, field
from typing import Type, TypeVar, Optional

# Importando o nosso salvador de vidas
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

T = TypeVar("T", bound="SystemConfig")

# --- ALERTA: Renomeado para não colidir com models.plda_config.PLDAConfig ---
@dataclass
class GlobalPLDAConfig(JSONSerializableMixin):
    latent_dim: int = 512
    use_fisher: bool = True

@dataclass
class SERFIQConfig(JSONSerializableMixin):
    t: int = 5
    alpha: float = 1.5

@dataclass
class OFIQConfig(JSONSerializableMixin):
    ofiq_exe_path: str = "bin/OFIQ/OFIQSampleApp.exe"
    ofiq_data_path: str = "bin/OFIQ/data"

@dataclass
class FFConfig(JSONSerializableMixin):
    model: str = 'sepaelv2'
    det_size: int = 320
    use_gpu: bool = False

@dataclass
class PathsConfig(JSONSerializableMixin):
    datasets: str = "data/datasets"
    pipelines: str = "data/pipelines"
    strategies: str = "data/strategies"
    templates: str = "data/templates"
    checkpoints: str = "data/checkpoints"
    mining_configs: str = "data/mining_configs" 
    split_configs: str = "data/split_configs"  # <--- [NOVO] Pasta global para regras de divisão

@dataclass
class SystemConfig(JSONSerializableMixin):
    """
    Configurações globais do sistema, caminhos de diretórios e flags de execução base.
    """
    paths: PathsConfig = field(default_factory=PathsConfig)
    serfiq: SERFIQConfig = field(default_factory=SERFIQConfig)
    ofiq: OFIQConfig = field(default_factory=OFIQConfig)
    ff: FFConfig = field(default_factory=FFConfig)
    plda: GlobalPLDAConfig = field(default_factory=GlobalPLDAConfig) 

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói a configuração aninhada delegando a injeção parcial 
        para os métodos from_dict gerados pelo JSONSerializableMixin de cada sub-classe.
        """
        if not data:
            return cls()
            
        return cls(
            paths=PathsConfig.from_dict(data.get("paths", {})),
            serfiq=SERFIQConfig.from_dict(data.get("serfiq", {})),
            ofiq=OFIQConfig.from_dict(data.get("ofiq", {})),
            ff=FFConfig.from_dict(data.get("ff", {})),
            plda=GlobalPLDAConfig.from_dict(data.get("plda", {}))
        )

    # --- MÉTODOS DE COMPATIBILIDADE (Para não quebrar o código existente) ---
    
    def save_to_file(self, file_path: str = "config.json"):
        """Mapeado para o novo método do Mixin."""
        self.to_json(file_path)

    @classmethod
    def load_from_file(cls, file_path: str = "config.json") -> 'SystemConfig':
        """Mapeado para o novo método do Mixin, garantindo retorno default em caso de falha."""
        import os
        if not os.path.exists(file_path):
            return cls()
        try:
            return cls.from_json(file_path)
        except Exception:
            return cls()

# =============================================================================
# O TRUQUE MÁGICO PARA NÃO QUEBRAR O SEU SISTEMA (ALIAS)
# =============================================================================
def __getattr__(name):
    """
    Interpela a importação. Se algum arquivo antigo tentar importar 'AppConfig',
    ele entrega o 'SystemConfig' e avisa no terminal que está na hora de atualizar.
    """
    if name == "AppConfig":
        warnings.warn(
            "A classe 'AppConfig' foi renomeada para 'SystemConfig' para manter o "
            "padrão de nomenclatura. Atualize seus imports!",
            DeprecationWarning,
            stacklevel=2
        )
        return SystemConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Para compatibilidade estática em algumas IDEs mais antigas
AppConfig = SystemConfig