# faceaudit/services/config_loader_service.py
import json
from pathlib import Path

# --- ATUALIZAÇÃO: AppConfig agora é SystemConfig ---
from faceaudit.models.system_config import SystemConfig

class ConfigLoaderService:
    """
    Serviço que garante a integridade das configurações do sistema.
    Assegura que o arquivo config.json sempre exista no disco.
    """

    def __init__(self, config_file: str = "config.json"):
        # ✅ CORREÇÃO: DI pode injetar None; evitamos Path(None)
        if not config_file:
            config_file = "config.json"

        self.config_path = Path(config_file)
        self._ensure_config_exists()

    def _ensure_config_exists(self):
        """
        Garante a existência do config.json. 
        Se for a primeira vez, gera caminhos absolutos baseados na raiz do projeto.
        """
        if not self.config_path.exists():
            print(f"⚙️ Gerando configuração inicial em: {self.config_path.absolute()}")

            # 1. Instancia com a classe atualizada
            default_config = SystemConfig()

            # 2. Transforma os caminhos padrão em absolutos baseados na raiz
            root = self.config_path.parent.absolute()

            default_config.paths.datasets = str(root / default_config.paths.datasets)
            default_config.paths.pipelines = str(root / default_config.paths.pipelines)
            default_config.paths.strategies = str(root / default_config.paths.strategies)

            # 3. Salva já com os caminhos completos
            self.save(default_config)

    def load(self) -> SystemConfig:
        """Carrega a configuração atualizada diretamente do disco."""
        # Mantemos o load_from_file pois ele já embute a lógica de fallback seguro 
        # que mantivemos para retrocompatibilidade
        return SystemConfig.load_from_file(str(self.config_path))

    def save(self, config: SystemConfig):
        """Salva as configurações no disco, garantindo a persistência."""
        # Delega para o método de compatibilidade (que por sua vez chama o Mixin de JSON)
        config.save_to_file(str(self.config_path))