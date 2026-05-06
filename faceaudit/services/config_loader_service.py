# faceaudit/services/config_loader_service.py
import json
from pathlib import Path

from faceaudit.models.system_config import SystemConfig


class ConfigLoaderService:
    """Serviço que garante a integridade das configurações do sistema.

    - Assegura que `config.json` existe no disco (cria com defaults se ausente).
    - Migra automaticamente formatos antigos (paths separados, parâmetros
      operacionais de FF/SER-FIQ/PLDA inline) para o formato novo
      (apenas `paths.library`, `paths.dir_templates`, `ff.use_gpu`,
      `ofiq.*`).
    """

    # Chaves antigas que sumiram no formato novo. Detectar pelo menos uma
    # = JSON está em formato legado.
    _LEGACY_TOP_KEYS = ("serfiq", "plda")
    _LEGACY_PATH_KEYS = (
        "datasets", "pipelines", "strategies", "mining_configs",
        "split_configs", "templates", "checkpoints",
    )
    _LEGACY_FF_KEYS = ("model", "det_size")

    def __init__(self, config_file: str = "config.json"):
        if not config_file:
            config_file = "config.json"

        self.config_path = Path(config_file)
        self._ensure_config_exists()
        self._migrate_legacy_if_needed()

    # ──────────────────────────────────────────────────────────────────
    # Bootstrap
    # ──────────────────────────────────────────────────────────────────

    def _ensure_config_exists(self):
        if self.config_path.exists():
            return

        print(f"⚙️ Gerando configuração inicial em: {self.config_path.absolute()}")
        default_config = SystemConfig()

        # Resolve `library` e `dir_templates` para paths absolutos
        # baseados na raiz do projeto (onde fica o config.json).
        root = self.config_path.parent.absolute()
        default_config.paths.library = str(root / default_config.paths.library)
        default_config.paths.dir_templates = str(
            root / default_config.paths.dir_templates
        )

        self.save(default_config)

    def _migrate_legacy_if_needed(self) -> None:
        """Detecta formato antigo e regrava em formato novo. Idempotente."""
        try:
            raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return

        if not isinstance(raw, dict):
            return

        if not self._is_legacy(raw):
            return

        print("⚙️ config.json em formato legado — migrando para o novo formato.")
        # `SystemConfig.from_dict` + `PathsConfig.from_dict` já contêm a
        # lógica de migração: leem chaves antigas, derivam library/dir_templates.
        migrated = SystemConfig.from_dict(raw)
        self.save(migrated)
        print(f"   ✅ Nova library: {migrated.paths.library}")
        print(f"   ✅ Novo dir_templates: {migrated.paths.dir_templates}")

    def _is_legacy(self, raw: dict) -> bool:
        # Top-level com chaves removidas no novo formato.
        if any(k in raw for k in self._LEGACY_TOP_KEYS):
            return True

        # paths com qualquer chave antiga.
        paths = raw.get("paths") or {}
        if isinstance(paths, dict):
            if any(k in paths for k in self._LEGACY_PATH_KEYS):
                return True
            # Sem `library`? — então também é legado (mesmo sem chave antiga).
            if "library" not in paths:
                return True

        # ff com chaves operacionais antigas.
        ff = raw.get("ff") or {}
        if isinstance(ff, dict) and any(k in ff for k in self._LEGACY_FF_KEYS):
            return True

        return False

    # ──────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────

    def load(self) -> SystemConfig:
        return SystemConfig.load_from_file(str(self.config_path))

    def save(self, config: SystemConfig):
        config.save_to_file(str(self.config_path))
