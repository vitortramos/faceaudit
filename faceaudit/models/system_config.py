from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

T = TypeVar("T", bound="SystemConfig")


@dataclass
class OFIQConfig(JSONSerializableMixin):
    """Caminhos do executável OFIQ e da pasta de dados/modelos."""
    ofiq_exe_path: str = "bin/OFIQ/OFIQSampleApp.exe"
    ofiq_data_path: str = "bin/OFIQ/data"

    @classmethod
    def from_dict(cls, data: dict) -> "OFIQConfig":
        if not data:
            return cls()
        # Filtra chaves desconhecidas para não quebrar com configs futuros
        # ou legacy fields que não existem mais.
        valid = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**valid)


@dataclass
class FFConfig(JSONSerializableMixin):
    """Capability de hardware do ForensicFace.

    Parâmetros operacionais (modelo, det_size, extended) foram movidos para
    `DatasetIngestionConfig` — viram escolha por execução (Novo/Enriquecer
    Dataset). Aqui só fica o que é da máquina.
    """
    use_gpu: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "FFConfig":
        if not data:
            return cls()
        # Filtra chaves desconhecidas — descarta `model`/`det_size` legados
        # silenciosamente; a migração no ConfigLoaderService já reescreveu
        # ou vai reescrever o JSON em formato novo.
        valid = {k: v for k, v in data.items() if k in cls.__annotations__}
        return cls(**valid)


@dataclass
class PathsConfig(JSONSerializableMixin):
    """Pastas globais do sistema.

    Estrutura em árvore:
        library/
            datasets/
            pipeline_configs/
            mining_configs/
            split_configs/
            strategy_templates/
            checkpoints/
        dir_templates/         (à parte — templates de ingestão)

    Apenas `library` e `dir_templates` são fields reais. As subpastas
    são acessadas via properties (`paths.datasets`, `paths.pipeline_configs`,
    etc.) que retornam `library/<sub>` — mantém retrocompat com callers
    que esperam acesso direto.
    """

    library: str = "data/library"
    dir_templates: str = "data/dir_templates"

    # ──────────────────────────────────────────────────────────────────
    # Subpastas da library (read-only, derivadas)
    # ──────────────────────────────────────────────────────────────────

    def _sub(self, name: str) -> str:
        # os.path.join preserva separador da plataforma; callers que
        # querem POSIX podem chamar Path(...).as_posix().
        return os.path.join(self.library, name)

    @property
    def datasets(self) -> str:
        return self._sub("datasets")

    @property
    def pipeline_configs(self) -> str:
        return self._sub("pipeline_configs")

    @property
    def mining_configs(self) -> str:
        return self._sub("mining_configs")

    @property
    def split_configs(self) -> str:
        return self._sub("split_configs")

    @property
    def strategy_templates(self) -> str:
        return self._sub("strategy_templates")

    @property
    def checkpoints(self) -> str:
        return self._sub("checkpoints")

    # Aliases legados — apontam pras mesmas subpastas que os nomes novos.
    # Removíveis na Fase 3 quando a arch antiga sair de cena.
    @property
    def pipelines(self) -> str:
        return self._sub("pipeline_configs")

    @property
    def strategies(self) -> str:
        return self._sub("strategy_templates")

    @property
    def templates(self) -> str:
        # Legacy alias — aponta pro `dir_templates` que é nome novo.
        return self.dir_templates

    # ──────────────────────────────────────────────────────────────────
    # Migração de config.json antigo
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "PathsConfig":
        """Aceita formato novo (`library`, `dir_templates`) e formato antigo
        (`datasets`, `pipelines`, ..., `templates`). No formato antigo,
        infere `library` da pasta-pai dos paths antigos."""
        if not data:
            return cls()

        if "library" in data:
            return cls(
                library=str(data.get("library") or "data/library"),
                dir_templates=str(
                    data.get("dir_templates") or data.get("templates") or "data/dir_templates"
                ),
            )

        # Migração: usuário tinha datasets/pipelines/strategies em paths
        # separados. Heurística — se todos compartilham um prefixo comum
        # (ex: "data/datasets", "data/pipelines" → "data/"), usamos esse
        # prefixo como library. Senão, default e o usuário ajusta no dialog.
        legacy_paths = [
            data.get(k) for k in
            ("datasets", "pipelines", "mining_configs", "split_configs", "strategies")
            if data.get(k)
        ]
        common_root = _common_parent(legacy_paths) if legacy_paths else None
        library = str(common_root or data.get("datasets") or "data/library")

        # `templates` legado vira `dir_templates`.
        dir_templates = str(data.get("templates") or "data/dir_templates")

        return cls(library=library, dir_templates=dir_templates)

    def to_dict(self) -> dict:
        # Apenas os fields reais — properties não são serializadas.
        return {
            "library": self.library,
            "dir_templates": self.dir_templates,
        }


def _common_parent(paths: list) -> str:
    """Retorna o diretório pai comum a todos os paths (string), ou
    string vazia se não houver convergência."""
    from pathlib import Path
    if not paths:
        return ""
    try:
        parts = [Path(p).resolve().parts for p in paths if p]
        if not parts:
            return ""
        # Maior prefixo comum sobre os parts.
        common = parts[0]
        for p in parts[1:]:
            limit = min(len(common), len(p))
            common = tuple(a for a, b in zip(common[:limit], p[:limit]) if a == b)
            if not common:
                return ""
        # Se todos os paths têm um nível final diferente (datasets, pipelines, ...),
        # o common_parent é o pai dele.
        return str(Path(*common))
    except Exception:
        return ""


@dataclass
class SystemConfig(JSONSerializableMixin):
    """Configurações globais — apenas o que é "ambiente da máquina".

    Parâmetros operacionais (modelo de embedding, det_size, T/alpha
    SER-FIQ, modelo PLDA) foram movidos para `DatasetIngestionConfig` ou
    para os DTOs de instrução da nova arquitetura. Aqui só sobra:
    - paths globais (library, dir_templates)
    - capability de hardware (ff.use_gpu)
    - paths para executáveis externos (ofiq.exe_path, ofiq.data_path)
    """
    paths: PathsConfig = field(default_factory=PathsConfig)
    ofiq: OFIQConfig = field(default_factory=OFIQConfig)
    ff: FFConfig = field(default_factory=FFConfig)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        if not data:
            return cls()
        return cls(
            paths=PathsConfig.from_dict(data.get("paths", {})),
            ofiq=OFIQConfig.from_dict(data.get("ofiq", {})),
            ff=FFConfig.from_dict(data.get("ff", {})),
        )

    # Métodos de retrocompat (callers antigos chamavam diretamente).
    def save_to_file(self, file_path: str = "config.json"):
        self.to_json(file_path)

    @classmethod
    def load_from_file(cls, file_path: str = "config.json") -> "SystemConfig":
        if not os.path.exists(file_path):
            return cls()
        try:
            return cls.from_json(file_path)
        except Exception:
            return cls()


# ─────────────────────────────────────────────────────────────────────
# AppConfig alias (retrocompat de imports antigos)
# ─────────────────────────────────────────────────────────────────────

def __getattr__(name):
    if name == "AppConfig":
        warnings.warn(
            "A classe 'AppConfig' foi renomeada para 'SystemConfig'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SystemConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


AppConfig = SystemConfig
