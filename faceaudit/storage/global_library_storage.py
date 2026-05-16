# faceaudit/storage/global_library_storage.py

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, List, Optional, Union

from faceaudit.enums.global_library_file_type import GlobalLibraryFileType
from faceaudit.enums.global_library_folder import GlobalLibraryFolder
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol
from faceaudit.services.hash_service import HashService


class GlobalLibraryStorage(ArtifactStorageProtocol):
    """
    Implementação de ArtifactStorageProtocol para a biblioteca global de recursos
    reutilizáveis entre projetos FaceAudit.

    Gerencia datasets, split configs, mining configs e templates de estratégia
    numa pasta central da máquina (configurada em config.json).

    Diferenças em relação a LocalProjectArtifactStorage:
    - Usa GlobalLibraryFolder e GlobalLibraryFileType (não ProjectFolder/StrategyFileType)
    - Só suporta tipos de artefato globais (GLOBAL_DATASET, SPLIT_CONFIG,
      MINING_CONFIG, STRATEGY_TEMPLATE)
    - Não implementa copy_artifact_to_project — esta operação flui no sentido
      inverso: o projeto copia DA biblioteca, não para ela

    Uso típico:
        lib = GlobalLibraryStorage(library_root=Path("/dados/faceaudit_library"))
        lib.create_library_structure()
        ref = lib.save_artifact(StrategyArtifactType.SPLIT_CONFIG, split_cfg, config_id="70_30_id")
        path = lib.resolve_artifact(ref)   # → absoluto, para copiar ao projeto
    """

    def __init__(
        self,
        library_root: Union[str, Path],
        hash_service: Optional[HashService] = None,
    ):
        """
        Args:
            library_root: Raiz da biblioteca global. Todos os recursos serão
                          salvos em subpastas relativas a este diretório.
            hash_service: Instância opcional de HashService para SHA256.
        """
        self._root = Path(library_root)
        self._hash = hash_service or HashService()

    # =========================================================================
    # PROPRIEDADES
    # =========================================================================

    @property
    def library_root(self) -> Path:
        """Raiz absoluta da biblioteca global. Somente leitura."""
        return self._root

    # =========================================================================
    # ESTRUTURA DA BIBLIOTECA
    # =========================================================================

    def create_library_structure(self) -> None:
        """
        Cria a árvore de subpastas da biblioteca a partir de GlobalLibraryFolder.

        Idempotente — seguro para chamar múltiplas vezes.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        for folder in GlobalLibraryFolder:
            if folder.value:
                (self._root / folder.value).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # ArtifactStorageProtocol — IMPLEMENTAÇÃO
    # =========================================================================

    def save_artifact(
        self,
        artifact_type: StrategyArtifactType,
        data: Any,
        **kwargs,
    ) -> StrategyArtifactReference:
        """
        Persiste um recurso na subpasta correta da biblioteca global.

        O path na referência retornada é sempre RELATIVO à library_root,
        permitindo que a localização da biblioteca mude sem quebrar referências.

        Args:
            artifact_type: Tipo do artefato global (GLOBAL_DATASET, SPLIT_CONFIG,
                           MINING_CONFIG ou STRATEGY_TEMPLATE).
            data: Objeto a ser persistido.
            **kwargs: Variáveis do template de nome (ex: config_id, dataset_id,
                      template_id).

        Raises:
            ValueError: Se artifact_type não for um tipo da biblioteca global,
                        ou se kwargs não tiver as variáveis necessárias.
        """
        file_type = GlobalLibraryFileType.for_artifact(artifact_type)
        relative_path = file_type.build_path(**kwargs)
        abs_path = self._root / relative_path

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        self._write(abs_path, data, file_type.extension)

        file_hash = self._hash.calculate_file_hash(abs_path)

        return StrategyArtifactReference(
            artifact_type=artifact_type,
            path=str(relative_path),
            hash=file_hash,
        )

    def copy_artifact_to_project(
        self,
        source_path: Union[str, Path],
        artifact_type: StrategyArtifactType,
        **kwargs,
    ) -> StrategyArtifactReference:
        """
        Copia um arquivo externo para a biblioteca global.

        Útil para importar um dataset ou template que estava em outro local.

        Args:
            source_path: Caminho absoluto do arquivo de origem.
            artifact_type: Tipo do artefato (deve ser um tipo da biblioteca global).
            **kwargs: Variáveis para o template de nome.

        Raises:
            FileNotFoundError: Se source_path não existir.
            ValueError: Se artifact_type não for um tipo da biblioteca global.
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Arquivo de origem não encontrado: {src}")

        file_type = GlobalLibraryFileType.for_artifact(artifact_type)
        relative_path = file_type.build_path(**kwargs)
        abs_dest = self._root / relative_path

        abs_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, abs_dest)

        file_hash = self._hash.calculate_file_hash(abs_dest)

        return StrategyArtifactReference(
            artifact_type=artifact_type,
            path=str(relative_path),
            hash=file_hash,
        )

    def resolve_artifact(
        self,
        artifact_ref: StrategyArtifactReference,
    ) -> Optional[Path]:
        """
        Converte a referência lógica em caminho físico absoluto.

        Retorna None se o arquivo não existir (não lança exceção), permitindo
        que o chamador decida como tratar a ausência.

        Args:
            artifact_ref: Referência com path relativo à library_root.

        Returns:
            Path absoluto se o arquivo existir, None caso contrário.
        """
        if not artifact_ref or not artifact_ref.path:
            return None

        abs_path = self._root / artifact_ref.path
        return abs_path if abs_path.exists() else None

    def list_artifacts(
        self,
        artifact_type: Optional[StrategyArtifactType] = None,
    ) -> List[StrategyArtifactReference]:
        """
        Lista os recursos presentes na biblioteca, opcionalmente filtrados por tipo.

        Varre apenas os tipos mapeados em GlobalLibraryFileType. Tipos sem
        mapeamento (artefatos de projeto) são silenciosamente ignorados.

        Args:
            artifact_type: Se fornecido, lista apenas artefatos deste tipo.
                           Se None, lista todos os tipos da biblioteca global.

        Returns:
            Lista de StrategyArtifactReference com path relativo e hash=None
            (hash não é recalculado na listagem por questão de performance).
        """
        types_to_scan = (
            [artifact_type]
            if artifact_type is not None
            else list(StrategyArtifactType)
        )

        found: List[StrategyArtifactReference] = []

        for a_type in types_to_scan:
            file_type = GlobalLibraryFileType.for_artifact_safe(a_type)
            if file_type is None:
                continue

            folder_path = self._root / file_type.folder.value
            if not folder_path.exists():
                continue

            # rglob percorre subpastas — usuário pode organizar a biblioteca
            # em subdiretórios (ex: datasets/cfp_frontal/, datasets/cfp_lateral/)
            # pra facilitar navegação quando há muitos recursos. O path relativo
            # preserva a hierarquia (ex: "datasets/cfp_frontal/foo.parquet").
            for path in folder_path.rglob(f"*{file_type.extension}"):
                if not path.is_file():
                    continue
                try:
                    relative = path.relative_to(self._root)
                    found.append(
                        StrategyArtifactReference(
                            artifact_type=a_type,
                            path=str(relative),
                            hash=None,
                        )
                    )
                except ValueError:
                    continue

        return found

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _write(self, path: Path, data: Any, extension: str) -> None:
        """
        Serializa data para o disco no formato adequado à extensão.

        Suporte:
        - .json  → dict, ou qualquer objeto com to_dict()
        - .parquet → pandas DataFrame, ou objeto com to_dataframe()

        Raises:
            NotImplementedError: Para extensões não suportadas na biblioteca global.
        """
        if extension == ".json":
            payload = data.to_dict() if hasattr(data, "to_dict") else data
            path.write_text(
                json.dumps(payload, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )

        elif extension == ".parquet":
            if hasattr(data, "to_parquet"):
                data.to_parquet(path)
            elif hasattr(data, "to_dataframe"):
                data.to_dataframe().to_parquet(path)
            else:
                raise NotImplementedError(
                    f"Não sei serializar {type(data).__name__} para .parquet. "
                    "O objeto deve implementar to_parquet() ou to_dataframe()."
                )

        else:
            raise NotImplementedError(
                f"Extensão '{extension}' não suportada por GlobalLibraryStorage."
            )
