# faceaudit/storage/local_project_artifact_storage.py

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any, List, Optional, Union

from faceaudit.enums.artifact_integrity_status import ArtifactIntegrityStatus
from faceaudit.enums.project_folder import ProjectFolder
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_file_type import StrategyFileType
from faceaudit.models.storage.artifact_integrity_result import ArtifactIntegrityResult
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol
from faceaudit.services.hash_service import HashService


class LocalProjectArtifactStorage(ArtifactStorageProtocol):
    """
    Implementação concreta de ArtifactStorageProtocol que garante o Pendrive Paradigm.

    Regras fundamentais desta implementação:
    - project_root é o único ponto de configuração — sem dependências de
      FileSystemService, ConfigLoaderService ou ProjectConfig.
    - Todos os paths em StrategyArtifactReference são RELATIVOS à project_root.
      Mover o diretório do projeto para outro disco ou máquina não quebra nenhuma
      referência armazenada.
    - A estrutura de subpastas segue ProjectFolder e StrategyFileType — a mesma
      taxonomia do projeto é compartilhada entre arquitetura antiga e nova.

    Uso típico:
        storage = LocalProjectArtifactStorage(project_root=Path("/media/pendrive/caso_01"))
        storage.create_project_structure()
        ref = storage.save_artifact(StrategyArtifactType.STRATEGY_DEFINITION, definition, strategy_id="ST_01")
        # ref.path == "definitions/ST_01_definition.json"  (sempre relativo)
    """

    def __init__(
        self,
        project_root: Union[str, Path],
        hash_service: Optional[HashService] = None,
    ):
        """
        Args:
            project_root: Raiz do projeto autocontido. Todos os artefatos serão
                          salvos em subpastas relativas a este diretório.
            hash_service: Instância opcional de HashService para cálculo de SHA256.
                          Se não fornecido, uma instância padrão é criada.
        """
        self._root = Path(project_root)
        self._hash = hash_service or HashService()

    # =========================================================================
    # PROPRIEDADES
    # =========================================================================

    @property
    def project_root(self) -> Path:
        """Raiz absoluta do projeto. Somente leitura."""
        return self._root

    # =========================================================================
    # ESTRUTURA DO PROJETO
    # =========================================================================

    def create_project_structure(self) -> None:
        """
        Cria a árvore de subpastas do projeto a partir do enum ProjectFolder.

        Idempotente — seguro para chamar múltiplas vezes.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        for folder in ProjectFolder:
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
        Persiste um artefato na subpasta correta do projeto e retorna a referência.

        O path na referência retornada é sempre RELATIVO à project_root.

        Args:
            artifact_type: Tipo semântico do artefato (define pasta e extensão).
            data: Objeto a ser persistido. Suporta dicts/objetos com to_dict() para
                  JSON, qualquer objeto Python para pickle, DataFrames para parquet,
                  e strings para logs.
            **kwargs: Variáveis para preencher o template de nome do StrategyFileType.
                      Exemplo: strategy_id="ST_01", step_id="pipe_a".

        Returns:
            StrategyArtifactReference com path relativo e hash SHA256.

        Raises:
            ValueError: Se kwargs não contiver as variáveis necessárias para o template.
        """
        import time as _time

        # progress_callback é meta — não vai para o template de path.
        progress_callback = kwargs.pop("progress_callback", None)

        def _emit(msg: str) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        t_start = _time.perf_counter()

        file_type = StrategyFileType.for_artifact(artifact_type)
        relative_path = file_type.build_path(**kwargs)
        abs_path = self._root / relative_path

        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # ETAPA 1 — escrita em disco
        t1 = _time.perf_counter()
        self._write(abs_path, data, file_type.extension, progress_callback=progress_callback)
        t_write = _time.perf_counter() - t1

        # ETAPA 2 — hash SHA-256 (lê o arquivo inteiro)
        t2 = _time.perf_counter()
        file_hash = self._hash.calculate_file_hash(abs_path)
        t_hash = _time.perf_counter() - t2

        # Telemetria: tempo total + tamanho do arquivo. Permite ao perito
        # ver onde o gargalo está (write ou hash) caso a persistência fique
        # lenta. Só emite se houver progress_callback (no fluxo normal de
        # execução de instrução; chamadas programáticas ficam silenciosas).
        try:
            file_mb = abs_path.stat().st_size / (1024 * 1024)
        except Exception:
            file_mb = 0.0
        t_total = _time.perf_counter() - t_start
        _emit(
            f"⏱  Persistência '{relative_path}': {file_mb:.1f} MB | "
            f"write={t_write:.2f}s, hash={t_hash:.2f}s, total={t_total:.2f}s"
        )

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
        Copia um artefato externo para a estrutura do projeto e retorna a referência.

        Útil para incorporar ao projeto um dataset ou pipeline que estava em outra
        localização (ex: repositório global da máquina).

        Args:
            source_path: Caminho absoluto do arquivo de origem.
            artifact_type: Tipo semântico do artefato.
            **kwargs: Variáveis para o template de nome.

        Returns:
            StrategyArtifactReference com path relativo e hash SHA256.

        Raises:
            FileNotFoundError: Se source_path não existir.
        """
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Arquivo de origem não encontrado: {src}")

        file_type = StrategyFileType.for_artifact(artifact_type)
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

        A resolução é direta: project_root / ref.path.
        Se o arquivo não existir, retorna None em vez de lançar exceção.

        Args:
            artifact_ref: Referência do artefato com path relativo.

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
        Lista os artefatos presentes no projeto, opcionalmente filtrados por tipo.

        Varre as subpastas conforme o mapeamento em StrategyFileType. Tipos sem
        mapeamento (ex: UNKNOWN) são silenciosamente ignorados.

        Args:
            artifact_type: Se fornecido, lista apenas artefatos deste tipo.
                           Se None, lista todos os tipos mapeados.

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
            file_type = StrategyFileType.for_artifact_safe(a_type)
            if file_type is None:
                continue

            folder_path = self._root / file_type.folder.value
            if not folder_path.exists():
                continue

            for path in folder_path.glob(f"*{file_type.extension}"):
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
    # VERIFICAÇÃO DE INTEGRIDADE
    # =========================================================================

    def verify_integrity(
        self,
        refs: List[StrategyArtifactReference],
    ) -> List[ArtifactIntegrityResult]:
        """
        Verifica a integridade dos artefatos referenciados.

        Para cada referência:
        - MISSING          → arquivo não encontrado em disco
        - HASH_UNAVAILABLE → arquivo presente, referência sem hash registrado
        - MODIFIED         → hash atual difere do hash registrado
        - OK               → hash confere

        Args:
            refs: Lista de StrategyArtifactReference a verificar.
                  Referências com path vazio são silenciosamente ignoradas.

        Returns:
            Lista de ArtifactIntegrityResult, uma entrada por referência válida.
            Ordem preservada.
        """
        results: List[ArtifactIntegrityResult] = []

        for ref in refs:
            if not ref or not ref.path:
                continue

            abs_path = self._root / ref.path

            if not abs_path.exists():
                results.append(ArtifactIntegrityResult(
                    artifact_ref=ref,
                    status=ArtifactIntegrityStatus.MISSING,
                    expected_hash=ref.hash,
                    actual_hash=None,
                    abs_path=None,
                ))
                continue

            if not ref.hash:
                results.append(ArtifactIntegrityResult(
                    artifact_ref=ref,
                    status=ArtifactIntegrityStatus.HASH_UNAVAILABLE,
                    expected_hash=None,
                    actual_hash=None,
                    abs_path=abs_path,
                ))
                continue

            actual_hash = self._hash.calculate_file_hash(abs_path)
            status = (
                ArtifactIntegrityStatus.OK
                if actual_hash == ref.hash
                else ArtifactIntegrityStatus.MODIFIED
            )

            results.append(ArtifactIntegrityResult(
                artifact_ref=ref,
                status=status,
                expected_hash=ref.hash,
                actual_hash=actual_hash,
                abs_path=abs_path,
            ))

        return results

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _write(self, path: Path, data: Any, extension: str, progress_callback=None) -> None:
        """
        Serializa data para o disco no formato adequado à extensão.

        Suporte:
        - .json  → dict, ou qualquer objeto com to_dict()
        - .pkl   → qualquer objeto Python (via pickle)
        - .ckpt  → mesmo que pkl (checkpoint de modelo)
        - .parquet → pandas DataFrame, ou objeto com to_dataframe()
        - .log   → string ou objeto com __str__
        - .png   → bytes brutos (imagem já gerada externamente)

        Raises:
            NotImplementedError: Para extensões não suportadas.
        """
        if extension == ".json":
            payload = data.to_dict() if hasattr(data, "to_dict") else data
            path.write_text(
                json.dumps(payload, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )

        elif extension in (".pkl", ".ckpt"):
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif extension == ".parquet":
            if hasattr(data, "to_parquet"):
                # Tenta passar progress_callback se a implementação aceitar.
                try:
                    data.to_parquet(path, progress_callback=progress_callback)
                except TypeError:
                    # pandas DataFrame não aceita o kwarg — chama sem ele.
                    data.to_parquet(path)
            elif hasattr(data, "to_dataframe"):
                # EmbeddingDataset ou similar com to_dataframe()
                data.to_dataframe().to_parquet(path)
            else:
                raise NotImplementedError(
                    f"Não sei serializar {type(data).__name__} para .parquet. "
                    "O objeto deve implementar to_parquet() ou to_dataframe()."
                )

        elif extension == ".log":
            content = data if isinstance(data, str) else str(data)
            path.write_text(content, encoding="utf-8")

        elif extension == ".png":
            if isinstance(data, bytes):
                path.write_bytes(data)
            else:
                raise NotImplementedError(
                    f"Para .png, forneça bytes da imagem. Recebido: {type(data).__name__}."
                )

        else:
            raise NotImplementedError(
                f"Extensão '{extension}' não suportada por LocalProjectArtifactStorage."
            )
