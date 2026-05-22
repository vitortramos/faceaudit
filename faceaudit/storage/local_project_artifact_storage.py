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
            path=relative_path.as_posix(),
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
            path=relative_path.as_posix(),
            hash=file_hash,
        )

    def import_dataset(
        self,
        source_path: Union[str, Path],
        dataset_id: str,
    ) -> StrategyArtifactReference:
        """Importa um parquet de dataset pro projeto, absolutizando paths
        se necessário.

        Diferente do `copy_artifact_to_project` genérico, este método sabe
        que um parquet pode declarar `path_mode=RELATIVE` na metadata —
        nesse caso, os paths das imagens são ancorados em
        `(source_parquet_dir / path_root)` e regravados como absolutos
        antes da cópia. Sem isso, ao trazer um dataset RELATIVE pra
        `<projeto>/datasets/`, os paths internos passariam a apontar pra
        `<projeto>/datasets/<path_root>/...` (onde as imagens não estão).

        Datasets já em `path_mode=ABSOLUTE` são copiados byte-a-byte —
        nada a transformar.

        Args:
            source_path: Caminho absoluto do parquet original.
            dataset_id: Identificador do dataset no projeto (usado pra
                montar o path destino via `StrategyFileType.SOURCE_DATASET`).

        Returns:
            StrategyArtifactReference com path relativo ao projeto e
            hash SHA-256 do parquet final (pós-absolutização, se houve).

        Raises:
            FileNotFoundError: Se `source_path` não existir.
        """
        from faceaudit.services.dataset_path_absolutizer import (
            absolutize_parquet_paths,
        )

        file_type = StrategyFileType.for_artifact(
            StrategyArtifactType.GLOBAL_DATASET,
        )
        relative_path = file_type.build_path(dataset_id=dataset_id)
        abs_dest = self._root / relative_path

        absolutize_parquet_paths(source_path, abs_dest)

        file_hash = self._hash.calculate_file_hash(abs_dest)
        return StrategyArtifactReference(
            artifact_type=StrategyArtifactType.GLOBAL_DATASET,
            path=relative_path.as_posix(),
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

            # rglob percorre subpastas — consistente com GlobalLibraryStorage.
            # No projeto isso raramente importa (a cópia da biblioteca achata
            # no destino), mas garante que arquivos colocados manualmente em
            # subpastas pelo user também sejam descobertos.
            for path in folder_path.rglob(f"*{file_type.extension}"):
                if not path.is_file():
                    continue
                try:
                    relative = path.relative_to(self._root)
                    found.append(
                        StrategyArtifactReference(
                            artifact_type=a_type,
                            path=relative.as_posix(),
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

    def cleanup_orphan_sidecars(self) -> List[str]:
        """Apaga sidecars (`_stats.json`/`_dist.json`/`_metrics.json`) cujo
        arquivo principal não existe mais no mesmo diretório.

        Convenção do sistema: sidecars são SEMPRE escritos ao lado de um
        arquivo principal (parquet/pkl) via `parent_path.with_name(stem + suffix)`.
        Quando o principal é re-gravado em um path diferente (ex: filename
        ganhou `_{id}` numa refactor de naming) ou é apagado, o sidecar
        antigo fica órfão. Como sidecars são 100% derivados (regerados
        sob demanda a partir do principal), apagá-los é seguro — zero
        perda de informação.

        Retorna a lista de paths relativos apagados (forward-slash POSIX),
        pra log/telemetria do caller. Falhas individuais são engolidas
        (filesystem locks, permission denied) — cleanup é best-effort.
        """
        deleted: List[str] = []
        for folder in self._ORPHAN_SCAN_FOLDERS:
            folder_path = self._root / folder.value
            if not folder_path.exists():
                continue

            for file_path in folder_path.rglob("*"):
                if not file_path.is_file():
                    continue

                # Detecta qual suffix bate, se algum.
                matched_suffix = None
                for suffix in self._SIDECAR_SUFFIXES:
                    if file_path.name.endswith(suffix):
                        matched_suffix = suffix
                        break
                if matched_suffix is None:
                    continue

                # Stem do arquivo pai (sem o sufixo do sidecar). Ex:
                # 'PLDA MODEL_TS-abcd_dist.json' → 'PLDA MODEL_TS-abcd'.
                parent_stem = file_path.name[: -len(matched_suffix)]

                # Pai pode ter qualquer extensão registrada (.parquet/.pkl
                # cobrem todos os tipos de sidecar atuais). Cheque genérica
                # via glob — qualquer sibling com mesmo stem qualifica.
                has_parent = any(
                    sibling.is_file()
                    and sibling.stem == parent_stem
                    and sibling.name != file_path.name
                    for sibling in file_path.parent.iterdir()
                )
                if has_parent:
                    continue

                try:
                    rel_path = file_path.relative_to(self._root).as_posix()
                    file_path.unlink()
                    deleted.append(rel_path)
                except Exception:
                    pass

        return deleted

    # Subpastas escaneadas em busca de arquivos órfãos. Mantemos apenas as pastas
    # cujos arquivos são tipicamente referenciados via StrategyArtifactReference.
    # GRAPHICS/REPORTS/CHECKPOINTS são produzidos por executores mas não entram em
    # `artifact_ref` (snapshots, exports, estados intermediários — todos derivativos).
    # DEFINITIONS é a fonte das próprias referências, IMAGES contém cópias originais
    # referenciadas por FaceInfo (não por artifact_ref).
    # MODELS/SCORES/RESULTS são as 3 pastas hierárquicas da nova arquitetura
    # (subpastas por strategy_id e tipo). ARTIFACTS é mantida pra detectar arquivos
    # de projetos com layout legado que ainda não foram migrados — rglob ignora
    # se a pasta não existir.
    _ORPHAN_SCAN_FOLDERS = (
        ProjectFolder.DATASETS,
        ProjectFolder.PIPELINES,
        ProjectFolder.SPLIT_CONFIGS,
        ProjectFolder.MINING_CONFIGS,
        ProjectFolder.MODELS,
        ProjectFolder.SCORES,
        ProjectFolder.RESULTS,
        ProjectFolder.ARTIFACTS,
    )

    # Sidecars produzidos pelo sistema (sempre derivados de um artefato principal).
    # Não são órfãos verdadeiros — se o artefato principal for órfão, o sidecar
    # vira lixo junto, mas reportá-lo separado polui a auditoria.
    _SIDECAR_SUFFIXES = (
        "_stats.json",
        "_dist.json",
        "_metrics.json",
    )

    def scan_orphans(
        self,
        referenced_paths: List[str],
    ) -> List[ArtifactIntegrityResult]:
        """
        Lista arquivos presentes em subpastas auditadas mas não referenciados.

        Para cada arquivo encontrado em `_ORPHAN_SCAN_FOLDERS` cujo caminho relativo
        (normalizado para forward-slash) não conste em `referenced_paths`, retorna
        um `ArtifactIntegrityResult` com status ORPHAN. Sidecars conhecidos
        (`_stats.json`, `_dist.json`, etc.) são ignorados.

        Args:
            referenced_paths: Lista de paths relativos (forward-slash) declarados
                              em `StrategyArtifactReference.path` por alguma
                              estratégia do projeto.

        Returns:
            Lista de `ArtifactIntegrityResult` com `status=ORPHAN`, uma entrada
            por arquivo órfão. `artifact_ref` é sintético (artifact_type=UNKNOWN,
            hash=None) e carrega o path relativo do órfão.
        """
        normalized_refs = {p.replace("\\", "/") for p in referenced_paths if p}
        results: List[ArtifactIntegrityResult] = []

        for folder in self._ORPHAN_SCAN_FOLDERS:
            folder_path = self._root / folder.value
            if not folder_path.exists():
                continue

            for file_path in folder_path.rglob("*"):
                if not file_path.is_file():
                    continue

                rel_path = file_path.relative_to(self._root).as_posix()

                if any(rel_path.endswith(suffix) for suffix in self._SIDECAR_SUFFIXES):
                    continue

                if rel_path in normalized_refs:
                    continue

                synthetic_ref = StrategyArtifactReference(
                    artifact_type=StrategyArtifactType.UNKNOWN,
                    path=rel_path,
                    hash=None,
                )
                results.append(ArtifactIntegrityResult(
                    artifact_ref=synthetic_ref,
                    status=ArtifactIntegrityStatus.ORPHAN,
                    expected_hash=None,
                    actual_hash=None,
                    abs_path=file_path,
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
