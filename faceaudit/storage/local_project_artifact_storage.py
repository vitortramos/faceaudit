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

    # Sufixos de companion files que SEMPRE acompanham um parquet/pkl
    # quando movido. Inclui sidecars regeneráveis (_stats/_dist/_metrics)
    # E o audit log (.audit.json — cadeia de custódia forense, NÃO deve
    # ser apagado por cleanups genéricos, mas DEVE ser movido junto).
    _ARTIFACT_COMPANION_SUFFIXES = (
        "_stats.json",
        "_dist.json",
        "_metrics.json",
        ".audit.json",  # dot-suffix: replaces parquet extension
    )

    def migrate_legacy_datasets_layout(
        self,
        definitions: List["LRStrategyDefinition"],  # type: ignore[name-defined]
    ) -> dict:
        """Migra layout antigo `datasets/<X>.parquet` (flat) pro novo:
          - PRIMARY: `datasets/primary/<X>.parquet`
          - DERIVED: `datasets/<strategy_folder>/<rid>_<id>.parquet`

        Iterates `definitions` pra encontrar artifact_refs apontando
        pra paths antigos. Pra cada um:
          1. Computa novo path canônico
          2. Move arquivo principal + companions (sidecars + audit.json)
          3. Atualiza `artifact_ref.path` na definition em memória

        O save_definition é DELEGADO ao caller (FaceAuditProject.check_integrity)
        pra evitar acoplar esse helper ao repository.

        Detecção de "antigo":
          - PRIMARY: path == `datasets/<X>.parquet` (sem subfolder)
          - DERIVED: path == `datasets/<sid>_<rid>...parquet` (com prefixo sid)

        Args:
            definitions: Lista de definitions do projeto.

        Returns:
            Dict com:
              - "moved_primary": List[(old_rel, new_rel)]
              - "moved_derived": List[(old_rel, new_rel)]
              - "updated_definitions": Set[strategy_id] cujas defns
                tiveram artifact_refs alteradas.
              - "failures": List[(old_rel, error_str)]
        """
        from faceaudit.enums.resource_category import ResourceCategory
        from faceaudit.enums.strategy_resource_type import StrategyResourceType
        from faceaudit.models.strategy.strategy_folder import strategy_folder_name

        moved_primary: List[tuple] = []
        moved_derived: List[tuple] = []
        updated_strategies: set = set()
        failures: List[tuple] = []

        for defn in definitions:
            strategy_id = defn.strategy_id
            strategy_name = defn.name or ""
            strategy_folder = strategy_folder_name(strategy_id, strategy_name)

            for res in defn.resources:
                if res.artifact_ref is None or not res.artifact_ref.path:
                    continue
                if res.resource_type != StrategyResourceType.DATASET:
                    continue
                old_path = res.artifact_ref.path.replace("\\", "/")
                if not old_path.startswith("datasets/"):
                    continue
                tail = old_path[len("datasets/"):]
                # Já em formato novo? Skip.
                if "/" in tail:
                    continue

                # Decide novo path por categoria.
                if res.category == ResourceCategory.PRIMARY:
                    new_path = f"datasets/primary/{tail}"
                elif res.category == ResourceCategory.DERIVED:
                    # Stripa prefixo `<sid>_` do filename, se presente.
                    if tail.startswith(f"{strategy_id}_"):
                        stripped = tail[len(strategy_id) + 1:]
                    else:
                        # Defensivo: derivado sem prefixo é caso raro;
                        # mantém nome.
                        stripped = tail
                    # Garantir sufixo `_<id>` antes da extensão. Se já
                    # tem, não duplica.
                    res_id = getattr(res, "id", "") or ""
                    if res_id and not stripped.endswith(
                        f"_{res_id}.parquet"
                    ):
                        # Adiciona `_<id>` antes da extensão.
                        if stripped.endswith(".parquet"):
                            stripped = (
                                stripped[: -len(".parquet")]
                                + f"_{res_id}.parquet"
                            )
                    new_path = f"datasets/{strategy_folder}/{stripped}"
                else:
                    continue

                if new_path == old_path:
                    continue

                # Move arquivo principal + companions
                ok, errs = self._move_artifact_with_companions(
                    old_path, new_path,
                )
                if not ok:
                    failures.append((old_path, "; ".join(errs)))
                    continue

                res.artifact_ref.path = new_path
                updated_strategies.add(strategy_id)
                if res.category == ResourceCategory.PRIMARY:
                    moved_primary.append((old_path, new_path))
                else:
                    moved_derived.append((old_path, new_path))

        return {
            "moved_primary": moved_primary,
            "moved_derived": moved_derived,
            "updated_definitions": updated_strategies,
            "failures": failures,
        }

    def _move_artifact_with_companions(
        self, old_rel: str, new_rel: str,
    ) -> tuple:
        """Move arquivo principal + companions (sidecars + audit.json)
        de `old_rel` pra `new_rel`, ambos relativos ao project_root.

        Returns:
            (ok: bool, errors: List[str])
        """
        import shutil

        errors: List[str] = []
        old_abs = self._root / old_rel
        new_abs = self._root / new_rel

        # Idempotência cross-strategy: o mesmo PRIMARY pode estar
        # referenciado por N estratégias. Quando a 1ª itera, move source
        # → dest. Quando as próximas iteram, source não existe mais MAS
        # dest existe. Trata como "já migrado" — ref deve ser atualizado
        # mesmo assim. Sem isso só a 1ª estratégia recebe o ref novo
        # (artifact_refs das outras ficam stale apontando flat path).
        if not old_abs.exists():
            if new_abs.exists():
                return (True, errors)
            errors.append(f"source missing: {old_rel}")
            return (False, errors)

        # Garante destination dir
        try:
            new_abs.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"mkdir failed: {e}")
            return (False, errors)

        # Move main file
        try:
            if new_abs.exists():
                # Idempotência: destino já existe (migração parcial
                # rodou antes?) — preserva destino, apaga source.
                old_abs.unlink()
            else:
                shutil.move(str(old_abs), str(new_abs))
        except Exception as e:
            errors.append(f"move main failed: {e}")
            return (False, errors)

        # Move companions (sidecars + audit log) — best-effort, falhas
        # individuais não abortam (parquet é o principal).
        old_stem = old_abs.stem
        for suffix in self._ARTIFACT_COMPANION_SUFFIXES:
            if suffix.startswith("."):
                # dot-suffix substitui extensão: `<stem>.audit.json`
                old_companion = old_abs.with_suffix(suffix)
                new_companion = new_abs.with_suffix(suffix)
            else:
                # underscore-suffix preserva extensão: `<stem>_stats.json`
                old_companion = old_abs.parent / (old_stem + suffix)
                new_companion = new_abs.parent / (new_abs.stem + suffix)
            if not old_companion.exists():
                continue
            try:
                if new_companion.exists():
                    old_companion.unlink()
                else:
                    shutil.move(str(old_companion), str(new_companion))
            except Exception as e:
                errors.append(f"companion {suffix}: {e}")

        return (True, errors)

    # Pastas hierárquicas onde cada estratégia ocupa uma subpasta. Usado
    # pelos cleanups que sabem que `<folder>/<X>/` é um agrupamento por
    # estratégia.
    # DATASETS entra aqui após a refactor que separou PRIMARY (em
    # `datasets/primary/`) de DERIVED (em `datasets/<strategy_folder>/`).
    _STRATEGY_SUBFOLDER_PARENTS = (
        ProjectFolder.MODELS,
        ProjectFolder.SCORES,
        ProjectFolder.RESULTS,
        ProjectFolder.GRAPHICS,
        ProjectFolder.DATASETS,
    )

    @staticmethod
    def _rmtree_force(path) -> bool:
        """`shutil.rmtree` resistente a Windows READONLY attr.

        Windows seta READONLY em pastas/arquivos por vários motivos
        (OneDrive sync, criação via copy de fonte READONLY, etc.).
        `rmtree` padrão falha com PermissionError quando encontra um
        item READONLY. `ignore_errors=True` mascara isso silenciosamente
        e o filesystem fica em estado inconsistente com o que o código
        loga.

        Esta versão usa o callback `onerror` pra tirar a flag READONLY
        e retentar. Retorna True se a pasta sumiu de fato (False se
        ainda persiste após tentativa).
        """
        import os
        import stat
        import shutil

        def _on_rm_error(func, target, exc_info):
            try:
                os.chmod(target, stat.S_IWRITE)
                func(target)
            except Exception:
                pass

        try:
            shutil.rmtree(path, onerror=_on_rm_error)
        except Exception:
            pass
        # Verifica deleção real — onerror swallow garante que rmtree
        # sempre retorna, mas a pasta pode permanecer se o chmod+retry
        # também falhou (ex: file lock por outro processo).
        from pathlib import Path as _P
        return not _P(path).exists()

    @staticmethod
    def _rmdir_force(path) -> bool:
        """`Path.rmdir` resistente a Windows READONLY attr.

        Análogo a `_rmtree_force` mas pra diretório vazio.
        """
        import os
        import stat
        try:
            path.rmdir()
        except PermissionError:
            try:
                os.chmod(path, stat.S_IWRITE)
                path.rmdir()
            except Exception:
                return False
        except OSError:
            return False
        return not path.exists()

    def cleanup_legacy_strategy_subfolders(
        self,
        strategy_pairs: List[tuple],
    ) -> List[str]:
        """Apaga subpastas legadas `<sid>/` quando a versão canônica
        `<name>_<sid>/` existe na mesma pasta hierárquica.

        Contexto: a refactor que adicionou `<name>_<sid>` nas subpastas
        de estratégia pode deixar resíduos do nome antigo (`<sid>` puro)
        quando a estratégia é re-executada com o novo template. Os
        arquivos novos vão pra `<name>_<sid>/`, e a pasta legada fica
        ou vazia (caso comum em models/scores/results — cleanup por
        arquivo já rodou) ou ainda com arquivos (caso de graphics/,
        cujos PNGs não são individualmente referenciados).

        Critério: legacy `<sid>/` é safe de apagar se há sibling
        `<name>_<sid>/` (com name != "") — sinal de que a estratégia já
        produziu artefatos no novo path e o legado é residual.

        Args:
            strategy_pairs: Lista de `(strategy_id, current_folder_name)`
                            pra cada estratégia. `current_folder_name`
                            vem de `strategy_folder_name(sid, name)`.

        Returns:
            Lista de paths relativos (POSIX) das subpastas apagadas.
        """
        deleted: List[str] = []
        for strategy_id, current_folder in strategy_pairs:
            if not strategy_id or not current_folder:
                continue
            if current_folder == strategy_id:
                # Sem rename / nome vazio — não há legacy distinto.
                continue
            for folder in self._STRATEGY_SUBFOLDER_PARENTS:
                parent = self._root / folder.value
                if not parent.is_dir():
                    continue
                current_dir = parent / current_folder
                legacy_dir = parent / strategy_id
                # Só apaga legacy se canônica existe (estratégia migrou).
                if not current_dir.is_dir():
                    continue
                if not legacy_dir.is_dir():
                    continue
                rel_path = legacy_dir.relative_to(self._root).as_posix()
                # _rmtree_force lida com Windows READONLY + verifica
                # deleção real. Só reporta como deletado se a pasta
                # SUMIU de fato.
                if self._rmtree_force(legacy_dir):
                    deleted.append(rel_path + "/")

        return deleted

    def cleanup_empty_strategy_subfolders(self) -> List[str]:
        """Apaga diretórios vazios em models/, scores/, results/, graphics/
        via walk bottom-up.

        Contexto: `_unlink_old_artifact_with_sidecars` apaga arquivos
        individuais mas não chama `rmdir` no pai. Quando todos os arquivos
        de `models/<sid>/calibration/` foram apagados (path-changed
        re-execução), a subpasta vazia fica como ruído visual no
        filesystem. Walk bottom-up garante que `models/<sid>/calibration/`
        é apagada ANTES de tentar `models/<sid>/` (caso fique tudo vazio).

        Retorna lista de paths relativos (POSIX) dos diretórios apagados.
        Falhas individuais (permission, race com outra process) são
        engolidas.
        """
        deleted: List[str] = []
        for folder in self._STRATEGY_SUBFOLDER_PARENTS:
            root = self._root / folder.value
            if not root.is_dir():
                continue
            # Sort por profundidade descendente — leaves primeiro.
            all_dirs = [p for p in root.rglob("*") if p.is_dir()]
            all_dirs.sort(key=lambda p: -len(p.parts))
            for path in all_dirs:
                try:
                    if any(path.iterdir()):
                        continue
                except Exception:
                    continue
                rel_path = path.relative_to(self._root).as_posix()
                # _rmdir_force lida com Windows READONLY + verifica
                # deleção real.
                if self._rmdir_force(path):
                    deleted.append(rel_path + "/")
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
