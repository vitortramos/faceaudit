# faceaudit/services/project_service.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

from faceaudit.models.project.face_audit_project_metadata import FaceAuditProjectMetadata
from faceaudit.models.storage.artifact_integrity_result import ArtifactIntegrityResult
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.repositories.strategy_definition_repository import StrategyDefinitionRepository
from faceaudit.repositories.strategy_trained_model_repository import StrategyTrainedModelRepository
from faceaudit.storage.local_project_artifact_storage import LocalProjectArtifactStorage


# Nome fixo do arquivo âncora dentro de qualquer pasta de projeto FaceAudit.
_FA_FILENAME = "project.fa"


class FaceAuditProject:
    """
    Handle de um projeto FaceAudit aberto.

    Agrega todos os componentes necessários para interagir com o projeto:
    - metadados (.fa)
    - storage físico de artefatos
    - repositórios de definições e modelos treinados

    Criado exclusivamente por FaceAuditProjectService.create() ou .open().
    Não instanciar diretamente.
    """

    def __init__(
        self,
        metadata: FaceAuditProjectMetadata,
        root: Path,
        storage: LocalProjectArtifactStorage,
        definition_repository: StrategyDefinitionRepository,
        model_repository: StrategyTrainedModelRepository,
        integrity_results: Optional[List[ArtifactIntegrityResult]] = None,
        has_legacy_layout: bool = False,
    ) -> None:
        self._metadata = metadata
        self._root = root
        self._storage = storage
        self._definition_repository = definition_repository
        self._model_repository = model_repository
        self._integrity_results: Optional[List[ArtifactIntegrityResult]] = integrity_results
        # True quando o projeto tem layout pré-reorg (pasta `artifacts/` populada
        # com pickles/parquets que a nova arch espera ver em `models/`, `scores/`,
        # `results/`). UI consome essa flag pra exibir diálogo de migração.
        self._has_legacy_layout = has_legacy_layout

    # =========================================================================
    # PROPRIEDADES — somente leitura
    # =========================================================================

    @property
    def metadata(self) -> FaceAuditProjectMetadata:
        """Metadados do projeto (nome, id, datas, versão)."""
        return self._metadata

    @property
    def root(self) -> Path:
        """Caminho absoluto da pasta raiz do projeto."""
        return self._root

    @property
    def fa_file(self) -> Path:
        """Caminho absoluto do arquivo .fa deste projeto."""
        return self._root / _FA_FILENAME

    @property
    def storage(self) -> LocalProjectArtifactStorage:
        """Storage de artefatos do projeto (Pendrive Paradigm)."""
        return self._storage

    @property
    def definition_repository(self) -> StrategyDefinitionRepository:
        """Repositório de definições de estratégia."""
        return self._definition_repository

    @property
    def model_repository(self) -> StrategyTrainedModelRepository:
        """Repositório de modelos treinados."""
        return self._model_repository

    @property
    def integrity_results(self) -> Optional[List[ArtifactIntegrityResult]]:
        """
        Resultados da última verificação de integridade.

        None se check_integrity() nunca foi chamado (nem automaticamente via open()).
        Lista vazia se a verificação rodou e todos os artefatos estão OK.
        Lista com entradas se algum artefato está ausente, modificado ou sem hash.
        """
        return self._integrity_results

    @property
    def has_legacy_layout(self) -> bool:
        """True quando o projeto tem `artifacts/` populada (layout pré-reorg).

        Sinaliza pra UI que precisa oferecer migração wipe-and-rerun: apagar
        recursos DERIVED de todas as estratégias + remover `artifacts/`, e
        deixar o usuário re-executar nas novas subpastas hierárquicas
        (`models/<sid>/...`, `scores/<sid>/...`, `results/<sid>/...`).
        """
        return self._has_legacy_layout

    @property
    def has_integrity_problems(self) -> bool:
        """
        True se a última verificação encontrou artefatos AUSENTES ou MODIFICADOS.
        False se não houve verificação ou se tudo está OK.
        """
        if not self._integrity_results:
            return False
        return any(r.is_problem for r in self._integrity_results)

    # =========================================================================
    # VERIFICAÇÃO DE INTEGRIDADE
    # =========================================================================

    def check_integrity(self) -> List[ArtifactIntegrityResult]:
        """
        Verifica a integridade de todos os artefatos declarados nas estratégias do projeto.

        Carrega todas as definições de estratégia presentes no projeto e coleta os
        artifact_ref de seus recursos. Para cada referência com hash registrado,
        recalcula o SHA-256 e compara.

        O resultado é armazenado em self.integrity_results e também retornado.

        Returns:
            Lista de ArtifactIntegrityResult, uma por artefato verificado.
            Entradas com status MODIFIED ou MISSING indicam problemas.
        """
        refs: List[StrategyArtifactReference] = []

        for definition in self._definition_repository.list_definitions():
            for resource in definition.resources:
                if resource.artifact_ref is not None:
                    refs.append(resource.artifact_ref)

        results = self._storage.verify_integrity(refs)
        results.extend(
            self._storage.scan_orphans([ref.path for ref in refs])
        )
        self._integrity_results = results
        return self._integrity_results

    # =========================================================================
    # MUTAÇÕES DE METADADOS
    # =========================================================================

    def rename(self, new_name: str) -> None:
        """
        Renomeia o projeto e persiste o .fa atualizado.

        Raises:
            ValueError: Se new_name for vazio.
        """
        self._metadata.rename(new_name)
        self._save_metadata()

    def set_description(self, description: str) -> None:
        """Altera a descrição do projeto e persiste o .fa atualizado."""
        self._metadata.set_description(description)
        self._save_metadata()

    # =========================================================================
    # PERSISTÊNCIA DE METADADOS
    # =========================================================================

    def _save_metadata(self) -> None:
        """Persiste os metadados atuais no arquivo .fa."""
        self.fa_file.write_text(
            json.dumps(self._metadata.to_dict(), indent=4, ensure_ascii=False),
            encoding="utf-8",
        )


class FaceAuditProjectService:
    """
    Serviço para criação e abertura de projetos FaceAudit na nova arquitetura.

    Responsabilidades:
    - Criar a estrutura de pastas de um novo projeto
    - Gerar e persistir o arquivo .fa com metadados iniciais
    - Abrir um projeto existente a partir de um arquivo .fa
    - Verificar se uma pasta é um projeto FaceAudit válido

    Não mantém estado — todos os métodos são estáticos.
    """

    FA_FILENAME: str = _FA_FILENAME

    @staticmethod
    def create(
        folder: Union[str, Path],
        name: str,
        description: str = "",
    ) -> FaceAuditProject:
        """
        Cria um novo projeto FaceAudit na pasta indicada.

        A pasta é criada se não existir. A estrutura de subpastas (definida por
        ProjectFolder) é criada automaticamente. Um arquivo project.fa é gerado
        com os metadados iniciais.

        Args:
            folder: Caminho da pasta raiz do novo projeto.
            name: Nome legível do projeto.
            description: Descrição opcional.

        Returns:
            FaceAuditProject: Handle do projeto recém-criado, pronto para uso.

        Raises:
            ValueError: Se name for vazio.
            FileExistsError: Se já existir um project.fa na pasta indicada.
        """
        if not name or not name.strip():
            raise ValueError("O nome do projeto não pode ser vazio.")

        root = Path(folder)
        root.mkdir(parents=True, exist_ok=True)

        fa_file = root / _FA_FILENAME
        if fa_file.exists():
            raise FileExistsError(
                f"Já existe um projeto FaceAudit nesta pasta: {fa_file}"
            )

        storage = LocalProjectArtifactStorage(project_root=root)
        storage.create_project_structure()

        metadata = FaceAuditProjectMetadata(name=name.strip(), description=description)
        fa_file.write_text(
            json.dumps(metadata.to_dict(), indent=4, ensure_ascii=False),
            encoding="utf-8",
        )

        return FaceAuditProject(
            metadata=metadata,
            root=root,
            storage=storage,
            definition_repository=StrategyDefinitionRepository(artifact_storage=storage),
            model_repository=StrategyTrainedModelRepository(artifact_storage=storage),
        )

    @staticmethod
    def open(
        fa_file: Union[str, Path],
        verify_integrity: bool = True,
    ) -> FaceAuditProject:
        """
        Abre um projeto FaceAudit existente a partir do arquivo .fa.

        Por padrão executa a verificação de integridade dos artefatos declarados
        nas estratégias do projeto logo após a abertura. O resultado fica disponível
        em project.integrity_results e project.has_integrity_problems.

        Args:
            fa_file: Caminho para o arquivo project.fa do projeto.
            verify_integrity: Se True (padrão), executa check_integrity() automaticamente
                              após a abertura. Passe False para abrir sem verificação
                              (útil quando a verificação já foi feita ou por performance).

        Returns:
            FaceAuditProject: Handle do projeto aberto, pronto para uso.

        Raises:
            FileNotFoundError: Se o arquivo .fa não existir.
            ValueError: Se o arquivo não tiver extensão .fa ou estiver corrompido.
        """
        fa_path = Path(fa_file)

        if not fa_path.exists():
            raise FileNotFoundError(f"Arquivo de projeto não encontrado: {fa_path}")

        if fa_path.suffix != ".fa":
            raise ValueError(
                f"Esperado um arquivo .fa, mas recebido: {fa_path.suffix}"
            )

        try:
            data = json.loads(fa_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Não foi possível ler o arquivo de projeto: {fa_path}"
            ) from exc

        metadata = FaceAuditProjectMetadata.from_dict(data)
        if metadata is None:
            raise ValueError(
                f"O arquivo .fa está vazio ou com formato inválido: {fa_path}"
            )

        root = fa_path.parent
        storage = LocalProjectArtifactStorage(project_root=root)

        # Detecta layout legado: pasta `artifacts/` com conteúdo. Nova arch
        # escreve em `models/`, `scores/`, `results/` — `artifacts/` populada
        # indica projeto criado antes da reorg, precisa de migração.
        legacy_dir = root / "artifacts"
        has_legacy_layout = legacy_dir.is_dir() and any(legacy_dir.iterdir())

        project = FaceAuditProject(
            metadata=metadata,
            root=root,
            storage=storage,
            definition_repository=StrategyDefinitionRepository(artifact_storage=storage),
            model_repository=StrategyTrainedModelRepository(artifact_storage=storage),
            has_legacy_layout=has_legacy_layout,
        )

        if verify_integrity:
            project.check_integrity()

        return project

    @staticmethod
    def find_project(folder: Union[str, Path]) -> Optional[Path]:
        """
        Retorna o caminho do arquivo .fa se a pasta for um projeto FaceAudit.

        Args:
            folder: Pasta a verificar.

        Returns:
            Path para o project.fa se existir, None caso contrário.
        """
        fa_file = Path(folder) / _FA_FILENAME
        return fa_file if fa_file.exists() else None

    @staticmethod
    def is_project_folder(folder: Union[str, Path]) -> bool:
        """
        Retorna True se a pasta contiver um projeto FaceAudit válido.

        Args:
            folder: Pasta a verificar.
        """
        return FaceAuditProjectService.find_project(folder) is not None
