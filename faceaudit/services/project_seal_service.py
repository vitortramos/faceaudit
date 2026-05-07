"""ProjectSealService — sela um projeto FaceAudit copiando todas as
imagens referenciadas pra dentro do `.fa` e reescrevendo paths como
relativos (PR L2.b).

Operação destrutiva: parquets de DATASET são reescritos in-place. Sem
rollback automático — audit log preserva a trilha.

Decisões herdadas do plano L2:
  - Substituição em-place (decisão #2 do usuário)
  - Suporta dry_run (decisão #3)
  - Processa só GLOBAL_DATASET, não INTERMEDIATE_DATASET (decisão #5)

Stateless. Construir uma instância por chamada ou compartilhar.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.localization_naming import NamingStrategy
from faceaudit.enums.path_mode import PathMode
from faceaudit.enums.project_folder import ProjectFolder
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_file_type import StrategyFileType
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.seal_result import SealResult
from faceaudit.services.audit_log_service import AuditLogService
from faceaudit.services.dataset_localization_service import (
    DatasetLocalizationService,
)


class ProjectSealService:
    """Sela um projeto: copia imagens pra dentro do `.fa` e reescreve
    parquets pra paths relativos."""

    def seal(
        self,
        project,  # FaceAuditProject — sem import pra evitar ciclo
        *,
        dry_run: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SealResult:
        """Sela o projeto.

        Args:
            project: handle de `FaceAuditProject` (de
                `FaceAuditProjectService.open()` ou `.create()`).
            dry_run: quando True, apenas simula — lista o que seria
                feito sem escrever em disco. Útil pra perito conferir
                escopo antes de selar de verdade.
            progress_callback: chamado por dataset com
                `(idx, total, dataset_id_msg)`.

        Returns:
            `SealResult` agregando totais.
        """
        start = time.time()
        project_root = Path(project.storage.project_root)

        # 1. Lista todos os artefatos GLOBAL_DATASET do projeto.
        artifacts = project.storage.list_artifacts(
            StrategyArtifactType.GLOBAL_DATASET,
        )
        total = len(artifacts)

        n_sealed = 0
        n_skipped = 0
        n_failed = 0
        total_imgs_copied = 0
        total_imgs_skipped = 0
        failures: List[Tuple[str, str]] = []

        loc_service = DatasetLocalizationService()

        for idx, ref in enumerate(artifacts):
            dataset_id = self._extract_dataset_id(ref.path)
            if progress_callback:
                progress_callback(idx, total, dataset_id)

            try:
                outcome = self._seal_one(
                    project_root,
                    ref.path,
                    dataset_id,
                    dry_run=dry_run,
                    loc_service=loc_service,
                )
                if outcome["status"] == "sealed":
                    n_sealed += 1
                    total_imgs_copied += outcome["copied"]
                    total_imgs_skipped += outcome["images_skipped"]
                elif outcome["status"] == "skipped":
                    n_skipped += 1
                else:  # "failed"
                    n_failed += 1
                    failures.append((dataset_id, outcome["error"]))
            except Exception as e:
                n_failed += 1
                failures.append((dataset_id, str(e)))

        return SealResult(
            project_root=project_root,
            n_datasets_processed=total,
            n_datasets_sealed=n_sealed,
            n_datasets_skipped=n_skipped,
            n_datasets_failed=n_failed,
            total_images_copied=total_imgs_copied,
            total_images_skipped=total_imgs_skipped,
            total_failures=tuple(failures),
            duration_seconds=time.time() - start,
            dry_run=dry_run,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _seal_one(
        self,
        project_root: Path,
        relative_path: str,
        dataset_id: str,
        *,
        dry_run: bool,
        loc_service: DatasetLocalizationService,
    ) -> dict:
        """Sela um único dataset. Retorna dict com `status` em
        {"sealed", "skipped", "failed"} + métricas."""
        parquet_path = project_root / relative_path
        if not parquet_path.exists():
            return {"status": "failed", "error": "parquet não encontrado"}

        ds = FaceDataset.from_parquet(parquet_path)

        # Idempotência: se já está em RELATIVE com path_root apontando
        # pra dentro do projeto, pula.
        if ds.path_mode == PathMode.RELATIVE:
            return {"status": "skipped"}

        # Dry-run: estima sem escrever
        if dry_run:
            return {
                "status": "sealed",
                "copied": len(ds.faces),
                "images_skipped": 0,
            }

        # Real: localiza pra `<project>/images/<dataset_id>/`
        target_images_dir = (
            project_root / ProjectFolder.IMAGES.value / dataset_id
        )
        result = loc_service.localize(
            ds,
            target_images_dir,
            mode=LocalizationMode.COPY_ORIGINALS,
            naming=NamingStrategy.PRESERVE,
            skip_existing=True,
        )

        # Ajusta path_root pro destino relativo ao parquet
        # (parquet em <datasets>/, imagens em <images>/<id>/images/)
        result.dataset.path_root = (
            f"../{ProjectFolder.IMAGES.value}/{dataset_id}/images"
        )

        # Sobrescreve o parquet original com a versão localizada
        result.dataset.to_parquet(parquet_path)

        # Audit log no parquet selado — registra COPY_IMAGES com
        # metadata do que foi feito.
        try:
            AuditLogService.append(
                parquet_path,
                AuditOperation.COPY_IMAGES,
                f"Projeto selado: {result.summary()}",
                metadata={
                    "operation": "project_seal",
                    "target_dir": str(target_images_dir),
                    "n_processed": result.n_processed,
                    "n_copied": result.n_copied,
                    "n_skipped": result.n_skipped,
                    "n_failed": result.n_failed,
                },
            )
        except Exception:
            pass  # audit é best-effort

        return {
            "status": "sealed",
            "copied": result.n_copied,
            "images_skipped": result.n_skipped,
        }

    @staticmethod
    def _extract_dataset_id(relative_path: str) -> str:
        """`datasets/foo.parquet` → `foo`."""
        return Path(relative_path).stem
