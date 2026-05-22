# tests/test_artifact_layout_migration.py
"""Detecção de projetos com layout pré-reorganização (pasta `artifacts/`
populada). A migração wipe-and-rerun em si vive no controller V2 (UI) —
aqui testamos apenas a sinalização `project.has_legacy_layout` que dispara
o diálogo na abertura."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faceaudit.services.project_service import FaceAuditProjectService


@pytest.fixture
def fresh_project(tmp_path):
    """Projeto recém-criado via service. Layout novo — sem artifacts/ populada."""
    return FaceAuditProjectService.create(
        folder=tmp_path / "proj_fresh",
        name="Fresh Project",
    )


class TestHasLegacyLayoutDetection:

    def test_fresh_project_has_no_legacy_layout(self, fresh_project):
        """Projeto criado pela versão nova nunca tem layout legado."""
        reopened = FaceAuditProjectService.open(
            fresh_project.fa_file, verify_integrity=False,
        )
        assert reopened.has_legacy_layout is False

    def test_empty_artifacts_folder_is_not_legacy(self, tmp_path):
        """Pasta `artifacts/` existe mas vazia → não conta como legado.
        `create_project_structure` cria todas as pastas raiz, então a
        existência sozinha não é sinal."""
        proj = FaceAuditProjectService.create(
            folder=tmp_path / "proj_empty_artifacts",
            name="Empty Artifacts",
        )
        # `artifacts/` foi criada vazia pelo create_project_structure
        assert (proj.root / "artifacts").is_dir()
        reopened = FaceAuditProjectService.open(
            proj.fa_file, verify_integrity=False,
        )
        assert reopened.has_legacy_layout is False

    def test_populated_artifacts_folder_marks_legacy(self, tmp_path):
        """Pasta `artifacts/` com qualquer arquivo → has_legacy_layout=True.
        Simula projeto criado em versão pré-reorg."""
        proj = FaceAuditProjectService.create(
            folder=tmp_path / "proj_legacy",
            name="Legacy Project",
        )
        # Simula 1 arquivo legado em artifacts/
        legacy_file = proj.root / "artifacts" / "ST_01_some_output.pkl"
        legacy_file.write_bytes(b"\x80\x04N.")  # pickle dump simples (None)

        reopened = FaceAuditProjectService.open(
            proj.fa_file, verify_integrity=False,
        )
        assert reopened.has_legacy_layout is True

    def test_artifacts_folder_missing_is_not_legacy(self, tmp_path):
        """Projeto sem `artifacts/` (apagada manualmente após reorg) não
        deve disparar migração."""
        proj = FaceAuditProjectService.create(
            folder=tmp_path / "proj_no_artifacts",
            name="No Artifacts",
        )
        # Apaga a pasta
        import shutil
        shutil.rmtree(proj.root / "artifacts")
        reopened = FaceAuditProjectService.open(
            proj.fa_file, verify_integrity=False,
        )
        assert reopened.has_legacy_layout is False

    def test_only_subfolder_in_artifacts_marks_legacy(self, tmp_path):
        """Mesmo só com subpastas (sem arquivos), pasta não-vazia conta.
        Cobre caso onde projeto antigo tem `artifacts/graphics/` (legado
        de uma versão muito antiga, antes do isolamento de graphics)."""
        proj = FaceAuditProjectService.create(
            folder=tmp_path / "proj_subfolder",
            name="Subfolder Only",
        )
        sub = proj.root / "artifacts" / "old_subfolder"
        sub.mkdir(parents=True)
        # iterdir() vê a subpasta — não-vazia
        reopened = FaceAuditProjectService.open(
            proj.fa_file, verify_integrity=False,
        )
        assert reopened.has_legacy_layout is True


class TestStrategyIsolationInHierarchicalLayout:
    """Sanity: o sweep `shutil.rmtree(folder/<sid>)` no _on_remove_strategy
    NÃO pode atingir outras estratégias na mesma pasta hierárquica."""

    def test_remove_one_strategy_preserves_others(self, tmp_path):
        """Dado projeto com 2 estratégias compartilhando models/, scores/,
        results/, apagar models/<sid_A>/ deixa models/<sid_B>/ intacto."""
        import shutil
        from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
        from faceaudit.models.transformation_pipeline_config import (
            TransformationPipelineConfig,
        )
        from faceaudit.storage.local_project_artifact_storage import (
            LocalProjectArtifactStorage,
        )

        s = LocalProjectArtifactStorage(project_root=tmp_path)
        s.create_project_structure()

        # Persiste 2 modelos sob estratégias diferentes
        cfg = TransformationPipelineConfig(name="pipe", transformators=[])
        ref_a = s.save_artifact(
            StrategyArtifactType.TRAINED_PIPELINE, cfg,
            strategy_id="SID_A", step_id="pipe_x", id="TP-aaaaaaaa",
        )
        ref_b = s.save_artifact(
            StrategyArtifactType.TRAINED_PIPELINE, cfg,
            strategy_id="SID_B", step_id="pipe_y", id="TP-bbbbbbbb",
        )

        path_a = tmp_path / ref_a.path
        path_b = tmp_path / ref_b.path
        assert path_a.exists() and path_b.exists()

        # Sweep da estratégia A (mesma lógica do _on_remove_strategy)
        shutil.rmtree(tmp_path / "models" / "SID_A", ignore_errors=True)

        assert not path_a.exists(), "Arquivo da estratégia A deveria ter sido apagado"
        assert path_b.exists(), "Arquivo da estratégia B NÃO deveria ter sido apagado"
        assert (tmp_path / "models" / "SID_B").is_dir()

    def test_sweep_idempotent_when_strategy_folder_absent(self, tmp_path):
        """shutil.rmtree com ignore_errors=True sobre folder inexistente
        não levanta exceção — garantia do uso pelo controller."""
        import shutil
        from faceaudit.storage.local_project_artifact_storage import (
            LocalProjectArtifactStorage,
        )
        s = LocalProjectArtifactStorage(project_root=tmp_path)
        s.create_project_structure()
        # Sweep de estratégia que nunca produziu output — só rodar
        shutil.rmtree(tmp_path / "models" / "SID_NEVER_RAN", ignore_errors=True)
        shutil.rmtree(tmp_path / "scores" / "SID_NEVER_RAN", ignore_errors=True)
        shutil.rmtree(tmp_path / "results" / "SID_NEVER_RAN", ignore_errors=True)
