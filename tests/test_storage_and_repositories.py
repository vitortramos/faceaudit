# tests/test_storage_and_repositories.py
"""
Testes para a camada de storage e repositórios da nova arquitetura declarativa.

Cobre:
- StrategyFileType (enums/strategy_file_type.py)
- LocalProjectArtifactStorage (storage/local_project_artifact_storage.py)
- StrategyDefinitionRepository (repositories/strategy_definition_repository.py)
- StrategyTrainedModelRepository (repositories/strategy_trained_model_repository.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faceaudit.enums.project_folder import ProjectFolder
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_file_type import StrategyFileType
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.lr_strategy_trained_model import LRStrategyTrainedModel
from faceaudit.repositories.strategy_definition_repository import StrategyDefinitionRepository
from faceaudit.repositories.strategy_trained_model_repository import StrategyTrainedModelRepository
from faceaudit.storage.local_project_artifact_storage import LocalProjectArtifactStorage


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def storage(tmp_path) -> LocalProjectArtifactStorage:
    """Storage com estrutura de pastas já criada em diretório temporário."""
    s = LocalProjectArtifactStorage(project_root=tmp_path)
    s.create_project_structure()
    return s


@pytest.fixture
def repos(storage):
    """Repositórios alimentados pelo mesmo storage temporário."""
    return {
        "storage": storage,
        "def_repo": StrategyDefinitionRepository(artifact_storage=storage),
        "model_repo": StrategyTrainedModelRepository(artifact_storage=storage),
    }


def _make_definition(strategy_id: str = "ST_01") -> LRStrategyDefinition:
    return LRStrategyDefinition(strategy_id=strategy_id, name=f"Estratégia {strategy_id}")


def _make_model(strategy_id: str = "ST_01") -> LRStrategyTrainedModel:
    return LRStrategyTrainedModel(strategy_id=strategy_id)


# =============================================================================
# SEÇÃO 1 — StrategyFileType
# =============================================================================

class TestStrategyFileType:

    def test_build_path_definition(self):
        p = StrategyFileType.STRATEGY_DEFINITION.build_path(strategy_id="ST_01")
        assert p == Path("definitions/ST_01_definition.json")

    def test_build_path_trained_model(self):
        p = StrategyFileType.TRAINED_STRATEGY_MODEL.build_path(strategy_id="ST_01")
        assert p == Path("models/ST_01/strategy_model.pkl")

    def test_build_path_trained_pipeline(self):
        p = StrategyFileType.TRAINED_PIPELINE.build_path(strategy_id="ST_01", step_id="pipe_a")
        assert p == Path("models/ST_01/pipeline/pipe_a.pkl")

    def test_build_path_trained_similarity_model(self):
        p = StrategyFileType.TRAINED_SIMILARITY_MODEL.build_path(
            strategy_id="ST_01", resource_id="SIM-1",
        )
        assert p == Path("models/ST_01/similarity/SIM-1.pkl")

    def test_build_path_similarity_scores(self):
        p = StrategyFileType.SIMILARITY_SCORES.build_path(
            strategy_id="ST_01", resource_id="SC-1",
        )
        assert p == Path("scores/ST_01/raw/SC-1.parquet")

    def test_build_path_mined_scores(self):
        p = StrategyFileType.MINED_SCORES.build_path(
            strategy_id="ST_01", resource_id="SC-MIN",
        )
        assert p == Path("scores/ST_01/mined/SC-MIN.parquet")

    def test_build_path_vectorized_scores(self):
        p = StrategyFileType.VECTORIZED_SCORES.build_path(
            strategy_id="ST_01", resource_id="SC-VEC",
        )
        assert p == Path("scores/ST_01/vectorized/SC-VEC.parquet")

    def test_build_path_calibrated_scores(self):
        p = StrategyFileType.CALIBRATED_SCORES.build_path(
            strategy_id="ST_01", resource_id="SC-CAL",
        )
        assert p == Path("scores/ST_01/calibrated/SC-CAL.parquet")

    def test_build_path_validation_result(self):
        p = StrategyFileType.VALIDATION_RESULT.build_path(
            strategy_id="ST_01", resource_id="VAL-1",
        )
        assert p == Path("results/ST_01/validation/VAL-1.pkl")

    def test_build_path_casework_visualization(self):
        p = StrategyFileType.CASEWORK_VISUALIZATION.build_path(
            strategy_id="ST_01", resource_id="CASE-1",
        )
        assert p == Path("results/ST_01/casework/CASE-1.pkl")

    def test_build_path_hubness_ablation_has_distinct_suffix(self):
        """HUBNESS_REPORT e HUBNESS_ABLATION_RESULT compartilham subpasta
        `results/<sid>/hubness/`. Pra não colidirem o ablation tem o
        sufixo `_ablation` no nome do arquivo."""
        report = StrategyFileType.HUBNESS_REPORT.build_path(
            strategy_id="ST_01", resource_id="HUB-1",
        )
        ablation = StrategyFileType.HUBNESS_ABLATION_RESULT.build_path(
            strategy_id="ST_01", resource_id="HUB-1",
        )
        assert report == Path("results/ST_01/hubness/HUB-1.pkl")
        assert ablation == Path("results/ST_01/hubness/HUB-1_ablation.pkl")
        assert report != ablation

    def test_build_path_missing_kwarg_raises(self):
        with pytest.raises(ValueError, match="strategy_id"):
            StrategyFileType.STRATEGY_DEFINITION.build_path()

    def test_for_artifact_definition(self):
        result = StrategyFileType.for_artifact(StrategyArtifactType.STRATEGY_DEFINITION)
        assert result is StrategyFileType.STRATEGY_DEFINITION

    def test_for_artifact_strategy_model(self):
        result = StrategyFileType.for_artifact(StrategyArtifactType.STRATEGY_MODEL)
        assert result is StrategyFileType.TRAINED_STRATEGY_MODEL

    def test_for_artifact_unknown_raises(self):
        with pytest.raises(ValueError):
            StrategyFileType.for_artifact(StrategyArtifactType.UNKNOWN)

    def test_for_artifact_safe_unknown_returns_none(self):
        result = StrategyFileType.for_artifact_safe(StrategyArtifactType.UNKNOWN)
        assert result is None

    def test_for_artifact_safe_known_returns_value(self):
        result = StrategyFileType.for_artifact_safe(StrategyArtifactType.STRATEGY_DEFINITION)
        assert result is StrategyFileType.STRATEGY_DEFINITION


# =============================================================================
# SEÇÃO 2 — LocalProjectArtifactStorage
# =============================================================================

class TestLocalProjectArtifactStorage:

    def test_create_project_structure(self, tmp_path):
        s = LocalProjectArtifactStorage(project_root=tmp_path)
        s.create_project_structure()
        for folder in ProjectFolder:
            if folder.value:
                assert (tmp_path / folder.value).is_dir(), \
                    f"Pasta '{folder.value}' não foi criada"

    def test_create_project_structure_is_idempotent(self, storage, tmp_path):
        # Chamar duas vezes não deve lançar exceção
        storage.create_project_structure()

    def test_save_and_resolve_json(self, storage):
        ref = storage.save_artifact(
            StrategyArtifactType.STRATEGY_DEFINITION,
            {"strategy_id": "ST_01", "name": "Teste"},
            strategy_id="ST_01",
        )
        resolved = storage.resolve_artifact(ref)
        assert resolved is not None
        assert resolved.exists()
        assert resolved.suffix == ".json"

    def test_save_and_resolve_pickle(self, storage):
        model = _make_model("ST_02")
        ref = storage.save_artifact(
            StrategyArtifactType.STRATEGY_MODEL,
            model,
            strategy_id="ST_02",
        )
        resolved = storage.resolve_artifact(ref)
        assert resolved is not None
        assert resolved.exists()
        assert resolved.suffix == ".pkl"

    def test_resolve_nonexistent_returns_none(self, storage):
        ref = StrategyArtifactReference(
            artifact_type=StrategyArtifactType.STRATEGY_DEFINITION,
            path="definitions/nao_existe.json",
            hash=None,
        )
        assert storage.resolve_artifact(ref) is None

    def test_resolve_none_ref_returns_none(self, storage):
        assert storage.resolve_artifact(None) is None

    def test_path_is_relative(self, storage):
        ref = storage.save_artifact(
            StrategyArtifactType.STRATEGY_DEFINITION,
            {"strategy_id": "ST_03"},
            strategy_id="ST_03",
        )
        p = Path(ref.path)
        assert not p.is_absolute(), f"Path deveria ser relativo, mas foi: {ref.path}"

    def test_hash_is_calculated(self, storage):
        ref = storage.save_artifact(
            StrategyArtifactType.STRATEGY_DEFINITION,
            {"strategy_id": "ST_04"},
            strategy_id="ST_04",
        )
        assert ref.hash is not None
        assert len(ref.hash) == 64  # SHA-256 em hex

    def test_list_artifacts_by_type_finds_saved(self, storage):
        storage.save_artifact(
            StrategyArtifactType.STRATEGY_DEFINITION,
            {"strategy_id": "ST_05"},
            strategy_id="ST_05",
        )
        found = storage.list_artifacts(StrategyArtifactType.STRATEGY_DEFINITION)
        assert len(found) == 1
        assert found[0].artifact_type == StrategyArtifactType.STRATEGY_DEFINITION

    def test_save_artifact_creates_nested_subdirs(self, storage, tmp_path):
        """Templates com slashes (`models/<sid>/pipeline/<step_id>`) devem
        criar subpastas automaticamente via `mkdir(parents=True)`."""
        from faceaudit.models.transformation_pipeline_config import (
            TransformationPipelineConfig,
        )
        cfg = TransformationPipelineConfig(name="pipe", transformators=[])
        ref = storage.save_artifact(
            StrategyArtifactType.TRAINED_PIPELINE,
            cfg,
            strategy_id="ST_09",
            step_id="pipe_a",
        )
        resolved = storage.resolve_artifact(ref)
        assert resolved is not None
        assert resolved.exists()
        # Path inclui subpastas — `models/ST_09/pipeline/pipe_a.pkl`
        assert "ST_09" in ref.path
        assert "pipeline" in ref.path
        assert (tmp_path / "models" / "ST_09" / "pipeline").is_dir()

    def test_list_artifacts_finds_files_in_nested_subdirs(self, storage):
        """`list_artifacts` usa rglob — deve achar arquivos em subpastas."""
        from faceaudit.models.transformation_pipeline_config import (
            TransformationPipelineConfig,
        )
        cfg = TransformationPipelineConfig(name="pipe", transformators=[])
        storage.save_artifact(
            StrategyArtifactType.TRAINED_PIPELINE,
            cfg,
            strategy_id="ST_NESTED",
            step_id="step_x",
        )
        found = storage.list_artifacts(StrategyArtifactType.TRAINED_PIPELINE)
        assert len(found) == 1
        assert "ST_NESTED" in found[0].path
        assert "step_x" in found[0].path

    def test_list_artifacts_empty_project(self, storage):
        found = storage.list_artifacts()
        assert found == []

    def test_list_artifacts_all_types(self, storage):
        storage.save_artifact(
            StrategyArtifactType.STRATEGY_DEFINITION,
            {"strategy_id": "ST_06"},
            strategy_id="ST_06",
        )
        storage.save_artifact(
            StrategyArtifactType.STRATEGY_MODEL,
            _make_model("ST_06"),
            strategy_id="ST_06",
        )
        found = storage.list_artifacts()
        assert len(found) >= 2

    def test_copy_artifact_to_project(self, storage, tmp_path):
        # Cria arquivo externo
        src = tmp_path / "externo.json"
        src.write_text(json.dumps({"strategy_id": "ST_07"}), encoding="utf-8")

        ref = storage.copy_artifact_to_project(
            source_path=src,
            artifact_type=StrategyArtifactType.STRATEGY_DEFINITION,
            strategy_id="ST_07",
        )
        resolved = storage.resolve_artifact(ref)
        assert resolved is not None
        assert resolved.exists()
        assert not Path(ref.path).is_absolute()

    def test_copy_nonexistent_raises(self, storage, tmp_path):
        with pytest.raises(FileNotFoundError):
            storage.copy_artifact_to_project(
                source_path=tmp_path / "fantasma.json",
                artifact_type=StrategyArtifactType.STRATEGY_DEFINITION,
                strategy_id="ST_08",
            )


# =============================================================================
# SEÇÃO 3 — Repositórios
# =============================================================================

class TestStrategyDefinitionRepository:

    def test_save_and_load_roundtrip(self, repos):
        repo = repos["def_repo"]
        original = _make_definition("ST_10")
        repo.save_definition(original)

        loaded = repo.load_definition("ST_10")
        assert loaded is not None
        assert loaded.strategy_id == original.strategy_id
        assert loaded.name == original.name

    def test_load_definition_not_found(self, repos):
        assert repos["def_repo"].load_definition("INEXISTENTE") is None

    def test_load_definition_empty_id(self, repos):
        assert repos["def_repo"].load_definition("") is None

    def test_list_definitions_empty(self, repos):
        assert repos["def_repo"].list_definitions() == []

    def test_list_definitions_multiple(self, repos):
        repo = repos["def_repo"]
        for i in range(3):
            repo.save_definition(_make_definition(f"ST_2{i}"))
        found = repo.list_definitions()
        assert len(found) == 3

    def test_list_definitions_tolerates_bad_file(self, repos, tmp_path):
        # Injeta arquivo JSON corrompido na pasta definitions/
        bad_file = tmp_path / "definitions" / "lixo_definition.json"
        bad_file.write_text("{ isso não é json válido !!!}", encoding="utf-8")

        repos["def_repo"].save_definition(_make_definition("ST_30"))
        found = repos["def_repo"].list_definitions()
        # Arquivo corrompido ignorado; apenas a definição válida é retornada
        assert len(found) == 1
        assert found[0].strategy_id == "ST_30"

    def test_list_definitions_ignores_non_canonical_filenames(self, repos, tmp_path):
        """Arquivos JSON válidos mas com nome fora da forma canônica
        `<strategy_id>_definition.json` (ex: backups manuais com sufixo
        `_definition.backup_<timestamp>.json`) NÃO devem ser carregados —
        senão duplicariam `artifact_ref`s e poluiriam a verificação de
        integridade com hash stale do mesmo path físico.
        """
        repo = repos["def_repo"]
        # Salva uma definition canônica
        repo.save_definition(_make_definition("ST_40"))

        # Cria um "backup" da mesma definition com nome não-canônico
        canonical = tmp_path / "definitions" / "ST_40_definition.json"
        backup = tmp_path / "definitions" / "ST_40_definition.backup_20260522.json"
        backup.write_text(canonical.read_text(encoding="utf-8"), encoding="utf-8")

        # Também cria um arquivo com strategy_id divergente do nome
        impostor = tmp_path / "definitions" / "ST_99_definition.json"
        defn_impostor = _make_definition("ST_41")  # strategy_id ≠ ST_99
        impostor.write_text(json.dumps(defn_impostor.to_dict()), encoding="utf-8")

        found = repo.list_definitions()
        # Apenas a definition canônica é retornada
        assert len(found) == 1
        assert found[0].strategy_id == "ST_40"

    def test_save_definition_none_raises(self, repos):
        with pytest.raises((ValueError, AttributeError)):
            repos["def_repo"].save_definition(None)


class TestStrategyTrainedModelRepository:

    def test_save_and_load_roundtrip(self, repos):
        repo = repos["model_repo"]
        original = _make_model("ST_40")
        repo.save_model(original)

        loaded = repo.load_model("ST_40")
        assert loaded is not None
        assert loaded.strategy_id == original.strategy_id

    def test_load_model_not_found(self, repos):
        assert repos["model_repo"].load_model("INEXISTENTE") is None

    def test_load_model_empty_id(self, repos):
        assert repos["model_repo"].load_model("") is None

    def test_list_models_empty(self, repos):
        assert repos["model_repo"].list_models() == []

    def test_list_models_multiple(self, repos):
        repo = repos["model_repo"]
        for i in range(3):
            repo.save_model(_make_model(f"ST_5{i}"))
        found = repo.list_models()
        assert len(found) == 3

    def test_save_model_none_raises(self, repos):
        with pytest.raises((ValueError, AttributeError)):
            repos["model_repo"].save_model(None)


class TestIntegration:

    def test_full_roundtrip_same_storage(self, repos):
        """
        Mesmo LocalProjectArtifactStorage alimenta ambos os repositórios.
        Salvar e carregar definição + modelo deve funcionar de forma independente.
        """
        def_repo = repos["def_repo"]
        model_repo = repos["model_repo"]

        definition = _make_definition("ST_99")
        model = _make_model("ST_99")

        def_repo.save_definition(definition)
        model_repo.save_model(model)

        loaded_def = def_repo.load_definition("ST_99")
        loaded_model = model_repo.load_model("ST_99")

        assert loaded_def is not None
        assert loaded_model is not None
        assert loaded_def.strategy_id == "ST_99"
        assert loaded_model.strategy_id == "ST_99"

        # Listar também deve funcionar
        assert len(def_repo.list_definitions()) == 1
        assert len(model_repo.list_models()) == 1
