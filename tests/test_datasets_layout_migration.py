# tests/test_datasets_layout_migration.py
"""Migração não-destrutiva do layout flat `datasets/` pra hierárquico:

    PRIMARY: datasets/<X>.parquet            → datasets/primary/<X>.parquet
    DERIVED: datasets/<sid>_<X>.parquet      → datasets/<strategy_folder>/<X>_<id>.parquet
    DERIVED: datasets/<sid>_<X>_<id>.parquet → datasets/<strategy_folder>/<X>_<id>.parquet

Companions (sidecars + audit log `.audit.json`) são movidos junto com
o parquet principal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faceaudit.enums.resource_category import ResourceCategory
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.storage.local_project_artifact_storage import LocalProjectArtifactStorage


def _ref(path: str) -> StrategyArtifactReference:
    return StrategyArtifactReference(
        artifact_type=StrategyArtifactType.GLOBAL_DATASET,
        path=path,
        hash=None,
    )


def _make_storage(tmp_path) -> LocalProjectArtifactStorage:
    s = LocalProjectArtifactStorage(project_root=tmp_path)
    s.create_project_structure()
    return s


def _make_defn(strategy_id: str, name: str, resources: list) -> LRStrategyDefinition:
    defn = LRStrategyDefinition(strategy_id=strategy_id, name=name)
    for r in resources:
        defn.add_resource(r)
    return defn


def _primary_resource(rid: str, rel_path: str) -> StrategyResource:
    return StrategyResource(
        resource_id=rid,
        name=rid,
        resource_type=StrategyResourceType.DATASET,
        artifact_ref=_ref(rel_path),
        category=ResourceCategory.PRIMARY,
    )


def _derived_resource(rid: str, rel_path: str, technical_id: str) -> StrategyResource:
    res = StrategyResource(
        resource_id=rid,
        name=rid,
        resource_type=StrategyResourceType.DATASET,
        artifact_ref=_ref(rel_path),
        category=ResourceCategory.DERIVED,
    )
    res.id = technical_id  # type: ignore[attr-defined]
    return res


class TestPrimaryMigration:

    def test_primary_moved_to_primary_subfolder(self, tmp_path):
        storage = _make_storage(tmp_path)
        # Cria arquivo no formato flat antigo
        old_path = tmp_path / "datasets" / "BFW Frontais.parquet"
        old_path.write_bytes(b"parquet-data")

        defn = _make_defn(
            "ST_01", "Test",
            [_primary_resource("BFW Frontais", "datasets/BFW Frontais.parquet")],
        )
        report = storage.migrate_legacy_datasets_layout([defn])

        assert len(report["moved_primary"]) == 1
        assert not old_path.exists()
        new_path = tmp_path / "datasets" / "primary" / "BFW Frontais.parquet"
        assert new_path.exists()
        # artifact_ref updated
        assert defn.resources[0].artifact_ref.path == "datasets/primary/BFW Frontais.parquet"

    def test_primary_sidecar_and_audit_moved_together(self, tmp_path):
        storage = _make_storage(tmp_path)
        parquet = tmp_path / "datasets" / "X.parquet"
        parquet.write_bytes(b"p")
        stats = tmp_path / "datasets" / "X_stats.json"
        stats.write_text("{}", encoding="utf-8")
        audit = tmp_path / "datasets" / "X.audit.json"
        audit.write_text("[]", encoding="utf-8")

        defn = _make_defn(
            "ST_01", "Test",
            [_primary_resource("X", "datasets/X.parquet")],
        )
        storage.migrate_legacy_datasets_layout([defn])

        prim = tmp_path / "datasets" / "primary"
        assert (prim / "X.parquet").exists()
        assert (prim / "X_stats.json").exists()
        assert (prim / "X.audit.json").exists()
        # Origens devem estar limpas
        assert not stats.exists()
        assert not audit.exists()

    def test_skip_already_migrated(self, tmp_path):
        """Primary já em `datasets/primary/` não é movido novamente."""
        storage = _make_storage(tmp_path)
        prim_dir = tmp_path / "datasets" / "primary"
        prim_dir.mkdir(parents=True)
        (prim_dir / "Y.parquet").write_bytes(b"p")

        defn = _make_defn(
            "ST_01", "Test",
            [_primary_resource("Y", "datasets/primary/Y.parquet")],
        )
        report = storage.migrate_legacy_datasets_layout([defn])

        assert report["moved_primary"] == []
        assert defn.resources[0].artifact_ref.path == "datasets/primary/Y.parquet"


class TestDerivedMigration:

    def test_derived_with_id_suffix_moved_to_strategy_folder(self, tmp_path):
        storage = _make_storage(tmp_path)
        old_name = "ST_01_CFP - FRONTAIS_DS-abc123.parquet"
        old_path = tmp_path / "datasets" / old_name
        old_path.write_bytes(b"data")

        defn = _make_defn(
            "ST_01", "Cosseno",
            [_derived_resource(
                "CFP - FRONTAIS",
                f"datasets/{old_name}",
                "DS-abc123",
            )],
        )
        report = storage.migrate_legacy_datasets_layout([defn])

        assert len(report["moved_derived"]) == 1
        assert not old_path.exists()
        # strategy_folder = "Cosseno_ST_01"
        new_path = tmp_path / "datasets" / "Cosseno_ST_01" / "CFP - FRONTAIS_DS-abc123.parquet"
        assert new_path.exists()
        assert defn.resources[0].artifact_ref.path == (
            "datasets/Cosseno_ST_01/CFP - FRONTAIS_DS-abc123.parquet"
        )

    def test_derived_without_id_suffix_gets_id_added(self, tmp_path):
        """Arquivo antigo `<sid>_<rid>.parquet` (pré-fix do _DS-id) recebe o id no novo path."""
        storage = _make_storage(tmp_path)
        old_name = "ST_01_FOO.parquet"
        old_path = tmp_path / "datasets" / old_name
        old_path.write_bytes(b"data")

        defn = _make_defn(
            "ST_01", "Cosseno",
            [_derived_resource("FOO", f"datasets/{old_name}", "DS-newid")],
        )
        storage.migrate_legacy_datasets_layout([defn])

        new_path = tmp_path / "datasets" / "Cosseno_ST_01" / "FOO_DS-newid.parquet"
        assert new_path.exists()

    def test_derived_companion_files_moved(self, tmp_path):
        storage = _make_storage(tmp_path)
        old_name = "ST_01_A_DS-x.parquet"
        (tmp_path / "datasets" / old_name).write_bytes(b"data")
        (tmp_path / "datasets" / "ST_01_A_DS-x_stats.json").write_text("{}", encoding="utf-8")

        defn = _make_defn(
            "ST_01", "S",
            [_derived_resource("A", f"datasets/{old_name}", "DS-x")],
        )
        storage.migrate_legacy_datasets_layout([defn])

        new_dir = tmp_path / "datasets" / "S_ST_01"
        assert (new_dir / "A_DS-x.parquet").exists()
        assert (new_dir / "A_DS-x_stats.json").exists()


class TestMigrationFailureCases:

    def test_idempotent_when_dest_already_exists(self, tmp_path):
        """Source + destino existem → preserva destino, apaga source."""
        storage = _make_storage(tmp_path)
        old = tmp_path / "datasets" / "A.parquet"
        old.write_bytes(b"old")
        (tmp_path / "datasets" / "primary").mkdir()
        new = tmp_path / "datasets" / "primary" / "A.parquet"
        new.write_bytes(b"new")

        defn = _make_defn(
            "ST_01", "S",
            [_primary_resource("A", "datasets/A.parquet")],
        )
        storage.migrate_legacy_datasets_layout([defn])

        # Source removida, destino preservado
        assert not old.exists()
        assert new.exists()
        assert new.read_bytes() == b"new"

    def test_skip_when_source_missing(self, tmp_path):
        """Source no JSON mas ausente no disco → reporta failure."""
        storage = _make_storage(tmp_path)
        defn = _make_defn(
            "ST_01", "S",
            [_primary_resource("A", "datasets/A.parquet")],
        )
        report = storage.migrate_legacy_datasets_layout([defn])

        assert len(report["moved_primary"]) == 0
        assert len(report["failures"]) == 1
        # artifact_ref não é tocado
        assert defn.resources[0].artifact_ref.path == "datasets/A.parquet"

    def test_primary_shared_across_strategies_all_refs_updated(self, tmp_path):
        """REGRESSION: o mesmo PRIMARY referenciado por N estratégias precisa
        ter o ref atualizado em TODAS, não apenas na primeira que itera.

        Bug original: a 1ª iteração movia source → dest (sucesso, ref
        atualizado). Iterações seguintes viam source missing e abortavam
        com (False, ['source missing']) — ref ficava stale apontando
        pro path flat.

        Fix: quando source não existe MAS dest existe, trata como
        idempotente (já migrado) e retorna (True, []) pra que o caller
        atualize o ref.
        """
        storage = _make_storage(tmp_path)
        # Arquivo único compartilhado por 3 strategies
        (tmp_path / "datasets" / "shared.parquet").write_bytes(b"data")

        defn1 = _make_defn(
            "ST_01", "A",
            [_primary_resource("shared", "datasets/shared.parquet")],
        )
        defn2 = _make_defn(
            "ST_02", "B",
            [_primary_resource("shared", "datasets/shared.parquet")],
        )
        defn3 = _make_defn(
            "ST_03", "C",
            [_primary_resource("shared", "datasets/shared.parquet")],
        )

        report = storage.migrate_legacy_datasets_layout([defn1, defn2, defn3])

        # Arquivo movido uma vez
        assert (tmp_path / "datasets" / "primary" / "shared.parquet").exists()
        assert not (tmp_path / "datasets" / "shared.parquet").exists()

        # TODAS as 3 definitions têm ref atualizado
        for d in [defn1, defn2, defn3]:
            assert d.resources[0].artifact_ref.path == "datasets/primary/shared.parquet"

        # Reporte conta 3 (uma por definition)
        assert len(report["moved_primary"]) == 3
        assert report["failures"] == []
