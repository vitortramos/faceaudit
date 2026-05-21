"""Testes do DirTemplate + extração via FaceDatasetAdapter.

Cobre migração de schema legado (1 separador) para o novo
(2 pares folder_/filename_), roteamento de colunas (built-in vs
custom → extra_data) e helpers (label_for / column_usages /
is_builtin / known_columns).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter
from faceaudit.models import FaceInfo
from faceaudit.models.dir_template import (
    BUILTIN_COLUMNS,
    IGNORE_TOKEN,
    DirTemplate,
)


# ─────────────────────────── DirTemplate ────────────────────────────


def test_defaults_are_safe():
    t = DirTemplate(name="t")
    assert t.use_folder_separator is False
    assert t.use_filename_separator is False
    assert t.folder_separator == "_"
    assert t.filename_separator == "_"
    assert t.custom_columns == []
    assert t.folder_map == {}
    assert t.filename_map == {}


def test_from_dict_migrates_legacy_single_separator():
    """JSON antigo com `separator` + `use_separator` deve preencher
    os dois pares novos com o mesmo valor."""
    legacy = {
        "name": "legacy",
        "separator": "-",
        "use_separator": True,
        "folder_map": {"0": ["identity"]},
        "filename_map": {"1": "gender"},
    }
    t = DirTemplate.from_dict(legacy)
    assert t.folder_separator == "-"
    assert t.filename_separator == "-"
    assert t.use_folder_separator is True
    assert t.use_filename_separator is True
    # Mapas migrados com chaves inteiras
    assert t.folder_map == {0: ["identity"]}
    assert t.filename_map == {1: "gender"}


def test_from_dict_preserves_new_schema():
    """JSON novo com folder_/filename_ separados não deve ser sobrescrito."""
    new_data = {
        "name": "novo",
        "folder_separator": "/",
        "use_folder_separator": False,
        "filename_separator": "-",
        "use_filename_separator": True,
        "custom_columns": ["pose"],
        "folder_map": {"0": ["identity"]},
        "filename_map": {"2": "pose"},
    }
    t = DirTemplate.from_dict(new_data)
    assert t.folder_separator == "/"
    assert t.use_folder_separator is False
    assert t.filename_separator == "-"
    assert t.use_filename_separator is True
    assert t.custom_columns == ["pose"]


def test_from_dict_legacy_does_not_override_new_when_both_present():
    """Cenário extremo: JSON com TANTO `separator` quanto os novos.
    Os novos têm precedência."""
    mixed = {
        "name": "x",
        "separator": "_",
        "use_separator": True,
        "folder_separator": "|",
        "use_folder_separator": False,
    }
    t = DirTemplate.from_dict(mixed)
    assert t.folder_separator == "|"
    assert t.use_folder_separator is False
    # filename_separator não foi explicitado no novo schema, então o
    # legado vence pra ele.
    assert t.filename_separator == "_"
    assert t.use_filename_separator is True


def test_to_dict_does_not_include_legacy_fields():
    t = DirTemplate(name="t", custom_columns=["pose"])
    d = t.to_dict()
    assert "separator" not in d
    assert "use_separator" not in d
    assert "folder_separator" in d
    assert "filename_separator" in d
    assert d["custom_columns"] == ["pose"]


def test_from_json_roundtrip(tmp_path: Path):
    original = DirTemplate(
        name="t",
        folder_separator="-",
        use_folder_separator=True,
        filename_separator="_",
        use_filename_separator=True,
        custom_columns=["pose", "device"],
        folder_map={0: ["identity"], 1: ["gender", "ethnicity"]},
        filename_map={2: "pose", 3: "device"},
        identity_from_parent=False,
    )
    p = tmp_path / "t.json"
    p.write_text(json.dumps(original.to_dict()), encoding="utf-8")
    loaded = DirTemplate.from_json(str(p))
    assert loaded == original


# ─────────────────────────── helpers de coluna ──────────────────────


def test_known_columns_combines_builtins_and_custom():
    t = DirTemplate(name="t", custom_columns=["pose", "device"])
    cols = t.known_columns()
    assert cols == ["identity", "gender", "ethnicity", "origin", "pose", "device"]


def test_label_for_builtin_returns_pt_br():
    t = DirTemplate(name="t")
    assert t.label_for("gender") == "Gênero"
    assert t.label_for("identity") == "Identidade"
    assert t.label_for("ethnicity") == "Etnia"
    assert t.label_for("origin") == "Origem"


def test_label_for_custom_returns_own_name():
    t = DirTemplate(name="t", custom_columns=["pose"])
    assert t.label_for("pose") == "pose"


def test_is_builtin():
    t = DirTemplate(name="t", custom_columns=["pose"])
    assert t.is_builtin("identity") is True
    assert t.is_builtin("pose") is False
    assert t.is_builtin("desconhecido") is False


def test_column_usages_lists_all_mappings():
    t = DirTemplate(
        name="t",
        custom_columns=["pose"],
        folder_map={0: ["pose", "identity"], 2: ["pose"]},
        filename_map={1: "pose", 3: "gender"},
    )
    u = t.column_usages("pose")
    # 2 em folder + 1 em filename
    assert len(u) == 3
    assert any("Nível 0" in s for s in u)
    assert any("Nível 2" in s for s in u)
    assert any("Parte 1" in s for s in u)

    assert t.column_usages("gender") == ["Nome do arquivo, Parte 3"]
    assert t.column_usages("não-mapeado") == []


# ─────────────────────── Adapter._apply_template ────────────────────


def _make_face(image_path: str) -> FaceInfo:
    return FaceInfo(image_path=image_path)


def test_apply_template_builtin_from_folder():
    template = DirTemplate(
        name="t",
        folder_map={0: ["identity"]},
    )
    face = _make_face("/data/john_doe/img.jpg")
    FaceDatasetAdapter._apply_template(
        face=face,
        file_path=Path("/data/john_doe/img.jpg"),
        root_path=Path("/data"),
        template=template,
    )
    assert face.identity == "john_doe"


def test_apply_template_custom_column_lands_in_extra_data():
    template = DirTemplate(
        name="t",
        custom_columns=["pose"],
        use_filename_separator=True,
        filename_separator="_",
        filename_map={1: "pose"},
    )
    face = _make_face("/data/img_frontal.jpg")
    FaceDatasetAdapter._apply_template(
        face=face,
        file_path=Path("/data/img_frontal.jpg"),
        root_path=Path("/data"),
        template=template,
    )
    assert face.extra_data == {"pose": "frontal"}
    # Não vazou para nenhum atributo built-in
    assert face.identity is None
    assert face.gender is None


def test_apply_template_independent_separators():
    """Folder usa '-', filename usa '_'. Ambos devem ser respeitados."""
    template = DirTemplate(
        name="t",
        custom_columns=["pose"],
        use_folder_separator=True,
        folder_separator="-",
        use_filename_separator=True,
        filename_separator="_",
        folder_map={0: ["identity", "gender"]},
        filename_map={2: "pose"},
    )
    face = _make_face("/root/john-male/img_01_frontal.jpg")
    FaceDatasetAdapter._apply_template(
        face=face,
        file_path=Path("/root/john-male/img_01_frontal.jpg"),
        root_path=Path("/root"),
        template=template,
    )
    assert face.identity == "john"
    assert face.gender == "male"
    assert face.extra_data["pose"] == "frontal"


def test_apply_template_ignore_token_skips():
    template = DirTemplate(
        name="t",
        use_filename_separator=True,
        filename_separator="_",
        filename_map={0: "identity", 1: IGNORE_TOKEN, 2: "gender"},
    )
    face = _make_face("/data/john_ignored_male.jpg")
    FaceDatasetAdapter._apply_template(
        face=face,
        file_path=Path("/data/john_ignored_male.jpg"),
        root_path=Path("/data"),
        template=template,
    )
    assert face.identity == "john"
    assert face.gender == "male"
    # Parte 1 nem foi parar em extra_data
    assert "ignored" not in (face.extra_data.values())


def test_apply_template_unknown_column_silently_skipped():
    """Coluna referenciada no map mas não declarada em custom_columns
    nem em built-ins: ignorada sem levantar exceção (template
    inconsistente — UI deve bloquear, mas adapter é tolerante)."""
    template = DirTemplate(
        name="t",
        # NÃO declaramos 'pose' em custom_columns
        filename_map={0: "pose"},
    )
    face = _make_face("/data/pose_val.jpg")
    FaceDatasetAdapter._apply_template(
        face=face,
        file_path=Path("/data/pose_val.jpg"),
        root_path=Path("/data"),
        template=template,
    )
    assert face.extra_data == {}


def test_apply_template_identity_from_parent_takes_precedence():
    template = DirTemplate(
        name="t",
        filename_map={0: "identity"},
        identity_from_parent=True,
    )
    face = _make_face("/root/alice/img.jpg")
    FaceDatasetAdapter._apply_template(
        face=face,
        file_path=Path("/root/alice/img.jpg"),
        root_path=Path("/root"),
        template=template,
    )
    # identity_from_parent é aplicado no fim e sobrescreve qualquer
    # extração anterior.
    assert face.identity == "alice"
