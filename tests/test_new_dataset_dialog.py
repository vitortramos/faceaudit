"""Diálogo "Novo dataset" redesenhado (Etapa 1) — view em 3 grupos + máquina de
estados condicional + validação reativa (offscreen, sem GPU/ForensicFace).

Cobre: estado inicial, visibilidade condicional (autocontido→tipo de imagem;
entrada alinhada→'face original' cinza; template→troca identidade↔template;
identidade única→campos), validação reativa (OK + bloqueio) e o mapeamento de
get_config para o DatasetIngestionConfig existente.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _ctrl(tmp_path, *, ff=True, ofiq=True, with_template=False,
         use_gpu=False, batch_size=32):
    from interface.tabs.dataset_manager.new_dataset_dialog.new_dataset_controller import (
        NewDatasetController,
    )
    tdir = tmp_path / "templates"
    tdir.mkdir(exist_ok=True)
    if with_template:
        (tdir / "t1.json").write_text(json.dumps({"name": "t1"}), encoding="utf-8")
    return NewDatasetController(None, templates_path=tdir, ff_available=ff,
                                ofiq_available=ofiq, default_dataset_dir="G:/datasets",
                                default_use_gpu=use_gpu, default_batch_size=batch_size)


def _ok(v):
    return v.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)


def test_initial_state(app, tmp_path):
    c = _ctrl(tmp_path); v = c.view   # segura o controller (senão GC mata o sinal changed→_revalidate)
    assert v.get_dataset_dir() == "G:/datasets"      # destino default preenchido
    assert v.rb_absolute.isChecked()                  # absoluto por padrão
    assert not v.row_imgtype.isVisibleTo(v)           # tipo de imagem só no autocontido
    assert v.row_identity.isVisibleTo(v)              # identidade visível
    assert not v.row_template.isVisibleTo(v)          # template oculto
    assert not _ok(v).isEnabled()                     # sem caminho → OK bloqueado


def test_validation_path_and_name(app, tmp_path):
    c = _ctrl(tmp_path); v = c.view   # segura o controller (senão GC mata o sinal changed→_revalidate)
    v.txt_path.setText("C:/imgs")
    assert not _ok(v).isEnabled()                     # falta nome
    v.txt_name.setText("ds1")
    assert _ok(v).isEnabled()
    assert v.lbl_blocker.text() == ""


def test_autocontido_reveals_image_type(app, tmp_path):
    c = _ctrl(tmp_path); v = c.view   # segura o controller (senão GC mata o sinal changed→_revalidate)
    v.rb_self_contained.setChecked(True)
    assert v.row_imgtype.isVisibleTo(v)
    assert v.rb_img_original.isEnabled()              # entrada = originais


def test_input_aligned_greys_face_original(app, tmp_path):
    c = _ctrl(tmp_path); v = c.view   # segura o controller (senão GC mata o sinal changed→_revalidate)
    v.rb_self_contained.setChecked(True)
    v.rb_input_aligned.setChecked(True)
    assert not v.rb_img_original.isEnabled()          # já alinhadas → sem "face original"
    assert v.rb_img_aligned.isChecked()               # força "face alinhada"


def test_template_swaps_identity_block(app, tmp_path):
    c = _ctrl(tmp_path, with_template=True)
    v = c.view
    v.txt_path.setText("C:/imgs"); v.txt_name.setText("ds1")
    v.chk_use_template.setChecked(True)
    assert v.row_template.isVisibleTo(v) and not v.row_identity.isVisibleTo(v)
    assert not _ok(v).isEnabled()                     # template ON + "Nenhum" → bloqueia
    v.cmb_template.setCurrentIndex(1)                 # seleciona t1
    assert _ok(v).isEnabled()
    cfg = c.get_config()
    assert cfg.recursive is True and cfg.template is not None and cfg.template.name == "t1"


def test_single_identity_requires_value(app, tmp_path):
    c = _ctrl(tmp_path)
    v = c.view
    v.txt_path.setText("C:/imgs"); v.txt_name.setText("ds1")
    v.rb_id_single.setChecked(True)
    assert v.row_id_single.isVisibleTo(v)
    assert not _ok(v).isEnabled()                     # falta identidade
    v.txt_identity_flat.setText("Fulano")
    v.cmb_gender.setCurrentText("Male"); v.txt_ethnicity.setText("Parda")
    assert _ok(v).isEnabled()
    cfg = c.get_config()
    assert cfg.recursive is False and cfg.default_identity == "Fulano"
    assert cfg.default_gender == "Male" and cfg.default_ethnicity == "Parda"


def test_subdir_identity_is_recursive_unknown(app, tmp_path):
    c = _ctrl(tmp_path)
    c.view.txt_path.setText("C:/imgs"); c.view.txt_name.setText("ds1")
    cfg = c.get_config()
    assert cfg.recursive is True and cfg.default_identity == "Unknown"


def test_align_now_mapping(app, tmp_path):
    # originais + autocontido + face alinhada → align_during + aligned_target_dir.
    c = _ctrl(tmp_path)
    v = c.view
    v.txt_path.setText("C:/imgs"); v.txt_name.setText("ds1")
    v.rb_self_contained.setChecked(True)
    v.rb_img_aligned.setChecked(True)
    cfg = c.get_config()
    assert cfg.is_aligned is False and cfg.align_during_ingestion is True
    assert str(cfg.aligned_target_dir) == str(__import__("pathlib").Path("G:/datasets"))


def test_face_original_no_align(app, tmp_path):
    c = _ctrl(tmp_path)
    v = c.view
    v.txt_path.setText("C:/imgs"); v.txt_name.setText("ds1")
    v.rb_self_contained.setChecked(True); v.rb_img_original.setChecked(True)
    cfg = c.get_config()
    assert cfg.align_during_ingestion is False and cfg.is_aligned is False


def test_input_aligned_sets_is_aligned(app, tmp_path):
    c = _ctrl(tmp_path)
    v = c.view
    v.txt_path.setText("C:/imgs"); v.txt_name.setText("ds1")
    v.rb_input_aligned.setChecked(True)
    cfg = c.get_config()
    assert cfg.is_aligned is True and cfg.align_during_ingestion is False


def test_quality_defaults(app, tmp_path):
    v = _ctrl(tmp_path).view
    assert v.chk_fiqa.isChecked()                      # FIQA marcado por padrão
    assert not v.chk_ofiq.isChecked()                  # OFIQ desmarcado (lento)


def test_service_availability_disables_stages(app, tmp_path):
    c = _ctrl(tmp_path, ff=False, ofiq=False); v = c.view
    assert all(not ch.isEnabled() for ch in v._ff_model_checks.values())
    assert not v.chk_ofiq.isEnabled() and not v.chk_ofiq.isChecked()
    # todos os atributos FF ficam desabilitados sem ForensicFace
    assert not v.chk_fiqa.isEnabled() and not v.chk_gender.isEnabled()
    assert not v.chk_yaw.isEnabled() and not v.chk_det_conf.isEnabled()
    assert not v.rb_img_aligned.isEnabled()           # sem FF não dá pra alinhar


def test_input_aligned_hides_original_only(app, tmp_path):
    c = _ctrl(tmp_path); v = c.view
    assert v._pose_row_w.isVisibleTo(v) and v._outros_row_w.isVisibleTo(v)
    assert v._ofiq_w.isVisibleTo(v) and v._ff_det_size_spin.isVisibleTo(v)
    v.rb_input_aligned.setChecked(True)
    # tudo que depende da foto original some; det size vira dica
    assert not v._pose_row_w.isVisibleTo(v)
    assert not v._outros_row_w.isVisibleTo(v)
    assert not v._ofiq_w.isVisibleTo(v)
    assert not v._ff_det_size_spin.isVisibleTo(v)
    assert v._det_size_hint.isVisibleTo(v)


def test_get_config_extended_and_kept(app, tmp_path):
    c = _ctrl(tmp_path); v = c.view
    v.txt_path.setText("C:/imgs"); v.txt_name.setText("ds1")
    cfg = c.get_config()
    assert cfg.ff_extended is True                     # FIQA+pose+gênero+idade on
    assert set(cfg.ff_kept_attributes) == {
        "fiqa", "yaw", "pitch", "roll", "gender", "age", "det_score", "ipd"}
    assert cfg.compute_serfiq is False
    # desmarca todos os atributos do bundle → extended desliga
    for ch in (v.chk_fiqa, v.chk_yaw, v.chk_pitch, v.chk_roll, v.chk_gender, v.chk_age):
        ch.setChecked(False)
    cfg2 = c.get_config()
    assert cfg2.ff_extended is False
    assert "gender" not in cfg2.ff_kept_attributes
    assert "det_score" in cfg2.ff_kept_attributes      # det/ipd não são do bundle


def test_ofiq_disabled_when_input_aligned(app, tmp_path):
    c = _ctrl(tmp_path); v = c.view
    v.txt_path.setText("C:/imgs"); v.txt_name.setText("ds1")
    v.chk_ofiq.setChecked(True)
    assert c.get_config().compute_ofiq is True
    v.rb_input_aligned.setChecked(True)
    assert c.get_config().compute_ofiq is False        # OFIQ exige a original


def test_gpu_batch_defaults_and_override(app, tmp_path):
    c = _ctrl(tmp_path, use_gpu=True, batch_size=16); v = c.view
    v.txt_path.setText("C:/imgs"); v.txt_name.setText("ds1")
    assert v.chk_use_gpu.isChecked() is True           # default do sistema
    assert v.spn_batch_size.value() == 16
    cfg = c.get_config()
    assert cfg.use_gpu is True and cfg.batch_size == 16
    v.chk_use_gpu.setChecked(False); v.spn_batch_size.setValue(64)
    cfg2 = c.get_config()
    assert cfg2.use_gpu is False and cfg2.batch_size == 64


def test_origin_is_a_dataset_column_value(app, tmp_path):
    # Origem é coluna por-face ('origin'), no grupo Conteúdo — não metadado geral.
    c = _ctrl(tmp_path); v = c.view
    v.txt_path.setText("C:/imgs"); v.txt_name.setText("ds1")
    assert c.get_config().origin == "Local"            # default
    v.txt_origin.setText("Caso 123")
    assert c.get_config().origin == "Caso 123"


def test_image_type_getter(app, tmp_path):
    c = _ctrl(tmp_path); v = c.view   # segura o controller (senão GC mata o sinal changed→_revalidate)
    v.rb_absolute.setChecked(True)
    assert v.get_image_type() == "aligned"            # default; só relevante no autocontido
    v.rb_self_contained.setChecked(True); v.rb_img_original.setChecked(True)
    assert v.get_image_type() == "original"
    assert v.get_self_contained() is True
