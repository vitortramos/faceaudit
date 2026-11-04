"""Cadeia de custódia de Juntar, Dividir e Auditar; justificativa obrigatória.

Lacunas encontradas numa conferência:
  - Juntar e Dividir produziam datasets novos sem nenhuma entrada de
    custódia — o arquivo derivado não dizia de onde veio (MERGE e SPLIT
    existiam no enum e ninguém os gravava);
  - exclusões, renomeações e mesclas feitas no diálogo Auditar não pediam
    justificativa nem geravam entrada — só o save seguinte ficava registrado;
  - a exclusão pela Tabela/Galeria aceitava justificativa vazia.

O Juntar produz um dataset em memória, sem arquivo: a entrada fica pendente
e tem de chegar à cadeia do arquivo quando ele for gravado.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from faceaudit.enums.audit_operation import AuditOperation  # noqa: E402
from faceaudit.models.face_dataset import FaceDataset  # noqa: E402
from faceaudit.models.face_info import FaceInfo  # noqa: E402
from faceaudit.models.filter import FilterConfig  # noqa: E402
from faceaudit.services import dataset_custody_service as custody  # noqa: E402
from faceaudit.services.audit_log_service import (  # noqa: E402
    AuditLogService,
    JustificationRequiredError,
)
from faceaudit.services.hash_service import HashService  # noqa: E402
from interface.tabs.dataset_manager import dataset_manager_controller as dmc  # noqa: E402
from interface.tabs.dataset_manager.dataset_analysis import (  # noqa: E402
    dataset_analysis_controller as dac,
)
from interface.tabs.dataset_manager.dataset_analysis.dataset_analysis_controller import (  # noqa: E402
    DatasetAnalysisController,
)
from interface.tabs.dataset_manager.justification_dialog import (  # noqa: E402
    JustificationDialog,
)
from interface.tabs.dataset_manager.remove_faces_dialog import (  # noqa: E402
    RemoveFacesDialog,
)

Ctrl = dmc.DatasetManagerController


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def sem_caixas(monkeypatch):
    """Todas as caixas de mensagem respondem sem abrir tela (Sim nas perguntas)."""
    for mod in (dmc, dac):
        monkeypatch.setattr(mod.QMessageBox, "information", lambda *a, **k: None)
        monkeypatch.setattr(mod.QMessageBox, "warning", lambda *a, **k: None)
        monkeypatch.setattr(
            mod.QMessageBox, "question",
            lambda *a, **k: mod.QMessageBox.StandardButton.Yes,
        )


def _faces(prefixo, n, ident="P"):
    return [
        FaceInfo(image_path=f"/img/{prefixo}{i}.png", file_name=f"{prefixo}{i}.png",
                 identity=f"{ident}{i % 3}", hash=f"{prefixo}h{i}")
        for i in range(n)
    ]


def _sha(p):
    return HashService().calculate_file_hash(p)


# ── Backend: justificativa obrigatória ─────────────────────────────────────


@pytest.mark.parametrize("motivo", [None, "", "   ", "\n\t"])
def test_cadeia_recusa_remove_faces_sem_justificativa(tmp_path, motivo):
    alvo = tmp_path / "d.parquet"
    with pytest.raises(JustificationRequiredError):
        AuditLogService.append(
            alvo, AuditOperation.REMOVE_FACES, "x",
            metadata={"justification": motivo},
        )
    assert AuditLogService.load(alvo) == []
    with pytest.raises(JustificationRequiredError):
        custody.removal_entry(_faces("a", 2), motivo)
    with pytest.raises(JustificationRequiredError):
        custody.queue_pending(SimpleNamespace(), AuditOperation.REMOVE_FACES, "x",
                              {"justification": motivo})


def test_remove_faces_com_justificativa_e_aceita(tmp_path):
    descricao, meta = custody.removal_entry(_faces("a", 3), "  duplicadas ")
    assert meta["justification"] == "duplicadas"
    assert meta["removed_count"] == 3
    assert len(meta["removed_set_sha256"]) == 64
    AuditLogService.append(tmp_path / "d.parquet", AuditOperation.REMOVE_FACES,
                           descricao, metadata=meta)


def test_dialogo_generico_so_confirma_com_texto(app):
    dlg = JustificationDialog(titulo_janela="t", titulo_html="<b>t</b>")
    try:
        assert not dlg.pode_confirmar()
        dlg.set_justificativa("motivo")
        assert dlg.pode_confirmar()
    finally:
        dlg.deleteLater()


# ── Juntar ─────────────────────────────────────────────────────────────────


class _MergeStub:
    """Controller reduzido ao que Juntar + Salvar Como tocam."""

    _AMOSTRA_VERIFICACAO_MERGE = Ctrl._AMOSTRA_VERIFICACAO_MERGE
    _amostrar_imagens_ausentes = Ctrl.__dict__["_amostrar_imagens_ausentes"]
    for _n in ("fluxo_merge", "_registrar_juntar", "_descrever_origem",
               "_descrever_conteudo_salvo", "_audit",
               "_escrever_pendencias_de_custodia",
               "_gravar_e_finalizar_salvar_como"):
        locals()[_n] = getattr(Ctrl, _n)
    del _n

    def __init__(self, dataset, path):
        self.dataset = dataset
        self.current_path = str(path) if path else None
        self.dataset_original = None
        self.current_filter_config = FilterConfig()
        self._last_path_rebase = None
        self.view = None
        self.dirty = False

    def sync_changes_from_table(self):
        pass

    def _get_initial_dir(self):
        return ""

    def is_dirty(self):
        return self.dirty

    def set_dirty(self, v):
        self.dirty = v

    def carregar_dataset_na_tela(self, dataset, path=None, display_df=None):
        self.dataset = dataset
        self.current_path = path


def _par_de_datasets(tmp_path, *, b_alinhado=False):
    a = FaceDataset(name="A", faces=_faces("a", 4))
    pa = tmp_path / "A.parquet"
    a.to_parquet(pa)
    AuditLogService.append(pa, AuditOperation.CREATE, "A criado")
    b = FaceDataset(name="B", faces=_faces("b", 3, ident="Q"), is_aligned=b_alinhado)
    pb = tmp_path / "B.parquet"
    b.to_parquet(pb)
    return FaceDataset.from_parquet(pa), pa, pb


def test_juntar_registra_origens_e_grava_na_cadeia_do_arquivo_novo(app, sem_caixas, tmp_path, monkeypatch):
    a, pa, pb = _par_de_datasets(tmp_path, b_alinhado=True)
    monkeypatch.setattr(dmc.QFileDialog, "getOpenFileName",
                        lambda *a_, **k: (str(pb), ""))
    ctrl = _MergeStub(a, pa)

    ctrl.fluxo_merge()

    unido = ctrl.dataset
    assert len(unido) == 7 and ctrl.current_path is None
    # Ainda sem arquivo: a entrada está pendente, não perdida.
    pend = custody.pending_entries(unido)
    assert [p["operation"] for p in pend] == [AuditOperation.MERGE]

    destino = tmp_path / "out" / "unido.parquet"
    ctrl._gravar_e_finalizar_salvar_como(str(destino))

    cadeia = AuditLogService.load(destino)
    assert [e.operation for e in cadeia] == [AuditOperation.MERGE, AuditOperation.SAVE_AS]
    ok, problemas = AuditLogService.verify_chain(cadeia)
    assert ok, problemas
    meta = cadeia[0].metadata
    fa, fb = meta["sources"]
    assert fa["parquet_path"] == str(pa) and fa["parquet_sha256"] == _sha(pa)
    assert fb["parquet_path"] == str(pb) and fb["parquet_sha256"] == _sha(pb)
    assert fa["rows"] == 4 and fb["rows"] == 3
    assert fa["audit_chain_head"] == AuditLogService.load(pa)[-1].entry_hash
    assert fb["audit_chain_head"] is None
    assert meta["rows"] == 7
    assert meta["mixed_alignment"] is True
    assert meta["mixed_alignment_confirmed_by_operator"] is True
    assert meta["is_aligned"] is False
    assert meta["written_on_save"] is True and meta["recorded_at"]
    assert "software" in meta
    assert cadeia[-1].dataset_hash == _sha(destino)
    # Gravado uma vez, não volta a ser escrito no próximo save.
    assert custody.pending_entries(ctrl.dataset) == []


def test_juntar_recusado_nao_registra_nada(app, tmp_path, monkeypatch, sem_caixas):
    a, pa, pb = _par_de_datasets(tmp_path, b_alinhado=True)
    monkeypatch.setattr(dmc.QFileDialog, "getOpenFileName",
                        lambda *a_, **k: (str(pb), ""))
    monkeypatch.setattr(dmc.QMessageBox, "question",
                        lambda *a_, **k: dmc.QMessageBox.StandardButton.No)
    ctrl = _MergeStub(a, pa)
    ctrl.fluxo_merge()
    assert ctrl.dataset is a
    assert custody.pending_entries(a) == []


def test_entrada_de_dataset_sem_arquivo_fica_pendente_ate_gravar(tmp_path):
    """Exclusão feita no resultado do Juntar, antes de ele ter arquivo."""
    ds = FaceDataset(name="mem", faces=_faces("m", 3))
    ctrl = _MergeStub(ds, None)
    descricao, meta = custody.removal_entry(ds.faces[:1], "fora do escopo")
    ctrl._audit(AuditOperation.REMOVE_FACES, descricao, metadata=meta)
    ctrl._audit(AuditOperation.CLOSE, "fechar")  # não descreve conteúdo: não adia
    assert [p["operation"] for p in custody.pending_entries(ds)] == [
        AuditOperation.REMOVE_FACES]

    destino = tmp_path / "mem.parquet"
    ctrl._gravar_e_finalizar_salvar_como(str(destino))
    ops = [e.operation for e in AuditLogService.load(destino)]
    assert ops == [AuditOperation.REMOVE_FACES, AuditOperation.SAVE_AS]


# ── Dividir ────────────────────────────────────────────────────────────────


class _SplitStub:
    fluxo_split = Ctrl.fluxo_split
    for _n in ("_registrar_divisao", "_descrever_origem",
               "_descrever_conteudo_salvo", "_audit"):
        locals()[_n] = getattr(Ctrl, _n)
    del _n

    def __init__(self, dataset, path, storage):
        self.dataset = dataset
        self.current_path = str(path)
        self.dataset_original = None
        self.current_filter_config = FilterConfig()
        self.pipeline = SimpleNamespace(storage_path=storage)
        self.view = None
        self.carregados = []

    def sync_changes_from_table(self):
        pass

    def is_dirty(self):
        return False

    def carregar_dataset_na_tela(self, dataset, path=None, display_df=None):
        self.carregados.append(path)


def _fake_split_dialog(config):
    class _Dlg:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Accepted

        def get_split_config(self):
            return dict(config)
    return _Dlg


@pytest.mark.parametrize("modo", ["ratio", "manual"])
def test_dividir_registra_origem_criterio_e_contagens_em_cada_metade(app, sem_caixas, tmp_path, monkeypatch, modo):
    origem = FaceDataset(name="orig", faces=_faces("o", 12))
    po = tmp_path / "orig.parquet"
    origem.to_parquet(po)
    AuditLogService.append(po, AuditOperation.CREATE, "criado")
    origem = FaceDataset.from_parquet(po)
    saida = tmp_path / "split"
    saida.mkdir()
    config = {"name_a": "parteA", "name_b": "parteB", "mode": modo,
              "ratio": 0.67, "manual_ids_b": {"P1"}, "post_action": 0}
    monkeypatch.setattr(dmc, "SplitDatasetDialogController", _fake_split_dialog(config))

    ctrl = _SplitStub(origem, po, saida)
    ctrl.fluxo_split()

    metas = {}
    for rotulo, nome in (("A", "parteA"), ("B", "parteB")):
        arq = saida / f"{nome}.parquet"
        cadeia = AuditLogService.load(arq)
        assert [e.operation for e in cadeia] == [AuditOperation.SPLIT]
        assert cadeia[0].dataset_hash == _sha(arq)
        meta = cadeia[0].metadata
        assert meta["part"] == rotulo
        assert meta["source"]["parquet_sha256"] == _sha(po)
        assert meta["source"]["parquet_path"] == str(po)
        assert meta["source"]["rows"] == 12
        assert meta["source"]["has_unsaved_changes"] is False
        assert meta["source"]["audit_chain_head"] == AuditLogService.load(po)[-1].entry_hash
        assert meta["rows"] == len(FaceDataset.from_parquet(arq))
        metas[rotulo] = meta
    assert metas["A"]["rows"] + metas["B"]["rows"] == 12
    assert metas["A"]["sibling"]["rows"] == metas["B"]["rows"]
    assert metas["A"]["sibling"]["name"] == "parteB"
    crit = metas["A"]["criterion"]
    assert crit["mode"] == modo
    if modo == "ratio":
        assert crit["ratio_identities_to_a"] == 0.67
        assert crit["random_seed"] == custody.default_split_seed() == 42
    else:
        assert crit["ids_for_b"] == ["P1"] and crit["ids_for_b_count"] == 1
        assert "random_seed" not in crit
    # A origem não é tocada.
    assert [e.operation for e in AuditLogService.load(po)] == [AuditOperation.CREATE]


# ── Auditar ────────────────────────────────────────────────────────────────


class _ViewStub(QtWidgets.QWidget):
    """Janela do Auditar reduzida a no-ops (precisa ser QWidget: é o pai
    dos diálogos de justificativa)."""

    def __getattr__(self, nome):
        return lambda *a, **k: None


class _AnaliseStub:
    """DatasetAnalysisController sem a janela: só a regra das ações."""

    for _n in ("_on_delete_issues", "_on_rename_identity", "_on_merge_identities",
               "_on_undo", "_apply_and_finalize_undo", "_push_action",
               "_sync_undo_button", "_compute_active_marks",
               "applied_actions", "undone_count",
               "_pedir_justificativa_exclusao", "_pedir_justificativa"):
        locals()[_n] = getattr(DatasetAnalysisController, _n)
    _colher_justificativa = DatasetAnalysisController.__dict__["_colher_justificativa"]
    for _n in ("_describe_delete", "_describe_rename", "_describe_merge",
               "_summarize_filenames", "_short_filename"):
        locals()[_n] = DatasetAnalysisController.__dict__[_n]
    del _n

    def __init__(self, dataset):
        self.dataset = dataset
        self.view = _ViewStub()
        self.dataset_modified = mock.MagicMock()
        self._action_history = []
        self._undone_count = 0
        self.session_id = "sessao-teste"

    def _invalidate_heavy_caches(self):
        pass

    def _run_quick_audits(self):
        pass


@pytest.fixture
def justificar(monkeypatch):
    """Diálogos de justificativa confirmam com o texto pedido."""
    estado = {"texto": "motivo pericial", "aceitar": True}

    def _exec(self):
        return (QtWidgets.QDialog.DialogCode.Accepted if estado["aceitar"]
                else QtWidgets.QDialog.DialogCode.Rejected)
    monkeypatch.setattr(JustificationDialog, "exec", _exec)
    monkeypatch.setattr(JustificationDialog, "justificativa",
                        lambda self: estado["texto"].strip())
    return estado


def test_auditar_excluir_pede_justificativa_e_cancelar_nao_exclui(app, sem_caixas, justificar):
    ds = FaceDataset(name="d", faces=_faces("a", 4))
    an = _AnaliseStub(ds)
    justificar["aceitar"] = False
    an._on_delete_issues(["/img/a0.png"])
    assert len(ds.faces) == 4 and an.applied_actions() == []

    justificar["aceitar"] = True
    justificar["texto"] = "   "
    an._on_delete_issues(["/img/a0.png"])
    assert len(ds.faces) == 4, "justificativa em branco não pode excluir"

    justificar["texto"] = "detecção errada"
    an._on_delete_issues(["/img/a0.png"])
    assert len(ds.faces) == 3
    [acao] = an.applied_actions()
    assert acao.kind == "delete" and acao.justification == "detecção errada"
    assert acao.applied_at


def test_auditar_registra_ao_fechar_so_o_que_ficou_aplicado(app, sem_caixas, justificar, tmp_path, monkeypatch):
    ds = FaceDataset(name="d", faces=_faces("a", 6))
    arq = tmp_path / "d.parquet"
    ds.to_parquet(arq)
    an = _AnaliseStub(ds)

    justificar["texto"] = "duplicadas"
    an._on_delete_issues(["/img/a0.png", "/img/a1.png"])
    # Renomear P2 → Fulano (afeta a2 e a5)
    monkeypatch.setattr(dac.QInputDialog, "getText", lambda *a, **k: ("Fulano", True))
    justificar["texto"] = "grafia conforme documento"
    an._on_rename_identity(["/img/a2.png"])
    # Mesclar P1 (a4) e Fulano → Fulano
    monkeypatch.setattr(dac.QInputDialog, "getItem", lambda *a, **k: ("Fulano", True))
    justificar["texto"] = "mesma pessoa"
    an._on_merge_identities(["/img/a4.png", "/img/a2.png"])
    # Desfaz a mescla: não pode aparecer na cadeia.
    an._on_undo()
    assert an.undone_count() == 1
    assert [a.kind for a in an.applied_actions()] == ["delete", "rename"]

    ctrl = SimpleNamespace(dataset=ds, dataset_original=None, current_path=str(arq))
    ctrl._audit = Ctrl._audit.__get__(ctrl)
    Ctrl._registrar_acoes_da_auditoria(ctrl, an)

    cadeia = AuditLogService.load(arq)
    assert [e.operation for e in cadeia] == [
        AuditOperation.REMOVE_FACES, AuditOperation.EDIT_LABEL]
    ok, problemas = AuditLogService.verify_chain(cadeia)
    assert ok, problemas
    rem, ren = (e.metadata for e in cadeia)
    assert rem["justification"] == "duplicadas"
    assert rem["removed_count"] == 2
    assert set(rem["removed_identifiers"]) == {"ah0", "ah1"}
    assert rem["origin"] == "audit_dialog" and rem["pending_until_save"] is True
    assert rem["audit_session_actions_undone"] == 1
    assert rem["audit_session_actions_applied"] == 2
    assert ren["action"] == "rename_identity"
    assert ren["old_identities"] == ["P2"] and ren["new_identity"] == "Fulano"
    assert ren["changed_count"] == 2
    assert set(ren["changed_identifiers"]) == {"ah2", "ah5"}
    assert ren["justification"] == "grafia conforme documento"
    assert ren["applied_at"]


def test_auditar_com_filtro_exclui_tambem_da_origem(app, sem_caixas, justificar, tmp_path):
    origem = FaceDataset(name="o", faces=_faces("a", 5))
    recorte = FaceDataset(name="r", faces=list(origem.faces[:3]))
    an = _AnaliseStub(recorte)
    an._on_delete_issues(["/img/a1.png"])
    ctrl = SimpleNamespace(dataset=recorte, dataset_original=origem, current_path=None)
    ctrl._audit = Ctrl._audit.__get__(ctrl)
    Ctrl._registrar_acoes_da_auditoria(ctrl, an)
    assert "/img/a1.png" not in [f.image_path for f in origem.faces]
    # Sem arquivo: a entrada ficou pendente no dataset raiz.
    assert [p["operation"] for p in custody.pending_entries(origem)] == [
        AuditOperation.REMOVE_FACES]
