"""Ações sobre registros selecionados: excluir e salvar como novo dataset.

As duas já existiam, escondidas no menu de contexto da galeria. Agora
valem também na tabela (a visão padrão) e na toolbar, e a exclusão entra
na cadeia de custódia.

Os testes cobrem o que quebra em silêncio:

- exclusão com filtro ativo precisa alcançar o dataset de ORIGEM — o
  recorte na tela e ele compartilham os mesmos objetos `FaceInfo`, e
  remover só da lista do recorte fazia "Limpar filtro" ressuscitar o que
  foi excluído;
- a entrada de auditoria precisa identificar o que saiu sem virar uma
  lista de 5 mil linhas;
- a seleção da tabela precisa sobreviver à ordenação (linha ≠ posição em
  `dataset.faces`), por isso a moeda comum é o `image_path`.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from faceaudit.enums.audit_operation import AuditOperation  # noqa: E402
from faceaudit.models.face_dataset import FaceDataset  # noqa: E402
from faceaudit.models.face_info import FaceInfo  # noqa: E402
from interface.tabs.dataset_manager.dataset_manager_controller import (  # noqa: E402
    DatasetManagerController,
    _MAX_AUDIT_IDENTIFIERS,
)
from interface.tabs.dataset_manager.remove_faces_dialog import (  # noqa: E402
    RemoveFacesDialog,
)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _dataset(n: int = 6) -> FaceDataset:
    return FaceDataset(
        name="base",
        faces=[
            FaceInfo(image_path=f"/fotos/f{i}.png", identity=f"P{i:03d}")
            for i in range(n)
        ],
    )


class _ControllerStub:
    """Controller reduzido ao que as ações de seleção tocam.

    Instanciar o `DatasetManagerController` de verdade sobe a aba inteira
    (tabela, galeria, toolbar, serviços de IA). Estes testes são sobre a
    REGRA — quem sai do dataset, o que vai para a custódia —, então
    reaproveitamos os métodos reais sobre um objeto mínimo.
    """

    excluir_selecionados = DatasetManagerController.excluir_selecionados
    _faces_from_paths = DatasetManagerController._faces_from_paths
    _audit_remocao_de_faces = DatasetManagerController._audit_remocao_de_faces

    def __init__(self, dataset, dataset_original=None):
        self.dataset = dataset
        self.dataset_original = dataset_original
        self.view = None
        self.auditorias = []
        self.refresh_chamado = 0

    # Colaboradores substituídos — o teste não exercita a UI.
    def sync_changes_from_table(self):
        pass

    def _refresh_after_structural_change(self):
        self.refresh_chamado += 1

    def _audit(self, operation, description, *, metadata=None, **kwargs):
        self.auditorias.append((operation, description, metadata or {}))


@pytest.fixture
def sem_dialogos(monkeypatch):
    """Confirma a exclusão com uma justificativa fixa, sem abrir tela."""
    monkeypatch.setattr(
        RemoveFacesDialog, "exec",
        lambda self: QtWidgets.QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(
        RemoveFacesDialog, "justificativa",
        lambda self: "faces com detecção incorreta",
    )
    monkeypatch.setattr(
        "interface.tabs.dataset_manager.dataset_manager_controller."
        "QMessageBox.information",
        lambda *a, **k: None,
    )


def test_exclusao_remove_do_dataset(app, sem_dialogos):
    ds = _dataset()
    ctrl = _ControllerStub(ds)

    ctrl.excluir_selecionados(["/fotos/f1.png", "/fotos/f3.png"])

    restantes = [f.image_path for f in ds.faces]
    assert restantes == [
        "/fotos/f0.png", "/fotos/f2.png", "/fotos/f4.png", "/fotos/f5.png",
    ]
    assert ctrl.refresh_chamado == 1


def test_exclusao_com_filtro_alcanca_o_dataset_de_origem(app, sem_dialogos):
    """O recorte e a origem compartilham as mesmas faces — as duas listas
    têm de perder o registro, senão "Limpar filtro" o traz de volta."""
    origem = _dataset()
    recorte = FaceDataset(name="recorte", faces=list(origem.faces[:4]))
    ctrl = _ControllerStub(recorte, dataset_original=origem)

    ctrl.excluir_selecionados(["/fotos/f2.png"])

    assert "/fotos/f2.png" not in [f.image_path for f in recorte.faces]
    assert "/fotos/f2.png" not in [f.image_path for f in origem.faces], (
        "a exclusão precisa valer também no dataset de origem"
    )
    assert len(origem.faces) == 5


def test_exclusao_entra_na_custodia_com_justificativa(app, sem_dialogos):
    ds = _dataset()
    ctrl = _ControllerStub(ds)

    ctrl.excluir_selecionados(["/fotos/f0.png", "/fotos/f5.png"])

    assert len(ctrl.auditorias) == 1
    operacao, descricao, meta = ctrl.auditorias[0]
    assert operacao is AuditOperation.REMOVE_FACES
    assert "detecção incorreta" in descricao
    assert meta["removed_count"] == 2
    assert meta["justification"] == "faces com detecção incorreta"
    assert meta["pending_until_save"] is True
    assert set(meta["removed_identifiers"]) == {"/fotos/f0.png", "/fotos/f5.png"}
    assert meta["removed_identifiers_truncated"] is False
    assert len(meta["removed_set_sha256"]) == 64


def test_custodia_trunca_a_lista_mas_mantem_o_hash(app, sem_dialogos):
    """Em exclusões grandes a lista é cortada; o hash do conjunto inteiro
    continua lá, e é ele que prova depois se um registro estava dentro."""
    total = _MAX_AUDIT_IDENTIFIERS + 20
    ds = _dataset(total)
    ctrl = _ControllerStub(ds)

    ctrl.excluir_selecionados([f.image_path for f in list(ds.faces)])

    _op, _desc, meta = ctrl.auditorias[0]
    assert meta["removed_count"] == total
    assert len(meta["removed_identifiers"]) == _MAX_AUDIT_IDENTIFIERS
    assert meta["removed_identifiers_truncated"] is True
    assert len(meta["removed_set_sha256"]) == 64
    assert ds.faces == []


def test_caminho_ambiguo_aborta_sem_excluir(app, sem_dialogos, monkeypatch):
    """Duas faces com o MESMO image_path e só uma selecionada: abortar é o
    certo — operar por caminho atingiria a que o usuário não escolheu."""
    ds = _dataset(3)
    ds.faces.append(FaceInfo(image_path="/fotos/f1.png", identity="P999"))
    ctrl = _ControllerStub(ds)
    avisos = []
    monkeypatch.setattr(
        "interface.tabs.dataset_manager.dataset_manager_controller."
        "QMessageBox.warning",
        lambda *a, **k: avisos.append(a),
    )

    ctrl.excluir_selecionados(["/fotos/f1.png"])

    assert len(ds.faces) == 4, "nada pode ser removido em caso ambíguo"
    assert avisos, "o usuário precisa saber por que a operação parou"
    assert ctrl.auditorias == []


def test_dialogo_de_exclusao_permite_justificativa_vazia(app):
    """Exigir texto produziria 'ok', 'teste', 'limpeza' — ruído na trilha."""
    dlg = RemoveFacesDialog(n_faces=3, nome_dataset="base")
    try:
        assert dlg.justificativa() == ""
    finally:
        dlg.deleteLater()
