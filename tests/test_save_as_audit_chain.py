"""Salvar Como preserva a cadeia de custódia da origem e registra o filtro.

Dois defeitos encontrados num caso real:
  - o Salvar Como MOVIA o `.audit.json` para o arquivo novo, e o dataset de
    origem ficava sem cadeia nenhuma (um dataset de vídeo perdeu o registro
    da própria extração e das exclusões feitas nele);
  - gravar a vista filtrada registrava só "rows: 83", sem o critério — não
    havia como reconstruir de onde vieram aquelas 83 linhas.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")

from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.filter import FilterConfig
from faceaudit.models.filter.column_filter import NumericColumnFilter
from faceaudit.models.filter.column_filter_data_type import ColumnFilterDataType
from faceaudit.services.audit_log_service import AuditLogService
from interface.tabs.dataset_manager.dataset_manager_controller import (
    DatasetManagerController,
)


def test_copy_sidecar_mantem_o_da_origem(tmp_path):
    origem, novo = tmp_path / "a.parquet", tmp_path / "b.parquet"
    AuditLogService.append(origem, AuditOperation.CREATE, "criado")
    assert AuditLogService.copy_sidecar(origem, novo)
    assert len(AuditLogService.load(origem)) == 1
    assert len(AuditLogService.load(novo)) == 1


def test_copy_sidecar_nao_sobrescreve_destino(tmp_path):
    origem, novo = tmp_path / "a.parquet", tmp_path / "b.parquet"
    AuditLogService.append(origem, AuditOperation.CREATE, "origem")
    AuditLogService.append(novo, AuditOperation.CREATE, "destino")
    assert not AuditLogService.copy_sidecar(origem, novo)
    assert AuditLogService.load(novo)[0].description == "destino"


def _face(i):
    return FaceInfo(image_path=f"/img/{i}.png", file_name=f"{i}.png",
                    identity="A", hash=f"h{i}")


def _controlador(tmp_path, *, filtrado):
    """Só o estado que o Salvar Como lê — sem montar a interface."""
    inteiro = FaceDataset(name="inteiro", faces=[_face(i) for i in range(4)])
    origem = tmp_path / "inteiro.parquet"
    inteiro.to_parquet(origem)
    AuditLogService.append(origem, AuditOperation.CREATE, "criado")
    AuditLogService.append(origem, AuditOperation.REMOVE_FACES, "1 excluída",
                           metadata={"pending_until_save": True})
    filtro = FilterConfig(name="fiqa", filters=[NumericColumnFilter(
        column_name="fiqa_score", data_type=ColumnFilterDataType.NUMERIC,
        ranges=[(1.6, float("inf"))])])
    na_tela = FaceDataset(name="vista", faces=inteiro.faces[:2]) if filtrado else inteiro
    ctrl = SimpleNamespace(
        dataset=na_tela, current_path=str(origem),
        dataset_original=inteiro if filtrado else None,
        current_filter_config=filtro if filtrado else FilterConfig(),
        _last_path_rebase=None,
        set_dirty=lambda *a: None,
        carregar_dataset_na_tela=lambda *a, **k: None,
    )
    for nome in ("_audit", "_descrever_conteudo_salvo"):
        setattr(ctrl, nome, getattr(DatasetManagerController, nome).__get__(ctrl))
    return ctrl, origem


def _salvar_como(ctrl, destino, **kw):
    DatasetManagerController._gravar_e_finalizar_salvar_como(ctrl, str(destino), **kw)


def test_salvar_como_copia_a_cadeia_e_a_origem_registra_que_nao_mudou(tmp_path):
    ctrl, origem = _controlador(tmp_path, filtrado=False)
    destino = tmp_path / "novo" / "novo.parquet"
    _salvar_como(ctrl, destino)

    cadeia_origem = AuditLogService.load(origem)
    cadeia_nova = AuditLogService.load(destino)
    # A origem mantém a história e ganha a entrada que explica o pendente.
    assert [e.operation for e in cadeia_origem] == [
        AuditOperation.CREATE, AuditOperation.REMOVE_FACES, AuditOperation.SAVE_AS]
    assert cadeia_origem[-1].metadata["this_file_unchanged"] is True
    assert cadeia_origem[-1].dataset_hash  # hash do arquivo de origem, intacto
    # O novo herda a história e acrescenta o próprio Salvar Como.
    assert [e.operation for e in cadeia_nova] == [
        AuditOperation.CREATE, AuditOperation.REMOVE_FACES, AuditOperation.SAVE_AS]
    assert cadeia_nova[-1].metadata["content"] == "full"
    for cadeia in (cadeia_origem, cadeia_nova):
        ok, problemas = AuditLogService.verify_chain(cadeia)
        assert ok, problemas


def test_vista_filtrada_registra_o_criterio(tmp_path):
    ctrl, _ = _controlador(tmp_path, filtrado=True)
    destino = tmp_path / "filtrado.parquet"
    _salvar_como(ctrl, destino)

    entrada = AuditLogService.load(destino)[-1]
    assert entrada.metadata["content"] == "filtered_view"
    faixa = entrada.metadata["filter"]["filters"][0]
    assert faixa["column_name"] == "fiqa_score"
    assert faixa["ranges"] == [[1.6, "inf"]]
    # O sidecar continua sendo JSON válido (sem `Infinity`).
    json.loads(AuditLogService.sidecar_path_for(destino).read_text(encoding="utf-8"),
               parse_constant=lambda c: pytest.fail(f"constante não-JSON: {c}"))
