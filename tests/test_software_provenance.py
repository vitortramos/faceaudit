"""Todo artefato diz qual FaceAudit o produziu: versão + commit."""

import json
import re
import subprocess
from pathlib import Path

import faceaudit
from faceaudit._version import __version__
from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.services import software_provenance_service as sp
from faceaudit.services.audit_log_service import AuditLogService

_REPO = Path(__file__).resolve().parents[1]


def _git(*args):
    r = subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def test_versao_tem_fonte_unica():
    assert faceaudit.__version__ == __version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
    pyproject = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    # O pyproject não declara número próprio: lê do _version.py.
    assert 'dynamic = ["version"]' in pyproject
    assert "faceaudit._version.__version__" in pyproject


def test_commit_e_o_do_checkout():
    prov = sp.software_provenance()
    assert prov["faceaudit_version"] == __version__
    head = _git("rev-parse", "HEAD")
    if head is None:  # fora de um checkout: não se sabe, e o registro diz isso
        assert prov["git_commit"] is None
        return
    assert prov["git_commit"] == head
    sujo = bool(_git("status", "--porcelain", "--untracked-files=no", "--", "faceaudit", "interface"))
    assert prov["git_dirty"] is sujo
    assert sp.commit_label() == head[:10] + ("-dirty" if sujo else "")


def test_sem_git_o_commit_fica_vazio(monkeypatch):
    monkeypatch.setattr(sp, "_git", lambda *a: None)
    sp._git_state.cache_clear()
    try:
        prov = sp.software_provenance()
        assert prov["faceaudit_version"] == __version__
        assert prov["git_commit"] is None and prov["git_dirty"] is None
        assert sp.commit_label() is None
    finally:
        sp._git_state.cache_clear()


def test_toda_entrada_da_custodia_leva_o_software(tmp_path):
    parquet = tmp_path / "ds.parquet"
    e1 = AuditLogService.append(parquet, AuditOperation.CREATE, "criado", metadata={"rows": 3})
    e2 = AuditLogService.append(parquet, AuditOperation.ENRICH, "enriquecido")
    for e in (e1, e2):
        assert e.metadata["software"] == sp.software_provenance()
    assert e1.metadata["rows"] == 3
    # Está dentro do entry_hash: adulterar o commit quebra a cadeia.
    entries = AuditLogService.load(parquet)
    assert AuditLogService.verify_chain(entries)[0]
    bruto = json.loads(AuditLogService.sidecar_path_for(parquet).read_text(encoding="utf-8"))
    bruto[0]["metadata"]["software"]["git_commit"] = "0" * 40
    AuditLogService.sidecar_path_for(parquet).write_text(json.dumps(bruto), encoding="utf-8")
    ok, problemas = AuditLogService.verify_chain(AuditLogService.load(parquet))
    assert not ok and problemas[0].code == "self_hash_mismatch"


def test_versoes_das_bibliotecas_que_decidem_o_resultado():
    from importlib import metadata
    import platform
    prov = sp.software_provenance()
    deps = prov["dependencies"]
    # O que está instalado entra com a versão exata do pip; o que não está fica de fora.
    for dist in sp._TRACKED_DISTRIBUTIONS:
        try:
            instalada = metadata.version(dist)
        except metadata.PackageNotFoundError:
            assert dist not in deps
        else:
            assert deps[dist] == instalada
    assert "forensicface" in deps and "numpy" in deps
    assert any(d.startswith("onnxruntime") for d in deps)
    assert prov["python"] == platform.python_version()
    assert prov["platform"].split("-")[0] in ("win32", "darwin", "linux")
    json.dumps(prov)  # vai para JSON (cadeia de custódia, manifestos)


def test_versoes_das_bibliotecas_ficam_na_custodia(tmp_path, monkeypatch):
    parquet = tmp_path / "ds.parquet"
    e = AuditLogService.append(parquet, AuditOperation.CREATE, "criado")
    assert e.metadata["software"]["dependencies"] == sp.dependency_versions()
    # Trocar a versão registrada de uma biblioteca quebra a cadeia.
    bruto = json.loads(AuditLogService.sidecar_path_for(parquet).read_text(encoding="utf-8"))
    bruto[0]["metadata"]["software"]["dependencies"]["forensicface"] = "9.9.9"
    AuditLogService.sidecar_path_for(parquet).write_text(json.dumps(bruto), encoding="utf-8")
    ok, _ = AuditLogService.verify_chain(AuditLogService.load(parquet))
    assert not ok
