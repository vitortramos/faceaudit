"""Nenhuma queda sem rastro: exceção não tratada vai para o log e não aborta."""
import faulthandler
import sys
import threading

import pytest

from interface import crash_log


@pytest.fixture
def instalado(tmp_path):
    antes = (sys.excepthook, threading.excepthook)
    log_dir = crash_log.install(tmp_path)
    yield log_dir
    sys.excepthook, threading.excepthook = antes
    faulthandler.enable()  # devolve ao stderr (o pytest também o usa)


def test_excecao_da_gui_vai_para_o_log(instalado, capsys):
    try:
        raise ValueError("falha num slot")
    except ValueError:
        sys.excepthook(*sys.exc_info())
    texto = (instalado / crash_log.EXCEPTIONS_LOG).read_text(encoding="utf-8")
    assert "ValueError: falha num slot" in texto and "Traceback" in texto
    assert "falha num slot" in capsys.readouterr().err


def test_excecao_de_thread_vai_para_o_log(instalado):
    def falha():
        raise RuntimeError("falha no worker")
    t = threading.Thread(target=falha, name="worker-teste")
    t.start(); t.join()
    texto = (instalado / crash_log.EXCEPTIONS_LOG).read_text(encoding="utf-8")
    assert "[thread worker-teste]" in texto and "RuntimeError: falha no worker" in texto


def test_faulthandler_grava_no_arquivo_da_sessao(instalado):
    assert faulthandler.is_enabled()
    texto = (instalado / crash_log.FAULTHANDLER_LOG).read_text(encoding="utf-8")
    assert "sessão iniciada" in texto


def test_log_grande_e_rotacionado(tmp_path, monkeypatch):
    monkeypatch.setattr(crash_log, "_MAX_LOG_BYTES", 10)
    alvo = tmp_path / crash_log.EXCEPTIONS_LOG
    alvo.write_text("x" * 100, encoding="utf-8")
    crash_log._open_rotated(alvo).close()
    assert (tmp_path / (crash_log.EXCEPTIONS_LOG + ".1")).exists()
    assert alvo.stat().st_size == 0


_SCRIPT_SLOT = r'''
import sys
from pathlib import Path
from PyQt6 import QtCore, QtWidgets
from interface import crash_log
crash_log.install(Path(sys.argv[1]))
avisos = []
QtWidgets.QMessageBox.critical = staticmethod(lambda *a, **k: avisos.append(a[2]))
app = QtWidgets.QApplication(sys.argv[:1])
def slot():
    raise KeyError("recurso sumiu")
QtCore.QTimer.singleShot(0, slot)
QtCore.QTimer.singleShot(200, app.quit)
app.exec()
print("SOBREVIVEU", len(avisos), "excecoes.log" in avisos[0])
'''


def test_excecao_em_slot_nao_derruba_o_app(tmp_path):
    """Sem o gancho, o PyQt6 aborta o processo (qFatal) — o crash do
    Gerenciador de Estratégias. Com ele, o app segue e o erro fica no log."""
    import os
    import subprocess
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONIOENCODING="utf-8",
               PYTHONPATH=str(raiz))
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT_SLOT, str(tmp_path)],
        capture_output=True, text=True, encoding="utf-8", env=env, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SOBREVIVEU 1 True" in proc.stdout
    texto = (tmp_path / crash_log.EXCEPTIONS_LOG).read_text(encoding="utf-8")
    assert "KeyError: 'recurso sumiu'" in texto
