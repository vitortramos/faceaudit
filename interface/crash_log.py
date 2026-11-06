"""Registro de falhas da interface: nenhuma queda sem rastro.

Sem `sys.excepthook` próprio, o PyQt6 chama `qFatal` quando uma exceção
Python escapa de um slot — o processo aborta (0xc0000409 no Windows) e o
traceback só existe no terminal, que costuma fechar junto. Aqui:

- exceção não tratada (thread da GUI ou worker) vai para `excecoes.log`,
  continua no stderr e aparece num aviso; o app NÃO aborta;
- queda nativa (Qt/C++) grava a pilha Python em `faulthandler.log`.

Os arquivos ficam em `~/.faceaudit/logs/`.
"""
from __future__ import annotations

import datetime as _dt
import faulthandler
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional, TextIO

DEFAULT_LOG_DIR = Path.home() / ".faceaudit" / "logs"
EXCEPTIONS_LOG = "excecoes.log"
FAULTHANDLER_LOG = "faulthandler.log"
_MAX_LOG_BYTES = 5 * 1024 * 1024

# Arquivo do faulthandler precisa ficar aberto a sessão inteira: ele escreve
# no descritor no momento da queda, quando já não dá para abrir nada.
_fault_file: Optional[TextIO] = None
_log_dir: Path = DEFAULT_LOG_DIR
_showing_dialog = False


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _open_rotated(path: Path) -> TextIO:
    """Abre para acréscimo; acima do teto, recomeça (guarda a anterior em .1)."""
    try:
        if path.exists() and path.stat().st_size > _MAX_LOG_BYTES:
            path.replace(path.with_suffix(path.suffix + ".1"))
    except OSError:
        pass
    return open(path, "a", encoding="utf-8")


def _write_exception(text: str) -> None:
    try:
        with _open_rotated(_log_dir / EXCEPTIONS_LOG) as fh:
            fh.write(f"\n===== {_now()} =====\n{text}")
    except OSError:
        pass


def _show_dialog(exc: BaseException) -> None:
    """Aviso na tela, só na thread da GUI e sem reentrância."""
    global _showing_dialog
    try:
        from PyQt6 import QtCore, QtWidgets
    except ImportError:
        return
    app = QtWidgets.QApplication.instance()
    if app is None or QtCore.QThread.currentThread() is not app.thread():
        return
    if _showing_dialog:
        return
    _showing_dialog = True
    try:
        QtWidgets.QMessageBox.critical(
            None,
            "Erro inesperado",
            "Ocorreu um erro inesperado e a operação em andamento pode ter "
            "ficado incompleta. O FaceAudit continua aberto, mas a tela pode "
            "não refletir o estado real: salve o que precisar e reabra o "
            "projeto.\n\n"
            f"{type(exc).__name__}: {exc}\n\n"
            f"Detalhes gravados em:\n{_log_dir / EXCEPTIONS_LOG}",
        )
    except Exception:
        pass
    finally:
        _showing_dialog = False


def _excepthook(exc_type, exc, tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    _write_exception(text)
    sys.stderr.write(text)
    _show_dialog(exc)


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "?"
    text = f"[thread {name}]\n" + "".join(
        traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
    )
    _write_exception(text)
    sys.stderr.write(text)


def install(log_dir: Optional[Path] = None) -> Path:
    """Instala os ganchos. Chamar uma vez, no início de `run.py`."""
    global _fault_file, _log_dir
    _log_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    try:
        _log_dir.mkdir(parents=True, exist_ok=True)
        if _fault_file is not None:
            _fault_file.close()
        _fault_file = _open_rotated(_log_dir / FAULTHANDLER_LOG)
        _fault_file.write(f"\n===== sessão iniciada {_now()} =====\n")
        _fault_file.flush()
        faulthandler.enable(file=_fault_file, all_threads=True)
    except OSError:
        faulthandler.enable()
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    return _log_dir
