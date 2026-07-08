"""Barra de título nativa seguindo o tema (claro/escuro) no Windows.

No Windows a barra de título (área não-cliente) é desenhada pelo SO e segue
o tema do **sistema**, não o da aplicação. Como o FaceAudit força seu próprio
tema, a barra fica clara mesmo com a app em modo escuro — visualmente
destoante. `DwmSetWindowAttribute` com `DWMWA_USE_IMMERSIVE_DARK_MODE`
instrui o DWM a pintar a barra de título escura.

No-op fora do Windows (no macOS a barra acompanha o conteúdo/appearance).
Robusto a falhas — nunca levanta.
"""

from __future__ import annotations

import sys

from PyQt6 import QtWidgets

# DWMWA_USE_IMMERSIVE_DARK_MODE: 20 no Win10 20H1+ (build 19041+), 19 em
# builds mais antigas. Tentamos 20 e caímos para 19.
_DWMWA_NEW = 20
_DWMWA_OLD = 19

# SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED
_SWP_FRAME_REFRESH = 0x0001 | 0x0002 | 0x0004 | 0x0020


def apply_titlebar_theme(widget: QtWidgets.QWidget, dark: bool) -> None:
    """Pinta a barra de título de `widget` (janela top-level) conforme o tema."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        hwnd = wintypes.HWND(int(widget.winId()))
        value = ctypes.c_int(1 if dark else 0)
        dwmapi = ctypes.windll.dwmapi

        applied = False
        for attr in (_DWMWA_NEW, _DWMWA_OLD):
            res = dwmapi.DwmSetWindowAttribute(
                hwnd, ctypes.c_int(attr),
                ctypes.byref(value), ctypes.sizeof(value),
            )
            if res == 0:
                applied = True
                break

        # Força o DWM a repintar a área não-cliente (title bar) sem mover
        # nem redimensionar — necessário quando o tema troca com a janela
        # já visível.
        if applied:
            ctypes.windll.user32.SetWindowPos(
                hwnd, None, 0, 0, 0, 0, _SWP_FRAME_REFRESH,
            )
    except Exception:
        # Barra de título é cosmética — falha aqui não deve derrubar a UI.
        pass


def apply_titlebar_theme_to_all(dark: bool) -> None:
    """Aplica a todas as janelas top-level atuais da aplicação."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    for w in app.topLevelWidgets():
        if w.isWindow():
            apply_titlebar_theme(w, dark)
