"""Helper para stylesheets temáveis (claro/escuro) com re-aplicação automática.

Padrão único de migração dos ~76 arquivos com QSS cravado:

    # ANTES
    self.setStyleSheet("QWidget { background: #f5f5f5; color: #333; }")

    # DEPOIS
    from interface.theme import apply_themed_stylesheet
    apply_themed_stylesheet(
        self,
        lambda t: f"QWidget {{ background: {t('SURFACE_RAISED')}; color: {t('TEXT')}; }}",
    )

`apply_themed_stylesheet(widget, builder)`:
- Aplica imediatamente `builder(t)` como stylesheet do widget, onde `t(token)`
  devolve o hex do token no modo atual.
- Reconstrói e re-aplica automaticamente quando o tema troca (via
  `ThemeManager.theme_changed`).
- O objeto que escuta o sinal é filho do widget (parent=widget), então é
  destruído junto com ele — sem vazamento de conexão nem widget zumbi.
"""

from __future__ import annotations

from typing import Callable

from PyQt6 import QtCore, QtWidgets

from .theme_manager import ThemeManager
from .tokens import color as _fallback_color

TokenFn = Callable[[str], str]
QssBuilder = Callable[[TokenFn], str]


def current_token_fn() -> TokenFn:
    """Retorna uma função `t(token) -> hex` que lê o modo ATUAL a cada
    chamada (fallback para o modo claro se o ThemeManager não existir)."""
    def t(name: str) -> str:
        mgr = ThemeManager.instance()
        if mgr is not None:
            return mgr.color(name)
        return _fallback_color(name, "light")
    return t


class _ThemedStyler(QtCore.QObject):
    """Recalcula e aplica o stylesheet de um widget a cada troca de tema."""

    def __init__(self, widget: QtWidgets.QWidget, builder: QssBuilder):
        super().__init__(widget)  # parent=widget → ciclo de vida atrelado
        self._widget = widget
        self._builder = builder
        mgr = ThemeManager.instance()
        if mgr is not None:
            mgr.theme_changed.connect(self._reapply)
        self._reapply()

    def _reapply(self, *_args) -> None:
        try:
            self._widget.setStyleSheet(self._builder(current_token_fn()))
        except RuntimeError:
            # Objeto C++ subjacente já destruído — nada a fazer.
            pass


def apply_themed_stylesheet(widget: QtWidgets.QWidget, builder: QssBuilder) -> None:
    """Aplica um stylesheet derivado de tokens e o re-aplica ao trocar de tema.

    Args:
        widget: alvo do `setStyleSheet`.
        builder: `callable(t)` que recebe `t(token)->hex` e devolve o QSS.
            Use f-strings com chaves QSS duplicadas: ``f"QLabel {{ ... }}"``.
    """
    _ThemedStyler(widget, builder)
