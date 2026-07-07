"""ThemeManager — aplicação e troca em runtime do tema claro/escuro.

Responsabilidades:
- Construir a QPalette a partir dos tokens (`build_palette`).
- Construir o QSS global mínimo a partir dos tokens (`build_global_qss`).
- Aplicar estilo Fusion + paleta + QSS na QApplication (`apply`).
- Trocar de modo em runtime e notificar a UI (`set_mode` + sinal
  `theme_changed`).

Singleton acessível globalmente via `ThemeManager.instance()`, para que
diálogos (ConfigDialog) e widgets migrados possam ler tokens e reagir a
mudanças de tema.

A regra de negócio permanece no backend — este módulo é puramente de
apresentação (frontend). A preferência do usuário é persistida em
config.json via `SystemConfig.theme.mode` (feito pelo caller).
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from .tokens import TOKENS, normalize_mode


def build_palette(mode: str) -> QtGui.QPalette:
    """Constrói a QPalette do modo a partir dos tokens.

    O modo ``light`` reproduz exatamente a paleta histórica de
    `_apply_light_palette` — ativar o tema não regride o visual claro.
    """
    t = TOKENS[normalize_mode(mode)]
    Role = QtGui.QPalette.ColorRole
    Group = QtGui.QPalette.ColorGroup

    pal = QtGui.QPalette()

    def C(key: str) -> QtGui.QColor:
        return QtGui.QColor(t[key])

    pal.setColor(Role.Window, C("WINDOW"))
    pal.setColor(Role.WindowText, C("WINDOW_TEXT"))
    pal.setColor(Role.Base, C("BASE"))
    pal.setColor(Role.AlternateBase, C("ALT_BASE"))
    pal.setColor(Role.Text, C("TEXT"))
    pal.setColor(Role.Button, C("BUTTON"))
    pal.setColor(Role.ButtonText, C("BUTTON_TEXT"))
    pal.setColor(Role.ToolTipBase, C("TOOLTIP_BASE"))
    pal.setColor(Role.ToolTipText, C("TOOLTIP_TEXT"))
    pal.setColor(Role.PlaceholderText, C("PLACEHOLDER"))
    pal.setColor(Role.Highlight, C("HIGHLIGHT"))
    pal.setColor(Role.HighlightedText, C("HIGHLIGHT_TEXT"))
    pal.setColor(Role.Link, C("LINK"))
    pal.setColor(Role.BrightText, C("BRIGHT_TEXT"))

    disabled = C("DISABLED_TEXT")
    for role in (Role.WindowText, Role.Text, Role.ButtonText, Role.HighlightedText):
        pal.setColor(Group.Disabled, role, disabled)

    return pal


def build_global_qss(mode: str) -> str:
    """QSS global aplicado no nível da QApplication.

    Modo claro retorna string vazia — mantém o comportamento histórico
    (sem stylesheet global, só paleta) byte-idêntico, sem risco de
    regressão. Modo escuro aplica um baseline conservador para elementos
    que a QPalette não cobre bem (tooltip, menus, splitter), ajudando as
    partes ainda-não-migradas a ficarem coerentes no escuro.
    """
    mode = normalize_mode(mode)
    if mode == "light":
        return ""

    t = TOKENS[mode]
    return f"""
        QToolTip {{
            background-color: {t['TOOLTIP_BASE']};
            color: {t['TOOLTIP_TEXT']};
            border: 1px solid {t['BORDER']};
        }}
        QMenuBar {{ background: {t['WINDOW']}; color: {t['TEXT']}; }}
        QMenuBar::item:selected {{
            background: {t['HIGHLIGHT']}; color: {t['HIGHLIGHT_TEXT']};
        }}
        QMenu {{
            background: {t['SURFACE']}; color: {t['TEXT']};
            border: 1px solid {t['BORDER']};
        }}
        QMenu::item:selected {{
            background: {t['HIGHLIGHT']}; color: {t['HIGHLIGHT_TEXT']};
        }}
        QSplitter::handle {{ background: {t['BORDER']}; }}
    """.strip()


class ThemeManager(QtCore.QObject):
    """Gerencia o tema ativo da QApplication e notifica a UI ao trocar."""

    #: Emitido após aplicar um novo modo (payload = "light" | "dark").
    theme_changed = QtCore.pyqtSignal(str)

    _instance: Optional["ThemeManager"] = None

    def __init__(self, app: QtWidgets.QApplication):
        super().__init__(app)
        self._app = app
        self._mode = "light"

    # ── Singleton ─────────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> Optional["ThemeManager"]:
        """Retorna o ThemeManager ativo (ou None se ainda não inicializado)."""
        return cls._instance

    @classmethod
    def initialize(cls, app: QtWidgets.QApplication, mode: str = "light") -> "ThemeManager":
        """Cria (ou reusa) o singleton e aplica o modo. Chamar uma vez no
        bootstrap, após ler a preferência de config.json."""
        mgr = cls._instance or cls(app)
        mgr._app = app
        cls._instance = mgr
        mgr.apply(mode)
        return mgr

    # ── Estado / leitura de tokens ────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    def color(self, token_name: str) -> str:
        """Hex do token no modo atual. Token desconhecido → preto."""
        return TOKENS[self._mode].get(token_name, "#000000")

    def qcolor(self, token_name: str) -> QtGui.QColor:
        return QtGui.QColor(self.color(token_name))

    # ── Aplicação ─────────────────────────────────────────────────────────

    def apply(self, mode: str) -> None:
        """Aplica estilo + paleta + QSS global e emite `theme_changed`."""
        mode = normalize_mode(mode)
        self._mode = mode
        try:
            self._app.setStyle("Fusion")
        except Exception:
            pass
        self._app.setPalette(build_palette(mode))
        self._app.setStyleSheet(build_global_qss(mode))
        self.theme_changed.emit(mode)

    def set_mode(self, mode: str) -> None:
        """Troca de tema em runtime (no-op se já for o modo atual)."""
        if normalize_mode(mode) != self._mode:
            self.apply(mode)
