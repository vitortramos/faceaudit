from PyQt6.QtWidgets import QPushButton, QFrame, QLabel
from PyQt6.QtCore import Qt

from interface.theme import apply_themed_stylesheet, current_token_fn


def _button_qss(style: str, t) -> str:
    """QSS de um ToolbarButton por estilo, derivado de tokens de tema.

    `t` é a função `t(token) -> hex` do modo atual. Preserva 100% a
    estrutura CSS original (padding/radius/font) — só as cores viram
    tokens.
    """
    if style == "primary":
        return (
            "QPushButton { background-color: " + t("PRIMARY") + "; color: "
            + t("HIGHLIGHT_TEXT") + "; font-weight: bold; border-radius: 4px; "
            "padding: 5px 12px; } QPushButton:hover { background-color: "
            + t("PRIMARY_STRONG") + "; }"
        )
    if style == "success":
        return (
            "QPushButton { background-color: " + t("SUCCESS") + "; color: "
            + t("HIGHLIGHT_TEXT") + "; font-weight: bold; border-radius: 4px; "
            "padding: 5px 12px; } QPushButton:hover { background-color: "
            + t("SUCCESS") + "; }"
        )
    if style == "danger":
        return (
            "QPushButton { background-color: " + t("DANGER_BG") + "; border: 1px solid "
            + t("DANGER") + "; padding: 5px 12px; border-radius: 4px; color: "
            + t("DANGER") + "; } QPushButton:hover { background-color: "
            + t("DANGER_BG") + "; }"
        )
    if style == "warning":
        return (
            "QPushButton { background-color: " + t("WARNING") + "; color: "
            + t("TEXT") + "; font-weight: bold; border-radius: 4px; "
            "padding: 5px 12px; } QPushButton:hover { background-color: "
            + t("WARNING") + "; }"
        )
    # default
    return (
        "QPushButton { background-color: " + t("SURFACE") + "; border: 1px solid "
        + t("BORDER") + "; padding: 5px 12px; border-radius: 4px; color: "
        + t("TEXT") + "; } QPushButton:hover { background-color: "
        + t("SURFACE_ALT") + "; }"
    )


class ToolbarButton(QPushButton):
    """
    Botão padronizado para a Toolbar.
    Aceita 'style_type' para definir cores predefinidas.

    Suporta modo COMPACT (icon-only) ativado pelo ribbon pai quando a
    largura disponível não comporta o texto. Tooltip preservado nos
    dois modos — em compact, hover ainda mostra o rótulo completo.
    """

    # Largura mínima padronizada — uniformiza a aparência dos botões da
    # toolbar pra rótulos curtos (Abrir, Salvar, Auditar, etc.) sem
    # cortar os mais longos (Salvar Como…, Estatísticas, Biblioteca):
    # esses crescem além do mínimo via tamanho natural do conteúdo.
    MIN_BUTTON_WIDTH = 110
    # Largura no modo compact — só o ícone/emoji. Quadrado, ~36px.
    MIN_BUTTON_WIDTH_COMPACT = 38

    def __init__(self, text, icon_char="", style="default", tooltip=""):
        super().__init__()

        self._label_text = text
        self._icon_char = icon_char
        self._compact = False

        # Tooltip — em compact ele assume o papel do rótulo. Se o caller
        # não passou tooltip explícito, usa o texto como fallback pra que
        # o usuário sempre consiga descobrir a função do botão por hover.
        self.setToolTip(tooltip if tooltip else text)
        self.style_type = style
        # QSS temável: recalculado ao trocar de tema. O estilo corrente
        # (`self.style_type`) é lido a cada rebuild — set_style muta o
        # atributo e re-aplica.
        apply_themed_stylesheet(
            self, lambda t: _button_qss(self.style_type, t),
        )
        self.setMinimumWidth(self.MIN_BUTTON_WIDTH)

        self._refresh_display_text()

    def _refresh_display_text(self):
        """Atualiza o texto exibido conforme o modo (compact ou normal)."""
        if self._compact:
            display = self._icon_char if self._icon_char else self._label_text
        else:
            if self._icon_char:
                display = f"{self._icon_char} {self._label_text}"
            else:
                display = self._label_text
        self.setText(display)

    def set_label(self, text: str, icon_char: str = None) -> None:
        """Atualiza o rótulo (e opcionalmente o ícone) exibido em runtime,
        respeitando o modo atual (compact/full). Usado por botões que mudam de
        função conforme o estado — ex.: o toggle Galeria↔Tabela."""
        self._label_text = text
        if icon_char is not None:
            self._icon_char = icon_char
        self._refresh_display_text()

    def set_compact(self, compact: bool) -> None:
        """Alterna entre modo full (icon + texto) e compact (só ícone)."""
        if compact == self._compact:
            return
        self._compact = compact
        if compact:
            self.setMinimumWidth(self.MIN_BUTTON_WIDTH_COMPACT)
            self.setMaximumWidth(self.MIN_BUTTON_WIDTH_COMPACT + 12)
        else:
            self.setMinimumWidth(self.MIN_BUTTON_WIDTH)
            self.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX
        self._refresh_display_text()

    def set_style(self, style_name):
        # Muta o estilo corrente e re-aplica o QSS temável com a cor certa
        # do tema atual (o builder registrado lê self.style_type).
        self.style_type = style_name
        self.setStyleSheet(_button_qss(self.style_type, current_token_fn()))

class ToolbarSeparator(QFrame):
    """Linha vertical separadora."""
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        apply_themed_stylesheet(
            self, lambda t: "color: " + t("BORDER") + ";",
        )

class StatusLabel(QLabel):
    """Label de status com fundo branco e borda."""
    def __init__(self, text="Nenhum dataset carregado."):
        super().__init__(text)
        apply_themed_stylesheet(
            self,
            lambda t: (
                "background-color: " + t("SURFACE") + "; padding: 2px 10px; "
                "border-radius: 3px; border: 1px solid " + t("BORDER") + ";"
            ),
        )