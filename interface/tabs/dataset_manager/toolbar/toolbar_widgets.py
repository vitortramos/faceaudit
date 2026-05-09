from PyQt6.QtWidgets import QPushButton, QFrame, QLabel
from PyQt6.QtCore import Qt

class ToolbarButton(QPushButton):
    """
    Botão padronizado para a Toolbar.
    Aceita 'style_type' para definir cores predefinidas.

    Suporta modo COMPACT (icon-only) ativado pelo ribbon pai quando a
    largura disponível não comporta o texto. Tooltip preservado nos
    dois modos — em compact, hover ainda mostra o rótulo completo.
    """
    # Estilos CSS centralizados
    STYLES = {
        "default": "QPushButton { background-color: #ffffff; border: 1px solid #ccc; padding: 5px 12px; border-radius: 4px; color: #333; } QPushButton:hover { background-color: #e6e6e6; }",
        "primary": "QPushButton { background-color: #0078d7; color: white; font-weight: bold; border-radius: 4px; padding: 5px 12px; } QPushButton:hover { background-color: #005a9e; }",
        "success": "QPushButton { background-color: #28a745; color: white; font-weight: bold; border-radius: 4px; padding: 5px 12px; } QPushButton:hover { background-color: #218838; }",
        "danger":  "QPushButton { background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 5px 12px; border-radius: 4px; color: #721c24; } QPushButton:hover { background-color: #f1b0b7; }",
        "warning": "QPushButton { background-color: #ffc107; color: black; font-weight: bold; border-radius: 4px; padding: 5px 12px; } QPushButton:hover { background-color: #e0a800; }"
    }

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
        self.setStyleSheet(self.STYLES.get(style, self.STYLES["default"]))
        self.setMinimumWidth(self.MIN_BUTTON_WIDTH)
        self.style_type = style

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
        self.setStyleSheet(self.STYLES.get(style_name, self.STYLES["default"]))

class ToolbarSeparator(QFrame):
    """Linha vertical separadora."""
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setStyleSheet("color: #ccc;")

class StatusLabel(QLabel):
    """Label de status com fundo branco e borda."""
    def __init__(self, text="Nenhum dataset carregado."):
        super().__init__(text)
        self.setStyleSheet("background-color: white; padding: 2px 10px; border-radius: 3px; border: 1px solid #ddd;")