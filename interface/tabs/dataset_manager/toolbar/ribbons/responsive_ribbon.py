"""Base responsiva para os ribbons da toolbar do Dataset Manager.

Em telas estreitas (ex: MacBook 14"), o ribbon Dataset com 14 botões
extrapola a largura disponível — botões eram cortados pelo lado direito
ou forçavam overflow horizontal (scroll). Esta classe resolve isso:

    1. Calcula a largura natural dos filhos no modo full (ícone + texto).
    2. Em cada `resizeEvent`, compara com a largura disponível.
    3. Se não couber, alterna TODOS os ToolbarButton da ribbon pra
       compact (icon-only). Tooltip preserva o rótulo completo.
    4. Se a largura voltar a comportar, alterna de volta pra full.

A altura da ribbon não muda — toggle é apenas largura. Mantém a
toolbar com altura previsível (130px) sem impacto na área útil abaixo.
"""

from typing import List

from PyQt6.QtWidgets import QWidget

from ..toolbar_widgets import ToolbarButton


class ResponsiveRibbon(QWidget):
    """Ribbon que alterna entre full e compact baseado em sua largura.

    Subclasses só precisam montar o layout normalmente; a chamada a
    `_install_responsive_behavior()` no fim do `__init__` ativa o
    comportamento adaptativo.
    """

    # Folga de segurança — se a largura disponível for ≤ natural width
    # por menos de SLACK_PX, ainda vai pra compact (evita corte residual
    # de borda + spacing imprevisível do layout).
    SLACK_PX = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_compact = False
        self._cached_full_width: int = 0

    def _install_responsive_behavior(self) -> None:
        """Chamar no fim do `__init__` da subclasse, depois de montar
        o layout completo. Cacheia a largura natural pro hot path do
        resizeEvent não ter que recomputar.
        """
        # ensurePolished + sizeHint() do layout dão a largura natural
        # em modo full (estado inicial dos botões).
        self.ensurePolished()
        self._cached_full_width = self.layout().sizeHint().width() if self.layout() else 0

    def _toolbar_buttons(self) -> List[ToolbarButton]:
        """Coleta todos os ToolbarButton filhos diretos da ribbon."""
        return self.findChildren(ToolbarButton)

    def _set_compact(self, compact: bool) -> None:
        if compact == self._is_compact:
            return
        for btn in self._toolbar_buttons():
            btn.set_compact(compact)
        self._is_compact = compact

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._cached_full_width <= 0:
            return
        available = self.width()
        threshold = self._cached_full_width - self.SLACK_PX
        if available < threshold:
            self._set_compact(True)
        elif available >= self._cached_full_width:
            # Histerese: só volta a full se houver folga maior que o
            # threshold de entrada — evita oscilar quando a janela está
            # perto do limite.
            self._set_compact(False)
