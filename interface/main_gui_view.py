import os
from PyQt6 import QtWidgets, uic, QtCore, QtGui
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton

from interface.theme import ThemeManager, apply_titlebar_theme
from interface.theme.tokens import color as _theme_color

class MainGuiView(QtWidgets.QMainWindow):
    """
    View da Janela Principal.
    Responsável apenas pela renderização da casca da aplicação e containers de abas.
    """
    config_requested = QtCore.pyqtSignal()

    def __init__(self, ui_path: str):
        super().__init__()
        try:
            uic.loadUi(ui_path, self)
        except Exception as e:
            raise RuntimeError(f"Falha crítica ao carregar arquivo de interface (.ui): {e}")

        self.main_tabs = self.findChild(QtWidgets.QTabWidget)
        self._apply_main_tabs_style()
        self._setup_corner_config_button()

        # Reaplica o chrome (tabs + botão de config) quando o usuário troca
        # de tema em runtime. O ThemeManager já foi inicializado no bootstrap
        # (interface/run.py) antes desta view existir.
        _theme_mgr = ThemeManager.instance()
        if _theme_mgr is not None:
            _theme_mgr.theme_changed.connect(self._on_theme_changed)
            # Barra de título nativa (Windows) acompanha o tema — evita título
            # claro sobre app escura. Aplicado aqui (antes do show) para a
            # janela já nascer com a barra correta.
            apply_titlebar_theme(self, _theme_mgr.mode == "dark")
        # CRÍTICO em telas pequenas: o QTabWidget propaga o MAX dos
        # minimumSizeHint de TODAS as tabs (não só a ativa). A tab
        # Dataset Manager (com ribbon de muitos botões) tem min ~1630,
        # arrastando a janela inteira pra fora da tela quando o
        # conteúdo da tab Strategy V2 colapsa.
        # Ignored faz o widget descartar sizeHint E minimumSizeHint —
        # única policy que quebra a propagação.
        if self.main_tabs is not None:
            self.main_tabs.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            self.main_tabs.setMinimumWidth(0)
        cw = self.centralWidget()
        if cw is not None:
            cw.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            cw.setMinimumWidth(0)
        # Trava a largura/altura máxima da janela à área disponível da
        # tela. Sem isso, widgets internos com `sizeHint` natural maior
        # que a tela (caso típico em MacBook 14" com Strategy Manager V2
        # quando o painel lateral é colapsado e o editor expande pra
        # ~1300px) arrastam a MainWindow pra fora dos limites do display
        # — usuário não consegue ver o canto direito.
        self._apply_screen_max_size()
        self.windowHandle = None
        # Re-aplica em mudança de tela (laptop conectado a monitor externo
        # e voltando à tela built-in, por exemplo).
        try:
            QtWidgets.QApplication.instance().screenAdded.connect(
                lambda _s: self._apply_screen_max_size()
            )
            QtWidgets.QApplication.instance().screenRemoved.connect(
                lambda _s: self._apply_screen_max_size()
            )
        except Exception:
            pass

    def _apply_screen_max_size(self) -> None:
        """Limita maximumSize do CENTRAL WIDGET (não da QMainWindow!) ao
        `availableGeometry` da tela atual (já desconta dock + menu bar do
        macOS).

        IMPORTANTE: `setMaximumSize` numa `QMainWindow` faz o Qt remover
        a flag `WS_MAXIMIZEBOX` da janela nativa no Windows → o botão de
        maximizar fica greyed out / inutilizável. Por isso aplicamos a
        constraint no central widget: o conteúdo interno continua capado
        (não arrasta a janela pra fora da tela), mas a janela em si fica
        livre pra ser maximizada/restaurada.
        """
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        screen = self.screen() or app.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        cw = self.centralWidget()
        if cw is not None:
            cw.setMaximumSize(avail.width(), avail.height())
        self._screen_max_w = avail.width()
        self._screen_max_h = avail.height()

    # resizeEvent override REMOVIDO — entrava em loop infinito porque
    # ao forçar resize(cap), o Qt imediatamente revertia pra
    # minimumSize do conteúdo (1630), disparando novo resizeEvent que
    # dispara novo singleShot, etc. Travava a UI. Procurar outra
    # abordagem: cortar o minimumSize NA ORIGEM em vez de tentar
    # corrigir a janela depois.

    def _tok(self, name: str) -> str:
        """Hex de um token de tema no modo atual. Fallback para o modo claro
        quando o ThemeManager ainda não foi inicializado (ex: testes)."""
        mgr = ThemeManager.instance()
        if mgr is not None:
            return mgr.color(name)
        return _theme_color(name, "light")

    def _on_theme_changed(self, mode: str) -> None:
        """Reaplica o chrome que usa QSS próprio quando o tema troca."""
        if self.main_tabs is not None:
            self._apply_main_tabs_style()
        if getattr(self, "btn_config", None) is not None:
            self._restyle_config_button()
        # Barra de título nativa acompanha a troca ao vivo.
        apply_titlebar_theme(self, mode == "dark")

    def _apply_main_tabs_style(self):
        """Estilo Office accent bar nas tabs principais — mesma família
        visual das tabs do ribbon (manager_toolbar_view), mas com
        accent navy (`PRIMARY_STRONG`) + barra mais grossa (4px) + padding
        maior pra reforçar a hierarquia: tabs principais (nível 1 —
        navegação no app) > tabs do ribbon (nível 2 — grupos de
        ferramentas). Cores derivadas de tokens de tema (claro/escuro)."""
        t = self._tok
        self.main_tabs.setStyleSheet(
            # Pane (área do conteúdo da aba)
            "QTabWidget::pane { "
            f"    border: 1px solid {t('BORDER')}; "
            f"    background: {t('SURFACE')}; "
            "    top: -1px; "
            "} "
            # Aba inativa
            "QTabBar::tab { "
            f"    background: {t('SURFACE_ALT')}; "
            f"    color: {t('TEXT_MUTED')}; "
            "    padding: 10px 24px; "
            "    font-size: 12px; "
            f"    border: 1px solid {t('BORDER')}; "
            "    border-top: 4px solid transparent; "
            "    border-right: none; "
            "    border-top-left-radius: 5px; "
            "    border-top-right-radius: 5px; "
            "    margin-right: 1px; "
            "} "
            # Hover
            "QTabBar::tab:hover:!selected { "
            f"    background: {t('BORDER')}; "
            f"    color: {t('PRIMARY_STRONG')}; "
            "} "
            # Aba selecionada — accent bar navy + texto navy + bold
            "QTabBar::tab:selected { "
            f"    background: {t('SURFACE')}; "
            f"    color: {t('PRIMARY_STRONG')}; "
            "    font-weight: bold; "
            f"    border-top: 4px solid {t('PRIMARY_STRONG')}; "
            f"    border-bottom: 1px solid {t('SURFACE')}; "
            "} "
            "QTabBar::tab:last { margin-right: 0; }"
        )

    def _restyle_config_button(self):
        """Aplica o estilo do botão ⚙️ a partir de tokens de tema."""
        t = self._tok
        self.btn_config.setStyleSheet(
            "QPushButton { border: none; background: transparent; "
            "font-size: 18px; color: " + t("TEXT") + "; } "
            "QPushButton:hover { background-color: " + t("SURFACE_ALT")
            + "; border-radius: 5px; }"
        )

    def add_tab(self, index: int, widget: QWidget, label: str):
        """Insere um widget MVC em uma posição específica do QTabWidget."""
        if self.main_tabs.count() > index:
            self.main_tabs.removeTab(index)
        self.main_tabs.insertTab(index, widget, label)

    def _setup_corner_config_button(self):
        """Posiciona o botão de engrenagem no canto superior direito das abas."""
        self.btn_config = QPushButton("⚙️")
        self.btn_config.setToolTip("Configurações do Sistema")
        self.btn_config.setFixedSize(35, 30)
        self._restyle_config_button()
        self.btn_config.clicked.connect(self.config_requested.emit)

        # Container para auxiliar o alinhamento vertical na barra de abas
        container = QWidget()
        l = QVBoxLayout(container)
        l.setContentsMargins(0, 0, 15, 0)
        l.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        l.addWidget(self.btn_config)

        self.main_tabs.setCornerWidget(container, QtCore.Qt.Corner.TopRightCorner)