import os
from PyQt6 import QtWidgets, uic, QtCore, QtGui
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton

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
        # Permite a janela encolher abaixo do sizeHint do conteúdo
        # (telas pequenas tipo MacBook 14" 1512px). Por default o Qt
        # impede a janela de ficar menor que o sizeHint mínimo dos
        # filhos — em telas estreitas isso fazia parte da app ficar
        # cortada à direita sem nem mostrar scrollbar.
        self.setMinimumSize(QtCore.QSize(800, 600))
        self._wrap_central_in_scroll_area()
        self._apply_main_tabs_style()
        self._setup_corner_config_button()

    def _wrap_central_in_scroll_area(self):
        """Envolve a tab widget central num QScrollArea pra que telas
        menores que o sizeHint do conteúdo (MacBook 14", 1512px) ganhem
        scrollbars automaticamente em vez de cortar a UI à direita.

        Estratégia:
        - QScrollArea com `setWidgetResizable(False)` faz o widget
          interno (tab widget) ocupar exatamente o seu sizeHint.
          Quando o viewport < sizeHint, surge scrollbar.
        - Listener de `resizeEvent` no scroll redimensiona o tab widget
          pra acompanhar o viewport quando há espaço de sobra
          (mantém o feeling de "ocupa a tela toda" em monitores grandes).
        - Largura mínima do tab widget bate o sizeHint reportado pela
          aba ativa, garantindo que conteúdos legítimamente largos
          (3 colunas do Strategy Manager V2, etc.) não sejam comprimidos
          até a ilegibilidade.
        """
        old_central = self.centralWidget()
        if old_central is None or self.main_tabs is None:
            return

        self.main_tabs.setParent(None)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("main_scroll_area")
        scroll.setWidgetResizable(False)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )
        scroll.setWidget(self.main_tabs)

        new_central = QWidget()
        layout = QVBoxLayout(new_central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.setCentralWidget(new_central)
        self._main_scroll_area = scroll

        # Bridge entre o scroll viewport e o tab widget: quando o
        # viewport encolhe abaixo do `minimumSizeHint` do tab widget,
        # mantemos o tab widget no seu tamanho mínimo + scrollbar.
        # Quando o viewport tem mais espaço, expandimos o tab widget
        # pra ocupar tudo (sem scrollbar dummy).
        scroll.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        # Sincroniza tab widget com o viewport do scroll quando o usuário
        # redimensiona a janela. Mantém o comportamento "ocupa tela toda"
        # em monitores grandes, com fallback pra scrollbar em telas
        # pequenas (≤ sizeHint do conteúdo).
        scroll = getattr(self, "_main_scroll_area", None)
        if (
            scroll is not None
            and obj is scroll.viewport()
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            tabs = self.main_tabs
            if tabs is not None:
                hint = tabs.minimumSizeHint()
                w = max(scroll.viewport().width(), hint.width())
                h = max(scroll.viewport().height(), hint.height())
                tabs.resize(w, h)
        return super().eventFilter(obj, event)

    def _apply_main_tabs_style(self):
        """Estilo Office accent bar nas tabs principais — mesma família
        visual das tabs do ribbon (manager_toolbar_view), mas com
        accent navy escuro (`#1a3d6e`) + barra mais grossa (4px) +
        padding maior pra reforçar a hierarquia: tabs principais (nível 1
        — navegação no app) > tabs do ribbon (nível 2 — grupos de
        ferramentas)."""
        self.main_tabs.setStyleSheet(
            # Pane (área do conteúdo da aba)
            "QTabWidget::pane { "
            "    border: 1px solid #c8c8c8; "
            "    background: white; "
            "    top: -1px; "
            "} "
            # Aba inativa
            "QTabBar::tab { "
            "    background: #e0e0e0; "
            "    color: #555; "
            "    padding: 10px 24px; "
            "    font-size: 12px; "
            "    border: 1px solid #c8c8c8; "
            "    border-top: 4px solid transparent; "
            "    border-right: none; "
            "    border-top-left-radius: 5px; "
            "    border-top-right-radius: 5px; "
            "    margin-right: 1px; "
            "} "
            # Hover
            "QTabBar::tab:hover:!selected { "
            "    background: #d2d2d2; "
            "    color: #1a3d6e; "
            "} "
            # Aba selecionada — accent bar navy + texto navy + bold
            "QTabBar::tab:selected { "
            "    background: white; "
            "    color: #1a3d6e; "
            "    font-weight: bold; "
            "    border-top: 4px solid #1a3d6e; "
            "    border-bottom: 1px solid white; "
            "} "
            "QTabBar::tab:last { margin-right: 0; }"
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
        self.btn_config.setStyleSheet("""
            QPushButton { border: none; background: transparent; font-size: 18px; color: #333; }
            QPushButton:hover { background-color: #e0e0e0; border-radius: 5px; }
        """)
        self.btn_config.clicked.connect(self.config_requested.emit)

        # Container para auxiliar o alinhamento vertical na barra de abas
        container = QWidget()
        l = QVBoxLayout(container)
        l.setContentsMargins(0, 0, 15, 0)
        l.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        l.addWidget(self.btn_config)

        self.main_tabs.setCornerWidget(container, QtCore.Qt.Corner.TopRightCorner)