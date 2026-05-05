# interface/tabs/strategy_manager_v2/strategy_manager_v2_view.py

from PyQt6 import QtWidgets, QtCore, QtGui


class StrategyManagerV2View(QtWidgets.QWidget):
    """
    View Container da aba Strategy Manager V2 (nova arquitetura declarativa).

    Layout:
    ┌─────────────────────────────────────────────────────┐
    │  Global Resources Toolbar (Pipeline/Mining/Split/…) │
    │  Project Toolbar (Novo/Abrir/Salvar/Fechar/…)       │
    ├──────────────┬──────────────────────────────────────┤
    │  Left Panel  │  Strategy Editor                     │
    │  (QSplitter) │  (sem aba — editor direto)           │
    │  - Estratégs │                                       │
    │  - Recursos  │  Botão "Comparar" no left panel abre │
    │  - Info      │  uma janela top-level dedicada.      │
    └──────────────┴──────────────────────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Estado de collapse do left panel.
        self._left_panel_collapsed = False
        # Tamanhos a restaurar quando o usuário expande o painel de novo.
        # Atualizados dinamicamente pelo splitter quando o usuário arrasta.
        self._saved_splitter_sizes = [280, 700]
        self._init_layout()
        self._init_shortcuts()

    def _init_shortcuts(self) -> None:
        # Cmd+B no macOS / Ctrl+B em Linux/Windows. Qt traduz automático.
        toggle_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+B"), self)
        toggle_shortcut.activated.connect(self.toggle_left_panel)

    def _init_layout(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(5, 5, 5, 5)

        # ── Toolbar ──────────────────────────────────────────────────────────
        self.toolbar_container = QtWidgets.QWidget()
        toolbar_layout = QtWidgets.QVBoxLayout(self.toolbar_container)
        toolbar_layout.setContentsMargins(0, 0, 0, 4)
        toolbar_layout.setSpacing(0)

        # Placeholder — preenchido pelo controller
        self.project_toolbar = QtWidgets.QToolBar("Projeto")
        self.project_toolbar.setMovable(False)

        toolbar_layout.addWidget(self.project_toolbar)
        root.addWidget(self.toolbar_container)

        # Separador visual
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: #dcdde1;")
        root.addWidget(sep)

        # ── Content: QStackedWidget (welcome ↔ workspace) ────────────────────
        self.content_stack = QtWidgets.QStackedWidget()

        self.welcome_page = self._create_welcome_page()
        self.content_stack.addWidget(self.welcome_page)       # index 0

        self.workspace_page = self._create_workspace_page()
        self.content_stack.addWidget(self.workspace_page)     # index 1

        root.addWidget(self.content_stack)
        self.show_welcome_screen()

    # ── Welcome screen ────────────────────────────────────────────────────────

    def _create_welcome_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        icon = QtWidgets.QLabel("⚖️")
        icon.setStyleSheet("font-size: 60px; margin-bottom: 15px;")
        icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        title = QtWidgets.QLabel("Gerenciador de Estratégias (Nova Arquitetura)")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #2c3e50;")

        subtitle = QtWidgets.QLabel(
            "Crie ou abra um projeto para:\n\n"
            "• Compor estratégias de cálculo de LR de forma declarativa\n"
            "• Executar e validar estratégias configuradas\n"
            "• Comparar resultados entre estratégias"
        )
        subtitle.setStyleSheet(
            "font-size: 14px; color: #5f6368; line-height: 180%; padding: 20px;"
        )
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        layout.addWidget(icon)
        layout.addWidget(title, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        return page

    # ── Workspace ─────────────────────────────────────────────────────────────

    def _create_workspace_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 4, 0, 0)

        # Splitter principal: left panel ↔ main area (redimensionável pelo usuário)
        self._workspace_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._workspace_splitter.setChildrenCollapsible(False)
        self._workspace_splitter.setHandleWidth(4)
        self._workspace_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #dcdde1;
            }
            QSplitter::handle:hover {
                background: #a0a0a0;
            }
        """)

        # Left panel placeholder — substituído por LeftPanelWidget pelo controller
        self.left_panel_placeholder = QtWidgets.QWidget()
        self.left_panel_placeholder.setMinimumWidth(200)
        self.left_panel_placeholder.setMaximumWidth(500)

        # Editor de estratégia — placeholder substituído pelo controller via
        # set_strategy_editor(). Vai direto no splitter, sem QTabWidget
        # intermediário (o conteúdo "Comparação" antigo virou janela top-level
        # acessada via botão "Comparar" no left_panel).
        self.strategy_editor_placeholder = QtWidgets.QWidget()

        self._workspace_splitter.addWidget(self.left_panel_placeholder)
        self._workspace_splitter.addWidget(self.strategy_editor_placeholder)
        self._workspace_splitter.setSizes([280, 700])

        layout.addWidget(self._workspace_splitter)
        return page

    def set_left_panel(self, widget: QtWidgets.QWidget) -> None:
        """Substitui o placeholder do painel esquerdo pelo widget real."""
        widget.setMinimumWidth(200)
        widget.setMaximumWidth(500)
        self._workspace_splitter.replaceWidget(0, widget)
        self.left_panel_placeholder.hide()
        self.left_panel_placeholder = widget

    def toggle_left_panel(self) -> None:
        """
        Alterna entre painel lateral visível e recolhido. Quando recolhido,
        o splitter dá toda a largura para a área principal — útil em telas
        pequenas (ex: MacBook 14&quot;) durante edição intensa de instruções.
        """
        if self._workspace_splitter is None:
            return
        if self._left_panel_collapsed:
            self._workspace_splitter.setSizes(self._saved_splitter_sizes)
            self._left_panel_collapsed = False
        else:
            current = self._workspace_splitter.sizes()
            if current and current[0] > 0:
                self._saved_splitter_sizes = list(current)
            # Truque: setChildrenCollapsible está False, então não conseguimos
            # ir a 0. Usamos minimumWidth do left panel como tamanho recolhido
            # mais próximo de 0; mas pra de fato &quot;sumir&quot;, ocultamos o widget.
            self.left_panel_placeholder.hide()
            self._left_panel_collapsed = True
        # Atualiza label do botão na toolbar, se existir.
        toolbar = getattr(self, "project_toolbar", None)
        if toolbar is not None and hasattr(toolbar, "set_left_panel_collapsed"):
            toolbar.set_left_panel_collapsed(self._left_panel_collapsed)
        # Mostra de volta após restaurar tamanhos (ordem importa pra evitar flicker).
        if not self._left_panel_collapsed:
            self.left_panel_placeholder.show()

    def set_strategy_editor(self, widget: QtWidgets.QWidget) -> None:
        """Substitui o placeholder do editor de estratégia pelo widget real."""
        # Editor é o segundo widget do splitter (índice 1) — left panel é o 0.
        self._workspace_splitter.replaceWidget(1, widget)
        self.strategy_editor_placeholder.hide()
        self.strategy_editor_placeholder = widget

    # ── Navigation ────────────────────────────────────────────────────────────

    def show_welcome_screen(self) -> None:
        self.content_stack.setCurrentIndex(0)

    def show_workspace_screen(self) -> None:
        self.content_stack.setCurrentIndex(1)
