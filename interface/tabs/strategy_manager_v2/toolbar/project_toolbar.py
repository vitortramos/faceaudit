"""Toolbar do Strategy Manager V2 no formato Ribbon (refator).

Antes: `QToolBar` single-row com labels separadores ("Projeto:" /
"Biblioteca Global:") agrupando os botões.

Agora: `QFrame` com `QTabWidget` interno + 2 ribbons (Projeto +
Biblioteca Global) + botão de toggle do painel lateral como
`cornerWidget` da QTabBar (sempre visível, independente da aba).

Mesmo padrão visual da toolbar do Dataset Manager — accent bar
azul (`#1976d2`) na aba selecionada, fundo branco contínuo com a
pane, hover escurecendo as inativas.

**Contrato preservado**: signals, button aliases (btn_new, btn_open,
…) e métodos públicos (`set_project_open`, `set_left_panel_collapsed`)
permanecem identicos — controller (`strategy_manager_v2_controller`)
não muda.
"""

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QFrame, QTabWidget, QVBoxLayout

from .ribbons.biblioteca_ribbon import BibliotecaRibbon
from .ribbons.projeto_ribbon import ProjetoRibbon


_LEFT_PANEL_LABEL_EXPANDED = "◀ Recolher painel"
_LEFT_PANEL_LABEL_COLLAPSED = "▶ Mostrar painel"


class ProjectToolbar(QFrame):
    """Toolbar Ribbon do Strategy Manager V2 — 2 abas (Projeto +
    Biblioteca Global) + botão de toggle no canto.

    Sinais (preservados — não mude sem atualizar
    `strategy_manager_v2_controller._connect_signals`):
        new_project_requested, open_project_requested,
        save_project_requested, close_project_requested,
        seal_project_requested,
        pipeline_requested, mining_requested, split_requested,
        manage_library_requested,
        toggle_left_panel_requested
    """

    new_project_requested = QtCore.pyqtSignal()
    open_project_requested = QtCore.pyqtSignal()
    save_project_requested = QtCore.pyqtSignal()
    close_project_requested = QtCore.pyqtSignal()
    seal_project_requested = QtCore.pyqtSignal()  # PR L2.b — selar projeto
    add_resources_requested = QtCore.pyqtSignal()  # bulk import da biblioteca
    pipeline_requested = QtCore.pyqtSignal()
    mining_requested = QtCore.pyqtSignal()
    split_requested = QtCore.pyqtSignal()
    manage_library_requested = QtCore.pyqtSignal()
    toggle_left_panel_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_style()
        self._init_ui()
        self._wire_button_aliases()
        self._connect_internal_signals()
        self.set_project_open(False)

    # ── Style + structure ────────────────────────────────────────────

    def _init_style(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMaximumHeight(110)  # tabs + ribbon row, compacto
        # Vertical = Maximum: nunca expandir além do sizeHint, mesmo se
        # o pai (QVBoxLayout do view) tiver espaço sobrando — evita que
        # o QFrame "incha" e cria um gap entre a aba principal e o ribbon.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        self.setStyleSheet(
            "QFrame { background-color: #f5f5f5; "
            "border-radius: 5px; border: 1px solid #e0e0e0; }"
        )

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        # Mesmo accent-bar azul (`#1976d2`) usado na toolbar do Dataset
        # Manager — coerência visual entre as duas abas principais.
        self.tabs.setStyleSheet(
            "QTabWidget::pane { "
            "    border: 1px solid #d0d0d0; "
            "    background: white; "
            "    border-radius: 4px; "
            "    top: -1px; "
            "} "
            "QTabBar::tab { "
            "    background: #e8e8e8; "
            "    color: #555; "
            "    padding: 7px 20px; "
            "    font-size: 11px; "
            "    border: 1px solid #d0d0d0; "
            "    border-top: 3px solid transparent; "
            "    border-right: none; "
            "    border-top-left-radius: 4px; "
            "    border-top-right-radius: 4px; "
            "    margin-right: 1px; "
            "} "
            "QTabBar::tab:hover:!selected { "
            "    background: #dcdcdc; "
            "    color: #333; "
            "} "
            "QTabBar::tab:selected { "
            "    background: white; "
            "    color: #1976d2; "
            "    font-weight: bold; "
            "    border-top: 3px solid #1976d2; "
            "    border-bottom: 1px solid white; "
            "} "
            "QTabBar::tab:last { margin-right: 0; }"
        )

        self.projeto_ribbon = ProjetoRibbon()
        self.biblioteca_ribbon = BibliotecaRibbon()

        self.tabs.addTab(self.projeto_ribbon, "Projeto")
        self.tabs.addTab(self.biblioteca_ribbon, "Biblioteca de Configurações")
        self.tabs.setCurrentIndex(0)

        # Botão de toggle do painel — corner widget da QTabBar.
        # Sempre visível independente da aba selecionada (faz sentido
        # porque é uma ação de visualização, não de projeto/biblioteca).
        self.btn_toggle_left_panel = QtWidgets.QPushButton(
            _LEFT_PANEL_LABEL_EXPANDED,
        )
        self.btn_toggle_left_panel.setToolTip(
            "Mostrar/esconder o painel lateral (Cmd+B)",
        )
        self.btn_toggle_left_panel.setStyleSheet(
            "QPushButton { "
            "    border: 1px solid #bbb; border-radius: 4px; "
            "    background: white; padding: 3px 12px; font-size: 11px; "
            "    margin: 4px 6px; "
            "} "
            "QPushButton:hover { background: #e8f0fe; border-color: #4a90d9; }"
        )
        self.tabs.setCornerWidget(
            self.btn_toggle_left_panel, QtCore.Qt.Corner.TopRightCorner,
        )

        main_layout.addWidget(self.tabs)

    def _wire_button_aliases(self) -> None:
        """Cria atalhos `self.btn_<x>` delegando aos ribbons. Preserva
        100% da API pública anterior (controller continua acessando
        `toolbar.btn_save`, etc.)."""
        # Projeto
        self.btn_new = self.projeto_ribbon.btn_new
        self.btn_open = self.projeto_ribbon.btn_open
        self.btn_save = self.projeto_ribbon.btn_save
        self.btn_close = self.projeto_ribbon.btn_close
        self.btn_seal = self.projeto_ribbon.btn_seal
        self.btn_add_resources = self.projeto_ribbon.btn_add_resources
        # Biblioteca
        self.btn_pipeline = self.biblioteca_ribbon.btn_pipeline
        self.btn_mining = self.biblioteca_ribbon.btn_mining
        self.btn_split = self.biblioteca_ribbon.btn_split
        self.btn_manage = self.biblioteca_ribbon.btn_manage

    def _connect_internal_signals(self) -> None:
        """Conecta os botões internos dos ribbons aos sinais públicos
        do toolbar (que o controller subscreve)."""
        self.btn_new.clicked.connect(self.new_project_requested)
        self.btn_open.clicked.connect(self.open_project_requested)
        self.btn_save.clicked.connect(self.save_project_requested)
        self.btn_close.clicked.connect(self.close_project_requested)
        self.btn_seal.clicked.connect(self.seal_project_requested)
        self.btn_add_resources.clicked.connect(self.add_resources_requested)
        self.btn_pipeline.clicked.connect(self.pipeline_requested)
        self.btn_mining.clicked.connect(self.mining_requested)
        self.btn_split.clicked.connect(self.split_requested)
        self.btn_manage.clicked.connect(self.manage_library_requested)
        self.btn_toggle_left_panel.clicked.connect(
            self.toggle_left_panel_requested,
        )

    # ── API pública (preservada — usada pelo controller) ────────────

    def set_left_panel_collapsed(self, collapsed: bool) -> None:
        """Atualiza o label do botão para refletir o estado atual do
        painel."""
        self.btn_toggle_left_panel.setText(
            _LEFT_PANEL_LABEL_COLLAPSED
            if collapsed else _LEFT_PANEL_LABEL_EXPANDED
        )

    def set_project_open(self, is_open: bool) -> None:
        """Habilita/desabilita botões que dependem de um projeto aberto."""
        self.btn_save.setEnabled(is_open)
        self.btn_close.setEnabled(is_open)
        self.btn_seal.setEnabled(is_open)  # PR L2.b
        self.btn_add_resources.setEnabled(is_open)
