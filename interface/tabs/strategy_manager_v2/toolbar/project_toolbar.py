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


# Labels curtas — só seta. Texto longo no corner widget do QTabWidget
# era promovido pra uma "linha" extra acima das tabs em telas
# pequenas e ainda sobrepunha a borda do QTabWidget::pane (linha fina
# horizontal aparente). Tooltip preserva a semântica completa.
_LEFT_PANEL_LABEL_EXPANDED = "◀"
_LEFT_PANEL_LABEL_COLLAPSED = "▶"


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
    verify_integrity_requested = QtCore.pyqtSignal()  # re-check on-demand
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
        # Painel inicia expandido — esconde o botão da toolbar
        # (equivalente vive no header do LeftPanelWidget). Aparece
        # automaticamente quando o usuário colapsa o painel.
        self.btn_toggle_left_panel.setVisible(False)

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

        # Botão de toggle do painel. Histórico:
        # - Tentativa 1: corner widget da QTabBar. Em telas pequenas
        #   (MacBook 14") o Qt promovia o widget pra uma linha SEPARADA
        #   acima das tabs e ainda sobrepunha a borda do
        #   `QTabWidget::pane` (linha fina aparente).
        # - Solução: empilhar [tabs | btn] num QHBoxLayout EXTERNO,
        #   garantindo que o botão fica na MESMA linha das tabs,
        #   alinhado verticalmente, sem reflow exótico.
        self.btn_toggle_left_panel = QtWidgets.QPushButton(
            _LEFT_PANEL_LABEL_EXPANDED,
        )
        self.btn_toggle_left_panel.setToolTip(
            "Mostrar/esconder o painel lateral (Cmd+B)",
        )
        self.btn_toggle_left_panel.setFixedSize(28, 24)
        self.btn_toggle_left_panel.setStyleSheet(
            "QPushButton { "
            "    border: 1px solid #bbb; border-radius: 4px; "
            "    background: white; font-size: 13px; "
            "    padding: 0; "
            "} "
            "QPushButton:hover { background: #e8f0fe; border-color: #4a90d9; }"
        )

        # Container horizontal: tabs (expand) | btn (fixo). Alinha o
        # botão no topo direito, no mesmo nível que a primeira linha de
        # tabs do QTabWidget.
        tabs_row = QtWidgets.QHBoxLayout()
        tabs_row.setContentsMargins(0, 0, 0, 0)
        tabs_row.setSpacing(4)
        tabs_row.addWidget(self.tabs, 1)
        btn_holder = QtWidgets.QWidget()
        btn_holder_layout = QtWidgets.QVBoxLayout(btn_holder)
        btn_holder_layout.setContentsMargins(0, 4, 4, 0)
        btn_holder_layout.addWidget(
            self.btn_toggle_left_panel,
            alignment=QtCore.Qt.AlignmentFlag.AlignTop,
        )
        btn_holder_layout.addStretch()
        tabs_row.addWidget(btn_holder, 0)

        main_layout.addLayout(tabs_row)

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
        self.btn_verify_integrity = self.projeto_ribbon.btn_verify_integrity
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
        self.btn_verify_integrity.clicked.connect(self.verify_integrity_requested)
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
        painel. Botão aqui na toolbar fica VISÍVEL APENAS quando o
        painel está colapsado — quando aberto, o botão equivalente
        vive dentro do próprio LeftPanelWidget (mais intuitivo, evita
        botão "perdido" no canto da toolbar)."""
        self.btn_toggle_left_panel.setText(
            _LEFT_PANEL_LABEL_COLLAPSED
            if collapsed else _LEFT_PANEL_LABEL_EXPANDED
        )
        self.btn_toggle_left_panel.setVisible(collapsed)

    def set_project_open(self, is_open: bool) -> None:
        """Habilita/desabilita botões que dependem de um projeto aberto."""
        self.btn_save.setEnabled(is_open)
        self.btn_close.setEnabled(is_open)
        self.btn_seal.setEnabled(is_open)  # PR L2.b
        self.btn_add_resources.setEnabled(is_open)
        self.btn_verify_integrity.setEnabled(is_open)
