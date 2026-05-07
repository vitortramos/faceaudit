# interface/tabs/strategy_manager_v2/toolbar/project_toolbar.py

from PyQt6 import QtWidgets, QtCore


_LEFT_PANEL_LABEL_EXPANDED = "◀ Recolher painel"
_LEFT_PANEL_LABEL_COLLAPSED = "▶ Mostrar painel"


class ProjectToolbar(QtWidgets.QToolBar):
    """
    Toolbar unificada: ações de projeto + biblioteca global em uma única linha.

    Sinais:
        new_project_requested
        open_project_requested
        save_project_requested
        close_project_requested
        pipeline_requested
        mining_requested
        split_requested
        manage_library_requested
    """

    new_project_requested = QtCore.pyqtSignal()
    open_project_requested = QtCore.pyqtSignal()
    save_project_requested = QtCore.pyqtSignal()
    close_project_requested = QtCore.pyqtSignal()
    seal_project_requested = QtCore.pyqtSignal()  # PR L2.b — selar projeto
    pipeline_requested = QtCore.pyqtSignal()
    mining_requested = QtCore.pyqtSignal()
    split_requested = QtCore.pyqtSignal()
    manage_library_requested = QtCore.pyqtSignal()
    toggle_left_panel_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Projeto", parent)
        self.setMovable(False)
        self.setStyleSheet("""
            QToolBar { border: none; background: #f4f4f4; spacing: 4px; padding: 2px 4px; }
        """)
        self._build()
        self.set_project_open(False)

    def _build(self) -> None:
        # ── Grupo Projeto ────────────────────────────────────────────────────
        lbl_proj = QtWidgets.QLabel("  Projeto:  ")
        lbl_proj.setStyleSheet("font-size: 11px; color: #555; font-weight: bold;")
        self.addWidget(lbl_proj)

        self.btn_new = self._make_button("Novo", "Criar novo projeto FaceAudit")
        self.btn_new.clicked.connect(self.new_project_requested)
        self.addWidget(self.btn_new)

        self.btn_open = self._make_button("Abrir", "Abrir projeto FaceAudit existente (.fa)")
        self.btn_open.clicked.connect(self.open_project_requested)
        self.addWidget(self.btn_open)

        self.btn_save = self._make_button("Salvar", "Salvar projeto atual")
        self.btn_save.clicked.connect(self.save_project_requested)
        self.addWidget(self.btn_save)

        self.btn_close = self._make_button("Fechar", "Fechar projeto atual")
        self.btn_close.clicked.connect(self.close_project_requested)
        self.addWidget(self.btn_close)

        # PR L2.b — Selar projeto (copia imagens pra dentro do .fa)
        self.btn_seal = self._make_button(
            "🔒 Selar",
            "Selar projeto: copia todas as imagens referenciadas pra "
            "dentro do .fa e reescreve paths como relativos. Empacota "
            "o projeto pra entrega forense (USB/zip/laudo).",
        )
        self.btn_seal.clicked.connect(self.seal_project_requested)
        self.addWidget(self.btn_seal)

        self.addSeparator()

        # ── Grupo Biblioteca Global ──────────────────────────────────────────
        lbl_lib = QtWidgets.QLabel("  Biblioteca Global:  ")
        lbl_lib.setStyleSheet("font-size: 11px; color: #555; font-weight: bold;")
        self.addWidget(lbl_lib)

        self.btn_pipeline = self._make_button("Pipeline", "Criar ou editar configuração de pipeline de transformação")
        self.btn_pipeline.clicked.connect(self.pipeline_requested)
        self.addWidget(self.btn_pipeline)

        self.btn_mining = self._make_button("Mining", "Criar ou editar configuração de mineração de pares")
        self.btn_mining.clicked.connect(self.mining_requested)
        self.addWidget(self.btn_mining)

        self.btn_split = self._make_button("Split", "Criar ou editar configuração de divisão de dataset")
        self.btn_split.clicked.connect(self.split_requested)
        self.addWidget(self.btn_split)

        self.btn_manage = self._make_button("Gerenciar Biblioteca", "Visualizar e remover recursos da biblioteca global")
        self.btn_manage.clicked.connect(self.manage_library_requested)
        self.addWidget(self.btn_manage)

        # Stretch espacial empurra o botão de toggle pra direita.
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.addWidget(spacer)

        self.btn_toggle_left_panel = self._make_button(
            _LEFT_PANEL_LABEL_EXPANDED,
            "Mostrar/esconder o painel lateral (Cmd+B)",
        )
        self.btn_toggle_left_panel.clicked.connect(self.toggle_left_panel_requested)
        self.addWidget(self.btn_toggle_left_panel)

    def set_left_panel_collapsed(self, collapsed: bool) -> None:
        """Atualiza o label do botão para refletir o estado atual do painel."""
        self.btn_toggle_left_panel.setText(
            _LEFT_PANEL_LABEL_COLLAPSED if collapsed else _LEFT_PANEL_LABEL_EXPANDED
        )

    def set_project_open(self, is_open: bool) -> None:
        """Habilita/desabilita botões que dependem de um projeto aberto."""
        self.btn_save.setEnabled(is_open)
        self.btn_close.setEnabled(is_open)
        self.btn_seal.setEnabled(is_open)  # PR L2.b

    @staticmethod
    def _make_button(text: str, tooltip: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #bbb; border-radius: 4px;
                background: white; padding: 3px 10px; font-size: 11px;
            }
            QPushButton:hover { background: #e8f0fe; border-color: #4a90d9; }
            QPushButton:pressed { background: #d2e3fc; }
            QPushButton:disabled { background: #f0f0f0; color: #aaa; border-color: #ddd; }
        """)
        return btn
