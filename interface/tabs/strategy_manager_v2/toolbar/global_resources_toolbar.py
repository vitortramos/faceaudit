# interface/tabs/strategy_manager_v2/toolbar/global_resources_toolbar.py

from PyQt6 import QtWidgets, QtCore


class GlobalResourcesToolbar(QtWidgets.QToolBar):
    """
    Toolbar de recursos globais (biblioteca reutilizável entre projetos).

    Sinais:
        pipeline_requested   — usuário clicou em "Pipeline"
        mining_requested     — usuário clicou em "Mining"
        split_requested      — usuário clicou em "Split"
        manage_library_requested — usuário clicou em "Gerenciar Biblioteca"
    """

    pipeline_requested = QtCore.pyqtSignal()
    mining_requested = QtCore.pyqtSignal()
    split_requested = QtCore.pyqtSignal()
    manage_library_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Recursos Globais", parent)
        self.setMovable(False)
        self.setStyleSheet("""
            QToolBar { border: none; background: #f0f0f0; spacing: 4px; padding: 2px 4px; }
        """)
        self._build()

    def _build(self) -> None:
        label = QtWidgets.QLabel("  Biblioteca Global:  ")
        label.setStyleSheet("font-size: 11px; color: #555; font-weight: bold;")
        self.addWidget(label)

        self.btn_pipeline = self._make_button("🔗 Pipeline", "Criar ou editar configuração de pipeline de transformação")
        self.btn_pipeline.clicked.connect(self.pipeline_requested)
        self.addWidget(self.btn_pipeline)

        self.btn_mining = self._make_button("⛏ Mining", "Criar ou editar configuração de mineração de pares")
        self.btn_mining.clicked.connect(self.mining_requested)
        self.addWidget(self.btn_mining)

        self.btn_split = self._make_button("✂ Split", "Criar ou editar configuração de divisão de dataset")
        self.btn_split.clicked.connect(self.split_requested)
        self.addWidget(self.btn_split)

        self.addSeparator()

        self.btn_manage = self._make_button("📚 Gerenciar Biblioteca", "Visualizar e remover recursos da biblioteca global")
        self.btn_manage.clicked.connect(self.manage_library_requested)
        self.addWidget(self.btn_manage)

    @staticmethod
    def _make_button(text: str, tooltip: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setStyleSheet("""
            QPushButton {
                border: 1px solid #bbb; border-radius: 4px;
                background: white; padding: 3px 10px; font-size: 12px;
            }
            QPushButton:hover { background: #e8f0fe; border-color: #4a90d9; }
            QPushButton:pressed { background: #d2e3fc; }
        """)
        return btn
