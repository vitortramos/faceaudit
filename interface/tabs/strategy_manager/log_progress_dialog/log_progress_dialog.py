# faceaudit/interface/tabs/strategy_manager/panels/preparation/cards/log_progress_dialog/log_progress_dialog.py

from PyQt6 import QtWidgets, QtCore, QtGui

from interface.theme import current_token_fn

class LogProgressDialog(QtWidgets.QDialog):
    """
    Janela modal para exibir logs. Suporta cancelamento.
    """
    cancellation_requested = QtCore.pyqtSignal()

    def __init__(self, title="Processando...", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 400)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        
        # Evita fechar pelo 'X' durante processamento (força usar o botão Cancelar)
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.WindowCloseButtonHint)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Área de Texto
        self.text_log = QtWidgets.QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet("""
            QTextEdit {
                font-family: Consolas, Monospace; 
                font-size: 9pt; 
                background-color: #1e1e1e; 
                color: #d4d4d4;
                border: 1px solid #555;
            }
        """)
        layout.addWidget(self.text_log)
        
        # Barra de Progresso
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminado
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Botão de Ação (Cancelar / Fechar)
        self.btn_action = QtWidgets.QPushButton("Cancelar")
        self.btn_action.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_action.clicked.connect(self._on_action_clicked)
        layout.addWidget(self.btn_action)
        
        self.finished_process = False

    def _on_action_clicked(self):
        if self.finished_process:
            self.accept() # Fecha a janela
        else:
            # Emite sinal de cancelamento e desabilita para não clicar 2x
            self.text_log.append("\n⚠️ Solicitando cancelamento...")
            self.btn_action.setEnabled(False)
            self.cancellation_requested.emit()

    def append_log(self, msg: str):
        self.text_log.append(msg)
        sb = self.text_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def finish(self, success: bool = True):
        """Muda o estado visual para finalizado."""
        self.finished_process = True
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowCloseButtonHint)
        self.show() # Refresh flags

        self.btn_action.setText("Fechar")
        self.btn_action.setEnabled(True)
        # Sempre revela o botão ao concluir — em fluxos sem cancelamento
        # (allow_cancel=False) o runner o esconde durante o processamento;
        # ao terminar, "Fechar" precisa reaparecer (senão a janela fica sem
        # botão de fechar embaixo).
        self.btn_action.setVisible(True)

        t = current_token_fn()
        if success:
            self.text_log.append("\n✅ PROCESSO CONCLUÍDO.")
            self.progress_bar.setStyleSheet(
                "QProgressBar::chunk { background-color: " + t("SUCCESS") + "; }"
            )
        else:
            self.text_log.append("\n❌ PROCESSO INTERROMPIDO OU FALHOU.")
            self.progress_bar.setStyleSheet(
                "QProgressBar::chunk { background-color: " + t("DANGER") + "; }"
            )