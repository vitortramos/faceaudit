from PyQt6 import QtCore
from .replace_dialog_view import ReplaceDialogView

class ReplaceDialogController(QtCore.QObject):
    """
    Controlador do Diálogo de Substituição.
    Responsabilidade: Iniciar a View, conectar sinais e retornar os dados processados.
    """
    def __init__(self, parent_view=None):
        super().__init__()
        self.view = ReplaceDialogView(parent_view)
        self._connect_signals()

    def _connect_signals(self):
        # Conecta os botões da View aos slots da própria View (accept/reject do QDialog)
        self.view.buttons.accepted.connect(self.view.accept)
        self.view.buttons.rejected.connect(self.view.reject)

    def exec(self) -> bool:
        """Exibe o diálogo e retorna True se o usuário clicou em OK."""
        return self.view.exec()

    def get_data(self):
        """
        Retorna uma tupla com os dados inseridos na View.
        Retorno: (find_text, replace_text, exact_match_bool)
        """
        return (
            self.view.input_find.text(),
            self.view.input_replace.text(),
            self.view.check_exact.isChecked()
        )