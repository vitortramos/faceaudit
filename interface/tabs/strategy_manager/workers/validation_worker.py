from PyQt6 import QtCore

class ValidationWorker(QtCore.QThread):
    """
    Worker assíncrono para executar a Arena de Validação.
    Garante que os cálculos pesados de matrizes (N x M) e extração de LRs
    não congelem a interface gráfica, reportando os logs em tempo real.
    """
    progress_log = QtCore.pyqtSignal(str)
    
    # Diferente do ProjectWorker que retorna apenas uma mensagem de sucesso,
    # este worker precisa devolver o "Laudo" (ValidationComparisonReport) para a UI desenhar os gráficos.
    finished_success = QtCore.pyqtSignal(object) 
    finished_error = QtCore.pyqtSignal(str)

    def __init__(self, validation_manager, parent=None):
        super().__init__(parent)
        self.validation_manager = validation_manager
        self._is_cancelled = False

    def run(self):
        try:
            def logger_bridge(msg: str):
                if not self._is_cancelled:
                    self.progress_log.emit(msg)

            # Executa a arena de validação no backend
            report = self.validation_manager.run_validation(logger_callback=logger_bridge)

            # Se o usuário não clicou em Cancelar durante o processo...
            if not self._is_cancelled:
                # Verifica se o relatório tem de fato resultados
                if report and report.has_results:
                    self.finished_success.emit(report)
                else:
                    self.finished_error.emit("A validação foi concluída, mas não gerou resultados. Verifique os logs e os datasets.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_error.emit(f"Erro crítico durante a validação: {str(e)}")

    def cancel(self):
        """
        Sinaliza a interrupção. 
        O loop de inferência do Manager poderá checar esse estado (se implementado)
        ou apenas o log_bridge deixará de emitir.
        """
        self._is_cancelled = True