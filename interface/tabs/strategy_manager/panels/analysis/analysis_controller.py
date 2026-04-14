from PyQt6 import QtCore

class AnalysisController(QtCore.QObject):
    """
    Controlador responsável pela aba de Análise de Caso (Coluna 3).
    Gerencia a aplicação do modelo forense em casos reais.
    """
    def __init__(self, view, config):
        super().__init__()
        self.view = view
        self.config = config
        self.current_strategy = None
        
        print("[AnalysisController] Iniciado (Coluna 3)")

    def set_current_strategy(self, strategy_config):
        """Recebe a estratégia carregada na Coluna 1"""
        self.current_strategy = strategy_config
        print(f"[AnalysisController] Estratégia recebida: {strategy_config.name}")