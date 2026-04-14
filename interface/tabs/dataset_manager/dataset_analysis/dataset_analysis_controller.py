from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QProgressDialog, QMessageBox
from typing import Optional

# Imports do Domínio
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.services.comparison_service import ComparisonService

# View local e Worker
from .dataset_analysis_view import DatasetAnalysisView
from interface.tabs.dataset_manager.workers.dataset_analysis_worker import DatasetAnalysisWorker

class DatasetAnalysisController(QtCore.QObject):
    """
    Controlador do Diálogo de Auditoria e Mineração Biométrica.
    """
    def __init__(self, 
                 dataset: FaceDataset, 
                 comparison_service: ComparisonService, 
                 parent_view: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent_view)
        
        self.dataset = dataset
        self.comparison_service = comparison_service # Recebido explicitamente!
        
        # Inicializa a View
        self.view = DatasetAnalysisView(dataset_name=self.dataset.name, parent=parent_view)
        
        # Conecta botões
        self.view.btn_run_mining.clicked.connect(self._run_biometric_mining)
        
        # Já roda a análise rápida na hora de abrir
        self._run_hash_analysis()

    def exec(self):
        """Exibe o diálogo modalmente."""
        return self.view.exec()

    def _run_hash_analysis(self):
        """
        Executa a verificação de integridade O(N) que já está pronta no modelo.
        """
        # Chama o método que acabamos de criar no FaceDataset
        results = self.dataset.analyze_hash_integrity()
        
        consistent = results.get("consistent", [])
        inconsistent = results.get("inconsistent", [])
        
        # Manda a View desenhar as tabelas
        self.view.build_hash_table(consistent, inconsistent)

    def _run_biometric_mining(self):
        """Inicia a autocomparação na thread secundária."""
        # Desabilita o botão para evitar duplos cliques
        self.view.btn_run_mining.setEnabled(False)

        # Configura o diálogo de progresso
        self.progress_mining = QProgressDialog("Preparando Mineração...", "Cancelar", 0, 100, self.view)
        self.progress_mining.setWindowTitle("Mineração Biométrica")
        self.progress_mining.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress_mining.setMinimumDuration(0)

        # Configuração da Thread
        self.thread = QtCore.QThread()
        # Aqui podemos configurar o 'top_n' para 100, 50, etc.
        self.worker = DatasetAnalysisWorker(self.dataset, self.comparison_service, top_n=100)
        self.worker.moveToThread(self.thread)

        # Sinais de Execução
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(lambda msg, pct: self.progress_mining.setLabelText(msg))
        self.worker.progress_update.connect(lambda msg, pct: self.progress_mining.setValue(pct))
        
        # Sinais de Conclusão e Erro
        self.worker.finished.connect(self._on_mining_success)
        self.worker.error.connect(self._on_mining_error)
        self.progress_mining.canceled.connect(self.worker.stop)
        
        # Limpeza de Memória da Thread (Extremamente Importante para Mineração)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def _on_mining_success(self, df_genuines, df_impostors):
        if hasattr(self, 'progress_mining'): 
            self.progress_mining.close()
            
        # ==========================================
        # 🐛 DEBUG: RASTREAMENTO DOS DATAFRAMES
        # ==========================================
        print("\n" + "="*40)
        print("🔍 DEBUG DE MINERAÇÃO BIOMÉTRICA")
        print("="*40)
        
        # Analisando Genuínos
        if df_genuines is not None:
            print(f"🔴 Genuínos -> Total de Linhas: {len(df_genuines)}")
            if not df_genuines.empty:
                print(f"   Colunas: {df_genuines.columns.tolist()}")
        else:
            print("🔴 Genuínos -> CHEGOU COMO NONE!")

        # Analisando Impostores
        if df_impostors is not None:
            print(f"🟠 Impostores -> Total de Linhas: {len(df_impostors)}")
            if not df_impostors.empty:
                print(f"   Colunas: {df_impostors.columns.tolist()}")
        else:
            print("🟠 Impostores -> CHEGOU COMO NONE!")
            
        print("="*40 + "\n")
        # ==========================================
        
        # Passa os DataFrames para a View desenhar as tabelas
        self.view.build_mining_tables(df_genuines, df_impostors)
        
        # Altera o visual do botão para dar feedback visual
        self.view.btn_run_mining.setText("✅ Mineração Concluída")
        self.view.btn_run_mining.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px;")

    def _on_mining_error(self, msg):
        if hasattr(self, 'progress_mining'): 
            self.progress_mining.close()
        self.view.btn_run_mining.setEnabled(True)
        QMessageBox.critical(self.view, "Erro de Processamento", msg)