from faceaudit.models.filter_config import FilterConfig
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from .enrich_dataset_view import EnrichDatasetView

class EnrichDatasetController:
    """
    Controlador da janela de Enriquecimento.
    Analisa o dataset atual para sugerir quais processamentos estão faltando.
    """
    def __init__(self, dataset, parent=None):
        self.dataset = dataset
        self.view = EnrichDatasetView(parent)
        
        # Analisa o dataset e preenche a UI com estatísticas
        self._analyze_and_populate()

    def exec(self):
        return self.view.exec()

    def get_config(self) -> DatasetIngestionConfig:
        """
        Gera um DatasetIngestionConfig estritamente compatível com o Dataclass fornecido.
        """
        from pathlib import Path # Garante o import caso não esteja no topo

        return DatasetIngestionConfig(
            # 1. Campos Obrigatórios (Necessários para o __init__ do Dataclass)
            # Passamos um caminho fictício pois o método 'enrich_existing_dataset' do pipeline ignora isso.
            input_path=Path("."), 
            dataset_name=self.dataset.name,
            
            # 2. Configurações de Processamento (Mapeando da View para o Config)
            compute_ff=self.view.is_checked('ff'),
            compute_ofiq=self.view.is_checked('ofiq'),
            compute_serfiq=self.view.is_checked('serfiq')
            
            # OBS: Removemos 'extract_features', 'enable_forensic', etc., 
            # pois eles não existem na definição da sua classe.
        )

    def _analyze_and_populate(self):
        """Conta quantos registros já têm FF, OFIQ e SERFIQ calculados."""
        total = len(self.dataset)
        if total == 0:
            return

        count_ff = 0
        count_ofiq = 0
        count_serfiq = 0

        # Scan rápido na memória (é muito rápido pois não lê disco)
        for face in self.dataset.faces:
            if face.ff_info is not None: count_ff += 1
            if face.ofiq_info is not None: count_ofiq += 1
            if face.serfiq_info is not None: count_serfiq += 1

        # Atualiza a UI com a lógica de sugestão
        self._update_ui_state('ff', count_ff, total)
        self._update_ui_state('ofiq', count_ofiq, total)
        self._update_ui_state('serfiq', count_serfiq, total)

    def _update_ui_state(self, key, count, total):
        pct = (count / total) * 100
        
        status_text = ""
        color = ""
        auto_check = False

        if count == 0:
            status_text = "⚠️ Ausente (0%)"
            color = "#d9534f" # Vermelho
            auto_check = True # Sugere fazer
            
        elif count == total:
            status_text = "✅ Completo (100%)"
            color = "#5cb85c" # Verde
            auto_check = False # Já tem, não precisa refazer (mas pode)
            
        else:
            status_text = f"🚧 Parcial ({int(pct)}%)"
            color = "#f0ad4e" # Laranja
            auto_check = True # Sugere terminar

        self.view.set_status(key, status_text, color)
        self.view.set_checked(key, auto_check)