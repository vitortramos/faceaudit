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
        Monta o DatasetIngestionConfig com os flags binários por serviço +
        params operacionais (FF/SER-FIQ) lidos dos painéis avançados.
        """
        from pathlib import Path

        return DatasetIngestionConfig(
            # Path fictício — o pipeline.enrich_existing_dataset ignora.
            input_path=Path("."),
            dataset_name=self.dataset.name,

            # Flags por serviço
            compute_ff=self.view.is_checked('ff'),
            compute_ofiq=self.view.is_checked('ofiq'),
            compute_serfiq=self.view.is_checked('serfiq'),

            # Params operacionais (defaults se o painel avançado não foi expandido)
            ff_models=self.view.get_ff_models(),
            ff_det_size=self.view.get_ff_det_size(),
            ff_extended=self.view.get_ff_extended(),
            serfiq_t=self.view.get_serfiq_t(),
            serfiq_alpha=self.view.get_serfiq_alpha(),

            # PR 18: idempotência — quando False (default), o pipeline pula
            # faces que já têm o enrichment solicitado. True força tudo.
            force_reprocess=self.view.chk_force_reprocess.isChecked(),
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