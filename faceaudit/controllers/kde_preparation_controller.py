# faceaudit/controllers/kde_preparation_controller.py

import logging
from typing import List

from faceaudit.models.kde_config import KDEConfig
from faceaudit.models import EmbeddingDataset
from faceaudit.services.embedding_math_service import EmbeddingMathService

class KDEPreparationController:
    """
    Controlador de Engenharia de Dados para o KDE.
    
    Estratégia de Processamento:
    --------------------------
    Processa Probe e Reference SEPARADAMENTE antes de fundir.
    Iso é crucial para preservar características de domínio (ex: Frontal vs Lateral).
    Se fundíssemos antes, a média (centróide) destruiria a variação de pose que
    queremos modelar.
    """

    def __init__(self, math_service: EmbeddingMathService):
        self.math_svc = math_service
        self.logger = logging.getLogger("KDEPrepCtrl")

    def prepare_dataset(
        self, 
        config: KDEConfig, 
        probe_raw: EmbeddingDataset, 
        reference_raw: EmbeddingDataset
    ) -> EmbeddingDataset:
        """
        1. Aplica Agregação no Probe (se o DTO existir).
        2. Aplica Agregação no Reference (se o DTO existir).
        3. Funde os resultados em um único Dataset para a calibração NxN.
        """
        
        # 1. Preparação Individual (Preserva domínios distintos)
        probe_ready = self._process_single_dataset(config, probe_raw, "Probe")
        reference_ready = self._process_single_dataset(config, reference_raw, "Reference")

        # 2. Fusão Final (Merge)
        # Unificamos as listas para que o KDE possa rodar a matriz NxN completa.
        combined_items = probe_ready.items + reference_ready.items
        
        final_name = f"Calib_{probe_raw.name}_{reference_raw.name}"
        
        # Checa diretamente a existência do DTO de agregação
        if config.aggregation:
            # SANEAMENTO: Acesso seguro ao value do Enum
            agg_method = config.aggregation.method
            agg_val = agg_method.value if hasattr(agg_method, 'value') else str(agg_method)
            final_name += f"_Agg{agg_val}"

        self.logger.info(
            f"Fusão Concluída. "
            f"Probe({len(probe_ready)}) + Ref({len(reference_ready)}) -> Total({len(combined_items)})"
        )

        return EmbeddingDataset(
            name=final_name,
            description="Dataset Unificado (Probe+Ref) processado separadamente.",
            items=combined_items
        )

    def _process_single_dataset(
        self, 
        config: KDEConfig, 
        dataset: EmbeddingDataset,
        label: str
    ) -> EmbeddingDataset:
        """Helper para aplicar a lógica de agregação em um dataset isolado."""
        
        # Se o DTO não existe (agregação desligada), retorna o original
        if not config.aggregation:
            return dataset

        agg_cfg = config.aggregation
        
        # SANEAMENTO: Extração do valor para o log
        agg_val = agg_cfg.method.value if hasattr(agg_cfg.method, 'value') else str(agg_cfg.method)

        self.logger.info(
            f"[{label}] Agregando via {agg_val.upper()} "
            f"(K={agg_cfg.combination_group_size}, Weighted={bool(agg_cfg.quality_key_str)})"
        )

        # Chama o MathService passando APENAS o DTO!
        return self.math_svc.apply_smart_aggregation(
            dataset=dataset,
            config=agg_cfg,
            logger_callback=self.logger.info  
        )