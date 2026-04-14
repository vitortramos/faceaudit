from pathlib import Path
from typing import Optional, Callable, Tuple

# Models & Enums
from faceaudit.models.plda_config import PLDAConfig
from faceaudit.models import EmbeddingDataset
from faceaudit.strategies.plda_strategy import PLDAStrategy

# Novos imports para Split e Mineração
from faceaudit.services.plda_split_service import PLDASplitService
from faceaudit.models.split_config import SplitConfig
from faceaudit.models.comparison_mining_config import ComparisonMiningConfig
from faceaudit.services.comparison_mining_service import ComparisonMiningService

class PLDATrainingController:
    """
    Controlador responsável pela orquestração do fluxo de treinamento do PLDA.

    Principais responsabilidades:
    1) Engenharia de Dados (Delega a divisão de treino/calibração ao PLDASplitService).
    2) Inicialização Dinâmica da Estratégia Matemática.
    3) Injeção de Regras (Passa o ComparisonMiningConfig para a calibração da Regressão Logística).
    4) Ciclo de Vida e Persistência do modelo (EM + LogReg).
    """

    def __init__(self,
                 config: PLDAConfig,
                 split_service: Optional[PLDASplitService] = None,
                 mining_service: Optional[ComparisonMiningService] = None, # <--- ADICIONE AQUI
                 logger: Optional[Callable[[str], None]] = None):
        self.config = config
        self.logger = logger or print
        
        self.split_service = split_service or PLDASplitService()
        self.mining_service = mining_service or ComparisonMiningService()

    def _log(self, msg: str):
        if self.logger:
            self.logger(msg)

    # =========================================================================
    # DATA PREP (Delegado ao Serviço de Split usando DTO Global)
    # =========================================================================

    def prepare_datasets_for_training(
        self,
        source_dataset: EmbeddingDataset,
        target_dataset: Optional[EmbeddingDataset],
        split_config: SplitConfig
    ) -> Tuple[EmbeddingDataset, EmbeddingDataset]:
        """
        Delega o split inteligente para o PLDASplitService aplicando a regra
        específica selecionada pelo perito.
        """
        self._log(f"✂️ [Engenharia de Dados] Cortando as bases de treino usando a regra '{split_config.name}'...")

        plda_set, calib_set = self.split_service.perform_split(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            strategy_type=self.config.strategy,
            config=split_config,
            logger_callback=self._log
        )
        
        return plda_set, calib_set

    # =========================================================================
    # EXECUÇÃO ORQUESTRADA
    # =========================================================================

    def run(
        self,
        source_dataset: EmbeddingDataset,
        target_dataset: Optional[EmbeddingDataset],
        split_config: SplitConfig,
        mining_config: ComparisonMiningConfig, # <--- CORREÇÃO AQUI TAMBÉM
        output_path: Path
    ) -> str:
        """
        Orquestra o ciclo completo de vida do modelo PLDA.
          1) Executa o Split configurado
          2) Valida se há dados residuais suficientes
          3) Instancia o modelo matemático
          4) Treina as matrizes (EM) e as probabilidades (LogReg) injetando a regra de Mineração
          5) Salva o binário na Cadeia de Custódia
        """
        self._log(f"🔧 [Treino] Configurando fluxo de execução para ID: {self.config.id}")

        # 1) Split Guiado Pelo JSON
        try:
            plda_data, calib_data = self.prepare_datasets_for_training(
                source_dataset, target_dataset, split_config
            )
        except Exception as e:
            self._log(f"❌ [Erro Critico] Falha na preparação dos dados: {e}")
            raise

        # 2) Safety checks contra cortes muito agressivos
        if len(plda_data) < 10:
            raise ValueError(
                f"Dados insuficientes para treino PLDA após o split. Restaram apenas {len(plda_data)} amostras."
            )

        unique_calib = len(set(i.identity for i in calib_data.items if i.identity is not None))
        if unique_calib < 2:
            raise ValueError(
                f"Calibração impossível: Apenas {unique_calib} identidade(s). Requer mínimo de 2."
            )

        # 3) Instancia estratégia Matemática (Parâmetros purificados)
        params = self.config.parameters
        
        # Constrói apenas os hiperparâmetros matemáticos que restaram após a refatoração
        strategy_kwargs = {
            "em_iterations": params.em_iterations,
            "plda_mode": params.plda_mode,
            "latent_dim": params.latent_dim,
            "logreg_C": params.logreg_C
        }

        strategy = PLDAStrategy(
            name=output_path.stem,
            **strategy_kwargs
        )

        mode_val = params.plda_mode.value if hasattr(params.plda_mode, "value") else str(params.plda_mode)
        self._log(f"🧠 [Treino] Inicializando PLDA (Mode: {mode_val}, Dim: {params.latent_dim})...")

        # 4) Fit (Injeta o ComparisonMiningConfig e o Service diretamente)
        strategy.fit(
            train_dataset=plda_data,
            calibration_dataset=calib_data,
            mining_service=self.mining_service, # <--- ADICIONE ESTA LINHA
            mining_config=mining_config, 
            logger_callback=self._log
        )

        # 5) Persistência Forense
        self._log(f"💾 [Persistência] Salvando modelo serializado em: {output_path.name}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        strategy.save(str(output_path))

        self._log("✅ [Treino] Processo concluído com sucesso.")
        return str(output_path)