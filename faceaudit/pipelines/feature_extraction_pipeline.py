import logging
from pathlib import Path
from typing import Optional, Callable

# Imports do Domínio
from faceaudit.models import FaceDataset
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter

# Imports dos Serviços
from faceaudit.services import (
    FFService, 
    HashService, 
    OFIQService, 
    SERFIQService
)

class FeatureExtractionPipeline:
    """
    Orquestrador ETL (Extract, Transform, Load) para Datasets Faciais.

    Esta classe é responsável por coordenar todo o ciclo de vida da ingestão de um dataset,
    desde a leitura dos arquivos em disco até a extração de características biométricas
    e persistência final.

    Responsabilidades:
        1. **Extract (Extração):** Utiliza o `FaceDatasetAdapter` para varrer diretórios,
           aplicar templates de metadados ou Regex, e criar a estrutura inicial do dataset.
        2. **Transform (Transformação):** Orquestra a chamada sequencial dos serviços de IA
           (Biometria, Qualidade ISO, Robustez) conforme configurado no objeto `DatasetIngestionConfig`.
        3. **Load (Carga):** Persiste o resultado processado em um arquivo Parquet otimizado
           para leitura rápida futura.

    Attributes:
        storage_path (Path): Diretório base onde os arquivos .parquet processados serão salvos.
        logger (logging.Logger): Logger para registro de eventos e erros do pipeline.
    """

    def __init__(self, 
                 hash_service: HashService,
                 ff_service: FFService,
                 ofiq_service: OFIQService,
                 serfiq_service: SERFIQService,
                 storage_path: Path):
        """
        Inicializa o pipeline injetando as dependências de serviço.

        Args:
            hash_service (HashService): Serviço para cálculo de integridade (SHA256).
            ff_service (FFService): Serviço de IA (ForensicFace) para detecção e embeddings.
            ofiq_service (OFIQService): Serviço de cálculo de qualidade padrão ISO/IEC 29794-5.
            serfiq_service (SERFIQService): Serviço de cálculo de robustez estocástica (SER-FIQ).
            storage_path (Path): Caminho do diretório onde os datasets processados serão salvos.
        """
        self.hash_service = hash_service
        self.ff_service = ff_service
        self.ofiq_service = ofiq_service
        self.serfiq_service = serfiq_service
        
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger("FeatureExtractionPipeline")

        # Garante a existência do diretório de armazenamento na inicialização
        if not self.storage_path.exists():
            try:
                self.storage_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.critical(f"Falha ao criar diretório de storage {self.storage_path}: {e}")

    def execute(self, 
                config: DatasetIngestionConfig, 
                progress_callback: Optional[Callable[[str, int], None]] = None
                ) -> FaceDataset:
        """
        Executa o fluxo completo de ingestão (ETL) baseado no objeto de configuração.
        (Sempre varre os diretórios do zero, ignorando caches antigos).

        Args:
            config (DatasetIngestionConfig): DTO contendo todas as regras de ingestão
                (caminhos, flags de processamento, templates, metadados padrão).
            progress_callback (Optional[Callable[[str, int], None]]): Função de callback para
                atualizar a interface gráfica. Deve aceitar `(mensagem: str, porcentagem: int)`.
                Para cancelar a operação, este callback deve lançar `InterruptedError`.

        Returns:
            FaceDataset: O objeto dataset processado e enriquecido.

        Raises:
            Exception: Propaga qualquer erro inesperado ocorrido durante o processamento
                       (exceto InterruptedError, que é tratado salvando o estado parcial).
        """
        
        # 1. Setup e Definição de Caminhos
        dataset_name = config.dataset_name
        cache_file = self.storage_path / f"{dataset_name}.parquet"

        # 2. Extract: Ingestão de Dados
        if progress_callback: 
            progress_callback("Lendo imagens e extraindo metadados...", 0)
        
        print(f"🚀 [Pipeline] Iniciando ingestão: {dataset_name}")
        
        # Delega a complexidade de leitura de disco/regex para o Adapter
        dataset = FaceDatasetAdapter.dataset_from_dir(
            dir_path=config.input_path,
            name=dataset_name,
            recursive=config.recursive,
            template=config.template,
            default_identity=config.default_identity,
            origin=config.origin,
            callback=progress_callback
        )
        
        if len(dataset) == 0:
            print("⚠️ [Pipeline] Dataset vazio ou nenhuma face encontrada.")
            return dataset

        # 3. Transform (Parte A): Aplicação de Metadados Manuais
        if config.default_gender:
            print(f" ℹ️ [Pipeline] Aplicando Gênero Fixo: {config.default_gender}")
            dataset._df['gender'] = config.default_gender
            
        if config.default_ethnicity:
             print(f" ℹ️ [Pipeline] Aplicando Etnia Fixa: {config.default_ethnicity}")
             dataset._df['ethnicity'] = config.default_ethnicity

        # 4. Transform (Parte B): Processamento Biométrico e de Qualidade
        active_stages = []
        if config.compute_ff: active_stages.append("Biometria (ForensicFace)")
        if config.compute_ofiq: active_stages.append("Qualidade (OFIQ)")
        if config.compute_serfiq: active_stages.append("Robustez (SER-FIQ)")
        
        total_stages = len(active_stages)
        current_stage_idx = 0

        try:
            # A. ForensicFace (Embeddings, Detecção, Pose, Idade)
            if config.compute_ff:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)

                ff_svc = self._resolve_ff_service(config)
                wanted_models = list(config.ff_models or ["sepaelv2"])

                # Callback adaptativo: escala o % interno do FFService (0-100)
                # para caber no slot atual do pipeline.
                def ff_callback(msg: str, internal_pct: int):
                    if progress_callback:
                        scaled_pct = base_pct + int((internal_pct / 100) * (next_pct - base_pct))
                        progress_callback(msg, scaled_pct)

                ff_svc.process_dataset(
                    dataset,
                    progress_callback=ff_callback,
                    models=wanted_models,
                )
                current_stage_idx += 1

            # B. OFIQ (Qualidade ISO)
            if config.compute_ofiq:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)
                msg = f"Calculando {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, base_pct)

                ofiq_tick = self._make_noarg_stage_callback(
                    msg, base_pct, next_pct, len(dataset), progress_callback
                )
                self.ofiq_service.process_dataset(dataset, callback=ofiq_tick)
                current_stage_idx += 1

            # C. SER-FIQ (Qualidade Estocástica)
            if config.compute_serfiq:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)
                msg = f"Calculando {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, base_pct)

                serfiq_tick = self._make_noarg_stage_callback(
                    msg, base_pct, next_pct, len(dataset), progress_callback
                )
                self._resolve_serfiq_service(config).process_dataset(
                    dataset,
                    callback=serfiq_tick,
                    t=config.serfiq_t,
                    alpha=config.serfiq_alpha,
                )
                current_stage_idx += 1

        except InterruptedError:
            print("\n🛑 [Pipeline] Processamento interrompido pelo usuário.")
            print("   ⚠️ Salvando estado parcial do dataset...")
            # Não relançamos o erro aqui para permitir que o bloco de persistência abaixo seja executado.
            
        except Exception as e:
            self.logger.error(f"Erro inesperado durante a execução do pipeline: {e}", exc_info=True)
            raise e # Erros não previstos devem ser tratados pelo Worker/Controller.

        # 5. Load: Persistência (Sobrescreve se existir)
        if progress_callback: 
            progress_callback("Salvando arquivo Parquet...", 99)
            
        try:
            print(f"💾 [Pipeline] Salvando dataset em: {cache_file}")
            dataset.to_parquet(cache_file)
        except Exception as e:
            msg = f"Erro crítico ao salvar o dataset em {cache_file}: {e}"
            print(f"❌ {msg}")
            self.logger.error(msg)
            raise e
        
        if progress_callback: 
            progress_callback("Processamento Concluído!", 100)

        return dataset

    def _calc_pct(self, current_stage: int, total_stages: int) -> int:
        if total_stages == 0: return 0
        # Reserva os 5% finais para persistência, distribuindo 95% entre as fases
        return int((current_stage / total_stages) * 95)

    # =========================================================================
    # Resolução dinâmica de serviços a partir do DatasetIngestionConfig
    # =========================================================================

    def _resolve_ff_service(self, config: DatasetIngestionConfig) -> FFService:
        """Retorna o FFService Singleton se ele já cobre os params do
        config; caso contrário, instancia um temporário com os params
        pedidos. O temporário paga ~5-10s de carga ONNX uma vez por
        execução — se você executar várias vezes seguidas com os mesmos
        params, o helper continua reutilizando."""
        wanted_models = list(config.ff_models or ["sepaelv2"])
        det_size = int(config.ff_det_size)
        extended = bool(config.ff_extended)

        if self.ff_service.matches_config(
            models=wanted_models, det_size=det_size, extended=extended
        ):
            return self.ff_service

        # Singleton não cobre — constrói FFService temporário. `use_gpu`
        # vem do Singleton (é capability da máquina, não da execução).
        use_gpu = getattr(self.ff_service, "_use_gpu", False)
        print(
            f"⚙️ [Pipeline] FFService Singleton não cobre os params pedidos "
            f"(models={wanted_models}, det_size={det_size}, extended={extended}). "
            f"Instanciando temporário…"
        )
        return FFService(
            models=wanted_models,
            det_size=det_size,
            use_gpu=use_gpu,
            extended=extended,
        )

    def _resolve_serfiq_service(
        self, config: DatasetIngestionConfig
    ) -> SERFIQService:
        """Retorna o SERFIQService Singleton, ajustando T/alpha in-place
        antes do uso. SERFIQService é leve (não carrega modelo extra além
        do FF embedded), então mutar atributos é mais barato que rebuild."""
        # SERFIQService.process_dataset aceita `t=` e `alpha=` por chamada
        # (override por execução), então não precisa rebuild — basta passar
        # adiante na chamada. Retorna o Singleton.
        return self.serfiq_service

    def _make_noarg_stage_callback(
        self,
        stage_msg: str,
        base_pct: int,
        next_pct: int,
        total_items: int,
        progress_callback: Optional[Callable[[str, int], None]],
    ) -> Callable[[], None]:
        """
        Adaptador para serviços que emitem callback sem args (OFIQ, SER-FIQ),
        equivalente ao `ff_callback` adaptativo usado pelo FFService.

        Conta itens internamente, escala o progresso pro slot do estágio
        (`base_pct` → `next_pct`) e chama `progress_callback(msg, pct)`.

        Por que isso existe: o `EnrichWorker._callback` checa o flag de
        cancelamento dentro do progress_callback. Se OFIQ/SER-FIQ rodam
        sem callback, o cancel fica preso até a fase terminar. Esse
        adaptador garante que cada item processado dispare a verificação.
        """
        if total_items <= 0 or progress_callback is None:
            return lambda: None

        state = {"count": 0}
        slot_width = max(0, next_pct - base_pct)

        def _tick() -> None:
            state["count"] += 1
            internal_pct = min(100, int((state["count"] / total_items) * 100))
            scaled = base_pct + int((internal_pct / 100) * slot_width)
            progress_callback(stage_msg, scaled)

        return _tick
    
    def enrich_existing_dataset(self, 
                                dataset: FaceDataset, 
                                config: DatasetIngestionConfig, 
                                progress_callback: Optional[Callable[[str, int], None]] = None
                                ) -> FaceDataset:
        """
        Processa um dataset já existente em memória, aplicando apenas os serviços de IA solicitados.
        Ignora as etapas de varredura de diretório (Extract) e carregamento de cache.
        """
        print(f"🚀 [Pipeline] Iniciando enriquecimento do dataset: {dataset.name}")

        active_stages = []
        if config.compute_ff: active_stages.append("Biometria (ForensicFace)")
        if config.compute_ofiq: active_stages.append("Qualidade (OFIQ)")
        if config.compute_serfiq: active_stages.append("Robustez (SER-FIQ)")
        
        total_stages = len(active_stages)
        current_stage_idx = 0
        
        if total_stages == 0:
            print("⚠️ [Pipeline] Nenhum processamento foi selecionado para enriquecimento.")
            return dataset

        try:
            if config.compute_ff:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)

                ff_svc = self._resolve_ff_service(config)
                wanted_models = list(config.ff_models or ["sepaelv2"])

                def ff_callback(msg: str, internal_pct: int):
                    if progress_callback:
                        scaled_pct = base_pct + int((internal_pct / 100) * (next_pct - base_pct))
                        progress_callback(msg, scaled_pct)

                ff_svc.process_dataset(
                    dataset,
                    progress_callback=ff_callback,
                    models=wanted_models,
                )
                current_stage_idx += 1

            if config.compute_ofiq:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)
                msg = f"Enriquecendo: {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, base_pct)

                ofiq_tick = self._make_noarg_stage_callback(
                    msg, base_pct, next_pct, len(dataset), progress_callback
                )
                self.ofiq_service.process_dataset(dataset, callback=ofiq_tick)
                current_stage_idx += 1

            if config.compute_serfiq:
                base_pct = self._calc_pct(current_stage_idx, total_stages)
                next_pct = self._calc_pct(current_stage_idx + 1, total_stages)
                msg = f"Enriquecendo: {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, base_pct)

                serfiq_tick = self._make_noarg_stage_callback(
                    msg, base_pct, next_pct, len(dataset), progress_callback
                )
                self._resolve_serfiq_service(config).process_dataset(
                    dataset,
                    callback=serfiq_tick,
                    t=config.serfiq_t,
                    alpha=config.serfiq_alpha,
                )
                current_stage_idx += 1

        except InterruptedError:
            print("\n🛑 [Pipeline] Enriquecimento interrompido pelo usuário.")
            
        except Exception as e:
            self.logger.error(f"Erro ao enriquecer dataset: {e}", exc_info=True)
            raise e

        # Persistência
        if dataset.name:
            cache_file = self.storage_path / f"{dataset.name}.parquet"
            if progress_callback: progress_callback("Salvando alterações...", 99)
            try:
                print(f"💾 [Pipeline] Salvando dataset enriquecido em: {cache_file}")
                dataset.to_parquet(cache_file)
            except Exception as e:
                self.logger.error(f"Erro ao salvar pós-enriquecimento: {e}")

        if progress_callback: 
            progress_callback("Enriquecimento Concluído!", 100)

        return dataset