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
                
                # Callback adaptativo: Escala o % interno do FFService (0-100) para caber no slot atual do pipeline
                def ff_callback(msg: str, internal_pct: int):
                    if progress_callback:
                        scaled_pct = base_pct + int((internal_pct / 100) * (next_pct - base_pct))
                        progress_callback(msg, scaled_pct)

                self.ff_service.process_dataset(dataset, progress_callback=ff_callback)
                current_stage_idx += 1

            # B. OFIQ (Qualidade ISO)
            if config.compute_ofiq:
                pct = self._calc_pct(current_stage_idx, total_stages)
                msg = f"Calculando {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, pct)
                
                self.ofiq_service.process_dataset(dataset)
                current_stage_idx += 1

            # C. SER-FIQ (Qualidade Estocástica)
            if config.compute_serfiq:
                pct = self._calc_pct(current_stage_idx, total_stages)
                msg = f"Calculando {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, pct)
                
                self.serfiq_service.process_dataset(dataset)
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
                
                def ff_callback(msg: str, internal_pct: int):
                    if progress_callback:
                        scaled_pct = base_pct + int((internal_pct / 100) * (next_pct - base_pct))
                        progress_callback(msg, scaled_pct)

                self.ff_service.process_dataset(dataset, progress_callback=ff_callback)
                current_stage_idx += 1

            if config.compute_ofiq:
                pct = self._calc_pct(current_stage_idx, total_stages)
                msg = f"Enriquecendo: {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, pct)
                
                self.ofiq_service.process_dataset(dataset)
                current_stage_idx += 1

            if config.compute_serfiq:
                pct = self._calc_pct(current_stage_idx, total_stages)
                msg = f"Enriquecendo: {active_stages[current_stage_idx]}..."
                if progress_callback: progress_callback(msg, pct)
                
                self.serfiq_service.process_dataset(dataset)
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