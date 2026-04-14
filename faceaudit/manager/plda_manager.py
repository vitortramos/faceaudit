import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable, List
from collections import Counter

from faceaudit.services.comparison_service import ComparisonService
from faceaudit.protocols.strategy_manager_protocol import StrategyManagerProtocol

# --- DEPENDENCY INJECTION ---
from dependency_injector.wiring import inject, Provide
from faceaudit.containers import ApplicationContainer 

# ENUMS DE DOMÍNIO E DTOs
from faceaudit.enums.project_file_type import ProjectFileType
from faceaudit.enums.project_folder import ProjectFolder
from faceaudit.models.project_file import ProjectFile

# MODELS & ENUMS
from faceaudit.models.plda_config import PLDAConfig
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.plda_validation_result import PLDAValidationResult, ValidationSeverity
from faceaudit.models.transformation_trajectory_data import TransformationTrajectoryData
from faceaudit.models.pipeline_summary import PipelineSummary
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.plda_validation_input import PLDAValidationInput
from faceaudit.models.split_simulation_result import SplitSimulationResult 
from faceaudit.models.split_config import SplitConfig 
from faceaudit.models.comparison_mining_config import ComparisonMiningConfig

# STRATEGY & ENTITIES
from faceaudit.strategies.plda_strategy import PLDAStrategy
from faceaudit.pipelines.transformation_pipeline import TransformationPipeline

# --- CONTROLLERS DE NEGÓCIO ---
from faceaudit.controllers.plda_transformation_pipelines_controller import PLDATransformationPipelinesController
from faceaudit.controllers.plda_training_controller import PLDATrainingController
from faceaudit.controllers.transformation_trajectory_controller import TransformationTrajectoryController

# SERVICES
from faceaudit.services.config_loader_service import ConfigLoaderService
from faceaudit.services.dataset_loader_service import DatasetLoaderService
from faceaudit.services.dataset_stats_service import DatasetStatsService
from faceaudit.services.plda_validation_service import PLDAValidationService
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.services.hash_service import HashService
from faceaudit.services.file_system_service import FileSystemService
from faceaudit.services.plda_split_service import PLDASplitService 
from faceaudit.services.comparison_mining_service import ComparisonMiningService

class PLDAManager(StrategyManagerProtocol):
    """
    Gerenciador de Sessão para estratégia PLDA.
    Refatorado para a Arquitetura Autocontida (Pendrive Paradigm) com Manifesto Forense.
    """

    @inject
    def __init__(self, 
                 config_loader: ConfigLoaderService = Provide[ApplicationContainer.config_loader],
                 dataset_loader: DatasetLoaderService = Provide[ApplicationContainer.dataset_loader],
                 stats_service: DatasetStatsService = Provide[ApplicationContainer.stats_service],
                 validation_service: PLDAValidationService = Provide[ApplicationContainer.plda_validation_service],
                 math_service: EmbeddingMathService = Provide[ApplicationContainer.math_service],
                 comparison_service: ComparisonService = Provide[ApplicationContainer.comparison_service], 
                 hash_service: HashService = Provide[ApplicationContainer.hash_service],
                 file_system_service: FileSystemService = Provide[ApplicationContainer.file_system_service],
                 split_service: PLDASplitService = Provide[ApplicationContainer.plda_split_service],
                 mining_service: ComparisonMiningService = Provide[ApplicationContainer.mining_service]): 
        
        self.config_loader = config_loader
        self.dataset_loader = dataset_loader
        self.stats_service = stats_service
        self.validation_service = validation_service
        self.math_service = math_service
        self.hash_service = hash_service 
        self.fs_service = file_system_service
        self.comparison_service = comparison_service
        self.split_service = split_service 
        self.mining_service = mining_service

        # --- Estado da Configuração ---
        self._card_config: Optional[PLDAConfig] = None
        
        # --- Observer de Modificação (Dirty Flag Callback) ---
        self._dirty_callback: Optional[Callable[[], None]] = None
        
        # --- Cache de Dados (Volátil) ---
        self._source_dataset: Optional[EmbeddingDataset] = None
        self._target_dataset: Optional[EmbeddingDataset] = None
        self._source_pipe_config: Optional[TransformationPipelineConfig] = None
        self._target_pipe_config: Optional[TransformationPipelineConfig] = None
        self._final_embedding_dim: Optional[int] = None
        self._post_prep_stats: Optional[Dict[str, Any]] = None

        # --- Artefatos de Preparação (Memória) ---
        self._fitted_source_pipeline: Optional[TransformationPipeline] = None
        self._fitted_target_pipeline: Optional[TransformationPipeline] = None
        self._transformed_source_data: Optional[EmbeddingDataset] = None 
        self._transformed_target_data: Optional[EmbeddingDataset] = None 
        
        # --- Ciclo de Vida do Modelo (Persistência) ---
        self._fitted_strategy: Optional[PLDAStrategy] = None
        self._last_model_path: Optional[Path] = None
        self._last_split_simulation: Optional[SplitSimulationResult] = None

    # =========================================================================
    # 0. NOTIFICAÇÕES DE ESTADO (DIRTY FLAG)
    # =========================================================================

    def set_dirty_callback(self, callback: Callable[[], None]) -> None:
        """Injeta a função que será chamada quando o estado do manager mudar."""
        self._dirty_callback = callback

    def _trigger_dirty(self) -> None:
        """Avisa o orquestrador (ProjectManager) que o projeto foi alterado."""
        if self._dirty_callback:
            self._dirty_callback()

    # =========================================================================
    # 1. GESTÃO DE CONFIGURAÇÃO E ESTADO
    # =========================================================================

    def update_config(self, new_config: PLDAConfig, logger_callback: Callable[[str], None] = print) -> None:
        if self._card_config is None:
            self._card_config = new_config
            self._trigger_dirty() # O primeiro carregamento marca o manager como vivo
            return

        old_config = self._card_config
        
        datasets_changed = (
            new_config.source_dataset_name != old_config.source_dataset_name or
            new_config.target_dataset_name != old_config.target_dataset_name or
            new_config.embedding_type != old_config.embedding_type
        )
        
        pipelines_changed = (
            new_config.source_pipeline_name != old_config.source_pipeline_name or
            new_config.target_pipeline_name != old_config.target_pipeline_name or
            new_config.parameters.use_pca != old_config.parameters.use_pca or
            new_config.parameters.pca_dim != old_config.parameters.pca_dim
        )
        
        params_changed = (
            new_config.parameters != old_config.parameters or
            new_config.split_config_name != old_config.split_config_name or
            new_config.mining_config_name != old_config.mining_config_name
        )
        
        if datasets_changed:
            self.clear_cache(logger_callback)
        elif pipelines_changed or params_changed:
            self._clear_artifacts(logger_callback)

        # Se literalmente qualquer coisa mudou, avisamos que o projeto está "sujo"
        if datasets_changed or pipelines_changed or params_changed:
            self._trigger_dirty()

        # Preserva a lista de rastreamento forense ao atualizar hiperparâmetros na UI
        new_config.project_files = old_config.project_files.copy()
        self._card_config = new_config

    def get_config(self) -> Optional[PLDAConfig]:
        return self._card_config
    
    def get_training_datasets(self) -> List[str]:
        """Retorna os datasets usados no treino do PLDA para auditoria de Leakage."""
        if not self._card_config:
            return []
        
        datasets = []
        if self._card_config.source_dataset_name and self._card_config.source_dataset_name != "Selecione...":
            datasets.append(self._card_config.source_dataset_name)
            
        if self._card_config.target_dataset_name and self._card_config.target_dataset_name != "Selecione...":
            datasets.append(self._card_config.target_dataset_name)
            
        return datasets

    # =========================================================================
    # 2. VALIDAÇÃO (Check de Sanidade)
    # =========================================================================

    def validate_all(self) -> PLDAValidationResult:
        if not self._card_config:
            return PLDAValidationResult(ValidationSeverity.NEUTRAL)
        inp = self._build_validation_input()
        return self.validation_service.validate(inp)

    def _build_validation_input(self) -> PLDAValidationInput:
        if not self._card_config: 
            raise ValueError("Config não inicializada.")

        def _get_stats_dynamically(dataset_name: str):
            if not dataset_name or dataset_name == "Selecione...": 
                return None
            path = self.fs_service.resolve_file(dataset_name, ProjectFileType.DATASET_METADATA)
            if path:
                return self.stats_service.get_stats(dataset_name, target_directory=path.parent)
            return None

        # 1. Carrega estatísticas básicas de domínio
        st_src = _get_stats_dynamically(self._card_config.source_dataset_name)
        st_tgt = _get_stats_dynamically(self._card_config.target_dataset_name)
        self.load_pipeline_configs()

        # 2. Resolve estatísticas de preparação (Pipelines)
        stats_to_use = self._post_prep_stats
        dim_to_use = self._final_embedding_dim
        has_stats_cache = False

        if not self.is_prepared() and self.fs_service.current_project_root:
            pf_stats = next((pf for pf in self._card_config.project_files 
                             if pf.file_type == ProjectFileType.PREPARATION_STATS), None)
            if pf_stats:
                abs_path = self.fs_service.current_project_root / pf_stats.path
                if abs_path.exists():
                    try:
                        with open(abs_path, "r", encoding="utf-8") as f:
                            stats_to_use = json.load(f)
                        dim_to_use = stats_to_use.get("final_embedding_dim")
                        has_stats_cache = True
                    except Exception as e:
                        print(f"⚠️ Aviso: Cache de estatísticas de pipeline corrompido: {e}")

        # 3. Resolve a Simulação de Split (Usa o método centralizado para Lazy Load)
        # Isso garante que se o arquivo existir no HD, ele será carregado agora.
        sim_result = self.get_split_simulation()

        return PLDAValidationInput(
            config=self._card_config, 
            train_stats=st_src, 
            calib_stats=st_tgt,
            source_pipeline_config=self._source_pipe_config,
            target_pipeline_config=self._target_pipe_config,
            final_embedding_dim=dim_to_use,
            is_prepared=self.is_prepared() or has_stats_cache,
            post_prep_stats=stats_to_use,
            split_simulation=sim_result # <--- Injeta o DTO carregado (RAM ou HD)
        )

    # =========================================================================
    # 3. I/O DE DADOS E PREPARAÇÃO
    # =========================================================================

    def get_available_datasets(self) -> Tuple[List[str], List[str]]:
        return self.fs_service.list_available_files(ProjectFileType.DATASET_METADATA)

    def get_available_pipelines(self) -> Tuple[List[str], List[str]]:
        return self.fs_service.list_available_files(ProjectFileType.PIPELINE_CONFIG)

    def get_available_splits(self) -> Tuple[List[str], List[str]]:
        return self.fs_service.list_available_files(ProjectFileType.SPLIT_CONFIG)

    def get_available_mining_configs(self) -> Tuple[List[str], List[str]]:
        return self.fs_service.list_available_files(ProjectFileType.MINING_CONFIG)
    
    def get_dataset_stats(self, dataset_name: str):
        if not dataset_name or dataset_name == "Selecione...": 
            return None
            
        clean_name = dataset_name.split(" (")[0]
        path = self.fs_service.resolve_file(clean_name, ProjectFileType.DATASET_METADATA)
        
        if path:
            return self.stats_service.get_stats(clean_name, target_directory=path.parent)
            
        return None

    def load_pipeline_configs(self) -> None:
        if not self._card_config: return

        def _load_pipe(name):
            if not name or name == "Selecione...": return None
            path = self.fs_service.resolve_file(name, ProjectFileType.PIPELINE_CONFIG)
            if not path: return None
            return TransformationPipelineConfig.from_json(path)

        self._source_pipe_config = _load_pipe(self._card_config.source_pipeline_name)
        self._target_pipe_config = _load_pipe(self._card_config.target_pipeline_name)
        
        if self._card_config.parameters.use_pca:
            dim = self._card_config.parameters.pca_dim
            if self._source_pipe_config: self._source_pipe_config.set_global_output_dimension(dim)
            if self._target_pipe_config: self._target_pipe_config.set_global_output_dimension(dim)

    def run_preparation(self, logger_callback: Callable[[str], None] = print) -> Dict[str, Any]:
        if not self._card_config: raise ValueError("Configuração não definida.")
        
        logger_callback("🔒 Auditando integridade dos dados de entrada...")
        self._lock_local_inputs(logger_callback)

        logger_callback("📥 Carregando datasets do disco para a memória...")
        if not self._source_dataset: self.load_input_datasets()
        self.load_pipeline_configs()

        if not self.fs_service.current_project_root:
            raise RuntimeError("Projeto não inicializado no sistema de arquivos.")

        proj_root = self.fs_service.current_project_root
        strategy_id = self._card_config.id
        prefix_1, prefix_2 = self._card_config.strategy.artifact_prefixes

        def get_clean_name(full_name: str) -> str:
            return full_name.split(" (")[0] if full_name and full_name != "Selecione..." else "unknown"

        src_pipe_name = get_clean_name(self._card_config.source_pipeline_name)
        tgt_pipe_name = get_clean_name(self._card_config.target_pipeline_name)

        rel_src_path = ProjectFileType.FITTED_PIPELINE.build_path(strategy_id=strategy_id, pipeline_name=f"{prefix_1}_{src_pipe_name}")
        rel_tgt_path = ProjectFileType.FITTED_PIPELINE.build_path(strategy_id=strategy_id, pipeline_name=f"{prefix_2}_{tgt_pipe_name}")

        path_src_fitted = proj_root / rel_src_path
        path_tgt_fitted = proj_root / rel_tgt_path
        path_src_fitted.parent.mkdir(parents=True, exist_ok=True)

        def _is_valid_cache(path: Path) -> bool:
            if not path.exists(): return False
            rel_path = path.relative_to(proj_root).as_posix()
            return any(pf.path == rel_path for pf in self._card_config.project_files)

        src_cached = _is_valid_cache(path_src_fitted)
        tgt_cached = True if not self._target_dataset else _is_valid_cache(path_tgt_fitted)

        if src_cached and tgt_cached:
            logger_callback("♻️ Pipelines matemáticos já treinados encontrados no disco (Manifesto).")
            logger_callback(" ↳ Pulando etapa de treinamento (Fit) para economizar tempo de CPU...")
        else:
            logger_callback(f"⚙️ Treinando pipelines de transformação ({prefix_1} e {prefix_2})...")
            controller = PLDATransformationPipelinesController()
            controller.train_and_persist(
                source_dataset=self._source_dataset, target_dataset=self._target_dataset,
                source_config=self._source_pipe_config, target_config=self._target_pipe_config,
                strategy=self._card_config.strategy, source_save_path=path_src_fitted,
                target_save_path=path_tgt_fitted, logger_callback=logger_callback
            )

            logger_callback("🔐 Assinando matrizes matemáticas (Cadeia de Custódia)...")
            self.add_project_file(ProjectFileType.FITTED_PIPELINE, path_src_fitted, logger_callback)
            if self._target_dataset:
                self.add_project_file(ProjectFileType.FITTED_PIPELINE, path_tgt_fitted, logger_callback)

        logger_callback("🔄 Carregando matrizes e aplicando transformações geométricas nos dados...")
        self._fitted_source_pipeline = TransformationPipeline.load(path_src_fitted)
        self._fitted_target_pipeline = TransformationPipeline.load(path_tgt_fitted)

        self._transformed_source_data = self._fitted_source_pipeline.transform(self._source_dataset)
        if self._target_dataset:
            self._transformed_target_data = self._fitted_target_pipeline.transform(self._target_dataset)

        self._final_embedding_dim = int(self._transformed_source_data.matrix.shape[1])
        
        logger_callback("📊 Calculando distribuição de classes pós-preparação...")
        self._compute_post_prep_split_stats(logger_callback)
        
        logger_callback("✅ Preparação concluída. Dados alocados na RAM.")
        return self._post_prep_stats

    def is_prepared(self) -> bool:
        return self._fitted_source_pipeline is not None and self._transformed_source_data is not None

    # =========================================================================
    # 4. CICLO DE VIDA DO MODELO (Persistência e Lazy Load)
    # =========================================================================

    def run_training(self, logger_callback: Callable[[str], None] = print) -> str:
        if not self.is_prepared(): 
            logger_callback("⚠️ Preparação não encontrada. Executando preparação forçada...")
            self.run_preparation(logger_callback=logger_callback)

        if not self.fs_service.current_project_root:
            raise RuntimeError("Raiz de projeto não inicializada.")

        if not self._card_config.split_config_name or self._card_config.split_config_name == "Selecione...":
            raise ValueError("Regra de Split não configurada. Selecione uma na interface.")
        if not self._card_config.mining_config_name or self._card_config.mining_config_name == "Selecione...":
            raise ValueError("Regra de Mineração não configurada. Selecione uma na interface.")

        clean_split = self._card_config.split_config_name.split(" (")[0]
        clean_mining = self._card_config.mining_config_name.split(" (")[0]

        path_split = self.fs_service.resolve_file(clean_split, ProjectFileType.SPLIT_CONFIG)
        path_mining = self.fs_service.resolve_file(clean_mining, ProjectFileType.MINING_CONFIG)

        if not path_split or not path_split.exists():
            raise FileNotFoundError(f"Arquivo de Split '{clean_split}' não encontrado.")
        if not path_mining or not path_mining.exists():
            raise FileNotFoundError(f"Arquivo de Mineração '{clean_mining}' não encontrado.")

        split_config = SplitConfig.from_json(path_split)
        mining_config = ComparisonMiningConfig.from_json(path_mining)
            
        proj_root = self.fs_service.current_project_root
        
        rel_model_path = ProjectFileType.TRAINED_STRATEGY.build_path(strategy_id=self._card_config.id)
        abs_model_path = proj_root / rel_model_path
        abs_model_path.parent.mkdir(parents=True, exist_ok=True)

        logger_callback("🧠 Iniciando treinamento do modelo PLDA...")
        controller = PLDATrainingController(
            config=self._card_config, 
            split_service=self.split_service,
            mining_service=self.mining_service, # <--- ADICIONE ESTA LINHA
            logger=logger_callback
        )

        model_path_str = controller.run(
            source_dataset=self._transformed_source_data,
            target_dataset=self._transformed_target_data,
            split_config=split_config,
            mining_config=mining_config,
            output_path=abs_model_path
        )
        
        logger_callback(f"💾 Modelo salvo com sucesso em: {abs_model_path.parent.name}/{abs_model_path.name}")
        self._last_model_path = Path(model_path_str)
        
        logger_callback("📥 Carregando modelo final na memória para inferência...")
        self.load_model(self._last_model_path) 
        
        logger_callback("🔐 Assinando binário do modelo treinado...")
        self.add_project_file(ProjectFileType.TRAINED_STRATEGY, self._last_model_path, logger_callback)
        
        logger_callback("✅ Treinamento concluído!")
        return model_path_str

    def get_model_path(self) -> Optional[Path]:
        return self._last_model_path

    def has_model_on_disk(self) -> bool:
        return self._last_model_path is not None and self._last_model_path.exists()

    def is_model_loaded(self) -> bool:
        return self._fitted_strategy is not None

    def load_model(self, path: Optional[Path] = None) -> bool:
        target_path = path or self._last_model_path
        if not target_path or not target_path.exists():
            return False

        try:
            strategy = PLDAStrategy(name=target_path.stem)
            strategy.load(str(target_path))
            self._fitted_strategy = strategy
            self._last_model_path = target_path
            return True
        except Exception as e:
            import traceback
            print(f"\n❌ ERRO FATAL AO CARREGAR PLDA: {str(e)}")
            traceback.print_exc()
            return False
    
    def load_input_datasets(self) -> None:
        if not self._card_config: return

        def _load_one(name: str) -> Optional[EmbeddingDataset]:
            if not name or name == "Selecione...": return None
            
            clean_name = name.split(" (")[0]
            path = self.fs_service.resolve_file(clean_name, ProjectFileType.DATASET_METADATA)
            
            if not path: return None
            
            raw_data = self.dataset_loader.load(path)
            if isinstance(raw_data, FaceDataset):
                return raw_data.to_embedding_dataset(self._card_config.embedding_type)
            return raw_data

        self._source_dataset = _load_one(self._card_config.source_dataset_name)
        self._target_dataset = _load_one(self._card_config.target_dataset_name)

    # =========================================================================
    # 5. GERENCIAMENTO DE ARQUIVOS (Manifesto Forense)
    # =========================================================================

    def add_project_file(self, file_type: ProjectFileType, abs_path: Path, logger_callback: Callable = print) -> None:
        if not self._card_config or not self.fs_service.current_project_root:
            return

        if not abs_path.exists():
            logger_callback(f"⚠️ Aviso: Arquivo não encontrado para rastreio: {abs_path.name}")
            return

        try:
            file_hash = self.hash_service.calculate_file_hash(abs_path)
            rel_path = abs_path.relative_to(self.fs_service.current_project_root).as_posix()
            
            existing_file = next((pf for pf in self._card_config.project_files if pf.path == rel_path), None)
            
            if existing_file:
                existing_file.hash = file_hash
                existing_file.file_type = file_type
            else:
                new_pf = ProjectFile(file_type=file_type, path=rel_path, hash=file_hash)
                self._card_config.project_files.append(new_pf)
                
            logger_callback(f"  ↳ Manifesto: [{file_type.name}] {Path(rel_path).name} assinado.")
            
            self._trigger_dirty()
            
        except Exception as e:
            logger_callback(f"⚠️ Erro ao registrar rastreio de {abs_path.name}: {e}")

    def delete_project_file(self, project_file: ProjectFile, logger_callback: Callable = print) -> None:
        if not self._card_config or not self.fs_service.current_project_root:
            return

        if project_file in self._card_config.project_files:
            self._card_config.project_files.remove(project_file)
            self._trigger_dirty() 
            
        abs_path = self.fs_service.current_project_root / project_file.path
        if abs_path.exists():
            try:
                abs_path.unlink()
                logger_callback(f"🗑️ Arquivo apagado do disco: {abs_path.name}")
            except Exception as e:
                logger_callback(f"⚠️ Não foi possível apagar o arquivo {abs_path.name}: {e}")

    def _cleanup_local_files(self, logger_callback: Callable[[str], None] = print) -> None:
        if not self._card_config or not self.fs_service.current_project_root:
            return
            
        proj_root = self.fs_service.current_project_root
        strategy_id = self._card_config.id
        
        valid_paths = {pf.path for pf in self._card_config.project_files}
        folders_to_check = [ProjectFolder.ARTIFACTS, ProjectFolder.STRATEGIES, ProjectFolder.GRAPHICS]
        
        has_deleted = False
        for folder in folders_to_check:
            folder_path = proj_root / folder.value
            if not folder_path.exists(): continue
                
            for file_path in folder_path.glob(f"{strategy_id}_*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(proj_root).as_posix()
                    if rel_path not in valid_paths:
                        try:
                            file_path.unlink()
                            has_deleted = True
                            logger_callback(f"🧹 Faxina Local: Lixo órfão removido -> {file_path.name}")
                        except Exception as e:
                            logger_callback(f"⚠️ Aviso: Não foi possível apagar {file_path.name}: {e}")
                            
        if has_deleted:
            self._trigger_dirty()

    # =========================================================================
    # 6. LIMPEZA E CACHE (RAM vs DISCO)
    # =========================================================================

    def clear_cache(self, logger_callback: Callable[[str], None] = print) -> None:
        self._source_dataset = None
        self._target_dataset = None
        self._clear_artifacts(logger_callback)

    def _clear_artifacts(self, logger_callback: Callable[[str], None] = print) -> None:
        self._fitted_source_pipeline = None
        self._fitted_target_pipeline = None
        self._transformed_source_data = None
        self._transformed_target_data = None
        self._final_embedding_dim = None
        self._post_prep_stats = None
        self._fitted_strategy = None
        self._last_split_simulation = None

    def clear_state(self, wipe_disk: bool = False, logger_callback: Callable[[str], None] = print) -> None:
        self.clear_cache(logger_callback)
        self._last_model_path = None
        
        if wipe_disk and self._card_config:
            logger_callback(f"🗑️ Excluindo rastros da estratégia {self._card_config.id} no disco...")
            
            dynamic_types = {ProjectFileType.FITTED_PIPELINE, ProjectFileType.TRAINED_STRATEGY, 
                             ProjectFileType.TRAJECTORY_DATA, ProjectFileType.TRAJECTORY_PLOT,
                             ProjectFileType.PREPARATION_STATS, ProjectFileType.SPLIT_SIMULATION_STATS}
            
            for pf in list(self._card_config.project_files):
                if pf.file_type in dynamic_types:
                    self.delete_project_file(pf, logger_callback)
            
            self._cleanup_local_files(logger_callback)

        self._card_config = None

    def invalidate_math_artifacts(self, logger_callback: Callable[[str], None] = print) -> None:
        self.clear_cache(logger_callback)
        
        if not self._card_config or not self.fs_service.current_project_root:
            return

        logger_callback("🧹 Invalidando cache de artefatos obsoletos no disco...")
        
        dynamic_types = {
            ProjectFileType.FITTED_PIPELINE, 
            ProjectFileType.TRAINED_STRATEGY, 
            ProjectFileType.TRAJECTORY_DATA, 
            ProjectFileType.TRAJECTORY_PLOT,
            ProjectFileType.PREPARATION_STATS,
            ProjectFileType.SPLIT_SIMULATION_STATS
        }
        
        for pf in list(self._card_config.project_files):
            if pf.file_type in dynamic_types:
                self.delete_project_file(pf, logger_callback)
        
        self._cleanup_local_files(logger_callback)

    # =========================================================================
    # 7. GESTÃO FORENSE DE ATIVOS (DTOs, Cópia e Invalidação)
    # =========================================================================

    def get_pipeline_config(self, pipeline_name: str) -> Optional[TransformationPipelineConfig]:
        if not pipeline_name or pipeline_name == "Selecione...": 
            return None
            
        clean_name = pipeline_name.split(" (")[0]
        path = self.fs_service.resolve_file(clean_name, ProjectFileType.PIPELINE_CONFIG)
        
        if path and path.exists():
            cfg = TransformationPipelineConfig.from_json(path)
            cfg.name = clean_name 
            return cfg
            
        return None

    def save_or_update_pipeline(self, config: TransformationPipelineConfig, logger_callback: Callable = print) -> None:
        if not self.fs_service.current_project_root:
            raise RuntimeError("Nenhum projeto ativo para salvar o pipeline.")

        config.name = "".join(x for x in config.name if x.isalnum() or x in ('_', '-'))

        rel_path = ProjectFileType.PIPELINE_CONFIG.build_path(pipeline_name=config.name)
        file_path = self.fs_service.current_project_root / rel_path
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config.to_json(file_path)
        logger_callback(f"💾 Pipeline '{config.name}' salvo/atualizado no projeto local.")

        self.add_project_file(ProjectFileType.PIPELINE_CONFIG, file_path, logger_callback)
        self.invalidate_math_artifacts(logger_callback)
    
    def _lock_local_inputs(self, logger_callback: Callable = print) -> None:
        if not self._card_config or not self.fs_service.current_project_root:
            return
            
        inputs = [
            (self._card_config.source_dataset_name, ProjectFileType.DATASET_METADATA),
            (self._card_config.target_dataset_name, ProjectFileType.DATASET_METADATA),
            (self._card_config.source_pipeline_name, ProjectFileType.PIPELINE_CONFIG),
            (self._card_config.target_pipeline_name, ProjectFileType.PIPELINE_CONFIG),
            (self._card_config.mining_config_name, ProjectFileType.MINING_CONFIG), 
            (self._card_config.split_config_name, ProjectFileType.SPLIT_CONFIG) 
        ]
        
        for name, f_type in inputs:
            if not name or name == "Selecione...": continue
            clean_name = name.split(" (")[0]
            
            path = self.fs_service.resolve_file(clean_name, f_type)
            if path and path.exists() and self.fs_service.current_project_root in path.parents:
                self.add_project_file(f_type, path, lambda x: None)
    
    def get_split_simulation(self) -> Optional[SplitSimulationResult]:
        """Lazy load seguro da simulação para a interface e validação."""
        if self._last_split_simulation is None and self._card_config and self.fs_service.current_project_root:
            pf_sim = next((pf for pf in self._card_config.project_files 
                           if pf.file_type == ProjectFileType.SPLIT_SIMULATION_STATS), None)
            if pf_sim:
                abs_path = self.fs_service.current_project_root / pf_sim.path
                if abs_path.exists():
                    try:
                        self._last_split_simulation = SplitSimulationResult.from_json(abs_path)
                    except Exception as e:
                        print(f"⚠️ Aviso: Cache de simulação corrompido: {e}")
        return self._last_split_simulation

    def get_split_config(self, split_name: str) -> Optional[SplitConfig]:
        if not split_name or split_name == "Selecione...": return None
        clean_name = split_name.split(" (")[0]
        path = self.fs_service.resolve_file(clean_name, ProjectFileType.SPLIT_CONFIG)
        if path and path.exists():
            cfg = SplitConfig.from_json(path)
            cfg.name = clean_name
            return cfg
        return None

    def save_or_update_split(self, config: SplitConfig, logger_callback: Callable = print) -> None:
        if not self.fs_service.current_project_root: raise RuntimeError("Nenhum projeto ativo para salvar.")
        config.name = "".join(x for x in config.name if x.isalnum() or x in ('_', '-'))
        rel_path = ProjectFileType.SPLIT_CONFIG.build_path(config_name=config.name)
        file_path = self.fs_service.current_project_root / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_json(file_path)
        
        logger_callback(f"💾 Split '{config.name}' salvo no projeto local.")
        self.add_project_file(ProjectFileType.SPLIT_CONFIG, file_path, logger_callback)
        
        # A regra matemática mudou, a simulação no cache não vale mais
        self._last_split_simulation = None 
        self._trigger_dirty()

    def get_mining_config(self, mining_name: str) -> Optional[ComparisonMiningConfig]:
        if not mining_name or mining_name == "Selecione...": return None
        clean_name = mining_name.split(" (")[0]
        path = self.fs_service.resolve_file(clean_name, ProjectFileType.MINING_CONFIG)
        if path and path.exists():
            cfg = ComparisonMiningConfig.from_json(path)
            cfg.name = clean_name
            return cfg
        return None

    def save_or_update_mining(self, config: ComparisonMiningConfig, logger_callback: Callable = print) -> None:
        if not self.fs_service.current_project_root: raise RuntimeError("Nenhum projeto ativo para salvar.")
        config.name = "".join(x for x in config.name if x.isalnum() or x in ('_', '-'))
        rel_path = ProjectFileType.MINING_CONFIG.build_path(config_name=config.name)
        file_path = self.fs_service.current_project_root / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_json(file_path)
        
        logger_callback(f"💾 Mineração '{config.name}' salva no projeto local.")
        self.add_project_file(ProjectFileType.MINING_CONFIG, file_path, logger_callback)
        
        self._last_split_simulation = None 
        self._trigger_dirty()

    # =========================================================================
    # 8. EXECUÇÃO E ANÁLISE (COLUNA 3) E SIMULAÇÕES 
    # =========================================================================
    
    def simulate_data_split(self, logger_callback: Callable[[str], None] = print) -> SplitSimulationResult:
        if not self._card_config:
            return SplitSimulationResult(success=False, error_message="Aba não configurada.")
            
        split_name = self._card_config.split_config_name
        if not split_name or split_name == "Selecione...":
            return SplitSimulationResult(success=False, error_message="Nenhuma regra de Split selecionada.")

        clean_split_name = split_name.split(" (")[0]
        split_path = self.fs_service.resolve_file(clean_split_name, ProjectFileType.SPLIT_CONFIG)
        
        if not split_path or not split_path.exists():
            return SplitSimulationResult(success=False, error_message=f"Arquivo '{clean_split_name}' não encontrado no disco.")

        try:
            split_config = SplitConfig.from_json(split_path)
            
            if not self._source_dataset:
                logger_callback("📥 Lendo dataset base para simular a matemática de divisão...")
                self.load_input_datasets()

            if not self._source_dataset:
                return SplitSimulationResult(success=False, error_message="Falha: Dataset de Origem ausente.")

            data_to_split = self._transformed_source_data if self._transformed_source_data else self._source_dataset
            target_data_to_split = self._transformed_target_data if self._transformed_target_data else self._target_dataset
            
            logger_callback(f"✂️ Simulando corte (Arca de Noé) usando regra: '{split_config.name}'...")
            
            plda_set, calib_set = self.split_service.perform_split(
                source_dataset=data_to_split,
                target_dataset=target_data_to_split,
                strategy_type=self._card_config.strategy,
                config=split_config,
                logger_callback=logger_callback
            )
            
            def calc_pairs(dataset: EmbeddingDataset) -> Tuple[int, int]:
                if not dataset or not dataset.items: return 0, 0
                counts = Counter([item.identity for item in dataset.items if item.identity])
                
                gen_pairs = sum((n * (n - 1)) // 2 for n in counts.values() if n > 1)
                
                total_samples = sum(counts.values())
                total_pairs = (total_samples * (total_samples - 1)) // 2
                imp_pairs = total_pairs - gen_pairs
                
                return gen_pairs, imp_pairs

            plda_gen, plda_imp = calc_pairs(plda_set)
            calib_gen, calib_imp = calc_pairs(calib_set)

            plda_ids = len(set(i.identity for i in plda_set.items if i.identity))
            calib_ids = len(set(i.identity for i in calib_set.items if i.identity))

            logger_callback("📊 Matemática de calibração concluída!")

            self._last_split_simulation = SplitSimulationResult(
                plda_identities_count=plda_ids,
                plda_images_count=len(plda_set.items),
                plda_genuine_pairs=plda_gen,
                plda_impostor_pairs=plda_imp,
                calib_identities_count=calib_ids,
                calib_images_count=len(calib_set.items),
                calib_genuine_pairs=calib_gen,
                calib_impostor_pairs=calib_imp,
                success=True
            )

            # SALVA NO DISCO (LAZY LOAD)
            if self.fs_service.current_project_root and self._card_config:
                rel_path = ProjectFileType.SPLIT_SIMULATION_STATS.build_path(strategy_id=self._card_config.id)
                abs_path = self.fs_service.current_project_root / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                
                logger_callback("💾 Persistindo métricas da simulação no projeto...")
                self._last_split_simulation.to_json(abs_path)
                self.add_project_file(ProjectFileType.SPLIT_SIMULATION_STATS, abs_path, logger_callback)

            return self._last_split_simulation
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return SplitSimulationResult(success=False, error_message=f"Erro interno ao simular: {str(e)}")

    def ensure_model_loaded(self) -> None:
        if self.is_model_loaded(): return
        
        if self.has_model_on_disk():
            success = self.load_model(self._last_model_path)
            if not success:
                raise RuntimeError(f"Falha ao carregar modelo em: {self._last_model_path}")
        else:
            raise FileNotFoundError("Nenhum modelo treinado disponível para carregar.")

    def set_model_path(self, path: Path) -> None:
        self._last_model_path = path

    def unload_model(self) -> None:
        self._fitted_strategy = None

    def run_inference(self, logger_callback: Any = print) -> Any:
        self.ensure_model_loaded()
        pass

    def run_validation_inference(self, 
                                 probe_dataset: FaceDataset, 
                                 reference_dataset: FaceDataset, 
                                 logger_callback: Callable[[str], None] = print) -> Tuple[List[float], List[float]]:
        self.ensure_model_loaded()

        emb_type = self._card_config.embedding_type
        logger_callback(f"     [PLDA] Convertendo faces para embeddings {emb_type.name}...")
        
        probe_emb = probe_dataset.to_embedding_dataset(emb_type)
        ref_emb = reference_dataset.to_embedding_dataset(emb_type)

        proj_root = self.fs_service.current_project_root
        strategy_id = self._card_config.id
        prefix_1, prefix_2 = self._card_config.strategy.artifact_prefixes

        def get_clean_name(name): return name.split(" (")[0] if name and name != "Selecione..." else "unknown"
        src_pipe_name = get_clean_name(self._card_config.source_pipeline_name)
        tgt_pipe_name = get_clean_name(self._card_config.target_pipeline_name)

        path_src_fitted = proj_root / ProjectFileType.FITTED_PIPELINE.build_path(strategy_id=strategy_id, pipeline_name=f"{prefix_1}_{src_pipe_name}")
        path_tgt_fitted = proj_root / ProjectFileType.FITTED_PIPELINE.build_path(strategy_id=strategy_id, pipeline_name=f"{prefix_2}_{tgt_pipe_name}")

        if not self._fitted_source_pipeline:
            self._fitted_source_pipeline = TransformationPipeline.load(path_src_fitted)
        if not self._fitted_target_pipeline:
            self._fitted_target_pipeline = TransformationPipeline.load(path_tgt_fitted)

        from faceaudit.enums.plda_training_strategy import PLDATrainingStrategy
        if self._card_config.strategy == PLDATrainingStrategy.SUPERVISED_ADAPTATION:
            val_probe_ready = self._fitted_target_pipeline.transform(probe_emb)
            val_ref_ready = self._fitted_source_pipeline.transform(ref_emb)
        else:
            val_probe_ready = self._fitted_target_pipeline.transform(probe_emb)
            val_ref_ready = self._fitted_target_pipeline.transform(ref_emb)

        logger_callback(f"     [PLDA] Calculando matriz densa de scores ({len(val_probe_ready)} x {len(val_ref_ready)})...")
        
        lr_matrix = self._fitted_strategy.predict_full_matrix(val_probe_ready.matrix, val_ref_ready.matrix)
        
        import numpy as np
        lr_matrix_clipped = np.clip(lr_matrix, 1e-15, None)
        log_lr_matrix = np.log(lr_matrix_clipped)

        ids_p = np.array([getattr(i, "identity", None) for i in val_probe_ready.items])
        ids_r = np.array([getattr(i, "identity", None) for i in val_ref_ready.items])
        
        logger_callback("     [PLDA] Separando Genuínos e Impostores (Broadcasting)...")

        is_auto_comparison = (probe_dataset.name == reference_dataset.name)

        if is_auto_comparison:
            n = len(ids_p)
            i_idx, j_idx = np.triu_indices(n, k=1)
            
            valid_mask = (ids_p[i_idx] != None) & (ids_r[j_idx] != None)
            i_valid, j_valid = i_idx[valid_mask], j_idx[valid_mask]
            
            is_same = (ids_p[i_valid] == ids_r[j_valid])
            scores = log_lr_matrix[i_valid, j_valid]
            
            genuine_lrs = scores[is_same].tolist()
            impostor_lrs = scores[~is_same].tolist()
            
        else:
            ids_p_grid = ids_p[:, None] 
            ids_r_grid = ids_r[None, :] 
            
            valid_mask = (ids_p_grid != None) & (ids_r_grid != None)
            is_same = (ids_p_grid == ids_r_grid) & valid_mask
            is_diff = (ids_p_grid != ids_r_grid) & valid_mask
            
            genuine_lrs = log_lr_matrix[is_same].tolist()
            impostor_lrs = log_lr_matrix[is_diff].tolist()
            
        logger_callback(f"     [PLDA] Concluído! Extraídos {len(genuine_lrs)} Genuínos e {len(impostor_lrs)} Impostores.")

        return genuine_lrs, impostor_lrs

    # =========================================================================
    # 9. MÉTODOS DE APOIO E TRAJETÓRIA
    # =========================================================================

    def get_ready_datasets(self) -> Tuple[EmbeddingDataset, Optional[EmbeddingDataset]]:
        if not self.is_data_loaded(): self.load_input_datasets()
        return self._source_dataset, self._target_dataset

    def get_trained_artifacts(self) -> Dict[str, Any]:
        if not self.is_prepared(): raise ValueError("Execute 'Preparar' primeiro.")
        return {
            'source_dataset_transformed': self._transformed_source_data,
            'target_dataset_transformed': self._transformed_target_data,
            'source_pipeline_fitted': self._fitted_source_pipeline,
            'target_pipeline_fitted': self._fitted_target_pipeline
        }

    def is_data_loaded(self) -> bool:
        return self._source_dataset is not None

    def get_pipeline_details(self, pipeline_name: str) -> PipelineSummary:
        if not pipeline_name or pipeline_name == "Selecione...": return PipelineSummary.empty()
        try:
            path = self.fs_service.resolve_file(pipeline_name, ProjectFileType.PIPELINE_CONFIG)
            if not path: return PipelineSummary.error("Arquivo não encontrado no projeto nem no diretório global.")
            
            cfg = TransformationPipelineConfig.from_json(path)
            return cfg.to_summary()
        except Exception as e:
            return PipelineSummary.error(str(e))
    
    def has_trajectory_data(self) -> bool:
        if not self._card_config or not self.fs_service.current_project_root:
            return False
        pf = next((f for f in self._card_config.project_files if f.file_type == ProjectFileType.TRAJECTORY_DATA), None)
        return pf is not None and (self.fs_service.current_project_root / pf.path).exists()

    def has_fitted_pipelines(self) -> bool:
        if not self._card_config or not self.fs_service.current_project_root:
            return False

        proj_root = self.fs_service.current_project_root
        strategy_id = self._card_config.id
        prefix_1, prefix_2 = self._card_config.strategy.artifact_prefixes

        def get_clean_name(full_name: str) -> str:
            return full_name.split(" (")[0] if full_name and full_name != "Selecione..." else "unknown"

        src_pipe_name = get_clean_name(self._card_config.source_pipeline_name)
        tgt_pipe_name = get_clean_name(self._card_config.target_pipeline_name)

        if src_pipe_name == "unknown": return False

        rel_src_path = ProjectFileType.FITTED_PIPELINE.build_path(strategy_id=strategy_id, pipeline_name=f"{prefix_1}_{src_pipe_name}").as_posix()
        rel_tgt_path = ProjectFileType.FITTED_PIPELINE.build_path(strategy_id=strategy_id, pipeline_name=f"{prefix_2}_{tgt_pipe_name}").as_posix()

        def _is_valid_cache(rel_path_str: str) -> bool:
            has_manifest = any(pf.path == rel_path_str for pf in self._card_config.project_files)
            exists_physically = (proj_root / rel_path_str).exists()
            return has_manifest and exists_physically

        src_ok = _is_valid_cache(rel_src_path)
        tgt_ok = True
        if self._card_config.target_dataset_name and self._card_config.target_dataset_name != "Selecione...":
            if tgt_pipe_name == "unknown": return False
            tgt_ok = _is_valid_cache(rel_tgt_path)

        return src_ok and tgt_ok

    def get_visualization_data(self, logger_callback: Any = print) -> TransformationTrajectoryData:
        viz_data = None
        
        if self._card_config and self.fs_service.current_project_root:
            pf = next((f for f in self._card_config.project_files 
                       if (hasattr(f.file_type, 'name') and f.file_type.name == "TRAJECTORY_DATA") 
                       or f.file_type == "TRAJECTORY_DATA"), None)
            if pf:
                abs_path = self.fs_service.current_project_root / pf.path
                if abs_path.exists():
                    logger_callback("📥 Carregando projeção 3D pré-calculada do disco (Cache)...")
                    try:
                        with open(abs_path, "rb") as f:
                            viz_data = pickle.load(f)
                    except Exception as e:
                        logger_callback(f"⚠️ Aviso: Cache da trajetória corrompido. Recalculando... ({e})")

        if viz_data is None:
            if not self.is_prepared():
                if self.has_fitted_pipelines():
                    logger_callback("🔄 Dados ausentes na RAM, mas pipelines encontrados no HD. Carregando automaticamente...")
                    self.run_preparation(logger_callback)
                else:
                    raise ValueError("É necessário executar a Preparação antes de visualizar a trajetória.")
                
            logger_callback("🌐 Iniciando cálculo da trajetória 3D (Isso pode levar alguns segundos)...")
            
            raw_source, raw_target = self.get_ready_datasets()
            artifacts = self.get_trained_artifacts()
            pipe_source = artifacts['source_pipeline_fitted']
            pipe_target = artifacts['target_pipeline_fitted']
            
            if not pipe_source: raise ValueError("Pipeline Source não encontrado.")
                
            traj_controller = TransformationTrajectoryController(math_service=self.math_service)
            
            logger_callback("  ↳ Modelando adaptação de domínio frame a frame...")
            frames = traj_controller.calculate_trajectory(
                source_dataset=raw_source,
                target_dataset=raw_target if raw_target else raw_source,
                source_pipeline=pipe_source,
                target_pipeline=pipe_target if pipe_target else pipe_source,
                logger_callback=logger_callback
            )
            
            logger_callback("  ↳ Reduzindo dimensionalidade via Joint PCA...")
            viz_data = traj_controller.create_3d_projection(frames)
            
            if self.fs_service.current_project_root and self._card_config:
                logger_callback("💾 Persistindo matrizes de trajetória no projeto...")
                rel_path_data = ProjectFileType.TRAJECTORY_DATA.build_path(strategy_id=self._card_config.id)
                abs_path_data = self.fs_service.current_project_root / rel_path_data
                abs_path_data.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    with open(abs_path_data, "wb") as f:
                        pickle.dump(viz_data, f)
                    self.add_project_file(ProjectFileType.TRAJECTORY_DATA, abs_path_data, logger_callback)
                except Exception as e:
                    logger_callback(f"⚠️ Falha ao salvar dados da trajetória: {e}")

        if viz_data and self.fs_service.current_project_root and self._card_config:
            has_plots = any((hasattr(f.file_type, 'name') and f.file_type.name == "TRAJECTORY_PLOT") 
                            or f.file_type == "TRAJECTORY_PLOT" 
                            for f in self._card_config.project_files)
            
            if not has_plots:
                traj_controller = TransformationTrajectoryController(math_service=self.math_service)
                
                rel_path_plot_template = ProjectFileType.TRAJECTORY_PLOT.build_path(
                    strategy_id=self._card_config.id, 
                    step="{step}"
                ).as_posix()
                
                abs_path_plot_template = str(self.fs_service.current_project_root / rel_path_plot_template)
                
                saved_pngs = traj_controller.export_headless_plots(
                    trajectory_data=viz_data, 
                    base_path_template=abs_path_plot_template,
                    logger_callback=logger_callback
                )
                
                for png_path in saved_pngs:
                    self.add_project_file(ProjectFileType.TRAJECTORY_PLOT, png_path, logger_callback)

        logger_callback("✅ Dados de visualização 3D prontos e garantidos!")
        return viz_data

    def _compute_post_prep_split_stats(self, logger_callback):
        if not self._transformed_source_data:
            return

        src_ids = len(set(i.identity for i in self._transformed_source_data.items if i.identity))
        
        self._post_prep_stats = {
            "source_transformed_imgs": len(self._transformed_source_data),
            "source_transformed_ids": src_ids,
            "final_embedding_dim": self._final_embedding_dim
        }

        if self._transformed_target_data:
            tgt_ids = len(set(i.identity for i in self._transformed_target_data.items if i.identity))
            self._post_prep_stats["target_transformed_imgs"] = len(self._transformed_target_data)
            self._post_prep_stats["target_transformed_ids"] = tgt_ids

        if self._card_config and self.fs_service.current_project_root:
            rel_path = ProjectFileType.PREPARATION_STATS.build_path(strategy_id=self._card_config.id)
            abs_path = self.fs_service.current_project_root / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger_callback("💾 Gravando cache de estatísticas de preparação no disco...")
            with open(abs_path, "w", encoding="utf-8") as f:
                json.dump(self._post_prep_stats, f, indent=4)
            
            self.add_project_file(ProjectFileType.PREPARATION_STATS, abs_path, logger_callback)