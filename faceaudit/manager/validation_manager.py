from typing import Callable, Optional, Tuple, List, Dict
from pathlib import Path

# --- DTOs ---
from faceaudit.models.validation_config import ValidationConfig
from faceaudit.models.validation_comparison_report import ValidationComparisonReport
from faceaudit.models.strategy_validation_result import StrategyValidationResult
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.enums.project_file_type import ProjectFileType
from faceaudit.models.arena_validation_input import ArenaValidationInput
from faceaudit.models.arena_validation_result import ArenaValidationResult
from faceaudit.models.project_file import ProjectFile

# --- SERVICES E MANAGERS ---
from faceaudit.manager.strategies_manager import StrategiesManager
from faceaudit.services.validation_math_service import ValidationMathService
from faceaudit.services.dataset_loader_service import DatasetLoaderService
from faceaudit.services.file_system_service import FileSystemService
from faceaudit.services.dataset_stats_service import DatasetStatsService
from faceaudit.services.arena_validation_service import ArenaValidationService
from faceaudit.services.hash_service import HashService


class ValidationManager:
    """
    Orquestrador e Gerenciador de Sessão da 'Arena de Validação'.
    Agora 100% alinhado com o Pendrive Paradigm e Cadeia de Custódia.
    """

    def __init__(self,
                 strategies_manager: StrategiesManager,
                 math_service: ValidationMathService,
                 fs_service: FileSystemService,
                 dataset_loader: DatasetLoaderService,
                 stats_service: DatasetStatsService,
                 arena_validator: ArenaValidationService,
                 hash_service: HashService):  # <--- HASH SERVICE ADICIONADO AQUI
        
        self.strategies_manager = strategies_manager
        self.math_service = math_service
        self.fs_service = fs_service
        self.dataset_loader = dataset_loader
        self.stats_service = stats_service
        self.arena_validator = arena_validator
        self.hash_service = hash_service
        
        self._config = ValidationConfig()
        self._is_dirty = False
        self._last_audit_result: Optional[ArenaValidationResult] = None
        self._last_report_path: Optional[Path] = None

    # =========================================================================
    # 1. GESTÃO DE ESTADO (SESSION & DIRTY FLAG)
    # =========================================================================
    
    def get_config(self) -> ValidationConfig:
        return self._config

    def update_config(self, new_config: ValidationConfig, run_audit: bool = True):
        # Guarda os project files antigos para não perdê-los no DTO novo
        old_files = self._config.project_files.copy() if self._config else []
        new_config.project_files = old_files
        
        self._config = new_config
        self.set_dirty(True)
        self._config.is_audit_passed = False 
        
        # A Trava de Custódia agora! (Assina com Hash usando os nomes do DTO)
        self._lock_local_inputs(logger_callback=lambda x: None)
        
        if run_audit:
            self.audit_arena()

    def set_dirty(self, state: bool):
        self._is_dirty = state

    def is_dirty(self) -> bool:
        return self._is_dirty

    def load_session(self, config: ValidationConfig):
        self._config = config
        self.set_dirty(False)
        self.audit_arena()

    # =========================================================================
    # 2. CICLO DE VIDA DO MODELO E GERENCIAMENTO DE MANIFESTO
    # =========================================================================

    def add_project_file(self, file_type: ProjectFileType, abs_path: Path, logger_callback: Callable = print) -> None:
        if not self._config or not self.fs_service.current_project_root: return

        if not abs_path.exists():
            logger_callback(f"⚠️ Aviso: Arquivo não encontrado para rastreio: {abs_path.name}")
            return

        try:
            file_hash = self.hash_service.calculate_file_hash(abs_path)
            rel_path = abs_path.relative_to(self.fs_service.current_project_root).as_posix()
            
            existing_file = next((pf for pf in self._config.project_files if pf.path == rel_path), None)
            if existing_file:
                existing_file.hash = file_hash
                existing_file.file_type = file_type
            else:
                new_pf = ProjectFile(file_type=file_type, path=rel_path, hash=file_hash)
                self._config.project_files.append(new_pf)
                
            logger_callback(f"  ↳ Manifesto da Arena: [{file_type.name}] {Path(rel_path).name} assinado.")
            self.set_dirty(True)
        except Exception as e:
            logger_callback(f"⚠️ Erro ao registrar rastreio de {abs_path.name} na Arena: {e}")

    def delete_project_file(self, project_file: ProjectFile, logger_callback: Callable = print) -> None:
        if not self._config or not self.fs_service.current_project_root: return

        if project_file in self._config.project_files:
            self._config.project_files.remove(project_file)
            self.set_dirty(True)
            
        abs_path = self.fs_service.current_project_root / project_file.path
        if abs_path.exists():
            try:
                abs_path.unlink()
                logger_callback(f"🗑️ Arquivo da Arena apagado do disco: {abs_path.name}")
            except Exception as e:
                logger_callback(f"⚠️ Não foi possível apagar o arquivo {abs_path.name}: {e}")

    def invalidate_math_artifacts(self, logger_callback: Callable[[str], None] = print) -> None:
        if not self._config or not self.fs_service.current_project_root: return
        
        # Tipos dinâmicos gerados PELA ARENA (que perdem validade se o Dataset mudar)
        dynamic_types = {ProjectFileType.VALIDATION_COMPARISON_DATA, ProjectFileType.DISTRIBUTION_GRAPH}
        
        for pf in list(self._config.project_files):
            if pf.file_type in dynamic_types:
                self.delete_project_file(pf, logger_callback)
                
    def _lock_local_inputs(self, logger_callback: Callable = print) -> None:
        """
        Varre os dados que a Arena precisa e os inclui/atualiza no manifesto com HASH real.
        Isso é o que garante que o ProjectManager vai poder confiar nesses arquivos no Deep Copy!
        """
        if not self._config or not self.fs_service.current_project_root:
            return
            
        inputs = [
            (self._config.probe_dataset_name, ProjectFileType.DATASET_METADATA),
            (self._config.reference_dataset_name, ProjectFileType.DATASET_METADATA)
        ]
        
        # Remove do manifesto quem não estiver mais selecionado nos combos
        active_dataset_clean_names = {name.split(" (")[0] for name, _ in inputs if name and name != "Selecione..."}
        
        for pf in list(self._config.project_files):
            if pf.file_type == ProjectFileType.DATASET_METADATA:
                pf_clean_name = Path(pf.path).stem
                if pf_clean_name not in active_dataset_clean_names:
                    # O usuário mudou de ideia no combo, esse arquivo não pertence mais à Arena
                    self._config.project_files.remove(pf)
                    self.set_dirty(True)
        
        # Adiciona/Atualiza os que estão ativos
        for name, f_type in inputs:
            if not name or name == "Selecione...": continue
            clean_name = name.split(" (")[0]
            
            path = self.fs_service.resolve_file(clean_name, f_type)
            # Só rastreia se o arquivo estiver local no projeto (o fs_service copia depois se for global)
            if path and path.exists() and self.fs_service.current_project_root in path.parents:
                self.add_project_file(f_type, path, logger_callback)

    # =========================================================================
    # 3. GESTÃO DE DADOS (UI E METADADOS)
    # =========================================================================
    
    def get_available_datasets(self) -> Tuple[List[str], List[str]]:
        return self.fs_service.list_available_files(ProjectFileType.DATASET_METADATA)
    
    def get_dataset_stats(self, dataset_name: str):
        if not dataset_name or dataset_name == "Selecione...": 
            return None
            
        clean_name = dataset_name.split(" (")[0]
        path = self.fs_service.resolve_file(clean_name, ProjectFileType.DATASET_METADATA)
        
        if path:
            return self.stats_service.get_stats(clean_name, target_directory=path.parent)
            
        return None

    def get_active_strategies_info(self) -> List[Dict[str, str]]:
        active_managers = self.strategies_manager.get_all_managers()
        info_list = []
        
        for strat_id, manager in active_managers.items():
            config = manager.get_config()
            if config:
                info_list.append({
                    "id": strat_id,
                    "name": getattr(config, 'name', strat_id),
                    "type": config.type.name.replace("_STRATEGY", "") if hasattr(config, 'type') else "UNKNOWN",
                    "is_trained": manager.has_model_on_disk() or manager.is_model_loaded()
                })
        return info_list

    # =========================================================================
    # 4. GATEKEEPER E VALIDAÇÃO
    # =========================================================================

    def audit_arena(self) -> ArenaValidationResult:
        probe_stats = self.get_dataset_stats(self._config.probe_dataset_name)
        ref_stats = self.get_dataset_stats(self._config.reference_dataset_name)
        
        history_stats_map = {}
        for strategy_id in self._config.strategy_ids:
            manager = self.strategies_manager.get_manager(strategy_id)
            if not manager: continue
                
            training_datasets = manager.get_training_datasets()
            stats_list = []
            for ds_name in training_datasets:
                st = self.get_dataset_stats(ds_name)
                if st: stats_list.append(st)
                    
            history_stats_map[strategy_id] = stats_list

        audit_input = ArenaValidationInput(
            probe_stats=probe_stats,
            reference_stats=ref_stats,
            strategies_history_stats=history_stats_map
        )
        
        result = self.arena_validator.validate(audit_input)
        self._last_audit_result = result
        self._config.is_audit_passed = result.is_pure
        
        if result.is_pure:
            self._config.audit_messages = ["✅ Ambiente Seguro. Nenhum vazamento detectado."]
        else:
            self._config.audit_messages = ["⚠️ Alerta de Auditoria: Verifique o painel para detalhes de Leakage."]
            
        return result
    
    def get_last_audit_result(self) -> Optional[ArenaValidationResult]:
        return self._last_audit_result

    def run_validation(self, logger_callback: Callable[[str], None] = print) -> ValidationComparisonReport:
        logger_callback("🚀 <b>Iniciando Arena de Validação Comparativa...</b>")

        # 1. Invalida os resultados gráficos anteriores (Cadeia de Custódia)
        self.invalidate_math_artifacts(logger_callback)

        if not self._config.is_audit_passed:
            logger_callback("⚠️ <b>Aviso:</b> A auditoria detectou problemas de Data Leakage neste cenário!")

        report = ValidationComparisonReport(
            probe_dataset_name=self._config.probe_dataset_name,
            reference_dataset_name=self._config.reference_dataset_name
        )

        # 2. Carrega o Gabarito uma única vez como FaceDataset (O material bruto)
        logger_callback(f"📥 Carregando Gabarito: {self._config.probe_dataset_name} vs {self._config.reference_dataset_name}")
        probe_ds_raw = self._load_dataset(self._config.probe_dataset_name)
        ref_ds_raw = self._load_dataset(self._config.reference_dataset_name)

        if not probe_ds_raw or not ref_ds_raw:
            raise ValueError("Falha ao carregar os datasets de validação. Verifique os arquivos no projeto.")

        # 3. Loop de Combate entre as Estratégias selecionadas
        for strategy_id in self._config.strategy_ids:
            logger_callback(f"\n⚙️ <b>Avaliando estratégia: {strategy_id}</b>...")
            
            manager = self.strategies_manager.get_manager(strategy_id)
            if not manager:
                logger_callback(f"  ⚠️ Aviso: Estratégia {strategy_id} não encontrada na RAM.")
                continue
                
            try:
                # Garante que o binário (.pkl) esteja pronto
                manager.ensure_model_loaded()
            except Exception as e:
                logger_callback(f"  ⚠️ Aviso: {strategy_id} não possui modelo treinado ou falhou ao carregar: {e}")
                continue

            try:
                strat_config = manager.get_config()
                strategy_name = getattr(strat_config, 'name', strategy_id) or strategy_id
                strategy_type = "UNKNOWN"
                if hasattr(strat_config, 'type'):
                    strategy_type = strat_config.type.name.replace("_STRATEGY", "")

                # --- O GRANDE MOMENTO (CONTRATO DO PROTOCOLO) ---
                # Passamos o FaceDataset. O Manager da estratégia se vira para 
                # converter para seus embeddings específicos e gerar os LRs.
                logger_callback(f"  ↳ Extraindo Likelihood Ratios (LRs)...")
                genuine_lrs, impostor_lrs = manager.run_validation_inference(
                    probe_dataset=probe_ds_raw, 
                    reference_dataset=ref_ds_raw,
                    logger_callback=logger_callback
                )
                
                if not genuine_lrs or not impostor_lrs:
                    logger_callback(f"  ❌ Falha: A estratégia não retornou scores suficientes para avaliação.")
                    continue

                # 4. CÁLCULO DAS MÉTRICAS (Responsabilidade da Arena)
                logger_callback(f"  ↳ Calculando EER, Cllr e curvas de desempenho...")
                strat_result: StrategyValidationResult = self.math_service.calculate_metrics(
                    strategy_id=strategy_id,
                    strategy_name=strategy_name,
                    strategy_type=strategy_type,
                    genuine_scores=genuine_lrs,
                    impostor_scores=impostor_lrs
                )
                
                # Registra no laudo final
                report.results[strategy_id] = strat_result
                logger_callback(f"  ✅ Sucesso! (Cllr: {strat_result.metrics.cllr:.3f} | EER: {strat_result.metrics.eer:.2%})")
                
            except Exception as e:
                import traceback
                logger_callback(f"  ❌ Erro crítico ao avaliar {strategy_id}: {e}")
                traceback.print_exc()

        logger_callback("\n🏁 <b>Validação Comparativa Concluída!</b>")
        return report

    def _load_dataset(self, raw_name: str) -> Optional[EmbeddingDataset]:
        if not raw_name or raw_name == "Selecione...":
            return None
            
        clean_name = raw_name.split(" (")[0]
        path = self.fs_service.resolve_file(clean_name, ProjectFileType.DATASET_METADATA)
        
        if path and path.exists():
            return self.dataset_loader.load(path)
        return None
    
    def save_report(self, report: ValidationComparisonReport, report_name: str) -> Path:
        """
        Salva o JSON final dos resultados e assina FORENSEMENTE no manifesto da Arena.
        """
        if not self.fs_service.current_project_root:
            raise RuntimeError("Nenhum projeto ativo para salvar o relatório.")

        rel_path = ProjectFileType.VALIDATION_COMPARISON_DATA.build_path(report_name=report_name)
        abs_path = self.fs_service.current_project_root / rel_path
        
        report.save_to_disk(abs_path)
        
        # A própria Arena se responsabiliza por assinar seus artefatos (como KDE e PLDA fazem)
        self.add_project_file(ProjectFileType.VALIDATION_COMPARISON_DATA, abs_path)
        self._last_report_path = abs_path
                
        return abs_path