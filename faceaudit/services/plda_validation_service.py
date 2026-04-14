from typing import Optional, Dict, Any, Tuple, Set, List
import numpy as np

# Imports de Enums
from faceaudit.enums.plda_training_strategy import PLDATrainingStrategy
from faceaudit.enums.embedding_type import EmbeddingType
from faceaudit.enums.transformation_type import TransformationType

# Models
from faceaudit.models.plda_validation_result import PLDAValidationResult, ValidationSeverity
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.plda_validation_input import PLDAValidationInput

class PLDAValidationService:
    """
    Serviço de Auditoria Prévia de Treinamento (Gatekeeper).
    
    Responsabilidade:
    -----------------
    Este serviço atua como um "Gatekeeper" (Guardião) que impede o início de treinamentos 
    que estejam matematicamente condenados ao fracasso.

    Pilares de Validação:
    ---------------------
    1. Validação de Dados: Integridade e Isolamento de Datasets.
    2. Validação de Pipeline: Coerência da engenharia de características (CORAL, PCA, LDA).
    3. Validação Geométrica (PLDA): Regra de Fisher (Identidades > Dimensões).
    4. Validação Logística (Calibração): Avalia se a Simulação do Split gerou pares suficientes.
    """

    # =========================================================================
    # 0. FACADE (MÉTODO UNIFICADO)
    # =========================================================================
    def validate(self, input_data: PLDAValidationInput) -> PLDAValidationResult:
        
        # 1. Validação de Dados (Regras de Negócio e Compatibilidade)
        data_res = self.validate_data_rules(input_data)
        
        # 2. Validação de Pipeline (Lógica de Transformação e Dimensão)
        pipe_res = self.validate_pipeline_logic(input_data)
        
        # 3. Validação de Estabilidade Numérica baseada na Simulação de Split
        plda_res = self.validate_strategy_params(input_data)

        # 4. Agregação Final da Severidade
        final_severity = self._worse(data_res.severity, pipe_res.severity)
        final_severity = self._worse(final_severity, plda_res.severity)

        return PLDAValidationResult(
            severity=final_severity,
            
            data_severity=data_res.data_severity,
            messages=data_res.messages,
            
            pipeline_severity=pipe_res.pipeline_severity,
            pipeline_message=pipe_res.pipeline_message,
            
            plda_severity=plda_res.plda_severity,
            plda_message=plda_res.plda_message,
            
            calib_severity=plda_res.calib_severity,
            calib_message=plda_res.calib_message,
            
            metrics={**data_res.metrics, **plda_res.metrics}
        )

    # =========================================================================
    # 1. VALIDAÇÃO DE DADOS
    # =========================================================================
    def validate_data_rules(self, input_data: PLDAValidationInput) -> PLDAValidationResult:
        def empty_result(sev=ValidationSeverity.NEUTRAL, msg=""):
            return PLDAValidationResult(
                severity=sev,
                data_severity=sev,
                messages=[msg] if msg else [],
                metrics={}
            )

        if input_data is None or getattr(input_data, "config", None) is None:
            return empty_result(ValidationSeverity.NEUTRAL, "Selecione os datasets e uma estratégia.")

        cfg = input_data.config
        strat: PLDATrainingStrategy = cfg.strategy

        if strat is None:
            return empty_result(ValidationSeverity.NEUTRAL, "Selecione uma estratégia.")

        st_src = input_data.train_stats
        st_tgt = input_data.calib_stats

        if st_src is None or st_src.n_imgs <= 0:
            return empty_result(ValidationSeverity.NEUTRAL, "Selecione o dataset Source (Treino).")

        if strat.requires_target_labels:
            if st_tgt is None or st_tgt.n_imgs <= 0:
                return empty_result(ValidationSeverity.NEUTRAL, "Selecione o dataset Target (com labels).")
        else:
            if st_tgt is None or st_tgt.n_imgs <= 0:
                return empty_result(ValidationSeverity.NEUTRAL, "Selecione o dataset de Calibração (Target).")

        severity = ValidationSeverity.OK
        messages = []
        metrics: Dict[str, Any] = {}

        emb_sev, emb_msgs = self._check_embedding_compatibility(cfg.embedding_type, st_src, st_tgt)
        if emb_sev != ValidationSeverity.OK:
            severity = self._worse(severity, emb_sev)
            messages.extend(emb_msgs)

        src_set = getattr(st_src, "identity_set", set()) or set()
        tgt_set = getattr(st_tgt, "identity_set", set()) or set()

        overlap_count = self._calculate_overlap(src_set, tgt_set)
        metrics["overlap"] = overlap_count

        requires_overlap = bool(getattr(strat, "requires_target_labels", False)) or \
                           bool(getattr(strat, "use_target_for_lda", False)) or \
                           bool(getattr(strat, "use_target_for_plda", False))

        if requires_overlap:
            if overlap_count == 0:
                severity = self._worse(severity, ValidationSeverity.CRITICAL)
                messages.append(f"❌ A estratégia '{strat.label}' exige identidades correspondentes entre Source e Target (Overlap=0).")
            elif st_tgt and st_tgt.n_ids > 0 and overlap_count < (st_tgt.n_ids * 0.5):
                severity = self._worse(severity, ValidationSeverity.WARNING)
                messages.append(f"⚠️ Apenas {overlap_count} identidades emparelhadas (Source↔Target). Adaptação pode ser ineficaz.")
            else:
                messages.append(f"✅ {overlap_count} identidades emparelhadas encontradas (Source↔Target).")
        else:
            if overlap_count > 0:
                severity = self._worse(severity, ValidationSeverity.CRITICAL)
                messages.append(f"❌ LEAKAGE: {overlap_count} identidades do Source aparecem também no Target (deve ser 0).")
            else:
                messages.append("✅ Dados disjuntos (sem vazamento de identidades) entre Source e Target.")

        return PLDAValidationResult(
            severity=severity,
            data_severity=severity,
            messages=messages,
            metrics=metrics
        )

    # =========================================================================
    # 2. VALIDAÇÃO LÓGICA DE PIPELINES
    # =========================================================================
    def validate_pipeline_logic(self, input_data: PLDAValidationInput) -> PLDAValidationResult:
        cfg = input_data.config
        strat = cfg.strategy
        p_src = input_data.source_pipeline_config
        p_tgt = input_data.target_pipeline_config

        pipe_sev = ValidationSeverity.OK
        messages = []

        if not p_src or not p_tgt or len(p_src.transformators) == 0 or len(p_tgt.transformators) == 0:
             return PLDAValidationResult(
                severity=ValidationSeverity.NEUTRAL, 
                pipeline_severity=ValidationSeverity.NEUTRAL,
                pipeline_message="Selecione pipelines válidos."
            )

        has_coral_src = self._has_step(p_src, TransformationType.CORAL_TRANSFORMATION)
        
        if strat.requires_source_adaptation and not has_coral_src:
            pipe_sev = self._worse(pipe_sev, ValidationSeverity.CRITICAL)
            messages.append(f"❌ '{strat.label}' exige um Pipeline Source com CORAL.")
        
        if not strat.allows_source_adaptation and has_coral_src:
            pipe_sev = self._worse(pipe_sev, ValidationSeverity.CRITICAL)
            messages.append(f"❌ '{strat.label}' proíbe CORAL no Source.")

        has_coral_tgt = self._has_step(p_tgt, TransformationType.CORAL_TRANSFORMATION)
        if has_coral_tgt and not strat.allows_target_adaptation:
                pipe_sev = self._worse(pipe_sev, ValidationSeverity.CRITICAL)
                messages.append(f"❌ Pipeline Target NÃO deve ter CORAL (Âncora).")

        if strat.requires_symmetric_pipelines:
            if not self._are_structurally_equal(p_src, p_tgt):
                pipe_sev = self._worse(pipe_sev, ValidationSeverity.WARNING)
                messages.append(f"⚠️ Standard: Pipelines deveriam ser iguais.")

        has_lda_tgt = self._has_step(p_tgt, TransformationType.LDA_TRANSFORMATION)
        if not strat.allows_target_supervised_steps and has_lda_tgt:
            pipe_sev = self._worse(pipe_sev, ValidationSeverity.CRITICAL)
            messages.append(f"❌ '{strat.label}' não recomenda LDA no Target.")

        if strat.use_target_for_whitening and not (p_src.has_pca or p_src.has_lda):
                pipe_sev = self._worse(pipe_sev, ValidationSeverity.WARNING)
                messages.append(f"⚠️ Otimiza Matriz, mas Pipeline não tem PCA ou LDA.")

        dim_src = self._get_declared_output_dim(p_src)
        dim_tgt = self._get_declared_output_dim(p_tgt)
        
        if dim_src is not None and dim_tgt is not None:
            if dim_src != dim_tgt:
                pipe_sev = self._worse(pipe_sev, ValidationSeverity.CRITICAL)
                messages.append(f"❌ Erro de Dimensão: Source sai com {dim_src}D, mas Target sai com {dim_tgt}D.")
        elif (dim_src is None) != (dim_tgt is None):
            defined_val = dim_src if dim_src else dim_tgt
            who_defined = "Source" if dim_src else "Target"
            pipe_sev = self._worse(pipe_sev, ValidationSeverity.CRITICAL)
            messages.append(f"❌ Erro de Dimensão: {who_defined} reduz para {defined_val}D, mas o outro mantém original.")

        msg_final = "\n".join(messages) if messages else "✅ Pipeline consistente."
        
        return PLDAValidationResult(
            severity=pipe_sev, 
            pipeline_severity=pipe_sev, 
            pipeline_message=msg_final
        )

    # =========================================================================
    # 3. VALIDAÇÃO GEOMÉTRICA E CALIBRAÇÃO (NOVO)
    # =========================================================================
    def validate_strategy_params(self, input_data: PLDAValidationInput) -> PLDAValidationResult:
        if not input_data or getattr(input_data, "config", None) is None:
            return PLDAValidationResult(severity=ValidationSeverity.NEUTRAL)

        sim_result = input_data.split_simulation
        if not sim_result or not sim_result.success:
            return PLDAValidationResult(
                severity=ValidationSeverity.NEUTRAL,
                plda_severity=ValidationSeverity.NEUTRAL,
                plda_message="⚠️ Execute a Simulação de Split para validar os parâmetros geométricos."
            )

        cfg = input_data.config
        metrics: Dict[str, Any] = {}
        severity = ValidationSeverity.OK
        messages: List[str] = []

        # ------------------------------------------------------------
        # 1. Dimensão de Entrada
        # ------------------------------------------------------------
        final_dim = input_data.final_embedding_dim
        if final_dim is not None:
            d_in = int(final_dim)
        else:
            pipe_out = self._get_declared_output_dim(input_data.source_pipeline_config)
            d_in = int(pipe_out) if pipe_out else 512

        metrics["input_dim"] = d_in

        # ------------------------------------------------------------
        # 2. Regra de Fisher (Identidades > Dimensões)
        # ------------------------------------------------------------
        n_classes = sim_result.plda_identities_count
        
        if d_in <= 0:
            return PLDAValidationResult(
                severity=ValidationSeverity.CRITICAL,
                plda_severity=ValidationSeverity.CRITICAL,
                plda_message="❌ Dimensão de entrada inválida (<= 0)."
            )

        if n_classes < 2:
            return PLDAValidationResult(
                severity=ValidationSeverity.CRITICAL,
                plda_severity=ValidationSeverity.CRITICAL,
                plda_message=f"❌ O Split PLDA separou apenas {n_classes} identidades. Requer mínimo 2."
            )

        fisher_ceiling = n_classes - 1

        if fisher_ceiling <= d_in:
            severity = self._worse(severity, ValidationSeverity.CRITICAL)
            msg = (
                f"❌ Singularidade de Fisher: O Treino possui {n_classes} Identidades, "
                f"o que suporta um espaço máximo de {fisher_ceiling} Dimensões.\n"
                f"Seu pipeline entrega vetores de {d_in}D.\n"
                f"Solução: Aumente as pessoas no Dataset ou adicione um PCA/LDA no pipeline."
            )
            messages.append(msg)
        else:
            messages.append(f"✅ Estabilidade Geométrica OK ({n_classes} IDs > {d_in}D).")

        # ------------------------------------------------------------
        # 3. Rank Efetivo (Low Rank vs Full Rank)
        # ------------------------------------------------------------
        config_params = cfg.parameters
        target_mode = config_params.plda_mode
        target_latent = int(getattr(config_params, "latent_dim", 0) or 0)

        if target_mode.name == "LOW_RANK":
            if target_latent > fisher_ceiling:
                severity = self._worse(severity, ValidationSeverity.CRITICAL)
                messages.append(
                    f"❌ Modo Low-Rank: Você pediu {target_latent}D, mas as {n_classes} identidades "
                    f"limitam o cálculo a {fisher_ceiling}D."
                )
            elif target_latent >= d_in:
                 severity = self._worse(severity, ValidationSeverity.WARNING)
                 messages.append(f"⚠️ Modo Low-Rank: Pedido {target_latent}D, mas a entrada já é {d_in}D.")
            else:
                 messages.append(f"✅ Redução Low-Rank configurada ({target_latent}D).")

        # ------------------------------------------------------------
        # 4. Viabilidade da Calibração (Pares LogReg)
        # ------------------------------------------------------------
        gen_pairs = sim_result.calib_genuine_pairs
        imp_pairs = sim_result.calib_impostor_pairs

        if gen_pairs < 100:
            severity = self._worse(severity, ValidationSeverity.CRITICAL)
            messages.append(f"❌ Calibração Falha: Apenas {gen_pairs} pares genuínos disponíveis. A Regressão Logística precisa de no mínimo 100 para criar a curva.")
        elif imp_pairs < 500:
             severity = self._worse(severity, ValidationSeverity.CRITICAL)
             messages.append(f"❌ Calibração Falha: Apenas {imp_pairs} pares impostores disponíveis.")
        else:
            messages.append(f"✅ Poder de Calibração OK ({gen_pairs} Genuínos, {imp_pairs} Impostores).")

        return PLDAValidationResult(
            severity=severity,
            plda_severity=severity, 
            plda_message="\n".join(messages),
            metrics=metrics
        )

    # =========================================================================
    # MÉTODOS AUXILIARES (PRIVADOS)
    # =========================================================================
    
    def _has_step(self, config, type_enum: TransformationType) -> bool:
        return any(t.type == type_enum for t in config.transformators)

    def _are_structurally_equal(self, cfg1, cfg2) -> bool:
        if len(cfg1.transformators) != len(cfg2.transformators): return False
        for t1, t2 in zip(cfg1.transformators, cfg2.transformators):
            if t1.type != t2.type: return False
        return True

    def _check_embedding_compatibility(self, req_type: EmbeddingType, stats_train, stats_calib):
        requirements = {
            EmbeddingType.SEPAEL_V2: {'embedding_v2', 'embedding', 'embedding_sepaelv2'},
            EmbeddingType.SEPAEL_V4: {'embedding_v4', 'embedding_sepaelv4'},
        }
        allowed_cols = requirements.get(req_type, set())
        msgs = []
        severity = ValidationSeverity.OK

        def check_ds(stats, label):
            cols = getattr(stats, 'embedding_columns', set())
            if not cols: return True 
            if not cols.intersection(allowed_cols): return False
            return True

        if not check_ds(stats_train, "Source"):
            severity = ValidationSeverity.CRITICAL
            msgs.append(f"❌ Dataset Source não contém embeddings '{req_type.display_name}'.")

        if not check_ds(stats_calib, "Target"):
            severity = ValidationSeverity.CRITICAL
            msgs.append(f"❌ Dataset Target não contém embeddings '{req_type.display_name}'.")
            
        return severity, msgs

    def _calculate_overlap(self, set_a: Set[str], set_b: Set[str]) -> int:
        return len(set_a.intersection(set_b))

    def _worse(self, s1: ValidationSeverity, s2: ValidationSeverity) -> ValidationSeverity:
        hierarchy = {ValidationSeverity.CRITICAL: 3, ValidationSeverity.WARNING: 2, ValidationSeverity.OK: 1, ValidationSeverity.NEUTRAL: 0}
        return s1 if hierarchy.get(s1, 0) > hierarchy.get(s2, 0) else s2
    
    def _get_declared_output_dim(self, pipeline_config) -> Optional[int]:
        if not pipeline_config or not hasattr(pipeline_config, 'steps') or not pipeline_config.steps:
            return None
        
        last_step = pipeline_config.steps[-1]
        dto_config = getattr(last_step, "config", None)
        
        if not dto_config:
            return None

        if hasattr(dto_config, 'n_components'):
            return dto_config.n_components
            
        if hasattr(dto_config, 'output_dim'):
            return dto_config.output_dim
            
        return None