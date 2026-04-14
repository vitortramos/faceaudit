# faceaudit/services/kde_validation_service.py
from typing import List, Optional, Tuple, Set

from faceaudit.models.kde_validation_input import KDEValidationInput
from faceaudit.models.kde_validation_result import KDEValidationResult, ValidationSeverity
from faceaudit.enums.aggregation_method import AggregationMethod

class KDEValidationService:
    """
    Serviço de Auditoria Prévia para a Estratégia KDE.
    Gatekeeper que impede execução de configurações estatisticamente inválidas.
    """

    def validate(self, input_data: KDEValidationInput) -> KDEValidationResult:
        """
        Executa a validação completa baseada no Input encapsulado (Aba de Preparação).
        """
        
        # 1. Validação de Preenchimento Básico
        if not input_data.config.probe_dataset_name or not input_data.config.reference_dataset_name:
             return KDEValidationResult(
                 severity=ValidationSeverity.NEUTRAL,
                 messages=["Selecione os datasets Probe e Reference."],
                 data_severity=ValidationSeverity.NEUTRAL
             )

        if not input_data.has_data:
            return KDEValidationResult(
                severity=ValidationSeverity.WARNING, 
                messages=["Aguardando carregamento das estatísticas dos datasets..."],
                data_severity=ValidationSeverity.WARNING
            )

        # Variáveis de estado
        config = input_data.config
        stats_probe = input_data.probe_stats
        stats_ref = input_data.reference_stats
        
        severity = ValidationSeverity.OK
        data_sev = ValidationSeverity.OK
        params_sev = ValidationSeverity.OK
        
        messages = []
        metrics = {}

        # =====================================================================
        # 2. VALIDAÇÃO DE DADOS (Interseção & Compatibilidade)
        # =====================================================================
        
        # A. Interseção de Identidades (CRÍTICA)
        overlap_count = 0
        if stats_probe.identity_set and stats_ref.identity_set:
            overlap = stats_probe.identity_set.intersection(stats_ref.identity_set)
            overlap_count = len(overlap)
            metrics['overlap_identities'] = overlap_count

        if overlap_count == 0:
            data_sev = ValidationSeverity.CRITICAL
            messages.append(
                f"❌ Sem Interseção: Probe ({stats_probe.n_ids} IDs) e Reference ({stats_ref.n_ids} IDs) são disjuntos. "
                "O KDE precisa de identidades comuns para gerar a curva de Genuínos."
            )
        elif overlap_count < 10:
            data_sev = ValidationSeverity.WARNING
            messages.append(f"⚠️ Baixa Interseção ({overlap_count} IDs). A curva de genuínos será instável.")
        else:
            messages.append(f"✅ {overlap_count} identidades comuns encontradas.")

        # B. Compatibilidade de Versão (Embeddings)
        type_p = getattr(stats_probe, 'embedding_type_name', None)
        type_r = getattr(stats_ref, 'embedding_type_name', None)
        
        if type_p and type_r and type_p != type_r:
            data_sev = ValidationSeverity.CRITICAL
            messages.append(f"❌ Incompatibilidade: Probe é {type_p}, Reference é {type_r}.")

        # =====================================================================
        # 3. VALIDAÇÃO DE PARÂMETROS (Agregação & Clamp)
        # =====================================================================

        # A. Agregação (Smart Enrollment) - [CORRIGIDO: Acessando via DTO AggregationConfig]
        if config.aggregation:
            agg_method = config.aggregation.method
            k = config.aggregation.combination_group_size
            
            # Média de imagens por identidade no Probe
            avg_imgs = stats_probe.n_imgs / stats_probe.n_ids if stats_probe.n_ids > 0 else 0
            
            if agg_method == AggregationMethod.COMBINATORIAL:
                if avg_imgs < k:
                    params_sev = self._worse(params_sev, ValidationSeverity.WARNING)
                    messages.append(
                        f"⚠️ Combinatória Otimista: K={k} > Média de {avg_imgs:.1f} imgs/id. "
                        "A maioria cairá para o fallback (Centróide)."
                    )
            
            elif agg_method == AggregationMethod.CENTROID:
                if overlap_count < 50:
                     params_sev = self._worse(params_sev, ValidationSeverity.WARNING)
                     messages.append(
                         f"⚠️ Escassez de Dados: Agregação por Centróide resultará em apenas {overlap_count} vetores genuínos."
                     )

        # B. Parâmetros de Clamp (Estatística Mínima)
        p = config.parameters
        min_stat = 20
        if p.genuines_min_count < min_stat or p.impostors_min_count < min_stat:
             params_sev = self._worse(params_sev, ValidationSeverity.WARNING)
             messages.append(f"⚠️ 'Min Count' muito baixo (<{min_stat}). Risco de curvas degeneradas.")

        # =====================================================================
        # 4. CONSOLIDAÇÃO
        # =====================================================================
        
        severity = self._worse(data_sev, params_sev)

        return KDEValidationResult(
            severity=severity,
            messages=messages,
            metrics=metrics,
            data_severity=data_sev,
            params_severity=params_sev
        )

    # =========================================================================
    # VALIDAÇÃO DO DATASET DE VALIDAÇÃO (HOLD-OUT)
    # =========================================================================

    def validate_validation_setup(
        self,
        probe_train_stats,
        ref_train_stats,
        probe_val_stats,
        ref_val_stats
    ) -> KDEValidationResult:
        """
        Valida se os Datasets de Validação são adequados.
        (Nenhuma mudança estrutural necessária aqui, apenas proteção de estado)
        """

        severity = ValidationSeverity.OK
        data_sev = ValidationSeverity.OK
        params_sev = ValidationSeverity.OK

        messages = []
        metrics = {}

        if not probe_val_stats or not ref_val_stats:
            return KDEValidationResult(
                severity=ValidationSeverity.NEUTRAL,
                messages=["Selecione os Datasets de Validação (Probe e Reference)."],
                data_severity=ValidationSeverity.NEUTRAL,
                params_severity=ValidationSeverity.NEUTRAL
            )

        # 1) VAZAMENTO DE DADOS (DATA LEAKAGE)
        leakage_count = 0

        if probe_val_stats.identity_set:
            if probe_train_stats and probe_train_stats.identity_set:
                overlap = probe_val_stats.identity_set.intersection(probe_train_stats.identity_set)
                if overlap:
                    data_sev = ValidationSeverity.CRITICAL
                    messages.append(f"❌ Vazamento (Probe Val): Compartilha {len(overlap)} IDs com o Treino (Probe).")
                    leakage_count += len(overlap)
            
            if ref_train_stats and ref_train_stats.identity_set:
                overlap = probe_val_stats.identity_set.intersection(ref_train_stats.identity_set)
                if overlap:
                    data_sev = ValidationSeverity.CRITICAL
                    messages.append(f"❌ Vazamento (Probe Val): Compartilha {len(overlap)} IDs com o Treino (Reference).")
                    leakage_count += len(overlap)

        if ref_val_stats.identity_set:
            if probe_train_stats and probe_train_stats.identity_set:
                overlap = ref_val_stats.identity_set.intersection(probe_train_stats.identity_set)
                if overlap:
                    data_sev = ValidationSeverity.CRITICAL
                    messages.append(f"❌ Vazamento (Ref Val): Compartilha {len(overlap)} IDs com o Treino (Probe).")
                    leakage_count += len(overlap)
            
            if ref_train_stats and ref_train_stats.identity_set:
                overlap = ref_val_stats.identity_set.intersection(ref_train_stats.identity_set)
                if overlap:
                    data_sev = ValidationSeverity.CRITICAL
                    messages.append(f"❌ Vazamento (Ref Val): Compartilha {len(overlap)} IDs com o Treino (Reference).")
                    leakage_count += len(overlap)

        metrics["data_leakage_ids"] = leakage_count

        # 2) INTERSEÇÃO ENTRE OS DATASETS DE VALIDAÇÃO
        val_overlap_count = 0
        if probe_val_stats.identity_set and ref_val_stats.identity_set:
            val_overlap = probe_val_stats.identity_set.intersection(ref_val_stats.identity_set)
            val_overlap_count = len(val_overlap)
        
        metrics["val_genuine_ids"] = val_overlap_count

        if val_overlap_count == 0:
            data_sev = ValidationSeverity.CRITICAL
            messages.append(
                f"❌ Sem Interseção na Validação: Probe Val ({probe_val_stats.n_ids} IDs) e Ref Val ({ref_val_stats.n_ids} IDs) "
                "não têm pessoas em comum."
            )
        elif val_overlap_count < 10:
            data_sev = self._worse(data_sev, ValidationSeverity.WARNING)
            messages.append(f"⚠️ Baixa Interseção ({val_overlap_count} IDs em comum).")

        # 3) COMPATIBILIDADE DE EMBEDDING
        types = {
            "Treino (Probe)": getattr(probe_train_stats, "embedding_type_name", None),
            "Val (Probe)": getattr(probe_val_stats, "embedding_type_name", None),
            "Val (Ref)": getattr(ref_val_stats, "embedding_type_name", None),
        }
        
        base_type = types["Treino (Probe)"]
        if base_type:
            for label, emb_type in types.items():
                if emb_type and emb_type != base_type:
                    data_sev = ValidationSeverity.CRITICAL
                    messages.append(f"❌ Incompatibilidade: {label} usa {emb_type}, mas o modelo foi treinado em {base_type}.")

        if data_sev == ValidationSeverity.OK:
            messages.append(f"✅ Setup de Validação íntegro e sem vazamentos.")

        return KDEValidationResult(
            severity=data_sev,
            messages=messages,
            metrics=metrics,
            data_severity=data_sev,
            params_severity=params_sev
        )

    def _worse(self, s1: ValidationSeverity, s2: ValidationSeverity) -> ValidationSeverity:
        hierarchy = {ValidationSeverity.CRITICAL: 3, ValidationSeverity.WARNING: 2, ValidationSeverity.OK: 1, ValidationSeverity.NEUTRAL: 0}
        return s1 if hierarchy.get(s1, 0) > hierarchy.get(s2, 0) else s2