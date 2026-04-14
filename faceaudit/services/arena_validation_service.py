from typing import Set

from faceaudit.models.arena_validation_input import ArenaValidationInput
from faceaudit.models.arena_validation_result import ArenaValidationResult
from faceaudit.models.strategy_audit_result import StrategyAuditResult
from faceaudit.models.plda_validation_result import ValidationSeverity

class ArenaValidationService:
    """
    Guardião da Arena de Validação.
    Audita a sanidade do gabarito (Probe vs Ref) e detecta Data Leakage 
    (Vazamento de Dados) cruzando o gabarito com o histórico de treino das estratégias.
    """

    def validate(self, input_data: ArenaValidationInput) -> ArenaValidationResult:
        result = ArenaValidationResult()

        if not input_data.probe_stats or not input_data.reference_stats:
            result.severity = ValidationSeverity.NEUTRAL
            result.arena_messages.append("ℹ️ Selecione os datasets Probe e Reference para habilitar a auditoria.")
            result.is_pure = False
            return result

        # 1. VALIDAÇÃO DE SANIDADE DA ARENA (Probe vs Reference)
        arena_sev, arena_msgs, probe_ids, ref_ids, probe_hashes, ref_hashes = self._audit_arena_sanity(
            input_data.probe_stats, 
            input_data.reference_stats
        )
        
        result.arena_messages.extend(arena_msgs)
        result.severity = self._worse(result.severity, arena_sev)

        # Se a Arena for inválida (ex: 0 pares genuínos), nem adianta checar leakage
        if arena_sev == ValidationSeverity.CRITICAL:
            result.is_pure = False
            return result

        # União do conjunto da Arena (Tudo que vai ser testado hoje)
        arena_all_ids = probe_ids.union(ref_ids)
        arena_all_hashes = probe_hashes.union(ref_hashes)

        # 2. VALIDAÇÃO DE DATA LEAKAGE (Arena vs Estratégias)
        for strategy_id, history_stats_list in input_data.strategies_history_stats.items():
            strat_audit = self._audit_strategy_leakage(
                strategy_id, 
                history_stats_list, 
                arena_all_ids, 
                arena_all_hashes
            )
            
            result.strategy_audits[strategy_id] = strat_audit
            result.severity = self._worse(result.severity, strat_audit.severity)
            if not strat_audit.is_pure:
                result.is_pure = False

        return result

    def _audit_arena_sanity(self, probe_stats, ref_stats):
        """Garante que é matematicamente possível rodar a validação."""
        severity = ValidationSeverity.OK
        messages = []

        p_ids: Set[str] = getattr(probe_stats, "identity_set", set()) or set()
        r_ids: Set[str] = getattr(ref_stats, "identity_set", set()) or set()
        
        p_hashes: Set[str] = getattr(probe_stats, "hash_set", set()) or set()
        r_hashes: Set[str] = getattr(ref_stats, "hash_set", set()) or set()

        shared_ids = p_ids.intersection(r_ids)
        if len(shared_ids) == 0:
            severity = ValidationSeverity.CRITICAL
            messages.append("❌ IMPOSSÍVEL AVALIAR: Probe e Reference não possuem NENHUMA identidade em comum. Não haverá pares genuínos.")
        elif len(shared_ids) < 5:
            severity = self._worse(severity, ValidationSeverity.WARNING)
            messages.append(f"⚠️ Alerta Estatístico: Apenas {len(shared_ids)} identidades em comum. Curvas DET e EER não serão confiáveis.")
        else:
            messages.append(f"✅ Gabarito viável: {len(shared_ids)} identidades emparelhadas para pares genuínos.")

        shared_hashes = p_hashes.intersection(r_hashes)
        if len(shared_hashes) > 0:
            severity = self._worse(severity, ValidationSeverity.WARNING)
            messages.append(f"⚠️ Atenção Forense: {len(shared_hashes)} imagens exatas aparecem tanto no Probe quanto no Reference. Isso vai inflar os scores de similaridade artificialmente para 1.0.")

        return severity, messages, p_ids, r_ids, p_hashes, r_hashes

    def _audit_strategy_leakage(self, strategy_id: str, history_stats_list: list, arena_ids: Set[str], arena_hashes: Set[str]) -> StrategyAuditResult:
        """Cruza o 'Passado' da estratégia com o 'Presente' da Arena."""
        audit = StrategyAuditResult()
        
        if not history_stats_list:
            audit.messages.append("ℹ️ Histórico de treino não encontrado (ignorado).")
            return audit

        history_ids: Set[str] = set()
        history_hashes: Set[str] = set()

        for stats in history_stats_list:
            history_ids.update(getattr(stats, "identity_set", set()) or set())
            history_hashes.update(getattr(stats, "hash_set", set()) or set())

        leakage_ids = arena_ids.intersection(history_ids)
        leakage_hashes = arena_hashes.intersection(history_hashes)

        audit.leakage_count_ids = len(leakage_ids)
        audit.leakage_count_hashes = len(leakage_hashes)

        if len(leakage_hashes) > 0:
            audit.is_pure = False
            audit.severity = ValidationSeverity.CRITICAL
            audit.messages.append(f"❌ DATA LEAKAGE CRÍTICO: O modelo memorizou {len(leakage_hashes)} imagens exatas que estão neste teste de validação.")
        
        elif len(leakage_ids) > 0:
            audit.is_pure = False
            audit.severity = ValidationSeverity.WARNING
            audit.messages.append(f"⚠️ IDENTITY LEAKAGE: O modelo já foi exposto a {len(leakage_ids)} identidades que estão na Arena (embora com fotos diferentes). Avaliação de generalização comprometida.")
            
        else:
            audit.messages.append("✅ Modelo puro. 0% de sobreposição com os dados da Arena.")

        return audit

    def _worse(self, s1: ValidationSeverity, s2: ValidationSeverity) -> ValidationSeverity:
        hierarchy = {ValidationSeverity.CRITICAL: 3, ValidationSeverity.WARNING: 2, ValidationSeverity.OK: 1, ValidationSeverity.NEUTRAL: 0}
        return s1 if hierarchy.get(s1, 0) > hierarchy.get(s2, 0) else s2