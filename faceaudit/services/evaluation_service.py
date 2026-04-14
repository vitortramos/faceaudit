import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from typing import List, Tuple

# Imports dos Modelos
from faceaudit.models import StrategyValidationResult, ValidationAnalysisData

class EvaluationService:
    """
    Serviço responsável por calcular métricas de performance forense (EER, Cllr, AUC)
    e consolidar os dados para análise visual.
    
    Este serviço não executa modelos. Ele apenas avalia os resultados produzidos por eles.
    """

    def evaluate(self, 
                 lrs: np.ndarray, 
                 ground_truth: np.ndarray, 
                 scores: np.ndarray,
                 strategy_name: str,
                 dataset_name: str) -> StrategyValidationResult:
        """
        Calcula todas as métricas forenses e retorna o objeto de relatório consolidado.

        Args:
            lrs: Array de Likelihood Ratios calculados.
            ground_truth: Array booleano (True=Mesma Pessoa/Hp, False=Diferente/Hd).
            scores: Array de scores brutos (apenas para plotar histogramas de separação).
            strategy_name: Nome da estratégia avaliada.
            dataset_name: Nome do dataset de teste usado.
        """
        # 1. Separação de Classes (Genuínos vs Impostores)
        # Hp: Hipótese do Promotor (Mesma pessoa) -> ground_truth == True
        # Hd: Hipótese da Defesa (Pessoas diferentes) -> ground_truth == False
        mask_gen = (ground_truth == True)
        mask_imp = (ground_truth == False)

        gen_lrs = lrs[mask_gen]
        imp_lrs = lrs[mask_imp]
        
        gen_scores = scores[mask_gen]
        imp_scores = scores[mask_imp]

        if len(gen_lrs) == 0 or len(imp_lrs) == 0:
            raise ValueError("Dados insuficientes: é necessário ter ambas as classes (Genuínos e Impostores).")

        # 2. Cálculo de Métricas Escalares
        eer, fpr, fnr = self._calculate_eer(ground_truth, scores) # EER usa scores brutos ou LRs (dá na mesma se monótono)
        auc = roc_auc_score(ground_truth, scores)
        cllr = self._calculate_cllr(gen_lrs, imp_lrs)
        cllr_min = self._calculate_min_cllr(gen_lrs, imp_lrs) # Opcional: Implementar PAV algorithm se precisar precisão extrema

        # 3. Montagem do DTO de Dados (ValidationAnalysisData)
        analysis_data = ValidationAnalysisData(
            gen_lrs=gen_lrs,
            imp_lrs=imp_lrs,
            gen_scores=gen_scores,
            imp_scores=imp_scores,
            det_fpr=fpr,
            det_fnr=fnr
        )

        # 4. Criação do Resultado Final
        return StrategyValidationResult(
            strategy_name=strategy_name,
            test_dataset_name=dataset_name,
            n_genuines=len(gen_lrs),
            n_impostors=len(imp_lrs),
            eer=eer,
            cllr=cllr,
            cllr_min=cllr_min,
            auc=auc,
            analysis_data=analysis_data
        )

    # =========================================================================
    # MÉTODOS MATEMÁTICOS INTERNOS
    # =========================================================================

    def _calculate_eer(self, y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Calcula o Equal Error Rate (ponto onde FAR = FRR)."""
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        fnr = 1 - tpr
        
        # Encontra o índice onde a diferença entre fnr e fpr é mínima
        idx_eer = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[idx_eer]
        
        return eer, fpr, fnr

    def _calculate_cllr(self, gen_lrs: np.ndarray, imp_lrs: np.ndarray) -> float:
        """
        Calcula o Cllr (Log-Likelihood Ratio Cost).
        Métrica padrão para avaliar a calibração de sistemas de LR.
        
        Fórmula: 
          Cllr = 0.5 * (mean(log2(1 + 1/LR_gen)) + mean(log2(1 + LR_imp)))
        """
        # Proteção numérica (epsilon) e conversão para Log2
        # LR_gen deve ser alto -> 1/LR_gen deve ser pequeno -> log deve ser pequeno.
        # LR_imp deve ser baixo -> LR_imp pequeno -> log pequeno.
        
        # Termo 1: Genuínos (Quanto maior o LR, melhor)
        # log2(1 + 1/LR)
        term_gen = np.log2(1 + (1 / (gen_lrs + 1e-15)))
        
        # Termo 2: Impostores (Quanto menor o LR, melhor)
        # log2(1 + LR)
        term_imp = np.log2(1 + imp_lrs)
        
        cllr = 0.5 * (np.mean(term_gen) + np.mean(term_imp))
        return float(cllr)

    def _calculate_min_cllr(self, gen_lrs: np.ndarray, imp_lrs: np.ndarray) -> float:
        """
        Calcula o Cllr Mínimo (minCllr).
        Isso representa a performance discriminativa "pura", assumindo calibração perfeita.
        
        Para calcular o minCllr exato, precisaríamos do algoritmo PAV (Pool Adjacent Violators).
        Como é complexo implementar do zero sem libs externas (como 'lir'), 
        aqui usaremos uma aproximação ou retornaremos o próprio cllr se não quisermos adicionar complexidade agora.
        
        *Por enquanto, retornamos 0.0 ou o próprio cllr como placeholder se não tivermos PAV.*
        *Se você tiver a lib 'lir' instalada, podemos usar lir.metrics.cllr_min*
        """
        # Placeholder para manter o código puro Python sem dependências obscuras.
        # Em produção, recomenda-se usar a lib 'li-metrics' ou similar.
        return 0.0