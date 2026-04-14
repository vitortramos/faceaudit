import numpy as np
from typing import List, Tuple
from sklearn.metrics import roc_curve, auc
from sklearn.isotonic import IsotonicRegression

from faceaudit.models.validation_metrics import ValidationMetrics
from faceaudit.models.validation_plot_data import ValidationPlotData
from faceaudit.models.strategy_validation_result import StrategyValidationResult

class ValidationMathService:
    """
    Serviço matemático puro para cálculo de métricas biométricas e forenses.
    
    Esta classe opera como um motor agnóstico. Ela não sabe se os dados vieram
    de um modelo PLDA, KDE ou de um pipeline de Deep Learning. Sua única
    responsabilidade é receber arrays de scores (tratados como Log-Likelihood Ratios
    na base neperiana 'e') e extrair métricas de performance rigorosas exigidas
    pela ENFSI (European Network of Forensic Science Institutes).
    """

    def calculate_metrics(self, 
                          strategy_id: str, 
                          strategy_name: str, 
                          strategy_type: str, 
                          genuine_scores: List[float], 
                          impostor_scores: List[float]) -> StrategyValidationResult:
        """
        Orquestra o cálculo de todas as métricas de validação e gera os vetores
        necessários para a plotagem dos gráficos de desempenho.

        Args:
            strategy_id (str): Identificador único da estratégia avaliada.
            strategy_name (str): Nome amigável de exibição da estratégia.
            strategy_type (str): Tipo da estratégia (ex: KDE, PLDA).
            genuine_scores (List[float]): Lista de Log-LRs de pares da mesma pessoa.
            impostor_scores (List[float]): Lista de Log-LRs de pares de pessoas diferentes.

        Returns:
            StrategyValidationResult: DTO contendo o pacote completo de métricas
            (EER, Cllr, min_Cllr, AUC) e dados para gráficos (DET, Tippett, ECE).

        Raises:
            ValueError: Se as listas de scores fornecidas estiverem vazias.
        """
        if not genuine_scores or not impostor_scores:
            raise ValueError("As listas de scores genuínos e impostores não podem estar vazias.")

        gen_arr = np.array(genuine_scores, dtype=np.float64)
        imp_arr = np.array(impostor_scores, dtype=np.float64)
        
        # 1. Arrays para Curva ROC e DET
        y_true = np.concatenate([np.ones_like(gen_arr), np.zeros_like(imp_arr)])
        y_scores = np.concatenate([gen_arr, imp_arr])
        
        # fpr = False Positive Rate (Falso Aceite) | tpr = True Positive Rate (Verdadeiro Aceite)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr  # Falso Rejeite
        roc_auc = float(auc(fpr, tpr))
        
        # 2. Cálculo do EER (Equal Error Rate)
        # Encontra o ponto de intersecção mais próximo entre a Falsa Rejeição e o Falso Aceite.
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer_value = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        eer_thresh = float(thresholds[eer_idx])

        # 3. Cálculo do Cllr (Desempenho Forense Real)
        # Penaliza duramente o modelo por dar certezas altas em decisões erradas.
        cllr_value = self._calculate_cllr(gen_arr, imp_arr)
        
        # 4. Calibração PAVA (Regressão Isotônica) e Min Cllr
        # Calcula qual seria o Cllr perfeito se o modelo estivesse otimamente calibrado.
        gen_cal, imp_cal = self._get_calibrated_llrs(gen_arr, imp_arr)
        min_cllr_value = self._calculate_cllr(gen_cal, imp_cal)

        # 5. Vetores para o Gráfico de Calibração (ECE)
        # Varia as chances a priori (Prior Odds) para mapear a perda de entropia ao longo de diferentes cenários.
        ece_x, ece_y, ece_min_y = self._calculate_ece_data(gen_arr, imp_arr, gen_cal, imp_cal)

        # Montagem do DTO de Métricas Escalares
        metrics = ValidationMetrics(
            eer=eer_value,
            eer_threshold=eer_thresh,
            cllr=cllr_value,
            cllr_min=min_cllr_value,
            auc=roc_auc
        )
        
        # Montagem do DTO Vetorial para Plotagem
        plot_data = ValidationPlotData(
            genuine_log_lrs=gen_arr,
            impostor_log_lrs=imp_arr,
            fpr_array=fpr,
            fnr_array=fnr,
            ece_prior_log_odds=ece_x,
            ece_values=ece_y,
            ece_min_values=ece_min_y
        )
        
        return StrategyValidationResult(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            n_genuines=len(gen_arr),
            n_impostors=len(imp_arr),
            metrics=metrics,
            plot_data=plot_data
        )

    def _calculate_cllr(self, gen_scores: np.ndarray, imp_scores: np.ndarray) -> float:
        """
        Calcula o Cost of Log-Likelihood Ratio (Cllr) empírico em Bits.
        
        O Cllr mede o custo informacional das decisões do sistema, avaliando não
        apenas se o sistema acerta, mas o grau de calibração do peso da evidência.
        
        Matematicamente equivalente a:
        Cllr = 0.5 * [ mean(log2(1 + e^(-s_gen))) + mean(log2(1 + e^(s_imp))) ]
        
        Utiliza `np.logaddexp(0, x)` (que equivale a ln(1 + exp(x))) para garantir
        estabilidade numérica e evitar overflow com LRs muito grandes.

        Args:
            gen_scores (np.ndarray): Log-LRs dos pares genuínos.
            imp_scores (np.ndarray): Log-LRs dos pares impostores.

        Returns:
            float: O valor do Cllr. Um modelo inútil terá Cllr próximo de 1.0.
        """
        # Divide por log(2) no final para converter a Entropia de base 'e' (Nats) para base '2' (Bits).
        term_gen = np.mean(np.logaddexp(0, -gen_scores)) / np.log(2)
        term_imp = np.mean(np.logaddexp(0, imp_scores)) / np.log(2)
        
        return float(0.5 * (term_gen + term_imp))

    def _calculate_min_cllr(self, gen_scores: np.ndarray, imp_scores: np.ndarray) -> float:
        """
        Calcula o Cllr mínimo possível (poder de discriminação isolado) removendo
        qualquer penalidade causada por má calibração do modelo.

        Args:
            gen_scores (np.ndarray): Log-LRs dos pares genuínos.
            imp_scores (np.ndarray): Log-LRs dos pares impostores.

        Returns:
            float: O valor ótimo do Cllr após aplicação de Regressão Isotônica.
        """
        gen_cal_llrs, imp_cal_llrs = self._get_calibrated_llrs(gen_scores, imp_scores)
        return self._calculate_cllr(gen_cal_llrs, imp_cal_llrs)

    def _get_calibrated_llrs(self, gen_scores: np.ndarray, imp_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica o algoritmo PAVA (Pool Adjacent Violators Algorithm) para extrair
        os Log-LRs otimamente calibrados.
        
        O PAVA assegura que as probabilidades de saída sejam monotonamente crescentes
        em relação aos scores de entrada, corrigindo problemas onde um modelo
        dá certezas exageradas em regiões esparsas da distribuição.

        Args:
            gen_scores (np.ndarray): Log-LRs reais dos pares genuínos.
            imp_scores (np.ndarray): Log-LRs reais dos pares impostores.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (gen_calibrated, imp_calibrated) Log-LRs após PAVA.
        """
        y_true = np.concatenate([np.ones_like(gen_scores), np.zeros_like(imp_scores)])
        y_scores = np.concatenate([gen_scores, imp_scores])
        
        # 1. Regressão Isotônica mapeia os scores para uma probabilidade [0, 1] monotonamente crescente.
        ir = IsotonicRegression(out_of_bounds='clip')
        p_calibrated = ir.fit_transform(y_scores, y_true)
        
        # Segurança contra div/0 no cálculo logarítmico a seguir.
        epsilon = 1e-14
        p_calibrated = np.clip(p_calibrated, epsilon, 1 - epsilon)
        
        # 2. Converte Probabilidade de volta para Log-Odds.
        # Subtraímos as odds a priori do dataset empírico para isolar a Razão de Verossimilhança (LR) pura.
        prior_log_odds = np.log(len(gen_scores) / len(imp_scores))
        calibrated_llrs = np.log(p_calibrated / (1 - p_calibrated)) - prior_log_odds
        
        return calibrated_llrs[:len(gen_scores)], calibrated_llrs[len(gen_scores):]

    def _calculate_ece_data(self, gen_scores: np.ndarray, imp_scores: np.ndarray, 
                            gen_cal: np.ndarray, imp_cal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula os vetores para o gráfico de Empirical Cross-Entropy (ECE).
        
        O gráfico ECE avalia o desempenho do modelo simulando um vasto espectro
        de cenários forenses, variando artificialmente as chances a priori (Prior Odds)
        de -3 a +3 na base Log10. Se a curva do modelo real se distanciar muito
        da curva mínima (PAVA), significa que o modelo possui uma alta Perda de Calibração.

        Args:
            gen_scores (np.ndarray): LRs Reais dos Genuínos.
            imp_scores (np.ndarray): LRs Reais dos Impostores.
            gen_cal (np.ndarray): LRs Calibrados (PAVA) dos Genuínos.
            imp_cal (np.ndarray): LRs Calibrados (PAVA) dos Impostores.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            (Eixo X Log10_Odds, Eixo Y ECE_Real, Eixo Y ECE_Min)
        """
        # Define o espectro do Eixo X: variação dos Prior Odds (Log base 10)
        prior_log10_odds = np.linspace(-3.5, 3.5, 100)
        
        # Converte o X para base neperiana para uso na fórmula da Entropia Sigmoidal
        prior_ln_odds = prior_log10_odds * np.log(10)
        
        # Calcula a proporção (Probabilidade P) associada a cada Prior Odds
        P = 1 / (1 + np.exp(-prior_ln_odds))
        
        # Usamos broadcasting do Numpy para calcular a matriz inteira (100 cenários x N scores) simultaneamente.
        # ece_real: Combina as penalidades genuínas e impostoras reais ponderadas pela Probabilidade (P).
        gen_term_real = np.mean(np.logaddexp(0, -(prior_ln_odds[:, None] + gen_scores[None, :])), axis=1) / np.log(2)
        imp_term_real = np.mean(np.logaddexp(0, prior_ln_odds[:, None] + imp_scores[None, :]), axis=1) / np.log(2)
        ece_real = P * gen_term_real + (1 - P) * imp_term_real

        # ece_min: Repete o cálculo usando os dados perfeitamente calibrados pelo PAVA.
        gen_term_min = np.mean(np.logaddexp(0, -(prior_ln_odds[:, None] + gen_cal[None, :])), axis=1) / np.log(2)
        imp_term_min = np.mean(np.logaddexp(0, prior_ln_odds[:, None] + imp_cal[None, :]), axis=1) / np.log(2)
        ece_min = P * gen_term_min + (1 - P) * imp_term_min

        return prior_log10_odds, ece_real, ece_min