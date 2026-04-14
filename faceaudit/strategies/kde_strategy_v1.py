import numpy as np
import pickle
from typing import Optional, Dict, Any
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

# Imports do Projeto
from faceaudit.protocols.lr_model_protocol import LRModelProtocol
from faceaudit.models import EmbeddingDataset, EmbeddingInfo, CaseResult, FaceInfo
from faceaudit.services import ComparisonService

class KDEStrategy:
    """
    Estratégia de cálculo de Likelihood Ratio (LR) baseada em KDE (Kernel Density Estimation).
    
    Esta estratégia modela a distribuição de scores de similaridade (Cosseno) para 
    pares genuínos e impostores usando métodos não-paramétricos (Gaussian Kernels).
    
    ### Fundamentação Teórica
    A LR é calculada como a razão das densidades de probabilidade:
    
    $$ LR(s) = \\frac{P(s | H_p)}{P(s | H_d)} $$
    
    Onde:
    - $s$: Score de similaridade (Cosseno).
    - $H_p$: Hipótese do Promotor (Mesma pessoa / Genuíno).
    - $H_d$: Hipótese da Defesa (Pessoas diferentes / Impostor).
    - $P(s|H)$: Densidade estimada via soma de Kernels Gaussianos centrados nos dados de treino.
    
    ### Otimizações de Performance
    - **Downsampling:** Para datasets massivos (milhões de pares), o método `fit` realiza uma 
      sub-amostragem aleatória (ex: max 100k pontos) para manter a inferência rápida sem 
      perda significativa de precisão estatística.
    - **Bandwidth Search:** Utiliza Grid Search com Validação Cruzada para encontrar o 
      fator de suavização ($h$) ideal.
    """

    def __init__(self, comparison_service: ComparisonService):
        """
        Inicializa a estratégia KDE.

        Args:
            comparison_service (ComparisonService): Serviço injetado responsável por 
                calcular a similaridade geométrica (Cosseno) de forma otimizada durante o treino.
        """
        self._name = "KDE_Cosine"
        self.comp_svc = comparison_service
        
        # Modelos internos do SciKit-Learn
        self.kde_gen: Optional[KernelDensity] = None
        self.kde_imp: Optional[KernelDensity] = None
        
        # Metadados para auditoria e rastreabilidade
        self.metadata: Dict[str, Any] = {}
        self.fitted = False

    @property
    def name(self) -> str:
        """Retorna o identificador único da estratégia (ex: 'KDE_Cosine')."""
        return self._name

    def fit(self, 
            train_dataset: Optional[EmbeddingDataset], 
            calibration_dataset: EmbeddingDataset) -> None:
        """
        Treina as densidades de probabilidade com OTIMIZAÇÃO DE AMOSTRAGEM.

        Este método gera milhões de scores de comparação, mas utiliza apenas um subconjunto
        representativo para construir o modelo KDE, garantindo performance na validação.

        Args:
            train_dataset: Ignorado por esta estratégia (KDE usa apenas scores, não vetores latentes).
            calibration_dataset: Dataset contendo as amostras para gerar as distribuições $H_p$ e $H_d$.

        Raises:
            ValueError: Se o dataset não tiver classes suficientes (min 5 pares por classe).
        """
        if train_dataset is not None:
             pass # KDE trabalha apenas no espaço de scores

        print(f"[{self.name}] Iniciando treino. Gerando scores para {len(calibration_dataset)} identidades...")

        # 1. Gera Matriz de Scores (NxN) via Serviço
        ds_scores = self.comp_svc.process_comparison(
            calibration_dataset, 
            calibration_dataset,
            comparison_name="KDE_Calibration_Fit"
        )
        
        df = ds_scores.to_pandas()
        
        # 2. Limpeza (Remove a diagonal principal e scores perfeitos artificiais)
        # Evita que a densidade de genuínos tenha um pico artificial em 1.0 (auto-comparação)
        if 'face_hash_1' in df.columns and df['face_hash_1'].notna().all():
             df = df[df['face_hash_1'] != df['face_hash_2']]
        else:
             df = df[df['cosine_similarity'] < 0.999999]

        if df.empty:
            raise ValueError(f"[{self.name}] Erro: Dataset de scores vazio após limpeza.")

        # 3. Separação de Classes (Genuínos vs Impostores)
        gen_scores = df[df['same_person'] == True]['cosine_similarity'].values.reshape(-1, 1)
        imp_scores = df[df['same_person'] == False]['cosine_similarity'].values.reshape(-1, 1)

        # === OTIMIZAÇÃO DE PERFORMANCE (DOWNSAMPLING) ===
        # O KDE "lembra" de cada ponto de treino. Se usarmos 7 milhões de impostores,
        # cada predição futura terá custo O(7.000.000). 
        # Limitamos a 100.000 pontos, o que desenha a mesma curva com custo O(100.000).
        MAX_SAMPLES = 100000
        rng = np.random.default_rng(42)

        if len(imp_scores) > MAX_SAMPLES:
            print(f"[{self.name}] ⚠️ Sub-amostrando Impostores: {len(imp_scores)} -> {MAX_SAMPLES} para performance.")
            imp_scores = rng.choice(imp_scores, size=MAX_SAMPLES, replace=False)
            
        if len(gen_scores) > MAX_SAMPLES:
            print(f"[{self.name}] ⚠️ Sub-amostrando Genuínos: {len(gen_scores)} -> {MAX_SAMPLES} para performance.")
            gen_scores = rng.choice(gen_scores, size=MAX_SAMPLES, replace=False)
        # =================================================

        # Validação estatística mínima pós-amostragem
        if len(gen_scores) < 5 or len(imp_scores) < 5:
            raise ValueError(f"[{self.name}] Dados insuficientes (Min 5 pares). Gen: {len(gen_scores)}, Imp: {len(imp_scores)}")

        # 4. Otimização e Ajuste (Fit)
        print(f"[{self.name}] Otimizando Bandwidth (Grid Search)...")
        
        bw_gen = self._optimize_bandwidth(gen_scores)
        bw_imp = self._optimize_bandwidth(imp_scores)

        print(f"[{self.name}] Ajustando modelos KDE finais...")
        self.kde_gen = KernelDensity(kernel='gaussian', bandwidth=bw_gen).fit(gen_scores)
        self.kde_imp = KernelDensity(kernel='gaussian', bandwidth=bw_imp).fit(imp_scores)
        
        # Salva metadados úteis para o relatório
        self.metadata = {
            'n_genuines_fit': len(gen_scores), 
            'n_impostors_fit': len(imp_scores),
            'bw_gen': bw_gen, 
            'bw_imp': bw_imp
        }
        
        self.fitted = True
        print(f"[{self.name}] ✅ Treinado com sucesso. BW: Gen={bw_gen:.4f}, Imp={bw_imp:.4f}")

    def compare(self, probe: FaceInfo, reference: FaceInfo) -> CaseResult:
        """
        Realiza a comparação forense completa e retorna o objeto estruturado.
        
        Este método atua como um wrapper para o `predict`, adicionando contexto
        semântico (Suporte Verbal) e estruturando o resultado para o pipeline.

        Args:
            probe: Informações da face questionada.
            reference: Informações da face padrão.

        Returns:
            CaseResult: Objeto contendo Score, LR e Interpretação Verbal.
        """
        # 1. Calcula LR
        lr = self.predict(probe, reference)
        
        # 2. Calcula Score Bruto (Cosseno) apenas para registro no laudo
        # Assumindo vetores normalizados, produto escalar = cosseno
        v1 = probe.embedding
        v2 = reference.embedding
        raw_score = float(np.dot(v1, v2))

        # 3. Define Suporte Verbal
        verbal = self._get_verbal_support(lr)

        return CaseResult(
            strategy_name=self.name,
            probe=probe,
            reference=reference,
            score=raw_score,
            lr=float(lr),
            verbal_support=verbal,
            ground_truth=None
        )

    def predict(self, probe: EmbeddingInfo, reference: EmbeddingInfo) -> float:
        """
        Calcula a Likelihood Ratio (LR) numérica para um par de embeddings.

        Args:
            probe: Embedding da imagem questionada.
            reference: Embedding da imagem padrão.

        Returns:
            float: LR calculada ($P(E|H_p) / P(E|H_d)$).
        
        Raises:
            RuntimeError: Se chamado antes de `fit()`.
        """
        if not self.fitted:
            raise RuntimeError(f"[{self.name}] Estratégia não treinada. Execute fit() primeiro.")

        # 1. Cálculo Geométrico
        v1 = probe.embedding
        v2 = reference.embedding
        score = np.dot(v1, v2) 
        
        # 2. Avaliação Estatística (Log-Densidade)
        # score_samples retorna ln(pdf)
        log_hp = self.kde_gen.score_samples([[score]])[0]
        log_hd = self.kde_imp.score_samples([[score]])[0]

        # 3. Conversão para Probabilidade e LR
        prob_hp = np.exp(log_hp)
        prob_hd = np.exp(log_hd)
        
        # Proteção contra divisão por zero (LR -> Infinito)
        if prob_hd < 1e-15: prob_hd = 1e-15
            
        return float(prob_hp / prob_hd)

    def save(self, path: str) -> None:
        """Persiste o estado completo da estratégia (modelos KDE e metadados) em disco."""
        state = {
            'kde_gen': self.kde_gen, 
            'kde_imp': self.kde_imp,
            'fitted': self.fitted, 
            'metadata': self.metadata
        }
        with open(path, 'wb') as f: pickle.dump(state, f)
        print(f"[{self.name}] Modelo salvo em {path}")

    def load(self, path: str) -> None:
        """Restaura o estado da estratégia a partir do disco."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.kde_gen = state['kde_gen']
            self.kde_imp = state['kde_imp']
            self.fitted = state.get('fitted', True)
            self.metadata = state.get('metadata', {})
        print(f"[{self.name}] Modelo carregado.")

    # --- Métodos Auxiliares ---

    def _optimize_bandwidth(self, data: np.ndarray) -> float:
        """Encontra o melhor bandwidth ($h$) usando Grid Search (Sub-amostrado se N > 2000)."""
        sample = data
        if len(data) > 2000:
            rng = np.random.default_rng(42)
            sample = data[rng.choice(len(data), 2000, replace=False)]
            
        params = {'bandwidth': np.linspace(0.01, 0.5, 20)}
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, cv=3)
        grid.fit(sample)
        return grid.best_params_['bandwidth']

    def _get_verbal_support(self, lr: float) -> str:
        """Converte o LR numérico em uma escala verbal (ENFSI/SWGDE)."""
        if lr < 1: return "Suporte para Diferentes Pessoas"
        if lr < 10: return "Suporte Fraco"
        if lr < 100: return "Suporte Moderado"
        if lr < 1000: return "Suporte Moderadamente Forte"
        if lr < 10000: return "Suporte Forte"
        return "Suporte Muito Forte"