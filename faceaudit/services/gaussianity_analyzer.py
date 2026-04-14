import numpy as np
from scipy.stats import chi2

from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.gaussianity_analysis_info import GaussianityAnalysisInfo

class GaussianityAnalyzer:
    """
    Serviço forense para avaliar a Normalidade Multivariada do espaço latente.
    """
    
    def compute_mahalanobis_normality(self, dataset: 'EmbeddingDataset') -> GaussianityAnalysisInfo:
        """
        Calcula as distâncias de Mahalanobis ao quadrado para todo o dataset
        e os quantis teóricos correspondentes da distribuição Qui-Quadrado.
        """
        print(f"Iniciando análise de Gaussianidade para: {dataset.name}...")
        
        matrix = dataset.matrix
        n_samples, dimensions = matrix.shape
        
        # 1. Encontra o Centróide Global (Vetor Médio)
        mean_vector = np.mean(matrix, axis=0)
        
        # 2. Centraliza os dados
        centered_matrix = matrix - mean_vector
        
        print("Calculando Matriz de Covariância e Pseudo-inversa...")
        # 3. Calcula a Matriz de Covariância (512x512)
        # rowvar=False indica que as colunas são as variáveis (dimensões)
        cov_matrix = np.cov(matrix, rowvar=False)
        
        # Usamos pinv (Pseudo-inversa) ao invés de inv, pois em 512D 
        # a rede neural quase sempre cria dimensões colineares/redundantes.
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        
        print("Calculando distâncias de Mahalanobis para todas as imagens...")
        # 4. Cálculo vetorizado da Distância de Mahalanobis ao quadrado
        # D^2 = (x - mu)^T * Sigma^-1 * (x - mu)
        left_term = np.dot(centered_matrix, inv_cov_matrix)
        mahalanobis_sq = np.sum(left_term * centered_matrix, axis=1)
        
        # Ordena as distâncias reais do menor para o maior
        mahalanobis_sq_sorted = np.sort(mahalanobis_sq)
        
        # 5. Calcula os Quantis Teóricos da Qui-Quadrado (Chi-Square)
        # Se os dados forem normais, as distâncias Mahalanobis ao quadrado 
        # devem seguir uma distribuição Chi^2 com graus de liberdade = dimensões.
        percentiles = (np.arange(1, n_samples + 1) - 0.5) / n_samples
        theoretical_quantiles = chi2.ppf(percentiles, df=dimensions)
        
        # 6. Métrica de Saúde da Nuvem (Regra Empírica dos 95%)
        # Limite teórico onde 95% dos dados deveriam estar
        threshold_95 = chi2.ppf(0.95, df=dimensions)
        pct_inside = (np.sum(mahalanobis_sq <= threshold_95) / n_samples) * 100
        
        print("Análise concluída!")
        
        return GaussianityAnalysisInfo(
            dataset_name=dataset.name,
            dimensions=dimensions,
            sample_size=n_samples,
            mahalanobis_sq_distances=mahalanobis_sq_sorted,
            theoretical_chi2_quantiles=theoretical_quantiles,
            pct_inside_95_ellipse=pct_inside
        )