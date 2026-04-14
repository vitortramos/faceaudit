# faceaudit/models/plda_validation_result.py
from dataclasses import dataclass, field

from typing import List, Dict, Any

from faceaudit.enums.validation_severity import ValidationSeverity



@dataclass
class PLDAValidationResult:
    """
    Data Transfer Object (DTO) que contém o diagnóstico completo da viabilidade 
    do treinamento do modelo PLDA.
    
    Agrega métricas, mensagens de erro/aviso e níveis de severidade para diferentes
    aspectos da validação:
    1. Dados (Overlap, Leakage).
    2. Pipelines (Consistência Lógica, Regras de Adaptação).
    3. PLDA (Estabilidade Matemática, Matriz Singular).
    4. Calibração (Densidade de Imagens).
    """
    
    # --- Status Geral ---
    # Representa o pior cenário entre todas as verificações.
    # Se qualquer sub-validação for CRITICAL, este campo será CRITICAL.
    severity: ValidationSeverity
    
    # --- Status Específico dos Dados (Grupo 2) ---
    # Indica se os dados passaram no teste de Overlap/Leakage.
    # Permite pintar o label de dados independentemente dos outros erros.
    data_severity: ValidationSeverity = ValidationSeverity.NEUTRAL
    
    # --- Status Específico dos Pipelines (Grupo 3) ---
    # [NOVO] Indica se a configuração dos pipelines (Source/Target) obedece 
    # às regras da estratégia (ex: Exige CORAL, Proíbe LDA, Simetria).
    pipeline_severity: ValidationSeverity = ValidationSeverity.NEUTRAL
    pipeline_message: str = "" # Mensagem específica para o label dentro do grupo de pipelines
    
    # --- Diagnósticos PLDA e Calibração (Grupo 4) ---
    # Mensagens e status dedicados para os labels de feedback na interface.
    
    plda_message: str = "Aguardando dados..."
    plda_severity: ValidationSeverity = ValidationSeverity.NEUTRAL
    
    calib_message: str = "Aguardando dados..."
    calib_severity: ValidationSeverity = ValidationSeverity.NEUTRAL
    

    # --- [NOVO] Pós-preparação (split real) ---
    post_prep_message: str = "Aguardando preparação..."
    post_prep_severity: ValidationSeverity = ValidationSeverity.NEUTRAL
    
    # --- Métricas Calculadas ---
    # Dicionário para armazenar valores numéricos relevantes.
    # Chaves esperadas:
    # - 'overlap': (int) Quantidade de identidades na interseção Source/Target.
    # - 'plda_ratio': (float) Razão Amostras / Parâmetros Livres (Estabilidade).
    # - 'calib_density': (float) Média de imagens por identidade no Target.
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # --- Mensagens de Contexto (Dados) ---
    # Lista de strings com avisos detalhados (Geralmente focada em Dados/Overlap).
    messages: List[str] = field(default_factory=list)
    
    # --- Helpers de Interface (UI) ---
    
    def get_color(self, severity: ValidationSeverity) -> str:
        """
        Retorna o código HEX da cor de fundo (Background) apropriada para o nível de severidade.
        Útil para estilizar componentes PyQt (Labels, Frames).
        """
        if severity == ValidationSeverity.CRITICAL:
            return "#f8d7da" # Vermelho claro (Bootstrap Danger)
        elif severity == ValidationSeverity.WARNING:
            return "#fff3cd" # Amarelo claro (Bootstrap Warning)
        elif severity == ValidationSeverity.OK:
            return "#d4edda" # Verde claro (Bootstrap Success)
        return "#e9ecef"     # Cinza claro (Bootstrap Light/Secondary)

    def get_text_color(self, severity: ValidationSeverity) -> str:
        """
        Retorna o código HEX da cor do texto (Foreground) para garantir contraste.
        """
        if severity == ValidationSeverity.CRITICAL:
            return "#721c24" # Vermelho escuro
        elif severity == ValidationSeverity.WARNING:
            return "#856404" # Amarelo escuro
        elif severity == ValidationSeverity.OK:
            return "#155724" # Verde escuro
        return "#6c757d"     # Cinza escuro