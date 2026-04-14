import pickle
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

# Imports do Projeto
from faceaudit.protocols.lr_model_protocol import LRModelProtocol
from faceaudit.models import EmbeddingInfo, CaseResult, StrategyValidationResult, EmbeddingDataset

class CaseworkPipeline:
    """
    Pipeline Operacional para Análise de Casos Forenses.
    
    Este pipeline atua como o orquestrador final do processo de verificação, 
    cruzando dados biométricos (Probe vs Reference) contra modelos estatísticos treinados (Estratégias).

    ### Responsabilidades
    1. **Gestão de Recursos:** Carregar e gerenciar modelos (PLDA, KDE) e relatórios de validação.
    2. **Processamento em Lote (Batch):** Executar confrontos 1:N ou N:M para testes em massa.
    3. **Processamento Unitário (Case):** Executar confronto 1:1 para casos reais.
    4. **Padronização:** Garantir que todos os resultados sigam o formato `CaseResult`.
    """

    def __init__(self):
        # Armazena as instâncias carregadas em memória {nome: objeto}
        self.strategies: Dict[str, LRModelProtocol] = {}
        self.validations: Dict[str, StrategyValidationResult] = {}
        self.is_loaded = False

    def load_resources(self, 
                       models_folder: str, 
                       validations_folder: str, 
                       strategy_templates: List[LRModelProtocol]):
        """
        Carrega os artefatos gerados pelos pipelines de Treino e Validação a partir do disco.
        Ideal para ambiente de produção ou inferência.

        Args:
            models_folder (str): Caminho para a pasta contendo os arquivos .pkl dos modelos.
            validations_folder (str): Caminho para a pasta contendo os relatórios de validação.
            strategy_templates (List): Lista de instâncias "vazias" das estratégias para saber o que carregar.
        """
        print("\n=== [Casework] Carregando Recursos Forenses ===")
        m_path = Path(models_folder)
        v_path = Path(validations_folder)

        if not m_path.exists(): 
            print(f"⚠️ Pasta {models_folder} não encontrada (ignorando se estiver rodando em memória).")

        for template in strategy_templates:
            name = template.name
            
            # 1. Carrega Modelo (O cérebro)
            model_file = m_path / f"{name}.pkl"
            if model_file.exists():
                template.load(str(model_file))
                self.strategies[name] = template
                print(f" >> Modelo carregado: {name}")
                
                # 2. Carrega Validação (O contexto) - Opcional
                val_file = v_path / f"{name}_validation.pkl"
                if val_file.exists():
                    with open(val_file, 'rb') as f:
                        self.validations[name] = pickle.load(f)
            else:
                print(f" ⚠️ Modelo {name} não encontrado no disco.")

        if self.strategies:
            self.is_loaded = True
            print("=== Recursos Carregados. Pronto para Análise. ===")

    def execute(self, 
                strategies: List[LRModelProtocol], 
                probes: EmbeddingDataset, 
                references: EmbeddingDataset) -> Dict[str, List[CaseResult]]:
        """
        Executa a comparação em lote (Batch Processing) para Notebooks e Testes.
        
        Permite injetar estratégias já treinadas em memória (sem carregar do disco),
        facilitando o ciclo de experimentação rápida.

        Args:
            strategies (List): Lista de estratégias (PLDA, KDE) já treinadas (.fit executado).
            probes (EmbeddingDataset): Dataset de imagens questionadas.
            references (EmbeddingDataset): Dataset de imagens padrão.

        Returns:
            Dict[str, List[CaseResult]]: Dicionário agrupado por estratégia contendo os laudos.
        """
        results: Dict[str, List[CaseResult]] = {}
        
        # 1. Injeta as estratégias recebidas na memória da instância
        # Isso permite usar o pipeline mesmo sem chamar load_resources()
        for strat in strategies:
            self.strategies[strat.name] = strat
        
        self.is_loaded = True 
        
        print(f"⚔️ Iniciando confronto em lote: {len(probes)} Probes x {len(references)} References")

        for name, strategy in self.strategies.items():
            print(f"   >> Executando estratégia: {name}...")
            batch_results = []
            
            # Loop NxM (Todos contra Todos)
            for probe_item in probes.items:
                for ref_item in references.items:
                    try:
                        # Reutiliza a lógica padronizada do protocolo .compare()
                        res = strategy.compare(probe=probe_item, reference=ref_item)
                        batch_results.append(res)
                            
                    except Exception as e:
                        print(f"      ❌ Erro em {probe_item.file_name} vs {ref_item.file_name}: {e}")
            
            results[name] = batch_results
            print(f"      ✅ {len(batch_results)} comparações realizadas.")
            
        return results

    def process_case(self, 
                     probe: EmbeddingInfo, 
                     reference: EmbeddingInfo, 
                     plot_context: bool = True) -> Dict[str, CaseResult]:
        """
        Processa um caso forense unitário (1:1).
        
        Args:
            probe: Imagem questionada.
            reference: Imagem padrão.
            plot_context: Se True, tenta gerar gráficos de contexto usando dados de validação.

        Returns:
            Dict[str, CaseResult]: Resultados de todas as estratégias carregadas para este par.
        """
        if not self.is_loaded:
            raise RuntimeError("Pipeline não inicializado. Carregue recursos ou execute inject.")

        results = {}
        print(f"\n🔎 Analisando Caso: {probe.file_name} vs {reference.file_name}")

        for name, strategy in self.strategies.items():
            try:
                # 1. Executa Comparação
                case_result = strategy.compare(probe, reference)
                results[name] = case_result
                
                # 2. Plotagem de Contexto (Onde este caso cai na curva de Tippett?)
                if plot_context and name in self.validations:
                    val_result = self.validations[name]
                    title = f"Contexto ({name})\nLR={case_result.lr:.1f} - {case_result.verbal_support}"
                    val_result.plot_case_context(case_result, title=title)
            
            except Exception as e:
                print(f" ❌ Erro na estratégia {name}: {e}")

        return results