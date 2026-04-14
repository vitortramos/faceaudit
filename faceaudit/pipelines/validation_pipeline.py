import pickle
import random
import numpy as np
from typing import List
from pathlib import Path

# Imports do Projeto
from faceaudit.protocols.lr_model_protocol import LRModelProtocol
from faceaudit.models import EmbeddingComparisonDataset
from faceaudit.services.evaluation_service import EvaluationService

class ValidationPipeline:
    """
    Orquestra a validação e auditoria da performance de múltiplas estratégias de calibração
    (modelos LRModelProtocol) em um dataset de comparação de embeddings.

    O pipeline segue um fluxo estruturado para garantir que cada estratégia seja
    avaliada de forma consistente e que os resultados sejam persistidos para
    análise posterior.

    Fluxo de Execução:
    1.  **Preparação do Dataset**: Recebe um `EmbeddingComparisonDataset` que pode
        conter pares desbalanceados (N x N) de comparações.
    2.  **Balanceamento (Opcional)**: Se configurado, o dataset de pares é balanceado
        para ter uma proporção 1:1 de pares genuínos e impostores. Isso é crucial
        para acelerar a validação em datasets muito grandes, reduzindo a carga
        computacional da inferência.
    3.  **Inferência das Estratégias**: Para cada estratégia fornecida, o pipeline
        itera sobre os pares preparados do dataset, aplicando o método `predict`
        da estratégia para obter os Log-Likelihood Ratios (LRs).
    4.  **Cálculo de Métricas**: Utiliza um `EvaluationService` para calcular
        métricas de desempenho como Equal Error Rate (EER) e Cost of Log-Likelihood Ratio (Cllr)
        com base nos LRs gerados e no ground truth.
    5.  **Persistência dos Resultados**: Salva o objeto `ValidationResult` (contendo
        as métricas e outros dados relevantes) em um arquivo `.pkl` na pasta de saída especificada
        para cada estratégia avaliada.
    """

    def __init__(self, strategies: List[LRModelProtocol], evaluation_service: EvaluationService):
        """
        Inicializa o ValidationPipeline com as estratégias de validação e o serviço de avaliação.

        Args:
            strategies (List[LRModelProtocol]): Uma lista de objetos que implementam
                                                o `LRModelProtocol`. Cada estratégia
                                                representa um método de calibração
                                                ou um modelo a ser avaliado.
            evaluation_service (EvaluationService): Uma instância do serviço responsável
                                                    pelo cálculo das métricas de desempenho
                                                    (EER, Cllr, etc.).
        """
        self.strategies = strategies
        self.eval_service = evaluation_service
        print(f"✅ ValidationPipeline inicializado com {len(strategies)} estratégias de calibração.")

    def execute(self,
                test_dataset: EmbeddingComparisonDataset,
                output_folder: str,
                balance_pairs: bool = True,
                random_seed: int = 42):
        """
        Executa o pipeline completo de validação para as estratégias configuradas.

        Este método orquestra a preparação dos dados, a execução da inferência
        para cada estratégia, o cálculo das métricas de desempenho e a
        persistência dos resultados.

        Args:
            test_dataset (EmbeddingComparisonDataset): O dataset contendo os pares de embeddings
                                                       (com ground truth conhecido) a serem usados
                                                       para a validação.
            output_folder (str): O caminho para a pasta onde os arquivos de relatório
                                 (resultados da validação em formato `.pkl`) serão salvos.
                                 A pasta será criada se não existir.
            balance_pairs (bool, optional): Se `True`, o pipeline sub-amostrará os pares
                                            impostores para igualar o número de pares
                                            genuínos. Isso é altamente recomendado para
                                            datasets N x N muito grandes, pois reduz
                                            drasticamente o tempo de validação sem
                                            comprometer significativamente a qualidade da avaliação
                                            para métricas como EER e Cllr.
                                            Se `False`, todos os pares do dataset serão processados.
                                            Padrão é `True`.
            random_seed (int, optional): Semente para o gerador de números pseudoaleatórios,
                                         utilizada para garantir a reprodutibilidade da
                                         sub-amostragem de pares quando `balance_pairs` é `True`.
                                         Padrão é `42`.
        """
        print("\n=== Iniciando Pipeline de Validação ===")
        print(f"Configurações: Dataset='{test_dataset.name}', Balancear Pares={balance_pairs}, Pasta de Saída='{output_folder}'")
        
        # Garante que a pasta de saída exista
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        print(f"Verificando/Criando pasta de saída: {Path(output_folder).resolve()}")

        # 1. Preparação e Balanceamento dos Dados
        items_to_process = list(test_dataset.items) # Cópia para permitir modificação

        if balance_pairs:
            print(f"⚖️ Balanceando dataset '{test_dataset.name}' (Target 1:1 Genuíno:Impostor)...")
            genuines = [item for item in items_to_process if item.same_person is True]
            impostors = [item for item in items_to_process if item.same_person is False]

            n_gen = len(genuines)
            n_imp_total = len(impostors)

            print(f"   Originalmente: {n_gen} Genuínos, {n_imp_total} Impostores.")

            if n_gen > 0 and n_imp_total > n_gen:
                random.seed(random_seed)
                # Seleciona impostores aleatórios na mesma quantidade dos genuínos
                impostors_sampled = random.sample(impostors, n_gen)
                items_to_process = genuines + impostors_sampled
                print(f"   📉 Redução: {len(test_dataset.items)} -> {len(items_to_process)} pares após balanceamento.")
                print(f"   ✅ Configuração balanceada: {len(genuines)} Genuínos vs {len(impostors_sampled)} Impostores.")
            else:
                print(f"   ⚠️ Balanceamento ignorado (Genuínos insuficientes ({n_gen}) ou Impostores ({n_imp_total}) não excedem genuínos para amostragem 1:1).")
                print(f"   Processando com {n_gen} Genuínos e {n_imp_total} Impostores (total: {len(items_to_process)}).")
        else:
            print(f"⚠️ Balanceamento de pares desativado. Processando dataset completo ({len(items_to_process)} pares). Isso pode ser lento para datasets grandes.")
            n_gen = sum(1 for item in test_dataset.items if item.same_person)
            n_imp_total = sum(1 for item in test_dataset.items if not item.same_person)
            print(f"   Dataset '{test_dataset.name}' contém {n_gen} Genuínos e {n_imp_total} Impostores.")

        n_items_final = len(items_to_process)
        ground_truth = np.array([item.same_person for item in items_to_process])
        scores_raw = np.array([item.cosine_similarity for item in items_to_process])
        print(f"   Dataset final preparado para processamento com {n_items_final} pares.")

        # 2. Execução das Estratégias
        for strategy in self.strategies:
            print(f"\n>> Validando Estratégia: {strategy.name}...")
            print(f"   Iniciando inferência para '{strategy.name}' em {n_items_final} pares.")

            lrs = []

            # Loop de Inferência
            for i, item in enumerate(items_to_process):
                # Ajuste o intervalo de impressão para datasets grandes para não poluir o console
                if n_items_final > 1000 and i % (max(1, n_items_final // 10)) == 0:
                    print(f"   Processando inferência: {i}/{n_items_final} pares ({i/n_items_final:.0%})...", end='\r')
                elif n_items_final <= 1000 and i % (max(1, n_items_final // 2)) == 0: # Para datasets menores, mais feedback
                     print(f"   Processando inferência: {i}/{n_items_final} pares ({i/n_items_final:.0%})...", end='\r')


                try:
                    # O protocolo exige predição par-a-par
                    lr = strategy.predict(item.embedding_1, item.embedding_2)
                    lrs.append(lr)
                except Exception as e:
                    # Em caso de erro na inferência para um par, logamos e atribuímos um LR neutro
                    # para permitir que a validação prossiga para outros pares.
                    print(f"\n   ❌ Erro na inferência para o par {i} "
                          f"(Ground Truth: {item.same_person}, Score: {item.cosine_similarity:.4f}) "
                          f"da estratégia '{strategy.name}': {e}")
                    lrs.append(1.0) # Atribui um LR neutro (1.0) em caso de erro
                    print(f"   Atribuindo LR neutro (1.0) para o par {i} e continuando. "
                          f"Considere investigar a causa do erro para '{strategy.name}'.")

            lrs_array = np.array(lrs)
            print(f"\n   Inferência concluída para '{strategy.name}' em {n_items_final} pares. Calculando métricas...")

            # 3. Avaliação Científica (Delega para o Service)
            try:
                print(f"   Iniciando avaliação de métricas (EER, Cllr) para '{strategy.name}'...")
                validation_result = self.eval_service.evaluate(
                    lrs=lrs_array,
                    ground_truth=ground_truth,
                    scores=scores_raw,
                    strategy_name=strategy.name,
                    dataset_name=test_dataset.name or "Unknown_Test_Set"
                )
                print(f"   Avaliação de métricas para '{strategy.name}' finalizada com sucesso.")

                # 4. Exibe Resumo
                print(f"   ✅ Resultado para '{strategy.name}': EER={validation_result.eer:.2%} | Cllr={validation_result.cllr:.4f} (com {len(lrs_array)} LRs processados)")

                # 5. Persistência
                filename = f"{strategy.name}_validation.pkl"
                save_path = Path(output_folder) / filename
                print(f"   Preparando para salvar relatório para '{strategy.name}' em: {save_path}")

                with open(save_path, 'wb') as f:
                    pickle.dump(validation_result, f)

                print(f"   ✔️ Relatório salvo com sucesso em: {save_path}")

            except Exception as e:
                print(f"\n   ❌ Falha crítica na avaliação ou persistência de resultados para a estratégia '{strategy.name}': {e}")
                import traceback
                traceback.print_exc() # Imprime o stack trace completo para depuração em caso de erro crítico

        print("\n=== Pipeline de Validação Concluído ===")
        print(f"Todos os resultados das estratégias foram processados e salvos (se bem-sucedidos) na pasta: {output_folder}")
