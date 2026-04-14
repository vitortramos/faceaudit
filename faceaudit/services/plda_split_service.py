import random
from typing import Tuple, Set, Dict, Iterable, Optional
from collections import Counter

from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.enums.plda_training_strategy import PLDATrainingStrategy
from faceaudit.models.split_config import SplitConfig, SplitMode, GroupingStrategy

# Injetando o Serviço de Comparação
from faceaudit.services.comparison_service import ComparisonService


class PLDASplitService:
    """
    Serviço responsável por dividir (split) os datasets de entrada em conjuntos mutuamente 
    exclusivos de Treinamento (Matriz PLDA) e Calibração (Regressão Logística).
    
    A regra de ouro deste serviço é: Identidades não podem vazar de um dataset para outro.
    Se a 'Pessoa A' foi para a calibração, ela não pode estar no treino.
    
    Destaque Arquitetural:
    Implementa a estratégia de 'Clusterização de Sósias' (Preserve Lookalikes). Para garantir
    que a Regressão Logística consiga ser treinada com casos difíceis (Hard Impostors), 
    este serviço identifica sósias previamente e os envia em pacotes indivisíveis para o 
    mesmo lado do split, preservando a densidade de dificuldade geométrica do mundo real.
    """

    def __init__(self, comparison_service: Optional[ComparisonService] = None):
        """
        Inicializa o serviço de divisão.

        Args:
            comparison_service (Optional[ComparisonService]): Serviço injetado usado para 
                calcular a Matriz de Confusão entre identidades. Se não for fornecido, 
                instancia um internamente.
        """
        self.comparison_service = comparison_service or ComparisonService()

    def perform_split(
        self,
        source_dataset: EmbeddingDataset,
        target_dataset: Optional[EmbeddingDataset],
        strategy_type,  # Pode ser o Enum ou a string correspondente
        config: SplitConfig,
        logger_callback: callable = print
    ) -> Tuple[EmbeddingDataset, EmbeddingDataset]:
        """
        Orquestra a divisão inteligente dos dados baseada na estratégia de domínio.

        Se a estratégia for Supervisionada (Cross-Domain), o serviço tenta priorizar identidades 
        que existam em ambos os domínios (Interseção) para a calibração. Caso contrário, 
        faz o split padrão na base de origem.

        Args:
            source_dataset (EmbeddingDataset): O dataset principal de origem.
            target_dataset (Optional[EmbeddingDataset]): O dataset de destino (usado em adaptação de domínio).
            strategy_type (Any): O tipo de estratégia PLDA sendo treinada (ex: SUPERVISED_ADAPTATION).
            config (SplitConfig): Motor de regras contendo cotas, limites e estratégia de agrupamento.
            logger_callback (callable): Função para injeção de logs na interface do usuário.

        Returns:
            Tuple[EmbeddingDataset, EmbeddingDataset]: Uma tupla contendo:
                1. Dataset de Treino (Para gerar a Matriz PLDA).
                2. Dataset de Calibração (Para treinar a Regressão Logística).
        """
        src_ids = self._sanitize_ids(source_dataset.identities)
        st_val = strategy_type.value if hasattr(strategy_type, 'value') else str(strategy_type)

        # =====================================================================
        # CASO 1: SUPERVISED_ADAPTATION (Requer cruzamento de domínios)
        # =====================================================================
        if st_val == PLDATrainingStrategy.SUPERVISED_ADAPTATION.value:
            if target_dataset is None:
                raise ValueError("A Estratégia Supervisionada exige um dataset Target (com labels).")

            logger_callback("🔄 [SplitService] Estratégia Supervisionada: Focando na interseção de domínios...")
            
            full_dataset = source_dataset + target_dataset
            tgt_ids = self._sanitize_ids(target_dataset.identities)
            intersection = src_ids.intersection(tgt_ids)

            # 1) Tenta bater a meta apenas com pessoas que existem em ambos os domínios
            calib_ids, metrics = self._select_ids(
                pool_dataset=full_dataset,
                candidate_ids=intersection,
                config=config,
                logger_callback=logger_callback
            )

            # 2) Fallback: Se a interseção for pequena, completa com identidades exclusivas do Target
            if metrics["total_pairs"] < config.min_calibration_pairs or len(calib_ids) < config.min_calibration_ids:
                logger_callback(
                    f"⚠️ [SplitService] Interseção gerou apenas {metrics['total_pairs']} pares. "
                    f"Completando com identidades exclusivas do Target..."
                )
                remaining_target = tgt_ids - calib_ids
                
                needed_pairs = max(0, config.min_calibration_pairs - metrics["total_pairs"])
                needed_ids = max(0, config.min_calibration_ids - len(calib_ids))
                
                temp_config = SplitConfig(
                    mode=SplitMode.ABSOLUTE, 
                    grouping_strategy=config.grouping_strategy,
                    cluster_size=config.cluster_size,
                    min_calibration_pairs=needed_pairs,
                    min_calibration_ids=needed_ids,
                    min_samples_per_id=config.min_samples_per_id,
                    seed=config.seed + 1 
                )
                
                extra_ids, _ = self._select_ids(
                    pool_dataset=full_dataset,
                    candidate_ids=remaining_target,
                    config=temp_config,
                    logger_callback=logger_callback
                )
                calib_ids.update(extra_ids)
                
            calib_set, plda_set = full_dataset.split_by_identities(calib_ids)

        # =====================================================================
        # CASO 2: BASELINE / NÃO-SUPERVISIONADAS (Apenas Source Domain)
        # =====================================================================
        else:
            logger_callback("🔹 [SplitService] Estratégia Base (Source Domain Only).")
            calib_ids, metrics = self._select_ids(
                pool_dataset=source_dataset,
                candidate_ids=src_ids,
                config=config,
                logger_callback=logger_callback
            )
            calib_set, plda_set = source_dataset.split_by_identities(calib_ids)

        # Logs finais
        n_plda_ids = len(self._sanitize_ids(plda_set.identities))
        logger_callback(f"✅ [SplitService] Divisão concluída!")
        logger_callback(f"   -> Calibração (Regressão): {len(calib_set)} amostras de {len(calib_ids)} identidades.")
        logger_callback(f"   -> Treino (Matriz PLDA): {len(plda_set)} amostras de {n_plda_ids} identidades.")

        return plda_set, calib_set

    # =========================================================================
    # MOTOR MATEMÁTICO DE SELEÇÃO (Com Preservação de Sósias)
    # =========================================================================

    def _select_ids(
        self, 
        pool_dataset: EmbeddingDataset, 
        candidate_ids: Set[str], 
        config: SplitConfig,
        logger_callback: callable
    ) -> Tuple[Set[str], Dict]:
        """
        Aplica a lógica matemática de extração para selecionar as identidades da calibração.

        Esta função calcula as cotas necessárias (proporção ou número absoluto) e percorre 
        as identidades de forma aleatória. Se a estratégia for PRESERVE_LOOKALIKES, ela 
        utiliza o ComparisonService para encontrar os piores sósias da identidade sorteada 
        e os agrupa no mesmo dataset, preservando a capacidade de treinar os casos limítrofes.

        Args:
            pool_dataset (EmbeddingDataset): O dataset completo onde as identidades estão.
            candidate_ids (Set[str]): Conjunto de identidades permitidas para o sorteio.
            config (SplitConfig): Regras de meta (IDs, Pares) e agrupamento.
            logger_callback (callable): Função para log.

        Returns:
            Tuple[Set[str], Dict]:
                1. Conjunto (Set) de strings com os nomes das identidades selecionadas.
                2. Dicionário com as métricas ('total_pairs', 'total_ids') alcançadas.
        """
        if not candidate_ids:
            return set(), {"total_pairs": 0, "total_ids": 0}

        def norm(x) -> Optional[str]:
            if x is None: return None
            s = str(x).strip()
            return s if s else None

        all_ids = (norm(i) for i in pool_dataset.identities)
        relevant_counts = Counter(i for i in all_ids if i is not None and i in candidate_ids)

        # 1. Filtra quem tem o mínimo de amostras necessárias (Ex: remover singletons)
        eligible_ids = [idn for idn, k in relevant_counts.items() if k >= config.min_samples_per_id]
        
        rng = random.Random(config.seed)
        eligible_ids.sort() # Ordenação pré-shuffle garante reprodutibilidade cross-platform
        rng.shuffle(eligible_ids) 

        # 2. Calcula Metas (Alvos de Calibração)
        target_ids_by_ratio = int(len(eligible_ids) * config.calibration_ratio) if config.mode == SplitMode.RATIO else 0
        target_ids = max(config.min_calibration_ids, target_ids_by_ratio)
        target_pairs = config.min_calibration_pairs

        # 3. Preparação do Motor de Clustering (Matriz de Confusão)
        confusion_dict = {}
        if config.grouping_strategy == GroupingStrategy.PRESERVE_LOOKALIKES:
            logger_callback(f"🔍 [SplitService] Clusterização ativada. Analisando sósias no dataset...")
            temp_pool, _ = pool_dataset.split_by_identities(set(eligible_ids))
            confusion_dict = self.comparison_service.get_identity_confusion_matrix(
                dataset=temp_pool, 
                logger_callback=logger_callback
            )

        selected = set()
        current_pairs = 0
        eligible_set = set(eligible_ids) # Set para busca rápida O(1)

        # 4. Seleção Cumulativa (Até bater a meta)
        for idn in eligible_ids:
            if idn in selected:
                continue

            # Inicia o pacote com a identidade sorteada
            package_to_add = [idn]

            # Injeta os sósias no pacote se a estratégia exigir
            if config.grouping_strategy == GroupingStrategy.PRESERVE_LOOKALIKES:
                lookalikes_needed = max(0, config.cluster_size - 1)
                
                if lookalikes_needed > 0 and idn in confusion_dict:
                    available_lookalikes = [
                        sim_id for sim_id in confusion_dict[idn] 
                        if sim_id in eligible_set and sim_id not in selected
                    ]
                    package_to_add.extend(available_lookalikes[:lookalikes_needed])

            # Adiciona o pacote inteiro ao conjunto selecionado e calcula as combinações C(n,2)
            for package_id in package_to_add:
                if package_id in selected: continue 
                
                k = relevant_counts[package_id]
                pairs = (k * (k - 1)) // 2  
                
                selected.add(package_id)
                current_pairs += pairs

            # Verifica condição de parada (Atingiu mínimo de IDs e Mínimo de Pares Genuínos)
            if len(selected) >= target_ids and current_pairs >= target_pairs:
                break

        return selected, {"total_pairs": current_pairs, "total_ids": len(selected)}

    def _sanitize_ids(self, ids_iter: Iterable) -> Set[str]:
        """Limpa IDs brutos, removendo nulos e espaços vazios."""
        out = set()
        for i in ids_iter:
            if i is None: continue
            s = str(i).strip()
            if s: out.add(s)
        return out