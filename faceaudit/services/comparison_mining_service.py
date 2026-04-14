import random
from collections import defaultdict
from typing import Tuple, List

from faceaudit.models.embedding_comparison import EmbeddingComparison
from faceaudit.models.embedding_comparison_dataset import EmbeddingComparisonDataset
from faceaudit.models.comparison_mining_config import (
    ComparisonMiningConfig,
    MiningRule,
    MiningStrategy,
    MiningFocus,
    GlobalSelectionMode
)


class ComparisonMiningService:
    """
    Serviço central de curadoria estatística e mineração de pares biométricos.
    
    Este serviço atua como um 'filtro inteligente' antes do treinamento de calibradores 
    (como a Regressão Logística do PLDA). Ele recebe uma matriz densa de todas as 
    comparações possíveis e extrai um subconjunto altamente representativo.
    
    Principais responsabilidades:
    - Evitar overfitting geográfico/demográfico limitando a contribuição por identidade.
    - Aplicar estratégias de fatiamento de dificuldade (HARD, EASY, TYPICAL).
    - Orquestrar proporções mistas sem permitir sobreposição ou duplicação de dados.
    """

    def mine_pairs(
        self, 
        dataset: EmbeddingComparisonDataset, 
        config: ComparisonMiningConfig
    ) -> Tuple[EmbeddingComparisonDataset, EmbeddingComparisonDataset]:
        """
        Executa o pipeline completo de curadoria e separação de pares.

        Lê o dataset bruto contendo comparações pré-calculadas, separa entre genuínos 
        (mesma identidade) e impostores (identidades diferentes), remove auto-comparações 
        (se configurado) e aplica as regras de mineração específicas para cada grupo.

        Args:
            dataset (EmbeddingComparisonDataset): Dataset bruto contendo todos os pares possíveis.
            config (ComparisonMiningConfig): Motor de regras definindo semente, limites e estratégias.

        Returns:
            Tuple[EmbeddingComparisonDataset, EmbeddingComparisonDataset]: 
                Uma tupla contendo (Dataset_Impostores_Minerados, Dataset_Genuínos_Minerados).
        """
        rng = random.Random(config.seed)
        
        # 1. Remoção de Auto-Comparações e Divisão Genuíno/Impostor
        gen_pool = []
        imp_pool = []
        
        for comp in dataset.items:
            # Pula comparações da imagem com a sua duplicata exata (mesmo arquivo/hash)
            if config.exclude_self_comparisons and comp.embedding_1.face_hash == comp.embedding_2.face_hash:
                continue
                
            id1 = comp.embedding_1.identity
            id2 = comp.embedding_2.identity
            
            # Pula casos sem anotação de identidade, pois não podem ser classificados
            if id1 is None or id2 is None:
                continue
                
            if id1 == id2:
                gen_pool.append(comp)
            else:
                imp_pool.append(comp)

        # 2. Aplicação das Regras Independentes
        final_gen = self._apply_rule(gen_pool, config.genuines_rule, rng)
        final_imp = self._apply_rule(imp_pool, config.impostors_rule, rng)

        # 3. Empacotamento
        ds_imp = EmbeddingComparisonDataset(name=f"{dataset.name}_Impostors_Mined", items=final_imp)
        ds_gen = EmbeddingComparisonDataset(name=f"{dataset.name}_Genuines_Mined", items=final_gen)

        return ds_imp, ds_gen

    def _apply_rule(
        self, 
        pairs: List[EmbeddingComparison], 
        rule: MiningRule, 
        rng: random.Random
    ) -> List[EmbeddingComparison]:
        """
        Roteador interno que decide qual motor de extração usar baseado no Foco e Modo de Seleção.

        Args:
            pairs (List[EmbeddingComparison]): A lista bruta de pares (só genuínos ou só impostores).
            rule (MiningRule): A regra atômica a ser aplicada.
            rng (random.Random): Instância do gerador de números aleatórios para reprodutibilidade.

        Returns:
            List[EmbeddingComparison]: A lista de pares filtrada e minerada.
        """
        if not pairs:
            return []

        # Caso A: Foco direto em Pares (Saco global único) sem balanceamento por indivíduo
        if rule.focus == MiningFocus.PAIRS or rule.global_selection_mode == GlobalSelectionMode.GLOBAL_ORDER:
            return self._slice_from_global_pool(pairs, rule, rng)
            
        # Caso B: Foco em Identidades com Balanceamento (Round Robin)
        elif rule.global_selection_mode == GlobalSelectionMode.ROUND_ROBIN_BY_PIVOT:
            return self._round_robin_identities(pairs, rule, rng)
            
        return pairs

    # =========================================================================
    # MOTOR 1: O BARALHO DE CARTAS (ROUND ROBIN)
    # =========================================================================

    def _round_robin_identities(
        self, 
        pairs: List[EmbeddingComparison], 
        rule: MiningRule, 
        rng: random.Random
    ) -> List[EmbeddingComparison]:
        """
        Executa a extração usando Revezamento Circular (Round Robin) e "Amostragem sem Reposição".

        Lógica:
        1. Agrupa todos os pares pela Identidade Pivô (a primeira face da comparação).
        2. Ordena os pares de cada identidade por score (LLR ou Similaridade).
        3. Coloca as identidades em uma fila circular.
        4. Visita cada identidade uma vez por rodada, removendo um par (pop) de acordo 
           com a estratégia exigida no momento (Hard, Typical, Random).
        5. Se uma identidade fica sem pares, ela é ejetada da fila silenciosamente.

        Isso garante que pessoas com 10.000 fotos não esmaguem pessoas com 2 fotos na 
        calibração do modelo, diluindo o viés de frequência do dataset.

        Args:
            pairs (List[EmbeddingComparison]): Lista de pares brutos.
            rule (MiningRule): Regra contendo a estratégia e o limite de pares.
            rng (random.Random): Gerador de números aleatórios.

        Returns:
            List[EmbeddingComparison]: Pares curados e balanceados por identidade.
        """
        max_global = rule.max_global_pairs
        if max_global is None:
            max_global = len(pairs)

        # 1. Agrupar em Baralhos por Identidade Pivô
        grouped = defaultdict(list)
        for comp in pairs:
            grouped[comp.embedding_1.identity].append(comp)

        # 2. Ordenar as cartas de CADA baralho do Mais Difícil (Score Alto) para o Mais Fácil
        decks = {}
        for pivot, p_list in grouped.items():
            decks[pivot] = sorted(p_list, key=lambda x: x.cosine_similarity, reverse=True)

        # 3. Formar a Fila Inicial (embaralhada para evitar viés alfabético na primeira extração)
        active_pivots = list(decks.keys())
        rng.shuffle(active_pivots)

        results = []
        strategy_counts = {s: 0 for s in MiningStrategy}

        def get_next_strategy(current_total: int) -> MiningStrategy:
            """
            Calcula qual estratégia deve ser alimentada na rodada atual do modo MIXED,
            baseando-se no maior déficit proporcional em relação à meta.
            """
            if rule.strategy != MiningStrategy.MIXED:
                return rule.strategy

            props = rule.mixed_proportions
            targets = {
                MiningStrategy.HARD: props.hard_ratio,
                MiningStrategy.TYPICAL: props.typical_ratio,
                MiningStrategy.RANDOM: props.random_ratio,
                MiningStrategy.EASY: props.easy_ratio
            }
            
            best_st = MiningStrategy.RANDOM
            max_deficit = -1.0
            
            for st, target_ratio in targets.items():
                if target_ratio <= 0: 
                    continue
                current_ratio = strategy_counts[st] / max(1, current_total)
                deficit = target_ratio - current_ratio
                if deficit > max_deficit:
                    max_deficit = deficit
                    best_st = st
            return best_st

        # 4. A Fila de Revezamento
        idx = 0
        while len(results) < max_global and active_pivots:
            pivot = active_pivots[idx % len(active_pivots)]
            deck = decks[pivot]

            # Decide qual carta tirar com base na necessidade atual (Déficit)
            st = get_next_strategy(len(results))
            
            # EXTRAÇÃO (POP) -> Remove a carta do baralho para evitar duplicatas absolutas
            if st == MiningStrategy.HARD:
                card = deck.pop(0)                   # O Topo (Pior Sósia)
            elif st == MiningStrategy.EASY:
                card = deck.pop(-1)                  # O Fundo (Mais Diferente)
            elif st == MiningStrategy.TYPICAL:
                card = deck.pop(len(deck) // 2)      # O Miolo (Mediana)
            else: # RANDOM
                rand_idx = rng.randrange(len(deck))
                card = deck.pop(rand_idx)            # Sorteio Uniforme
            
            results.append(card)
            strategy_counts[st] += 1
            
            # Drop-out: Se a pessoa ficou sem cartas, ela sai da fila de revezamento
            if not deck:
                active_pivots.remove(pivot)
                # O idx não é incrementado pois a lista encolheu e o próximo ocupou a posição atual
            else:
                idx += 1

        return results

    # =========================================================================
    # MOTOR 2: FATIAMENTO DO SACO GLOBAL (GLOBAL_ORDER)
    # =========================================================================

    def _slice_from_global_pool(
        self, 
        pool: List[EmbeddingComparison], 
        rule: MiningRule, 
        rng: random.Random
    ) -> List[EmbeddingComparison]:
        """
        Ordena todos os pares numa lista única e faz recortes geométricos.

        Diferente do Round Robin, esta abordagem não protege contra o viés de 
        identidades frequentes. Ela apenas olha para a matemática global do score.
        Para o modo MIXED, a função destrói o array à medida que fatia para garantir
        que não haja sobreposição de pares (Ex: o HARD não pode ser sorteado pelo RANDOM).

        Args:
            pool (List[EmbeddingComparison]): A piscina global de pares.
            rule (MiningRule): Regra contendo as proporções e estratégia global.
            rng (random.Random): Gerador de números aleatórios.

        Returns:
            List[EmbeddingComparison]: Lista final fatiada.
        """
        max_global = rule.max_global_pairs
        if max_global is None or max_global >= len(pool):
            return pool.copy()

        # Ordena do maior (mais similar) para o menor
        sorted_pool = sorted(pool, key=lambda x: x.cosine_similarity, reverse=True)

        if rule.strategy == MiningStrategy.HARD:
            return sorted_pool[:max_global]
            
        elif rule.strategy == MiningStrategy.EASY:
            return sorted_pool[-max_global:]
            
        elif rule.strategy == MiningStrategy.TYPICAL:
            mid = len(sorted_pool) // 2
            half = max_global // 2
            start = max(0, mid - half)
            return sorted_pool[start : start + max_global]
            
        elif rule.strategy == MiningStrategy.RANDOM:
            return rng.sample(sorted_pool, max_global)
            
        elif rule.strategy == MiningStrategy.MIXED:
            # Fatiamento cirúrgico sem sobreposição (Destructive Slicing)
            props = rule.mixed_proportions
            n_hard = int(max_global * props.hard_ratio)
            n_easy = int(max_global * props.easy_ratio)
            n_typ = int(max_global * props.typical_ratio)
            n_rnd = max_global - (n_hard + n_easy + n_typ) # O resto vai pro random

            results = []
            
            # 1. Pega os mais difíceis (Topo) e os remove da piscina principal
            if n_hard > 0:
                results.extend(sorted_pool[:n_hard])
                sorted_pool = sorted_pool[n_hard:]
                
            # 2. Pega os mais fáceis (Fundo) e os remove da piscina
            if n_easy > 0 and sorted_pool:
                n_easy = min(n_easy, len(sorted_pool))
                results.extend(sorted_pool[-n_easy:])
                sorted_pool = sorted_pool[:-n_easy]
                
            # 3. Pega os típicos (Novo Miolo) e os remove reconstruindo o array
            if n_typ > 0 and sorted_pool:
                n_typ = min(n_typ, len(sorted_pool))
                mid = len(sorted_pool) // 2
                half = n_typ // 2
                start = max(0, mid - half)
                results.extend(sorted_pool[start : start + n_typ])
                sorted_pool = sorted_pool[:start] + sorted_pool[start + n_typ:]
                
            # 4. Sorteia os aleatórios do que sobrou intacto na piscina
            if n_rnd > 0 and sorted_pool:
                n_rnd = min(n_rnd, len(sorted_pool))
                results.extend(rng.sample(sorted_pool, n_rnd))
                
            return results