import random
from collections import defaultdict
from typing import List

from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample
from faceaudit.models.comparison_mining_config import (
    ComparisonMiningConfig,
    MiningRule,
    MiningStrategy,
    MiningFocus,
    GlobalSelectionMode,
)


class ComparisonMiningService:
    """
    Serviço central de curadoria estatística e mineração de pares biométricos.

    Atua como 'filtro inteligente' antes do treinamento de calibradores
    (Logistic Regression, KDE). Recebe um SimilarityScoreDataset com todos os
    pares pré-pontuados pelo SIMILARITY_SCORING e extrai um subconjunto
    representativo segundo as regras configuradas (HARD/EASY/TYPICAL/RANDOM/
    MIXED, foco em pares ou identidades, balanceamento round-robin).

    Output é um único SimilarityScoreDataset (impostores minerados +
    genuínos minerados concatenados), pronto para alimentar o
    CALIBRATION_MODEL_TRAINING — separação interna por classe é dada pelo
    campo `same_person`/`label` em cada sample.
    """

    def mine_pairs(
        self,
        dataset: SimilarityScoreDataset,
        config: ComparisonMiningConfig,
    ) -> SimilarityScoreDataset:
        """
        Executa o pipeline completo de curadoria.

        Lê o `SimilarityScoreDataset` bruto, separa genuínos e impostores via
        `sample.same_person`, opcionalmente remove auto-comparações por
        face_hash, aplica as regras independentes (`genuines_rule` e
        `impostors_rule`) e retorna **um** SimilarityScoreDataset com os
        pares curados (impostores primeiro, depois genuínos — a ordem é
        irrelevante para o treino do calibrador, mas mantém compatibilidade
        com o histórico do service).
        """
        rng = random.Random(config.seed)

        # 1. Separação genuíno/impostor (com remoção opcional de auto-comparações)
        gen_pool: List[SimilarityScoreSample] = []
        imp_pool: List[SimilarityScoreSample] = []

        for sample in dataset.items:
            # Auto-comparação: mesmo face_hash em probe e reference
            if (
                config.exclude_self_comparisons
                and sample.probe.face_hash is not None
                and sample.probe.face_hash == sample.reference.face_hash
            ):
                continue

            # Sem ground truth → não classificável; descarta.
            if not sample.has_ground_truth:
                continue

            if sample.same_person:
                gen_pool.append(sample)
            else:
                imp_pool.append(sample)

        # 2. Regras independentes
        final_gen = self._apply_rule(gen_pool, config.genuines_rule, rng)
        final_imp = self._apply_rule(imp_pool, config.impostors_rule, rng)

        # 3. Empacotamento — output único concatenado
        mined_items: List[SimilarityScoreSample] = [*final_imp, *final_gen]

        return SimilarityScoreDataset(
            name=f"{dataset.name}_Mined",
            description=(
                f"Pares minerados de {dataset.name} via ComparisonMiningService "
                f"(impostores={len(final_imp)}, genuínos={len(final_gen)})."
            ),
            items=mined_items,
            metadata={
                "source_dataset_name": dataset.name,
                "n_genuines_mined": len(final_gen),
                "n_impostors_mined": len(final_imp),
            },
        )

    def _apply_rule(
        self,
        pairs: List[SimilarityScoreSample],
        rule: MiningRule,
        rng: random.Random,
    ) -> List[SimilarityScoreSample]:
        """
        Roteador interno: PAIRS/GLOBAL_ORDER → fatiamento global;
        IDENTITIES + ROUND_ROBIN_BY_PIVOT → revezamento por identidade.
        """
        if not pairs:
            return []

        if rule.focus == MiningFocus.PAIRS or rule.global_selection_mode == GlobalSelectionMode.GLOBAL_ORDER:
            return self._slice_from_global_pool(pairs, rule, rng)

        elif rule.global_selection_mode == GlobalSelectionMode.ROUND_ROBIN_BY_PIVOT:
            return self._round_robin_identities(pairs, rule, rng)

        return pairs

    # =========================================================================
    # MOTOR 1: ROUND ROBIN POR IDENTIDADE PIVÔ
    # =========================================================================

    def _round_robin_identities(
        self,
        pairs: List[SimilarityScoreSample],
        rule: MiningRule,
        rng: random.Random,
    ) -> List[SimilarityScoreSample]:
        """
        Revezamento circular sem reposição entre identidades pivô (probe).

        Agrupa os pares por `sample.probe.identity`, ordena cada baralho por
        `raw_score` descendente, e visita cada identidade uma vez por rodada
        retirando uma carta segundo a estratégia (HARD = topo, EASY = fundo,
        TYPICAL = miolo, RANDOM = sorteio uniforme; MIXED escolhe segundo
        déficit proporcional). Identidades sem pares são ejetadas
        silenciosamente. Garante que pessoas com muitas amostras não
        dominem o conjunto minerado.
        """
        max_global = rule.max_global_pairs
        if max_global is None:
            max_global = len(pairs)

        # 1. Baralhos por identidade pivô
        grouped = defaultdict(list)
        for sample in pairs:
            grouped[sample.probe.identity].append(sample)

        # 2. Ordenação interna de cada baralho — Top score = mais difícil
        decks = {}
        for pivot, p_list in grouped.items():
            decks[pivot] = sorted(p_list, key=lambda s: s.raw_score, reverse=True)

        # 3. Fila inicial embaralhada (evita viés alfabético)
        active_pivots = list(decks.keys())
        rng.shuffle(active_pivots)

        results: List[SimilarityScoreSample] = []
        strategy_counts = {s: 0 for s in MiningStrategy}

        def get_next_strategy(current_total: int) -> MiningStrategy:
            if rule.strategy != MiningStrategy.MIXED:
                return rule.strategy

            props = rule.mixed_proportions
            targets = {
                MiningStrategy.HARD: props.hard_ratio,
                MiningStrategy.TYPICAL: props.typical_ratio,
                MiningStrategy.RANDOM: props.random_ratio,
                MiningStrategy.EASY: props.easy_ratio,
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

        # 4. Loop de revezamento
        idx = 0
        while len(results) < max_global and active_pivots:
            pivot = active_pivots[idx % len(active_pivots)]
            deck = decks[pivot]

            st = get_next_strategy(len(results))

            if st == MiningStrategy.HARD:
                card = deck.pop(0)
            elif st == MiningStrategy.EASY:
                card = deck.pop(-1)
            elif st == MiningStrategy.TYPICAL:
                card = deck.pop(len(deck) // 2)
            else:  # RANDOM
                rand_idx = rng.randrange(len(deck))
                card = deck.pop(rand_idx)

            results.append(card)
            strategy_counts[st] += 1

            if not deck:
                active_pivots.remove(pivot)
            else:
                idx += 1

        return results

    # =========================================================================
    # MOTOR 2: FATIAMENTO DO POOL GLOBAL
    # =========================================================================

    def _slice_from_global_pool(
        self,
        pool: List[SimilarityScoreSample],
        rule: MiningRule,
        rng: random.Random,
    ) -> List[SimilarityScoreSample]:
        """
        Ordena globalmente por `raw_score` (desc) e fatia segundo a estratégia.

        Diferente do Round Robin, não protege contra viés de identidades
        frequentes — olha apenas a estatística global do score.
        Modo MIXED faz `destructive slicing`: HARD/EASY/TYPICAL são removidos
        do pool antes do RANDOM, para evitar sobreposição.
        """
        max_global = rule.max_global_pairs
        if max_global is None or max_global >= len(pool):
            return pool.copy()

        sorted_pool = sorted(pool, key=lambda s: s.raw_score, reverse=True)

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
            props = rule.mixed_proportions
            n_hard = int(max_global * props.hard_ratio)
            n_easy = int(max_global * props.easy_ratio)
            n_typ = int(max_global * props.typical_ratio)
            n_rnd = max_global - (n_hard + n_easy + n_typ)

            results: List[SimilarityScoreSample] = []

            if n_hard > 0:
                results.extend(sorted_pool[:n_hard])
                sorted_pool = sorted_pool[n_hard:]

            if n_easy > 0 and sorted_pool:
                n_easy = min(n_easy, len(sorted_pool))
                results.extend(sorted_pool[-n_easy:])
                sorted_pool = sorted_pool[:-n_easy]

            if n_typ > 0 and sorted_pool:
                n_typ = min(n_typ, len(sorted_pool))
                mid = len(sorted_pool) // 2
                half = n_typ // 2
                start = max(0, mid - half)
                results.extend(sorted_pool[start : start + n_typ])
                sorted_pool = sorted_pool[:start] + sorted_pool[start + n_typ:]

            if n_rnd > 0 and sorted_pool:
                n_rnd = min(n_rnd, len(sorted_pool))
                results.extend(rng.sample(sorted_pool, n_rnd))

            return results

        return pool.copy()
