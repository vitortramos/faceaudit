from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Iterable, List, Dict, Any
import random
from collections import defaultdict

from faceaudit.models.embedding_comparison_dataset import EmbeddingComparisonDataset
from faceaudit.models.embedding_comparison import EmbeddingComparison

@dataclass(frozen=True)
class ScoreModeInfo:
    """Estrutura de metadados para exibição amigável na Interface Gráfica."""
    label: str
    summary: str

class ScoreBasedStrategyMode(Enum):
    """
    Define a estratégia de Mineração e Curadoria de Pares (Pair Mining Strategy) para o KDE.
    O KDE aprende a densidade de probabilidade dos scores de Genuínos e Impostores.
    """

    # --- 1. MODOS GLOBAIS (Ignoram limites por ID) ---
    STANDARD = ("standard", "STANDARD", "Usa pares globais aleatórios. Reflete a distribuição real.")
    HARD_IMPOSTORS = ("hard_impostors", "HARD_IMPOSTORS (Global)", "Prioriza impostores com score alto (falsos positivos).")
    HARD_GENUINES = ("hard_genuines", "HARD_GENUINES (Global)", "Prioriza genuínos com score baixo (falsos negativos).")
    BOTH_HARD = ("both_hard", "BOTH_HARD (Global)", "Treina com os piores casos de ambos os lados (Conservador).")

    # --- 2. MODOS BALANCEADOS (Respeitam limites por ID) ---
    BALANCED_PER_ID = ("balanced_per_id", "BALANCED_RANDOM", "Amostragem aleatória com teto por identidade.")
    BALANCED_HARD_IMPOSTORS = ("balanced_hard_impostors", "BALANCED_HARD_IMP", "Impostores difíceis limitados por identidade.")
    BALANCED_HARD_GENUINES = ("balanced_hard_genuines", "BALANCED_HARD_GEN", "Genuínos difíceis limitados por identidade.")
    BALANCED_BOTH_HARD = ("balanced_both_hard", "BALANCED_BOTH_HARD", "Hard Mining balanceado para ambos os lados.")

    def __new__(cls, value: str, label: str, summary: str):
        # Pattern canônico de Enum com valor composto — ver
        # typicality_reference_mode.py pra explicação completa.
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        # label_pt alias — labels deste enum são identificadores técnicos
        # (STANDARD, HARD_IMPOSTORS, ...). Mantido por consistência de
        # API com outros enums do projeto.
        obj.label_pt = label
        obj.summary = summary
        return obj

    @property
    def info(self) -> ScoreModeInfo:
        """Retorna metadados ricos para a UI."""
        return ScoreModeInfo(self.label, self.summary)

    def __str__(self) -> str:
        return self.label_pt

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna o identificador técnico (ex: 'hard_impostors') para o JSON."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> ScoreBasedStrategyMode:
        """Reconstrói o Enum a partir da string com fallback seguro."""
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.STANDARD

    # =========================================================================
    # API PRINCIPAL DE MINERAÇÃO (Preservada conforme original)
    # =========================================================================

    def split_pairs(
        self,
        comparisons: EmbeddingComparisonDataset,
        *,
        impostors_ratio: float = 1.0,
        impostors_min: int = 1_000,
        impostors_max: int = 1_000_000,
        genuines_ratio: float = 1.0,
        genuines_min: int = 1_000,
        genuines_max: int = 1_000_000,
        per_identity_ratio: float = 1.0,
        per_identity_min: int = 10,
        per_identity_max: int = 500,
        exclude_self: bool = True,
        seed: int = 42
    ) -> Tuple[EmbeddingComparisonDataset, EmbeddingComparisonDataset]:
        """Executa a separação e curadoria dos pares baseada no modo selecionado."""
        cleaned = list(self._iter_valid_pairs(comparisons, exclude_self=exclude_self))
        
        impostors_items = [c for c in cleaned if not c.same_person]
        genuines_items = [c for c in cleaned if c.same_person]

        target_imp_global = self._calculate_target_count(
            len(impostors_items), impostors_ratio, impostors_min, impostors_max
        )
        target_gen_global = self._calculate_target_count(
            len(genuines_items), genuines_ratio, genuines_min, genuines_max
        )

        if self in [ScoreBasedStrategyMode.STANDARD, ScoreBasedStrategyMode.HARD_IMPOSTORS, 
                    ScoreBasedStrategyMode.HARD_GENUINES, ScoreBasedStrategyMode.BOTH_HARD]:
            
            if self in [ScoreBasedStrategyMode.HARD_IMPOSTORS, ScoreBasedStrategyMode.BOTH_HARD]:
                impostors_items.sort(key=lambda x: float(x.cosine_similarity), reverse=True)
                impostors_items = impostors_items[:target_imp_global]
            else: 
                random.seed(seed)
                random.shuffle(impostors_items)
                impostors_items = impostors_items[:target_imp_global]

            if self in [ScoreBasedStrategyMode.HARD_GENUINES, ScoreBasedStrategyMode.BOTH_HARD]:
                genuines_items.sort(key=lambda x: float(x.cosine_similarity), reverse=False)
                genuines_items = genuines_items[:target_gen_global]
            else: 
                random.seed(seed) 
                random.shuffle(genuines_items)
                genuines_items = genuines_items[:target_gen_global]
        
        else:
            sort_imp_desc = self in [ScoreBasedStrategyMode.BALANCED_HARD_IMPOSTORS, ScoreBasedStrategyMode.BALANCED_BOTH_HARD]
            sort_gen_asc = self in [ScoreBasedStrategyMode.BALANCED_HARD_GENUINES, ScoreBasedStrategyMode.BALANCED_BOTH_HARD]
            
            shuffle_imp = not sort_imp_desc
            shuffle_gen = not sort_gen_asc

            impostors_items, genuines_items = self._balance_per_identity(
                impostors_items, genuines_items,
                ratio=per_identity_ratio, min_val=per_identity_min, max_val=per_identity_max,
                sort_impostors_desc=sort_imp_desc, sort_genuines_asc=sort_gen_asc,
                shuffle_impostors=shuffle_imp, shuffle_genuines=shuffle_gen, seed=seed
            )
            
            if len(impostors_items) > target_imp_global:
                random.seed(seed)
                random.shuffle(impostors_items)
                impostors_items = impostors_items[:target_imp_global]
            
            if len(genuines_items) > target_gen_global:
                random.seed(seed)
                random.shuffle(genuines_items)
                genuines_items = genuines_items[:target_gen_global]

        return (
            self._create_dataset(comparisons, impostors_items, "Impostors"),
            self._create_dataset(comparisons, genuines_items, "Genuines")
        )

    # --- Helpers Internos ---

    def _calculate_target_count(self, total: int, ratio: float, min_val: int, max_val: int) -> int:
        if total == 0: return 0
        target = int(total * ratio)
        target = max(target, min_val) 
        target = min(target, max_val) 
        return min(target, total) 

    def _create_dataset(self, base, items, type_label) -> EmbeddingComparisonDataset:
        return EmbeddingComparisonDataset(
            name=f"{base.name}__{type_label}__{self.value}",
            description=f"{type_label} pairs selected via {self.label}",
            items=items,
        )

    def _iter_valid_pairs(self, comparisons, *, exclude_self: bool) -> Iterable[EmbeddingComparison]:
        seen_pairs = set()
        for c in comparisons.items:
            e1, e2 = getattr(c, "embedding_1", None), getattr(c, "embedding_2", None)
            h1, h2 = getattr(e1, "face_hash", None), getattr(e2, "face_hash", None)
            n1, n2 = getattr(e1, "file_name", None), getattr(e2, "file_name", None)

            if exclude_self:
                if (h1 is not None and h1 == h2) or (n1 is not None and n1 == n2) or (e1 is e2):
                    continue

            k1 = h1 if h1 else (n1 if n1 else id(e1))
            k2 = h2 if h2 else (n2 if n2 else id(e2))
            pair_key = tuple(sorted((str(k1), str(k2))))

            if pair_key in seen_pairs:
                continue 
            
            seen_pairs.add(pair_key)
            yield c

    def _balance_per_identity(self, impostors, genuines, **kwargs) -> Tuple[List, List]:
        rng = random.Random(kwargs.get('seed', 42))
        imp_groups = defaultdict(list)
        gen_groups = defaultdict(list)

        for c in impostors:
            imp_groups[getattr(c.embedding_1, "identity", "unknown")].append(c)
        for c in genuines:
            gen_groups[getattr(c.embedding_1, "identity", "unknown")].append(c)

        def process(groups, sort_desc, sort_asc, do_shuffle):
            final = []
            for id_key, items in groups.items():
                count = self._calculate_target_count(len(items), kwargs['ratio'], kwargs['min_val'], kwargs['max_val'])
                if sort_desc: items.sort(key=lambda x: float(x.cosine_similarity), reverse=True)
                elif sort_asc: items.sort(key=lambda x: float(x.cosine_similarity), reverse=False)
                elif do_shuffle: rng.shuffle(items)
                final.extend(items[:count])
            return final

        new_imp = process(imp_groups, kwargs['sort_impostors_desc'], False, kwargs['shuffle_impostors'])
        new_gen = process(gen_groups, False, kwargs['sort_genuines_asc'], kwargs['shuffle_genuines'])
        return new_imp, new_gen