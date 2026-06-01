# faceaudit/services/identity_disjoint_pair_kfold.py

"""Splitter de K folds com identidades totalmente disjuntas entre treino
e teste. Para uso no GridSearchCV interno do calibrador LogReg quando
`gridsearch_cv_mode = STRICT_IDENTITY` (V2.10).

Pontos centrais:

1. As identidades únicas (probe ∪ reference) são particionadas em K
   grupos disjuntos uma única vez no construtor, via round-robin sobre
   permutação reprodutível (`random_state`). Cada identidade pertence a
   exatamente um grupo.

2. Para cada fold g (de 0 a K−1):
   - **test_idx**: índices de pares cujas duas identidades pertencem ao
     grupo g (Hp + Hd intra-grupo).
   - **train_idx**: índices de pares cujas duas identidades NÃO
     pertencem ao grupo g (Hp + Hd intra-grupo nos demais grupos).
   - **excluído desse fold** (mas usado em outros): pares boundary com
     uma identidade em g e outra fora — não estão nem em train nem em
     test.

3. Pares Hp (mesma identidade dos dois lados) pertencem trivialmente ao
   grupo dessa identidade.

4. A propriedade chave: para cada fold,
   `identities(train) ∩ identities(test) = ∅` — disjunção total.

5. O conjunto efetivo de calibração não é reduzido globalmente. A
   filtragem é dinâmica por fold; em cada fold, ≈ ((K−1)/K)² do conjunto
   Hd fica disponível no treino, e ≈ 1/K² no teste. Pares boundary são
   excluídos apenas do fold corrente.

6. O fit final do GridSearchCV (com `refit=True`) ou do
   `_select_monotonic_C` é em todo o D_cal — o splitter só governa o CV
   interno de seleção de C.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class IdentityDisjointPairKFold(BaseCrossValidator):
    """K-fold com identidades totalmente disjuntas entre treino e teste.

    Parameters
    ----------
    probe_identities : Sequence[Any]
        Identidade do probe por linha do conjunto de calibração.
    reference_identities : Sequence[Any]
        Identidade da referência por linha do conjunto de calibração.
        Mesma cardinalidade que `probe_identities`.
    n_splits : int
        Número de grupos (folds). Deve ser ≥ 2.
    random_state : int
        Seed para o embaralhamento das identidades antes do round-robin.
    """

    def __init__(
        self,
        probe_identities: Sequence[Any],
        reference_identities: Sequence[Any],
        n_splits: int = 5,
        random_state: int = 42,
        assignment_strategy: str = "balanced",
    ) -> None:
        if len(probe_identities) != len(reference_identities):
            raise ValueError(
                "probe_identities e reference_identities devem ter o "
                "mesmo comprimento."
            )
        if n_splits < 2:
            raise ValueError("n_splits deve ser >= 2.")
        if assignment_strategy not in ("balanced", "round_robin"):
            raise ValueError(
                "assignment_strategy deve ser 'balanced' (LPT scheduling) "
                "ou 'round_robin' (V2.9 legado)."
            )

        self._probe = np.asarray(probe_identities)
        self._reference = np.asarray(reference_identities)
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.assignment_strategy = assignment_strategy

        # Carga por identidade = total de aparições (probe + reference).
        # Importante: identidades com mais capturas geram mais pares e
        # devem ser balanceadas entre grupos para que folds tenham massa
        # similar.
        unique_ids: List[Any] = sorted(
            set(self._probe.tolist()) | set(self._reference.tolist())
        )
        rng = np.random.default_rng(self.random_state)

        if assignment_strategy == "round_robin":
            # V2.9 / V2.10 default original: shuffle + round-robin puro.
            shuffled = np.array(unique_ids, dtype=object)
            rng.shuffle(shuffled)
            id_to_group: dict = {}
            for i, ident in enumerate(shuffled):
                id_to_group[ident] = int(i % self.n_splits)
        else:
            # V2.10 balanced (default): Longest Processing Time (LPT)
            # scheduling — ordena identidades por carga (desc) e atribui
            # cada uma ao grupo com menor carga acumulada. Empates
            # quebrados por embaralhamento prévio reprodutível.
            # Resultado: cargas por grupo dentro de (max_load − min_load)
            # ≤ max_id_load, ótimo conhecido para o problema de
            # particionamento balanceado (Graham 1969 4/3 − 1/(3K)).
            from collections import Counter
            counts = Counter(self._probe.tolist())
            counts.update(self._reference.tolist())
            id_load: List[tuple] = [(ident, counts[ident]) for ident in unique_ids]
            # Embaralha antes pra romper empates de carga de forma estável.
            rng.shuffle(id_load)
            id_load.sort(key=lambda x: -x[1])  # carga desc; estável quebra empate por shuffle

            group_loads = np.zeros(self.n_splits, dtype=np.int64)
            id_to_group = {}
            for ident, load in id_load:
                g = int(np.argmin(group_loads))
                id_to_group[ident] = g
                group_loads[g] += load
            self._balance_group_loads = group_loads.tolist()
        self._id_to_group = id_to_group

        # Grupos de probe e reference por linha — cacheados pra split rápido.
        self._probe_group = np.array(
            [id_to_group[p] for p in self._probe], dtype=np.int64
        )
        self._reference_group = np.array(
            [id_to_group[r] for r in self._reference], dtype=np.int64
        )

        # Estatísticas de auditoria.
        per_group: List[int] = [0] * self.n_splits
        for g in id_to_group.values():
            per_group[g] += 1
        self._n_identities_per_group = per_group
        self._n_unique_identities = len(unique_ids)

    # ── BaseCrossValidator interface ────────────────────────────────

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None) -> Iterator[np.ndarray]:
        n_rows = len(self._probe_group)
        for g in range(self.n_splits):
            test_mask = (self._probe_group == g) & (self._reference_group == g)
            yield np.arange(n_rows, dtype=np.int64)[test_mask]

    def split(
        self,
        X=None,
        y=None,
        groups=None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_rows = len(self._probe_group)
        all_idx = np.arange(n_rows, dtype=np.int64)
        for g in range(self.n_splits):
            test_mask = (self._probe_group == g) & (self._reference_group == g)
            # Train: par com AMBAS as identidades fora do grupo g.
            train_mask = (self._probe_group != g) & (self._reference_group != g)
            # Boundary (uma id em g, outra fora) fica excluído deste fold.
            yield all_idx[train_mask], all_idx[test_mask]

    # ── Auditoria ────────────────────────────────────────────────────

    def get_fold_stats(self, labels: Optional[np.ndarray] = None) -> dict:
        """Estatísticas por fold para persistência em metadata.

        Parameters
        ----------
        labels : Optional[np.ndarray]
            Labels binários (1 = Hp, 0 = Hd). Se fornecido, computa
            contagens Hp/Hd por fold em train/test.
        """
        n_rows = len(self._probe_group)
        train_counts: List[int] = []
        test_counts: List[int] = []
        excluded_boundary_counts: List[int] = []
        train_hp_counts: List[int] = []
        train_hd_counts: List[int] = []
        test_hp_counts: List[int] = []
        test_hd_counts: List[int] = []

        if labels is not None:
            labels_arr = np.asarray(labels)

        for g in range(self.n_splits):
            test_mask = (self._probe_group == g) & (self._reference_group == g)
            train_mask = (self._probe_group != g) & (self._reference_group != g)
            boundary_mask = ~test_mask & ~train_mask

            train_counts.append(int(train_mask.sum()))
            test_counts.append(int(test_mask.sum()))
            excluded_boundary_counts.append(int(boundary_mask.sum()))

            if labels is not None:
                train_hp_counts.append(int((train_mask & (labels_arr == 1)).sum()))
                train_hd_counts.append(int((train_mask & (labels_arr == 0)).sum()))
                test_hp_counts.append(int((test_mask & (labels_arr == 1)).sum()))
                test_hd_counts.append(int((test_mask & (labels_arr == 0)).sum()))

        stats = {
            "n_unique_identities": self._n_unique_identities,
            "n_identities_per_group": list(self._n_identities_per_group),
            "assignment_strategy": self.assignment_strategy,
            "random_state": self.random_state,
            # Mapeamento explícito identidade → grupo (fold de teste).
            # Persistido como dict {identity: group_index} para
            # reprodução exata dos folds sem depender de rerodar com
            # mesmo random_state. Chaves stringificadas para
            # serialização JSON-friendly via JSONSerializableMixin.
            "id_to_group_mapping": {
                str(ident): int(g) for ident, g in self._id_to_group.items()
            },
            "train_counts_by_fold": train_counts,
            "test_counts_by_fold": test_counts,
            "excluded_boundary_counts_by_fold": excluded_boundary_counts,
            "train_retained_fraction_by_fold": [
                float(c) / float(n_rows) if n_rows else 0.0
                for c in train_counts
            ],
            "test_retained_fraction_by_fold": [
                float(c) / float(n_rows) if n_rows else 0.0
                for c in test_counts
            ],
        }
        if hasattr(self, "_balance_group_loads"):
            stats["balance_group_loads"] = list(self._balance_group_loads)

        if labels is not None:
            stats.update(
                {
                    "train_hp_counts_by_fold": train_hp_counts,
                    "train_hd_counts_by_fold": train_hd_counts,
                    "test_hp_counts_by_fold": test_hp_counts,
                    "test_hd_counts_by_fold": test_hd_counts,
                }
            )
            # Health checks formalizados por fold (V2.10).
            health = self._compute_fold_health(
                train_hp=train_hp_counts,
                train_hd=train_hd_counts,
                test_hp=test_hp_counts,
                test_hd=test_hd_counts,
                excluded_boundary=excluded_boundary_counts,
            )
            stats["fold_health"] = health
        return stats

    @staticmethod
    def _compute_fold_health(
        train_hp: List[int],
        train_hd: List[int],
        test_hp: List[int],
        test_hd: List[int],
        excluded_boundary: List[int],
    ) -> dict:
        """Avalia cada fold por critérios mínimos para CV estatisticamente
        sensato. Retorna dict com flags binárias e níveis.

        Critérios (V2.10):
        - `MIN_HP_TEST = 10`: cada fold deve ter ≥ 10 pares Hp no test.
        - `MIN_HD_TEST = 10`: idem para Hd.
        - `MIN_HP_TRAIN = 50`: train do fold deve ter ≥ 50 Hp.
        - `MIN_HD_TRAIN = 50`: idem Hd.
        - Excluded boundary fraction > 50% gera flag de alerta
          (indicação de que muitos pares ficaram fora desse fold).

        Severidade por fold:
        - `ok`: todos os mínimos satisfeitos.
        - `warning`: ≥ 1 mínimo violado mas todas as contagens > 0.
        - `critical`: ≥ 1 contagem == 0 (fold inutilizável).
        """
        MIN_HP_TEST = 10
        MIN_HD_TEST = 10
        MIN_HP_TRAIN = 50
        MIN_HD_TRAIN = 50

        n_folds = len(train_hp)
        per_fold_status: List[str] = []
        per_fold_violations: List[List[str]] = []
        for k in range(n_folds):
            vios: List[str] = []
            if train_hp[k] == 0:
                vios.append("train_hp_zero")
            elif train_hp[k] < MIN_HP_TRAIN:
                vios.append(f"train_hp_below_min({train_hp[k]}<{MIN_HP_TRAIN})")
            if train_hd[k] == 0:
                vios.append("train_hd_zero")
            elif train_hd[k] < MIN_HD_TRAIN:
                vios.append(f"train_hd_below_min({train_hd[k]}<{MIN_HD_TRAIN})")
            if test_hp[k] == 0:
                vios.append("test_hp_zero")
            elif test_hp[k] < MIN_HP_TEST:
                vios.append(f"test_hp_below_min({test_hp[k]}<{MIN_HP_TEST})")
            if test_hd[k] == 0:
                vios.append("test_hd_zero")
            elif test_hd[k] < MIN_HD_TEST:
                vios.append(f"test_hd_below_min({test_hd[k]}<{MIN_HD_TEST})")
            # Boundary excessivo no fold.
            total_in_fold = train_hp[k] + train_hd[k] + test_hp[k] + test_hd[k]
            total_kept = total_in_fold + excluded_boundary[k]
            if total_kept > 0 and excluded_boundary[k] / total_kept > 0.5:
                vios.append(
                    f"excluded_boundary_excessive({excluded_boundary[k]}/{total_kept})"
                )
            critical = any(v.endswith("_zero") for v in vios)
            if critical:
                status = "critical"
            elif vios:
                status = "warning"
            else:
                status = "ok"
            per_fold_status.append(status)
            per_fold_violations.append(vios)

        overall = "ok"
        if any(s == "critical" for s in per_fold_status):
            overall = "critical"
        elif any(s == "warning" for s in per_fold_status):
            overall = "warning"

        return {
            "thresholds": {
                "min_hp_test": MIN_HP_TEST,
                "min_hd_test": MIN_HD_TEST,
                "min_hp_train": MIN_HP_TRAIN,
                "min_hd_train": MIN_HD_TRAIN,
            },
            "per_fold_status": per_fold_status,
            "per_fold_violations": per_fold_violations,
            "overall_status": overall,
        }
        if labels is not None:
            stats.update(
                {
                    "train_hp_counts_by_fold": train_hp_counts,
                    "train_hd_counts_by_fold": train_hd_counts,
                    "test_hp_counts_by_fold": test_hp_counts,
                    "test_hd_counts_by_fold": test_hd_counts,
                }
            )
        return stats
