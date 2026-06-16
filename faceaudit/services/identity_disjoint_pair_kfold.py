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
    assignment_strategy : str
        `"balanced"` (default V2.10 — LPT scheduling) ou
        `"round_robin"` (legado V2.9).
    lookalike_clusters : Optional[Sequence[Sequence[Any]]]
        V2.11.2 — lista de clusters de identidades parecidas
        (lookalikes). Identidades de um mesmo cluster são tratadas
        como **unidade atômica** na atribuição a grupos: todas
        recebem o mesmo `group_index`, evitando que um par Hd
        entre lookalikes apareça com uma identidade no train e
        outra no test (leakage por similaridade visual). A carga
        do cluster no LPT scheduling é o somatório das cargas
        individuais. Identidades não declaradas em nenhum cluster
        funcionam como clusters singletons. Default: None
        (comportamento V2.10).
    """

    def __init__(
        self,
        probe_identities: Sequence[Any],
        reference_identities: Sequence[Any],
        n_splits: int = 5,
        random_state: int = 42,
        assignment_strategy: str = "balanced",
        lookalike_clusters: Optional[Sequence[Sequence[Any]]] = None,
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
        unique_ids: List[Any] = sorted(
            set(self._probe.tolist()) | set(self._reference.tolist())
        )
        rng = np.random.default_rng(self.random_state)

        # ── V2.11.2: super-identidades para lookalike clusters ──────
        # Cada cluster de lookalikes vira uma "super-identidade" com
        # carga = soma das cargas das identidades membros. Identidades
        # não-clusterizadas funcionam como singletons. O LPT scheduling
        # opera sobre super-identidades; cada identidade individual
        # herda o grupo da sua super.
        from collections import Counter
        counts = Counter(self._probe.tolist())
        counts.update(self._reference.tolist())

        id_to_super: dict = {}
        super_to_members: dict = {}
        n_clustered_identities = 0
        n_clusters = 0
        if lookalike_clusters:
            for i, cluster in enumerate(lookalike_clusters):
                if not cluster:
                    continue
                # Apenas identidades realmente presentes no dataset entram.
                members_present = [m for m in cluster if m in counts]
                if len(members_present) < 2:
                    # cluster com 0 ou 1 identidade efetiva é trivial.
                    continue
                super_id = f"__cluster_{i}__"
                super_to_members[super_id] = list(members_present)
                for ident in members_present:
                    id_to_super[ident] = super_id
                n_clusters += 1
                n_clustered_identities += len(members_present)

        # Identidades não-clusterizadas viram super-ids singletons.
        for ident in unique_ids:
            if ident not in id_to_super:
                id_to_super[ident] = ident
                super_to_members[ident] = [ident]

        unique_supers = sorted(super_to_members.keys(), key=str)
        super_loads_input = {
            sup: sum(counts[m] for m in members)
            for sup, members in super_to_members.items()
        }

        self._n_lookalike_clusters = n_clusters
        self._n_clustered_identities = n_clustered_identities
        self._lookalike_used = lookalike_clusters is not None and n_clusters > 0

        if assignment_strategy == "round_robin":
            # V2.9 / V2.10 round-robin sobre super-ids.
            shuffled = np.array(unique_supers, dtype=object)
            rng.shuffle(shuffled)
            super_to_group: dict = {}
            for i, sup in enumerate(shuffled):
                super_to_group[sup] = int(i % self.n_splits)
        else:
            # V2.10 balanced (default): LPT scheduling sobre super-ids
            # com carga = soma das cargas das identidades membros.
            super_load: List[tuple] = [
                (sup, super_loads_input[sup]) for sup in unique_supers
            ]
            rng.shuffle(super_load)
            super_load.sort(key=lambda x: -x[1])

            group_loads = np.zeros(self.n_splits, dtype=np.int64)
            super_to_group = {}
            for sup, load in super_load:
                g = int(np.argmin(group_loads))
                super_to_group[sup] = g
                group_loads[g] += load
            self._balance_group_loads = group_loads.tolist()

        # Expande para mapeamento identidade → grupo.
        id_to_group: dict = {}
        for ident in unique_ids:
            id_to_group[ident] = super_to_group[id_to_super[ident]]
        self._id_to_group = id_to_group
        self._super_to_group = super_to_group
        self._id_to_super = id_to_super

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
            # V2.11.2 — clusters de lookalikes tratados como unidade atômica.
            "lookalike_clusters_used": self._lookalike_used,
            "n_lookalike_clusters": self._n_lookalike_clusters,
            "n_clustered_identities": self._n_clustered_identities,
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
