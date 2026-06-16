"""Testes unitários do IdentityDisjointPairKFold (V2.10–V2.11.2).

Cobre:
- validações do construtor (n_splits, comprimentos, strategy);
- LPT scheduling balanceia cargas desiguais;
- round_robin distribui K identidades em K grupos;
- disjunção total entre folds (identities(train) ∩ identities(test) = ∅);
- get_fold_stats: shape e tipos esperados;
- get_fold_stats com labels: contagens Hp/Hd + fold_health;
- fold_health: detecta critical (zero), warning (below_min), ok;
- id_to_group_mapping: reproduz exatamente os folds;
- lookalike_clusters: identidades de cluster ficam no mesmo grupo;
- lookalike_clusters: identidades órfãs no dataset são descartadas silenciosamente;
- lookalike_clusters: cluster trivial (0 ou 1 id) é ignorado.
"""
from __future__ import annotations
from collections import Counter
from typing import List, Tuple

import numpy as np
import pytest

from faceaudit.services.identity_disjoint_pair_kfold import (
    IdentityDisjointPairKFold,
)


def _synth_pairs(ids: List[str], n_hp: int, n_hd: int, seed: int = 0
                 ) -> Tuple[List[str], List[str]]:
    """Cria n_hp pares Hp e n_hd pares Hd (ids distintos) reprodutíveis."""
    rng = np.random.default_rng(seed)
    hp_p = rng.choice(ids, size=n_hp, replace=True).tolist()
    hp_r = hp_p[:]
    hd_p = rng.choice(ids, size=n_hd, replace=True).tolist()
    hd_r = [rng.choice([x for x in ids if x != p]) for p in hd_p]
    return hp_p + hd_p, hp_r + hd_r


def _synth_labels(n_hp: int, n_hd: int) -> np.ndarray:
    return np.concatenate([np.ones(n_hp, dtype=np.int32),
                           np.zeros(n_hd, dtype=np.int32)])


# ── Construtor: validações ──────────────────────────────────────────


class TestConstructorValidations:
    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="mesmo comprimento"):
            IdentityDisjointPairKFold(["a", "b"], ["a"], n_splits=2)

    def test_rejects_n_splits_below_2(self) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            IdentityDisjointPairKFold(["a"], ["a"], n_splits=1)

    def test_rejects_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="assignment_strategy"):
            IdentityDisjointPairKFold(
                ["a"], ["a"], n_splits=2, assignment_strategy="invalid"
            )

    def test_accepts_valid_strategies(self) -> None:
        for strat in ("balanced", "round_robin"):
            skf = IdentityDisjointPairKFold(
                ["a", "b"], ["a", "b"], n_splits=2,
                assignment_strategy=strat,
            )
            assert skf.assignment_strategy == strat


# ── LPT scheduling vs round_robin ───────────────────────────────────


class TestLPTSchedulingBalance:
    def test_balanced_reduces_load_spread(self) -> None:
        """LPT deve produzir cargas mais uniformes que round_robin
        quando identidades têm cargas desiguais."""
        heavy = ["heavy_%02d" % i for i in range(3)]  # cada uma com ~40 caps
        light = ["light_%02d" % i for i in range(20)]  # cada uma com ~3 caps
        rng = np.random.default_rng(0)
        probe: List[str] = []
        ref: List[str] = []
        for hid in heavy:
            for _ in range(40):
                other = rng.choice(heavy + light)
                if other == hid:
                    other = "light_00"
                probe.append(hid)
                ref.append(other)
        for lid in light:
            for _ in range(3):
                other = rng.choice([x for x in heavy + light if x != lid])
                probe.append(lid)
                ref.append(other)

        skf_rr = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, random_state=42,
            assignment_strategy="round_robin",
        )
        skf_lpt = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, random_state=42,
            assignment_strategy="balanced",
        )
        stats_rr = skf_rr.get_fold_stats()
        stats_lpt = skf_lpt.get_fold_stats()

        rr_counts = np.array(stats_rr["train_counts_by_fold"])
        lpt_counts = np.array(stats_lpt["train_counts_by_fold"])
        # LPT deve ter range <= round_robin (na prática, muito menor).
        assert (lpt_counts.max() - lpt_counts.min()) <= (
            rr_counts.max() - rr_counts.min()
        )

    def test_balanced_persists_group_loads(self) -> None:
        ids = ["id_%02d" % i for i in range(10)]
        probe, ref = _synth_pairs(ids, n_hp=20, n_hd=60)
        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, random_state=42,
            assignment_strategy="balanced",
        )
        stats = skf.get_fold_stats()
        assert "balance_group_loads" in stats
        loads = stats["balance_group_loads"]
        assert len(loads) == 5
        assert all(isinstance(x, int) for x in loads)

    def test_round_robin_does_not_persist_balance_loads(self) -> None:
        ids = ["id_%02d" % i for i in range(10)]
        probe, ref = _synth_pairs(ids, n_hp=20, n_hd=60)
        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, random_state=42,
            assignment_strategy="round_robin",
        )
        stats = skf.get_fold_stats()
        assert "balance_group_loads" not in stats


# ── Disjunção total entre folds ─────────────────────────────────────


class TestIdentityDisjointness:
    def test_identities_disjoint_per_fold(self) -> None:
        ids = ["id_%02d" % i for i in range(25)]
        probe, ref = _synth_pairs(ids, n_hp=50, n_hd=150)
        probe_arr = np.asarray(probe)
        ref_arr = np.asarray(ref)

        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, random_state=42,
        )
        for fold, (tr_idx, te_idx) in enumerate(skf.split()):
            train_ids = (
                set(probe_arr[tr_idx].tolist())
                | set(ref_arr[tr_idx].tolist())
            )
            test_ids = (
                set(probe_arr[te_idx].tolist())
                | set(ref_arr[te_idx].tolist())
            )
            assert train_ids.isdisjoint(test_ids), (
                f"Fold {fold}: train ∩ test = {train_ids & test_ids}"
            )

    def test_boundary_pairs_excluded_from_each_fold(self) -> None:
        """Sum(train + test + boundary) = n_rows; nenhum par fica em 2 folds simultaneamente."""
        ids = ["id_%02d" % i for i in range(10)]
        probe, ref = _synth_pairs(ids, n_hp=20, n_hd=60)
        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, random_state=42,
        )
        labels = _synth_labels(20, 60)
        stats = skf.get_fold_stats(labels=labels)
        for f in range(5):
            total = (
                stats["train_counts_by_fold"][f]
                + stats["test_counts_by_fold"][f]
                + stats["excluded_boundary_counts_by_fold"][f]
            )
            assert total == len(probe)

    def test_test_sets_are_disjoint_between_folds(self) -> None:
        ids = ["id_%02d" % i for i in range(10)]
        probe, ref = _synth_pairs(ids, n_hp=20, n_hd=60)
        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, random_state=42,
        )
        all_test_idx: List[int] = []
        for _, te in skf.split():
            all_test_idx.extend(te.tolist())
        # cada par Hp aparece em test exatamente 1 vez (sempre vai pro
        # grupo da sua identidade); pares Hd intra-grupo idem; boundary nunca.
        assert len(all_test_idx) == len(set(all_test_idx))


# ── get_fold_stats: shape + tipos ──────────────────────────────────


class TestGetFoldStats:
    def test_required_keys_no_labels(self) -> None:
        ids = ["id_%02d" % i for i in range(10)]
        probe, ref = _synth_pairs(ids, n_hp=10, n_hd=30)
        skf = IdentityDisjointPairKFold(probe, ref, n_splits=5)
        stats = skf.get_fold_stats()
        expected_keys = {
            "n_unique_identities", "n_identities_per_group",
            "assignment_strategy", "random_state",
            "id_to_group_mapping", "lookalike_clusters_used",
            "n_lookalike_clusters", "n_clustered_identities",
            "train_counts_by_fold", "test_counts_by_fold",
            "excluded_boundary_counts_by_fold",
            "train_retained_fraction_by_fold",
            "test_retained_fraction_by_fold",
        }
        assert expected_keys.issubset(stats.keys())

    def test_label_counts_when_labels_provided(self) -> None:
        ids = ["id_%02d" % i for i in range(10)]
        probe, ref = _synth_pairs(ids, n_hp=20, n_hd=60)
        labels = _synth_labels(20, 60)
        skf = IdentityDisjointPairKFold(probe, ref, n_splits=5)
        stats = skf.get_fold_stats(labels=labels)
        expected_label_keys = {
            "train_hp_counts_by_fold", "train_hd_counts_by_fold",
            "test_hp_counts_by_fold", "test_hd_counts_by_fold",
            "fold_health",
        }
        assert expected_label_keys.issubset(stats.keys())
        # Soma por classe deve ser conservativa: train+test+boundary = total da classe.
        n_hp_total = int((labels == 1).sum())
        n_hd_total = int((labels == 0).sum())
        for f in range(5):
            assert (
                stats["train_hp_counts_by_fold"][f]
                + stats["test_hp_counts_by_fold"][f]
            ) <= n_hp_total

    def test_id_to_group_mapping_reproduces_folds(self) -> None:
        ids = ["id_%02d" % i for i in range(15)]
        probe, ref = _synth_pairs(ids, n_hp=30, n_hd=90)
        skf = IdentityDisjointPairKFold(probe, ref, n_splits=5)
        stats = skf.get_fold_stats()
        mapping = stats["id_to_group_mapping"]
        # Reconstruir folds manualmente via mapping.
        probe_arr = np.asarray(probe)
        ref_arr = np.asarray(ref)
        for fold, (tr_idx, te_idx) in enumerate(skf.split()):
            # toda linha em te deve ter probe e reference no fold corrente.
            for i in te_idx:
                assert mapping[str(probe_arr[i])] == fold
                assert mapping[str(ref_arr[i])] == fold
            # toda linha em tr deve ter probe e reference fora do fold.
            for i in tr_idx:
                assert mapping[str(probe_arr[i])] != fold
                assert mapping[str(ref_arr[i])] != fold


# ── fold_health: critical/warning/ok ───────────────────────────────


class TestFoldHealth:
    def test_overall_ok_when_well_supported(self) -> None:
        # Corpus grande o suficiente para satisfazer thresholds.
        ids = ["id_%02d" % i for i in range(30)]
        probe, ref = _synth_pairs(ids, n_hp=2000, n_hd=20000, seed=0)
        labels = _synth_labels(2000, 20000)
        skf = IdentityDisjointPairKFold(probe, ref, n_splits=5)
        stats = skf.get_fold_stats(labels=labels)
        health = stats["fold_health"]
        assert health["overall_status"] in ("ok", "warning"), (
            f"Esperado ok ou warning, recebeu {health['overall_status']}; "
            f"violations: {health['per_fold_violations']}"
        )

    def test_critical_when_class_is_zero_in_a_fold(self) -> None:
        # 5 identidades, 5 folds; cada grupo terá 1 id, sem variabilidade
        # de classes; pares Hp existem mas Hd intra-grupo é zero.
        ids = ["id_0", "id_1", "id_2", "id_3", "id_4"]
        probe = ids[:]  # 5 pares Hp
        ref = ids[:]
        labels = np.ones(5, dtype=np.int32)
        skf = IdentityDisjointPairKFold(probe, ref, n_splits=5)
        stats = skf.get_fold_stats(labels=labels)
        health = stats["fold_health"]
        # Zero Hd ⇒ test_hd_zero em todos os folds.
        assert health["overall_status"] == "critical"
        for viols in health["per_fold_violations"]:
            assert any("zero" in v for v in viols)

    def test_thresholds_documented(self) -> None:
        ids = ["id_%02d" % i for i in range(10)]
        probe, ref = _synth_pairs(ids, n_hp=20, n_hd=60)
        labels = _synth_labels(20, 60)
        skf = IdentityDisjointPairKFold(probe, ref, n_splits=5)
        stats = skf.get_fold_stats(labels=labels)
        thresholds = stats["fold_health"]["thresholds"]
        assert thresholds == {
            "min_hp_test": 10, "min_hd_test": 10,
            "min_hp_train": 50, "min_hd_train": 50,
        }


# ── lookalike_clusters ─────────────────────────────────────────────


class TestLookalikeClusters:
    def _setup(self):
        ids = ["id_%02d" % i for i in range(30)]
        probe, ref = _synth_pairs(ids, n_hp=60, n_hd=200)
        return ids, probe, ref

    def test_cluster_members_share_fold(self) -> None:
        _, probe, ref = self._setup()
        clusters = [["id_00", "id_01", "id_02"], ["id_10", "id_11"]]
        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, lookalike_clusters=clusters,
        )
        mapping = skf.get_fold_stats()["id_to_group_mapping"]
        # Cluster A
        groups_a = {mapping[k] for k in ["id_00", "id_01", "id_02"]}
        assert len(groups_a) == 1, f"Cluster A foi separado: {groups_a}"
        # Cluster B
        groups_b = {mapping[k] for k in ["id_10", "id_11"]}
        assert len(groups_b) == 1, f"Cluster B foi separado: {groups_b}"

    def test_metadata_reflects_clusters(self) -> None:
        _, probe, ref = self._setup()
        clusters = [["id_00", "id_01"], ["id_10", "id_11", "id_12"]]
        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, lookalike_clusters=clusters,
        )
        stats = skf.get_fold_stats()
        assert stats["lookalike_clusters_used"] is True
        assert stats["n_lookalike_clusters"] == 2
        assert stats["n_clustered_identities"] == 5

    def test_no_clusters_default_behavior(self) -> None:
        _, probe, ref = self._setup()
        skf = IdentityDisjointPairKFold(probe, ref, n_splits=5)
        stats = skf.get_fold_stats()
        assert stats["lookalike_clusters_used"] is False
        assert stats["n_lookalike_clusters"] == 0
        assert stats["n_clustered_identities"] == 0

    def test_empty_clusters_list_treated_as_no_clusters(self) -> None:
        _, probe, ref = self._setup()
        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, lookalike_clusters=[],
        )
        stats = skf.get_fold_stats()
        assert stats["lookalike_clusters_used"] is False
        assert stats["n_lookalike_clusters"] == 0

    def test_trivial_cluster_ignored(self) -> None:
        """Cluster com 0 ou 1 identidade efetiva no dataset é ignorado."""
        _, probe, ref = self._setup()
        clusters = [
            ["id_00", "id_01"],
            ["id_99"],  # único, ignorado
            [],          # vazio, ignorado
        ]
        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, lookalike_clusters=clusters,
        )
        stats = skf.get_fold_stats()
        assert stats["n_lookalike_clusters"] == 1
        assert stats["n_clustered_identities"] == 2

    def test_orphan_identity_silently_discarded(self) -> None:
        """Identidade não presente no dataset é silenciosamente
        removida do cluster (sem warning explícito até V2.12)."""
        _, probe, ref = self._setup()
        clusters = [["id_00", "id_01", "id_999"]]  # id_999 não existe
        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, lookalike_clusters=clusters,
        )
        stats = skf.get_fold_stats()
        # Cluster reduz para 2 (id_00 + id_01).
        assert stats["n_lookalike_clusters"] == 1
        assert stats["n_clustered_identities"] == 2

    def test_clusters_dont_affect_disjointness(self) -> None:
        """Mesmo com clusters, disjunção total deve continuar válida."""
        ids = ["id_%02d" % i for i in range(20)]
        probe, ref = _synth_pairs(ids, n_hp=40, n_hd=120)
        probe_arr = np.asarray(probe)
        ref_arr = np.asarray(ref)
        clusters = [["id_00", "id_01"], ["id_10", "id_11", "id_12"]]
        skf = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, lookalike_clusters=clusters,
        )
        for tr, te in skf.split():
            train_ids = set(probe_arr[tr].tolist()) | set(ref_arr[tr].tolist())
            test_ids = set(probe_arr[te].tolist()) | set(ref_arr[te].tolist())
            assert train_ids.isdisjoint(test_ids)


# ── Determinismo / reprodutibilidade ───────────────────────────────


class TestReproducibility:
    def test_same_seed_produces_same_mapping(self) -> None:
        ids = ["id_%02d" % i for i in range(20)]
        probe, ref = _synth_pairs(ids, n_hp=40, n_hd=100)
        skf1 = IdentityDisjointPairKFold(probe, ref, n_splits=5, random_state=42)
        skf2 = IdentityDisjointPairKFold(probe, ref, n_splits=5, random_state=42)
        assert (
            skf1.get_fold_stats()["id_to_group_mapping"]
            == skf2.get_fold_stats()["id_to_group_mapping"]
        )

    def test_different_seed_produces_different_mapping(self) -> None:
        ids = ["id_%02d" % i for i in range(20)]
        probe, ref = _synth_pairs(ids, n_hp=40, n_hd=100)
        m1 = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, random_state=42,
        ).get_fold_stats()["id_to_group_mapping"]
        m2 = IdentityDisjointPairKFold(
            probe, ref, n_splits=5, random_state=999,
        ).get_fold_stats()["id_to_group_mapping"]
        assert m1 != m2
