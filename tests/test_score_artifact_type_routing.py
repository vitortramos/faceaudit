# tests/test_score_artifact_type_routing.py
"""StrategyResource.to_artifact_type() rota outputs de score pra subpastas
distintas com base no metadata do payload:

    payload.metadata.is_score_vector=True → VECTORIZED_SCORES
    payload.metadata.is_mined=True        → MINED_SCORES
    (nenhum dos dois)                     → SIMILARITY_SCORES (raw)

Esse roteamento garante que `scores/<sid>/{raw,mined,vectorized}/` sejam
populados pelas instruções corretas (SIMILARITY_SCORING / PAIR_MINING /
SCORE_VECTORIZATION) sem que cada executor precise carimbar o
artifact_type explicitamente.
"""

from __future__ import annotations

from faceaudit.enums.resource_category import ResourceCategory
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.strategy.strategy_resource import StrategyResource


def _make_score_dataset(**metadata) -> SimilarityScoreDataset:
    """Cria um SimilarityScoreDataset vazio com metadata custom — suficiente
    pra testar o roteamento (não precisa de items reais)."""
    ds = SimilarityScoreDataset(name="test")
    ds.metadata.update(metadata)
    return ds


def _make_derived_resource(payload) -> StrategyResource:
    return StrategyResource(
        resource_id="SC-1",
        name="SC-1",
        resource_type=StrategyResourceType.SIMILARITY_SCORES,
        payload=payload,
        artifact_ref=None,
        category=ResourceCategory.DERIVED,
    )


class TestScoreArtifactTypeRouting:

    def test_default_raw_scores(self):
        """Sem flags de metadata, default é SIMILARITY_SCORES (raw)."""
        ds = _make_score_dataset()
        resource = _make_derived_resource(ds)
        assert resource.to_artifact_type() == StrategyArtifactType.SIMILARITY_SCORES

    def test_mined_flag_routes_to_mined_scores(self):
        """metadata.is_mined=True → MINED_SCORES (scores/<sid>/mined/)."""
        ds = _make_score_dataset(is_mined=True)
        resource = _make_derived_resource(ds)
        assert resource.to_artifact_type() == StrategyArtifactType.MINED_SCORES

    def test_score_vector_flag_routes_to_vectorized_scores(self):
        """metadata.is_score_vector=True → VECTORIZED_SCORES
        (scores/<sid>/vectorized/)."""
        ds = _make_score_dataset(is_score_vector=True)
        resource = _make_derived_resource(ds)
        assert resource.to_artifact_type() == StrategyArtifactType.VECTORIZED_SCORES

    def test_score_vector_takes_priority_over_mined(self):
        """Caso patológico mas defensivo: se ambas as flags estão setadas,
        is_score_vector vence (vectorized é mais específico)."""
        ds = _make_score_dataset(is_score_vector=True, is_mined=True)
        resource = _make_derived_resource(ds)
        assert resource.to_artifact_type() == StrategyArtifactType.VECTORIZED_SCORES

    def test_primary_returns_none(self):
        """PRIMARY não tem destino físico via executor (já está em disco)."""
        ds = _make_score_dataset()
        resource = StrategyResource(
            resource_id="SC-PRIM",
            name="SC-PRIM",
            resource_type=StrategyResourceType.SIMILARITY_SCORES,
            payload=ds,
            artifact_ref=None,
            category=ResourceCategory.PRIMARY,
        )
        assert resource.to_artifact_type() is None

    def test_no_payload_returns_base(self):
        """Resource sem payload hidratado retorna o base type (raw) — não
        explode com AttributeError. Defensivo: pode acontecer quando o
        StrategyResource ainda é um placeholder."""
        resource = StrategyResource(
            resource_id="SC-NOPAY",
            name="SC-NOPAY",
            resource_type=StrategyResourceType.SIMILARITY_SCORES,
            payload=None,
            artifact_ref=None,
            category=ResourceCategory.DERIVED,
        )
        # Sem payload, default = SIMILARITY_SCORES (raw)
        assert resource.to_artifact_type() == StrategyArtifactType.SIMILARITY_SCORES

    def test_calibrated_scores_unaffected(self):
        """CALIBRATED_SCORES não passa pelo override — vai sempre pra
        scores/<sid>/calibrated/ via o map base."""
        ds = _make_score_dataset()
        resource = StrategyResource(
            resource_id="SC-CAL",
            name="SC-CAL",
            resource_type=StrategyResourceType.CALIBRATED_SCORES,
            payload=ds,
            artifact_ref=None,
            category=ResourceCategory.DERIVED,
        )
        assert resource.to_artifact_type() == StrategyArtifactType.CALIBRATED_SCORES
