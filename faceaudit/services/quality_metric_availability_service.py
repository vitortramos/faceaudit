# faceaudit/services/quality_metric_availability_service.py

"""
Helper para descobrir quais métricas de qualidade (FIQA, OFIQ, SERFIQ,
EMBEDDING_NORM) estão disponíveis em um dataset, sem precisar carregar
o dataset inteiro em memória.

Usado pela UI para filtrar dinamicamente o dropdown de `quality_metric`
da instrução DATASET_BALANCING — só mostra métricas que de fato existem
no dataset selecionado.

Estratégia em camadas (mais barata → mais cara):
1. EMBEDDING_NORM: SEMPRE disponível (calculada on-the-fly via
   np.linalg.norm — não requer coluna no parquet).
2. Payload em memória (FaceDataset/EmbeddingDataset): itera os items e
   conta cobertura de cada métrica (FIQA via FaceInfo.ff_info / OFIQ via
   ofiq_info / SERFIQ via serfiq_info — ou via item.quality_scores no
   EmbeddingDataset).
3. Apenas artifact_ref (parquet em disco): leitura colunar via pyarrow
   somente das colunas relevantes (`fiqa_score`, `ofiq_score`,
   `serfiq_score`) — fast path, sem hidratar o dataset inteiro.

Threshold de "disponível": ≥ 50% dos samples com valor não-nulo.
Configurável via parâmetro mas o default é o esperado para uso forense
(garante massa estatística suficiente para ordenação por qualidade).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Set, Union

from faceaudit.enums.quality_score_type import QualityScoreType


_PARQUET_COLUMN_BY_METRIC = {
    QualityScoreType.FIQA: "fiqa_score",
    QualityScoreType.OFIQ: "ofiq_score",
    QualityScoreType.SERFIQ: "serfiq_score",
}


class QualityMetricAvailabilityService:
    """Stateless — todos os métodos são utilitários."""

    DEFAULT_COVERAGE_THRESHOLD = 0.5

    @classmethod
    def available_metrics(
        cls,
        resource: Any,  # StrategyResource (mas evita import circular)
        project_root: Optional[Path] = None,
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    ) -> Set[QualityScoreType]:
        """
        Retorna o conjunto de métricas disponíveis no dataset do recurso.

        Args:
            resource: StrategyResource cujo payload é FaceDataset/EmbeddingDataset
                ou cujo artifact_ref aponta para um parquet em disco.
            project_root: raiz do projeto, necessária quando o payload está
                em disco (artifact_ref.path é relativo).
            coverage_threshold: fração mínima de samples com a métrica
                preenchida para considerá-la "disponível". Default 0.5.

        Returns:
            Conjunto de QualityScoreType disponíveis. Sempre inclui
            EMBEDDING_NORM. Em caso de erro (recurso ausente, parquet
            corrompido), retorna conjunto vazio + EMBEDDING_NORM.
        """
        # EMBEDDING_NORM sempre está disponível.
        available: Set[QualityScoreType] = {QualityScoreType.EMBEDDING_NORM}

        if resource is None:
            return available

        # 1. Payload em memória (caminho mais barato e exato).
        payload = getattr(resource, "payload", None)
        if payload is not None:
            return available | cls._inspect_payload(payload, coverage_threshold)

        # 2. Disco — leitura colunar do parquet (fast path).
        artifact_ref = getattr(resource, "artifact_ref", None)
        if artifact_ref is None or not getattr(artifact_ref, "path", None):
            # Nem payload nem disk — retorna o mínimo (sem filtrar).
            return available

        if project_root is None:
            return available

        abs_path = Path(project_root) / artifact_ref.path
        if not abs_path.exists():
            return available

        try:
            return available | cls._inspect_parquet(abs_path, coverage_threshold)
        except Exception:
            # Falha na leitura do parquet (corrompido, formato inesperado etc).
            # Não bloqueia a UI — retorna o que já temos.
            return available

    # ── Inspetores ────────────────────────────────────────────────────────────

    @classmethod
    def _inspect_payload(
        cls, payload: Any, coverage_threshold: float
    ) -> Set[QualityScoreType]:
        """
        Inspeção em memória — funciona para FaceDataset (FaceInfo) e
        EmbeddingDataset (EmbeddingInfo).
        """
        from faceaudit.models.embedding_dataset import EmbeddingDataset
        from faceaudit.models.face_dataset import FaceDataset

        result: Set[QualityScoreType] = set()

        if isinstance(payload, FaceDataset):
            items = list(payload.faces or [])
            if not items:
                return result
            n = len(items)
            for metric in (QualityScoreType.FIQA,
                           QualityScoreType.OFIQ,
                           QualityScoreType.SERFIQ):
                non_null = sum(
                    1 for face in items
                    if metric.extract_from(face) is not None
                )
                if non_null / n >= coverage_threshold:
                    result.add(metric)
            return result

        if isinstance(payload, EmbeddingDataset):
            items = list(payload.items or [])
            if not items:
                return result
            n = len(items)
            for metric in (QualityScoreType.FIQA,
                           QualityScoreType.OFIQ,
                           QualityScoreType.SERFIQ):
                key = metric.name.lower()
                non_null = sum(
                    1 for it in items
                    if isinstance(getattr(it, "quality_scores", None), dict)
                    and it.quality_scores.get(key) is not None
                )
                if non_null / n >= coverage_threshold:
                    result.add(metric)
            return result

        return result

    @classmethod
    def _inspect_parquet(
        cls, path: Path, coverage_threshold: float
    ) -> Set[QualityScoreType]:
        """
        Leitura colunar via pyarrow — só as colunas de métrica + null_count.
        Custo: O(colunas existentes), não O(N).
        """
        import pyarrow.parquet as pq

        result: Set[QualityScoreType] = set()
        pq_file = pq.ParquetFile(path)
        schema_columns = set(pq_file.schema_arrow.names)
        n_total = pq_file.metadata.num_rows
        if n_total == 0:
            return result

        cols_to_read = []
        col_to_metric = {}
        for metric, col_name in _PARQUET_COLUMN_BY_METRIC.items():
            if col_name in schema_columns:
                cols_to_read.append(col_name)
                col_to_metric[col_name] = metric

        if not cols_to_read:
            return result

        # Lê só as colunas relevantes — evita carregar embeddings.
        tbl = pq.read_table(path, columns=cols_to_read)
        for col_name, metric in col_to_metric.items():
            col = tbl.column(col_name)
            non_null_count = col.length() - col.null_count
            if non_null_count / n_total >= coverage_threshold:
                result.add(metric)

        return result
