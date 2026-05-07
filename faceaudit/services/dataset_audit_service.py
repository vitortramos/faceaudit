"""Auditoria de datasets — detecções backend reaproveitadas pela UI e pelas
validation rules da V2.

`DatasetAuditService` é stateless. Cada método de auditoria recebe um
`FaceDataset` e retorna `AuditFinding` (ou `List[AuditFinding]` quando
produz mais de uma). Composição via `audit_all_quick` (só auditorias O(N))
e `audit_all_complete` (incluindo as O(N²) — definidas em PRs futuros).

Schema-by-role: quando o dataset traz `column_specs` (formato pós-PR 3),
as auditorias usam o role index pra encontrar colunas (embedding, quality,
etc.). Pra parquets legados sem specs, fallback de convenção (nomes
literais como `"embedding_v2"`, `"fiqa_score"`).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.models.audit import AuditFinding, AuditIssue, AuditSeverity
from faceaudit.models.column_catalog import ColumnRole, build_role_index
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.services.comparison_service import ComparisonService
from faceaudit.services.dataset_stats_service import DatasetStatsService


# Categorias canônicas usadas no campo `AuditFinding.category` —
# a UI agrupa nós da árvore por essas strings.
CATEGORY_PHYSICAL = "Integridade Física"
CATEGORY_LABELS = "Integridade Lógica (rótulos)"
CATEGORY_QUALITY = "Qualidade das Amostras"
CATEGORY_BIOMETRIC = "Auditoria Biométrica"


class DatasetAuditService:
    """Detecta problemas em `FaceDataset` agrupando por categoria.

    Stateless — sem cache na instância. Dependências injetadas via
    construtor pra permitir mocking em testes (ex: `path_exists` pode ser
    substituído pra simular file system).
    """

    def __init__(
        self,
        comparison_service: Optional[ComparisonService] = None,
        stats_service: Optional[DatasetStatsService] = None,
        hubness_factory: Optional[Callable[..., Any]] = None,
        path_exists: Callable[[str], bool] = os.path.exists,
    ):
        # Comparison/stats podem ser None nas auditorias O(N) — só são
        # exigidos pelas O(N²) (PR 13). Os métodos que precisam validam.
        self._comparison = comparison_service
        self._stats = stats_service
        self._hubness_factory = hubness_factory
        self._path_exists = path_exists

    # =========================================================================
    # Auditorias O(N) baratas
    # =========================================================================

    def audit_orphan_paths(self, dataset: FaceDataset) -> AuditFinding:
        """Lista faces cujo `image_path` não existe no filesystem."""
        issues: List[AuditIssue] = []
        for idx, face in enumerate(dataset.faces):
            path = getattr(face, "image_path", None)
            if not path or not self._path_exists(path):
                issues.append(AuditIssue(
                    locators=(path or "",),
                    description=f"Arquivo não encontrado: {path or '(sem path)'}",
                    row_indices=(idx,),
                    metadata={"identity": getattr(face, "identity", None)},
                ))
        return AuditFinding(
            audit_id="orphan_paths",
            audit_label="Paths órfãos",
            category=CATEGORY_PHYSICAL,
            severity=AuditSeverity.CRITICAL,
            issues=tuple(issues),
            summary=f"{len(issues)} face(s) com path inexistente em disco",
        )

    def audit_hash_conflicts(self, dataset: FaceDataset) -> AuditFinding:
        """Detecta hash duplicado com identidades diferentes (bug real)."""
        groups: Dict[str, List[Tuple[int, Any]]] = defaultdict(list)
        for idx, face in enumerate(dataset.faces):
            h = getattr(face, "hash", None)
            if h:
                groups[h].append((idx, face))

        issues: List[AuditIssue] = []
        for h, members in groups.items():
            if len(members) < 2:
                continue
            ids_seen = {getattr(f, "identity", None) for _, f in members}
            ids_seen.discard(None)
            if len(ids_seen) <= 1:
                continue  # mesma identidade, mesma foto → benigno
            indices = tuple(idx for idx, _ in members)
            paths = tuple(getattr(f, "image_path", "") for _, f in members)
            issues.append(AuditIssue(
                locators=paths,
                description=(
                    f"Hash compartilhado entre {len(ids_seen)} identidades: "
                    f"{', '.join(sorted(ids_seen))}"
                ),
                row_indices=indices,
                metadata={"hash": h, "identities": sorted(ids_seen)},
            ))
        return AuditFinding(
            audit_id="hash_conflicts",
            audit_label="Hash compartilhado entre identidades distintas",
            category=CATEGORY_PHYSICAL,
            severity=AuditSeverity.CRITICAL,
            issues=tuple(issues),
            summary=f"{len(issues)} grupo(s) de hash conflitante",
        )

    def audit_singleton_identities(
        self, dataset: FaceDataset, min_samples: int = 2,
    ) -> AuditFinding:
        """Identidades com menos de `min_samples` amostras — não podem ser usadas
        em LR forense (split sem genuíno cross-foto, balanceamento quebra)."""
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, face in enumerate(dataset.faces):
            identity = getattr(face, "identity", None)
            if identity:
                groups[str(identity)].append(idx)

        issues: List[AuditIssue] = []
        for identity, idx_list in groups.items():
            if len(idx_list) < min_samples:
                paths = tuple(
                    getattr(dataset.faces[i], "image_path", "")
                    for i in idx_list
                )
                issues.append(AuditIssue(
                    locators=paths,
                    description=(
                        f"Identidade '{identity}' tem apenas {len(idx_list)} "
                        f"amostra(s) (mínimo: {min_samples})"
                    ),
                    row_indices=tuple(idx_list),
                    metadata={"identity": identity, "n_samples": len(idx_list)},
                ))
        return AuditFinding(
            audit_id="singleton_identities",
            audit_label=f"Identidades com < {min_samples} amostra(s)",
            category=CATEGORY_LABELS,
            severity=AuditSeverity.WARNING,
            issues=tuple(issues),
            summary=f"{len(issues)} identidade(s) com amostras insuficientes",
            metadata={"min_samples": min_samples},
        )

    def audit_missing_embeddings(self, dataset: FaceDataset) -> AuditFinding:
        """Faces sem embedding extraído — não podem ir pra estratégia."""
        # Convenção: face.ff_info.embedding_sepaelv2 é o embedding primário.
        # Quando falta, FF não rodou ou a face foi rejeitada na extração.
        issues: List[AuditIssue] = []
        for idx, face in enumerate(dataset.faces):
            ff = getattr(face, "ff_info", None)
            emb = getattr(ff, "embedding_sepaelv2", None) if ff else None
            if emb is None:
                issues.append(AuditIssue(
                    locators=(getattr(face, "image_path", ""),),
                    description="Sem embedding sepaelv2 — extração FF falhou ou não rodou",
                    row_indices=(idx,),
                    metadata={"identity": getattr(face, "identity", None)},
                ))
        return AuditFinding(
            audit_id="missing_embeddings",
            audit_label="Faces sem embedding",
            category=CATEGORY_PHYSICAL,
            severity=AuditSeverity.CRITICAL,
            issues=tuple(issues),
            summary=f"{len(issues)} face(s) sem embedding primário",
        )

    def audit_incomplete_enrichment(
        self, dataset: FaceDataset,
    ) -> List[AuditFinding]:
        """Cobertura incompleta dos enrichers opcionais (OFIQ, SERFIQ).

        Retorna uma `AuditFinding` por enricher faltando em pelo menos uma
        face. Sem severity CRITICAL — esses enrichers são opcionais; só
        sinaliza pro usuário decidir se quer rodar.
        """
        out: List[AuditFinding] = []

        # Cada entry: (audit_id, audit_label, attribute_path, summary_template)
        enrichers = [
            ("missing_ofiq", "Faces sem OFIQ",
             lambda f: getattr(f, "ofiq_info", None) is not None),
            ("missing_serfiq", "Faces sem SER-FIQ",
             lambda f: getattr(f, "serfiq_info", None) is not None),
        ]

        for audit_id, label, predicate in enrichers:
            issues: List[AuditIssue] = []
            for idx, face in enumerate(dataset.faces):
                if not predicate(face):
                    issues.append(AuditIssue(
                        locators=(getattr(face, "image_path", ""),),
                        description=f"{label.lower()} para esta face",
                        row_indices=(idx,),
                    ))
            out.append(AuditFinding(
                audit_id=audit_id,
                audit_label=label,
                category=CATEGORY_QUALITY,
                severity=AuditSeverity.INFO,
                issues=tuple(issues),
                summary=f"{len(issues)} face(s) sem este enrichment",
            ))
        return out

    def audit_low_quality(
        self,
        dataset: FaceDataset,
        metric: QualityScoreType = QualityScoreType.FIQA,
        threshold: float = 30.0,
    ) -> AuditFinding:
        """Faces com quality score abaixo do threshold.

        Métrica resolvida via `QualityScoreType.column_name_in(specs)` quando
        o dataset tem `column_specs`; fallback ao extractor literal do enum
        (`extract_from(face)`) pra parquets legados.
        """
        # Determina label da coluna pra mensagem
        col_name: Optional[str] = None
        if dataset.column_specs:
            col_name = metric.column_name_in(dataset.column_specs)

        issues: List[AuditIssue] = []
        for idx, face in enumerate(dataset.faces):
            value = metric.extract_from(face)
            if value is None:
                continue  # ausente ≠ baixa qualidade — outra auditoria cobre
            if value < threshold:
                issues.append(AuditIssue(
                    locators=(getattr(face, "image_path", ""),),
                    description=(
                        f"{metric.display_name}={value:.2f} < {threshold:.2f}"
                    ),
                    row_indices=(idx,),
                    metadata={"value": float(value), "metric": metric.name},
                ))

        col_suffix = f" ({col_name})" if col_name else ""
        return AuditFinding(
            audit_id=f"low_quality_{metric.name.lower()}",
            audit_label=f"{metric.display_name} < {threshold:g}{col_suffix}",
            category=CATEGORY_QUALITY,
            severity=AuditSeverity.WARNING,
            issues=tuple(issues),
            summary=f"{len(issues)} face(s) abaixo do threshold {threshold:g}",
            metadata={
                "metric": metric.name,
                "threshold": float(threshold),
                "column_name": col_name,
            },
        )

    # =========================================================================
    # Composição
    # =========================================================================

    def audit_all_quick(
        self,
        dataset: FaceDataset,
        *,
        singleton_min_samples: int = 2,
        low_quality_metric: QualityScoreType = QualityScoreType.FIQA,
        low_quality_threshold: float = 30.0,
    ) -> List[AuditFinding]:
        """Roda todas as auditorias O(N) em sequência. Custo total proporcional
        ao tamanho do dataset (sem comparações entre faces)."""
        out: List[AuditFinding] = []
        out.append(self.audit_orphan_paths(dataset))
        out.append(self.audit_hash_conflicts(dataset))
        out.append(self.audit_missing_embeddings(dataset))
        out.append(self.audit_singleton_identities(dataset, min_samples=singleton_min_samples))
        out.extend(self.audit_incomplete_enrichment(dataset))
        out.append(self.audit_low_quality(
            dataset, metric=low_quality_metric, threshold=low_quality_threshold,
        ))
        return out

    # =========================================================================
    # Helpers internos (compartilhados com auditorias O(N²) em PRs futuros)
    # =========================================================================

    def _resolve_role_columns(
        self, dataset: FaceDataset, role: ColumnRole,
    ) -> List[str]:
        """Devolve nomes de coluna do dataset que carregam o role pedido.

        Quando `column_specs` é vazio (parquet legado), retorna lista vazia
        — o caller decide o fallback (usar nomes convencionais).
        """
        if not dataset.column_specs:
            return []
        index = build_role_index(dataset.column_specs)
        return list(index.get(role, []))
