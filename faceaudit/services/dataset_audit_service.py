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

import numpy as np

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


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Fallback quando `comparison_service` não foi injetado.
    Mesmo cálculo que `ComparisonService._l2_normalize_rows`."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


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

    def audit_exact_duplicates(self, dataset: FaceDataset) -> AuditFinding:
        """Detecta linhas com hash idêntico — duplicatas exatas (PR 24, Tema 5.2).

        Distinto de `audit_hash_conflicts`:
          - `audit_hash_conflicts` flaga apenas quando o mesmo hash aparece
            com identidades DIFERENTES (CRITICAL — bug de rotulagem).
          - `audit_exact_duplicates` flaga qualquer hash repetido,
            independente da identidade — caso típico: a mesma foto
            importada duas vezes na ingestão (WARNING — pode ser bug de
            workflow ou intencional, mas o usuário precisa saber).

        Faces sem hash são ignoradas (sem critério pra dizer se duplicam).
        """
        groups: Dict[str, List[Tuple[int, Any]]] = defaultdict(list)
        for idx, face in enumerate(dataset.faces):
            h = getattr(face, "hash", None)
            if h:
                groups[h].append((idx, face))

        issues: List[AuditIssue] = []
        for h, members in groups.items():
            if len(members) < 2:
                continue
            indices = tuple(idx for idx, _ in members)
            paths = tuple(getattr(f, "image_path", "") for _, f in members)
            ids = sorted({str(getattr(f, "identity", None) or "?") for _, f in members})
            files = [getattr(f, "file_name", None) or "?" for _, f in members]
            issues.append(AuditIssue(
                locators=paths,
                description=(
                    f"{len(members)} linhas com hash idêntico"
                    + (f" (identidade: {ids[0]})" if len(ids) == 1 else f" entre {len(ids)} identidades")
                    + f" — {', '.join(files[:3])}"
                    + ("…" if len(files) > 3 else "")
                ),
                row_indices=indices,
                metadata={"hash": h, "count": len(members), "identities": ids},
            ))

        return AuditFinding(
            audit_id="exact_duplicates",
            audit_label="Duplicatas exatas (mesmo hash)",
            category=CATEGORY_PHYSICAL,
            severity=AuditSeverity.WARNING,
            issues=tuple(issues),
            summary=f"{len(issues)} grupo(s) de duplicatas exatas",
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

    def audit_missing_embeddings(
        self, dataset: FaceDataset,
    ) -> List[AuditFinding]:
        """Faces sem embedding extraído — uma `AuditFinding` por coluna de
        embedding declarada no schema.

        Schema-by-role: itera sobre as colunas do `dataset.column_specs` que
        têm `role=EMBEDDING`, gerando uma auditoria por modelo. Cada
        modelo conhecido (sepaelv2, sepaelv4, serfiq_mean) tem um predicate
        que sabe onde o embedding mora no `FaceInfo`. Custom embedding
        models (model_name desconhecido) emitem finding INFO indicando
        que o auditor não sabe verificar — provavelmente o pipeline
        externo é responsável por essa coluna.

        Fallback legacy: quando `column_specs` é vazio (parquet pré-PR 3),
        comporta-se como antes — 1 finding pra `embedding_sepaelv2`.
        """
        # Caminho legacy: convenção pré-PR 3.
        if not dataset.column_specs:
            return [self._audit_legacy_missing_embedding(dataset)]

        embedding_cols = self._resolve_role_columns(dataset, ColumnRole.EMBEDDING)
        if not embedding_cols:
            # Specs populadas mas sem nenhuma coluna de embedding declarada.
            return [AuditFinding(
                audit_id="missing_embeddings",
                audit_label="Faces sem embedding",
                category=CATEGORY_PHYSICAL,
                severity=AuditSeverity.CRITICAL,
                issues=(),
                summary="Schema do dataset não declara nenhuma coluna de embedding",
                metadata={"reason": "no_embedding_columns_in_schema"},
            )]

        out: List[AuditFinding] = []
        for col_name in embedding_cols:
            spec = dataset.column_specs[col_name]
            model_name = getattr(spec, "model_name", None)
            label = getattr(spec, "label", col_name)

            predicate = self._make_embedding_presence_check(model_name)
            if predicate is None:
                # Modelo desconhecido — não conseguimos verificar via FaceInfo.
                # Emite finding informativo pra avisar o usuário em vez de
                # mentir que está OK ou que está ruim.
                out.append(AuditFinding(
                    audit_id=f"missing_embedding_{col_name}",
                    audit_label=f"Faces sem {label} (verificação indisponível)",
                    category=CATEGORY_PHYSICAL,
                    severity=AuditSeverity.INFO,
                    issues=(),
                    summary=(
                        f"Modelo de embedding '{model_name or '?'}' desconhecido "
                        f"pelo auditor — pulando '{col_name}'"
                    ),
                    metadata={"column_name": col_name, "model_name": model_name},
                ))
                continue

            issues: List[AuditIssue] = []
            for idx, face in enumerate(dataset.faces):
                if not predicate(face):
                    issues.append(AuditIssue(
                        locators=(getattr(face, "image_path", ""),),
                        description=(
                            f"Sem {label} — extração FF/SERFIQ falhou ou não rodou"
                        ),
                        row_indices=(idx,),
                        metadata={
                            "identity": getattr(face, "identity", None),
                            "column_name": col_name,
                            "model_name": model_name,
                        },
                    ))
            out.append(AuditFinding(
                audit_id=f"missing_embedding_{col_name}",
                audit_label=f"Faces sem {label}",
                category=CATEGORY_PHYSICAL,
                severity=AuditSeverity.CRITICAL,
                issues=tuple(issues),
                summary=f"{len(issues)} face(s) sem embedding '{col_name}'",
                metadata={"column_name": col_name, "model_name": model_name},
            ))
        return out

    def _audit_legacy_missing_embedding(
        self, dataset: FaceDataset,
    ) -> AuditFinding:
        """Caminho de retrocompat pra parquets sem `column_specs`.

        Comportamento idêntico ao da versão anterior do `audit_missing_embeddings`:
        verifica `face.ff_info.embedding_sepaelv2` (convenção pré-migração).
        """
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
            audit_label="Faces sem embedding (legacy)",
            category=CATEGORY_PHYSICAL,
            severity=AuditSeverity.CRITICAL,
            issues=tuple(issues),
            summary=f"{len(issues)} face(s) sem embedding primário",
            metadata={"reason": "legacy_no_column_specs"},
        )

    @staticmethod
    def _make_embedding_presence_check(model_name: Optional[str]):
        """Retorna predicate `(face) -> bool` que indica presença do embedding
        do modelo correspondente em `FaceInfo`. None quando o modelo é
        desconhecido (custom externo, plugin não registrado, etc.).

        Os 3 modelos built-in mapeiam pros atributos atuais de FFInfo /
        SERFIQInfo. Quando entrar mais um (ex: AdaFace via outro path),
        adiciona aqui. Pra evitar drift entre auditor e produtor, vale
        consolidar isso num registro central no futuro.
        """
        if not model_name:
            return None
        registry = {
            "sepaelv2": lambda f: bool(
                getattr(getattr(f, "ff_info", None), "embedding_sepaelv2", None) is not None
            ),
            "sepaelv4": lambda f: bool(
                getattr(getattr(f, "ff_info", None), "embedding_sepaelv4", None) is not None
            ),
            "serfiq_mean": lambda f: bool(
                getattr(getattr(f, "serfiq_info", None), "mean_embedding", None) is not None
            ),
        }
        return registry.get(model_name)

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
    # Auditorias O(N²) biométricas (PR 13)
    # =========================================================================

    def audit_duplicate_identities(
        self,
        dataset: FaceDataset,
        threshold: float = 0.85,
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ) -> AuditFinding:
        """Detecta identidades duplicadas — mesma pessoa marcada com 2+ labels.

        Estratégia: computa centróides L2-normalizados por identidade, mede
        cosseno entre centróides e flagra pares acima do threshold. Cada par
        vira uma `AuditIssue` com TODAS as faces das duas identidades nos
        `locators` (até o limite do preview), com `identities` posição-
        alinhadas pra caption por face.

        Custo: O(K²·D) onde K = número de identidades e D = dimensão do
        embedding. Pra K=5000, D=256 leva poucos segundos.
        """
        cb = progress_cb or (lambda *_a: None)
        cb("Convertendo para embedding dataset…", 5)

        emb_ds = dataset.to_embedding_dataset()
        if not emb_ds.items:
            return self._empty_finding(
                "duplicate_identities", "Identidades duplicadas",
                CATEGORY_BIOMETRIC, AuditSeverity.WARNING,
                "Dataset sem embeddings — não foi possível computar centróides",
            )

        cb("Calculando centróides por identidade…", 20)
        centroids, identity_order, faces_by_identity = self._compute_identity_centroids(emb_ds)
        if len(identity_order) < 2:
            return self._empty_finding(
                "duplicate_identities", "Identidades duplicadas",
                CATEGORY_BIOMETRIC, AuditSeverity.WARNING,
                f"Apenas {len(identity_order)} identidade(s) — comparação inviável",
            )

        cb(f"Comparando {len(identity_order)} identidades…", 50)
        # Cosine sim entre centróides já L2-normalizados = produto interno
        sim_matrix = centroids @ centroids.T
        # Zera diagonal pra não comparar identidade consigo mesma
        np.fill_diagonal(sim_matrix, -np.inf)

        cb("Coletando pares acima do threshold…", 80)
        issues: List[AuditIssue] = []
        # Iteração no triângulo superior pra evitar duplicar (a, b) e (b, a)
        n = len(identity_order)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim < threshold:
                    continue
                id_a, id_b = identity_order[i], identity_order[j]
                faces_a = faces_by_identity[id_a]
                faces_b = faces_by_identity[id_b]
                # Locators concatenados (até cap razoável)
                locators = tuple(faces_a + faces_b)
                identities_aligned = (
                    [id_a] * len(faces_a) + [id_b] * len(faces_b)
                )
                issues.append(AuditIssue(
                    locators=locators,
                    description=(
                        f"'{id_a}' ↔ '{id_b}' — centróides com similaridade "
                        f"{sim:.3f} (≥ {threshold:.2f})"
                    ),
                    metadata={
                        "identity_a": id_a,
                        "identity_b": id_b,
                        "similarity": sim,
                        "identities": identities_aligned,
                        "n_faces_a": len(faces_a),
                        "n_faces_b": len(faces_b),
                    },
                ))

        # Ordena issues do mais confuso pro menos confuso
        issues.sort(key=lambda x: -x.metadata.get("similarity", 0.0))

        cb("Concluído.", 100)
        return AuditFinding(
            audit_id="duplicate_identities",
            audit_label="Identidades duplicadas (pessoa em 2+ labels)",
            category=CATEGORY_BIOMETRIC,
            severity=AuditSeverity.WARNING,
            issues=tuple(issues),
            summary=(
                f"{len(issues)} par(es) de identidades com centróide ≥ "
                f"{threshold:.2f}"
            ),
            metadata={"threshold": threshold, "n_identities": n},
        )

    def audit_mixed_identities(
        self,
        dataset: FaceDataset,
        outlier_threshold: float = 0.4,
        min_samples: int = 3,
        progress_cb: Optional[Callable[[str, int], None]] = None,
    ) -> AuditFinding:
        """Detecta identidades misturadas — mesmo label cobrindo 2+ pessoas.

        Estratégia: pra cada identidade com ≥ `min_samples` amostras, computa
        centróide e mede cosseno de cada face vs o próprio centróide. Faces
        com cosseno < `outlier_threshold` são consideradas outliers — provável
        anotação errada (face de outra pessoa rotulada como esta).

        Cada identidade com pelo menos 1 outlier vira uma issue. Locators
        listam outliers PRIMEIRO (visual em destaque) seguidos das típicas
        (até completar 8 — limite do preview), com `locator_labels` posição-
        alinhado distinguindo "outlier" / "típica" pra caption.

        Custo: O(N) onde N = número de faces.
        """
        cb = progress_cb or (lambda *_a: None)
        cb("Convertendo para embedding dataset…", 5)

        emb_ds = dataset.to_embedding_dataset()
        if not emb_ds.items:
            return self._empty_finding(
                "mixed_identities", "Identidades misturadas",
                CATEGORY_BIOMETRIC, AuditSeverity.WARNING,
                "Dataset sem embeddings — não foi possível computar centróides",
            )

        cb("Agrupando faces por identidade…", 15)
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, item in enumerate(emb_ds.items):
            ident = getattr(item, "identity", None)
            if ident:
                groups[str(ident)].append(idx)

        # Filtra só identidades com massa estatística suficiente
        groups = {k: v for k, v in groups.items() if len(v) >= min_samples}
        if not groups:
            return self._empty_finding(
                "mixed_identities", "Identidades misturadas",
                CATEGORY_BIOMETRIC, AuditSeverity.WARNING,
                f"Nenhuma identidade com ≥ {min_samples} amostras — análise inviável",
            )

        cb(f"Analisando outliers em {len(groups)} identidades…", 30)
        matrix = emb_ds.matrix
        norm_matrix = self._comparison._l2_normalize_rows(matrix) if self._comparison else _l2_normalize(matrix)

        issues: List[AuditIssue] = []
        n_processed = 0
        n_total = len(groups)
        for identity, idx_list in groups.items():
            n_processed += 1
            if n_processed % max(1, n_total // 10) == 0:
                pct = 30 + int(60 * n_processed / n_total)
                cb(f"Identidade {n_processed}/{n_total}…", min(pct, 90))

            embs = norm_matrix[idx_list]
            centroid = embs.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)

            # Cosine de cada face com o centróide
            sims = embs @ centroid

            outlier_local_idx = np.where(sims < outlier_threshold)[0]
            if len(outlier_local_idx) == 0:
                continue

            # Ordena outliers do pior pro melhor (mais distantes primeiro)
            outlier_local_idx = outlier_local_idx[np.argsort(sims[outlier_local_idx])]
            # Paths das outliers (em ordem)
            outlier_paths = [
                getattr(emb_ds.items[idx_list[i]], "image_path", "")
                for i in outlier_local_idx
            ]
            # Paths típicas pra contraste visual — top-2 mais alinhadas
            typical_local_idx = np.argsort(-sims)[:2]
            typical_paths = [
                getattr(emb_ds.items[idx_list[i]], "image_path", "")
                for i in typical_local_idx
                if sims[i] >= outlier_threshold  # de fato típica
            ]

            locators = tuple(outlier_paths + typical_paths)
            labels = (
                ["outlier"] * len(outlier_paths)
                + ["típica"] * len(typical_paths)
            )

            issues.append(AuditIssue(
                locators=locators,
                description=(
                    f"'{identity}' — {len(outlier_local_idx)}/{len(idx_list)} "
                    f"face(s) com cos < {outlier_threshold:.2f} ao centróide"
                ),
                metadata={
                    "identity": identity,
                    "n_outliers": int(len(outlier_local_idx)),
                    "n_total": len(idx_list),
                    "min_similarity": float(sims.min()),
                    "mean_similarity": float(sims.mean()),
                    "locator_labels": labels,
                },
            ))

        # Ordena issues por severidade — mais outliers em primeiro
        issues.sort(key=lambda x: -x.metadata.get("n_outliers", 0))

        cb("Concluído.", 100)
        return AuditFinding(
            audit_id="mixed_identities",
            audit_label="Identidades misturadas (label cobre 2+ pessoas)",
            category=CATEGORY_BIOMETRIC,
            severity=AuditSeverity.WARNING,
            issues=tuple(issues),
            summary=(
                f"{len(issues)} identidade(s) com outliers intra-classe "
                f"(cos < {outlier_threshold:.2f})"
            ),
            metadata={
                "outlier_threshold": outlier_threshold,
                "min_samples": min_samples,
            },
        )

    def _compute_identity_centroids(self, emb_dataset):
        """Helper compartilhado: agrupa embeddings por identidade, retorna
        matriz de centróides L2-normalizados + ordem das identidades + dict
        identidade → list de image_paths.
        """
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, item in enumerate(emb_dataset.items):
            ident = getattr(item, "identity", None)
            if ident:
                groups[str(ident)].append(idx)

        identity_order = sorted(groups.keys())
        if not identity_order:
            return np.zeros((0, 0)), [], {}

        matrix = emb_dataset.matrix
        norm_matrix = (
            self._comparison._l2_normalize_rows(matrix)
            if self._comparison else _l2_normalize(matrix)
        )

        D = matrix.shape[1]
        centroids = np.zeros((len(identity_order), D), dtype=np.float64)
        faces_by_identity: Dict[str, List[str]] = {}
        for i, ident in enumerate(identity_order):
            idxs = groups[ident]
            centroids[i] = norm_matrix[idxs].mean(axis=0)
            faces_by_identity[ident] = [
                getattr(emb_dataset.items[k], "image_path", "") for k in idxs
            ]

        # L2 normaliza os próprios centróides pra que produto interno = cosseno
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        centroids = centroids / norms

        return centroids, identity_order, faces_by_identity

    @staticmethod
    def _empty_finding(
        audit_id: str, label: str, category: str,
        severity: AuditSeverity, summary: str,
    ) -> AuditFinding:
        """Atalho pra finding sem issues — usado nos casos limite."""
        return AuditFinding(
            audit_id=audit_id, audit_label=label, category=category,
            severity=severity, issues=(), summary=summary,
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
        # PR 24 (Tema 5.2) — duplicatas exatas: mesmo hash em rows
        # distintos, independente da identidade. Complementa
        # `audit_hash_conflicts` (que cobre só conflitos cross-identity).
        out.append(self.audit_exact_duplicates(dataset))
        # missing_embeddings agora retorna List[AuditFinding] — uma por
        # coluna de embedding declarada no schema.
        out.extend(self.audit_missing_embeddings(dataset))
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
