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
from pathlib import Path as _Path
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
CATEGORY_HISTOGRAMS = "Histogramas"
CATEGORY_CORRELATIONS = "Correlações"
CATEGORY_BALANCE = "Balanceamento"
CATEGORY_HUBNESS = "Hubness"
CATEGORY_VISUALIZATION = "Visualização"


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
        """Lista faces cujo `image_path` não existe no filesystem.

        Caso especial: se 100% das faces têm path vazio, o dataset não
        referencia imagens (foi gerado a partir de embeddings extraídos
        sem manter os arquivos originais — `hash_skipped=true` no audit
        log). Em vez de gerar 1 issue por face (entulha tabela e tree),
        emite UM finding informativo único.
        """
        n_total = len(dataset.faces)
        if n_total == 0:
            return AuditFinding(
                audit_id="orphan_paths",
                audit_label="Paths órfãos",
                category=CATEGORY_PHYSICAL,
                severity=AuditSeverity.CRITICAL,
                issues=(),
                summary="0 face(s) — dataset vazio",
            )

        n_empty_paths = 0
        n_orphan = 0
        issues: List[AuditIssue] = []
        for idx, face in enumerate(dataset.faces):
            raw = getattr(face, "image_path", None)
            if not raw:
                n_empty_paths += 1
                continue
            # Resolve via dataset — em modo RELATIVE o `face.image_path`
            # é só o nome do arquivo (ex: "0010_01.png") e precisa ser
            # ancorado em `_parquet_dir/path_root` pra checar
            # existência. Sem isso, datasets do Image Pipeline
            # apareciam com 100% das linhas marcadas como órfãs.
            resolved = str(dataset.resolve_image_path(face))
            if not self._path_exists(resolved):
                n_orphan += 1
                issues.append(AuditIssue(
                    locators=(resolved,),
                    description=f"Arquivo não encontrado: {resolved}",
                    row_indices=(idx,),
                    metadata={"identity": getattr(face, "identity", None)},
                ))

        # Caso "dataset sem paths" — 100% dos paths vazios → finding
        # único informativo, severity INFO (não CRITICAL — não é bug
        # do dataset, é uma escolha de não armazenar os paths).
        if n_empty_paths == n_total:
            return AuditFinding(
                audit_id="orphan_paths",
                audit_label="Dataset sem paths de imagem",
                category=CATEGORY_PHYSICAL,
                severity=AuditSeverity.INFO,
                issues=(),
                summary=(
                    f"{n_total:,} face(s) sem path armazenado — "
                    "dataset gerado a partir de embeddings já extraídos. "
                    "Inspeção visual de imagens não disponível."
                ),
                metadata={
                    "reason": "all_paths_empty",
                    "n_faces": n_total,
                },
            )

        # Caminho normal: lista os paths quebrados (e flag os vazios
        # parciais como 1 issue agregada se houver alguns).
        if n_empty_paths > 0:
            issues.append(AuditIssue(
                locators=("",),
                description=(
                    f"{n_empty_paths:,} face(s) sem path armazenado "
                    "(agregado — não é possível listar individualmente)"
                ),
                metadata={"n_empty_paths": n_empty_paths},
            ))

        return AuditFinding(
            audit_id="orphan_paths",
            audit_label="Paths órfãos",
            category=CATEGORY_PHYSICAL,
            severity=AuditSeverity.CRITICAL,
            issues=tuple(issues),
            summary=(
                f"{n_orphan:,} face(s) com path inexistente em disco"
                + (f" + {n_empty_paths:,} sem path" if n_empty_paths else "")
            ),
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
            # Arrays paralelos a `locators` — view consome pra renderizar
            # colunas customizadas (ID1/Foto1/ID2/Foto2/Hash) e o helper
            # `_captions_for_issue` mostra a identidade certa em cada thumb.
            identities_aligned = [
                str(getattr(f, "identity", "") or "") for _, f in members
            ]
            file_names_aligned = [
                str(getattr(f, "file_name", "") or "") for _, f in members
            ]
            issues.append(AuditIssue(
                locators=paths,
                description=(
                    f"Hash compartilhado entre {len(ids_seen)} identidades: "
                    f"{', '.join(sorted(ids_seen))}"
                ),
                row_indices=indices,
                metadata={
                    "hash": h,
                    "identities": identities_aligned,        # parallel
                    "file_names": file_names_aligned,        # parallel
                    "unique_identities": sorted(ids_seen),   # set de IDs
                },
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
            # Arrays paralelos a `locators` — view consome pra renderizar
            # colunas customizadas e captions corretas no preview.
            identities_aligned = [
                str(getattr(f, "identity", "") or "") for _, f in members
            ]
            file_names_aligned = [
                str(getattr(f, "file_name", "") or "") for _, f in members
            ]
            unique_ids = sorted(set(identities_aligned) - {""})
            issues.append(AuditIssue(
                locators=paths,
                description=(
                    f"{len(members)} linhas com hash idêntico"
                    + (
                        f" (identidade: {unique_ids[0]})"
                        if len(unique_ids) == 1
                        else f" entre {len(unique_ids)} identidades"
                    )
                    + f" — {', '.join(file_names_aligned[:3])}"
                    + ("…" if len(file_names_aligned) > 3 else "")
                ),
                row_indices=indices,
                metadata={
                    "hash": h,
                    "count": len(members),
                    "identities": identities_aligned,        # parallel
                    "file_names": file_names_aligned,        # parallel
                    "unique_identities": unique_ids,
                },
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
        sinaliza pro usuário decidir se quer rodar. Categoria é PHYSICAL
        (integridade física do dataset) — escore baixo é QUALITY, mas
        ausência total do dado é PHYSICAL.
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
                        metadata={
                            "identity": str(getattr(face, "identity", "") or ""),
                        },
                    ))
            out.append(AuditFinding(
                audit_id=audit_id,
                audit_label=label,
                category=CATEGORY_PHYSICAL,
                severity=AuditSeverity.INFO,
                issues=tuple(issues),
                summary=f"{len(issues)} face(s) sem este enrichment",
            ))
        return out

    def audit_histograms(
        self, dataset: FaceDataset,
    ) -> List[AuditFinding]:
        """Coleta valores de métricas de qualidade pra plotagem de
        histogramas. Não é uma auditoria de "problemas" — é puramente
        informativa, pra dar ao perito um overview da distribuição
        empírica das métricas no dataset.

        Por enquanto: FIQA (quando disponível) + EMBEDDING_NORM (quando
        há embeddings). Cada finding carrega `metadata["values"]` com
        os escores de todas as faces que têm aquela métrica populada;
        a view consome isso e renderiza um histograma matplotlib em vez
        da tabela tradicional. `issues=()` (count=0) — bolinha verde na
        tree, indicando "informativo, sem problemas".
        """
        out: List[AuditFinding] = []
        target_metrics = [
            QualityScoreType.FIQA,
            QualityScoreType.EMBEDDING_NORM,
        ]
        for metric in target_metrics:
            values: List[float] = []
            image_paths: List[str] = []
            for face in dataset.faces:
                v = metric.extract_from(face)
                if v is not None:
                    values.append(float(v))
                    image_paths.append(str(getattr(face, "image_path", "") or ""))
            if not values:
                continue
            out.append(AuditFinding(
                audit_id=f"histogram_{metric.name.lower()}",
                audit_label=f"Histograma de {metric.display_name}",
                category=CATEGORY_HISTOGRAMS,
                severity=AuditSeverity.INFO,
                issues=(),
                summary=(
                    f"{len(values)} face(s) com escore — "
                    f"min={min(values):.3f}, max={max(values):.3f}, "
                    f"média={sum(values) / len(values):.3f}"
                ),
                metadata={
                    "values": values,
                    # Paralelo a `values` — permite à UI inspecionar as
                    # caudas (top-10 menores e top-10 maiores escores)
                    # mostrando as imagens correspondentes embaixo do
                    # histograma.
                    "image_paths": image_paths,
                    "metric_name": metric.name,
                    "metric_display": metric.display_name,
                },
            ))
        return out

    def audit_correlations(
        self,
        dataset: FaceDataset,
        *,
        x_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[AuditFinding]:
        """Coleta pares (x, y) de métricas relacionadas pra plotagem
        scatter. Útil pra detectar inconsistências entre métricas — ex:
        FIQA bom + norma baixa indica embedding instável apesar do escore
        de qualidade alto.

        Por enquanto: FIQA × Norma do embedding (quando ambas
        disponíveis). Cada finding carrega `metadata["x_values"]` e
        `["y_values"]` com os pares; opcionalmente `x_threshold` /
        `y_threshold` pra desenhar linhas guia de quadrante. Pearson r
        é pré-calculado e incluído no summary pra ficar visível na tree.
        """
        import numpy as np

        out: List[AuditFinding] = []
        thresholds = x_thresholds or {}

        # FIQA × Norma do embedding
        x_metric = QualityScoreType.FIQA
        y_metric = QualityScoreType.EMBEDDING_NORM
        xs: List[float] = []
        ys: List[float] = []
        for face in dataset.faces:
            x_val = x_metric.extract_from(face)
            y_val = y_metric.extract_from(face)
            if x_val is not None and y_val is not None:
                xs.append(float(x_val))
                ys.append(float(y_val))

        if xs:
            r = (
                float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1])
                if len(xs) > 1 else 0.0
            )
            out.append(AuditFinding(
                audit_id="correlation_fiqa_norm",
                audit_label="FIQA × Norma do embedding",
                category=CATEGORY_CORRELATIONS,
                severity=AuditSeverity.INFO,
                issues=(),
                summary=(
                    f"{len(xs)} face(s) com ambas as métricas — "
                    f"Pearson r = {r:+.3f}"
                ),
                metadata={
                    "x_values": xs,
                    "y_values": ys,
                    "x_label": x_metric.display_name,
                    "y_label": y_metric.display_name,
                    "x_metric_name": x_metric.name,
                    "y_metric_name": y_metric.name,
                    "x_threshold": thresholds.get(x_metric.name),
                    "y_threshold": thresholds.get(y_metric.name),
                    "pearson_r": r,
                },
            ))
        return out

    def audit_balance(
        self, dataset: FaceDataset,
    ) -> List[AuditFinding]:
        """Análise de balanceamento do dataset por identidade. Reusa o
        `DatasetStats.balance_metrics` — métricas oficiais consumidas
        também pelo `ResourceInfoWidget`. Produz 2 findings:

          - `histogram_samples_per_identity`: histograma das contagens
            de fotos por identidade. As tail strips mostram as 10 IDs
            com menos fotos (esquerda — singletons/quase-singletons) e
            10 com mais fotos (direita — possíveis "celebridades" que
            dominam o dataset).
          - `balance_metrics`: card com verdict (Balanceado / Aceitável
            / Moderado / Severo) + CV + razão max/min + min/max/mediana
            /média/std das contagens.
        """
        from faceaudit.models.dataset_stats import DatasetStats

        out: List[AuditFinding] = []

        # Conta amostras por identidade + guarda primeira face por ID
        # (representante visual nas tail strips).
        samples_per_id: Dict[str, int] = {}
        first_face_path_by_id: Dict[str, str] = {}
        for face in dataset.faces:
            ident = str(getattr(face, "identity", "") or "(sem identidade)")
            samples_per_id[ident] = samples_per_id.get(ident, 0) + 1
            if ident not in first_face_path_by_id:
                first_face_path_by_id[ident] = str(
                    getattr(face, "image_path", "") or ""
                )

        if not samples_per_id:
            return out

        # ── Finding 1: histograma das contagens ──
        # Para cada IDENTIDADE temos: count, primeira foto, label.
        identities = list(samples_per_id.keys())
        values = [samples_per_id[i] for i in identities]
        image_paths = [first_face_path_by_id[i] for i in identities]
        # Caption truncada pra caber nos thumbs (80px de largura)
        captions = [
            (
                f"{(i if len(i) <= 12 else i[:11] + '…')}\n"
                f"n={samples_per_id[i]}"
            )
            for i in identities
        ]

        out.append(AuditFinding(
            audit_id="histogram_samples_per_identity",
            audit_label="Histograma de amostras por identidade",
            category=CATEGORY_BALANCE,
            severity=AuditSeverity.INFO,
            issues=(),
            summary=(
                f"{len(identities):,} identidade(s) — "
                f"min={min(values)}, max={max(values)}, "
                f"mediana={sorted(values)[len(values) // 2]}"
            ),
            metadata={
                "values": values,
                "image_paths": image_paths,
                "captions": captions,
                "metric_name": "SAMPLES_PER_IDENTITY",
                "metric_display": "Amostras por identidade",
            },
        ))

        # ── Finding 2: card de métricas ──
        # Usa o property oficial do DatasetStats — mesma lógica de
        # verdict que o ResourceInfoWidget.
        stats = DatasetStats(samples_per_identity=dict(samples_per_id))
        bm = stats.balance_metrics
        if bm is not None:
            out.append(AuditFinding(
                audit_id="balance_metrics",
                audit_label="Métricas de balanceamento",
                category=CATEGORY_BALANCE,
                severity=AuditSeverity.INFO,
                issues=(),
                summary=(
                    f"{bm.verdict} — CV={bm.cv:.3f}, "
                    f"max/min={bm.imbalance_ratio:.1f}:1"
                ),
                metadata={
                    "verdict": bm.verdict,
                    "n_identities": bm.n_identities,
                    "min_samples": bm.min_samples,
                    "max_samples": bm.max_samples,
                    "mean_samples": bm.mean_samples,
                    "median_samples": bm.median_samples,
                    "std_samples": bm.std_samples,
                    "cv": bm.cv,
                    "imbalance_ratio": bm.imbalance_ratio,
                    "n_total_samples": sum(samples_per_id.values()),
                },
            ))
        return out

    def audit_stratified_boxplots(
        self, dataset: FaceDataset,
    ) -> List[AuditFinding]:
        """Boxplots de uma métrica Y estratificada por quartis de uma
        métrica X. Útil pra distinguir uma correlação real (medianas
        monótonas entre quartis) de um artefato de outliers (medianas
        flat, com a "correlação" vinda só dos extremos).

        Por enquanto: norma do embedding estratificada por quartis de
        FIQA. Cada finding carrega `metadata["groups"]` com 4 listas
        (uma por quartil) e `["labels"]` com os rótulos dos quartis.
        """
        import numpy as np

        out: List[AuditFinding] = []
        x_metric = QualityScoreType.FIQA
        y_metric = QualityScoreType.EMBEDDING_NORM

        xs: List[float] = []
        ys: List[float] = []
        for face in dataset.faces:
            x_val = x_metric.extract_from(face)
            y_val = y_metric.extract_from(face)
            if x_val is not None and y_val is not None:
                xs.append(float(x_val))
                ys.append(float(y_val))

        if len(xs) < 4:
            return out

        x_arr = np.asarray(xs)
        y_arr = np.asarray(ys)
        q1, q2, q3 = (float(v) for v in np.quantile(x_arr, [0.25, 0.5, 0.75]))

        # Particiona Y nos 4 quartis de X. `np.digitize` com bins
        # crescentes garante atribuição estável mesmo com empates.
        bins = [q1, q2, q3]
        idx = np.digitize(x_arr, bins, right=True)  # 0..3
        groups: List[List[float]] = [[], [], [], []]
        for bin_idx, y_val in zip(idx, y_arr):
            groups[int(bin_idx)].append(float(y_val))

        labels = [
            f"Q1\n({x_metric.display_name} ≤ {q1:.2f})\nn={len(groups[0])}",
            f"Q2\n({q1:.2f} – {q2:.2f}]\nn={len(groups[1])}",
            f"Q3\n({q2:.2f} – {q3:.2f}]\nn={len(groups[2])}",
            f"Q4\n(> {q3:.2f})\nn={len(groups[3])}",
        ]
        # Sumário com mediana por quartil — leitura rápida na tree.
        # Monotonicidade decrescente das medianas confirma correlação
        # negativa real; medianas flat sugerem artefato de outliers.
        medians = [
            (float(np.median(g)) if g else float("nan"))
            for g in groups
        ]
        out.append(AuditFinding(
            audit_id="boxplot_norm_by_fiqa_quartile",
            audit_label="Norma por quartil de FIQA (boxplot)",
            category=CATEGORY_CORRELATIONS,
            severity=AuditSeverity.INFO,
            issues=(),
            summary=(
                f"medianas: Q1={medians[0]:.2f} | Q2={medians[1]:.2f} | "
                f"Q3={medians[2]:.2f} | Q4={medians[3]:.2f}"
            ),
            metadata={
                "groups": groups,
                "labels": labels,
                "x_label": x_metric.display_name,
                "y_label": y_metric.display_name,
                "medians": medians,
            },
        ))
        return out

    def available_quality_metrics(
        self, dataset: FaceDataset,
    ) -> List[QualityScoreType]:
        """Descobre quais métricas de qualidade são auditáveis no dataset.

        Estratégia em camadas:
        1. EMBEDDING_NORM: SEMPRE disponível (calculada do embedding).
        2. FIQA/OFIQ/SERFIQ: disponível se `column_specs` tem coluna com
           `role=quality` e `metric_name` correspondente, OU (fallback
           pra parquets legados) se pelo menos uma face populou o atributo.

        Retorna na ordem canônica: FIQA, OFIQ, SERFIQ, EMBEDDING_NORM.
        """
        available: List[QualityScoreType] = []
        for metric in (QualityScoreType.FIQA, QualityScoreType.OFIQ,
                       QualityScoreType.SERFIQ):
            # Schema check (rápido)
            if dataset.column_specs and metric.column_name_in(
                dataset.column_specs,
            ):
                available.append(metric)
                continue
            # Runtime fallback: alguma face tem o valor populado?
            for face in dataset.faces:
                if metric.extract_from(face) is not None:
                    available.append(metric)
                    break
        # EMBEDDING_NORM sempre, no final (pode ser computada se há embeddings)
        for face in dataset.faces:
            if QualityScoreType.EMBEDDING_NORM.extract_from(face) is not None:
                available.append(QualityScoreType.EMBEDDING_NORM)
                break
        return available

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
                    metadata={
                        "value": float(value),
                        "metric": metric.name,
                        "metric_display": metric.display_name,
                        "identity": str(getattr(face, "identity", "") or ""),
                    },
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
        threshold: float = 0.90,
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
                "duplicate_identities", "Pessoas com mais de um rótulo",
                CATEGORY_BIOMETRIC, AuditSeverity.WARNING,
                "Dataset sem embeddings — não foi possível computar centróides",
            )

        cb("Calculando centróides por identidade…", 20)
        centroids, identity_order, faces_by_identity = self._compute_identity_centroids(emb_ds)
        if len(identity_order) < 2:
            return self._empty_finding(
                "duplicate_identities", "Pessoas com mais de um rótulo",
                CATEGORY_BIOMETRIC, AuditSeverity.WARNING,
                f"Apenas {len(identity_order)} identidade(s) — comparação inviável",
            )

        n = len(identity_order)
        cb(f"Comparando {n} identidades (chunked)…", 50)
        # Cosine sim entre centróides já L2-normalizados = produto interno.
        #
        # Versão chunked: a matriz K×K completa pesa K² × 4 bytes
        # (float32) — pra K=49.213 isso são 9.6 GB e OOM. Em vez disso,
        # processamos em blocos de `BLOCK` linhas de cada vez:
        # `centroids[i:i+B] @ centroids.T` produz só uma faixa B×K
        # (~4 MB pra B=512). O bloco é descartado depois de coletar os
        # pares acima do threshold. Memória pico cai pra O(B·K) em vez
        # de O(K²); CPU/wall-time é equivalente (mesmo total de
        # multiplicações).
        BLOCK = 512
        issues: List[AuditIssue] = []
        for start in range(0, n, BLOCK):
            stop = min(start + BLOCK, n)
            # Faixa B × K
            block_sim = centroids[start:stop] @ centroids.T
            # Pra cada linha local i_local = i - start, comparamos só
            # com j > i (triângulo superior global) — evita duplicar
            # (a, b)/(b, a) e zera diagonal naturalmente.
            for i_local in range(stop - start):
                i = start + i_local
                # row[j>i]: triângulo superior estrito da linha i
                row = block_sim[i_local, i + 1:]
                # Índices locais (em row) acima do threshold
                hits = np.where(row >= threshold)[0]
                if hits.size == 0:
                    continue
                # Mapeia pra índices globais de identidade B
                for offset in hits:
                    j = i + 1 + int(offset)
                    sim = float(row[offset])
                    id_a, id_b = identity_order[i], identity_order[j]
                    faces_a = faces_by_identity[id_a]
                    faces_b = faces_by_identity[id_b]
                    locators = tuple(faces_a + faces_b)
                    identities_aligned = (
                        [id_a] * len(faces_a) + [id_b] * len(faces_b)
                    )
                    issues.append(AuditIssue(
                        locators=locators,
                        description=(
                            f"'{id_a}' ↔ '{id_b}' — centróides com "
                            f"similaridade {sim:.3f} (≥ {threshold:.2f})"
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
            # Progress 50% → 95% conforme avançamos no varredura linear
            pct = 50 + int(45 * stop / n)
            cb(
                f"Comparando identidades (chunked)… {stop}/{n}",
                min(pct, 95),
            )

        cb("Ordenando pares…", 96)
        # Ordena issues do mais confuso pro menos confuso
        issues.sort(key=lambda x: -x.metadata.get("similarity", 0.0))

        cb("Concluído.", 100)
        return AuditFinding(
            audit_id="duplicate_identities",
            audit_label="Pessoas com mais de um rótulo",
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

        Estratégia (leave-one-out): pra cada face X de uma identidade com
        ≥ `min_samples` amostras, computa o centroide das **outras** faces
        do mesmo ID (excluindo X), normaliza, e mede cosseno X vs esse
        centroide. Faces com cosseno < `outlier_threshold` são outliers —
        provável anotação errada (face de outra pessoa rotulada como esta).

        Por que leave-one-out: se a própria face X entrasse no centroide,
        ela puxaria o vetor médio na sua direção e o cosseno X vs centroide
        ficaria artificialmente alto (auto-influência mascararia o outlier).
        Excluindo X do cálculo, o sinal é limpo.

        Cada **face outlier** vira uma issue própria (uma row na tabela),
        ordenadas das mais discrepantes (cosseno mais baixo) pras menos.

        Custo: O(N·D) com truques vetorizados — recalcular o centroide
        leave-one-out por face é (sum_all - X) / (n-1), feito numa
        passada por identidade.
        """
        cb = progress_cb or (lambda *_a: None)
        cb("Convertendo para embedding dataset…", 5)

        emb_ds = dataset.to_embedding_dataset()
        if not emb_ds.items:
            return self._empty_finding(
                "mixed_identities", "Faces com erro de identidade",
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
                "mixed_identities", "Faces com erro de identidade",
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

            n = len(idx_list)
            if n < 2:
                continue  # com 1 amostra não há "outras" pra comparar
            embs = norm_matrix[idx_list]            # (n, D), L2-normalized

            # Leave-one-out vetorizado:
            # centroid_loo[i] = média das embs[k] para k != i
            #                = (sum_all - embs[i]) / (n - 1)
            sum_all = embs.sum(axis=0)              # (D,)
            centroids_loo = (sum_all - embs) / (n - 1)   # (n, D)
            # L2-normaliza cada linha
            norms = np.linalg.norm(centroids_loo, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            centroids_loo /= norms

            # cosseno linha-a-linha: X[i] · centroid_loo[i]
            sims = (embs * centroids_loo).sum(axis=1)   # (n,)

            outlier_local_idx = np.where(sims < outlier_threshold)[0]
            if len(outlier_local_idx) == 0:
                continue

            # Ordena outliers do pior pro melhor (mais discrepantes primeiro).
            outlier_local_idx = outlier_local_idx[np.argsort(sims[outlier_local_idx])]

            id_paths = [
                getattr(emb_ds.items[idx_list[i]], "image_path", "")
                for i in range(n)
            ]
            id_filenames = [
                getattr(emb_ds.items[idx_list[i]], "file_name", "")
                or _Path(id_paths[i]).name
                for i in range(n)
            ]

            for local_i in outlier_local_idx:
                outlier_path = id_paths[local_i]
                outlier_name = id_filenames[local_i]
                # Paths das OUTRAS faces (que formaram o centroide).
                other_paths = [
                    p for j, p in enumerate(id_paths) if j != local_i
                ]
                locators = tuple([outlier_path] + other_paths)

                issues.append(AuditIssue(
                    locators=locators,
                    description=(
                        f"'{identity}' — {outlier_name}: cos={float(sims[local_i]):.3f} "
                        f"vs centroide das outras {n - 1} foto(s)"
                    ),
                    metadata={
                        "identity": identity,
                        "file_name": outlier_name,
                        "outlier_path": outlier_path,
                        "n_others": n - 1,
                        "similarity": float(sims[local_i]),
                        "other_paths": tuple(other_paths),
                    },
                ))

        # Ordena issues primariamente por identidade (alfabético), e
        # secundariamente por similaridade ascendente (mais discrepante
        # primeiro dentro do mesmo ID). Mantém faces da mesma identidade
        # agrupadas na tabela — facilita revisão.
        issues.sort(key=lambda x: (
            x.metadata.get("identity", ""),
            x.metadata.get("similarity", 1.0),
        ))

        cb("Concluído.", 100)
        return AuditFinding(
            audit_id="mixed_identities",
            audit_label="Faces com erro de identidade",
            category=CATEGORY_BIOMETRIC,
            severity=AuditSeverity.WARNING,
            issues=tuple(issues),
            summary=(
                f"{len(issues)} face(s) com cos < {outlier_threshold:.2f} "
                f"vs centroide das demais (leave-one-out)"
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
        low_quality_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[AuditFinding]:
        """Roda todas as auditorias O(N) em sequência. Custo total proporcional
        ao tamanho do dataset (sem comparações entre faces).

        `low_quality_thresholds`: dict `{metric.name: threshold}` (e.g.
        `{"FIQA": 1.50, "OFIQ": 25.0}`). Cada métrica disponível no dataset
        com threshold configurado gera uma auditoria `low_quality_<metric>`.
        Métricas sem threshold são puladas (não auditadas), mesmo se
        disponíveis no schema.
        """
        thresholds = low_quality_thresholds or {}
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
        # `Faces sem OFIQ/SERFIQ` agora caem em CATEGORY_PHYSICAL
        # (integridade física do dataset).
        out.extend(self.audit_incomplete_enrichment(dataset))
        # Uma auditoria de baixa qualidade por métrica disponível no
        # dataset que tenha threshold configurado pelo usuário.
        for metric in self.available_quality_metrics(dataset):
            threshold = thresholds.get(metric.name)
            if threshold is None:
                continue
            out.append(self.audit_low_quality(
                dataset, metric=metric, threshold=float(threshold),
            ))
        # Histogramas — categoria informativa, não-acionável
        out.extend(self.audit_histograms(dataset))
        # Correlações — scatter plots entre métricas de qualidade.
        # Passa os thresholds configurados pra que as linhas de quadrante
        # no plot batam com os limites de auditoria do usuário.
        out.extend(self.audit_correlations(
            dataset, x_thresholds=thresholds,
        ))
        # Boxplots estratificados — distinguem correlação real de
        # artefato de outliers via medianas por quartil.
        out.extend(self.audit_stratified_boxplots(dataset))
        # Balanceamento — histograma de samples/ID + card de métricas
        out.extend(self.audit_balance(dataset))
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
