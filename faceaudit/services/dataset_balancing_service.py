"""DatasetBalancingService — balanceamento por identidade.

Extrai a lógica do `DatasetBalancingInstructionExecutor` pra um service
standalone reutilizável. Permite que o Dataset Manager (botão Balancear)
chame a mesma lógica sem depender da máquina de instruções do Strategy
Manager.

Stateless — use uma instância por chamada ou compartilhe. Aceita
`FaceDataset` ou `EmbeddingDataset`. Lógica de cap/floor é compartilhada
entre os dois tipos; difere a função extratora de identidade e de
score de qualidade.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from faceaudit.enums.dataset_balancing_mode import DatasetBalancingMode
from faceaudit.enums.dataset_balancing_selection import DatasetBalancingSelection
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset


@dataclass(frozen=True)
class DatasetBalancingResult:
    """Resultado de uma operação de balanceamento."""
    dataset: Any  # FaceDataset ou EmbeddingDataset
    input_size: int
    input_n_ids: int
    output_size: int
    output_n_ids: int
    effective_n: int
    discarded_identities: int
    quality_fallback_count: int

    def summary(self) -> str:
        """Mensagem curta pra status/log."""
        parts = [
            f"{self.input_size}→{self.output_size} samples",
            f"{self.input_n_ids}→{self.output_n_ids} IDs",
            f"N={self.effective_n}",
        ]
        if self.discarded_identities > 0:
            parts.append(f"⚠ {self.discarded_identities} ID(s) descartada(s)")
        if self.quality_fallback_count > 0:
            parts.append(
                f"⚠ {self.quality_fallback_count} sample(s) sem métrica"
            )
        return " · ".join(parts)


class DatasetBalancingService:
    """Aplica balanceamento por identidade num dataset."""

    def balance(
        self,
        dataset: Any,
        *,
        mode: DatasetBalancingMode,
        target_n: int,
        selection: DatasetBalancingSelection,
        random_seed: int = 42,
        quality_metric: QualityScoreType = QualityScoreType.FIQA,
        output_name: Optional[str] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> DatasetBalancingResult:
        """Balanceia o dataset por identidade.

        Args:
            dataset: FaceDataset ou EmbeddingDataset de entrada (não mutado).
            mode: estratégia de cap/floor (ver DatasetBalancingMode).
            target_n: alvo de samples por ID (ignorado em AUTO_MIN_BALANCE).
            selection: RANDOM (sorteio reprodutível) ou BY_QUALITY (top-N).
            random_seed: seed pro RANDOM (irrelevante em BY_QUALITY).
            quality_metric: métrica usada em BY_QUALITY (FIQA/OFIQ/SERFIQ/
                EMBEDDING_NORM). Maior = melhor.
            output_name: nome do dataset de saída (default: `<input>_Balanced`).
            logger: callback opcional pra logs intermediários.

        Returns:
            DatasetBalancingResult com o dataset balanceado + estatísticas.
        """
        log = logger or (lambda _msg: None)
        uses_auto_min = mode == DatasetBalancingMode.AUTO_MIN_BALANCE
        uses_floor_and_cap = mode == DatasetBalancingMode.FLOOR_AND_CAP
        uses_random_selection = selection == DatasetBalancingSelection.RANDOM

        items = self._get_items(dataset)
        items_by_id = self._group_by_identity(items, dataset)

        input_size = len(items)
        input_n_ids = len(items_by_id)

        # N efetivo
        if uses_auto_min:
            counts = [len(v) for v in items_by_id.values()]
            effective_n = min(counts) if counts else 0
            log(
                f"  ↳ AUTO_MIN_BALANCE: N computado = {effective_n} "
                f"(menor entre {input_n_ids} identidades)."
            )
        else:
            effective_n = target_n

        # Floor (descarta IDs inteiros) só em FLOOR_AND_CAP
        if uses_floor_and_cap:
            before = len(items_by_id)
            items_by_id = {
                k: v for k, v in items_by_id.items() if len(v) >= effective_n
            }
            discarded_identities = before - len(items_by_id)
            if discarded_identities > 0:
                log(
                    f"  ↳ FLOOR_AND_CAP: {discarded_identities} identidade(s) "
                    f"descartada(s) por terem < {effective_n} amostras."
                )
        else:
            discarded_identities = 0

        # Cap por identidade
        rng = random.Random(random_seed)
        quality_fallback_count = 0
        is_face_dataset = isinstance(dataset, FaceDataset)
        kept_items: List[Any] = []

        for _identity, id_items in items_by_id.items():
            if len(id_items) <= effective_n:
                kept_items.extend(id_items)
                continue

            if uses_random_selection:
                shuffled = list(id_items)
                rng.shuffle(shuffled)
                kept_items.extend(shuffled[:effective_n])
            else:
                sorted_items, fallbacks = self._sort_by_quality(
                    id_items=id_items,
                    quality_metric=quality_metric,
                    is_face_dataset=is_face_dataset,
                )
                quality_fallback_count += fallbacks
                kept_items.extend(sorted_items[:effective_n])

        if quality_fallback_count > 0:
            log(
                f"  ↳ ⚠ {quality_fallback_count} sample(s) sem "
                f"`{quality_metric.name}` — empurrados ao final."
            )

        balanced = self._build_output_dataset(
            input_dataset=dataset,
            kept_items=kept_items,
            mode=mode,
            selection=selection,
            output_name=output_name,
        )

        output_size = len(self._get_items(balanced))
        output_n_ids = len({
            self._extract_identity(it, is_face_dataset)
            for it in self._get_items(balanced)
            if self._extract_identity(it, is_face_dataset) is not None
        })

        return DatasetBalancingResult(
            dataset=balanced,
            input_size=input_size,
            input_n_ids=input_n_ids,
            output_size=output_size,
            output_n_ids=output_n_ids,
            effective_n=effective_n,
            discarded_identities=discarded_identities,
            quality_fallback_count=quality_fallback_count,
        )

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _get_items(dataset: Any) -> List[Any]:
        if isinstance(dataset, FaceDataset):
            return list(dataset.faces)
        if isinstance(dataset, EmbeddingDataset):
            return list(dataset.items)
        return list(dataset)

    @staticmethod
    def _extract_identity(item: Any, is_face_dataset: bool) -> Optional[str]:
        identity = getattr(item, "identity", None)
        if identity is None:
            return None
        return str(identity)

    def _group_by_identity(
        self, items: List[Any], input_dataset: Any,
    ) -> Dict[str, List[Any]]:
        is_face_dataset = isinstance(input_dataset, FaceDataset)
        groups: Dict[str, List[Any]] = defaultdict(list)
        for item in items:
            identity = self._extract_identity(item, is_face_dataset)
            if identity is None:
                continue
            groups[identity].append(item)
        return dict(groups)

    def _sort_by_quality(
        self,
        id_items: List[Any],
        quality_metric: QualityScoreType,
        is_face_dataset: bool,
    ) -> Tuple[List[Any], int]:
        scored: List[Tuple[float, str, Any]] = []
        fallback_count = 0
        for item in id_items:
            score = self._extract_quality_score(
                item, quality_metric, is_face_dataset,
            )
            if score is None:
                fallback_count += 1
                score_for_sort = float("-inf")
            else:
                score_for_sort = float(score)
            tie_key = self._stable_tie_key(item, is_face_dataset)
            scored.append((score_for_sort, tie_key, item))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [t[2] for t in scored], fallback_count

    @staticmethod
    def _extract_quality_score(
        item: Any,
        quality_metric: QualityScoreType,
        is_face_dataset: bool,
    ) -> Optional[float]:
        if is_face_dataset:
            try:
                return quality_metric.extract_from(item)
            except Exception:
                return None
        if quality_metric == QualityScoreType.EMBEDDING_NORM:
            emb = getattr(item, "embedding", None)
            if emb is None:
                return None
            try:
                return float(np.linalg.norm(emb))
            except Exception:
                return None
        scores = getattr(item, "quality_scores", None)
        if not isinstance(scores, dict):
            return None
        val = scores.get(quality_metric.name.lower())
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _stable_tie_key(item: Any, is_face_dataset: bool) -> str:
        for attr in ("face_hash", "hash", "file_name", "image_path"):
            v = getattr(item, attr, None)
            if v:
                return hashlib.sha1(str(v).encode("utf-8")).hexdigest()[:16]
        return hashlib.sha1(repr(item).encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _build_output_dataset(
        input_dataset: Any,
        kept_items: List[Any],
        mode: DatasetBalancingMode,
        selection: DatasetBalancingSelection,
        output_name: Optional[str],
    ) -> Any:
        new_name = (
            output_name
            or f"{getattr(input_dataset, 'name', 'dataset')}_Balanced"
        )
        new_desc = (
            f"Balanceado ({mode.to_str()} / {selection.to_str()})"
        )
        if isinstance(input_dataset, FaceDataset):
            ds = FaceDataset(
                name=new_name,
                description=new_desc,
                hash=getattr(input_dataset, "hash", None),
                faces=kept_items,
                path_mode=getattr(input_dataset, "path_mode", FaceDataset().path_mode),
                path_root=getattr(input_dataset, "path_root", None),
                is_aligned=getattr(input_dataset, "is_aligned", False),
                column_specs=dict(
                    getattr(input_dataset, "column_specs", None) or {}
                ),
            )
            ds._parquet_dir = getattr(input_dataset, "_parquet_dir", None)
            return ds
        if isinstance(input_dataset, EmbeddingDataset):
            ds = EmbeddingDataset(
                name=new_name,
                description=new_desc,
                items=kept_items,
            )
            # Preserva metadados se a classe expuser (paridade com o
            # comportamento do executor original).
            loaded_md = getattr(input_dataset, "loaded_metadata", None)
            if isinstance(loaded_md, dict):
                ds.loaded_metadata = loaded_md.copy()
            return ds
        return kept_items
