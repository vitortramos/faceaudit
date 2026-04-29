# faceaudit/models/similarity/similarity_score_dataset.py

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample


T = TypeVar("T", bound="SimilarityScoreDataset")


@dataclass
class SimilarityScoreDataset(JSONSerializableMixin):
    """
    Coleção de amostras de scores de similaridade.

    Esta classe é a ponte natural entre:
    - modelos de similaridade, que produzem scores brutos
    - modelos de calibração, que aprendem a partir de scores e labels
    - rotinas de auditoria, análise de erro e exportação

    Diferentemente do DTO antigo preso ao cosseno, aqui usamos `raw_score`
    de forma neutra para suportar:
    - cosine
    - PLDA
    - outros modelos futuros
    """

    name: str
    description: Optional[str] = None
    items: List[SimilarityScoreSample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, sample: SimilarityScoreSample) -> None:
        self.items.append(sample)
        # Invalida o cache de contagens — `total_genuines`/`total_impostors`
        # serão recomputados na próxima leitura.
        self.__dict__.pop("_cached_n_gen", None)
        self.__dict__.pop("_cached_n_imp", None)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> SimilarityScoreSample:
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def _ensure_class_counts_cached(self) -> None:
        """
        Computa contagens de genuínos/impostores em UMA passada e cacheia.

        Necessário porque `_refresh_produced_resources` na UI chamava
        `total_genuines` e `total_impostors` a cada refresh e cada acesso
        percorria os 5M+ items — travando a main thread Qt por segundos.
        Com cache, o custo só é pago uma vez por dataset.
        """
        if "_cached_n_gen" in self.__dict__ and "_cached_n_imp" in self.__dict__:
            return
        n_gen = 0
        n_imp = 0
        for item in self.items:
            sp = item.same_person
            if sp is True:
                n_gen += 1
            elif sp is False:
                n_imp += 1
        self._cached_n_gen = n_gen
        self._cached_n_imp = n_imp

    @property
    def total_genuines(self) -> int:
        self._ensure_class_counts_cached()
        return self._cached_n_gen

    @property
    def total_impostors(self) -> int:
        self._ensure_class_counts_cached()
        return self._cached_n_imp

    @property
    def scores(self) -> np.ndarray:
        return np.asarray([float(item.raw_score) for item in self.items], dtype=np.float64)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([int(item.label) for item in self.items], dtype=np.int32)

    def to_pandas(self) -> pd.DataFrame:
        """
        Converte todas as amostras em um DataFrame achatado.
        """
        if not self.items:
            return pd.DataFrame()

        rows = [item.to_dict() for item in self.items]
        df = pd.DataFrame(rows)

        df.insert(0, "similarity_score_dataset_name", self.name)

        if self.description is not None:
            df.insert(1, "similarity_score_dataset_description", self.description)

        return df

    def to_parquet(
        self,
        path: Union[str, Path],
        progress_callback=None,
    ) -> None:
        """
        Salva o dataset em disco para auditoria posterior.

        Estratégia para datasets MUITO grandes (testado até 5M+ pares):

        1. **Colunas mínimas** — apenas raw_score, label, same_person,
           probe_identity, reference_identity. file_name/image_path/face_hash
           descartados (consumidores downstream — calibração, plot — só
           precisam de score+label+identity).

        2. **GC desativado durante a escrita**. Com 5M+ samples no heap, cada
           coleção cíclica de gen-2 do Python percorre todos os objetos
           rastreados, causando pausas de SEGUNDOS com a GIL retida — eis
           a causa real do "Não respondendo". `__slots__` em
           SimilarityScoreSample já reduz o tracking; pausar gc.disable()
           durante o write elimina o resíduo.

        3. **Chunks de 25k linhas** com `time.sleep(0.005)` entre eles —
           limita hold de GIL Python a ~30ms por chunk e dá janela ao main
           thread.

        4. **progress_callback opcional** — invocado antes/depois de cada
           chunk com mensagem legível, para que o worker possa propagar
           progresso ao dialog em tempo real durante a persistência.
        """
        import gc
        import time
        import pyarrow as pa
        import pyarrow.parquet as pq

        def _emit(msg: str) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        n = len(self.items)

        # `log_lr` é uma coluna OPCIONAL: presente sempre que algum sample do
        # dataset tem `metadata["log_lr"]` preenchido (ex: output da
        # instrução SCORE_CALIBRATION). Para datasets de scores brutos
        # (output de SIMILARITY_SCORING), esta coluna não aparece — mantém
        # compatibilidade com parquets antigos.
        has_log_lr = any(
            isinstance(getattr(s, "metadata", None), dict)
            and s.metadata.get("log_lr") is not None
            for s in self.items
        )

        fields = [
            ("similarity_score_dataset_name", pa.string()),
            ("probe_identity",     pa.string()),
            ("reference_identity", pa.string()),
            ("same_person", pa.bool_()),
            ("label",       pa.int8()),
            ("raw_score",   pa.float64()),
        ]
        if self.description is not None:
            fields.insert(1, ("similarity_score_dataset_description", pa.string()))
        if has_log_lr:
            fields.append(("log_lr", pa.float64()))
        schema = pa.schema(fields)

        if n == 0:
            with pq.ParquetWriter(path_obj, schema) as writer:
                writer.write_table(pa.table({f.name: pa.array([], type=f.type) for f in schema}))
            return

        CHUNK = 25_000
        YIELD_SECONDS = 0.005

        # ─── Anti-freeze: pausa o coletor cíclico do GC ────────────────────
        # Com milhões de objetos no heap, gc gen-2 pode pausar a thread por
        # muitos segundos enquanto percorre tudo. Pausar aqui evita esse
        # custo (refcount continua liberando memória normalmente).
        gc_was_enabled = gc.isenabled()
        gc.collect()
        gc.disable()

        n_chunks = (n + CHUNK - 1) // CHUNK
        _emit(f"  ↳ Iniciando gravação parquet em {n_chunks} bloco(s) de até {CHUNK:,} linhas…".replace(",", "."))

        try:
            with pq.ParquetWriter(path_obj, schema) as writer:
                for chunk_idx, chunk_start in enumerate(range(0, n, CHUNK), start=1):
                    chunk_end = min(chunk_start + CHUNK, n)
                    size = chunk_end - chunk_start

                    scores      = np.empty(size, dtype=np.float64)
                    labels      = np.empty(size, dtype=np.int8)
                    same_person = np.empty(size, dtype=object)
                    probe_id    = np.empty(size, dtype=object)
                    ref_id      = np.empty(size, dtype=object)
                    log_lr_arr  = np.empty(size, dtype=np.float64) if has_log_lr else None

                    items_local = self.items
                    for j in range(size):
                        s = items_local[chunk_start + j]
                        scores[j]      = s.raw_score
                        labels[j]      = s.label
                        same_person[j] = s.same_person
                        probe_id[j]    = s.probe.identity
                        ref_id[j]      = s.reference.identity
                        if has_log_lr:
                            md = getattr(s, "metadata", None)
                            v = md.get("log_lr") if isinstance(md, dict) else None
                            log_lr_arr[j] = float(v) if v is not None else np.nan

                    cols = {
                        "similarity_score_dataset_name": pa.array([self.name] * size, type=pa.string()),
                        "probe_identity":     pa.array(probe_id, type=pa.string()),
                        "reference_identity": pa.array(ref_id, type=pa.string()),
                        "same_person": pa.array(same_person, type=pa.bool_()),
                        "label":       pa.array(labels, type=pa.int8()),
                        "raw_score":   pa.array(scores, type=pa.float64()),
                    }
                    if self.description is not None:
                        cols["similarity_score_dataset_description"] = pa.array(
                            [self.description] * size, type=pa.string()
                        )
                    if has_log_lr:
                        cols["log_lr"] = pa.array(log_lr_arr, type=pa.float64())

                    writer.write_table(pa.table(cols, schema=schema))

                    # Libera referências locais para o GC desligado não
                    # acumular trabalho — refcount cuida da limpeza.
                    del scores, labels, same_person, probe_id, ref_id, cols
                    if log_lr_arr is not None:
                        del log_lr_arr

                    if chunk_idx == 1 or chunk_idx == n_chunks or (chunk_idx % 10 == 0):
                        pct = 100.0 * chunk_end / n
                        _emit(
                            f"  ↳ Bloco {chunk_idx}/{n_chunks} "
                            f"({chunk_end:,}/{n:,} linhas, {pct:.0f}%)".replace(",", ".")
                        )

                    # Pausa real do worker — garante janela ao main thread
                    # para processar eventos Qt.
                    time.sleep(YIELD_SECONDS)
        finally:
            if gc_was_enabled:
                gc.enable()

    def get_lowest_score_genuines(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna os pares genuínos com os menores scores brutos.
        """
        df = self.to_pandas()
        if df.empty:
            return df
        genuines = df[df["same_person"] == True]
        return genuines.nsmallest(top_n, "raw_score")

    def get_highest_score_impostors(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna os pares impostores com os maiores scores brutos.
        """
        df = self.to_pandas()
        if df.empty:
            return df
        impostors = df[df["same_person"] == False]
        return impostors.nlargest(top_n, "raw_score")

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o dataset hidratando corretamente os itens.
        """
        if not data:
            return None

        processed_data = data.copy()

        if "items" in processed_data and isinstance(processed_data["items"], list):
            processed_data["items"] = [
                SimilarityScoreSample.from_dict(item) if isinstance(item, dict) else item
                for item in processed_data["items"]
            ]

        return cls(**processed_data)