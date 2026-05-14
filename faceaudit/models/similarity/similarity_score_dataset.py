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
        """
        Retorna labels binários (1=Hp, 0=Hd) por sample.

        Samples sem ground truth (`label is None`) são representados como
        sentinela **-1**. Callers que treinam calibradores devem filtrar
        `labels >= 0` antes de usar — alternativamente, validar via
        `is_casework` no metadata do dataset (se True, NÃO chamar `labels`
        com expectativa de int).
        """
        return np.asarray(
            [int(item.label) if item.label is not None else -1 for item in self.items],
            dtype=np.int32,
        )

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

        # Schema metadata — flags do dataset que não cabem como coluna
        # (constantes pra todas as linhas). `is_casework` é crítico:
        # sem persistir explicitamente, o loader cai numa inferência
        # frágil baseada em `all(same_person is None)`, que pode falhar
        # silenciosamente (pd.NA vs None, valores residuais, etc).
        schema_metadata = {}
        if bool(self.metadata.get("is_casework", False)):
            schema_metadata[b"faceaudit:is_casework"] = b"true"
        schema = pa.schema(fields, metadata=schema_metadata or None)

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
                        # label = -1 sentinela para casework (s.label is None)
                        labels[j]      = s.label if s.label is not None else -1
                        same_person[j] = s.same_person  # None preservado como null no Parquet bool
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

        # Sidecar de distribuição pré-computada — evita iterar 5M+ samples
        # toda vez que o ScoreDistributionWidget vai desenhar o gráfico.
        # Tamanho típico: 30-80 KB. Falha silenciosa não impede a escrita
        # do parquet em si.
        try:
            self._write_distribution_summary_sidecar(path_obj)
        except Exception as exc:
            _emit(f"  ⚠ falha ao gravar sidecar de distribuição: {exc}")

    def _write_distribution_summary_sidecar(self, parquet_path: Path) -> None:
        """
        Grava um JSON pequeno ao lado do parquet com histograma + KDE pré-
        computados (genuínos vs impostores). O ScoreDistributionWidget lê
        esse sidecar quando existe e desenha sem precisar carregar o
        parquet inteiro nem rodar gaussian_kde de novo.
        """
        import json

        summary = self._compute_distribution_summary()
        if summary is None:
            return
        sidecar_path = parquet_path.with_name(parquet_path.stem + "_dist.json")
        sidecar_path.write_text(
            json.dumps(summary, ensure_ascii=False), encoding="utf-8"
        )

    def _compute_distribution_summary(self) -> Optional[dict]:
        """
        Computa o dict de distribuição (histograma + KDE) usado tanto
        pelo sidecar `_dist.json` quanto pelo PNG snapshot em
        `<project>/graphics/`. Retorna None se não houver pares.

        Detecta automaticamente o tipo de score relevante:
        - se há `metadata["log_lr"]` em pelo menos um sample → modo "log_lr"
          (output de SCORE_CALIBRATION)
        - caso contrário → modo "raw" (output de SIMILARITY_SCORING / PAIR_MINING)
        """
        if not self.items:
            return None

        # Decide o eixo X com base na presença de log_lr.
        has_log_lr = any(
            isinstance(getattr(s, "metadata", None), dict)
            and s.metadata.get("log_lr") is not None
            for s in self.items
        )
        score_type = "log_lr" if has_log_lr else "raw"

        gen_vals: list = []
        imp_vals: list = []
        # case_vals = pares casework (sem ground truth — `same_person is None`).
        # SIMILARITY_SCORING em modo casework produz dataset puramente case;
        # sem incluí-los aqui, o widget renderizava "Total: 0 pares".
        case_vals: list = []
        for s in self.items:
            sp = s.same_person
            if sp is True:
                arr = gen_vals
            elif sp is False:
                arr = imp_vals
            else:
                arr = case_vals
            if has_log_lr:
                md = getattr(s, "metadata", None)
                v = md.get("log_lr") if isinstance(md, dict) else None
                if v is None:
                    continue
                arr.append(float(v))
            else:
                arr.append(float(s.raw_score))

        gen_arr = np.asarray(gen_vals, dtype=np.float64)
        imp_arr = np.asarray(imp_vals, dtype=np.float64)
        case_arr = np.asarray(case_vals, dtype=np.float64)
        n_gen = int(gen_arr.size)
        n_imp = int(imp_arr.size)
        n_case = int(case_arr.size)
        if n_gen == 0 and n_imp == 0 and n_case == 0:
            return None

        # Eixo X cobre todas as 3 séries (gen, imp, case).
        non_empty = [a for a in (gen_arr, imp_arr, case_arr) if a.size > 0]
        all_vals = np.concatenate(non_empty) if non_empty else np.zeros(0)
        x_min = float(np.min(all_vals))
        x_max = float(np.max(all_vals))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        # Histograma — 60 bins, density=True (mesmo formato que o widget).
        bins = np.linspace(x_min, x_max, 60)
        gen_hist, _ = (
            np.histogram(gen_arr, bins=bins, density=True)
            if n_gen
            else (np.zeros(bins.size - 1), bins)
        )
        imp_hist, _ = (
            np.histogram(imp_arr, bins=bins, density=True)
            if n_imp
            else (np.zeros(bins.size - 1), bins)
        )
        # Caso: histograma só faz sentido com n>=30 (pelo menos uma dezena de
        # bins razoavelmente populados). Pra n<30 grava `case_values` cru pro
        # widget desenhar como linhas verticais anotadas.
        case_hist_payload = None
        case_values_payload: Optional[list] = None
        if n_case >= 30:
            case_hist, _ = np.histogram(case_arr, bins=bins, density=True)
            case_hist_payload = case_hist.tolist()
        elif n_case > 0:
            case_values_payload = case_arr.tolist()

        # KDE — 400 pontos no eixo X.
        kde_x = np.linspace(x_min, x_max, 400)
        kde_gen_y = None
        kde_imp_y = None
        kde_case_y = None
        try:
            from scipy.stats import gaussian_kde
            if n_gen > 1:
                kde_gen_y = gaussian_kde(gen_arr)(kde_x).tolist()
            if n_imp > 1:
                kde_imp_y = gaussian_kde(imp_arr)(kde_x).tolist()
            if n_case > 30:
                kde_case_y = gaussian_kde(case_arr)(kde_x).tolist()
        except Exception:
            pass

        return {
            "schema_version": 2,
            "score_type": score_type,
            "name": self.name,
            "n_gen": n_gen,
            "n_imp": n_imp,
            "n_case": n_case,
            "x_min": x_min,
            "x_max": x_max,
            "bins": bins.tolist(),
            "gen_hist_density": gen_hist.tolist(),
            "imp_hist_density": imp_hist.tolist(),
            "case_hist_density": case_hist_payload,
            "case_values": case_values_payload,
            "kde_x": kde_x.tolist(),
            "kde_gen_y": kde_gen_y,
            "kde_imp_y": kde_imp_y,
            "kde_case_y": kde_case_y,
        }

    def write_distribution_png(self, target_png_path: Path) -> bool:
        """Renderiza a distribuição como PNG (matplotlib Agg headless).

        Reusa o sidecar dict via `_compute_distribution_summary` e delega
        o desenho ao helper puro `distribution_plot.render_distribution_png`,
        garantindo idéntico visual ao que o ScoreDistributionWidget exibe.

        Retorna True se gravou, False se o dataset estava vazio (sem PNG
        a gerar) ou se a renderização falhou. Falha NÃO levanta — caller
        decide se quer logar.
        """
        try:
            summary = self._compute_distribution_summary()
            if summary is None:
                return False
            from faceaudit.models.similarity.distribution_plot import (
                render_distribution_png,
            )
            render_distribution_png(summary, target_png_path)
            return True
        except Exception:
            return False

    @staticmethod
    def try_load_distribution_summary(parquet_path: Union[str, Path]) -> Optional[dict]:
        """
        Tenta ler o sidecar `<stem>_dist.json` ao lado do parquet.

        Retorna o dict pronto pra plotar (compatível com o que o
        ScoreDistributionWidget espera) ou None se ausente / inválido /
        mais antigo que o parquet (stale).
        """
        import json

        p = Path(parquet_path)
        sidecar_path = p.with_name(p.stem + "_dist.json")
        if not sidecar_path.exists():
            return None
        try:
            # Stale check: se o parquet foi reescrito após o sidecar,
            # ignora (igual ao padrão usado em ResourceInfoWidget para
            # outros sidecars).
            if p.exists() and p.stat().st_mtime > sidecar_path.stat().st_mtime + 1.0:
                return None
            data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            # Aceita v1 (legacy) e v2 (atual com case_*). Os consumidores
            # leem case_* via `.get(..., default)`, então a leitura de
            # sidecar v1 fica compatível. Rejeição estrita a "== 1" era
            # um bug — toda escrita atual cria v2 (ver
            # `_compute_distribution_summary`), e a leitura jogava fora
            # o sidecar, forçando "n/a" nas tabelas que dependem dele.
            schema_version = data.get("schema_version")
            if schema_version not in (1, 2):
                return None
            return data
        except Exception:
            return None

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