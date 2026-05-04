# faceaudit/models/dataset_stats.py
import math
import statistics
from dataclasses import dataclass, field
from typing import Set, Dict, Optional


@dataclass(frozen=True)
class BalanceMetrics:
    """
    Métricas derivadas do balanceamento de amostras por identidade.

    `cv` é o coeficiente de variação (std / mean). `imbalance_ratio` é
    max/min de samples por identidade. `verdict` consolida os dois (pior
    caso) em um rótulo legível para o perito.
    """
    n_identities: int
    min_samples: int
    max_samples: int
    mean_samples: float
    median_samples: float
    std_samples: float
    cv: float
    imbalance_ratio: float
    verdict: str

    @property
    def verdict_emoji(self) -> str:
        return self.verdict.split(" ", 1)[0] if self.verdict else ""

    @property
    def verdict_label(self) -> str:
        parts = self.verdict.split(" ", 1)
        return parts[1] if len(parts) > 1 else self.verdict


@dataclass
class DatasetStats:
    """
    Representa metadados agnósticos de um dataset facial.
    Enriquecido com histogramas demográficos.
    """
    # Totais
    n_imgs: int = 0
    n_ids: int = 0

    # Dimensão dos embeddings (0 se não aplicável ou dataset vazio)
    embedding_dim: int = 0

    # Identificação
    identity_set: Set[str] = field(default_factory=set)
    hash_set: Set[str] = field(default_factory=set)

    embedding_columns: Set[str] = field(default_factory=set)

    # Estatísticas Demográficas (Histogramas)
    # Ex: {'Male': 150, 'Female': 120}
    gender_counts: Dict[str, int] = field(default_factory=dict)
    ethnicity_counts: Dict[str, int] = field(default_factory=dict)
    origin_counts: Dict[str, int] = field(default_factory=dict)

    # Balanceamento por identidade — { identity: n_samples }
    # Permite calcular dispersão (CV, max/min) sem recarregar o dataset.
    samples_per_identity: Dict[str, int] = field(default_factory=dict)

    @property
    def balance_metrics(self) -> Optional[BalanceMetrics]:
        """
        Retorna métricas de balanceamento por identidade ou None quando
        não aplicável (sem identidades, ou só uma identidade — caso em
        que "balanceamento" não tem sentido estatístico).
        """
        counts = list(self.samples_per_identity.values())
        if len(counts) < 2:
            return None

        mn = int(min(counts))
        mx = int(max(counts))
        mean = float(statistics.fmean(counts))
        median = float(statistics.median(counts))
        std = float(statistics.pstdev(counts))
        cv = (std / mean) if mean > 0 else 0.0
        ratio = (mx / mn) if mn > 0 else math.inf

        # Verdict = pior caso entre CV e imbalance ratio.
        # Ordem: balanceado < aceitável < moderado < severo.
        cv_level = (
            0 if cv == 0.0
            else 1 if cv <= 0.35
            else 2 if cv <= 0.75
            else 3
        )
        ratio_level = (
            0 if ratio == 1.0
            else 1 if ratio <= 4.0
            else 2 if ratio <= 10.0
            else 3
        )
        level = max(cv_level, ratio_level)
        verdict = {
            0: "✅ Balanceado",
            1: "🟢 Aceitável",
            2: "🟡 Moderadamente desbalanceado",
            3: "🔴 Severamente desbalanceado",
        }[level]

        return BalanceMetrics(
            n_identities=len(counts),
            min_samples=mn,
            max_samples=mx,
            mean_samples=mean,
            median_samples=median,
            std_samples=std,
            cv=cv,
            imbalance_ratio=ratio,
            verdict=verdict,
        )