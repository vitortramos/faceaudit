# faceaudit/models/calibration/bi_gaussianized_calibration_model_config.py

"""DTO de configuração do calibrador Bi-Gaussianized (Morrison 2024).

Mapeia scores para um sistema bi-gaussiano perfeitamente calibrado com σ²
alvo derivado de Cllr ou EER do treino. Quatro decisões:

  - `target_metric`     — Cllr (numérico, ~10ms) ou EER (analítico, exato).
  - `initial_calibrator`— calibrador inicial usado pra derivar Cllr/EER.
                          NONE pra scores já em espaço de ln(Λ) (ex: PLDA);
                          LOGREG quando rápido; KDE quando há outliers
                          (variante recomendada pelo paper §3.6).
  - `inversion_method`  — interp1d (rápido) ou Brent root-finding (preciso)
                          para inverter a CDF da mistura de duas gaussianas.
  - `sigma_shrinkage_factor` — encolhimento opcional do σ alvo (§3.5),
                               puxa LRs grandes pra perto de 1.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Type, TypeVar

from faceaudit.enums.calibrator_cv_split_mode import CalibratorCVSplitMode
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="BiGaussianizedCalibrationModelConfig")


class BiGaussTargetMetric(Enum):
    """Métrica usada pra derivar σ² do sistema bi-gaussiano alvo."""
    CLLR = "CLLR"   # via fit ln(Cllr) = b·(exp(-c·σ²) - 1) → Eq. (8)
    EER  = "EER"    # σ = -2·Φ⁻¹(EER) → Eq. (10), analítico

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value) -> "BiGaussTargetMetric":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.CLLR
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.CLLR


class BiGaussInitialCalibrator(Enum):
    """Calibrador inicial aplicado antes do passo de bi-gaussianização."""
    NONE   = "NONE"     # scores já em espaço de ln(Λ) (ex: PLDA log-LR)
    LOGREG = "LOGREG"   # rápido; assume linearidade no espaço dos scores
    KDE    = "KDE"      # mais robusto a outliers; §3.6 do paper recomenda

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value) -> "BiGaussInitialCalibrator":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.NONE
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.NONE


class BiGaussInversionMethod(Enum):
    """Método pra inverter a CDF da mistura 0.5·N(+σ²/2,σ²) + 0.5·N(-σ²/2,σ²)."""
    INTERP1D    = "INTERP1D"     # interpolação linear sobre grid pré-computado
    BRENT_ROOT  = "BRENT_ROOT"   # scipy.optimize.brentq por ponto; preciso, lento

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value) -> "BiGaussInversionMethod":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.INTERP1D
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.INTERP1D


class BiGaussGxMethod(Enum):
    """Método pra construir a CDF empírica G(x) — passo 5 do paper Morrison §2.7."""
    EMPIRICAL_LINEAR = "EMPIRICAL_LINEAR"  # piecewise linear nos breakpoints (paper default)
    KDE_SMOOTHED     = "KDE_SMOOTHED"      # G(x) = 0.5·F̂_g(x) + 0.5·F̂_i(x) via KDE Gaussiana

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value) -> "BiGaussGxMethod":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.EMPIRICAL_LINEAR
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.EMPIRICAL_LINEAR


class BiGaussSigmaEstimationMode(Enum):
    """Como estimar Cllr (ou EER) que alimenta σ² alvo."""
    IN_SAMPLE     = "IN_SAMPLE"      # calibrador inicial fit+predict no mesmo dataset (default)
    CROSS_FITTED  = "CROSS_FITTED"   # k-fold estratificado; reduz overfit do inicial calibrator

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value) -> "BiGaussSigmaEstimationMode":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.IN_SAMPLE
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.IN_SAMPLE


class BiGaussScoreDirection(Enum):
    """Direção dos scores: qual classe tem score mais alto."""
    AUTO                = "AUTO"                  # detecta via mean(gen) vs mean(imp)
    HIGHER_IS_GENUINE   = "HIGHER_IS_GENUINE"     # convenção padrão (PLDA, similaridade)
    LOWER_IS_GENUINE    = "LOWER_IS_GENUINE"      # convenção invertida (distância)

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value) -> "BiGaussScoreDirection":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.AUTO
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.AUTO


@dataclass
class BiGaussianizedCalibrationModelConfig(JSONSerializableMixin):
    """
    DTO de configuração do modelo de calibração Bi-Gaussianized.

    Bi-Gaussianized Calibration (Morrison 2024, Law, Probability and Risk):
    score → ln(Λ) via mapeamento monotônico não-linear pra distribuições alvo
    `Hp ~ N(+σ²/2, σ²)` e `Hd ~ N(-σ²/2, σ²)`, com σ² calculado de modo que
    o Cllr (ou EER) do sistema alvo bata com o do treino.
    """

    target_metric: BiGaussTargetMetric = field(
        default=BiGaussTargetMetric.CLLR,
        metadata={
            "help": (
                "<html><b>📊 Qual métrica do treino determina o σ alvo do "
                "modelo bi-gaussiano</b>. O σ controla o quão extremos podem "
                "ser os log-LRs calibrados — sistemas que separam bem geram "
                "σ maior (LRs mais confiantes).<br/><br/>"
                "<ul>"
                "<li><b>CLLR</b> (padrão) — usa o custo médio de informação. "
                "Aproveita toda a distribuição dos scores. <b>Recomendado</b> "
                "quando há ≥ 50 fontes distintas e o calibrador inicial é "
                "LOGREG ou KDE.</li>"
                "<li><b>EER</b> — usa só o ponto de igual erro. Fórmula "
                "analítica direta dos scores brutos (não usa calibrador "
                "inicial). Discretiza em passos de 1/N, ruidoso com poucos "
                "same-source. Use quando initial_calibrator não pode ser "
                "confiável (poucos dados pra ele rodar).</li>"
                "</ul></html>"
            )
        },
    )

    initial_calibrator: BiGaussInitialCalibrator = field(
        default=BiGaussInitialCalibrator.LOGREG,
        metadata={
            "help": (
                "<html><b>🎯 Calibrador inicial</b> — remapeia scores brutos "
                "pra espaço log-LR antes de calcular o Cllr que alimenta σ². "
                "<b>Crítico</b>: scores PLDA não estão em escala log-LR "
                "natural (gen mean +50, imp mean -60 em datasets típicos). "
                "Sem remapeamento, Cllr explode e σ² fica errado.<br/><br/>"
                "<ul>"
                "<li><b>LOGREG</b> (padrão recomendado) — sigmoide linear. "
                "Robusto, ~10ms, baixo overfit. <b>Escolha primeira</b>.</li>"
                "<li><b>NONE</b> — só use se os scores já são log-LRs "
                "calibrados (saída de outro calibrador, NÃO PLDA cru). "
                "Pode produzir Cllr inflado se mal escolhido.</li>"
                "<li><b>KDE</b> — não-paramétrico. Capacidade muito alta, "
                "overfita o Cllr de treino (use junto com "
                "sigma_estimation_mode=CROSS_FITTED se escolher).</li>"
                "</ul>"
                "Só afeta a estimação de σ² alvo. O mapeamento final "
                "não-linear sempre usa a CDF empírica G(x).<br/>"
                "<i>Ignorado quando target_metric=EER.</i></html>"
            )
        },
    )

    inversion_method: BiGaussInversionMethod = field(
        default=BiGaussInversionMethod.INTERP1D,
        metadata={
            "help": (
                "<html><b>Método pra inverter F(z) — CDF da mistura "
                "bi-gaussiana alvo.</b> Padrão (INTERP1D) é adequado pra "
                "uso prático. Só mude se precisar de precisão extra nas "
                "caudas (LRs muito extremos).<br/><br/>"
                "<ul>"
                "<li><b>INTERP1D</b> (padrão) — grid pré-computado + "
                "interpolação linear. O(N) total.</li>"
                "<li><b>BRENT_ROOT</b> — root-finding por ponto, ~50-100× "
                "mais lento. Use só se INTERP1D mostrar artefatos.</li>"
                "</ul></html>"
            )
        },
    )

    sigma_shrinkage_factor: float = field(
        default=1.0,
        metadata={
            "help": (
                "<html><b>Encolhe o σ alvo derivado</b> (§3.5 do paper "
                "Morrison). Valor < 1.0 puxa todos os LRs pra perto de 0 "
                "(decisão mais cautelosa).<br/><br/>"
                "<ul>"
                "<li><b>1.0</b> (padrão) — sem encolhimento. Use no caso "
                "normal.</li>"
                "<li><b>0.8–0.9</b> — encolhimento leve, defensável se "
                "houver suspeita de overfit que CROSS_FITTED não captou.</li>"
                "<li><b>&lt; 0.7</b> — agressivo. Geralmente piora Cllr; só "
                "use em diagnóstico/ablação.</li>"
                "</ul></html>"
            )
        },
    )

    enable_clipping: bool = field(
        default=False,
        metadata={
            "help": (
                "<html><b>Aplica clipping no log-LR calibrado</b> a "
                "<code>[min_log_lr, max_log_lr]</code>. Quando desligado "
                "(padrão), o calibrador não trunca a saída — log-LRs "
                "extremos (gerados por scores fora da faixa do treino ou "
                "por caudas da bi-gaussiana) ficam intactos. <br/><br/>"
                "Ligar pode ser útil em laudos onde se quer reportar log-LR "
                "com magnitude limitada (ex: ±20), mas via de regra "
                "<b>introduz perda de informação</b> e satura métricas como "
                "<i>cllr_min</i>. Mantenha desligado a menos que haja um "
                "motivo específico.<br/><br/>"
                "<b>⚠️ Importante:</b> este toggle controla <b>apenas</b> o "
                "clip final <code>np.clip(log_lr, min_log_lr, max_log_lr)</code>. "
                "Mesmo desligado, o BiGauss tem dois clamps internos que "
                "produzem floors/ceilings naturais e <b>não são afetados "
                "por este flag</b>:<br/>"
                "<ul>"
                "<li><code>gx_clamp_to_training_range</code> (configurável, "
                "default <b>True</b>): scores brutos fora do range visto no "
                "treino mapeiam pra <code>G(x_min)</code> ou "
                "<code>G(x_max)</code>. Todos esses pontos caem no mesmo "
                "percentil e produzem o <b>mesmo</b> log-LR de saída — "
                "aparece como um patamar visual nas caudas da curva de "
                "calibração.</li>"
                "<li>Clamp de probabilidade <code>u ∈ [ε, 1−ε]</code> "
                "(hardcoded, ε ≈ 1e-12): evita <code>F⁻¹(0) = −∞</code> e "
                "<code>F⁻¹(1) = +∞</code>. Define o piso/teto <i>matemático</i> "
                "do que o BiGauss consegue emitir — depende de σ alvo.</li>"
                "</ul>"
                "Pra eliminar o patamar visual nas caudas, é preciso "
                "desligar também <code>gx_clamp_to_training_range</code> "
                "(disponível quando <code>gx_method=KDE_SMOOTHED</code>) — "
                "mas isso permite extrapolação confiante em regiões não "
                "treinadas, o que <b>não é recomendado</b> em forense.</html>"
            )
        },
    )

    min_log_lr: float = field(
        default=-20.0,
        metadata={"help": (
            "Limite inferior do log-LR calibrado. Só aplica se "
            "<code>enable_clipping=True</code>."
        )}
    )

    max_log_lr: float = field(
        default=20.0,
        metadata={"help": (
            "Limite superior do log-LR calibrado. Só aplica se "
            "<code>enable_clipping=True</code>."
        )}
    )

    interpolation_points: int = field(
        default=10000,
        metadata={
            "help": (
                "Quantidade de pontos do grid usado pra construir G(x) (CDF "
                "empírica) e F⁻¹ (inversa da mistura). Só aplica se "
                "inversion_method=INTERP1D."
            )
        },
    )

    brent_xtol: float = field(
        default=1e-6,
        metadata={"help": "Tolerância absoluta do Brent (só aplica se BRENT_ROOT)."}
    )

    brent_search_range: float = field(
        default=50.0,
        metadata={
            "help": (
                "Range simétrico [-R, +R] em ln(Λ) usado como bracket inicial "
                "do Brent. Tem que conter as caudas; default 50 cobre LR de até e^50."
            )
        },
    )

    initial_logreg_C: float = field(
        default=1.0,
        metadata={"help": "C da LogReg inicial (só aplica se initial_calibrator=LOGREG)."}
    )

    initial_kde_bandwidth: float = field(
        default=0.0,
        metadata={
            "help": (
                "Bandwidth do KDE inicial. 0.0 = automático via Silverman "
                "(padrão do paper). Só aplica se initial_calibrator=KDE."
            )
        },
    )

    initial_kde_max_samples_per_class: int = field(
        default=10000,
        metadata={
            "help": (
                "<html><b>Subsampling do KDE inicial</b> (só aplica se "
                "initial_calibrator=KDE). Limita N por classe usado em fit + "
                "score_samples. Razão: sklearn KernelDensity tem custo O(N²) "
                "no pior caso (atol=0) — com 42k impostores o fit demora ~2 "
                "min sem feedback intermediário. Sample de 10k já é suficiente "
                "para estimativa de Cllr que alimenta σ². <br/>"
                "<code>0</code> = sem subsampling (usa todos os pontos, lento)."
                "</html>"
            )
        },
    )

    random_seed: int = field(
        default=42,
        metadata={"help": "Semente para amostragens internas."}
    )

    # ── G(x) — construção da CDF empírica ───────────────────────────────────
    gx_method: BiGaussGxMethod = field(
        default=BiGaussGxMethod.KDE_SMOOTHED,
        metadata={
            "help": (
                "<html><b>🔧 Método de construção da CDF empírica G(x)</b> "
                "que mapeia scores → percentil → log-LR pela inversa "
                "F⁻¹ da mistura bi-gaussiana. <b>Knob de maior impacto na "
                "calibração final</b> (junto com gx_kde_bandwidth_factor).<br/><br/>"
                "<ul>"
                "<li><b>KDE_SMOOTHED</b> (padrão recomendado) — G(x) = "
                "0.5·F̂_g(x) + 0.5·F̂_i(x), com F̂ obtidas por integração "
                "analítica de KDEs Gaussianos. <b>Permite controlar overfit "
                "do mapeamento</b> via gx_kde_bandwidth_factor.</li>"
                "<li><b>EMPIRICAL_LINEAR</b> — paper Morrison §2.7 puro. "
                "Soma cumulativa ponderada nos breakpoints. Preserva "
                "exatamente os dados, mas <b>overfita</b> o mapeamento "
                "(scores próximos do treino mapeiam exato; scores ligeiramente "
                "diferentes da validação caem em lugares diferentes). "
                "Não tem knob de suavização. Use quando quiser fidelidade "
                "estrita ao paper ou quando o dataset é muito pequeno.</li>"
                "</ul></html>"
            )
        },
    )

    gx_kde_bandwidth_gen: float = field(
        default=0.0,
        metadata={
            "help": (
                "Bandwidth do KDE dos genuínos pra construir G(x). "
                "0.0 = Silverman automático. Só aplica se gx_method=KDE_SMOOTHED."
            )
        },
    )

    gx_kde_bandwidth_imp: float = field(
        default=0.0,
        metadata={
            "help": (
                "Bandwidth do KDE dos impostores pra construir G(x). "
                "0.0 = Silverman automático. Só aplica se gx_method=KDE_SMOOTHED."
            )
        },
    )

    gx_kde_bandwidth_factor: float = field(
        default=5.0,
        metadata={
            "help": (
                "<html><b>🎚️ Multiplicador da bandwidth do KDE de G(x)</b> — "
                "<b>knob crítico de generalização</b>. Aplicado SOBRE a "
                "bandwidth resolvida (Silverman automática ou explícita "
                "via gx_kde_bandwidth_*).<br/><br/>"
                "Valor alto = G(x) mais suavizada → scores treino/validação "
                "vizinhos mapeiam pro mesmo lugar → calibração mais robusta. "
                "Valor baixo = G(x) fiel aos dados → overfit do mapeamento.<br/><br/>"
                "<ul>"
                "<li><b>1.0</b> — Silverman puro (otimiza estimativa de "
                "densidade, NÃO calibração). Overfita G(x).</li>"
                "<li><b>1.5–3.0</b> — oversmoothing moderado, melhora "
                "calibração.</li>"
                "<li><b>5.0</b> (padrão recomendado) — <b>sweet spot</b> "
                "validado empiricamente em pipelines AdaFace + CORAL + "
                "Whitening + PLDA shrinkage. Cllr cai ~22%, calibração "
                "compete com LogReg.</li>"
                "<li><b>&gt; 7.0</b> — bias alto, oversmoothing destrói "
                "estrutura discriminativa. Cllr piora.</li>"
                "</ul>"
                "Tunar com sweep em conjunto de validação se mudar de "
                "pipeline.<br/>"
                "<i>Só aplica se gx_method=KDE_SMOOTHED.</i></html>"
            )
        },
    )

    gx_clamp_to_training_range: bool = field(
        default=True,
        metadata={
            "help": (
                "Preserva a propriedade do paper §2.7: scores fora do range do "
                "treino recebem o G(x_min) ou G(x_max). Recomendado True para "
                "evitar extrapolação confiante em regiões não-treinadas. "
                "Só aplica se gx_method=KDE_SMOOTHED."
            )
        },
    )

    gx_grid_points: int = field(
        default=2000,
        metadata={
            "help": (
                "Quantidade de pontos do grid usado pra cachear G(x). "
                "2000 é suficiente — o resto vem da interpolação linear do "
                "scipy. Só aplica se gx_method=KDE_SMOOTHED."
            )
        },
    )

    # ── Estimação de σ² via cross-validation ────────────────────────────────
    sigma_estimation_mode: BiGaussSigmaEstimationMode = field(
        default=BiGaussSigmaEstimationMode.CROSS_FITTED,
        metadata={
            "help": (
                "<html><b>🎯 Como estimar Cllr que alimenta σ² alvo</b>. O Cllr "
                "do calibrador inicial é a base do cálculo de σ; se ele estiver "
                "otimista (overfit), σ² fica superestimado e a calibração final "
                "vira excessivamente confiante.<br/><br/>"
                "<ul>"
                "<li><b>CROSS_FITTED</b> (padrão recomendado) — k-fold "
                "estratificado: cada amostra recebe predição do calibrador "
                "inicial treinado em <i>outras</i> dobras. Cllr é não-viesado, "
                "σ² realista. Custo: fit ~k× mais lento (default 5×, ainda "
                "rápido).</li>"
                "<li><b>IN_SAMPLE</b> — fit + predict do calibrador inicial no "
                "mesmo conjunto. Cllr otimista demais (especialmente com "
                "initial=KDE). Use só quando velocidade é crítica ou dataset "
                "é pequeno demais pra CV (validação automática reduz k).</li>"
                "</ul>"
                "<i>Sem efeito se initial_calibrator=NONE (não há fitting "
                "pra cross-validar). Ignorado quando target_metric=EER.</i></html>"
            )
        },
    )

    sigma_cv_folds: int = field(
        default=5,
        metadata={
            "help": (
                "k de StratifiedKFold em sigma_estimation_mode=CROSS_FITTED. "
                "Default 5. Reduzido automaticamente para min(classe_minoritária, k) "
                "se houver poucos genuínos."
            )
        },
    )

    # ── Cross-validation diagnóstico do calibrador completo (opt-in) ────
    # DISTINTO de sigma_cv_folds: este aqui mede Cllr CV do BiGauss INTEIRO
    # (incluindo G(x) + F⁻¹ + sigma já alvo). O sigma_cv_folds acima é CV
    # interna pra estimar σ² antes de construir G(x).
    k_folds: int = field(
        default=0,
        metadata={"help": (
            "<html><b>🧪 K-fold CV diagnóstico do calibrador BiGauss completo</b> "
            "(opt-in, DISTINTO de <code>sigma_cv_folds</code>).<br/><br/>"
            "<code>0</code> (padrão) — desativado.<br/>"
            "<code>5</code> — recomendado. Roda 5 folds estratificados após "
            "o fit principal, refitando todo o pipeline BiGauss (G(x) + F⁻¹) "
            "com hiperparâmetros congelados (<code>gx_kde_bandwidth_*</code>, "
            "<code>sigma_shrinkage_factor</code>, etc.). Anexa "
            "<code>calibrator_cllr_cv</code> ao metadata.<br/><br/>"
            "<b>Diferença vs sigma_cv_folds</b>: aquele é CV interna do "
            "calibrador inicial (LogReg/KDE) só pra estimar σ². Este aqui "
            "mede generalização do BiGauss completo.<br/><br/>"
            "<b>Atenção</b>: Cllr CV mede generalização DENTRO do domínio "
            "de calibração, NÃO performance em validação externa."
            "</html>"
        )}
    )

    cv_split_mode: CalibratorCVSplitMode = field(
        default=CalibratorCVSplitMode.STRATIFIED_PAIR,
        metadata={"help": (
            "<html><b>Estratégia de partição dos folds</b> em k-fold CV do "
            "calibrador completo (só aplicável quando <code>k_folds &gt; 0</code>).<br/><br/>"
            "<b>STRATIFIED_PAIR</b> (padrão) — sklearn.StratifiedKFold.<br/>"
            "<b>STRATIFIED_GROUP</b> — sklearn.StratifiedGroupKFold por "
            "<code>probe_identity</code>. <b>Forense-defensável</b>, "
            "recomendado em laudos.</html>"
        )}
    )

    # ── Direção dos scores (defensivo) ──────────────────────────────────────
    score_direction: BiGaussScoreDirection = field(
        default=BiGaussScoreDirection.AUTO,
        metadata={
            "help": (
                "<html><b>Direção dos scores</b>. PLDA e similaridade-tipo: "
                "score alto = mais genuíno. Distância-tipo: score baixo = mais "
                "genuíno. Definir errado produz log-LRs com sinal trocado SILENCIOSAMENTE.<br/>"
                "<ul>"
                "<li><b>AUTO</b> (padrão) — detecta via comparação de médias gen/imp "
                "no treino. Loga decisão e ativa flip interno se necessário.</li>"
                "<li><b>HIGHER_IS_GENUINE</b> — força convenção padrão.</li>"
                "<li><b>LOWER_IS_GENUINE</b> — força convenção invertida; flip "
                "aplicado em fit + calibrate_scores.</li>"
                "</ul></html>"
            )
        },
    )

    def __post_init__(self) -> None:
        # Normaliza enums vindos de string (caso from_dict externo)
        if not isinstance(self.target_metric, BiGaussTargetMetric):
            self.target_metric = BiGaussTargetMetric.from_str(self.target_metric)
        if not isinstance(self.initial_calibrator, BiGaussInitialCalibrator):
            self.initial_calibrator = BiGaussInitialCalibrator.from_str(self.initial_calibrator)
        if not isinstance(self.inversion_method, BiGaussInversionMethod):
            self.inversion_method = BiGaussInversionMethod.from_str(self.inversion_method)
        if not isinstance(self.gx_method, BiGaussGxMethod):
            self.gx_method = BiGaussGxMethod.from_str(self.gx_method)
        if not isinstance(self.sigma_estimation_mode, BiGaussSigmaEstimationMode):
            self.sigma_estimation_mode = BiGaussSigmaEstimationMode.from_str(self.sigma_estimation_mode)
        if not isinstance(self.score_direction, BiGaussScoreDirection):
            self.score_direction = BiGaussScoreDirection.from_str(self.score_direction)
        if not isinstance(self.cv_split_mode, CalibratorCVSplitMode):
            self.cv_split_mode = CalibratorCVSplitMode.from_str(self.cv_split_mode)

        if not (0.0 < self.sigma_shrinkage_factor <= 1.0):
            raise ValueError("sigma_shrinkage_factor deve estar em (0, 1].")
        # min/max só são restritivos quando o clipping está ligado.
        if self.enable_clipping and self.min_log_lr >= self.max_log_lr:
            raise ValueError("min_log_lr deve ser menor que max_log_lr.")
        if self.interpolation_points < 100:
            raise ValueError("interpolation_points deve ser pelo menos 100.")
        if self.brent_xtol <= 0:
            raise ValueError("brent_xtol deve ser positivo.")
        if self.brent_search_range <= 0:
            raise ValueError("brent_search_range deve ser positivo.")
        if self.initial_logreg_C <= 0:
            raise ValueError("initial_logreg_C deve ser positivo.")
        if self.initial_kde_bandwidth < 0:
            raise ValueError("initial_kde_bandwidth não pode ser negativo.")
        if self.initial_kde_max_samples_per_class < 0:
            raise ValueError("initial_kde_max_samples_per_class não pode ser negativo.")
        if self.gx_kde_bandwidth_gen < 0:
            raise ValueError("gx_kde_bandwidth_gen não pode ser negativo.")
        if self.gx_kde_bandwidth_imp < 0:
            raise ValueError("gx_kde_bandwidth_imp não pode ser negativo.")
        if self.gx_kde_bandwidth_factor <= 0:
            raise ValueError("gx_kde_bandwidth_factor deve ser positivo.")
        if self.gx_grid_points < 100:
            raise ValueError("gx_grid_points deve ser pelo menos 100.")
        if self.sigma_cv_folds < 2:
            raise ValueError("sigma_cv_folds deve ser pelo menos 2.")

        # k_folds diagnóstico (DISTINTO de sigma_cv_folds)
        if self.k_folds < 0:
            raise ValueError("k_folds não pode ser negativo.")
        if self.k_folds == 1:
            raise ValueError("k_folds deve ser 0 (desativado) ou >= 2.")

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return cls()
        payload = dict(data)
        if "target_metric" in payload:
            payload["target_metric"] = BiGaussTargetMetric.from_str(
                payload["target_metric"]
            )
        if "initial_calibrator" in payload:
            payload["initial_calibrator"] = BiGaussInitialCalibrator.from_str(
                payload["initial_calibrator"]
            )
        if "inversion_method" in payload:
            payload["inversion_method"] = BiGaussInversionMethod.from_str(
                payload["inversion_method"]
            )
        if "gx_method" in payload:
            payload["gx_method"] = BiGaussGxMethod.from_str(payload["gx_method"])
        if "sigma_estimation_mode" in payload:
            payload["sigma_estimation_mode"] = BiGaussSigmaEstimationMode.from_str(
                payload["sigma_estimation_mode"]
            )
        if "score_direction" in payload:
            payload["score_direction"] = BiGaussScoreDirection.from_str(
                payload["score_direction"]
            )
        if "cv_split_mode" in payload:
            payload["cv_split_mode"] = CalibratorCVSplitMode.from_str(
                payload["cv_split_mode"]
            )
        return cls(**payload)

    # ── Visibilidade dinâmica de fields na UI ───────────────────────────────

    # Campos cuja mudança deve disparar rebuild do nested form (afetam
    # visibilidade ou estado de outros fields). Lido pelo nested_dto_widget.
    REACTIVE_FIELDS = (
        "target_metric",
        "initial_calibrator",
        "inversion_method",
        "gx_method",
        "sigma_estimation_mode",
        "enable_clipping",
    )

    def get_hidden_fields(self):
        """Retorna o conjunto de fields a OCULTAR pela UI conforme o estado atual.

        Regras:
        - target_metric=EER: oculta initial_calibrator + todos seus params
          (EER analítico não usa calibrador inicial) e sigma_estimation_mode
          (CROSS_FITTED só faz sentido pra calibrador inicial).
        - initial_calibrator filtra por modo:
            NONE   → oculta initial_logreg_C, initial_kde_*
                     + sigma_estimation_mode (no-op sem fitting)
            LOGREG → oculta initial_kde_*
            KDE    → oculta initial_logreg_C
        - inversion_method=INTERP1D: oculta brent_*
        - inversion_method=BRENT_ROOT: oculta interpolation_points
        - gx_method=EMPIRICAL_LINEAR: oculta todos os gx_kde_* + gx_clamp_*
          + gx_grid_points (só KDE_SMOOTHED usa).
        - sigma_estimation_mode=IN_SAMPLE: oculta sigma_cv_folds.
        """
        hidden = set()

        # Métrica
        is_eer = self.target_metric == BiGaussTargetMetric.EER
        if is_eer:
            # EER analítico não usa calibrador inicial → esconde toda a família
            hidden |= {
                "initial_calibrator",
                "initial_logreg_C",
                "initial_kde_bandwidth",
                "initial_kde_max_samples_per_class",
                "sigma_estimation_mode",
                "sigma_cv_folds",
            }
        else:
            # Filtra os params do calibrador inicial conforme o modo escolhido
            if self.initial_calibrator == BiGaussInitialCalibrator.NONE:
                hidden |= {
                    "initial_logreg_C",
                    "initial_kde_bandwidth",
                    "initial_kde_max_samples_per_class",
                    # CROSS_FITTED é no-op sem fitting; esconde pra não confundir
                    "sigma_estimation_mode",
                    "sigma_cv_folds",
                }
            elif self.initial_calibrator == BiGaussInitialCalibrator.LOGREG:
                hidden |= {
                    "initial_kde_bandwidth",
                    "initial_kde_max_samples_per_class",
                }
            elif self.initial_calibrator == BiGaussInitialCalibrator.KDE:
                hidden |= {"initial_logreg_C"}

            # sigma_cv_folds só é usado em CROSS_FITTED
            if self.sigma_estimation_mode != BiGaussSigmaEstimationMode.CROSS_FITTED:
                hidden.add("sigma_cv_folds")

        # Inversion method gatekeeps interpolation_points vs brent_*
        if self.inversion_method == BiGaussInversionMethod.INTERP1D:
            hidden |= {"brent_xtol", "brent_search_range"}
        else:  # BRENT_ROOT
            hidden.add("interpolation_points")

        # G(x) method gatekeeps toda a família gx_kde_* e gx_clamp_*
        if self.gx_method == BiGaussGxMethod.EMPIRICAL_LINEAR:
            hidden |= {
                "gx_kde_bandwidth_gen",
                "gx_kde_bandwidth_imp",
                "gx_kde_bandwidth_factor",
                "gx_clamp_to_training_range",
                "gx_grid_points",
            }

        # Clipping toggle: min/max só aparecem quando ligado.
        if not self.enable_clipping:
            hidden |= {"min_log_lr", "max_log_lr"}

        return hidden
